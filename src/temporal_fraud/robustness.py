import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from .data import digest, load_development
from .experiment import run as run_trees
from .graph_experiment import run as run_graph
from .graph_model import TemporalGraphClassifier, predict
from .metrics import evaluate
from .risk import (MonotoneCalibration, available_labels, capacity_threshold,
                   closed_hour_alerts, population_shift, reliability)
from .runtime import peak_memory_bytes


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def metrics(frame, fraction):
    return {name: evaluate(part.label, part.score, part.event_id, fraction)
            for name, part in [('validation', frame), ('cold', frame.loc[frame.cold]), ('warm', frame.loc[~frame.cold])]
            if len(part)}


class HistoryIntervention:
    def __init__(self, index, mode):
        if mode not in ['no_neighbors', 'no_edge_time']:
            raise ValueError('Unknown graph intervention')
        self.index, self.mode = index, mode

    def sample(self, rows, neighbors):
        history, age, direction = self.index.sample(rows, neighbors)
        if self.mode == 'no_neighbors':
            return np.full_like(history, -1), np.zeros_like(age), np.zeros_like(direction)
        return history, np.zeros_like(age), direction


def calibration_analysis(frame, config, model_fit_at):
    instant, delay = config['calibration_at'], config['calibration_delay_hours']
    fit = MonotoneCalibration.fit(frame, instant, delay, model_fit_at, config['calibration_min_positives'])
    matured = available_labels(frame.timestamp, instant, delay) & (frame.timestamp >= pd.Timestamp(model_fit_at)).to_numpy()
    calibration = frame.loc[matured]
    evaluation = frame.loc[(frame.timestamp >= pd.Timestamp(instant)) &
                           (frame.timestamp < pd.Timestamp(config['evaluation_end']))].copy()
    if not len(evaluation):
        raise ValueError('No post-calibration evaluation events')
    reference = fit.apply(calibration.score)
    raw = evaluation.score.to_numpy().copy()
    evaluation['score'] = fit.transform(raw, evaluation.timestamp)
    threshold = capacity_threshold(reference, config['alert_fraction'])
    selected, hourly = closed_hour_alerts(evaluation, threshold, config['alert_fraction'], config['evaluation_end'], config['seed'])
    evaluation['raw_score'], evaluation['alert'] = raw, selected
    positives = int(evaluation.label.sum())
    true_alerts = int(evaluation.loc[selected, 'label'].sum())
    report = {'calibrator': fit.to_dict(), 'threshold': threshold,
              'evaluation_first_event': str(evaluation.timestamp.min()), 'evaluation_last_event': str(evaluation.timestamp.max()),
              'evaluation_events': len(evaluation), 'evaluation_positives': positives,
              'raw': reliability(evaluation.label, raw, calibration.score),
              'calibrated': reliability(evaluation.label, evaluation.score, reference),
              'raw_ranking': evaluate(evaluation.label, raw, evaluation.event_id, config['alert_fraction']),
              'calibrated_ranking': evaluate(evaluation.label, evaluation.score, evaluation.event_id, config['alert_fraction']),
              'alerts': {'count': int(selected.sum()), 'budget_sum': sum(h['budget'] for h in hourly),
                         'true_alerts': true_alerts, 'recall': true_alerts / positives if positives else None,
                         'precision': true_alerts / selected.sum() if selected.any() else None,
                         'fraction': float(selected.mean()), 'threshold_only_fraction': float((evaluation.score >= threshold).mean()) if threshold is not None else 0.0,
                         'hourly': hourly},
              'feedback_event_ids_sha256': hashlib.sha256(calibration.event_id.to_numpy(dtype='<i8').tobytes()).hexdigest()}
    return report, evaluation


def run(data, baseline, graph, output, config, data_config, graph_config, device='cpu', resume=False):
    baseline, graph, output = Path(baseline), Path(graph), Path(output)
    if output.exists() and not resume:
        raise ValueError('New output directory or explicit resume required')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(graph_config['threads'])
    base_report = json.loads((baseline / 'report.json').read_text())
    graph_report = json.loads((graph / 'report.json').read_text())
    frozen = {**{'baseline/' + n: sha for n, sha in base_report['artifact_sha256'].items()},
              **{'graph/' + n: sha for n, sha in graph_report['artifact_sha256'].items()}}
    def verify_frozen():
        for name, expected in frozen.items():
            prefix, relative = name.split('/', 1)
            if digest((baseline if prefix == 'baseline' else graph) / relative) != expected:
                raise ValueError('Frozen artifact changed: ' + name)
    verify_frozen()
    report_path = output / 'study.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'stage': 'development_validation', 'test_scored': False, 'protocol': config,
        'data_config': data_config, 'graph_config': graph_config, 'frozen_artifact_sha256': frozen,
        'delays': {}, 'calibration': {}, 'ablations': {}, 'drift': {}, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    if report['protocol'] != config or report['data_config'] != data_config or report['graph_config'] != graph_config or report['frozen_artifact_sha256'] != frozen:
        raise ValueError('Resume protocol or frozen inputs differ')
    if report.get('completed'):
        for name, expected in report['artifact_sha256'].items():
            if digest(output / name) != expected:
                raise ValueError('Completed artifact changed')
        return report
    for delay in config['delay_hours']:
        if str(delay) in report['delays']:
            continue
        if delay == data_config['label_delay_hours']:
            tree_report = {'models': {'history_xgboost': metrics(
                pd.read_parquet(baseline / 'history_xgboost-validation.parquet'), config['alert_fraction'])}}
            neural_report = graph_report
            source = 'unchanged frozen 24-hour runs; not retrained'
        else:
            arm = output / f'delay-{delay}'
            arm.mkdir(exist_ok=True)
            arm_config = {**data_config, 'label_delay_hours': delay}
            tree_path = arm / 'trees/report.json'
            if tree_path.exists():
                tree_report = json.loads(tree_path.read_text())
            else:
                events, counts = load_development(data, arm_config)
                tree_report = run_trees(events, arm_config, arm / 'trees')
                tree_report['data_counts'] = counts
                tree_report['artifact_sha256'] = {p.name: digest(p) for p in (arm / 'trees').iterdir() if p.is_file()}
                write_json(tree_path, tree_report)
                del events
                gc.collect()
            neural_path = arm / 'graph'
            neural_report = run_graph({**graph_config, 'neighbors': [32]}, arm_config, data, neural_path,
                                      device=device, resume=neural_path.exists())
            source = 'new fixed-budget label-delay sensitivity arm'
        report['delays'][str(delay)] = {'source': source, 'training_events': neural_report['train_events'],
                                       'training_positives': neural_report['train_positives'],
                                       'history_xgboost': {g: tree_report['models']['history_xgboost'][g] for g in ['validation', 'cold', 'warm']},
                                       'temporal_attention_32': {g: neural_report['models']['temporal_attention_32'][g] for g in ['validation', 'cold', 'warm']}}
        write_json(report_path, report)
        print(json.dumps({'completed_delay_hours': delay, 'train_events': neural_report['train_events']}), flush=True)
    frames = {'history_xgboost': pd.read_parquet(baseline / 'history_xgboost-validation.parquet'),
              'temporal_attention_32': pd.read_parquet(graph / 'temporal_attention_32-validation.parquet')}
    for name in config['calibration_models']:
        if name not in report['calibration']:
            result, predictions = calibration_analysis(frames[name], config, data_config['fit_at'])
            predictions.to_parquet(output / (name + '-calibrated-policy.parquet'), index=False)
            write_json(output / (name + '-calibrator.json'), {'model': name, 'model_checkpoint_sha256': digest(baseline / 'history_xgboost.ubj') if name == 'history_xgboost' else graph_report['models'][name]['training']['checkpoint_sha256'], 'calibrator': result['calibrator'], 'threshold': result['threshold'], 'capacity_fraction': config['alert_fraction'], 'seed': config['seed']})
            report['calibration'][name] = result
            write_json(report_path, report)
            print(json.dumps({'calibration_model': name, 'matured_labels': result['calibrator']['examples'], 'evaluation_events': result['evaluation_events'], 'raw_brier': result['raw']['brier'], 'calibrated_brier': result['calibrated']['brier']}), flush=True)
    membership = pd.read_parquet(graph / 'membership.parquet')
    validation_rows = np.flatnonzero(membership.validation.to_numpy())
    x_host = np.load(graph / 'features.npy', mmap_mode='r')
    if not report['drift']:
        valid = membership.loc[membership.validation].reset_index(drop=True)
        reference = valid.timestamp.dt.strftime('%Y-%m-%d').eq(config['reference_drift_day']).to_numpy()
        observed = valid.timestamp.dt.strftime('%Y-%m-%d').eq(config['drift_day']).to_numpy()
        feature_drift = {name: population_shift(x_host[validation_rows[reference], column], x_host[validation_rows[observed], column])
                         for name, column in [('standardized_log_paid', 0), ('standardized_source_out_count', 6), ('standardized_source_age', 8)]}
        model_drift = {}
        for name, frame in frames.items():
            model_drift[name] = {'score_shift': population_shift(frame.score[reference], frame.score[observed]),
                                'days': {day: metrics(frame.loc[mask], config['alert_fraction']) for day, mask in [(config['reference_drift_day'], reference), (config['drift_day'], observed)]}}
        report['drift'] = {'reference_day': config['reference_drift_day'], 'observed_day': config['drift_day'],
                           'features': feature_drift, 'models': model_drift,
                           'caveat': 'Two development days only; observed distribution changes are not a diagnosis of production drift.'}
        write_json(report_path, report)
    if any(mode not in report['ablations'] for mode in config['graph_interventions']):
        x = torch.tensor(np.asarray(x_host), device=device)
        index = joblib.load(graph / 'neighborhood.joblib')
        state = torch.load(graph / 'neighbors-32-latest.pt', map_location=device, weights_only=True)
        model = TemporalGraphClassifier(x.shape[1], graph_config['hidden'], 32).to(device)
        model.load_state_dict(state['state_dict'])
        for mode in config['graph_interventions']:
            if mode in report['ablations']:
                continue
            before = time.perf_counter()
            scores = predict(model, x, HistoryIntervention(index, mode), validation_rows, graph_config['batch_size'])
            frame = frames['temporal_attention_32'].copy()
            frame['score'] = scores
            frame.to_parquet(output / (mode + '-validation.parquet'), index=False)
            report['ablations'][mode] = {'metrics': metrics(frame, config['alert_fraction']),
                                        'seconds': time.perf_counter() - before, 'retrained': False,
                                        'interpretation': 'Fixed-checkpoint inference intervention; distribution mismatch, not a retrained causal effect.'}
            write_json(report_path, report)
            print(json.dumps({'ablation': mode, 'average_precision': report['ablations'][mode]['metrics']['validation']['average_precision']}), flush=True)
    verify_frozen()
    arm_peaks = [json.loads((output / f'delay-{delay}/graph/report.json').read_text())['runtime']['peak_gpu_allocated_bytes']
                 for delay in config['delay_hours'] if delay != data_config['label_delay_hours']]
    report['runtime'] = {'seconds_this_invocation': time.perf_counter() - started, 'device': device,
                         'peak_process_memory_bytes': peak_memory_bytes(),
                         'peak_gpu_allocated_bytes': max([torch.cuda.max_memory_allocated(), *arm_peaks]) if device == 'cuda' else 0}
    report['completed'] = True
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['artifact_sha256'] = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p != report_path}
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/robustness-v1.json')
    parser.add_argument('--data-config', default='configs/ibm-hi-small.json')
    parser.add_argument('--graph-config', default='configs/temporal-v1.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.data, args.baseline, args.graph, args.output, json.loads(Path(args.config).read_text()),
        json.loads(Path(args.data_config).read_text()), json.loads(Path(args.graph_config).read_text()), args.device, args.resume)


if __name__ == '__main__':
    main()
