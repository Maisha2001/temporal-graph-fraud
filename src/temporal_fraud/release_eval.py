import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from .data import HEADER, NAMES, digest, normalize, split_events
from .features import causal_features
from .graph_model import TemporalGraphClassifier, predict
from .metrics import evaluate
from .risk import MonotoneCalibration, closed_hour_alerts
from .runtime import peak_memory_bytes
from .streaming import graph_features
from .temporal import TemporalNeighborhood

ARMS = ['train_prior', 'transaction_xgboost', 'history_xgboost', 'history_mlp',
        'temporal_attention_8', 'temporal_attention_32']
SOURCE_FILES = ['data.py', 'features.py', 'graph_model.py', 'metrics.py', 'risk.py',
                'runtime.py', 'streaming.py', 'temporal.py', 'release_eval.py']


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def frozen_sources(baseline, graph, risk):
    hashes = {}
    for key, folder, report in [('baseline', baseline, 'report.json'), ('graph', graph, 'report.json'), ('risk', risk, 'study.json')]:
        folder = Path(folder)
        values = json.loads((folder / report).read_text())
        for name, expected in values['artifact_sha256'].items():
            if digest(folder / name) != expected:
                raise ValueError('Frozen source changed: ' + key + '/' + name)
            hashes[key + '/' + name] = expected
        hashes[key + '/' + report] = digest(folder / report)
    return hashes


def label_free_events(path, config):
    if digest(path) != config['sha256']:
        raise ValueError('Dataset digest mismatch')
    import csv
    with Path(path).open(newline='', encoding='utf-8') as stream:
        if next(csv.reader(stream)) != HEADER:
            raise ValueError('Unexpected header')
    pieces, offset = [], 0
    for chunk in pd.read_csv(path, names=NAMES, header=0, usecols=NAMES[:-1], dtype='string', chunksize=250000):
        chunk['event_id'] = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        chunk['timestamp'] = pd.to_datetime(chunk.timestamp, format='%Y/%m/%d %H:%M', errors='raise')
        # Placeholder satisfies shared schema validation; the label column is never read.
        chunk['label'] = '0'
        pieces.append(normalize(chunk).drop(columns='label'))
    return pd.concat(pieces, ignore_index=True).sort_values(['timestamp', 'event_id']).reset_index(drop=True)


def source_hashes():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in SOURCE_FILES}


def freeze(config, baseline, graph, risk, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_config': config, 'source_sha256': source_hashes(),
        'artifact_sha256': frozen_sources(baseline, graph, risk), 'arms': ARMS,
        'selection': 'Every September 8 onward event, including sparse September 11-18 tail; no row/label exclusions.',
        'models': 'Exact existing final-epoch checkpoints and preprocessing; no retraining or threshold/calibration updates.',
        'labels': 'Prediction loader excludes label column; persist all scores and hashes before separately opening final labels.',
        'graph': 'Original pre-fit entity exclusion; earlier unlabeled evaluation events may update graph, never same-minute edges.',
        'metrics': 'AP, Brier, analytical uniform boundary-tie capacity recall/precision; all/cold/warm and each calendar day.',
        'alert_policy': 'Existing two September 7 noon calibrators and thresholds; closed-hour floor(1% traffic), seed 42.',
        'interpretation': 'Single synthetic temporal holdout; account hash buckets are excluded at fit, not permanently unseen nodes.',
        'test_labels_opened': False}
    save(output / 'protocol.json', protocol)
    return protocol


def predict_frozen(data, baseline, graph, risk, output, device='cpu'):
    output, baseline, graph, risk = map(Path, [output, baseline, graph, risk])
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol['source_sha256'] != source_hashes() or protocol['artifact_sha256'] != frozen_sources(baseline, graph, risk):
        raise ValueError('Evaluation source or frozen inputs changed after protocol freeze')
    if (output / 'predictions.parquet').exists() or (output / 'predictions-frozen.json').exists():
        raise ValueError('Prediction output exists; never overwrite a completed run')
    started = time.perf_counter()
    torch.set_num_threads(4)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    config = protocol['data_config']
    events = label_free_events(data, config)
    masks = split_events(events, config)
    replay = events.loc[masks['replay']].reset_index(drop=True)
    cold = masks['cold'][masks['replay']]
    test = replay.timestamp.ge(pd.Timestamp(config['test_start'])).to_numpy()
    rows = np.flatnonzero(test)
    features = causal_features(replay)
    predictions = replay.loc[test, ['event_id', 'timestamp']].copy().reset_index(drop=True)
    predictions['cold'] = cold[test]
    baseline_report = json.loads((baseline / 'report.json').read_text())
    predictions['train_prior'] = baseline_report['train_positives'] / baseline_report['train_events']
    timing = {}
    for name in ['transaction_xgboost', 'history_xgboost']:
        begin = time.perf_counter()
        preprocessing = joblib.load(baseline / (name + '.joblib'))
        model = XGBClassifier()
        model.load_model(baseline / (name + '.ubj'))
        model.set_params(n_jobs=4)
        predictions[name] = model.predict_proba(preprocessing.transform(features.loc[test]).astype(np.float32))[:, 1]
        timing[name] = time.perf_counter() - begin
    x = graph_features(features, joblib.load(graph / 'preprocessing.joblib'))
    index = TemporalNeighborhood(replay)
    membership = pd.read_parquet(graph / 'membership.parquet', columns=['event_id'])
    before = replay.timestamp.lt(pd.Timestamp(config['test_start'])).to_numpy()
    if not np.array_equal(replay.loc[before, 'event_id'], membership.event_id):
        raise ValueError('Original development replay membership changed')
    if not np.array_equal(x[before], np.load(graph / 'features.npy')):
        raise ValueError('Original development feature vectors changed')
    replay[['event_id', 'timestamp', 'source', 'target']].to_parquet(output / 'membership.parquet', index=False)
    np.save(output / 'features.npy', x)
    joblib.dump(index, output / 'neighborhood.joblib')
    del features, events, membership
    gc.collect()
    tensor = torch.as_tensor(x, device=device)
    for neighbors, name in [(0, 'history_mlp'), (8, 'temporal_attention_8'), (32, 'temporal_attention_32')]:
        begin = time.perf_counter()
        state = torch.load(graph / f'neighbors-{neighbors}-latest.pt', weights_only=True, map_location=device)
        model = TemporalGraphClassifier(state['features'], state['config']['hidden'], neighbors).to(device)
        model.load_state_dict(state['state_dict'])
        predictions[name] = predict(model, tensor, index, rows, 4096)
        timing[name] = time.perf_counter() - begin
        print(json.dumps({'predicted': name, 'events': len(rows), 'seconds': timing[name]}), flush=True)
    for name in ['history_xgboost', 'temporal_attention_32']:
        policy = json.loads((risk / (name + '-calibrator.json')).read_text())
        predictions[name + '_calibrated'] = MonotoneCalibration(**policy['calibrator']).transform(predictions[name], predictions.timestamp)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    freeze_record = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': digest(output / 'protocol.json'),
        'test_labels_opened': False, 'events': len(predictions), 'replay_events': len(replay),
        'source_sha256': source_hashes(), 'artifact_sha256': {name: digest(output / name) for name in
            ['predictions.parquet', 'features.npy', 'neighborhood.joblib', 'membership.parquet']},
        'runtime': {'seconds': time.perf_counter() - started, 'model_seconds': timing, 'device': device,
                    'cpu_threads': 4, 'peak_process_bytes': peak_memory_bytes(),
                    'peak_torch_allocated_bytes': int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0}}
    save(output / 'predictions-frozen.json', freeze_record)
    return freeze_record


def heldout_labels(data, config):
    offset, pieces = 0, []
    for chunk in pd.read_csv(data, names=NAMES, header=0, usecols=['timestamp', 'label'], dtype='string', chunksize=250000):
        chunk['event_id'] = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        chunk['timestamp'] = pd.to_datetime(chunk.timestamp, format='%Y/%m/%d %H:%M', errors='raise')
        selected = chunk.loc[chunk.timestamp >= pd.Timestamp(config['test_start'])].copy()
        selected['label'] = pd.to_numeric(selected.label, errors='raise').astype('int8')
        if not selected.label.isin([0, 1]).all():
            raise ValueError('Nonbinary final label')
        pieces.append(selected)
    return pd.concat(pieces).sort_values(['timestamp', 'event_id']).reset_index(drop=True)


def score_frozen(data, risk, output):
    output, risk = Path(output), Path(risk)
    if (output / 'report.json').exists():
        raise ValueError('Final result exists; never rerun or overwrite scoring')
    protocol = json.loads((output / 'protocol.json').read_text())
    frozen = json.loads((output / 'predictions-frozen.json').read_text())
    if digest(output / 'protocol.json') != frozen['protocol_sha256'] or source_hashes() != frozen['source_sha256']:
        raise ValueError('Protocol or code changed before scoring')
    for name, expected in frozen['artifact_sha256'].items():
        if digest(output / name) != expected:
            raise ValueError('Frozen prediction input changed')
    config = protocol['data_config']
    if digest(data) != config['sha256']:
        raise ValueError('Dataset changed before scoring')
    opened = datetime.now(timezone.utc).isoformat()
    predictions = pd.read_parquet(output / 'predictions.parquet')
    labels = heldout_labels(data, config)
    if not np.array_equal(labels.event_id, predictions.event_id) or not np.array_equal(labels.timestamp, predictions.timestamp):
        raise ValueError('Final labels do not align with frozen predictions')
    predictions['label'] = labels.label
    labels.to_parquet(output / 'labels.parquet', index=False)
    groups = {'all': np.ones(len(labels), dtype=bool), 'cold': predictions.cold.to_numpy(), 'warm': ~predictions.cold.to_numpy()}
    groups.update({str(day): predictions.timestamp.dt.date.eq(day).to_numpy() for day in sorted(predictions.timestamp.dt.date.unique())})
    names = ARMS + ['history_xgboost_calibrated', 'temporal_attention_32_calibrated']
    report = {
        'stage': 'final_frozen_temporal_holdout', 'test_scored': True, 'completed': True,
        'labels_opened_at_utc': opened, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': digest(output / 'protocol.json'), 'prediction_freeze_sha256': digest(output / 'predictions-frozen.json'),
        'events': len(labels), 'positives': int(labels.label.sum()), 'first_event': str(labels.timestamp.min()),
        'last_event': str(labels.timestamp.max()), 'runtime': frozen['runtime'], 'models': {}, 'policies': {}}
    for name in names:
        report['models'][name] = {group: evaluate(predictions.loc[mask, 'label'], predictions.loc[mask, name],
                                                 predictions.loc[mask, 'event_id'], config['alert_fraction'])
                                  for group, mask in groups.items() if mask.any()}
    for name in ['history_xgboost', 'temporal_attention_32']:
        filename = name + '-calibrator.json'
        if digest(risk / filename) != protocol['artifact_sha256']['risk/' + filename]:
            raise ValueError('Frozen calibrator changed')
        policy = json.loads((risk / filename).read_text())
        frame = predictions[['event_id', 'timestamp', 'label', name + '_calibrated']].rename(columns={name + '_calibrated': 'score'})
        selected, hours = closed_hour_alerts(frame, policy['threshold'], policy['capacity_fraction'], frame.timestamp.max().ceil('h') + pd.Timedelta(hours=1), policy['seed'])
        hits = int(frame.loc[selected, 'label'].sum())
        report['policies'][name] = {'threshold': policy['threshold'], 'alerts': int(selected.sum()), 'true_alerts': hits,
            'recall': hits / report['positives'] if report['positives'] else None, 'precision': hits / int(selected.sum()) if selected.any() else None,
            'hourly': hours, 'capacity_fraction': policy['capacity_fraction'], 'policy_refitted': False}
        frame['selected'] = selected
        frame.to_parquet(output / (name + '-alerts.parquet'), index=False)
    report['limitations'] = [
        'Single temporal test on independent synthetic AML data; no claim of real financial validity.',
        'Later test events may use earlier unlabeled test interactions, but never same-minute or future interactions.',
        'Cold is a fixed account holdout at original fit; those accounts may develop history later.',
        'All sparse tail days retained; per-day estimates may have few or zero positives.',
        'No model, hyperparameter, calibrator or threshold changed using final outcomes.',
        'Hourly decisions can wait up to one hour; most alerts may be false positives.',
        'One neural training seed; no confidence interval or bitwise CUDA retraining guarantee.']
    report['artifact_sha256'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    save(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'predict', 'score'])
    for name in ['baseline', 'graph', 'risk', 'data', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--config', default='configs/ibm-hi-small.json')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.phase == 'freeze':
        result = freeze(json.loads(Path(args.config).read_text()), args.baseline, args.graph, args.risk, args.output)
    elif args.phase == 'predict':
        result = predict_frozen(args.data, args.baseline, args.graph, args.risk, args.output, args.device)
    else:
        result = score_frozen(args.data, args.risk, args.output)
    print(json.dumps({k: result[k] for k in ['stage', 'events', 'positives', 'completed', 'runtime'] if k in result}), flush=True)


if __name__ == '__main__':
    main()
