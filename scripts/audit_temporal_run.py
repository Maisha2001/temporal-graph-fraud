"""Audit trusted local run artifacts without training or reading held-out rows."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from temporal_fraud.data import digest
from temporal_fraud.graph_model import TemporalGraphClassifier, predict


def independent_metrics(frame, fraction):
    grouped = frame.groupby('score', sort=True).label.agg(['sum', 'count']).iloc[::-1]
    positives, counts = grouped['sum'].to_numpy(), grouped['count'].to_numpy()
    total = int(positives.sum())
    cumulative = np.cumsum(counts)
    ap = float(np.sum(np.cumsum(positives) / cumulative * positives) / total)
    budget = int(np.floor(len(frame) * fraction))
    chosen = np.maximum(0, np.minimum(counts, budget - (cumulative - counts)))
    expected = float(np.sum(positives * chosen / counts))
    return {'average_precision': ap, 'recall_at_capacity': expected / total,
            'precision_at_capacity': expected / budget,
            'brier': float(np.mean((frame.score.to_numpy(dtype=float) - frame.label) ** 2))}


def audit(run, baseline, repository):
    run, baseline, repository = Path(run), Path(baseline), Path(repository)
    report = json.loads((run / 'report.json').read_text())
    for name, expected in report['artifact_sha256'].items():
        assert digest(run / name) == expected, name
    original = json.loads((repository / 'results/baseline-v1/report.json').read_text())
    for name, expected in original['artifact_sha256'].items():
        assert digest(baseline / name) == expected, name
    membership = pd.read_parquet(run / 'membership.parquet')
    prior = pd.read_parquet(baseline / 'membership.parquet')
    pd.testing.assert_frame_equal(membership[prior.columns], prior)
    assert membership.timestamp.max() < pd.Timestamp(report['data_config']['test_start'])
    train = membership.loc[membership.train]
    assert ((train.timestamp + pd.Timedelta(hours=report['data_config']['label_delay_hours'])) < pd.Timestamp(report['data_config']['fit_at'])).all()
    assert not train.cold.any()
    index = joblib.load(run / 'neighborhood.joblib')
    queries = np.unique(np.r_[0, np.linspace(0, len(membership) - 1, 64, dtype=int)])
    sampled, ages, directions = index.sample(queries, 32)
    exhaustive_checks = 0
    for i, row in enumerate(queries):
        for role, endpoint in enumerate(['source', 'target']):
            node = membership.iloc[row][endpoint]
            past = np.flatnonzero(((membership.source == node) | (membership.target == node)) &
                                 (membership.timestamp < membership.timestamp.iloc[row]))
            selected = sampled[i, role][sampled[i, role] >= 0]
            assert len(selected) == min(32, len(past))
            assert len(np.unique(selected)) == len(selected)
            assert np.isin(selected, past).all()
            if len(past) <= 32:
                np.testing.assert_array_equal(np.sort(selected), past)
            elif len(selected):
                cutoff = np.sort(index.minutes[past])[-32]
                assert (index.minutes[selected] >= cutoff).all()
            exhaustive_checks += 1
    torch.set_num_threads(4)
    x = torch.tensor(np.load(run / 'features.npy'))
    validation_rows = np.flatnonzero(membership.validation.to_numpy())
    chosen = np.unique(np.r_[0, np.linspace(0, len(validation_rows) - 1, 256, dtype=int)])
    base_predictions = pd.read_parquet(baseline / 'history_xgboost-validation.parquet')
    errors = {}
    for name, model_report in report['models'].items():
        predictions = pd.read_parquet(run / (name + '-validation.parquet'))
        pd.testing.assert_frame_equal(predictions[['event_id', 'timestamp', 'label', 'cold']], base_predictions[['event_id', 'timestamp', 'label', 'cold']])
        for group, frame in [('validation', predictions), ('cold', predictions.loc[predictions.cold]), ('warm', predictions.loc[~predictions.cold])]:
            metrics = independent_metrics(frame, report['data_config']['alert_fraction'])
            for key, value in metrics.items():
                assert abs(value - model_report[group][key]) < 1e-10, (name, group, key)
        neighbors = 0 if name == 'history_mlp' else int(name.rsplit('_', 1)[1])
        state = torch.load(run / f'neighbors-{neighbors}-latest.pt', map_location='cpu', weights_only=True)
        assert state['epoch'] == report['config']['epochs'] and state['updates'] == int(np.ceil(len(train) / report['config']['batch_size'])) * report['config']['epochs']
        assert state['fingerprint'] == report['input_manifest_sha256']
        model = TemporalGraphClassifier(x.shape[1], report['config']['hidden'], neighbors)
        model.load_state_dict(state['state_dict'])
        restored = predict(model, x, index, validation_rows[chosen])
        error = float(np.max(np.abs(restored - predictions.score.to_numpy()[chosen])))
        assert error < 1e-5, (name, error)
        errors[name] = error
    return {'passed': True, 'test_scored': False, 'original_baseline_artifacts_verified': len(original['artifact_sha256']),
            'graph_artifacts_verified': len(report['artifact_sha256']), 'membership_rows_identical': len(membership),
            'exhaustive_endpoint_queries': exhaustive_checks, 'independently_recomputed_metric_groups': 9,
            'cpu_checkpoint_predictions_per_model': len(chosen), 'cpu_vs_saved_gpu_max_absolute_difference': errors,
            'run_report_sha256': digest(run / 'report.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.run, args.baseline, Path(__file__).resolve().parents[1])
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('Refusing to overwrite audit')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
