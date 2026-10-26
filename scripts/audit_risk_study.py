"""Independently check trusted local risk-study artifacts without retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from audit_temporal_run import independent_metrics
from temporal_fraud.data import digest


def audit(run, baseline, graph):
    run, baseline, graph = Path(run), Path(baseline), Path(graph)
    report = json.loads((run / 'study.json').read_text())
    assert report['completed'] and report['test_scored'] is False
    for name, expected in report['artifact_sha256'].items():
        assert digest(run / name) == expected, name
    for name, expected in report['frozen_artifact_sha256'].items():
        prefix, relative = name.split('/', 1)
        assert digest((baseline if prefix == 'baseline' else graph) / relative) == expected, name
    original = pd.read_parquet(baseline / 'history_xgboost-validation.parquet')
    checks = 0
    for delay in [0, 48]:
        arm = run / f'delay-{delay}'
        membership = pd.read_parquet(arm / 'graph/membership.parquet')
        mask = ((membership.timestamp + pd.Timedelta(hours=delay)) < pd.Timestamp(report['data_config']['fit_at'])) & ~membership.cold
        np.testing.assert_array_equal(mask, membership.train)
        assert mask.sum() == report['delays'][str(delay)]['training_events']
        assert membership.loc[mask, 'label'].sum() == report['delays'][str(delay)]['training_positives']
        assert membership.timestamp.max() < pd.Timestamp(report['data_config']['test_start'])
        for model, folder in [('history_xgboost', 'trees'), ('temporal_attention_32', 'graph')]:
            predictions = pd.read_parquet(arm / folder / (model + '-validation.parquet'))
            pd.testing.assert_frame_equal(predictions[['event_id', 'timestamp', 'label', 'cold']], original[['event_id', 'timestamp', 'label', 'cold']])
            for group, frame in [('validation', predictions), ('cold', predictions.loc[predictions.cold]), ('warm', predictions.loc[~predictions.cold])]:
                for metric, value in independent_metrics(frame, report['protocol']['alert_fraction']).items():
                    assert abs(value - report['delays'][str(delay)][model][group][metric]) < 1e-10
                checks += 1
    policy_checks, calibration_checks = 0, {}
    for model in report['protocol']['calibration_models']:
        source = pd.read_parquet((baseline if model == 'history_xgboost' else graph) / (model + '-validation.parquet'))
        record = report['calibration'][model]
        calibration = record['calibrator']
        as_of = pd.Timestamp(calibration['fitted_at'])
        feedback = source.loc[(source.timestamp + pd.Timedelta(hours=calibration['delay_hours']) < as_of) &
                              (source.timestamp >= pd.Timestamp(calibration['model_fit_at']))]
        assert len(feedback) == calibration['examples'] and feedback.label.sum() == calibration['positives']
        assert hashlib.sha256(feedback.event_id.to_numpy(dtype='<i8').tobytes()).hexdigest() == record['feedback_event_ids_sha256']
        predicted = pd.read_parquet(run / (model + '-calibrated-policy.parquet'))
        assert not set(feedback.event_id).intersection(predicted.event_id)
        assert predicted.timestamp.min() >= as_of and predicted.timestamp.max() < pd.Timestamp(report['protocol']['evaluation_end'])
        expected = expit(calibration['slope'] * logit(np.clip(predicted.raw_score.to_numpy(dtype=float), 1e-7, 1-1e-7)) + calibration['intercept'])
        np.testing.assert_allclose(predicted.score, expected, rtol=0, atol=1e-15)
        reference = expit(calibration['slope'] * logit(np.clip(feedback.score.to_numpy(dtype=float), 1e-7, 1-1e-7)) + calibration['intercept'])
        budget = int(len(reference) * report['protocol']['alert_fraction'])
        assert record['threshold'] == np.sort(reference)[-budget]
        for kind, scores in [('raw', predicted.raw_score.to_numpy(dtype=float)), ('calibrated', expected)]:
            assert abs(np.mean((scores - predicted.label) ** 2) - record[kind]['brier']) < 1e-12
            clipped = np.clip(scores, 1e-12, 1-1e-12)
            assert abs(log_loss(predicted.label, clipped, labels=[0, 1]) - record[kind]['log_loss']) < 1e-10
            assert sum(x['events'] for x in record[kind]['bins']) == len(predicted)
        for hour, batch in predicted.groupby(predicted.timestamp.dt.floor('h')):
            capacity = int(len(batch) * report['protocol']['alert_fraction'])
            eligible = batch.loc[batch.score >= record['threshold']]
            keys = {int(row.event_id): hashlib.sha256(f"{report['protocol']['seed']}:{row.event_id}".encode()).hexdigest()[:16]
                    for row in eligible.itertuples()}
            expected_ids = [int(row.event_id) for row in sorted(eligible.itertuples(), key=lambda x: (-x.score, keys[int(x.event_id)]))[:capacity]]
            assert set(expected_ids) == set(batch.loc[batch.alert, 'event_id'])
            assert batch.alert.sum() <= capacity
            assert hour + pd.Timedelta(hours=1) <= pd.Timestamp(report['protocol']['evaluation_end'])
            policy_checks += 1
        actual = record['alerts']
        assert int(predicted.alert.sum()) == actual['count']
        assert int(predicted.loc[predicted.alert, 'label'].sum()) == actual['true_alerts']
        calibration_checks[model] = {'feedback_events': len(feedback), 'feedback_positives': int(feedback.label.sum()),
                                     'evaluation_events': len(predicted), 'selected_alerts': actual['count']}
    for mode in report['protocol']['graph_interventions']:
        frame = pd.read_parquet(run / (mode + '-validation.parquet'))
        pd.testing.assert_frame_equal(frame[['event_id', 'timestamp', 'label', 'cold']], original[['event_id', 'timestamp', 'label', 'cold']])
        for group, selected in [('validation', frame), ('cold', frame.loc[frame.cold]), ('warm', frame.loc[~frame.cold])]:
            for metric, value in independent_metrics(selected, report['protocol']['alert_fraction']).items():
                assert abs(value - report['ablations'][mode]['metrics'][group][metric]) < 1e-10
            checks += 1
    return {'passed': True, 'test_scored': False, 'new_artifacts_verified': len(report['artifact_sha256']),
            'frozen_artifacts_unchanged': len(report['frozen_artifact_sha256']), 'independent_metric_groups': checks,
            'closed_hour_policies_verified': policy_checks, 'calibration_partitions': calibration_checks,
            'study_sha256': digest(run / 'study.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['run', 'baseline', 'graph', 'output']:
        parser.add_argument('--' + argument, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.baseline, args.graph)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite existing audit')
    output.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(result))
