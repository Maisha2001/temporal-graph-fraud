import json

import numpy as np
import pandas as pd
import pytest

from temporal_fraud.risk import (MonotoneCalibration, available_labels, capacity_threshold,
                                 closed_hour_alerts, population_shift, reliability)


def predictions():
    return pd.DataFrame({'event_id': np.arange(200), 'timestamp': pd.date_range('2022-09-06', periods=200, freq='min'),
                         'score': np.linspace(0.05, 0.95, 200), 'label': np.tile([0, 0, 0, 1], 50)})


def test_feedback_is_strict_and_later_labels_do_not_fit():
    frame = predictions()
    instant = '2022-09-07 02:00'
    selected = available_labels(frame.timestamp, instant, 24)
    assert selected.sum() == 120 and not selected[120]
    first = MonotoneCalibration.fit(frame, instant, 24, '2022-09-06', min_positives=10)
    frame['label'] = frame.label.astype(object)
    frame.loc[120:, 'label'] = 'UNAVAILABLE'
    second = MonotoneCalibration.fit(frame, instant, 24, '2022-09-06', min_positives=10)
    assert first == second
    assert first.examples == 120 and pd.Timestamp(first.latest_label_available_at) < pd.Timestamp(instant)
    with pytest.raises(ValueError, match='before its fit'):
        first.transform([0.3], ['2022-09-07 01:59'])
    assert MonotoneCalibration(**json.loads(json.dumps(first.to_dict()))) == first


def test_in_sample_predictions_and_unmatured_labels_excluded():
    frame = predictions()
    fit = MonotoneCalibration.fit(frame, '2022-09-08', 24, '2022-09-06 01:00', min_positives=10)
    assert fit.examples == 140
    with pytest.raises(ValueError, match='Insufficient'):
        MonotoneCalibration.fit(frame, '2022-09-06', 24, '2022-09-06')
    with pytest.raises(ValueError):
        available_labels(frame.timestamp, '2022-09-08', -1)


def test_calibration_is_monotone_and_improves_known_miscalibration():
    rng = np.random.default_rng(4)
    truth = np.tile([0.01, 0.05, 0.15, 0.4], 2000)
    frame = pd.DataFrame({'event_id': np.arange(len(truth)), 'timestamp': pd.Timestamp('2022-09-06'),
                          'score': np.minimum(0.99, truth * 2 + 0.1), 'label': rng.binomial(1, truth)})
    fit = MonotoneCalibration.fit(frame, '2022-09-08', 24, '2022-09-06')
    grid = fit.apply(np.linspace(0, 1, 100))
    assert np.isfinite(grid).all() and np.all(np.diff(grid) >= 0)
    calibrated = fit.apply(frame.score)
    assert np.mean((calibrated - truth) ** 2) < np.mean((frame.score - truth) ** 2) / 10


def test_alert_capacity_ties_are_order_independent_and_label_free():
    frame = predictions()
    frame['score'] = 0.1
    selected, hours = closed_hour_alerts(frame, 0.1, 0.1, '2022-09-07')
    assert all(h['alerts'] <= h['budget'] == int(h['events'] * 0.1) for h in hours)
    assert selected.sum() == sum(h['budget'] for h in hours)
    reordered = frame.iloc[::-1].copy()
    reordered['label'] = 'UNKNOWN'
    other, _ = closed_hour_alerts(reordered, 0.1, 0.1, '2022-09-07')
    assert set(frame.event_id[selected]) == set(reordered.event_id[other])
    assert not np.array_equal(frame.event_id[selected][:6], np.arange(6))


def test_threshold_is_frozen_and_hour_must_close():
    frame = predictions()
    threshold = capacity_threshold(frame.score[:100], 0.1)
    assert threshold == frame.score.iloc[90]
    selected, _ = closed_hour_alerts(frame, threshold, 0.1, '2022-09-07')
    assert (frame.score[selected] >= threshold).all()
    with pytest.raises(ValueError, match='has not closed'):
        closed_hour_alerts(frame, threshold, 0.1, '2022-09-06 01:30')
    assert capacity_threshold([0.9], 0.01) is None
    assert not closed_hour_alerts(frame, None, 0.01, '2022-09-07')[0].any()


@pytest.mark.parametrize('scores', [[np.nan], [-1], [1.1]])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError):
        capacity_threshold(scores, 0.1)


def test_reliability_reports_empty_bins_and_true_error():
    report = reliability([0, 1], [0.0, 1.0], np.linspace(0, 1, 100))
    assert report['brier'] == 0 and report['expected_calibration_error'] == 0
    assert sum(b['events'] for b in report['bins']) == 2
    assert any(b['events'] == 0 for b in report['bins'])


def test_drift_handles_constant_reference_and_disjoint_support():
    assert population_shift([1] * 20, [1] * 20)['population_stability_index'] == 0
    assert population_shift([1] * 20, [2] * 20)['population_stability_index'] > 0
    reference = np.linspace(0, 1, 100)
    assert population_shift(reference, reference)['population_stability_index'] == 0
    assert np.isfinite(population_shift(reference, reference + 10)['population_stability_index'])


def test_calibration_pipeline_uses_only_matured_labels_and_later_events():
    pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    from temporal_fraud.robustness import calibration_analysis
    frame = pd.DataFrame({'event_id': np.arange(1000), 'timestamp': pd.date_range('2022-09-06', periods=1000, freq='3min'),
                          'score': np.tile(np.linspace(0.02, 0.9, 100), 10), 'label': np.tile([0, 0, 0, 1], 250), 'cold': False})
    config = {'calibration_at': '2022-09-07T12:00:00', 'calibration_delay_hours': 24,
              'calibration_min_positives': 10, 'evaluation_end': '2022-09-08', 'alert_fraction': 0.1, 'seed': 42}
    report, result = calibration_analysis(frame, config, '2022-09-06')
    assert report['calibrator']['examples'] == 240
    assert result.timestamp.min() == pd.Timestamp(config['calibration_at'])
    assert result.timestamp.max() < pd.Timestamp(config['evaluation_end'])
    changed = frame.copy()
    changed.loc[changed.timestamp >= pd.Timestamp(config['calibration_at']), 'label'] = 1 - changed.label
    other, alerts = calibration_analysis(changed, config, '2022-09-06')
    assert other['calibrator'] == report['calibrator'] and other['threshold'] == report['threshold']
    np.testing.assert_array_equal(result.alert, alerts.alert)
    assert all(h['alerts'] <= h['budget'] for h in report['alerts']['hourly'])


def test_graph_intervention_keeps_edges_causal_and_only_removes_requested_signal():
    pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    from temporal_fraud.robustness import HistoryIntervention
    from temporal_fraud.temporal import TemporalNeighborhood
    from test_pipeline import events
    index = TemporalNeighborhood(events(pd.date_range('2022-09-01', periods=5, freq='min')))
    original = index.sample([4], 2)
    zeroed = HistoryIntervention(index, 'no_edge_time').sample([4], 2)
    np.testing.assert_array_equal(zeroed[0], original[0])
    np.testing.assert_array_equal(zeroed[2], original[2])
    assert not zeroed[1].any()
    absent = HistoryIntervention(index, 'no_neighbors').sample([4], 2)
    assert (absent[0] == -1).all()
    with pytest.raises(ValueError):
        HistoryIntervention(index, 'unknown')


def test_complete_study_runs_on_independent_fixture_and_preserves_inputs(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    import csv
    from temporal_fraud.data import HEADER, digest, entity_bucket, load_development
    from temporal_fraud.experiment import run as trees
    from temporal_fraud.graph_experiment import run as graph
    from temporal_fraud.robustness import run, write_json
    from pathlib import Path
    torch.set_num_threads(2)
    root = Path(__file__).parents[1]
    data_config = json.loads((root / 'configs/ibm-hi-small.json').read_text())
    data_config.update({'trees': 2, 'max_depth': 1, 'threads': 2, 'alert_fraction': 0.5})
    warm = next(str(i) for i in range(100) if entity_bucket('1:' + str(i)) != 0)
    cold = next(str(i) for i in range(100) if entity_bucket('1:' + str(i)) == 0)
    times = list(pd.date_range('2022-09-01', periods=120, freq='h')) + list(pd.date_range('2022-09-06', periods=96, freq='30min'))
    data = tmp_path / 'events.csv'
    with data.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(HEADER)
        for i, timestamp in enumerate(times):
            source = cold if i >= 120 and i % 3 == 0 else warm
            writer.writerow([timestamp.strftime('%Y/%m/%d %H:%M'), 1, source, 1, warm, i+1, 'USD', i+1, 'USD', 'Wire', int(i % 4 == 0)])
        writer.writerow(['2022/09/08 00:00', 1, cold, 1, warm, 1, 'USD', 1, 'USD', 'Wire', 'UNREAD'])
    data_config['sha256'] = digest(data)
    events, _ = load_development(data, data_config)
    baseline, neural, output = tmp_path / 'baseline', tmp_path / 'graph', tmp_path / 'study'
    base = trees(events, data_config, baseline)
    base['artifact_sha256'] = {p.name: digest(p) for p in baseline.iterdir()}
    write_json(baseline / 'report.json', base)
    graph_config = {'seed': 42, 'epochs': 1, 'batch_size': 32, 'hidden': 8, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'neighbors': [32], 'threads': 2}
    graph(graph_config, data_config, data, neural)
    protocol = json.loads((root / 'configs/robustness-v1.json').read_text())
    protocol.update({'calibration_min_positives': 2, 'alert_fraction': 0.5})
    before = {str(p): digest(p) for folder in [baseline, neural] for p in folder.iterdir()}
    result = run(data, baseline, neural, output, protocol, data_config, graph_config)
    assert result['completed'] and not result['test_scored']
    assert result['delays']['0']['training_events'] > result['delays']['24']['training_events'] > result['delays']['48']['training_events']
    assert set(result['ablations']) == {'no_neighbors', 'no_edge_time'}
    assert before == {str(p): digest(p) for folder in [baseline, neural] for p in folder.iterdir()}
    study_before = digest(output / 'study.json')
    assert run(data, baseline, neural, output, protocol, data_config, graph_config, resume=True) == result
    assert digest(output / 'study.json') == study_before
