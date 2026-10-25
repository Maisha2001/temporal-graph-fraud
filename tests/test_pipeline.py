import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from temporal_fraud.data import HEADER, entity_bucket, load_development, split_events
from temporal_fraud.experiment import run
from temporal_fraud.features import HISTORY, causal_features
from temporal_fraud.metrics import evaluate


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / 'configs/ibm-hi-small.json').read_text())


def events(times, source=None, target=None):
    n = len(times)
    return pd.DataFrame({'event_id': np.arange(n), 'timestamp': pd.to_datetime(times),
                         'source': source or ['1:A'] * n, 'target': target or ['1:B'] * n,
                         'paid': np.arange(n) + 1.0, 'received': np.arange(n) + 1.0,
                         'receiving_currency': ['USD'] * n, 'payment_currency': ['USD'] * n,
                         'format': ['Wire'] * n, 'label': np.arange(n) % 2})


def test_same_timestamp_edges_are_invisible():
    frame = events(['2022-09-01 00:00', '2022-09-01 00:00', '2022-09-01 00:01'])
    history = causal_features(frame)
    np.testing.assert_allclose(history['log_source_out'], [0, 0, np.log1p(2)])
    np.testing.assert_allclose(history['source_age_minutes'], [-1, -1, 1])


def test_future_events_and_labels_do_not_change_past_features():
    frame = events(pd.date_range('2022-09-01', periods=5, freq='min'))
    expected = causal_features(frame.iloc[:3])
    frame.loc[3:, ['source', 'target']] = ['0:NEW', '0:FUTURE']
    frame['label'] = 1 - frame['label']
    actual = causal_features(frame).iloc[:3]
    pd.testing.assert_frame_equal(expected, actual)


def test_permutation_inside_timestamp_does_not_change_features():
    frame = events(['2022-09-01'] * 3 + ['2022-09-02'])
    expected = causal_features(frame)
    reordered = frame.iloc[[2, 0, 1, 3]]
    actual = causal_features(reordered).sort_index()
    pd.testing.assert_frame_equal(expected, actual)


def test_self_loop_counts_once_per_direction():
    frame = events(['2022-09-01', '2022-09-02'], ['1:A'] * 2, ['1:A'] * 2)
    features = causal_features(frame)
    assert features.loc[1, 'log_source_out'] == pytest.approx(np.log(2))
    assert features.loc[1, 'log_target_in'] == pytest.approx(np.log(2))


@pytest.mark.parametrize('mutation', ['unordered', 'duplicate_id'])
def test_invalid_replay_rejected(mutation):
    frame = events(['2022-09-01', '2022-09-02'])
    if mutation == 'unordered':
        frame = frame.iloc[::-1]
    else:
        frame['event_id'] = 0
    with pytest.raises(ValueError):
        causal_features(frame)


def test_label_delay_and_entity_holdout(config):
    warm = next(f'1:{i}' for i in range(100) if entity_bucket(f'1:{i}') != 0)
    cold = next(f'1:{i}' for i in range(100) if entity_bucket(f'1:{i}') == 0)
    frame = events(['2022-09-04 23:59', '2022-09-05 00:00', '2022-09-05 23:59', '2022-09-06 00:00', '2022-09-07 00:00'], [warm, warm, cold, cold, warm], [warm] * 5)
    masks = split_events(frame, config)
    assert masks['train'].tolist() == [True, False, False, False, False]
    assert masks['validation'].tolist() == [False, False, False, True, True]
    assert masks['replay'].tolist() == [True, True, False, True, True]
    replay = frame.loc[masks['replay']].reset_index(drop=True)
    assert causal_features(replay).iloc[2]['log_source_out'] == 0


def write_csv(path, rows, header=HEADER):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)


def row(timestamp='2022/09/01 00:00', label='0', paid='1'):
    return [timestamp, '010', 'A', '10', 'B', '1', 'USD', paid, 'USD', 'Wire', label]


def test_loader_canonical_ids_sort_and_reserved_labels_unread(tmp_path, config):
    path = tmp_path / 'sample.csv'
    write_csv(path, [row('2022/09/07 00:00'), row(), row('2022/09/08 00:00', 'INTENTIONALLY_UNREAD')])
    data, counts = load_development(path, config, verify=False)
    assert data['event_id'].tolist() == [1, 0]
    assert set(data['source']) == {'10:A'}
    assert set(data['target']) == {'10:B'}
    assert counts == {'raw_rows': 3, 'reserved_test_rows': 1, 'development_rows': 2}


@pytest.mark.parametrize('bad_row', [row(label='2'), row(paid='-1'), row(paid='inf'), row(timestamp='bad')])
def test_bad_data_rejected(tmp_path, config, bad_row):
    path = tmp_path / 'bad.csv'; write_csv(path, [bad_row])
    with pytest.raises((ValueError, TypeError)):
        load_development(path, config, verify=False)


def test_digest_and_schema_rejected(tmp_path, config):
    path = tmp_path / 'bad.csv'; write_csv(path, [row()], HEADER[::-1])
    with pytest.raises(ValueError, match='digest'):
        load_development(path, config)
    with pytest.raises(ValueError, match='header'):
        load_development(path, config, verify=False)


def test_alert_budget_ties_are_independent_of_input_order():
    result = evaluate([0, 1, 1, 0], [0.5] * 4, [4, 1, 2, 3], fraction=0.5)
    permuted = evaluate([1, 0, 0, 1], [0.5] * 4, [2, 3, 4, 1], fraction=0.5)
    assert result == permuted
    assert result['alerts'] == 2 and result['recall_at_capacity'] == 1


def test_no_positives_and_zero_budget_report_nulls():
    result = evaluate([0, 0], [0.1, 0.2], [1, 2], fraction=0.01)
    assert result['average_precision'] is None and result['recall_at_capacity'] is None
    assert result['alerts'] == 0 and result['precision_at_capacity'] is None


@pytest.mark.parametrize('scores', [[float('nan')], [1.1], [-0.1]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        evaluate([1], scores, [0])


def test_small_end_to_end_training_unknown_categories_and_saved_model(tmp_path, config):
    import joblib
    from xgboost import XGBClassifier
    warm = next(f'1:{i}' for i in range(100) if entity_bucket(f'1:{i}') != 0)
    dates = list(pd.date_range('2022-09-01', periods=40, freq='h')) + list(pd.date_range('2022-09-06', periods=10, freq='h'))
    frame = events(dates, [warm] * 50, [warm] * 50)
    frame.loc[40:, 'format'] = 'VALIDATION_ONLY'
    config.update(trees=3, threads=1)
    report = run(frame, config, tmp_path)
    assert report['test_scored'] is False and report['train_events'] == 40
    preprocessing = joblib.load(tmp_path / 'history_xgboost.joblib')
    assert not any('VALIDATION_ONLY' in name for name in preprocessing.get_feature_names_out())
    model = XGBClassifier(); model.load_model(tmp_path / 'history_xgboost.ubj')
    actual = model.predict_proba(preprocessing.transform(causal_features(frame).iloc[40:]))[:, 1]
    saved = pd.read_parquet(tmp_path / 'history_xgboost-validation.parquet')['score']
    np.testing.assert_array_equal(actual, saved)
