import json
import numpy as np
import pandas as pd
import pytest

from temporal_fraud.data import HEADER, digest
from temporal_fraud.release_eval import label_free_events, heldout_labels
from temporal_fraud.features import causal_features
from temporal_fraud.temporal import TemporalNeighborhood


def csv_file(tmp_path, labels):
    path = tmp_path / 'authored.csv'
    rows = [['2022/09/07 23:59', '1', 'A', '2', 'B', '10', 'USD', '10', 'USD', 'ACH', labels[0]],
            ['2022/09/08 00:00', '1', 'A', '2', 'B', '20', 'USD', '20', 'USD', 'ACH', labels[1]],
            ['2022/09/08 00:00', '2', 'B', '1', 'A', '30', 'USD', '30', 'USD', 'ACH', labels[2]],
            ['2022/09/18 12:00', '1', 'A', '3', 'C', '5', 'USD', '5', 'USD', 'ACH', labels[3]]]
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)
    return path, {'sha256': digest(path), 'test_start': '2022-09-08'}


def test_prediction_loader_never_parses_labels_and_keeps_sparse_tail(tmp_path):
    path, config = csv_file(tmp_path, ['unread', 'bad', 'hidden', 'never-read'])
    frame = label_free_events(path, config)
    assert 'label' not in frame and len(frame) == 4
    assert frame.timestamp.iloc[-1] == pd.Timestamp('2022-09-18T12:00')
    with pytest.raises(ValueError):
        heldout_labels(path, config)


def test_labels_align_separately_and_do_not_change_features(tmp_path):
    path, config = csv_file(tmp_path, [0, 1, 0, 1])
    before = label_free_events(path, config)
    labels = heldout_labels(path, config)
    assert labels.event_id.tolist() == [1, 2, 3] and labels.label.tolist() == [1, 0, 1]
    path, config = csv_file(tmp_path, [1, 0, 1, 0])
    after = label_free_events(path, config)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(causal_features(before), causal_features(after))


def test_final_graph_queries_exclude_entire_current_minute(tmp_path):
    path, config = csv_file(tmp_path, [0, 1, 1, 0])
    frame = label_free_events(path, config)
    index = TemporalNeighborhood(frame)
    rows, ages, _ = index.sample([1, 2], 32)
    assert set(rows[rows >= 0]) == {0}
    assert (ages[rows >= 0] == 1).all()
    with pytest.raises(ValueError, match='digest'):
        label_free_events(path, {**config, 'sha256': 'bad'})
