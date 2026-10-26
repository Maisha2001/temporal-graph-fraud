import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from temporal_fraud.data import entity_bucket
from temporal_fraud.temporal import TemporalNeighborhood
from test_pipeline import events


def test_sampler_matches_exhaustive_past_incident_edges():
    rng = np.random.default_rng(10)
    frame = events(pd.Timestamp('2022-09-01') + pd.to_timedelta(np.sort(rng.integers(0, 30, 100)), unit='min'),
                   [str(x) for x in rng.integers(0, 6, 100)], [str(x) for x in rng.integers(0, 6, 100)])
    index = TemporalNeighborhood(frame)
    history, ages, directions = index.sample(np.arange(len(frame)), 100)
    for row in range(len(frame)):
        for role, node in enumerate([frame.source[row], frame.target[row]]):
            expected = np.flatnonzero(((frame.source == node) | (frame.target == node)) & (frame.timestamp < frame.timestamp[row]))
            selected = history[row, role]
            np.testing.assert_array_equal(np.sort(selected[selected >= 0]), expected)
            assert (ages[row, role][selected >= 0] > 0).all()
            expected_direction = np.where(frame.source.iloc[selected[selected >= 0]] == node, 1, -1)
            np.testing.assert_array_equal(directions[row, role][selected >= 0], expected_direction)


def test_sampler_same_minute_future_labels_and_row_ids():
    frame = events(['2022-09-01'] * 4 + ['2022-09-02', '2022-09-03'])
    base = TemporalNeighborhood(frame)
    assert (base.sample([0, 1, 2, 3], 2)[0] == -1).all()
    chosen = base.sample([4], 2)[0]
    changed = frame.copy()
    changed.loc[5, ['source', 'target', 'paid']] = ['NEW', 'NEW2', 1e9]
    changed['label'] = 1 - changed.label
    changed['event_id'] = changed.event_id[::-1].to_numpy() + 100
    np.testing.assert_array_equal(TemporalNeighborhood(changed).sample([4], 2)[0], chosen)
    reordered = frame.iloc[[3, 1, 0, 2, 4, 5]].reset_index(drop=True)
    other = TemporalNeighborhood(reordered).sample([4], 2)[0]
    np.testing.assert_array_equal(frame.event_id.to_numpy()[chosen], reordered.event_id.to_numpy()[other])


def test_self_loop_and_new_account_have_no_duplicate_neighbors():
    frame = events(['2022-09-01', '2022-09-02', '2022-09-03'], ['A', 'A', 'NEW'], ['A', 'A', 'NEW'])
    history = TemporalNeighborhood(frame).sample([1, 2], 4)[0]
    np.testing.assert_array_equal(history[0], [[0, -1, -1, -1]] * 2)
    assert (history[1] == -1).all()


@pytest.mark.parametrize('rows,k', [([-1], 2), ([4], 2), ([0], 0)])
def test_invalid_query_rejected(rows, k):
    frame = events(['2022-09-01', '2022-09-02'])
    with pytest.raises(ValueError):
        TemporalNeighborhood(frame).sample(rows, k)


def graph_modules():
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    torch.set_num_threads(2)
    from temporal_fraud.graph_model import TemporalGraphClassifier, predict
    from temporal_fraud.graph_experiment import fit_model, prepare
    return torch, TemporalGraphClassifier, predict, fit_model, prepare


def training_fixture():
    warm = next(f'1:{i}' for i in range(100) if entity_bucket(f'1:{i}') != 0)
    cold = next(f'1:{i}' for i in range(100) if entity_bucket(f'1:{i}') == 0)
    frame = events(list(pd.date_range('2022-09-01', periods=48, freq='h')) + list(pd.date_range('2022-09-06', periods=12, freq='h')),
                   [warm] * 48 + [cold] * 12, [warm] * 60)
    config = json.loads((Path(__file__).parents[1] / 'configs/ibm-hi-small.json').read_text())
    training = {'seed': 42, 'epochs': 2, 'hidden': 8, 'batch_size': 12, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'threads': 2, 'neighbors': [0, 2]}
    return frame, config, training


def test_preprocessing_and_graph_respect_training_and_cold_split():
    torch, cls, predict, fit, prepare = graph_modules()
    frame, config, training = training_fixture()
    x, index, membership = prepare(frame, config, None)
    modified = frame.copy()
    modified.loc[48:, 'paid'] = 1e12
    modified.loc[48:, 'format'] = 'UNKNOWN'
    other, _, _ = prepare(modified, config, None)
    np.testing.assert_array_equal(x[:48], other[:48])
    assert np.all(index.sample([48], 4)[0][0, 0] == -1)
    assert membership.loc[48:, 'cold'].all()
    assert not membership.loc[48:, 'train'].any()


def test_model_has_past_messages_but_no_future_information():
    torch, cls, predict, fit, prepare = graph_modules()
    frame, config, training = training_fixture()
    x, index, membership = prepare(frame, config, None)
    torch.manual_seed(42)
    model = cls(x.shape[1], 8, 2)
    x = torch.tensor(x, requires_grad=True)
    model(x, torch.tensor([10]), index.sample([10], 2)).sum().backward()
    assert x.grad[8:10].abs().sum() > 0
    assert x.grad[11:].abs().sum() == 0
    frame.loc[49:, ['paid', 'received']] = 1e10
    frame['label'] = 1 - frame.label
    other, other_index, _ = prepare(frame, config, None)
    np.testing.assert_array_equal(predict(model, x.detach(), index, np.array([48])),
                                  predict(model, torch.tensor(other), other_index, np.array([48])))
    assert np.isfinite(predict(model, x.detach(), index, np.array([0]))).all()


@pytest.mark.parametrize('neighbors', [0, 2])
def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch, neighbors):
    torch, cls, predict, fit, prepare = graph_modules()
    import temporal_fraud.graph_experiment as experiment
    frame, config, training = training_fixture()
    x, index, membership = prepare(frame, config, None)
    x = torch.tensor(x)
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    full.mkdir()
    interrupted.mkdir()
    expected, _ = fit(x, index, membership, training, neighbors, full, 'fixture')
    original = experiment.save_checkpoint
    def stop_after_epoch(*args):
        original(*args)
        raise RuntimeError('simulate interruption after durable checkpoint')
    monkeypatch.setattr(experiment, 'save_checkpoint', stop_after_epoch)
    with pytest.raises(RuntimeError, match='simulate interruption'):
        fit(x, index, membership, training, neighbors, interrupted, 'fixture')
    monkeypatch.setattr(experiment, 'save_checkpoint', original)
    actual, report = fit(x, index, membership, training, neighbors, interrupted, 'fixture', resume=True)
    assert report['updates'] == 8
    for a, b in zip(expected.parameters(), actual.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='mismatch'):
        fit(x, index, membership, training, neighbors, interrupted, 'changed-data', resume=True)


def test_full_command_path_and_completed_resume_preserve_evidence(tmp_path):
    torch, cls, predict, fit, prepare = graph_modules()
    import csv
    from temporal_fraud.data import HEADER, digest
    from temporal_fraud.graph_experiment import run
    frame, config, training = training_fixture()
    data = tmp_path / 'fixture.csv'
    with data.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for row in frame.itertuples():
            source_bank, source_account = row.source.split(':')
            target_bank, target_account = row.target.split(':')
            writer.writerow([row.timestamp.strftime('%Y/%m/%d %H:%M'), source_bank, source_account,
                             target_bank, target_account, row.received, row.receiving_currency,
                             row.paid, row.payment_currency, row.format, row.label])
        writer.writerow(['2022/09/08 00:00', 1, 'X', 1, 'Y', 1, 'USD', 1, 'USD', 'Wire', 'UNREAD_TEST_LABEL'])
    config['sha256'] = digest(data)
    output = tmp_path / 'run'
    report = run(training, config, data, output)
    assert report['data_counts']['reserved_test_rows'] == 1
    assert report['test_scored'] is False
    assert set(report['models']) == {'history_mlp', 'temporal_attention_2'}
    before = {p.name: digest(p) for p in output.iterdir()}
    restored = run(training, config, data, output, resume=True)
    assert restored == report
    assert before == {p.name: digest(p) for p in output.iterdir()}
    with pytest.raises(ValueError, match='new output directory'):
        run(training, config, data, output)
    (output / 'features.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='digest mismatch'):
        run(training, config, data, output, resume=True)
