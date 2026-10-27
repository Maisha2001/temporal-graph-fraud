import argparse
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from temporal_fraud.data import CATEGORIES, digest, load_development, split_events
from temporal_fraud.streaming import MinuteBatch, empty_account, minute_number


def save(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')


def prepare(data, baseline, graph, risk, output, config, start, minutes=30):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    baseline, graph, risk = map(Path, [baseline, graph, risk])
    pinned = {}
    for folder, report_name in [(baseline, 'report.json'), (graph, 'report.json'), (risk, 'study.json')]:
        report = json.loads((folder / report_name).read_text())
        for name, expected in report['artifact_sha256'].items():
            if digest(folder / name) != expected:
                raise ValueError('Frozen input artifact changed: ' + name)
            pinned[str(folder.name) + '/' + name] = expected
    models = output / 'models'
    models.mkdir()
    for folder, names in [(baseline, ['history_xgboost.ubj', 'history_xgboost.joblib']),
                          (graph, ['preprocessing.joblib', 'neighbors-32-latest.pt'])]:
        for name in names:
            shutil.copyfile(folder / name, models / name)
    for name in ['history_xgboost', 'temporal_attention_32']:
        policy = json.loads((risk / (name + '-calibrator.json')).read_text())
        save(models / (name + '-policy.json'), policy)
    save(models / 'manifest.json', {'first_minute': '2022-09-07T12:00:00', 'end_minute': config['test_start'],
                                   'sha256': {p.name: digest(p) for p in sorted(models.iterdir())}})
    events, counts = load_development(data, config)
    masks = split_events(events, config)
    replay = events.loc[masks['replay']].reset_index(drop=True).drop(columns='label')
    membership = pd.read_parquet(graph / 'membership.parquet', columns=['event_id'])
    if not np.array_equal(membership.event_id, replay.event_id):
        raise ValueError('Saved model replay does not match input data')
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(minutes=minutes)
    if start < pd.Timestamp('2022-09-07T12:00:00') or end > pd.Timestamp(config['test_start']):
        raise ValueError('Only calibrated development replay is permitted')
    selected = replay.loc[(replay.timestamp >= start) & (replay.timestamp < end)]
    if not len(selected):
        raise ValueError('Empty replay interval')
    owners = sorted(set(selected.source) | set(selected.target))
    index = joblib.load(graph / 'neighborhood.joblib')
    x = np.load(graph / 'features.npy', mmap_mode='r')
    nodes = pd.factorize(pd.concat([replay.source, replay.target]), sort=True)[1]
    owner_codes = nodes.get_indexer(owners)
    now = minute_number(start)
    past = replay.loc[replay.timestamp < start]
    out_count, in_count = past.source.value_counts(), past.target.value_counts()
    last_out, last_in = past.groupby('source').timestamp.max(), past.groupby('target').timestamp.max()
    accounts = {}
    priorities = pd.util.hash_pandas_object(replay[['source', 'target', 'paid', 'received', *CATEGORIES]], index=False).to_numpy()
    for owner, code in zip(owners, owner_codes):
        state = empty_account()
        state.update(out=int(out_count.get(owner, 0)), **{'in': int(in_count.get(owner, 0))},
                     last_out=minute_number(last_out[owner]) if owner in last_out else -1,
                     last_in=minute_number(last_in[owner]) if owner in last_in else -1)
        stop = np.searchsorted(index.keys, code * 100_000_000 + now, side='left')
        positions = np.arange(stop - 1, max(index.starts[code], stop - 32) - 1, -1)
        for position in positions:
            row = int(index.rows[position])
            state['neighbors'].append({'event_id': int(replay.event_id.iloc[row]), 'minute': int(index.minutes[row]),
                                       'priority': int(priorities[row]), 'direction': int(index.directions[position]),
                                       'x': x[row].tolist()})
        accounts[owner] = state
    snapshot = {'model_fingerprint': digest(models / 'manifest.json'),
                'last_minute': (start - pd.Timedelta(minutes=1)).isoformat(), 'accounts': accounts}
    save(output / 'snapshot.json', snapshot)
    with (output / 'batches.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for sequence, (minute, group) in enumerate(selected.groupby('timestamp', sort=True)):
            batch = MinuteBatch(sequence=sequence, minute=minute.to_pydatetime(),
                                events=group.drop(columns='timestamp').to_dict('records'))
            stream.write(batch.model_dump_json() + '\n')
    selected_ids = selected.event_id.to_numpy()
    reference = selected[['event_id', 'timestamp']].copy()
    for folder, name in [(baseline, 'history_xgboost'), (graph, 'temporal_attention_32')]:
        pred = pd.read_parquet(folder / (name + '-validation.parquet'), columns=['event_id', 'score'])
        reference[name] = reference.event_id.map(pred.set_index('event_id').score)
    assert not reference.isna().any().any()
    reference.to_parquet(output / 'reference.parquet', index=False)
    report = {'test_scored': False, 'counts': counts, 'events': len(selected_ids),
              'minutes': int(selected.timestamp.nunique()), 'accounts': len(owners),
              'first_minute': str(start), 'end_minute': str(end), 'prepare_seconds': time.perf_counter() - started,
              'source_artifacts': pinned, 'model_fingerprint': snapshot['model_fingerprint'],
              'scope': 'Prior state for every endpoint in the bounded replay; unknown endpoints are rejected.',
              'sha256': {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}}
    save(output / 'preparation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['data','baseline','graph','risk','output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--config', default='configs/ibm-hi-small.json')
    parser.add_argument('--start', default='2022-09-07T12:00:00')
    parser.add_argument('--minutes', type=int, default=30)
    args = vars(parser.parse_args())
    args['config'] = json.loads(Path(args['config']).read_text())
    result = prepare(**args)
    print(json.dumps({k: result[k] for k in ['events','minutes','accounts','prepare_seconds','model_fingerprint']}))
