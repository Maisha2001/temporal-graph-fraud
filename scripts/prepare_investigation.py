import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from temporal_fraud.data import digest, load_development, split_events
from temporal_fraud.release_eval import save


def prepare(data, graph, measured, output, config, limit=60):
    output, graph, measured = map(Path, [output, graph, measured])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = json.loads((measured / 'report.json').read_text())
    for name, expected in report['artifact_sha256'].items():
        if digest(measured / name) != expected:
            raise ValueError('Saved service measurement changed')
    scores = []
    for line in (measured / 'responses.jsonl').read_text().splitlines():
        response = json.loads(line)
        if response['stream'] == report['streams']['serial']:
            for event in response['scores']:
                scores.append({**event, 'minute_index': response['sequence']})
    scored = pd.DataFrame(scores).drop_duplicates('event_id').set_index('event_id')
    if len(scored) != report['events_per_stream']:
        raise ValueError('Saved serial receipt count mismatch')
    selected_ids = set()
    for name in ['history_xgboost', 'temporal_attention_32']:
        selected_ids.update(scored.sort_values([name + '_calibrated'], ascending=False, kind='stable').head(limit).index)
    events, _ = load_development(data, config)
    masks = split_events(events, config)
    replay = events.loc[masks['replay']].reset_index(drop=True)
    cold = masks['cold'][masks['replay']]
    membership = pd.read_parquet(graph / 'membership.parquet', columns=['event_id'])
    if not np.array_equal(membership.event_id, replay.event_id):
        raise ValueError('Replay membership changed')
    index = joblib.load(graph / 'neighborhood.joblib')
    rows = np.flatnonzero(replay.event_id.isin(selected_ids))
    sampled, _, _ = index.sample(rows, 8)
    cases = []
    columns = ['event_id', 'timestamp', 'source', 'target', 'paid', 'payment_currency', 'format']
    for i, row in enumerate(rows):
        event = replay.iloc[row]
        context_rows = np.unique(sampled[i][sampled[i] >= 0])
        context = []
        for neighbor in context_rows:
            edge = replay.iloc[neighbor][columns].to_dict()
            edge['event_id'] = int(edge['event_id']); edge['timestamp'] = edge['timestamp'].isoformat()
            if edge['timestamp'] >= event.timestamp.isoformat():
                raise ValueError('Future investigation edge')
            context.append(edge)
        case = event[columns].to_dict()
        case.update(event_id=int(event.event_id), timestamp=event.timestamp.isoformat(),
                    label=int(event.label), cold=bool(cold[row]), context=context,
                    **scored.loc[event.event_id].to_dict())
        case['minute_index'] = int(case['minute_index'])
        cases.append(case)
    bundle = {
        'title': 'Transaction Atlas', 'mode': 'Saved model replay',
        'period': 'September 7, 2022 · 12:00–12:30 simulator time',
        'events_scored': report['events_per_stream'], 'minutes': 30,
        'selection': f'Union of the top {limit} calibrated cases per model in the measured replay; not an exhaustive alert queue.',
        'data_notice': 'IBM HI-Small synthetic AML data. Local derived data under CDLA-Sharing-1.0; retain IBM attribution and identify modifications when redistributing.',
        'limitations': 'Ranked review candidates, not closed-hour capacity decisions. Graph edges show strictly earlier observed context, not causal explanations. Simulation outcomes are hidden by default.',
        'service_report_sha256': digest(measured / 'report.json'),
        'model_fingerprint': report['model_fingerprint'], 'test_data_used': False,
        'latency_p95_ms': report['serial']['batch_latency']['p95_ms'],
        'cases': cases}
    save(output / 'cases.json', bundle)
    manifest = {'cases_sha256': digest(output / 'cases.json'), 'source_report_sha256': digest(measured / 'report.json'),
                'graph_membership_sha256': digest(graph / 'membership.parquet'), 'case_count': len(cases),
                'context_edges': sum(len(c['context']) for c in cases), 'prepare_seconds': time.perf_counter() - started,
                'test_data_used': False, 'license': 'CDLA-Sharing-1.0'}
    save(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['data', 'graph', 'measured', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--config', default='configs/ibm-hi-small.json')
    args = vars(parser.parse_args());args['config'] = json.loads(Path(args['config']).read_text())
    print(json.dumps(prepare(**args)))
