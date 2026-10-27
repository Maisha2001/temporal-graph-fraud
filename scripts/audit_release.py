import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score
import torch

from temporal_fraud.data import digest
from temporal_fraud.graph_model import TemporalGraphClassifier, predict
from temporal_fraud.release_eval import frozen_sources, heldout_labels, source_hashes, save


def audit(run, baseline, graph, risk):
    run, baseline, graph, risk = map(Path, [run, baseline, graph, risk])
    report = json.loads((run / 'report.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    frozen = json.loads((run / 'predictions-frozen.json').read_text())
    assert protocol['source_sha256'] == source_hashes()
    assert protocol['artifact_sha256'] == frozen_sources(baseline, graph, risk)
    assert digest(run / 'protocol.json') == report['protocol_sha256'] == frozen['protocol_sha256']
    assert digest(run / 'predictions-frozen.json') == report['prediction_freeze_sha256']
    assert datetime.fromisoformat(protocol['frozen_at_utc']) < datetime.fromisoformat(frozen['frozen_at_utc']) < datetime.fromisoformat(report['labels_opened_at_utc'])
    for name, expected in report['artifact_sha256'].items():
        assert digest(run / name) == expected, name
    predictions = pd.read_parquet(run / 'predictions.parquet')
    assert 'label' not in predictions
    labels = pd.read_parquet(run / 'labels.parquet')
    assert predictions.event_id.equals(labels.event_id) and predictions.timestamp.equals(labels.timestamp)
    assert len(predictions) == 1346673 and len(predictions) == report['events']
    assert predictions.event_id.is_unique
    frame = predictions.copy();frame['label'] = labels.label
    metric_groups = 0
    brier_errors = []
    dates = frame.timestamp.dt.strftime("%Y-%m-%d")
    masks = {g: (np.ones(len(frame), dtype=bool) if g == "all" else frame.cold.to_numpy() if g == "cold" else ~frame.cold.to_numpy() if g == "warm" else dates.eq(g).to_numpy()) for g in next(iter(report["models"].values()))}
    for arm, groups in report['models'].items():
        for group, expected in groups.items():
            mask = masks[group]
            batch = frame.loc[mask]
            y, scores = batch.label.to_numpy(), batch[arm].to_numpy()
            assert len(batch) == expected['events'] and int(y.sum()) == expected['positives']
            ap = float(average_precision_score(y, scores)) if y.sum() else None
            assert ap == expected['average_precision'] or abs(ap - expected['average_precision']) < 1e-12
            brier_error = abs(np.mean((scores.astype(float)-y)**2) - expected['brier'])
            brier_errors.append(float(brier_error))
            # sklearn preserves float32 score arithmetic; this independent calculation uses float64.
            assert brier_error < 1e-8
            budget = int(len(batch) * .01)
            if budget:
                boundary = np.sort(scores)[-budget]
                above, tied = scores > boundary, scores == boundary
                hits = y[above].sum() + (budget - above.sum()) * y[tied].mean()
                assert abs(hits - expected['expected_true_alerts']) < 1e-9
            metric_groups += 1
    hours_checked = 0
    for arm, expected in report['policies'].items():
        p = json.loads((risk / (arm + '-calibrator.json')).read_text())
        calibrated = expit(p['calibrator']['slope'] * logit(np.clip(frame[arm].to_numpy(dtype=float), 1e-7, 1-1e-7)) + p['calibrator']['intercept'])
        np.testing.assert_allclose(calibrated, frame[arm + '_calibrated'], rtol=0, atol=1e-15)
        assert p['threshold'] == expected['threshold']
        decisions = pd.read_parquet(run / (arm + '-alerts.parquet'))
        assert decisions.event_id.equals(frame.event_id)
        total_hits, total_alerts = 0, 0
        for hour, batch in decisions.groupby(decisions.timestamp.dt.floor('h')):
            budget = len(batch) // 100
            eligible = batch.loc[batch.score >= p['threshold']]
            selected = sorted(eligible.itertuples(), key=lambda row: (-row.score, hashlib.sha256(f"{p['seed']}:{row.event_id}".encode()).hexdigest()[:16]))[:budget]
            expected_ids = {row.event_id for row in selected}
            assert expected_ids == set(batch.loc[batch.selected, 'event_id'])
            total_alerts += len(expected_ids);total_hits += int(batch.loc[batch.selected,'label'].sum())
            hours_checked += 1
        assert total_alerts == expected['alerts'] and total_hits == expected['true_alerts']
    membership = pd.read_parquet(run / 'membership.parquet')
    index = joblib.load(run / 'neighborhood.joblib')
    rows = np.flatnonzero(membership.timestamp >= pd.Timestamp(protocol['data_config']['test_start']))
    np.testing.assert_array_equal(membership.event_id.to_numpy()[rows], frame.event_id)
    selected = rows[np.unique(np.linspace(0, len(rows)-1, 64, dtype=int))]
    neighbors, _, _ = index.sample(selected, 32)
    for i, row in enumerate(selected):
        for side, name in enumerate(['source','target']):
            owner = membership.iloc[row][name]
            earlier = np.flatnonzero(((membership.source==owner)|(membership.target==owner)) & (membership.timestamp < membership.timestamp.iloc[row]))
            actual = neighbors[i,side][neighbors[i,side]>=0]
            assert len(actual)==min(32,len(earlier)) and np.isin(actual,earlier).all()
            if len(earlier)>32:
                assert (index.minutes[actual]>=np.sort(index.minutes[earlier])[-32]).all()
    torch.set_num_threads(4)
    x = torch.from_numpy(np.load(run/'features.npy'))
    subset = np.unique(np.linspace(0,len(rows)-1,256,dtype=int))
    errors = {}
    for n,arm in [(0,'history_mlp'),(8,'temporal_attention_8'),(32,'temporal_attention_32')]:
        state = torch.load(graph/f'neighbors-{n}-latest.pt',weights_only=True,map_location='cpu')
        model=TemporalGraphClassifier(state['features'],state['config']['hidden'],n)
        model.load_state_dict(state['state_dict'])
        scores=predict(model,x,index,rows[subset])
        errors[arm]=float(np.max(np.abs(scores-frame[arm].to_numpy()[subset])))
        assert errors[arm]<1e-6
    return {'passed':True,'test_scored':True,'events':len(frame),'positives':int(frame.label.sum()),
            'source_artifacts_verified':len(protocol['artifact_sha256']),'result_artifacts_verified':len(report['artifact_sha256']),
            'independent_metric_groups':metric_groups,'float64_vs_recorded_brier_max_difference':max(brier_errors),'closed_hour_policies_checked':hours_checked,
            'exhaustive_endpoint_queries':128,'cpu_reload_predictions_per_neural_arm':len(subset),
            'cpu_vs_saved_max_absolute_difference':errors,'freeze_precedes_label_open':True,
            'report_sha256':digest(run/'report.json'),'prediction_sha256':digest(run/'predictions.parquet')}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['run','baseline','graph','risk','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Audit output exists')
    r=audit(a.run,a.baseline,a.graph,a.risk);save(a.output,r);print(json.dumps(r))
