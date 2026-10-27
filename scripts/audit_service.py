import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from scipy.special import expit, logit

from temporal_fraud.data import digest


def audit(prepared, measured, graph):
    prepared,measured,graph=map(Path,[prepared,measured,graph])
    prep=json.loads((prepared/'preparation.json').read_text())
    report=json.loads((measured/'report.json').read_text())
    assert report['completed'] and not report['test_scored']
    for folder,manifest in [(prepared,prep['sha256']),(measured,report['artifact_sha256'])]:
        for name,expected in manifest.items():
            assert digest(folder/name)==expected,name
    assert digest(prepared/'preparation.json')==report['source_preparation_sha256']
    assert report['api_restart_pid_changed'] and len(set(report['api_process_ids']))==2
    assert report['http_api_peak_process_bytes']>100_000_000
    responses=[json.loads(line) for line in (measured/'responses.jsonl').read_text().splitlines()]
    ref=pd.read_parquet(prepared/'reference.parquet').set_index('event_id').sort_index()
    errors={}
    for kind,stream in report['streams'].items():
        groups=[r for r in responses if r['stream']==stream]
        assert sorted(r['sequence'] for r in groups)==list(range(30))
        actual=pd.DataFrame([s for r in groups for s in r['scores']]).set_index('event_id').sort_index()
        assert actual.index.is_unique and actual.index.equals(ref.index)
        errors[kind]={}
        for model in ['history_xgboost','temporal_attention_32']:
            raw=np.asarray(ref[model],dtype=float)
            fit=json.loads((prepared/'models'/(model+'-policy.json')).read_text())['calibrator']
            calibrated=expit(fit['slope']*logit(np.clip(raw,1e-7,1-1e-7))+fit['intercept'])
            errors[kind][model]=float(np.max(np.abs(actual[model]-raw)))
            errors[kind][model+'_calibrated']=float(np.max(np.abs(actual[model+'_calibrated']-calibrated)))
            assert max(errors[kind].values())<1e-6
    timings=pd.read_parquet(measured/'http-latencies.parquet')
    assert len(timings)==90 and timings.events.sum()==3*len(ref)
    for name,subset in [('serial',timings.loc[timings.stream=='serial']),
                        ('concurrent_two_streams',timings.loc[timings.stream.str.startswith('parallel')])]:
        for quantile,label in [(.5,'p50_ms'),(.95,'p95_ms'),(.99,'p99_ms')]:
            assert abs(float(subset.seconds.quantile(quantile)*1000)-report[name]['batch_latency'][label])<1e-8
    records=json.loads((measured/'kafka-latencies.json').read_text())
    assert len(records)==30 and sum(x['duplicate'] for x in records)==1
    assert all(x['processing_seconds']>0 and x['enqueue_to_database_seconds']>0 for x in records)
    membership=pd.read_parquet(graph/'membership.parquet',columns=['event_id','timestamp','source','target'])
    index=joblib.load(graph/'neighborhood.joblib')
    x=np.load(graph/'features.npy',mmap_mode='r')
    end=pd.Timestamp(prep['end_minute'])
    before=membership.loc[membership.timestamp<end]
    out_count,in_count=before.source.value_counts(),before.target.value_counts()
    last_out,last_in=before.groupby('source').timestamp.max(),before.groupby('target').timestamp.max()
    nodes=pd.factorize(pd.concat([membership.source,membership.target]),sort=True)[1]
    codes={owner:i for i,owner in enumerate(nodes)}
    minute=int(np.datetime64(end,'m').astype(np.int64))
    accounts_checked=0;neighbors_checked=0
    with psycopg.connect(os.environ['FRAUD_DSN']) as conn:
        for stream in report['streams'].values():
            with conn.cursor(name='audit-'+stream) as cursor:
                cursor.execute('SELECT owner,state FROM fraud_accounts WHERE stream=%s',(stream,))
                for owner,state in cursor:
                    assert state['out']==int(out_count.get(owner,0))
                    assert state['in']==int(in_count.get(owner,0))
                    for label,source in [('last_out',last_out),('last_in',last_in)]:
                        expected=int(np.datetime64(source[owner],'m').astype(np.int64)) if owner in source else -1
                        assert state[label]==expected
                    code=codes[owner]
                    stop=np.searchsorted(index.keys,code*100_000_000+minute,side='left')
                    positions=np.arange(stop-1,max(index.starts[code],stop-32)-1,-1)
                    assert len(state['neighbors'])==len(positions)
                    for edge,position in zip(state['neighbors'],positions):
                        row=int(index.rows[position])
                        assert edge['event_id']==int(membership.event_id.iloc[row])
                        assert edge['minute']==int(index.minutes[row])
                        assert edge['direction']==int(index.directions[position])
                        np.testing.assert_array_equal(np.asarray(edge['x'],dtype=np.float32),x[row])
                        neighbors_checked+=1
                    accounts_checked+=1
    return {'passed':True,'test_scored':False,'prepared_artifacts_verified':len(prep['sha256']),
            'measurement_artifacts_verified':len(report['artifact_sha256']),
            'scored_event_copies_checked':4*len(ref),'metric_groups_checked':8,
            'persistent_account_states_checked':accounts_checked,'persistent_neighbor_vectors_checked':neighbors_checked,
            'http_latency_samples_checked':len(timings),'kafka_ack_recovery_duplicates':1,
            'raw_and_calibrated_max_absolute_errors':errors,'report_sha256':digest(measured/'report.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',required=True);parser.add_argument('--measured',required=True)
    parser.add_argument('--graph',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit(args.prepared,args.measured,args.graph)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))
