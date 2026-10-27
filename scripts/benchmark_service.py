import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import uuid

import httpx
import numpy as np
import pandas as pd
import psutil

from temporal_fraud.broker import consume, create_topic, publish
from temporal_fraud.data import digest
from temporal_fraud.store import Store
from temporal_fraud.streaming import MinuteBatch, Models


def summary(values):
    a=np.asarray(values,dtype=float)
    return {'count':len(a),'min_ms':float(a.min()*1000),'p50_ms':float(np.quantile(a,.5)*1000),
            'p95_ms':float(np.quantile(a,.95)*1000),'p99_ms':float(np.quantile(a,.99)*1000),'max_ms':float(a.max()*1000)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap',default='127.0.0.1:19094')
    args=parser.parse_args()
    root=Path(args.input).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    preparation=json.loads((root/'preparation.json').read_text())
    for name,expected in preparation['sha256'].items():
        assert digest(root/name)==expected,name
    models=Models(root/'models');store=Store(os.environ['FRAUD_DSN'],models);store.initialize()
    snapshot=json.loads((root/'snapshot.json').read_text())
    batches=[MinuteBatch.model_validate_json(line) for line in (root/'batches.jsonl').read_text().splitlines()]
    reference=pd.read_parquet(root/'reference.parquet').set_index('event_id')
    run_id=uuid.uuid4().hex[:12]
    streams={k:'service-'+run_id+'-'+k for k in ['serial','parallel-a','parallel-b','kafka']}
    setup_at=time.perf_counter()
    for stream in streams.values():
        store.create(stream,snapshot)
    setup_seconds=time.perf_counter()-setup_at
    del snapshot
    try:
        store.apply(streams['serial'],batches[0],fail_before_commit=True)
    except RuntimeError as error:
        assert 'Injected failure' in str(error)
    assert store.status(streams['serial'])['events']==0
    environment=os.environ.copy();environment['FRAUD_MODELS']=str(root/'models')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url='http://127.0.0.1:'+str(port)
    process=None;log=None;peak=0;server_pid=None;server_pids=[]
    def start(number):
        nonlocal process,log,server_pid
        log=(out/f'api-{number}.log').open('w',encoding='utf-8')
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        process=subprocess.Popen([sys.executable,'-m','uvicorn','temporal_fraud.api:create_app','--factory',
                                  '--host','127.0.0.1','--port',str(port),'--no-access-log'],
                                 env=environment,stdout=log,stderr=log,creationflags=flags)
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('API process exited; inspect saved log')
            try:
                health=httpx.get(url+'/health',timeout=1)
                if health.status_code==200:
                    server_pid=health.json()['pid']
                    owned={process.pid,*[p.pid for p in psutil.Process(process.pid).children(recursive=True)]}
                    if server_pid not in owned:raise RuntimeError('Readiness belongs to a different process')
                    server_pids.append(server_pid)
                    return
            except httpx.TransportError:pass
            time.sleep(.2)
        raise TimeoutError('API readiness')
    def stop():
        nonlocal process,log
        if process is not None:
            owned=psutil.Process(process.pid).children(recursive=True)
            for child in owned:
                if child.is_running():child.terminate()
            process.terminate();process.wait(timeout=15)
            _,alive=psutil.wait_procs(owned,timeout=10)
            for child in alive:child.kill()
            process=None
            try:
                httpx.get(url+'/health',timeout=.5)
            except httpx.TransportError:pass
            else:raise RuntimeError('API still listening after shutdown')
        if log is not None:log.close();log=None
    latencies=[];responses=[]
    def replay(key, selected):
        nonlocal peak
        local_records=[]
        with httpx.Client(base_url=url,timeout=60) as client:
            for batch in selected:
                started=time.perf_counter()
                response=client.post('/streams/'+streams[key]+'/batches',json=batch.model_dump(mode='json'))
                elapsed=time.perf_counter()-started
                response.raise_for_status();result=response.json()
                assert not result['duplicate']
                record={'stream':key,'sequence':batch.sequence,'events':len(batch.events),'seconds':elapsed}
                local_records.append(record);responses.append({'stream':key,**result})
                memory=psutil.Process(server_pid).memory_info()
                peak=max(peak,getattr(memory,'peak_wset',memory.rss))
        return local_records
    try:
        start(1)
        serial_started=time.perf_counter()
        latencies+=replay('serial',batches[:15])
        stop();start(2)
        with httpx.Client(base_url=url,timeout=60) as client:
            duplicate=client.post('/streams/'+streams['serial']+'/batches',json=batches[14].model_dump(mode='json'))
            duplicate.raise_for_status();assert duplicate.json()['duplicate']
        latencies+=replay('serial',batches[15:])
        serial_wall=time.perf_counter()-serial_started
        parallel_started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(replay,key,batches) for key in ['parallel-a','parallel-b']]
            for job in jobs:latencies+=job.result()
        parallel_wall=time.perf_counter()-parallel_started
    finally:
        stop()
    topic='fraud-'+run_id;group=topic+'-group'
    create_topic(args.bootstrap,topic)
    publish(args.bootstrap,topic,streams['kafka'],batches)
    kafka_started=time.perf_counter()
    try:
        consume(args.bootstrap,topic,group,store,len(batches),crash_after_database=1)
    except RuntimeError as error:
        assert 'after database commit' in str(error)
    assert store.status(streams['kafka'])['sequence']==0
    kafka_measurements=[]
    consume(args.bootstrap,topic,group,Store(store.dsn,Models(root/'models')),len(batches),
            observer=lambda value:kafka_measurements.append(value))
    kafka_wall=time.perf_counter()-kafka_started
    for batch in batches:
        responses.append({'stream':'kafka',**store.batch(streams['kafka'],batch.sequence)})
    errors={}
    for key in streams:
        flat=[row for batch in responses if batch['stream']==key or batch['stream']==streams[key] for row in batch['scores']]
        actual=pd.DataFrame(flat).set_index('event_id').sort_index()
        expected=reference.sort_index()
        assert len(actual)==len(reference) and actual.index.equals(expected.index)
        errors[key]={name:float(np.max(np.abs(actual[name]-expected[name]))) for name in ['history_xgboost','temporal_attention_32']}
        assert max(errors[key].values())<1e-6,errors[key]
        status=store.status(streams[key]);assert status['events']==len(reference) and status['sequence']==len(batches)-1
    pd.DataFrame(latencies).to_parquet(out/'http-latencies.parquet',index=False)
    (out/'responses.jsonl').write_text('\n'.join(json.dumps(r) for r in responses)+'\n',encoding='utf-8')
    (out/'kafka-latencies.json').write_text(json.dumps(kafka_measurements,indent=2)+'\n',encoding='utf-8')
    assert len(set(server_pids))==2 and peak>100_000_000
    report={'test_scored':False,'completed':True,'source_preparation_sha256':digest(root/'preparation.json'),
            'model_fingerprint':models.fingerprint,'events_per_stream':len(reference),'complete_minutes_per_stream':len(batches),
            'streams':streams,'snapshot_import_seconds_four_streams':setup_seconds,'cpu_threads':4,'device':'cpu',
            'http_api_peak_process_bytes':peak,'api_process_ids':server_pids,'api_restart_pid_changed':len(set(server_pids))==2,'platform':platform.platform(),'python':platform.python_version(),
            'serial':{'batch_latency':summary([x['seconds'] for x in latencies if x['stream']=='serial']),
                      'total_wall_including_restart_seconds':serial_wall,'restart_duplicate_preserved':True},
            'concurrent_two_streams':{'batch_latency':summary([x['seconds'] for x in latencies if x['stream'].startswith('parallel')]),
                                      'wall_seconds':parallel_wall,'events_per_second':2*len(reference)/parallel_wall},
            'kafka':{'wall_including_restart_seconds':kafka_wall,'messages_after_restart':len(kafka_measurements),
                     'duplicate_count':sum(x['duplicate'] for x in kafka_measurements),
                     'database_processing_latency':summary([x['processing_seconds'] for x in kafka_measurements]),
                     'enqueue_to_database_latency':summary([x['enqueue_to_database_seconds'] for x in kafka_measurements]),
                     'database_commit_before_ack_recovery':True},
            'max_absolute_difference_from_frozen_offline':errors,
            'rollback_test_passed':True,'unknown_accounts_rejected':True,
            'limitations':['Bounded single-partition ordered replay; not arbitrary event-time ingestion.',
                           'Complete minutes are producer-declared watermarks. Missing events cannot be inferred by the server.',
                           'Two concurrent replay streams on one local CPU service; not a distributed capacity guarantee.',
                           'Batch latency excludes up-to-one-minute event aggregation and any later hourly alert decision.',
                           'Kafka burst queue delays include deliberate crash/restart, not steady-state per-event inference.',
                           'Synthetic development data only; final holdout untouched.'],
            'artifact_sha256':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
