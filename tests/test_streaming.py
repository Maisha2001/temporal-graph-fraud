import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import os
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from temporal_fraud.api import create_app
from temporal_fraud.features import causal_features
from temporal_fraud.store import Conflict, Store
from temporal_fraud.streaming import (MinuteBatch, advance_accounts, empty_account,
                                     graph_batch, online_features)
from temporal_fraud.temporal import TemporalNeighborhood


def packet(sequence=0, minute='2022-09-07T12:00:00', event_id=1):
    return MinuteBatch(sequence=sequence, minute=minute, events=[
        dict(event_id=event_id, source='a', target='b', paid=10., received=10.,
             receiving_currency='USD', payment_currency='USD', format='ACH')])


@pytest.mark.parametrize('change', [
    {'events': []}, {'sequence': True}, {'minute': '2022-09-07T12:00:01'},
    {'minute': '2022-09-07T12:00:00Z'}, {'label': 1}
])
def test_invalid_batch(change):
    value = packet().model_dump()
    value.update(change)
    with pytest.raises(ValidationError):
        MinuteBatch(**value)


@pytest.mark.parametrize('change', [{'paid': float('nan')}, {'paid': -1}, {'label': 1}, {'source': ''}])
def test_invalid_transaction(change):
    value = packet().model_dump()
    value['events'][0].update(change)
    with pytest.raises(ValidationError):
        MinuteBatch(**value)


def test_causal_features_and_neighbors_match_offline():
    frames = []
    accounts = {k: empty_account() for k in ['a','b','c']}
    online = []
    for step in range(4):
        value = packet(step, datetime(2022,9,7,12,step), step * 2).model_dump()
        value['events'] += [{**value['events'][0], 'event_id': step * 2 + 1, 'source':'b', 'target':'c'}]
        frame = MinuteBatch(**value).frame()
        feature = online_features(frame, accounts)
        vectors = np.asarray([[step, row, 1] for row in range(2)], dtype=np.float32)
        packed, sample = graph_batch(frame, vectors, accounts, 32)
        if step == 0:
            assert (sample[0] == -1).all()
        else:
            assert (sample[1][sample[0] >= 0] > 0).all()
        online.append(feature);frames.append(frame)
        advance_accounts(frame, vectors, accounts)
    full = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(pd.concat(online, ignore_index=True), causal_features(full))
    index = TemporalNeighborhood(full)
    history, ages, direction = index.sample(np.array([6,7]), 32)
    before = {k: empty_account() for k in accounts}
    for step, frame in enumerate(frames[:3]):
        advance_accounts(frame, np.asarray([[step,row,1] for row in range(2)], dtype=np.float32), before)
    packed, sample = graph_batch(frames[3], np.zeros((2,3), dtype=np.float32), before, 32)
    np.testing.assert_array_equal(sample[1], ages)
    np.testing.assert_array_equal(sample[2], direction)
    for row, side, j in zip(*np.where(history >= 0)):
        old = history[row,side,j]
        np.testing.assert_array_equal(packed[sample[0][row,side,j]], [old // 2, old % 2, 1])


def test_same_minute_order_invariance_and_self_edge():
    batch = packet()
    value = batch.model_dump()
    value['events'].append({**value['events'][0], 'event_id':2, 'target':'a'})
    original = MinuteBatch(**value)
    value['events'].reverse()
    reordered = MinuteBatch(**value)
    assert original.fingerprint() == reordered.fingerprint()
    accounts = {k:empty_account() for k in ['a','b']}
    frame = original.frame()
    features = online_features(frame, accounts)
    assert (features.log_source_out == 0).all()
    advance_accounts(frame, np.zeros((2,3)), accounts)
    assert accounts['a']['out']==2 and accounts['a']['in']==1
    assert len(accounts['a']['neighbors'])==2


class FakeModels:
    fingerprint = 'fixture-model'
    first_minute = datetime(2022,9,7,12)
    end_minute = datetime(2022,9,8)

    def score(self, frame, accounts):
        features = online_features(frame, accounts)
        return ([{'event_id':int(row.event_id), 'history_xgboost':float(features.log_source_out.iloc[i])}
                 for i,row in enumerate(frame.itertuples())], np.zeros((len(frame),3), dtype=np.float32))


@pytest.fixture
def db():
    dsn = os.environ.get('FRAUD_TEST_DSN')
    if not dsn:
        pytest.skip('PostgreSQL integration runs in the separate service CI job')
    store = Store(dsn, FakeModels());store.initialize()
    stream = 'test-' + uuid.uuid4().hex
    store.create(stream, {'model_fingerprint':store.models.fingerprint, 'last_minute':'2022-09-07T11:59:00',
                          'accounts':{k:empty_account() for k in ['a','b']}})
    return store,stream


def test_database_rollback_restart_and_idempotence(db):
    store,stream=db
    with pytest.raises(RuntimeError):
        store.apply(stream, packet(), fail_before_commit=True)
    assert store.status(stream)['events']==0
    expected=store.apply(stream,packet())
    restored=Store(store.dsn,FakeModels())
    assert restored.apply(stream,packet())=={**expected,'duplicate':True}
    assert restored.status(stream)['events']==1
    with pytest.raises(Conflict):
        restored.apply(stream,packet(event_id=2))
    with pytest.raises(Conflict):
        restored.apply(stream,packet(1,'2022-09-07T12:01:00'))
    result=restored.apply(stream,packet(1,'2022-09-07T12:01:00',2))
    assert result['scores'][0]['history_xgboost']==pytest.approx(np.log(2),abs=1e-7)


def test_database_serializes_duplicate_writers(db):
    store,stream=db
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses=list(pool.map(lambda _:store.apply(stream,packet()),range(4)))
    assert sum(not r['duplicate'] for r in responses)==1
    assert store.status(stream)['events']==1


def test_api_conflicts_scope_holdout_and_validation(db):
    store,stream=db
    client=TestClient(create_app(store))
    path=f'/streams/{stream}/batches'
    assert client.get('/health').status_code==200
    assert client.get('/streams/absent').status_code==404
    assert client.post(path,json=packet(2).model_dump(mode='json')).status_code==409
    assert client.post(path,json=packet(minute='2022-09-08T00:00:00').model_dump(mode='json')).status_code==409
    value=packet().model_dump(mode='json');value['events'][0]['source']='unknown'
    assert client.post(path,json=value).status_code==409
    value=packet().model_dump(mode='json');value['events'][0]['label']=1
    assert client.post(path,json=value).status_code==422
    assert client.post(path,content=b'x'*8_000_001).status_code==413
    assert client.post(path,json=packet().model_dump(mode='json')).status_code==200
    assert client.get(f'/streams/{stream}/batches/0').json()['scores'][0]['event_id']==1


def test_kafka_redelivery_after_database_commit(db):
    bootstrap=os.environ.get('FRAUD_TEST_KAFKA')
    if not bootstrap:
        pytest.skip('Kafka integration runs in the separate service CI job')
    from temporal_fraud.broker import create_topic,publish,consume
    store,stream=db
    topic='fraud-test-'+uuid.uuid4().hex;group=topic+'-group'
    create_topic(bootstrap,topic)
    publish(bootstrap,topic,stream,[packet(),packet(1,'2022-09-07T12:01:00',2)])
    with pytest.raises(RuntimeError,match='after database commit'):
        consume(bootstrap,topic,group,store,limit=2,crash_after_database=1)
    assert store.status(stream)['events']==1
    assert consume(bootstrap,topic,group,Store(store.dsn,FakeModels()),limit=2)==2
    assert store.status(stream)['events']==2
