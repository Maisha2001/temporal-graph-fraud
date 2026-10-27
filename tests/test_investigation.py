import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from temporal_fraud.data import digest
from temporal_fraud.investigation import create_app, load_bundle


def authored_bundle(path):
    path.mkdir(parents=True, exist_ok=True)
    cases = []
    for i in range(3):
        cases.append({'event_id': i + 1, 'timestamp': f'2022-09-07T12:0{i}:00', 'source': 'A', 'target': chr(66+i),
                      'paid': 10+i, 'payment_currency': 'USD', 'format': 'ACH', 'minute_index': i,
                      'cold': i==1, 'label': i%2, 'history_xgboost': .4-i*.1,
                      'history_xgboost_calibrated': .04-i*.01, 'temporal_attention_32': .2+i*.1,
                      'temporal_attention_32_calibrated': .02+i*.01,
                      'context':[{'event_id':10+i,'timestamp':'2022-09-07T11:59:00','source':'X','target':'A','paid':5,'payment_currency':'USD','format':'ACH'}]})
    value={'title':'Transaction Atlas','mode':'Authored test fixture','period':'Authored fixture - September 7',
           'events_scored':3,'minutes':30,'selection':'Authored fixture, not model evidence.',
           'data_notice':'Original authored fixture; no IBM records.','limitations':'Synthetic fixture only.',
           'model_fingerprint':'authored','test_data_used':False,'latency_p95_ms':1,'cases':cases}
    (path/'cases.json').write_text(json.dumps(value),encoding='utf-8')
    (path/'manifest.json').write_text(json.dumps({'cases_sha256':digest(path/'cases.json')}),encoding='utf-8')
    return path


def test_case_ranking_time_search_and_labels(tmp_path):
    client=TestClient(create_app(authored_bundle(tmp_path/'demo')))
    assert client.get('/health').json()['test_data_used'] is False
    assert client.get('/api/overview').json()['case_count']==3
    assert [x['event_id'] for x in client.get('/api/cases').json()]==[3,2,1]
    assert [x['event_id'] for x in client.get('/api/cases?model=history_xgboost').json()]==[1,2,3]
    assert len(client.get('/api/cases?minute=1').json())==1
    assert client.get('/api/cases?cold=true').json()[0]['event_id']==2
    assert client.get('/api/cases?search=unknown').json()==[]
    assert 'label' not in client.get('/api/cases/2').json()
    assert client.get('/api/cases/2?outcomes=true').json()['label']==1
    assert all('label' not in x and 'context' not in x for x in client.get('/api/cases').json())


@pytest.mark.parametrize('route',['/api/cases?model=bad','/api/cases?minute=0','/api/cases?minute=31'])
def test_invalid_controls(tmp_path,route):
    client=TestClient(create_app(authored_bundle(tmp_path/'demo')))
    assert client.get(route).status_code==422


def test_integrity_causality_and_unknown_case(tmp_path):
    folder=authored_bundle(tmp_path/'demo')
    client=TestClient(create_app(folder))
    assert client.get('/api/cases/999').status_code==404
    assert client.get('/').status_code==200 and client.get('/assets/app.js').status_code==200
    p=folder/'cases.json';data=json.loads(p.read_text());data['cases'][0]['context'][0]['timestamp']='2022-09-08T00:00:00';p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='digest'):load_bundle(folder)
    (folder/'manifest.json').write_text(json.dumps({'cases_sha256':digest(p)}))
    with pytest.raises(ValueError,match='future'):load_bundle(folder)
