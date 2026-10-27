import json
import pytest
from temporal_fraud.tracking import log_release


def test_local_tracking_roundtrip(tmp_path):
    pytest.importorskip('mlflow')
    run=tmp_path/'run';run.mkdir();models=tmp_path/'models';models.mkdir()
    report={'events':10,'protocol_sha256':'authored','runtime':{'cpu_threads':4},
            'models':{'history_xgboost':{'all':{'average_precision':.5,'brier':.1,'recall_at_capacity':.2,'precision_at_capacity':.3}}}}
    for name,value in [('report.json',report),('protocol.json',{'authored':True}),('predictions-frozen.json',{'authored':True})]:
        (run/name).write_text(json.dumps(value))
    (models/'authored.txt').write_text('Original test artifact')
    result=log_release(run,models,tmp_path/'mlruns')
    assert result['status']=='FINISHED' and result['metrics_verified']==4 and result['model_artifacts_verified']==1
