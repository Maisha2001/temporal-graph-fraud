import argparse
import json
from pathlib import Path

from .data import digest


def log_release(run_directory, models, destination):
    import mlflow
    from mlflow.tracking import MlflowClient
    run_directory, models, destination = map(Path, [run_directory, models, destination])
    destination.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(destination.resolve().as_uri())
    mlflow.set_experiment('temporal-fraud-frozen-release')
    report = json.loads((run_directory / 'report.json').read_text())
    with mlflow.start_run(run_name='frozen-heldout-evaluation') as active:
        mlflow.set_tags({'evaluation_stage': 'final-heldout', 'training_performed': 'false',
                         'protocol_sha256': report['protocol_sha256'], 'data': 'IBM synthetic AML'})
        mlflow.log_params({'test_events': report['events'], 'cpu_threads': report['runtime']['cpu_threads'],
                           'model_selection': 'fixed existing checkpoints; no outcome-driven updates'})
        for arm, groups in report['models'].items():
            for metric in ['average_precision', 'brier', 'recall_at_capacity', 'precision_at_capacity']:
                value = groups['all'][metric]
                if value is not None:
                    mlflow.log_metric(arm + '.' + metric, value)
        for name in ['report.json', 'protocol.json', 'predictions-frozen.json']:
            mlflow.log_artifact(str(run_directory / name), artifact_path='evaluation')
        for path in sorted(models.iterdir()):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path='models')
        run_id = active.info.run_id
    client = MlflowClient()
    saved = client.get_run(run_id)
    if saved.info.status != 'FINISHED':
        raise RuntimeError('Tracking run did not finish')
    expected = report['models']['history_xgboost']['all']['average_precision']
    if saved.data.metrics['history_xgboost.average_precision'] != expected:
        raise RuntimeError('Logged metric mismatch')
    restored = client.download_artifacts(run_id, 'evaluation/report.json')
    if digest(restored) != digest(run_directory / 'report.json'):
        raise RuntimeError('Logged report mismatch')
    artifact_count = 0
    for path in models.iterdir():
        if path.is_file():
            restored = client.download_artifacts(run_id, 'models/' + path.name)
            if digest(restored) != digest(path):
                raise RuntimeError('Logged model mismatch')
            artifact_count += 1
    return {'run_id': run_id, 'status': saved.info.status, 'metrics_verified': len(saved.data.metrics),
            'model_artifacts_verified': artifact_count, 'report_sha256': digest(run_directory / 'report.json'),
            'tracking_mode': 'local file store; no cloud or remote API', 'model_training_performed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['run-directory', 'models', 'destination', 'output']:
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    path = Path(a.output)
    if path.exists():
        raise ValueError('Refusing to replace tracking evidence')
    result = log_release(a.run_directory, a.models, a.destination)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
