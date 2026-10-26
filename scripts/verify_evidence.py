from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
count = 0
for manifest_path in sorted((root / 'results').glob('*/checksums.json')):
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Evidence mismatch: {name}')
        count += 1
report = json.loads((root / 'results/baseline-v1/report.json').read_text())
assert report['test_scored'] is False
assert report['data_sha256'] == report['config']['sha256']
assert report['models']['history_xgboost']['validation']['events'] == report['validation_events']
corrected = json.loads((root / 'results/capacity-tie-audit-v1/report.json').read_text())
assert corrected['test_scored'] is False
assert corrected['artifact_sha256'] == report['artifact_sha256']
prior = corrected['models']['train_prior']['validation']
assert abs(prior['recall_at_capacity'] - prior['alerts'] / prior['events']) < 1e-12
for model in report['models']:
    assert corrected['models'][model]['validation']['average_precision'] == report['models'][model]['validation']['average_precision']
print(f'Verified {count} evidence files; final holdout remains unscored.')

graph = json.loads((root / 'results/temporal-v1/report.json').read_text())
audit = json.loads((root / 'results/temporal-v1/audit.json').read_text())
comparison = json.loads((root / 'results/temporal-v1/comparison.json').read_text())
assert graph['test_scored'] is False and audit['test_scored'] is False and comparison['test_scored'] is False
assert graph['train_events'] == report['train_events']
assert graph['validation_events'] == report['validation_events']
assert graph['data_config'] == report['config']
assert audit['passed'] and audit['original_baseline_artifacts_verified'] == 7
assert audit['run_report_sha256'] == comparison['graph_report_sha256']
for name, model in graph['models'].items():
    assert len(model['training']['losses']) == graph['config']['epochs']
    assert model['training']['updates'] == 1359
    assert comparison['models'][name]['validation'] == model['validation']
for name, model in corrected['models'].items():
    assert comparison['models'][name]['validation'] == model['validation']
print('Graph comparison, baseline preservation and independent audit are consistent.')
