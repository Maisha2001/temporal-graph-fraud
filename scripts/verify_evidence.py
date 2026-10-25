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
