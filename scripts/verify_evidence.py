from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'results/baseline-v1/checksums.json').read_text())
for name, expected in manifest.items():
    path = (root / name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Evidence mismatch: {name}')
report = json.loads((root / 'results/baseline-v1/report.json').read_text())
assert report['test_scored'] is False
assert report['data_sha256'] == report['config']['sha256']
assert report['models']['history_xgboost']['validation']['events'] == report['validation_events']
print(f'Verified {len(manifest)} evidence files; final holdout remains unscored.')
