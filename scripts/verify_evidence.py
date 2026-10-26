from pathlib import Path
import hashlib
import json
import argparse
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--revision')
mode.add_argument('--staged', action='store_true')
args = parser.parse_args()
blobs = None
if args.revision or args.staged:
    command = ['git', 'ls-files', '-s', '-z'] if args.staged else ['git', 'ls-tree', '-r', '-z', args.revision]
    entries = subprocess.check_output(command, cwd=root).decode().split('\0')
    blobs = {}
    for entry in filter(None, entries):
        metadata, name = entry.split('\t', 1)
        fields = metadata.split()
        blobs[name] = fields[1] if args.staged else fields[2]

def read(name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evidence path escapes repository')
    if blobs is None:
        return path.read_bytes()
    return subprocess.check_output(['git', 'cat-file', 'blob', blobs[name]], cwd=root)

def load(name):
    return json.loads(read(name))

manifests = sorted(name for name in blobs if name.startswith('results/') and name.endswith('/checksums.json')) if blobs is not None else [p.relative_to(root).as_posix() for p in sorted((root / 'results').glob('*/checksums.json'))]
count = 0
for manifest_name in manifests:
    manifest = load(manifest_name)
    for name, expected in manifest.items():
        if hashlib.sha256(read(name)).hexdigest() != expected:
            raise SystemExit(f'Evidence mismatch: {name}')
        count += 1
report = load('results/baseline-v1/report.json')
assert report['test_scored'] is False
assert report['data_sha256'] == report['config']['sha256']
assert report['models']['history_xgboost']['validation']['events'] == report['validation_events']
corrected = load('results/capacity-tie-audit-v1/report.json')
assert corrected['test_scored'] is False
assert corrected['artifact_sha256'] == report['artifact_sha256']
prior = corrected['models']['train_prior']['validation']
assert abs(prior['recall_at_capacity'] - prior['alerts'] / prior['events']) < 1e-12
for model in report['models']:
    assert corrected['models'][model]['validation']['average_precision'] == report['models'][model]['validation']['average_precision']
print(f'Verified {count} evidence files; final holdout remains unscored.')

graph = load('results/temporal-v1/report.json')
audit = load('results/temporal-v1/audit.json')
comparison = load('results/temporal-v1/comparison.json')
assert graph['test_scored'] is False and audit['test_scored'] is False and comparison['test_scored'] is False
assert graph['train_events'] == report['train_events']
assert graph['validation_events'] == report['validation_events']
assert graph['data_config'] == report['config']
assert audit['passed'] and audit['original_baseline_artifacts_verified'] == 7
assert audit['run_report_sha256'] == comparison['raw_run_report_sha256']
assert hashlib.sha256(read('results/temporal-v1/report.json')).hexdigest() == comparison['graph_report_sha256']
for name, model in graph['models'].items():
    assert len(model['training']['losses']) == graph['config']['epochs']
    assert model['training']['updates'] == 1359
    assert comparison['models'][name]['validation'] == model['validation']
for name, model in corrected['models'].items():
    assert comparison['models'][name]['validation'] == model['validation']
print('Graph comparison, baseline preservation and independent audit are consistent.')
