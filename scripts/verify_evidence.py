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
print(f'Verified {count} evidence files; historical development evidence preserved.')

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

risk = load('results/robustness-v1/report.json')
risk_audit = load('results/robustness-v1/audit.json')
assert risk['completed'] and risk['test_scored'] is False and risk_audit['test_scored'] is False
assert risk_audit['passed'] and risk_audit['frozen_artifacts_unchanged'] == 27
assert hashlib.sha256(read('results/robustness-v1/report.json')).hexdigest() == risk_audit['study_sha256']
assert risk['delays']['0']['training_events'] > risk['delays']['24']['training_events'] > risk['delays']['48']['training_events']
for name, calibration in risk['calibration'].items():
    fit = calibration['calibrator']
    assert fit['latest_label_available_at'] < fit['fitted_at'] <= calibration['evaluation_first_event']
    assert calibration['evaluation_last_event'] < risk['protocol']['evaluation_end'].replace('T', ' ')
    assert calibration['alerts']['count'] <= calibration['alerts']['budget_sum']
    assert all(hour['alerts'] <= hour['budget'] for hour in calibration['alerts']['hourly'])
assert risk['frozen_artifact_sha256'] == {**{'baseline/' + n: h for n, h in report['artifact_sha256'].items()}, **{'graph/' + n: h for n, h in graph['artifact_sha256'].items()}}
print('Delayed-label, calibration, hourly capacity and frozen-artifact evidence are consistent.')

service = load('results/service-v1/report.json')
service_audit = load('results/service-v1/audit.json')
assert service['completed'] and service['test_scored'] is False
assert service_audit['passed'] and service_audit['test_scored'] is False
assert hashlib.sha256(read('results/service-v1/report.json')).hexdigest() == service_audit['report_sha256']
assert service['events_per_stream'] * 4 == service_audit['scored_event_copies_checked']
assert service['api_restart_pid_changed'] and len(set(service['api_process_ids'])) == 2
assert service['http_api_peak_process_bytes'] > 100_000_000
assert service['kafka']['duplicate_count'] == 1 and service['rollback_test_passed']
assert service_audit['persistent_account_states_checked'] == 33044
assert service_audit['persistent_neighbor_vectors_checked'] == 772000
assert all(max(errors.values()) < 1e-6 for errors in service_audit['raw_and_calibrated_max_absolute_errors'].values())
print('Stateful service parity, persistence, recovery and measured load evidence are consistent.')


final = load('results/heldout-v1/report.json')
final_audit = load('results/heldout-v1/audit.json')
protocol = load('results/heldout-v1/protocol.json')
predictions = load('results/heldout-v1/predictions-frozen.json')
assert final['test_scored'] is True and final_audit['passed']
assert final['events'] == 1346673 and final['positives'] == 2150
assert hashlib.sha256(read('results/heldout-v1/report.json')).hexdigest() == final_audit['report_sha256']
assert final_audit['freeze_precedes_label_open']
assert hashlib.sha256(read('results/heldout-v1/protocol.json')).hexdigest() == final['protocol_sha256'] == predictions['protocol_sha256']
assert hashlib.sha256(read('results/heldout-v1/predictions-frozen.json')).hexdigest() == final['prediction_freeze_sha256']
assert protocol['frozen_at_utc'] < predictions['frozen_at_utc'] < final['labels_opened_at_utc']
assert final_audit['source_artifacts_verified'] == 74
assert final_audit['independent_metric_groups'] == 112
assert final_audit['closed_hour_policies_checked'] == 434
for name, hashes in load('results/heldout-v1/source-provenance.json')['files'].items():
    assert protocol['source_sha256'][name] == hashes['frozen_windows_bytes_sha256']
    assert hashlib.sha256(read('src/temporal_fraud/' + name).replace(b'\r\n',b'\n')).hexdigest() == hashes['canonical_git_lf_sha256']
browser = load('results/release-v1/browser.json')
tracking = load('results/release-v1/tracking.json')
assert browser['passed'] and browser['actual_saved_model_cases'] == 93 and browser['cold_filter_cases'] == 14
assert browser['server_stopped'] and browser['test_data_used'] is False
assert tracking['status'] == 'FINISHED' and tracking['metrics_verified'] == 32
assert tracking['model_artifacts_verified'] == 14 and tracking['report_sha256'] == final_audit['report_sha256']
print('Final evaluation, canonical source provenance, investigation and local MLflow evidence verified.')
