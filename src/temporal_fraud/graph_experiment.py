import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn

from .data import CATEGORIES, digest, load_development, split_events
from .features import NUMERIC, HISTORY, causal_features
from .graph_model import TemporalGraphClassifier, predict
from .metrics import evaluate
from .runtime import peak_memory_bytes
from .temporal import TemporalNeighborhood


def prepare(events, data_config, output):
    masks = split_events(events, data_config)
    replay = events.loc[masks['replay']].reset_index(drop=True)
    masks = {key: value[masks['replay']] for key, value in masks.items()}
    features = causal_features(replay.drop(columns='label'))
    for name in ['source_age_minutes', 'target_age_minutes']:
        features[name] = np.sign(features[name]) * np.log1p(np.abs(features[name]))
    preprocessing = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC + HISTORY),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), CATEGORIES),
    ])
    preprocessing.fit(features.loc[masks['train']])
    x = preprocessing.transform(features).astype(np.float32)
    index = TemporalNeighborhood(replay.drop(columns='label'))
    membership = replay[['event_id', 'timestamp', 'source', 'target', 'label']].copy()
    for name in ['train', 'validation', 'cold']:
        membership[name] = masks[name]
    if output is not None:
        output = Path(output)
        np.save(output / 'features.npy', x)
        joblib.dump(preprocessing, output / 'preprocessing.joblib')
        joblib.dump(index, output / 'neighborhood.joblib')
        membership.to_parquet(output / 'membership.parquet', index=False)
    return x, index, membership


def save_checkpoint(path, model, optimizer, epoch, updates, config, fingerprint, losses):
    temporary = path.with_suffix('.partial')
    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'updates': updates, 'config': config, 'fingerprint': fingerprint,
                'features': model.features, 'neighbors': model.neighbors,
                'losses': losses, 'torch_rng': torch.get_rng_state()}, temporary)
    temporary.replace(path)


def fit_model(x, index, membership, config, neighbors, output, fingerprint, resume=False):
    output = Path(output)
    checkpoint = output / f'neighbors-{neighbors}-latest.pt'
    torch.manual_seed(config['seed'])
    model = TemporalGraphClassifier(x.shape[1], config['hidden'], neighbors).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    train_rows = np.flatnonzero(membership.train.to_numpy())
    labels = torch.as_tensor(membership.label.to_numpy().copy(), device=x.device, dtype=torch.float32)
    positives = int(membership.loc[membership.train, 'label'].sum())
    if positives == 0 or positives == len(train_rows):
        raise ValueError('Training needs both classes')
    weight = np.sqrt((len(train_rows) - positives) / positives)
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=x.device, dtype=torch.float32))
    first_epoch, updates, losses = 0, 0, []
    if checkpoint.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        state = torch.load(checkpoint, map_location=x.device, weights_only=True)
        if state['fingerprint'] != fingerprint or state['config'] != config or state['neighbors'] != neighbors:
            raise ValueError('Checkpoint input/config mismatch')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu())
        first_epoch, updates, losses = state['epoch'], state['updates'], state['losses']
    started = time.perf_counter()
    for epoch in range(first_epoch, config['epochs']):
        model.train()
        rows = np.random.default_rng(config['seed'] + epoch).permutation(train_rows)
        total_loss = 0.0
        epoch_started = time.perf_counter()
        for start in range(0, len(rows), config['batch_size']):
            batch = rows[start:start + config['batch_size']]
            sample = index.sample(batch, neighbors) if neighbors else None
            device_rows = torch.as_tensor(batch, device=x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x, device_rows, sample), labels[device_rows])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
            total_loss += loss.item() * len(batch)
        losses.append({'epoch': epoch + 1, 'weighted_bce': total_loss / len(rows),
                       'seconds': time.perf_counter() - epoch_started})
        save_checkpoint(checkpoint, model, optimizer, epoch + 1, updates, config, fingerprint, losses)
        # Retain every completed epoch; selection is the fixed final epoch, never validation AP.
        epoch_path = output / f'neighbors-{neighbors}-epoch-{epoch + 1}.pt'
        if epoch_path.exists():
            raise ValueError('Refusing to replace historical epoch checkpoint')
        epoch_path.write_bytes(checkpoint.read_bytes())
        print(json.dumps({'neighbors': neighbors, **losses[-1], 'updates': updates}), flush=True)
    return model, {'losses': losses, 'updates': updates, 'fit_seconds_this_invocation': time.perf_counter() - started,
                   'positive_weight': float(weight), 'parameters': sum(p.numel() for p in model.parameters()),
                   'checkpoint_sha256': digest(checkpoint)}


def run(config, data_config, data, output, device='cpu', resume=False):
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError('Use a new output directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(config['threads'])
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable')
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    manifest_path = output / 'inputs.json'
    if resume and manifest_path.exists():
        inputs = json.loads(manifest_path.read_text())
        if inputs['config'] != config or inputs['data_config'] != data_config or digest(data) != data_config['sha256']:
            raise ValueError('Resume data/config mismatch')
        for name, expected in inputs['sha256'].items():
            if digest(output / name) != expected:
                raise ValueError('Cached input digest mismatch: ' + name)
        x = np.load(output / 'features.npy')
        index = joblib.load(output / 'neighborhood.joblib')
        membership = pd.read_parquet(output / 'membership.parquet')
    else:
        events, counts = load_development(data, data_config)
        x, index, membership = prepare(events, data_config, output)
        del events
        gc.collect()
        inputs = {'config': config, 'data_config': data_config, 'data_counts': counts,
                  'sha256': {name: digest(output / name) for name in ['features.npy', 'preprocessing.joblib', 'neighborhood.joblib', 'membership.parquet']}}
        manifest_path.write_text(json.dumps(inputs, indent=2) + '\n')
    preprocessing_seconds = time.perf_counter() - started
    fingerprint = digest(manifest_path)
    x = torch.as_tensor(x, device=device)
    validation_rows = np.flatnonzero(membership.validation.to_numpy())
    labels = membership.label.to_numpy()[validation_rows]
    event_ids = membership.event_id.to_numpy()[validation_rows]
    cold = membership.cold.to_numpy()[validation_rows]
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text()) if resume and report_path.exists() else {
        'stage': 'development_validation', 'test_scored': False, 'config': config, 'data_config': data_config,
        'input_manifest_sha256': fingerprint, 'data_counts': inputs['data_counts'],
        'train_events': int(membership.train.sum()), 'train_positives': int(membership.loc[membership.train, 'label'].sum()),
        'validation_events': len(validation_rows), 'models': {}}
    if resume and 'completed_at_utc' in report:
        for name, expected in report['artifact_sha256'].items():
            if digest(output / name) != expected:
                raise ValueError('Completed run artifact mismatch: ' + name)
        return report
    for neighbors in config['neighbors']:
        name = 'history_mlp' if neighbors == 0 else f'temporal_attention_{neighbors}'
        if resume and name in report['models']:
            for filename, expected in report['models'][name]['artifacts'].items():
                if digest(output / filename) != expected:
                    raise ValueError('Completed model artifact mismatch')
            continue
        model, training = fit_model(x, index, membership, config, neighbors, output, fingerprint, resume)
        scoring_started = time.perf_counter()
        scores = predict(model, x, index, validation_rows, config['batch_size'])
        score_seconds = time.perf_counter() - scoring_started
        state = torch.load(output / f'neighbors-{neighbors}-latest.pt', map_location=device, weights_only=True)
        reloaded = TemporalGraphClassifier(x.shape[1], config['hidden'], neighbors).to(device)
        reloaded.load_state_dict(state['state_dict'])
        check_rows = validation_rows[:min(512, len(validation_rows))]
        reference = predict(model, x, index, check_rows, config['batch_size'])
        restored = predict(reloaded, x, index, check_rows, config['batch_size'])
        difference = float(np.max(np.abs(reference - restored)))
        if difference > 1e-6:
            raise RuntimeError('Checkpoint reload predictions differ')
        predictions = membership.loc[membership.validation, ['event_id', 'timestamp', 'label', 'cold']].copy()
        predictions['score'] = scores
        pred_name = name + '-validation.parquet'
        predictions.to_parquet(output / pred_name, index=False)
        metrics = {group: evaluate(labels[mask], scores[mask], event_ids[mask], data_config['alert_fraction'])
                   for group, mask in [('validation', np.ones(len(scores), dtype=bool)), ('cold', cold), ('warm', ~cold)] if mask.any()}
        report['models'][name] = {**metrics, 'training': training, 'score_seconds': score_seconds,
                                 'reload_max_absolute_difference': difference, 'reload_events': len(check_rows),
                                 'artifacts': {filename: digest(output / filename) for filename in [pred_name, f'neighbors-{neighbors}-latest.pt']}}
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'model': name, 'validation': metrics['validation'], 'score_seconds': score_seconds}), flush=True)
        del model, reloaded
    report['runtime'] = {'device': device, 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                         'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                         'peak_process_memory_bytes': peak_memory_bytes(), 'preprocessing_seconds': preprocessing_seconds,
                         'total_seconds_this_invocation': time.perf_counter() - started,
                         'python': platform.python_version(), 'platform': platform.platform(),
                         'packages': {name: importlib.metadata.version(name) for name in ['torch', 'torch-geometric', 'numpy', 'pandas', 'scikit-learn']}}
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['artifact_sha256'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'report.json'}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/temporal-v1.json')
    parser.add_argument('--data-config', default='configs/ibm-hi-small.json')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), json.loads(Path(args.data_config).read_text()),
        args.data, args.output, args.device, args.resume)


if __name__ == '__main__':
    main()
