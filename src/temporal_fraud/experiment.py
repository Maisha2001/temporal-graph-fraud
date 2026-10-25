import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .data import CATEGORIES, digest, download, load_development, split_events
from .features import NUMERIC, HISTORY, causal_features
from .metrics import evaluate
from .runtime import peak_memory_bytes


def run(events, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    masks = split_events(events, config)
    replay = events.loc[masks['replay']].reset_index(drop=True)
    masks = {key: value[masks['replay']] for key, value in masks.items()}
    features = causal_features(replay.drop(columns='label'))
    feature_seconds = time.perf_counter() - started
    train, valid = masks['train'], masks['validation']
    y = replay['label'].to_numpy()
    if len(np.unique(y[train])) != 2:
        raise ValueError('Training requires both classes')
    membership = replay[['event_id', 'timestamp', 'source', 'target']].copy()
    membership['train'], membership['validation'], membership['cold'] = train, valid, masks['cold']
    membership.to_parquet(output / 'membership.parquet', index=False)
    report = {
        'stage': 'development_validation', 'test_scored': False, 'config': config,
        'train_events': int(train.sum()), 'train_positives': int(y[train].sum()),
        'validation_events': int(valid.sum()), 'replay_events': len(replay),
        'excluded_pre_fit_entity_edges': int((~split_events(events, config)['replay']).sum()),
        'training_latest_event': str(replay.loc[train, 'timestamp'].max()),
        'validation_first_event': str(replay.loc[valid, 'timestamp'].min()),
        'feature_seconds': feature_seconds, 'models': {},
    }
    prior = np.full(valid.sum(), y[train].mean())
    report['models']['train_prior'] = {'validation': evaluate(y[valid], prior, replay.loc[valid, 'event_id'], config['alert_fraction'])}
    for name, numeric in [('transaction_xgboost', NUMERIC), ('history_xgboost', NUMERIC + HISTORY)]:
        preprocessing = ColumnTransformer([
            ('numeric', 'passthrough', numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore', dtype=np.float32), CATEGORIES),
        ], sparse_threshold=1.0)
        transform_started = time.perf_counter()
        x_train = preprocessing.fit_transform(features.loc[train]).astype(np.float32)
        x_valid = preprocessing.transform(features.loc[valid]).astype(np.float32)
        positive = int(y[train].sum())
        model = XGBClassifier(n_estimators=config['trees'], max_depth=config['max_depth'],
                              learning_rate=config['learning_rate'], tree_method='hist', max_bin=128,
                              n_jobs=config['threads'], random_state=config['seed'],
                              scale_pos_weight=np.sqrt((train.sum() - positive) / positive),
                              eval_metric='logloss', device='cpu')
        fit_started = time.perf_counter()
        model.fit(x_train, y[train])
        fit_seconds = time.perf_counter() - fit_started
        scoring_started = time.perf_counter()
        scores = model.predict_proba(x_valid)[:, 1]
        scoring_seconds = time.perf_counter() - scoring_started
        model.save_model(output / (name + '.ubj'))
        joblib.dump(preprocessing, output / (name + '.joblib'))
        evaluation = {'validation': evaluate(y[valid], scores, replay.loc[valid, 'event_id'], config['alert_fraction']),
                      'fit_seconds': fit_seconds, 'score_seconds': scoring_seconds,
                      'transform_fit_score_seconds': time.perf_counter() - transform_started,
                      'features': list(preprocessing.get_feature_names_out()),
                      'model_sha256': digest(output / (name + '.ubj'))}
        for group, selected in [('cold', masks['cold'][valid]), ('warm', ~masks['cold'][valid])]:
            if selected.any():
                evaluation[group] = evaluate(y[valid][selected], scores[selected], replay.loc[valid, 'event_id'].to_numpy()[selected], config['alert_fraction'])
        predictions = replay.loc[valid, ['event_id', 'timestamp', 'label']].copy()
        predictions['score'], predictions['cold'] = scores, masks['cold'][valid]
        predictions.to_parquet(output / (name + '-validation.parquet'), index=False)
        report['models'][name] = evaluation
        print(json.dumps({'model': name, 'validation': evaluation['validation'], 'fit_seconds': fit_seconds}), flush=True)
    report['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                         'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'pyarrow']},
                         'total_seconds': time.perf_counter() - started,
                         'peak_process_memory_bytes': peak_memory_bytes(), 'device': 'cpu', 'threads': config['threads']}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ibm-hi-small.json')
    parser.add_argument('--data', default='data/HI-Small_Trans.csv')
    parser.add_argument('--output', default='artifacts/baseline')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.download:
        download(config, args.data)
    loaded_at = time.perf_counter()
    events, counts = load_development(args.data, config)
    load_seconds = time.perf_counter() - loaded_at
    report = run(events, config, args.output)
    report.update(data_counts=counts, load_seconds=load_seconds, data_sha256=digest(args.data),
                  config_sha256=digest(args.config), completed_at_utc=datetime.now(timezone.utc).isoformat())
    output = Path(args.output)
    report['artifact_sha256'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'report.json'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(output / 'report.json'), 'data': counts, 'seconds': report['runtime']['total_seconds']}))


if __name__ == '__main__':
    main()
