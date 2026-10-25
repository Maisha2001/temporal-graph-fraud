from pathlib import Path
import csv
import hashlib
import urllib.request

import numpy as np
import pandas as pd

HEADER = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account',
          'Amount Received', 'Receiving Currency', 'Amount Paid',
          'Payment Currency', 'Payment Format', 'Is Laundering']
NAMES = ['timestamp', 'source_bank', 'source_account', 'target_bank',
         'target_account', 'received', 'receiving_currency', 'paid',
         'payment_currency', 'format', 'label']
CATEGORIES = ['receiving_currency', 'payment_currency', 'format']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(config, destination):
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != config['sha256']:
            raise ValueError('Existing data does not match the pinned digest')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    with urllib.request.urlopen(config['source'], timeout=120) as response, temporary.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if digest(temporary) != config['sha256']:
        raise ValueError('Download digest differs from the pinned dataset')
    temporary.replace(destination)
    return destination


def entity_bucket(entity, modulus=10):
    return int.from_bytes(hashlib.sha256(entity.encode('utf-8')).digest()[:8], 'big') % modulus


def load_development(path, config, verify=True):
    if verify and digest(path) != config['sha256']:
        raise ValueError('Dataset digest mismatch')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        if next(csv.reader(stream)) != HEADER:
            raise ValueError('Unexpected IBM transaction header')
    stop = pd.Timestamp(config['test_start'])
    pieces, row_offset, reserved = [], 0, 0
    for chunk in pd.read_csv(path, header=0, names=NAMES, dtype='string', chunksize=250_000):
        chunk['event_id'] = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)
        row_offset += len(chunk)
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='%Y/%m/%d %H:%M', errors='raise')
        if chunk['timestamp'].isna().any():
            raise ValueError('Missing timestamp')
        # Never convert, summarize or score reserved test labels in development.
        selected = chunk['timestamp'] < stop
        reserved += int((~selected).sum())
        pieces.append(normalize(chunk.loc[selected].copy()))
    frame = pd.concat(pieces, ignore_index=True).sort_values(['timestamp', 'event_id']).reset_index(drop=True)
    if not len(frame):
        raise ValueError('No development events')
    return frame, {'raw_rows': row_offset, 'reserved_test_rows': reserved, 'development_rows': len(frame)}


def normalize(frame):
    if frame[NAMES].isna().any().any():
        raise ValueError('Missing transaction value')
    for direction in ['source', 'target']:
        bank = pd.to_numeric(frame[direction + '_bank'], errors='raise')
        if (bank < 0).any() or (bank % 1 != 0).any():
            raise ValueError('Invalid bank identifier')
        account = frame[direction + '_account'].str.strip().str.upper()
        if account.eq('').any():
            raise ValueError('Empty account identifier')
        frame[direction] = bank.astype('int64').astype('string') + ':' + account
    for name in ['paid', 'received']:
        frame[name] = pd.to_numeric(frame[name], errors='raise').astype('float64')
        if not np.isfinite(frame[name]).all() or (frame[name] < 0).any():
            raise ValueError('Invalid transaction amount')
    frame['label'] = pd.to_numeric(frame['label'], errors='raise')
    if not frame['label'].isin([0, 1]).all():
        raise ValueError('Labels must be binary')
    frame['label'] = frame['label'].astype('int8')
    for name in CATEGORIES:
        if frame[name].str.strip().eq('').any():
            raise ValueError('Empty transaction category')
    return frame[['event_id', 'timestamp', 'source', 'target', 'paid', 'received', *CATEGORIES, 'label']]


def split_events(frame, config):
    fit_at, test_start = pd.Timestamp(config['fit_at']), pd.Timestamp(config['test_start'])
    if not fit_at < test_start or config['label_delay_hours'] < 0:
        raise ValueError('Invalid split or label delay')
    nodes = pd.unique(pd.concat([frame['source'], frame['target']], ignore_index=True))
    buckets = {str(n): entity_bucket(str(n), config['entity_holdout_modulus']) for n in nodes}
    cold = (frame['source'].map(buckets).eq(config['entity_holdout_bucket']) |
            frame['target'].map(buckets).eq(config['entity_holdout_bucket'])).to_numpy()
    before_fit = frame['timestamp'].lt(fit_at).to_numpy()
    available = (frame['timestamp'] + pd.Timedelta(hours=config['label_delay_hours'])).lt(fit_at).to_numpy()
    train = before_fit & available & ~cold
    validation = frame['timestamp'].ge(fit_at).to_numpy() & frame['timestamp'].lt(test_start).to_numpy()
    replay = ~(before_fit & cold)
    if not train.any() or not validation.any():
        raise ValueError('Empty training or validation split')
    return {'train': train, 'validation': validation, 'cold': cold, 'replay': replay}
