import numpy as np
import pandas as pd

from .data import CATEGORIES

NUMERIC = ['log_paid', 'log_received', 'self_transfer', 'currency_change', 'hour_sin', 'hour_cos']
HISTORY = ['log_source_out', 'log_target_in', 'source_age_minutes', 'target_age_minutes']


def transaction_features(events):
    hour = events['timestamp'].dt.hour + events['timestamp'].dt.minute / 60
    return pd.DataFrame({
        'log_paid': np.log1p(events['paid']), 'log_received': np.log1p(events['received']),
        'self_transfer': events['source'].eq(events['target']).astype('float32'),
        'currency_change': events['payment_currency'].ne(events['receiving_currency']).astype('float32'),
        'hour_sin': np.sin(2 * np.pi * hour / 24), 'hour_cos': np.cos(2 * np.pi * hour / 24),
        **{name: events[name].astype(str) for name in CATEGORIES},
    }, index=events.index)


def causal_features(events):
    if not len(events) or not events['timestamp'].is_monotonic_increasing or not events['event_id'].is_unique:
        raise ValueError('Replay requires ordered timestamps and unique event IDs')
    features = transaction_features(events)
    n = len(events)
    codes, nodes = pd.factorize(pd.concat([events['source'], events['target']], ignore_index=True), sort=True)
    source, target = codes[:n], codes[n:]
    out_count = np.zeros(len(nodes), dtype=np.int64)
    in_count = np.zeros(len(nodes), dtype=np.int64)
    last_out = np.full(len(nodes), -1, dtype=np.int64)
    last_in = np.full(len(nodes), -1, dtype=np.int64)
    minutes = events['timestamp'].to_numpy(dtype='datetime64[m]').astype('int64')
    history = np.zeros((n, len(HISTORY)), dtype=np.float32)
    boundaries = np.r_[0, np.flatnonzero(np.diff(minutes)) + 1, n]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        src, dst, now = source[lo:hi], target[lo:hi], minutes[lo]
        history[lo:hi, 0] = np.log1p(out_count[src])
        history[lo:hi, 1] = np.log1p(in_count[dst])
        history[lo:hi, 2] = np.where(last_out[src] < 0, -1, now - last_out[src])
        history[lo:hi, 3] = np.where(last_in[dst] < 0, -1, now - last_in[dst])
        # A whole timestamp group is scored before any of its edges enter history.
        np.add.at(out_count, src, 1)
        np.add.at(in_count, dst, 1)
        last_out[src], last_in[dst] = now, now
    for column, values in zip(HISTORY, history.T):
        features[column] = values
    return features
