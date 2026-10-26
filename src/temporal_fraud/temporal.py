import numpy as np
import pandas as pd

from .data import CATEGORIES


class TemporalNeighborhood:
    """Index incident interactions; queries exclude the entire current minute."""

    def __init__(self, events):
        if not len(events) or not events.timestamp.is_monotonic_increasing or not events.event_id.is_unique:
            raise ValueError('Ordered events with unique IDs required')
        if events[['timestamp', 'source', 'target']].isna().any().any():
            raise ValueError('Missing graph endpoint or time')
        self.minutes = events.timestamp.to_numpy(dtype='datetime64[m]').astype(np.int64)
        if np.any(self.minutes < 0) or np.any(self.minutes >= 100_000_000):
            raise ValueError('Timestamp outside index range')
        codes, nodes = pd.factorize(pd.concat([events.source, events.target]), sort=True)
        n = len(events)
        self.endpoints = np.stack([codes[:n], codes[n:]], axis=1)
        # Observable-content ordering prevents source-row IDs from selecting tied neighbors.
        columns = ['source', 'target', 'paid', 'received', *CATEGORIES]
        priority = pd.util.hash_pandas_object(events[columns], index=False).to_numpy()
        nonself = codes[:n] != codes[n:]
        rows = np.r_[np.arange(n), np.arange(n)[nonself]]
        owners = np.r_[codes[:n], codes[n:][nonself]]
        directions = np.r_[np.ones(n, dtype=np.int8), -np.ones(nonself.sum(), dtype=np.int8)]
        order = np.lexsort((priority[rows], self.minutes[rows], owners))
        self.rows = rows[order].astype(np.int32)
        self.directions = directions[order]
        self.keys = owners[order] * 100_000_000 + self.minutes[self.rows]
        self.starts = np.r_[0, np.cumsum(np.bincount(owners, minlength=len(nodes)))]

    def sample(self, rows, neighbors):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(self.minutes)) or neighbors < 1:
            raise ValueError('Invalid query rows or neighbor budget')
        endpoints = self.endpoints[rows].reshape(-1)
        times = np.repeat(self.minutes[rows], 2)
        ends = np.searchsorted(self.keys, endpoints * 100_000_000 + times, side='left')
        positions = ends[:, None] - 1 - np.arange(neighbors)
        valid = positions >= self.starts[endpoints, None]
        positions = np.maximum(positions, 0)
        history = self.rows[positions]
        age = times[:, None] - self.minutes[history]
        history = np.where(valid, history, -1)
        age = np.where(valid, age, 0).astype(np.float32)
        direction = np.where(valid, self.directions[positions], 0).astype(np.float32)
        return history.reshape(-1, 2, neighbors), age.reshape(-1, 2, neighbors), direction.reshape(-1, 2, neighbors)
