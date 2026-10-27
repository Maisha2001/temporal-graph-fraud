import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator
import torch
from xgboost import XGBClassifier

from .data import CATEGORIES, digest
from .features import HISTORY, transaction_features
from .graph_model import TemporalGraphClassifier
from .risk import MonotoneCalibration


class Transaction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: StrictInt = Field(ge=0)
    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)
    paid: FiniteFloat = Field(ge=0, le=1e15)
    received: FiniteFloat = Field(ge=0, le=1e15)
    receiving_currency: str = Field(min_length=1, max_length=64)
    payment_currency: str = Field(min_length=1, max_length=64)
    format: str = Field(min_length=1, max_length=64)


class MinuteBatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sequence: StrictInt = Field(ge=0)
    minute: datetime
    events: list[Transaction] = Field(min_length=1, max_length=5000)

    @field_validator('minute')
    @classmethod
    def exact_minute(cls, value):
        if value.tzinfo is not None or value.second or value.microsecond:
            raise ValueError('Use the dataset local clock at an exact minute, without timezone')
        return value

    @model_validator(mode='after')
    def unique_events(self):
        if len({x.event_id for x in self.events}) != len(self.events):
            raise ValueError('Duplicate event ID within batch')
        return self

    def fingerprint(self):
        value = self.model_dump(mode='json')
        value['events'] = sorted(value['events'], key=lambda e: e['event_id'])
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def frame(self):
        frame = pd.DataFrame([x.model_dump() for x in self.events]).sort_values('event_id').reset_index(drop=True)
        frame['timestamp'] = pd.Timestamp(self.minute)
        return frame


def empty_account():
    return {'out': 0, 'in': 0, 'last_out': -1, 'last_in': -1, 'neighbors': []}


def minute_number(value):
    return int(np.datetime64(value, 'm').astype(np.int64))


def online_features(frame, accounts):
    features = transaction_features(frame)
    now = minute_number(frame.timestamp.iloc[0])
    history = []
    for event in frame.itertuples():
        src, dst = accounts[event.source], accounts[event.target]
        history.append([np.log1p(src['out']), np.log1p(dst['in']),
                        -1 if src['last_out'] < 0 else now - src['last_out'],
                        -1 if dst['last_in'] < 0 else now - dst['last_in']])
    for name, values in zip(HISTORY, np.asarray(history, dtype=np.float32).T):
        features[name] = values
    return features


def graph_features(features, preprocessing):
    features = features.copy()
    for name in ['source_age_minutes', 'target_age_minutes']:
        features[name] = np.sign(features[name]) * np.log1p(np.abs(features[name]))
    return preprocessing.transform(features).astype(np.float32)


def graph_batch(frame, query, accounts, neighbors):
    now = minute_number(frame.timestamp.iloc[0])
    vectors = list(query)
    sample = np.full((len(frame), 2, neighbors), -1, dtype=np.int64)
    ages = np.zeros(sample.shape, dtype=np.float32)
    directions = np.zeros(sample.shape, dtype=np.float32)
    for row, event in enumerate(frame.itertuples()):
        for side, owner in enumerate([event.source, event.target]):
            for j, edge in enumerate(accounts[owner]['neighbors'][:neighbors]):
                if edge['minute'] >= now:
                    raise ValueError('Noncausal saved graph state')
                sample[row, side, j] = len(vectors)
                vectors.append(edge['x'])
                ages[row, side, j] = now - edge['minute']
                directions[row, side, j] = edge['direction']
    return np.asarray(vectors, dtype=np.float32), (sample, ages, directions)


def advance_accounts(frame, vectors, accounts, neighbors=32):
    now = minute_number(frame.timestamp.iloc[0])
    columns = ['source', 'target', 'paid', 'received', *CATEGORIES]
    priorities = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    for row, event in enumerate(frame.itertuples()):
        src, dst = accounts[event.source], accounts[event.target]
        src['out'] += 1
        dst['in'] += 1
        src['last_out'], dst['last_in'] = now, now
        edge = {'event_id': int(event.event_id), 'minute': now, 'priority': int(priorities[row]),
                'x': vectors[row].tolist(), 'direction': 1}
        src['neighbors'].append(edge)
        if event.target != event.source:
            dst['neighbors'].append({**edge, 'direction': -1})
    for state in accounts.values():
        state['neighbors'] = sorted(state['neighbors'],
                                    key=lambda x: (x['minute'], x['priority'], x['event_id']), reverse=True)[:neighbors]


class Models:
    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, expected in manifest['sha256'].items():
            if Path(name).name != name or digest(directory / name) != expected:
                raise ValueError('Model bundle digest mismatch')
        self.fingerprint = digest(directory / 'manifest.json')
        self.first_minute = datetime.fromisoformat(manifest['first_minute'])
        self.end_minute = datetime.fromisoformat(manifest['end_minute'])
        torch.set_num_threads(4)
        self.tree = XGBClassifier()
        self.tree.load_model(directory / 'history_xgboost.ubj')
        self.tree.set_params(n_jobs=4)
        self.tree_preprocessing = joblib.load(directory / 'history_xgboost.joblib')
        self.graph_preprocessing = joblib.load(directory / 'preprocessing.joblib')
        state = torch.load(directory / 'neighbors-32-latest.pt', map_location='cpu', weights_only=True)
        self.graph = TemporalGraphClassifier(state['features'], state['config']['hidden'], state['neighbors'])
        self.graph.load_state_dict(state['state_dict'])
        self.graph.eval()
        self.calibrators = {}
        for name in ['history_xgboost', 'temporal_attention_32']:
            policy = json.loads((directory / (name + '-policy.json')).read_text())
            self.calibrators[name] = MonotoneCalibration(**policy['calibrator'])

    def score(self, frame, accounts):
        features = online_features(frame, accounts)
        tree = self.tree.predict_proba(self.tree_preprocessing.transform(features).astype(np.float32))[:, 1]
        query = graph_features(features, self.graph_preprocessing)
        vectors, sample = graph_batch(frame, query, accounts, self.graph.neighbors)
        with torch.inference_mode():
            graph = self.graph(torch.from_numpy(vectors), torch.arange(len(frame)), sample).sigmoid().numpy()
        scores = {'history_xgboost': tree, 'temporal_attention_32': graph}
        calibrated = {name: self.calibrators[name].transform(values, frame.timestamp) for name, values in scores.items()}
        result = [{'event_id': int(event_id), **{name: float(values[i]) for name, values in scores.items()},
                   **{name + '_calibrated': float(values[i]) for name, values in calibrated.items()}}
                  for i, event_id in enumerate(frame.event_id)]
        return result, query
