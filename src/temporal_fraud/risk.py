from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit


def available_labels(timestamps, as_of, delay_hours):
    if not np.isfinite(delay_hours) or delay_hours < 0:
        raise ValueError('Label delay must be finite and nonnegative')
    times = pd.DatetimeIndex(timestamps)
    if times.hasnans:
        raise ValueError('Missing feedback timestamp')
    return np.asarray(times + pd.Timedelta(hours=delay_hours) < pd.Timestamp(as_of))


def probabilities(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Finite probabilities in [0,1] required')
    return values


@dataclass
class MonotoneCalibration:
    slope: float
    intercept: float
    fitted_at: str
    delay_hours: float
    model_fit_at: str
    examples: int
    positives: int
    latest_label_available_at: str

    @classmethod
    def fit(cls, predictions, as_of, delay_hours, model_fit_at, min_positives=20):
        if not predictions.event_id.is_unique:
            raise ValueError('Duplicate feedback event IDs')
        eligible = available_labels(predictions.timestamp, as_of, delay_hours)
        eligible &= (predictions.timestamp >= pd.Timestamp(model_fit_at)).to_numpy()
        selected = predictions.loc[eligible]
        y = selected.label.to_numpy(dtype=float)
        if not np.isin(y, [0, 1]).all() or y.sum() < min_positives or y.sum() == len(y):
            raise ValueError('Insufficient mature positive/negative calibration labels')
        x = logit(np.clip(probabilities(selected.score), 1e-7, 1 - 1e-7))
        def objective(parameters):
            z = parameters[0] * x + parameters[1]
            error = expit(z) - y
            return float(np.mean(np.logaddexp(0, z) - y * z)), np.array([np.mean(error * x), np.mean(error)])
        result = minimize(objective, [1.0, float(logit(y.mean()) - x.mean())], jac=True,
                          method='L-BFGS-B', bounds=[(0, None), (None, None)],
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError('Calibration optimization failed: ' + result.message)
        latest = selected.timestamp.max() + pd.Timedelta(hours=delay_hours)
        return cls(float(result.x[0]), float(result.x[1]), str(pd.Timestamp(as_of)), float(delay_hours),
                   str(pd.Timestamp(model_fit_at)), len(y), int(y.sum()), str(latest))

    def transform(self, scores, timestamps):
        if len(scores) != len(timestamps):
            raise ValueError('Aligned scores and timestamps required')
        if np.any(pd.DatetimeIndex(timestamps) < pd.Timestamp(self.fitted_at)):
            raise ValueError('Calibrator cannot be applied before its fit instant')
        return self.apply(scores)

    def apply(self, scores):
        return expit(self.slope * logit(np.clip(probabilities(scores), 1e-7, 1 - 1e-7)) + self.intercept)

    def to_dict(self):
        return asdict(self)


def capacity_threshold(reference_scores, fraction):
    scores = probabilities(reference_scores)
    if not 0 < fraction <= 1 or not len(scores):
        raise ValueError('Nonempty reference and fraction in (0,1] required')
    budget = int(np.floor(len(scores) * fraction))
    return None if budget == 0 else float(np.partition(scores, len(scores) - budget)[len(scores) - budget])


def closed_hour_alerts(predictions, threshold, fraction, as_of, seed=42):
    if not 0 < fraction <= 1 or not predictions.event_id.is_unique:
        raise ValueError('Valid capacity and unique event IDs required')
    scores = probabilities(predictions.score)
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError('Invalid probability threshold')
    timestamps = pd.DatetimeIndex(predictions.timestamp)
    if timestamps.hasnans:
        raise ValueError('Missing alert timestamp')
    hours = timestamps.floor('h')
    if np.any(hours + pd.Timedelta(hours=1) > pd.Timestamp(as_of)):
        raise ValueError('Cannot decide an hour that has not closed')
    keys = np.array([int.from_bytes(hashlib.sha256(f'{seed}:{i}'.encode()).digest()[:8], 'big')
                     for i in predictions.event_id], dtype=np.uint64)
    selected = np.zeros(len(scores), dtype=bool)
    eligible = np.zeros(len(scores), dtype=bool) if threshold is None else scores >= threshold
    reports = []
    for hour in hours.unique().sort_values():
        rows = np.flatnonzero(hours == hour)
        budget = int(np.floor(len(rows) * fraction))
        candidates = rows[eligible[rows]]
        ranked = candidates[np.lexsort((keys[candidates], -scores[candidates]))]
        selected[ranked[:budget]] = True
        reports.append({'hour_start': str(hour), 'decision_at': str(hour + pd.Timedelta(hours=1)),
                        'events': len(rows), 'budget': budget, 'threshold_eligible': len(candidates),
                        'alerts': int(selected[rows].sum())})
    return selected, reports


def reliability(labels, scores, reference_scores, bins=10):
    y, scores, reference = np.asarray(labels, dtype=float), probabilities(scores), probabilities(reference_scores)
    if len(y) != len(scores) or not len(y) or not np.isin(y, [0, 1]).all() or not len(reference):
        raise ValueError('Nonempty aligned binary labels and reference scores required')
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    memberships = np.searchsorted(edges, scores, side='right')
    groups, ece = [], 0.0
    for i in range(len(edges) + 1):
        selected = memberships == i
        n = int(selected.sum())
        predicted = float(scores[selected].mean()) if n else None
        observed = float(y[selected].mean()) if n else None
        if n:
            ece += n / len(y) * abs(predicted - observed)
        groups.append({'bin': i, 'events': n, 'mean_probability': predicted, 'observed_fraction': observed})
    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    return {'brier': float(np.mean((scores - y) ** 2)),
            'log_loss': float(np.mean(-y * np.log(clipped) - (1-y) * np.log1p(-clipped))),
            'expected_calibration_error': float(ece), 'reference_quantile_boundaries': edges.tolist(), 'bins': groups}


def population_shift(reference, observed, bins=10):
    reference, observed = np.asarray(reference, dtype=float), np.asarray(observed, dtype=float)
    if not len(reference) or not len(observed) or not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError('Nonempty finite drift samples required')
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    count = len(edges) + 1
    ref = np.bincount(np.searchsorted(edges, reference, side='right'), minlength=count)
    obs = np.bincount(np.searchsorted(edges, observed, side='right'), minlength=count)
    if np.ptp(reference) == 0:
        value = reference[0]
        count = 3
        ref = np.array([0, len(reference), 0])
        obs = np.array([(observed < value).sum(), (observed == value).sum(), (observed > value).sum()])
    a, b = (ref + 0.5) / (len(reference) + count * 0.5), (obs + 0.5) / (len(observed) + count * 0.5)
    return {'population_stability_index': float(np.sum((b-a) * np.log(b/a))),
            'reference_quantile_boundaries': edges.tolist(), 'reference_counts': ref.tolist(),
            'observed_counts': obs.tolist(), 'reference_events': len(reference), 'observed_events': len(observed),
            'constant_reference_bins': 'below/equal/above' if np.ptp(reference) == 0 else None,
            'interpretation': 'descriptive binned shift, not a significance test or universal alarm threshold'}
