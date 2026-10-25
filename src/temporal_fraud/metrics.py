import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def evaluate(labels, scores, event_ids, fraction=0.01):
    labels, scores, event_ids = map(np.asarray, (labels, scores, event_ids))
    if not (len(labels) == len(scores) == len(event_ids)) or not len(labels):
        raise ValueError('Metric inputs must have the same nonzero length')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Expected binary labels and finite probabilities')
    if not 0 < fraction <= 1 or len(np.unique(event_ids)) != len(event_ids):
        raise ValueError('Invalid alert budget or duplicate event IDs')
    budget = int(np.floor(fraction * len(labels)))
    positive = int(labels.sum())
    expected, minimum, maximum, tied, slots = 0.0, 0, 0, 0, 0
    if budget:
        boundary = np.partition(scores, len(scores) - budget)[len(scores) - budget]
        above, at_boundary = scores > boundary, scores == boundary
        slots = budget - int(above.sum())
        tied = int(at_boundary.sum())
        above_positive, tied_positive = int(labels[above].sum()), int(labels[at_boundary].sum())
        # Expected yield of a uniform draw among boundary ties; row order never ranks alerts.
        expected = above_positive + slots * tied_positive / tied
        minimum = above_positive + max(0, slots - (tied - tied_positive))
        maximum = above_positive + min(slots, tied_positive)
    return {'events': len(labels), 'positives': positive, 'prevalence': float(labels.mean()),
            'average_precision': float(average_precision_score(labels, scores)) if positive else None,
            'brier': float(brier_score_loss(labels, scores)), 'alert_fraction': fraction,
            'alerts': budget, 'expected_true_alerts': expected,
            'true_alerts_min': minimum, 'true_alerts_max': maximum,
            'boundary_tied_events': tied, 'boundary_slots': slots,
            'tie_policy': 'analytical expectation under uniform random selection within boundary score ties',
            'recall_at_capacity': expected / positive if positive else None,
            'precision_at_capacity': expected / budget if budget else None}
