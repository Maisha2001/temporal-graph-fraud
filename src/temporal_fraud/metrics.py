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
    chosen = np.lexsort((event_ids, -scores))[:budget]
    positive = int(labels.sum())
    true_alerts = int(labels[chosen].sum())
    return {'events': len(labels), 'positives': positive, 'prevalence': float(labels.mean()),
            'average_precision': float(average_precision_score(labels, scores)) if positive else None,
            'brier': float(brier_score_loss(labels, scores)), 'alert_fraction': fraction,
            'alerts': budget, 'true_alerts': true_alerts,
            'recall_at_capacity': true_alerts / positive if positive else None,
            'precision_at_capacity': true_alerts / budget if budget else None}
