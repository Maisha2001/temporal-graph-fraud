# Development baseline results

Recorded October 25, 2026, using the pinned IBM HI-Small file and configuration.
The final September 8 onward holdout remains unscored.

Training contains 1,851,794 eligible transactions (1,187 positives); development
validation contains 964,840 transactions (1,028 positives, prevalence 0.1065%).
The replay excludes 536,404 pre-fit edges touching held-out accounts. There are
1,346,673 reserved test rows. No validation class balancing or downsampling is used.

| Model | Average precision | Recall at 1% capacity | Precision at 1% capacity | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Training prior | 0.00107 | 40.18% | 4.28% | 0.001065 |
| Transaction attributes | 0.05950 | 43.09% | 4.59% | 0.004580 |
| Attributes + past account activity | 0.17228 | 63.62% | 6.78% | 0.005858 |

Past account activity improves validation ranking over transaction attributes.
Nevertheless, 6.8% precision means most selected alerts are false positives. The
weighted history model has worse Brier score than the attributes-only model,
and both are worse than the class prior. Its probabilities are not calibrated.

The history model's cold-account slice has 208,312 events and 207 positives:
average precision 0.04565, recall 42.03% and precision 4.18% at its separate 1%
budget. The warm slice has 756,528 events and 821 positives: average precision
0.21114 and recall 68.94%. Cold accounts are materially harder. These point
estimates are development evidence, without uncertainty intervals or a claim
about real financial data.

The Windows/Python 3.13.13 run used four CPU threads and no GPU. Loading took
20.88 seconds; feature building, fitting and scoring took 29.99 seconds. The
transaction and history fits took 4.14 and 4.88 seconds respectively. Peak process
working set was 4,096,028,672 bytes (3.81 GiB). Downloading, package installation
and final artifact hashing are outside those timers.

All 19 regression and integration tests passed, including same-minute ordering,
future-event invariance, entity isolation, delayed-label boundaries, malformed
data, unseen categories, alert ties and saved-model prediction reproduction.
An independent artifact audit checked membership, training-label availability,
account holdouts, saved prediction order, ranked alert counts and all seven saved
artifact hashes. See [audit.json](audit.json) and the full [report.json](report.json).

The private generated artifacts can be recreated with the README command.
Only aggregate reports and hashes are distributed here. See the
[evaluation protocol](../../docs/evaluation.md) for assumptions and limitations.
