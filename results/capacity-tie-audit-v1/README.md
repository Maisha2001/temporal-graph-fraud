# Capacity metric audit

These are the current baseline metrics. The original first-commit report remains
unchanged in `results/baseline-v1/` for comparison; its row-order tie policy is
superseded for capacity metrics.

Review found that selecting equal scores by original CSV row number gave the
constant-score prior 40.18% recall in a 1% alert budget. This is an ordering
artifact, not evidence of discrimination. Source-row IDs are not model inputs,
but they were improperly allowed to decide tied rankings in the metric.

The corrected metric selects all events above the boundary score, then computes
the analytical expected positive count of a uniform random draw for remaining
slots among boundary ties. It uses labels only to evaluate that expectation,
never to choose alerts. It also reports minimum/maximum possible true positives,
boundary group size and remaining slots. Expected true-alert counts can be
fractional. This is an offline ranking measure, not an implemented online policy.

No model was retrained. All seven model, preprocessor, membership and saved
prediction files match their original hashes. All original average-precision
values remain exactly unchanged. September 8 onward remains unscored. An
independent grouping calculation reproduced every corrected capacity result.

Validation: 964,840 transactions, 1,028 positives; budget: 9,648 alerts.

| Model | Average precision | Expected recall at 1% | Expected precision at 1% | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Constant training prior | 0.00107 | 1.00% | 0.11% | 0.001065 |
| Transaction attributes | 0.05950 | 43.09% | 4.59% | 0.004580 |
| Attributes + past activity | 0.17228 | 63.62% | 6.78% | 0.005858 |

The history model ranks better than transaction attributes on this development
partition, but most alerts remain false positives and probability calibration is
poor. This simulated-data result is not a claim of real-world detection quality.
The constant prior now has the expected approximately 1% recall, rather than
benefiting from the publisher's source ordering.

All 21 tests pass. Additional regression tests cover source-ID invariance and
partially selected boundary ties. [report.json](report.json) preserves full group
metrics, original training settings, runtimes and the correction explanation.
The timing/memory fields refer to the unchanged original training run.
