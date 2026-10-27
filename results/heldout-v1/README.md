# Frozen final temporal evaluation

This is the one final evaluation of the previously reserved September 8-18
period. All 1,346,673 events and 2,150 positive simulator labels were retained.
The [protocol](protocol.json) was frozen before prediction; label-free predictions
were hashed before a separate label-reading step. No model, threshold or
calibrator changed using final outcomes.

| Model | AP | Global 1% recall | Global 1% precision |
|---|---:|---:|---:|
| Constant prior | .00160 | 1.00% | .16% |
| Transaction XGBoost | .09098 | 44.74% | 7.14% |
| History XGBoost | .17657 | 53.58% | 8.55% |
| History MLP | .09489 | 44.65% | 7.13% |
| Graph 8 | .10340 | 55.49% | 8.86% |
| Graph 32 | .14363 | 56.23% | 8.98% |

Global capacity metrics use analytical expectation within boundary ties.
They are ranking diagnostics, not the realized operational policy.

Frozen threshold plus closed-hour floor(1% traffic) policies select 12,784 tree
alerts (731 positive; 34.00% recall, 5.72% precision) and 13,024 graph alerts
(727 positive; 33.81% recall, 5.58% precision). Unused slots do not carry and
decisions wait for hour close. Full hourly counts, cold/warm and daily results
are in [report.json](report.json).

The tree has higher AP; graph32 has higher global capacity recall. Neither
dominates every measure. Most operational alerts are false positives.
Tree Brier improves from .010531 to .001437 with the frozen calibrator;
graph32 from .006837 to .001493. This does not establish calibration elsewhere.

September 8-10 contain 1,345,565 events; the last eight days contain only 1,108.
Those sparse days have unusually high positive prevalence and unstable small
sample metrics. They remain included and separately reported. No real financial
validity, significance, repeated-seed stability or permanent cold-start claim.

[Independent audit](audit.json): 74 unchanged source artifacts, nine result
artifacts, 112 metric groups, 434 hourly policies, 128 exhaustive endpoint queries
and 256 CPU reload scores per neural arm. Maximum CPU/GPU difference 7.46e-8.
Independent float64 Brier differed from recorded float32 by at most 7.04e-9;
the auditor uses 1e-8 tolerance, with no result changes.

Prediction took 65.13 seconds on RTX 5070 with four CPU threads, peak process
5,909,512,192 bytes and Torch allocation 1,181,807,104 bytes. Scoring/audit and
driver/other applications are outside those measurements.
Historical development reports remain unchanged. See the
[model card](../../docs/model-card.md) for limitations and
[source provenance](source-provenance.json) for Windows/Git newline hashes.
