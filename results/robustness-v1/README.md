# Delayed feedback and risk-policy study

This fixed development study preserves the earlier artifacts and leaves all 1,346,673 September 8 onward events unscored. New 0/48-hour training arms, chronological calibration, capped hourly alerts, graph interventions and daily drift checks use development data only.

## Label-delay sensitivity

Recall below uses the full-development 1% ranking capacity. The 24-hour arm reuses frozen predictions. Zero delay is an idealized availability scenario. More/newer labels, preprocessing and optimization effects are confounded.

| Delay | Training events / positives | History tree AP / recall | Graph32 AP / recall |
| --- | ---: | ---: | ---: |
| 0 h | 2,230,428 / 1,559 | 0.19236 / 65.76% | 0.14468 / 59.92% |
| 24 h | 1,851,794 / 1,187 | 0.17228 / 63.62% | 0.10365 / 56.91% |
| 48 h | 1,688,786 / 862 | 0.13603 / 58.66% | 0.07603 / 54.57% |

## Chronological calibration

Calibration fits 248,642 September 6 predictions with 242 mature positives at September 7 noon. The latest feedback is available at 11:59, strictly before fitting. Evaluation covers 233,800 later September 7 events with 267 positives, disjoint from training and calibration. These are previously used development days, not an independent final test.

| Model | Brier raw / calibrated | Log loss raw / calibrated | Reliability ECE raw / calibrated |
| --- | ---: | ---: | ---: |
| history_xgboost | 0.005913 / 0.001009 | 0.028461 / 0.005069 | 0.022266 / 0.000145 |
| temporal_attention_32 | 0.005011 / 0.001090 | 0.021494 / 0.005692 | 0.015725 / 0.000369 |

Brier, log loss and reference-binned reliability improve in this slice. ECE is bin-dependent, and 267 positives over half a day cannot establish stable deployment calibration. The graph remains weaker than the history tree. All bins and counts, including empty bins, are retained.

| Model | Alerts / slots | True alerts | Recall / precision | Threshold-only volume | Capped volume |
| --- | ---: | ---: | ---: | ---: | ---: |
| history_xgboost | 2,119 / 2,332 | 169 | 63.30% / 7.98% | 1.07% | 0.91% |
| temporal_attention_32 | 2,332 / 2,332 | 143 | 53.56% / 6.13% | 1.55% | 1.00% |

Decisions occur at the end of each simulator hour, with at most floor(1% of hourly traffic) alerts and no slot carryover. The 12 hours provide 2,332 total slots. A threshold alone exceeds 1% volume for both models; the explicit cap prevents overflow. The tree leaves some slots unused. These are actual hash-tie selections, distinct from analytical expected tie yield in ranking metrics. Most alerts remain false positives.

## Graph interventions and drift

| Fixed graph32 checkpoint | AP | Recall at 1% capacity |
| --- | ---: | ---: |
| Original | 0.10365 | 56.91% |
| no_neighbors | 0.04983 | 39.20% |
| no_edge_time | 0.10523 | 57.00% |

Removing messages degrades AP. Zeroing edge elapsed-time encodings slightly improves AP here, so their benefit is not established. Historical gaps and causal selection remain. Weights are unchanged; distribution mismatch prevents an isolated causal effect claim.

September 6 to 7 descriptive PSI: standardized log amount 0.00000234, source activity count 0.08815, source age 0.05306, tree score 0.01214 and graph score 0.00243. Increasing account history can explain movement without an adverse regime change. The report retains daily cold/warm prevalence and performance. No universal alarm threshold, significance or long-term drift diagnosis is supported.

## Verification and reproduction

The CUDA study took 263.29 seconds. Peak process memory was 5,273,214,976 bytes (4.91 GiB), and Torch GPU allocation 892,139,520 bytes (0.83 GiB), excluding driver/other applications. These are local experiment timings, not service benchmarks. New graph arms use three fixed epochs and 32 neighbors; tree settings remain fixed.

The independent audit verifies 44 new artifacts and 27 unchanged frozen artifacts, recomputes 18 metric groups, and checks all 24 model/hour policies with an independent sort/hash implementation. It checks strict label eligibility, disjoint calibration/evaluation IDs, float64 sigmoid reproduction, reference-only thresholds and selected positives. Transaction-level predictions and model checkpoints stay local.

See [the protocol and command](../../docs/risk-policy.md). Audit trusted local artifacts without training:

```sh
python scripts/audit_risk_study.py --run artifacts/robustness-v1 --baseline artifacts/baseline --graph artifacts/temporal-v1 --output artifacts/risk-audit.json
```

Public aggregate JSON uses LF line endings and is hash-checked against working files, the staged index or committed Git blobs. The independent synthetic AML data does not establish real financial deployment validity. Original baseline and temporal-model evidence remains unchanged.
