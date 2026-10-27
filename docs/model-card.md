# Model and evaluation card

## Intended use and data

Research and local investigation demonstrations on IBM HI-Small synthetic AML
data under CDLA-Sharing-1.0. The binary target is simulator laundering, not a
human investigation decision. No real customer data, banking integration or
deployment validity is claimed.

The pinned source contains 5,078,345 events. Fit occurs September 6 midnight with
24-hour label availability and fixed SHA256 account holdout: 1,851,794 eligible
training events. Development has 964,840 events / 1,028 positives. Calibration
uses 248,642 matured September 6 events outside training. September 7 afternoon
is a later development slice, not a fresh independent test.

## Models and selection

Transaction and history XGBoost, history MLP, and one-hop temporal attention with
8 or 32 recent interactions per endpoint. There are no account-ID embeddings or
recurrent TGN memory. Graph queries use strictly earlier minutes. Models train
all eligible events for three fixed neural epochs / 1,359 updates; development
does not select epochs. Only one training seed is measured.

Full-data final scores use the original saved weights, preprocessors, two
calibrators and two thresholds. Prediction reads exclude final labels; predictions
and hashes are saved before a separate scoring step opens them. No final outcome
changed training, exclusions, thresholds, calibration or checkpoint selection.
Earlier unlabeled test transactions may update history for later transactions.
Cold accounts were held out at fitting, but can accumulate later observable
history. Cold does not mean permanently unseen.

## Final results and limits

The [final report](../results/heldout-v1/report.json) includes all/cold/warm
metrics, each day and every hourly policy. History XGBoost AP is 0.17657 versus
graph32 0.14363. Graph32 global 1% capacity recall is 56.23% versus tree 53.58%.
These mixed results do not support unqualified graph superiority. In the
288,672-event fit-held-out account slice (404 positives), tree / graph32 AP is
0.16694 / 0.12779 and 1% capacity recall is 56.19% / 56.44%.

Frozen closed-hour decisions give tree 731 true positives / 12,784 alerts and
graph 727 / 13,024, or 34.00% / 33.81% recall and 5.72% / 5.58% precision.
These realized policies differ from a full-period ranking diagnostic: thresholds
were fixed on development, each hour has floor(1% traffic) capacity, and unused
slots do not carry. Decisions wait until hour close, up to an hour after arrival.

Calibration improves Brier from 0.010531 to 0.001437 for the tree and from
0.006837 to 0.001493 for graph32. Most alerts still are false positives.
No calibrated confidence interval, statistical significance or independent
replication across seeds is claimed.

All September 8-18 events were retained. September 8-10 have 482,773 / 654,467 /
208,325 events; the remaining eight days together have only 1,108, with very high
positive prevalence. Sparse-tail daily AP can approach one on tiny samples.
This simulator distribution shift makes aggregate transfer claims inappropriate.
Full daily results and counts are supplied rather than selecting favorable days.

The no-neighbor and no-time-encoding interventions are frozen inference
diagnostics, not retrained causal ablations. Removing time encoding slightly
improved development AP; the evidence does not establish benefit from that
encoding. Delay comparisons change eligible counts, recency and optimization
together. Descriptive PSI values are not universal drift alarms.

## Runtime and reproducibility

Measured graph training: 103.29 seconds, four CPU threads, 4.05 GB process peak
and 0.892 GB Torch GPU allocation. Final prediction: 65.13 seconds, 5.91 GB
process peak and 1.182 GB Torch allocation on an RTX 5070. These exclude other
applications/driver allocation; prediction timing excludes later scoring/audit.
CPU reload differences from saved GPU scores are below 7.46e-8 on the audit sample.
Bitwise CUDA retraining is not promised. Exact original weights are downloadable.

The service is bounded to declared account coverage and complete-minute
watermarks. Unknown accounts and late/conflicting batches reject. Omitted
events cannot be detected. Kafka is at-least-once with idempotent SQL effects,
not a distributed exactly-once transaction. Local load measurements do not
include minute aggregation or hourly alert decisions and do not show distributed
production capacity. The UI ranks saved development cases; graph links do not
establish wrongdoing or causality.
