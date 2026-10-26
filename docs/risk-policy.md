# Delayed feedback, calibration and alert capacity

`configs/robustness-v1.json` fixes the experiment before new outcomes are read.
All analysis remains inside the already-used development period. September 8
onward is still excluded by the loader and reserved for the final evaluation.
Nothing in this study revises the earlier models, reports or saved predictions.

## Label availability sensitivity

The original model fit instant remains September 6 at midnight. Labels become
eligible only when event time plus delay is strictly earlier than that instant.
New 0-hour and 48-hour arms use the original fixed tree settings and the graph's
three epochs, 32-neighbor budget and seed. They use all eligible labels, fit their
own preprocessing and recompute the training class weight. The 24-hour arm reuses
the frozen earlier models and predictions without retraining. Validation events,
account holdout and causal graph replay are identical across delay settings.
This measures sensitivity to fewer/newer labels; it does not separate data volume,
label recency, changed preprocessing and optimization effects.

## Calibration chronology

The deployed model weights are frozen at September 6 midnight. At September 7
noon, only September 6 predictions whose labels have matured under the 24-hour
rule enter calibration: event time must be strictly before September 6 noon.
These predictions are outside model training. A monotone sigmoid maps clipped
raw logits to probabilities, fitting unweighted log loss with nonnegative slope.
At least 20 mature positives and both classes are required; there is no tuning
search or class reweighting in calibration. The original model is not refit.

Evaluation covers September 7 noon to September 8 midnight, using a disjoint,
later subset of development events. Labels from those events cannot affect the
calibrator, threshold or decisions. This is a development analysis, not a new
independent test set: the full two-day development results were previously seen.
The calibrated policy is a candidate for later evaluation, not a deployment claim.

Brier score, unweighted log loss and reliability bins are all reported. Bin
boundaries come from calibration predictions, not evaluation labels. Quantile
bins can merge under tied scores; empty bins remain explicit. Brier and log loss
measure more than calibration alone, so use the reliability table as well.
See the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
for these distinctions. Rare positives, shared accounts and a short evaluation
period limit confidence in any improvement.

## Capacity policy

A threshold is frozen at the top 1% boundary of calibrated reference scores.
A threshold alone cannot guarantee future capacity under distribution changes,
so the report shows its uncapped alert fraction as well as capped decisions.

Events are ranked within a *closed simulator hour*. Only threshold-eligible
transactions can alert, and at most floor(0.01 times that hour's event count) are
selected. Ties use SHA-256(seed:event ID), which is independent of labels and
input order, rather than direct source-row ordering. Decisions occur when the
hour closes and may wait up to one hour. There is no claim of instantaneous
streaming alerts. Unused hourly slots are not carried forward, and a high frozen
threshold can leave capacity unused. This policy can be used by the later replay
service without seeing future hours or labels.

The concrete policy reports actual selected positives. Earlier ranking metrics
continue to use analytical expected yield within score ties; those are distinct
from realized decisions. Calibration can create ties through clipping or a zero
slope. The policy remains bounded even if every score is identical.

## Graph interventions and drift

The fixed 32-neighbor checkpoint is scored with no neighbor messages and with
edge elapsed-time encodings zeroed. The latter leaves timestamps for causal
selection, historical feature gaps and direction intact. Neither intervention
retrains weights. Distribution mismatch means these are sensitivity diagnostics,
not isolated causal estimates of the value of graph structure or time. The
previous retrained 8/32-neighbor and history-MLP comparisons remain available.

September 6 defines the reference for descriptive feature/score shift on
September 7. Population stability indices use reference quantile bins and a
0.5-count pseudocount. Constant references use below/equal/above bins. Daily
performance is reported separately for all events and cold/warm accounts.
No universal alarm cutoff or statistical significance is claimed, and two days
cannot establish long-term production drift. The dataset is independently
generated synthetic AML data, not real customer investigations.

## Reproduce

After reproducing the baseline and graph runs using the README:

```sh
fraud-risk-study --data data/HI-Small_Trans.csv --baseline artifacts/baseline --graph artifacts/temporal-v1 --output artifacts/robustness-v1 --device cuda
```

Use `--device cpu` when needed. The command writes new per-delay runs, calibration
JSON, capped alert decisions, intervention predictions, drift reports and artifact
hashes. Earlier artifact hashes are checked before and after. A new output folder
is required unless explicitly resuming. Completed runs verify their saved hashes
and return without modifying evidence. Interrupted runs resume the current study;
no final held-out labels are converted or scored. Load only your own trusted saved model artifacts.
