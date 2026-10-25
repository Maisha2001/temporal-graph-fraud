# Evaluation protocol

The configuration was fixed before model fitting. The baseline uses all eligible
rows, without balancing or selecting validation examples by label.

## Chronology and labels

The model is fitted as of September 6, 2022 at 00:00 simulator time. A training
label is available only when the event timestamp plus 24 hours is **strictly
earlier** than that instant. The delay is an explicit experimental assumption,
not an observed property of the publisher's labels. September 5 transactions can
contribute unlabeled graph history but their labels cannot train the model.

September 6–7 is development validation. September 8 onward is reserved for a
later frozen final evaluation. The development loader scans timestamps to route
rows but never converts, summarizes or scores reserved labels. Validation may
inform later development; it must never be reported as untouched final testing.

## Accounts and graph history

Accounts with `int.from_bytes(SHA256(bank:account)[:8], 'big') % 10 == 0` form a
fixed account holdout. Before the fit instant, every edge touching such an account
is removed from both model training and graph history. These accounts first enter
the replay during validation. Report both cold edges (at least one held-out
endpoint) and warm edges. A cold edge can connect to a known endpoint, and later
cold edges may have prior validation history; this is not a claim that both
endpoints remain perpetually unseen.

Each timestamp group is scored before any edge in that group updates state.
History features are sender outgoing count, receiver incoming count, and time
since each endpoint's previous transaction in that role. They never use labels,
future neighbors, future amounts, graph-wide degrees or same-minute ordering.
Validation edges update unlabeled state after their timestamp group, as they
would in a chronological replay. Preprocessing fits only on eligible training
rows and ignores unseen categories at prediction time.

Transaction inputs are log amounts, currencies, payment format, self-transfer,
currency-change indicator and cyclic hour. The task assumes the posted
transaction's two amounts and currencies are already observable; it does not
claim pre-authorization detection. Historical amounts are not summed across
different currencies.

## Fixed models and metrics

Compare the training class prior, an XGBoost model using transaction inputs, and
the same model with four causal history inputs. Both tree models use 150 trees,
depth 4, learning rate 0.08, histogram training with 128 bins, seed 42 and four CPU
threads. Positive weight is the square root of the training negative-to-positive
ratio. There is no early stopping, validation-driven parameter search or test
tuning in this baseline. Hyperparameters are in `configs/ibm-hi-small.json`.

Average precision is the non-interpolated summary from scikit-learn, often
reported as PR-AUC; it is not trapezoidal PR area. Also report Brier score and
recall/precision in the top floor(1% × number of events). Ties use the original
row ID, independently of labels. The capacity metric ranks a whole evaluation
partition retrospectively; it is not an online threshold, daily workload promise
or calibrated alert policy. Group-specific metrics give each group its own 1%
budget. Zero-positive groups have null average precision and recall.

## Limits and reproduction

Simulated laundering patterns and the shared generator family can make learning
easier than real use. The small dataset, short time span, label-delay assumption,
unweighted point estimates and retrospective alert ranking limit the conclusions.
No confidence intervals, calibrated thresholds or neural graph comparison are
claimed here. Those require separate measured experiments. The final holdout
must preserve all eligible rows and failures, with its protocol frozen before
scoring.

`fraud-baseline` recreates membership, predictions and both saved tree models.
The report records package versions, runtimes, peak process memory, configuration
and artifact hashes. CPU/OS/library differences can change timings and potentially
floating-point results; exact hashes describe the recorded run, not a universal
bitwise guarantee.
