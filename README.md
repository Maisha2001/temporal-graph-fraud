# Temporal Graph Fraud Detection

A reproducible study and local investigation application for IBM's independently
generated synthetic AML transactions. It combines causal transaction histories,
one-hop temporal graph attention, delayed labels, calibration, hourly alert
budgets, PostgreSQL state and Kafka replay.

**Version 1.0.0:** [release and exact model weights](https://github.com/Maisha2001/temporal-graph-fraud/releases/tag/v1.0.0).
[Transaction Atlas setup](docs/workbench.md) opens the investigation demo locally.
[Reproduction](docs/reproduction.md) covers training, frozen evaluation, model
loading, local MLflow and the installed wheel.

## Final temporal evaluation

All 1,346,673 September 8-18 transactions were retained, including 2,150 positive
simulator labels and the sparse final days. Models and policies were frozen
before final labels were opened. AP is average precision; capacity recall below
is a global ranking diagnostic with an analytical expectation inside tied scores.

| Model | AP | Recall at 1% capacity |
|---|---:|---:|
| Transaction XGBoost | 0.09098 | 44.74% |
| History XGBoost | **0.17657** | 53.58% |
| History MLP | 0.09489 | 44.65% |
| Temporal graph, 8 neighbors | 0.10340 | 55.49% |
| Temporal graph, 32 neighbors | 0.14363 | **56.23%** |

The history tree has higher AP; graph32 has higher global capacity recall.
The separate frozen **closed-hour operational policies** recover 34.00% / 33.81%
of positives for tree / graph, with only 5.72% / 5.58% precision.
Most alerts are false positives. Calibration improves Brier scores, but does not
make these probabilities reliable for another population.

See [complete final results](results/heldout-v1/README.md),
[model limitations](docs/model-card.md), and the
[frozen protocol](results/heldout-v1/protocol.json).
The sparse tail has very different prevalence. These are synthetic laundering
labels, not evidence of effectiveness on real financial transactions.

## Try the workbench

Use Python 3.13 in a separate environment:

~~~sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements/locked.txt -r requirements/service.txt
python -m pip install --no-deps -e .
~~~

Set FRAUD_CASES=examples/investigation in your shell, then:

~~~sh
python -m uvicorn temporal_fraud.investigation:create_app --factory --host 127.0.0.1 --port 8035
~~~

Open http://127.0.0.1:8035. The included three-case example is authored UI data,
clearly labeled as such. The verified IBM development replay contains 93 ranked
cases from 9,567 scored events, with 1,067 strictly earlier context interactions.
[Workbench instructions](docs/workbench.md) explain reproducing that export.
Case replay reads saved scores; it does not run inference again.

## Train, serve and verify

- [Data provenance and licensing](docs/data.md): 5,078,345 IBM HI-Small transactions,
  pinned download bytes, chronology and independent source.
- [Temporal models](docs/temporal-model.md): full eligible training, fixed three
  epochs, 8/32 recent interactions per endpoint, checkpointed comparisons.
- [Delay, calibration and drift](docs/risk-policy.md): 0/24/48-hour sensitivity,
  matured calibration labels, graph interventions and descriptive shift.
- [PostgreSQL / HTTP / Kafka setup](docs/serving.md): bounded complete-minute
  ingestion, atomic graph state and scores, idempotent retries and recovery.
- [Service measurements](results/service-v1/README.md): 582 ms serial batch p95;
  two concurrent local streams achieved 652 events/s. This excludes aggregation
  and later alert decisions, and does not establish distributed capacity.
- [End-to-end reproduction and tests](docs/reproduction.md): fresh environment,
  exact model loading, browser, live service, MLflow and evidence verification.

The original [baseline tie audit](results/capacity-tie-audit-v1/README.md),
[graph comparison](results/temporal-v1/README.md) and
[robustness study](results/robustness-v1/README.md) remain unchanged development
evidence. Their historical unscored-holdout flags describe those earlier stages.
CI checks authored examples, real local service integration and committed
evidence; it does not retrain the full dataset.

Code is MIT licensed. IBM data remains CDLA-Sharing-1.0. This repository contains
aggregate results and authored example cases, not IBM transaction tables.
Model release files are computational outputs; the code license does not
relicense IBM data. Load only trusted, checksum-verified model artifacts.
