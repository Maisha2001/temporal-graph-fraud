# Temporal Graph Fraud Detection

Chronological transaction classification on IBM's independently generated,
synthetic anti-money-laundering data. The working experiments compare transaction and history XGBoost baselines,
a history MLP, and a temporal graph attention model with two neighborhood budgets.
They include a verified download, account holdouts, delayed-label training,
checkpoint recovery, saved predictions and leakage regression tests.

This is an experimental implementation. Streaming service integration and the
analyst application are not implemented yet. Results concern
simulated laundering labels; they do not establish real-world fraud detection.

## Run

Use Python 3.13 and a separate environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements/locked.txt
python -m pip install --no-deps -e .
python -m pytest -q
fraud-baseline --download --output artifacts/baseline
python scripts/verify_evidence.py
```

The public download is about 476 MB. Its SHA-256 is checked before use. If the
download endpoint is unavailable, obtain `HI-Small_Trans.csv` from the
[original Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
then pass `--data /path/to/HI-Small_Trans.csv`. Different bytes are rejected.
Allow several GB of RAM and local disk for data and generated artifacts. The
recorded run uses four CPU threads; it requires no GPU or paid API.

The command writes a report, split membership, validation predictions, XGBoost
models and fitted preprocessors into the output directory. Treat joblib files as
trusted local artifacts only; do not load unknown files. Data and generated
artifacts are excluded from Git.

## Temporal graph experiments

After the baseline setup, install the graph dependencies in the same separate environment:

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/graph.txt
python -m pip install --no-deps -e .
python -m pytest -q
fraud-graph --data data/HI-Small_Trans.csv --output artifacts/temporal-v1
```

For the measured NVIDIA run, install the same Torch version using
`https://download.pytorch.org/whl/cu128` instead of the CPU index, then add
`--device cuda`. No compiled PyG extensions are required. The command refuses an
existing output directory unless `--resume` is supplied. Resume verifies cached
inputs and restores model/optimizer state from the last completed epoch.
See [the model and reproduction notes](docs/temporal-model.md) and
[graph comparison results](results/temporal-v1/README.md).

## Delayed feedback and alert policy

The [risk study](docs/risk-policy.md) compares label delays, fits calibration only
from matured out-of-training feedback, and evaluates a frozen threshold with a
hard hourly capacity cap on later development events. It also measures graph
interventions and daily feature/score shift. Reproduce it with `fraud-risk-study`
after the baseline and graph runs. [Measured results](results/robustness-v1/README.md)
include limitations and both uncapped threshold volume and actual capped decisions.

## Evaluation

See [the frozen protocol](docs/evaluation.md), [data provenance](docs/data.md)
and [measured results and tie audit](results/capacity-tie-audit-v1/README.md). September 8 onward is
reserved and has not been scored. The tie audit supersedes source-row ordering for capacity metrics in the original
protocol; original results are retained for comparison. The committed evidence is a development
validation result, not a final held-out benchmark. CI tests the pipeline with
small authored examples and verifies the recorded evidence hashes; it does not
download or retrain on the full dataset.

Code is MIT licensed. IBM data remains under CDLA-Sharing-1.0; no raw transactions
or derived transaction tables are included in this repository.

## Stateful local serving

[HTTP/Kafka setup](docs/serving.md) replays complete transaction minutes through
frozen tree and graph models with PostgreSQL state and retry-safe receipts.
[Service measurements](results/service-v1/README.md) cover prediction parity,
process restart, redelivery, concurrent replay and latency. This remains a
bounded synthetic development application; the final holdout is unscored.
