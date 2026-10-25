# Temporal Graph Fraud Detection

Chronological transaction classification on IBM's independently generated,
synthetic anti-money-laundering data. The current working baseline compares
transaction attributes with strictly prior account activity. It includes a
verified data download, account holdouts, delayed-label training, saved models,
and leakage regression tests.

This is an experimental foundation. Temporal neural modeling, streaming service
integration and the analyst application are not implemented yet. Results concern
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
