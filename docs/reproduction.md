# Reproduction and release setup

Use Python 3.13 in an isolated environment. The measured environment is Windows,
Python 3.13.13, Torch 2.11.0+cu128, PyG 2.8.1 and RTX 5070. CPU inference is also
tested. Keep several GB of memory and disk available; full training/replay uses
more than the small UI example. No paid APIs or cloud tracking are needed.

## Install the package and exact weights

Clone this repository at tag v1.0.0 for scripts, requirements and authored examples.
Create/activate a fresh environment, then:

~~~sh
python -m pip install -r requirements/locked.txt -r requirements/service.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/graph.txt
~~~

Download temporal_graph_fraud-1.0.0-py3-none-any.whl, fraud-models-v1.0.0.zip
and SHA256SUMS.txt from the
[v1.0.0 release](https://github.com/Maisha2001/temporal-graph-fraud/releases/tag/v1.0.0).
Compare each SHA256 to the published manifest before loading model files.
Install the wheel and extract the ZIP into an empty directory:

~~~sh
python -m pip install --no-deps temporal_graph_fraud-1.0.0-py3-none-any.whl
python -m zipfile -e fraud-models-v1.0.0.zip artifacts/released
python scripts/score_example.py --models artifacts/released/models
~~~

The scoring example uses an authored transaction with empty history and exact
tree/graph/calibrator files. Its successful execution verifies installed model
access, not predictive accuracy. release-models.json covers 13 files plus its
own separately recorded hash; manifest.json retains the six original service
model hashes. Only load trusted joblib/pickle artifacts. The wheel includes SQL
schema and all web assets. Start the [authored workbench](workbench.md) without
data or a database. Editable development uses pip install --no-deps -e . instead.

## Reproduce experiments in new output directories

Download/hash-check the independent data and train the original baselines:

~~~sh
fraud-baseline --download --output artifacts/baseline
fraud-graph --data data/HI-Small_Trans.csv --output artifacts/temporal-v1 --device cpu
fraud-risk-study --data data/HI-Small_Trans.csv --baseline artifacts/baseline --graph artifacts/temporal-v1 --output artifacts/robustness-v1 --device cpu
~~~

Use the pinned CUDA wheel index and --device cuda for a supported NVIDIA
machine. Do not overwrite recorded evidence. Graph resume restores the last
completed epoch; a fresh independent retraining can differ numerically.
The downloadable weights preserve the exact measured models.

To run a new frozen final evaluation, execute the three phases in order with the
same arguments, replacing <phase> with freeze, predict, then score:

~~~sh
python -m temporal_fraud.release_eval <phase> --data data/HI-Small_Trans.csv --baseline artifacts/baseline --graph artifacts/temporal-v1 --risk artifacts/robustness-v1 --output artifacts/heldout --device cpu
python scripts/audit_release.py --baseline artifacts/baseline --graph artifacts/temporal-v1 --risk artifacts/robustness-v1 --run artifacts/heldout --output artifacts/heldout-audit.json
~~~

Freeze pins source bytes, all source artifacts, arms and policies. Prediction
does not read the target column and saves every test score before scoring reads
labels. Existing prediction outputs refuse replacement. The recorded run is
complete; these commands are for a new independent reproduction. Do not tune
parameters against its outcomes.

Public source provenance preserves the original Windows byte hashes and
separately records canonical Git LF hashes. Only newline normalization differs.
The final report completed_at_utc field was captured when metric computation
started; verification completion is recorded separately in release evidence.
The embedded historical data configuration's development-only description is
retained unchanged; the final protocol and report specify the final test.

## Local tracking and end-to-end service

~~~sh
python -m pip install -r requirements/release.txt
python -m temporal_fraud.tracking --run-directory artifacts/heldout --models artifacts/released/models --destination artifacts/mlruns --output artifacts/tracking.json
~~~

This writes a local MLflow file store, logs 32 metrics and the model/evaluation
artifacts, then reads back the metrics and artifact hashes. It does not train
or contact a remote tracking service. The recorded run verified 14 model/package
artifacts and the report. MLflow-skinny supplies the local client/store; a hosted
MLflow UI is not required.

Follow [serving.md](serving.md), using artifacts/temporal-v1 and
artifacts/robustness-v1 for graph/risk paths consistently. Prepare a new
service snapshot, start only that Compose project's PostgreSQL/Kafka services,
run the benchmark and audit, then export [investigation cases](workbench.md).
The benchmark genuinely restarts its owned API process and tests SQL rollback,
Kafka redelivery and concurrent requests. Stop the project services afterward
without deleting volumes.

## Tests and evidence

~~~sh
python -m pip install -r requirements/release.txt
python -m playwright install chromium
python -m pytest -q
python scripts/verify_evidence.py
python scripts/verify_evidence.py --revision HEAD
~~~

Set FRAUD_BROWSER_TESTS=1 to include browser tests. Set FRAUD_TEST_DSN and
FRAUD_TEST_KAFKA=127.0.0.1:19094 with local services running to enable integration
tests. Otherwise those tests explicitly skip. CI runs Windows and Ubuntu
unit/evidence checks, a real Kafka/PostgreSQL job and a Chromium/MLflow job.
It never downloads the full data or reruns the benchmark. The independent final
audit covers 112 metric groups, 434 hourly policies, 128 exhaustive neighborhood
queries and 256 CPU checkpoint predictions per neural arm.
