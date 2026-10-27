# Local transaction replay service

The service scores **complete, ordered transaction minutes** with the frozen history
XGBoost and 32-neighbor graph models, plus the separately fitted calibrators.
It is a bounded synthetic-data replay, not a live banking integration.
The September 8 final holdout is rejected by the current model bundle.

## State and delivery contract

A stream starts from a hash-checked snapshot containing counts, last interaction
times and the most recent 32 incident feature vectors for every endpoint in its
replay scope. Unknown accounts are rejected rather than silently losing their
earlier history. The default preparation covers September 7, 12:00-12:30.

Each batch has an increasing sequence and a strictly later minute. Its producer
declares the minute complete; all events use state from earlier minutes.
The service cannot detect omitted events or a falsely declared watermark.
Empty gaps are allowed, late batches are rejected, and an accepted minute cannot
be extended. Use one ordered writer per stream. Requests forbid labels, duplicate
event IDs, nonfinite amounts and extra fields; maximum 5,000 events / 8 MB.

PostgreSQL locks the stream row while it scores the batch. Account state, event
IDs, scores, the batch receipt and sequence advance commit together. A retry
with identical content returns its existing scores; different content at an
existing sequence conflicts. Reused event IDs also conflict. All state is in SQL,
so restarting the API or consumer does not rebuild history from a partial cache.
Read status uses one database snapshot.

Kafka uses one partition, acknowledged production and manual synchronous offset
commits **after** the database transaction. A crash between database commit and
Kafka acknowledgement causes redelivery, which the receipt handles idempotently.
This is at-least-once transport with idempotent database effects, not a distributed
Kafka/PostgreSQL exactly-once transaction. Invalid messages stop consumption;
inspect and repair the producer before continuing. No automatic skipping.

## Setup

Use Python 3.13 and the existing model reproduction instructions in
[temporal-model.md](temporal-model.md) and [risk-policy.md](risk-policy.md).
Install into a separate environment:

```text
python -m pip install -r requirements/locked.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/graph.txt -r requirements/service.txt
python -m pip install --no-deps -e .
```

Set a local database password and connection string in your shell. PowerShell:

```powershell
$env:FRAUD_POSTGRES_PASSWORD = '<choose-a-local-password>'
$env:FRAUD_DSN = 'postgresql://fraud:<URL-encoded-password>@127.0.0.1:55434/fraud'
docker compose -f deploy/compose.yml up -d --wait
```

The Compose project is named `portfolio-fraud`. It has separate persistent
volumes, Kafka on loopback port 19094 and PostgreSQL on loopback 55434.
Pinned image digests are in the Compose file: Kafka 4.1.2, PostgreSQL 17.6.
Kafka is limited to 1 GiB / two CPUs, PostgreSQL to 768 MiB / two CPUs.
A small initialization container sets ownership only on this project's Kafka
volume; Kafka itself runs as its unprivileged image user. No host bind mounts.
These local plaintext ports have no production authentication or TLS;
keep them bound to loopback. Do not reuse the example CI password elsewhere.

Prepare a new output folder from the existing frozen runs:

```text
python scripts/prepare_service.py --data data/HI-Small_Trans.csv --baseline artifacts/baseline --graph artifacts/temporal --risk artifacts/robustness --output artifacts/service --minutes 30
python scripts/create_stream.py --prepared artifacts/service --stream demo
```

This verifies all source artifact hashes and copies models without training.
Prior state uses strictly earlier events, the original account holdout and the
saved graph features. Replay account coverage is selected from observable
endpoints; this controls scope only, not predictive features. The preparation
manifest pins the snapshot, requests, references and model bundle.
Only load trusted locally reproduced model/pickle artifacts.

For HTTP scoring:

```powershell
$env:FRAUD_MODELS = (Resolve-Path artifacts/service/models).Path
python -m uvicorn temporal_fraud.api:create_app --factory --host 127.0.0.1 --port 8034
```

Open `http://127.0.0.1:8034/docs`. POST each line of `batches.jsonl` in order to
`/streams/demo/batches`. GET `/streams/demo` shows the committed sequence and
event count; GET `/streams/demo/batches/0` retrieves an immutable result.
The API returns raw and calibrated probabilities, not immediate alert decisions.
The existing capacity policy waits for the hour to close and enforces its budget;
a probability threshold alone does not guarantee alert capacity.

For Kafka, create a **different new stream** and use the same requests:

```text
python scripts/create_stream.py --prepared artifacts/service --stream kafka-demo
python -m temporal_fraud.broker --topic fraud-demo --stream kafka-demo --publish artifacts/service/batches.jsonl
python -m temporal_fraud.broker --topic fraud-demo --group fraud-demo-consumer --limit 30
```

The consumer reads `FRAUD_DSN` and `FRAUD_MODELS`. Keep a distinct topic/group
for each new experiment. Do not feed the same stream from HTTP and Kafka
concurrently unless retries have identical content and ordering.

## Verification and measured scope

```text
python scripts/benchmark_service.py --input artifacts/service --output artifacts/service-load
python scripts/audit_service.py --prepared artifacts/service --measured artifacts/service-load --graph artifacts/temporal --output artifacts/service-audit.json
```

The benchmark uses fresh stream IDs and a temporary loopback HTTP port, tests
a genuine worker-process restart, a database rollback and redelivery after
database commit. It stops only the API processes it started. It leaves its SQL
records and Kafka topic for inspection. The independent audit compares every
persisted account count, age and neighbor vector to the offline index and checks
raw/calibrated predictions and latency quantiles. Run tests with `FRAUD_TEST_DSN`
and `FRAUD_TEST_KAFKA=127.0.0.1:19094` for live integration coverage.
CI uses these isolated containers on Ubuntu; Windows and Ubuntu also run the
portable model/API tests.

See [measured results](../results/service-v1/README.md). Measurements include SQL,
JSON, graph construction, model scoring and HTTP round trips per complete minute.
They exclude waiting up to one minute for aggregation and any later hourly alert
decision. Two concurrent local streams do not establish distributed capacity.

Stop the local project services while retaining data:

```text
docker compose -f deploy/compose.yml stop
```

The data/model limitations and CDLA-Sharing-1.0 obligations in [data.md](data.md)
still apply. Kafka, PostgreSQL and the Python packages retain their upstream
licenses. Implementation references:
[Apache Kafka Docker](https://kafka.apache.org/41/getting-started/docker/),
[Confluent Python client](https://docs.confluent.io/kafka-clients/python/current/overview.html),
[Psycopg transactions](https://www.psycopg.org/psycopg3/docs/basic/transactions.html).
