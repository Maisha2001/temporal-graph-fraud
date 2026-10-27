# Transaction Atlas

The local application ranks saved transaction scores, compares tree and graph
probabilities, filters fit-held-out accounts, replays arrivals by minute and
draws strictly earlier account interactions. The context is an observed
neighborhood, not a causal explanation. Optional simulation outcomes are hidden
by default and never affect ranking. Export omits labels unless explicitly shown.

## Quick start

Install the base and service requirements as in the README. No database, GPU or
dataset is needed for the included authored example.

PowerShell:

~~~powershell
$env:FRAUD_CASES = (Resolve-Path examples/investigation).Path
python -m uvicorn temporal_fraud.investigation:create_app --factory --host 127.0.0.1 --port 8035
~~~

POSIX shell:

~~~sh
FRAUD_CASES=examples/investigation python -m uvicorn temporal_fraud.investigation:create_app --factory --host 127.0.0.1 --port 8035
~~~

Open http://127.0.0.1:8035. Change ranking model, select a case, inspect the graph,
move the replay slider, filter accounts, reveal outcomes or export JSON.
Stop the server with Ctrl+C. Keep it on loopback; it has no access control.

The authored example's scores and timing are illustrative. It is an interface
smoke test, not measured model evidence. Browser tests cover ranking, selection,
replay, filters, hidden outcomes, export and a 390-pixel mobile layout.

## Real measured development replay

First reproduce the [model experiments](reproduction.md) and
[service benchmark](serving.md). Use consistent directories from those commands:

~~~sh
python scripts/prepare_investigation.py --data data/HI-Small_Trans.csv --graph artifacts/temporal-v1 --measured artifacts/service-load --output artifacts/investigation
~~~

Set FRAUD_CASES to that new output folder and start the same server. Preparation
verifies the service report and receipts, then selects the union of the top 60
calibrated cases from each model. The recorded export has 93 distinct cases and
1,067 earlier context interactions from 9,567 September 7 transactions.
Neither final heldout data nor its results are used in the application.

The displayed candidates are a review sample, not the closed-hour capped alert
queue. Probabilities alone do not enforce capacity. The app is a read-only view
of measured model output; HTTP/Kafka inference runs in the separate service.

The export is local derived IBM data under CDLA-Sharing-1.0, with attribution and
modification notices. Keep the license and notices when redistributing. No such
transaction bundle is committed here. The private verification used the actual
93-case export, checked both rankings and all 14 cold cases, hid outcomes on
export, and captured desktop/mobile screenshots with no browser errors.
