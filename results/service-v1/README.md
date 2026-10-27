# Stateful service measurements

Frozen tree/graph models scored 9,567 development events in 30 complete minutes
per stream (September 7, 12:00-12:30). Four streams cover sequential HTTP with an
API process restart, two concurrent HTTP streams, and Kafka redelivery after a
database commit. All 38,268 scored copies match saved offline predictions:
tree maximum error 0; graph 1.79e-7; calibrated graph 4.94e-8.

| Path | Batch median | Batch p95 | Batch p99 |
|---|---:|---:|---:|
| Sequential HTTP, 30 requests | 494 ms | 582 ms | 636 ms |
| Two concurrent streams, 60 requests | 981 ms | 1,138 ms | 1,170 ms |
| Kafka to database, 30 deliveries including one duplicate | 507 ms | 613 ms | 628 ms |

Concurrent replay processed 19,134 events in 29.35 seconds (652 events/second).
These are complete-minute **batch** latencies, not per-event inference times.
Requests include graph state reads/writes, model inference, serialization and
transport. They exclude up to one minute of aggregation and later hourly alert
selection. Results are one local workload, not a distributed scaling claim.

The Kafka burst includes deliberate failure and consumer restart. Its queued
message-to-database p95 is 16.07 seconds; do not confuse this with the 613 ms
processing p95. One redelivered batch returned the existing result, with no
extra events or state updates. Database rollback and four simultaneous duplicate
writers are tested separately. Actual API worker PIDs changed on restart and
both workers stopped. The initial diagnostic run inspected the Windows virtual
environment launcher instead of the API worker for memory; it remains saved
privately and is excluded from this summary. A separate corrected run produced
these measurements, without changing models or predictions.

Windows 11 / Python 3.13.13, four Torch CPU threads, no GPU inference. API peak
worker working set was 1,101,275,136 bytes (1.03 GiB), excluding Kafka/PostgreSQL
and the driver. Importing four 8,261-account snapshots took 16.39 seconds, outside
request timings. PostgreSQL is limited to 768 MiB; Kafka to 1 GiB, two CPUs each.
The 80.3 MB JSON snapshot is a correctness-oriented bounded replay, not a compact
production state format. Persistent SQL event storage grows with replay history.

Independent audit verified 33,044 final account states and 772,000 neighbor
vectors against the offline graph, all scores/calibrations, 90 HTTP samples,
10 prepared artifact hashes and five private measurement artifact hashes.
Local tests: 60 passed, including real PostgreSQL and Kafka. CI separates portable
Windows/Ubuntu tests from a live-container Ubuntu integration job.

Raw batch timings and Kafka observations are included here. report.json retains
hashes for private logs/full response records; audit.json records their checks.
Public canonical files have their own checksums.json, so Windows newline changes
cannot silently invalidate committed evidence.

The final holdout remains unscored. Synthetic AML limitations, fixed model
comparisons and calibration caveats remain unchanged. This service rejects
unknown accounts outside its bootstrapped scope, late minutes, sequence gaps,
conflicting retries, repeated event IDs and the final-holdout interval.
See [setup and protocol](../../docs/serving.md).
