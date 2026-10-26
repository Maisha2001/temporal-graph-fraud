# Temporal graph comparison

The fixed comparison uses all 1,851,794 eligible training events and the same
964,840 development events (1,028 positives) as the saved tree baselines. Every
neural arm trains for three epochs / 1,359 updates. September 8 onward remains
unscored. Baseline models and their seven local artifacts are unchanged.

| Model | AP | Recall at 1% capacity | Precision at capacity | Brier |
| --- | ---: | ---: | ---: | ---: |
| train_prior | 0.00107 | 1.00% | 0.11% | 0.001065 |
| transaction_xgboost | 0.05950 | 43.09% | 4.59% | 0.004580 |
| history_xgboost | 0.17228 | 63.62% | 6.78% | 0.005858 |
| history_mlp | 0.07101 | 44.07% | 4.70% | 0.005135 |
| temporal_attention_8 | 0.06978 | 54.38% | 5.79% | 0.003394 |
| temporal_attention_32 | 0.10365 | 56.91% | 6.06% | 0.004175 |

The 32-neighbor graph improves on the history MLP in AP and capacity recall,
but the history XGBoost baseline remains stronger on both. The 8-neighbor graph
has slightly lower AP than the MLP despite higher capacity recall. No result
justifies a claim that graph learning is universally better. Different parameter
counts and a single fixed seed/budget limit causal claims about architecture.
No epoch or hyperparameter was selected using these validation outcomes.

Cold-account AP is 0.02076 (MLP), 0.02602 (8 neighbors), and 0.02973 (32 neighbors),
versus 0.04565 for history XGBoost. Both graph arms recover 32.37% of cold-account
positives at their separate 1% capacity; this is not the full-population ranking.
Class-weighted scores remain poorly calibrated; most alerts are false positives.
The independent synthetic AML labels do not establish real-world performance.

## Runtime and verification

Windows 11, Python 3.13.13, Torch 2.11.0+cu128, PyG 2.8.1, RTX 5070 and four CPU
threads. Total measured run: 103.29 seconds, including 38.85 seconds preparation.
Fits took 2.99 seconds (MLP), 21.51 seconds (8 neighbors), and 31.19 seconds
(32 neighbors). Peak process working set was 4,051,165,184 bytes (3.77 GiB);
peak Torch GPU allocation was 892,179,968 bytes (0.83 GiB), excluding driver and
other application allocations. These are local measurements, not service latency.

All 3,195,268 replay membership rows exactly match the original baseline. The
independent audit verified 20 new and seven original artifacts, checked 128
endpoint queries against exhaustive past-edge lookup, and recomputed nine metric
groups from saved predictions. CPU checkpoint reloads on 256 spread-out validation
events per model differ from saved GPU predictions by at most 5.97e-8. The normal
run also checks 512 predictions per checkpoint on the training device.

`report.json` retains exact losses, parameters, hashes, cold/warm metrics and
runtime. `comparison.json` combines the unchanged baseline and new neural metrics.
`audit.json` records independent checks. Reproduction instructions and limitations
are in [the model notes](../../docs/temporal-model.md).

To audit your own trusted local artifacts without training:

```sh
python scripts/audit_temporal_run.py --run artifacts/temporal-v1 --baseline artifacts/baseline --output artifacts/temporal-audit.json
```

The repository contains aggregate evidence; generated transaction tables and
checkpoints remain local. CI uses small authored fixtures on Windows and Ubuntu,
checks interruption recovery, and verifies evidence hashes. It does not retrain
the full IBM dataset.

Published JSON uses LF line endings. The private raw run report keeps its original
bytes. `comparison.json` records both raw and published report hashes; model,
prediction and input artifact hashes refer to the unchanged private artifacts.
The initial graph commit hashed two Windows CRLF files before Git normalized
them. The subsequent evidence correction fixes those text hashes and adds
verification of staged and committed Git blobs. Metrics and models are unchanged.
