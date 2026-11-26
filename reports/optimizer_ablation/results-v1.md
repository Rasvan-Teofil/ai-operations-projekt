# Optimizer ablation — exploratory development results

Status: **running**, 8/32 runs complete. Updated: 2026-11-26T10:07:41.499606+00:00.

159 fit / 81 development articles, original Train fold 0; no final-Test inference. Two seeds.
Current serving model and frozen Test results remain unchanged.

| Run | Seed | Clip | LR | AdamW decay | Warmup | DE/EN F1 | JA F1 | Clipped steps |
|---|---|---|---|---|---|---|---|---|
| run-01 | 20261110 | 1.0 | 0.0001 | 0.01 | 0.1 | 0.7634 | 0.8030 | 96.7% |
| run-02 | 20261110 | None | 5e-05 | 0.01 | 0.1 | 0.7111 | 0.7196 | 0.0% |
| run-03 | 20261110 | 1.0 | 0.0001 | 0.0 | 0.1 | 0.7634 | 0.8030 | 96.7% |
| run-04 | 20261110 | 1.0 | 0.0001 | 0.0 | 0.0 | 0.7634 | 0.8030 | 96.7% |
| run-05 | 20261110 | None | 0.0001 | 0.0 | 0.0 | 0.7634 | 0.7516 | 0.0% |
| run-06 | 20261110 | 1.0 | 5e-05 | 0.0 | 0.0 | 0.7634 | 0.8030 | 93.3% |
| run-07 | 20261110 | 1.0 | 0.0001 | 0.01 | 0.0 | 0.7634 | 0.8030 | 96.7% |
| run-08 | 20261110 | None | 0.0001 | 0.01 | 0.1 | 0.7634 | 0.7516 | 0.0% |

These are descriptive effects for this fixed split and parameter range. Shared articles and two seeds
do not support independent-sample significance claims or a proven causal mechanism.
AdamW itself is fixed; only its weight decay varies. No deployment selection is made.
Training telemetry can support hypotheses about clipping/stability; it does not establish why F1 changed.
