# Optimizer ablation — exploratory development results

Status: **running**, 1/32 runs complete. Updated: 2026-11-26T09:35:37.126646+00:00.

159 fit / 81 development articles, original Train fold 0; no final-Test inference. Two seeds.
Current serving model and frozen Test results remain unchanged.

| Run | Seed | Clip | LR | AdamW decay | Warmup | DE/EN F1 | JA F1 | Clipped steps |
|---|---|---|---|---|---|---|---|---|
| run-01 | 20261110 | 1.0 | 0.0001 | 0.01 | 0.1 | 0.7634 | 0.8030 | 96.7% |

These are descriptive effects for this fixed split and parameter range. Shared articles and two seeds
do not support independent-sample significance claims or a proven causal mechanism.
AdamW itself is fixed; only its weight decay varies. No deployment selection is made.
Training telemetry can support hypotheses about clipping/stability; it does not establish why F1 changed.
