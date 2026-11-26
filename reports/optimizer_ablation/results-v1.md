# Optimizer ablation — exploratory development results

Status: **complete**, 32/32 runs complete. Updated: 2026-11-26T11:52:22.233686+00:00.

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
| run-09 | 20261110 | None | 5e-05 | 0.0 | 0.0 | 0.7634 | 0.7442 | 0.0% |
| run-10 | 20261110 | 1.0 | 5e-05 | 0.0 | 0.1 | 0.7634 | 0.8030 | 93.3% |
| run-11 | 20261110 | 1.0 | 5e-05 | 0.01 | 0.1 | 0.7634 | 0.8030 | 93.3% |
| run-12 | 20261110 | None | 0.0001 | 0.0 | 0.1 | 0.7634 | 0.7516 | 0.0% |
| run-13 | 20261110 | None | 5e-05 | 0.0 | 0.1 | 0.7634 | 0.7442 | 0.0% |
| run-14 | 20261110 | None | 0.0001 | 0.01 | 0.0 | 0.7634 | 0.6967 | 0.0% |
| run-15 | 20261110 | 1.0 | 5e-05 | 0.01 | 0.0 | 0.7634 | 0.7442 | 96.7% |
| run-16 | 20261110 | None | 5e-05 | 0.01 | 0.0 | 0.7634 | 0.7196 | 0.0% |
| run-17 | 20261111 | 1.0 | 0.0001 | 0.01 | 0.1 | 0.7505 | 0.7516 | 96.7% |
| run-18 | 20261111 | 1.0 | 0.0001 | 0.01 | 0.0 | 0.7634 | 0.7516 | 93.3% |
| run-19 | 20261111 | 1.0 | 5e-05 | 0.01 | 0.0 | 0.7850 | 0.7442 | 93.3% |
| run-20 | 20261111 | None | 0.0001 | 0.0 | 0.0 | 0.7505 | 0.7283 | 0.0% |
| run-21 | 20261111 | 1.0 | 0.0001 | 0.0 | 0.0 | 0.7634 | 0.6967 | 93.3% |
| run-22 | 20261111 | None | 5e-05 | 0.0 | 0.1 | 0.7111 | 0.7196 | 0.0% |
| run-23 | 20261111 | None | 5e-05 | 0.01 | 0.1 | 0.7111 | 0.7442 | 0.0% |
| run-24 | 20261111 | 1.0 | 0.0001 | 0.0 | 0.1 | 0.7634 | 0.7516 | 96.7% |
| run-25 | 20261111 | 1.0 | 5e-05 | 0.0 | 0.1 | 0.7634 | 0.7442 | 96.7% |
| run-26 | 20261111 | 1.0 | 5e-05 | 0.0 | 0.0 | 0.7850 | 0.7442 | 96.7% |
| run-27 | 20261111 | None | 5e-05 | 0.0 | 0.0 | 0.7634 | 0.7442 | 0.0% |
| run-28 | 20261111 | None | 0.0001 | 0.01 | 0.1 | 0.7214 | 0.6751 | 0.0% |
| run-29 | 20261111 | None | 0.0001 | 0.01 | 0.0 | 0.7505 | 0.7283 | 0.0% |
| run-30 | 20261111 | 1.0 | 5e-05 | 0.01 | 0.1 | 0.7634 | 0.7442 | 96.7% |
| run-31 | 20261111 | None | 5e-05 | 0.01 | 0.0 | 0.7280 | 0.7442 | 0.0% |
| run-32 | 20261111 | None | 0.0001 | 0.0 | 0.1 | 0.7737 | 0.7283 | 0.0% |

## Matched factor contrasts

High minus low, absolute F1 percentage points, averaged over matched settings and both seeds.

| Factor | Mean effect (pp) | Seed 1 (pp) | Seed 2 (pp) |
|---|---|---|---|
| gradient_clip | +1.75 | +0.65 | +2.85 |
| learning_rate | +0.49 | +0.65 | +0.33 |
| weight_decay | -0.96 | -0.65 | -1.26 |
| warmup_fraction | -1.15 | -0.65 | -1.64 |

Two-factor interactions and all per-seed paired differences are in `summary-v1.json`.

These are descriptive effects for this fixed split and parameter range. Shared articles and two seeds
do not support independent-sample significance claims or a proven causal mechanism.
AdamW itself is fixed; only its weight decay varies. No deployment selection is made.
Training telemetry can support hypotheses about clipping/stability; it does not establish why F1 changed.
