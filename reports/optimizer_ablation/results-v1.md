# Optimizer ablation — exploratory development results

Status: **blocked-gpu-xid-79-awaiting-reboot**, 0/32 runs complete. Updated: 2026-11-26T09:23:58.797733+00:00.

159 fit / 81 development articles, original Train fold 0; no final-Test inference. Two seeds.
Current serving model and frozen Test results remain unchanged.

| Run | Seed | Clip | LR | AdamW decay | Warmup | DE/EN F1 | JA F1 | Clipped steps |
|---|---|---|---|---|---|---|---|---|

These are descriptive effects for this fixed split and parameter range. Shared articles and two seeds
do not support independent-sample significance claims or a proven causal mechanism.
AdamW itself is fixed; only its weight decay varies. No deployment selection is made.
Training telemetry can support hypotheses about clipping/stability; it does not establish why F1 changed.
