# Development v2: separate ablations and combination

All values come from the same three source-group folds **inside the original train split**. The old test scores are not mixed into this curve. Validation and the original test stayed untouched.

| Stage | macro-F1 mean ± fold SD | DE | EN | Negative recall | ms/text | MiB | HTML exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_frozen | 0.6306 ± 0.0214 | 0.6566 | 0.6059 | 0.7029 | 2.497 | 465.1 | 33% |
| R0_reference | 0.6724 ± 0.0159 | 0.7053 | 0.6425 | 0.7183 | 2.519 | 465.1 | 33% |
| R1_quality | 0.6739 ± 0.0153 | 0.7076 | 0.6430 | 0.7230 | 2.489 | 465.1 | 33% |
| R2_article | 0.6727 ± 0.0157 | 0.7053 | 0.6430 | 0.7183 | 2.508 | 465.1 | 100% |
| R3_deploy | 0.6729 ± 0.0158 | 0.7062 | 0.6427 | 0.7188 | 2.381 | 240.7 | 33% |
| R4_combined | 0.6740 ± 0.0144 | 0.7065 | 0.6441 | 0.7220 | 2.419 | 240.7 | 100% |

![Measured development](development_curve.png)

**Selected before the comparison with the first iteration: `R3_deploy`.** Rule: highest mean source-CV macro-F1; within 0.005, latency wins, then size.

B0 is refitted per fold; R0 is the original fine-tuning approach, also retrained only on the respective fit fold. The original full-train model was not evaluated on training folds. R1 only changes the weighting from class to language×class; R2 uses R0 weights with token-weighted chunk aggregation and improved HTML extraction; R3 stores and computes R0 in BF16. R4 combines R1, R2 and R3.

- `R0_reference` vs. `B0_frozen`: Δmacro-F1 +0.0418, Δms/text +0.022. No automatic benefit from combining is assumed.
- `R1_quality` vs. `R0_reference`: Δmacro-F1 +0.0014, Δms/text -0.030. No automatic benefit from combining is assumed.
- `R2_article` vs. `R0_reference`: Δmacro-F1 +0.0003, Δms/text -0.011. No automatic benefit from combining is assumed.
- `R3_deploy` vs. `R0_reference`: Δmacro-F1 +0.0004, Δms/text -0.138. No automatic benefit from combining is assumed.
- `R4_combined` vs. `R0_reference`: Δmacro-F1 +0.0016, Δms/text -0.101. No automatic benefit from combining is assumed.

The 18 synthetic article texts and six HTML fixtures are controlled technical checks, not news articles labelled independently by humans. Their high hit rate therefore makes no claim about real-world article quality. Source CV keeps the same source out of both fit and dev, but not the same event across different publishers. Three folds with one fixed seed give fold spread, not a significance test. An independent benchmark follows only after this selection.

The original v1 test and its one-off evaluation are kept as a historical milestone. The additional fine-tuning ablations are the explicitly requested extension of the original single-model protocol. The harness abort is documented in `harness_incident.json` and kept as a failed MLflow run.

Reproduction: `python -m second_path.development run`, then `summarize`, `release`, `activate`, `python -m second_path.service_benchmark` and `python -m second_path.development_report`. On this tower, completed runs are locked; a fresh clone has no local data or models. First create the data and the original frozen baseline with `second_path.prepare` / `second_path.train frozen`. For fresh reproductions export `SENTIMENT_DEV_REPORTS=reports/reproduction` so the committed historical results stay untouched. The original v1 test and fine-tuning commands are not needed for v2.

MLflow: experiment `article-sentiment-second-path`, tags `phase=development-v2`, `stage`, `fold`. Every fold evaluation contains accuracy, per-class F1, macro-F1, per-language scores, negative precision/recall, latency, bytes and confusion matrix. `decision.json`, `runs.json`, `curve.csv`, `folds.json` and complete local model artefacts make every decision traceable.

Methodological references: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [PyTorch mixed precision](https://docs.pytorch.org/docs/stable/amp.html).

## Local load and recovery test

Deliberately low 10ms bounded queue exposes backpressure; production default is 2s. Synthetic local workload; no URL/network latency or real traffic claim.

| Profile | Steady p50/p95 ms | Burst 200 / 503 | Burst p95 ms | Recovered |
| --- | ---: | ---: | ---: | --- |
| reference | 3.06 / 4.19 | 40 / 0 | 221.59 | True |
| bounded_bf16 | 3.08 / 3.38 | 5 / 35 | 20.70 | True |
