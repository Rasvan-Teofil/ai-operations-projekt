# Measured comparison (second iteration)

The table is the historical v1 test from branch `second-path` and does not replace [`model_comparison.md`](model_comparison.md) of the first iteration. The numbers below are taken over unchanged.

# Measured comparison

Own shared test: 1,787 DE/EN sentences, seed 20261018. One-off final evaluation on 18 Oct 2026 on an RTX 5060 Ti (CUDA 12.8). Not comparable with the test split on main.

| Model | Accuracy | macro-F1 | F1 negative | F1 neutral | F1 positive | ms/text | MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.6676 | 0.6656 | 0.6822 | 0.6813 | 0.6333 | 2.511 | 465.14 |
| finetuned | 0.7230 | 0.7187 | 0.7182 | 0.7416 | 0.6962 | 2.500 | 465.14 |
| centroid | 0.6402 | 0.6410 | 0.6758 | 0.6372 | 0.6100 | 2.563 | 465.14 |
| majority | 0.4841 | 0.2174 | 0.0000 | 0.6523 | 0.0000 | 0.005 | 0.00 |

**We ship our own GPU fine-tuning instead of the frozen baseline because it raises macro-F1 on the shared test from 0.6656 to 0.7187 (+0.0531) at practically the same artefact size and inference time.**

The gap is above the pre-registered tie threshold of 0.005, so latency does not decide here. Both encoder artefacts take about 465 MiB; the extra fine-tuning time is 40.3 s after loading the initialisation (baseline training 11.3 s without download/loading). The majority control is smaller and faster but misses every negative and positive text. Centroid costs about the same at inference and is less accurate.

Fine-tuned per language: DE macro-F1 0.7541 (baseline 0.7008), EN 0.6913 (baseline 0.6383). English remains the weaker language. These are point estimates from our own test, not a comparison with main and not a significance test.

## Run IDs

- `frozen`: `6f942a83f7c949b2a3a70d9036b2d12d`
- `finetuned`: `5b1864e1e45e49cc99a74c33a39b5660`
- `centroid`: `d130c7f51fe749a5a17326f5503f4d78`
- `majority`: `8a7d115af9d54edc83a2d029e034df35`

Experiment: `article-sentiment-second-path`, local SQLite database `mlflow.db` (as on branch `second-path`; on `main` the store is `./mlruns` and `python -m src.import_recorded` loads the numbers). Confusion matrices: rows true, columns predicted, class order negative/neutral/positive. Latency: 20 independent warm-ups, batch 1, tokenisation and transfer included, CUDA synchronisation, no loading or network. All complete model and tokenizer files count towards size.

The validation file was only hashed for an integrity check. There were no validation metrics and no checkpoint selection. Fine-tuning end point: epoch 3, 8,335 rows each; the first and last encoder layers show changed weights compared with the frozen baseline (see training_evidence.json).

A warm-up abort without test predictions is documented separately in preflight_incident.json; it produced no scores and no model change. Before the test, the repaired inference path was only checked on hand-written example texts. Once the test scores were known, models and inference were no longer changed.
