<!--
English draft of main's README.md (main @ 2c612d8, 14 Nov 2026 state).
Prepared on daily-refinement on 2026-11-25. Not yet applied to main.
To apply: copy this file over README.md on main and delete this comment.
Links are unchanged; several linked docs are still German (see docs/translation-checklist.md).
-->
# Article Sentiment

**AI Operations · Hochschule Heilbronn · M.Sc. Wirtschaftsinformatik · Winter semester 2026/27**

Media monitoring for native German, English and Japanese news: classify a full text or exactly one HTTP(S) URL as `negative`, `neutral` or `positive`. **The current project state is entirely on `main`.** Earlier studies are kept in the archive.

## Status · 14 November 2026

The full-article study is complete: 27 train-internal development evaluations, four fits on the same 240 train articles and one shared fresh test set with **691 articles: DE 273, EN 118, JA 300**. Qwen LoRA was selected before the test set was collected and is now the local API default.

**References reviewed by the project team**: the project lead reviewed the created or classified data and confirmed it on 14 November 2026. The original AI-assisted annotation and the later review are documented separately in the [review record](docs/data/project-review.md). The headline metric weights DE and EN macro-F1 equally; JA is reported separately. The English test set was not artificially padded.

| Method | DE/EN macro-F1 | DE | EN | JA | warm ms/article | total MiB |
|---|---:|---:|---:|---:|---:|---:|
| frozen128 | 0.5150 | 0.5281 | 0.5019 | 0.6257 | 3.72 | 240.73 |
| minilm128 | 0.5928 | 0.4734 | 0.7122 | 0.5655 | 3.66 | 240.73 |
| qwen-zero | 0.7796 | 0.7508 | 0.8085 | 0.8143 | 172.33 | 7687.49 |
| qwen-lora | 0.8218 | 0.7612 | 0.8825 | 0.8132 | 184.26 | 7698.77 |

- **[Open data and original sources](docs/data/README.md)** · [Review status](docs/data/project-review.md)
- [Results with uncertainty, cost and run IDs](reports/next_stage/comparison-v1.md) · [full JSON](reports/next_stage/comparison-v1.json)
- [Model cards and limitations](reports/next_stage/model-cards-v1.md) · [unchanged references and error cases](reports/next_stage/error-analysis-v1.json)
- [Reproduce](docs/reproduce-full-article.md) · [Repository structure and file placement](docs/repository-structure.md)
- [Sources and licences](licenses/README.md) · [Inventory](licenses/inventory.json)
- [Historical studies and commands](docs/archive/project-history-through-v5.md) · [Phase log](docs/next-stage.md)

## Languages: German, English and Japanese

| Language | Native train articles | Native test articles | Qwen LoRA macro-F1 |
|---|---:|---:|---:|
| German | 78 | 273 | 0.7612 |
| English | 71 | 118 | 0.8825 |
| **Japanese / 日本語** | **91** | **300** | **0.8132** |

Japanese is implemented and evaluated separately. These are native Japanese
full articles, not translated DE/EN test texts.
[Japanese: data, results, usage and limitations](docs/languages/japanese.md).
DE/EN remain the pre-registered headline metric; JA is not added to it after the fact.

## Explaining the project from main

1. [Problem, target users and architecture](docs/project-overview.md).
2. [Current results including Japanese](reports/next_stage/comparison-v1.md).
3. [Repository structure following the lecture template](docs/repository-structure.md).
4. [Presentation guide](docs/presentation-guide.md) and [reproduction](docs/reproduce-full-article.md).
5. [Sources and licences](licenses/README.md), [verification records](reports/next_stage/verification-final-v1.json)
   and [original main state](docs/archive/main-before-full-article/README.md).
6. [Code walkthrough for Qwen and the best model](docs/qwen-code-walkthrough.md)
   and [course requirements audit](docs/course-completion-audit.md).

All linked files, the current application and its tests are on this main branch.
`second-path` survives only as a historical development label in run logs and MLflow.
No branch switch is needed for the presentation.

**Public demo:** not deployed yet. The running application and MLflow are local so far.
Free hosting without the tower machine is a separate, still open deployment task; a GitHub
Page alone does not run the model. [Hosting status and free options](docs/public-hosting.md).

## What follows from this

Qwen LoRA reaches the highest agreement on the equally weighted DE/EN headline metric. The paired 95% bootstrap interval of the difference to Qwen Zero is [0.0070; 0.0826], conditional on this source pool and the fixed reference labels. This is not evidence of 95% accuracy. On Japanese, LoRA is slightly below zero-shot; MiniLM's improvement also does not hold for every language.

The fine-tuned MiniLM needs about 3.66 instead of 184.26 warm ms/article and about 241 instead of 7699 MiB. These sizes include the base weights and adapters each model needs on its own. Times include tokenisation, transfer, full context and aggregation; model loading and URL fetching are excluded. GPU: RTX 5060 Ti, batch size 1.

The training size of 240 is not a proven optimum: the small learning curve had not saturated. 3000 was an upper bound, 6000 remains future work. A single archive, limited direct quotes, at most 20000 characters / 4096 Qwen article tokens, incomplete detection of semantic links and possible pre-training overlap limit transferability. The project-side review is documented; an additional blinded double annotation with measured inter-annotator agreement was not recorded.

## Data and test lock

Train: DE 78 / EN 71 / JA 91 after two reading passes plus rights and event checks. In the fresh test source frame, 856 articles were collected and each read fully twice. 41 class disagreements were resolved with a stated reason before any model predictions; both original reviews are kept. Four still unresolvable annotations, plus open rights, source-format problems and protected event links, are excluded and documented. Exclusion reasons can overlap. Whole groups are handled together.

691 cases in 658 observed groups remain after the pre-registered source order and per-language cap. The English 1500-page sample source was exhausted; this does not mean there are too few English news articles worldwide. Source losses and language/class frequencies are in the [test manifest](reports/next_stage/final-test-v1/test-manifest.json). No class balancing, text rewrites, duplicates or synthetic test cases. All 856 raw cases stay permanently reserved for testing.

Four candidates × 691 successful stored test predictions, no second test pass for timing and no post-hoc tuning. Warm-ups use train data only. Uncertainty and error analysis use the same stored predictions. The eight historical guard files, old reports and MLflow runs are kept; v1–v5 and full-article scores are not joined into one progress curve.

## Run locally

Python 3.12 with an existing CUDA PyTorch environment; raw data, weights and the MLflow database are git-ignored and live outside Git. A fresh clone first contains evidence and code only. Starting the API trains nothing and downloads no weights.

```bash
source .venv/bin/activate
nvidia-smi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Check for an existing listener and do not start a second service. [Local app](http://127.0.0.1:8000/) · [API docs](http://127.0.0.1:8000/docs) · [MLflow](http://127.0.0.1:5000/). `127.0.0.1` is reachable locally only; no public deployment is claimed.

```bash
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"Das Unternehmen meldet einen Gewinneinbruch."}'
```

Exactly one of `text` or `url`. Optional query aliases: `article_frozen_v1`, `article_minilm_v1`, `article_qwen_zero_v1`, `article_qwen_lora_v1`. Historical models remain selectable. `/predict` returns `label`, three normalised `scores`, the actual `model_version`, `input_policy` and, for URLs, an excerpt. Scores are not calibrated. `/analyze` adds sentence labels with original-text offsets, at most 80 sentences; colours are additional predictions, not causal word explanations.

Full-article Qwen rejects inputs over 4096 article tokens and does not cut anything from the middle. MiniLM processes all 128-token chunks as equally averaged features before the article head. Historical Qwen keeps its 3072-token head/tail rule, historical MiniLM profiles keep their own aggregation.

Model-independent URL fetching with `full-article-v5`, without silent character truncation. Private addresses, redirects, robots disallows and detected login/paywall pages stay blocked. 422/403/502/503 and optional `X-Article-Error` diagnostics are kept. Broken or changed publisher layouts can still fail; pasting the text is always possible.

## Verification and reproduction

**189 offline tests pass**; synthetic GPU/HTTP checks of all four models in DE/EN/JA, Japanese sentence offsets, context limits and a browser check are documented separately. No test texts were re-classified for this. See the [verification record for the source catalogue and review view](reports/data-review/verification-v1.json) and the [original study verification record](reports/next_stage/verification-final-v1.json).

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -p no:cacheprovider
python -m src.next_stage_reproduce --name review-only
```

By default the second command only checks. An explicitly started reproduction run uses its own namespace and new MLflow IDs. The [guide](docs/reproduce-full-article.md) covers weights, exact data restoration, failures on differing rendering and the test lock. Source attribution and the old licence restrictions are kept.

## Team

Teofil Wetzel and Nick Fronius · AI Operations, Hochschule Heilbronn.
