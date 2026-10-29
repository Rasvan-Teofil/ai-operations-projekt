# Article Sentiment

**Course:** AI Operations  
**University:** HHN Hochschule Heilbronn (Heilbronn University of Applied Sciences)  
**Programme:** M.Sc. Business Informatics (Wirtschaftsinformatik)  
**Semester:** Winter 2026/27

The tool reads a news text or exactly one article URL and estimates its sentiment: negative, neutral or positive. German and English.

## Summary

Built for newsrooms, PR & communications teams and analysts. Later, not yet labelled or modelled here: a score for judgemental or loaded language.

## Two iterations

Both paths live in this state of the repo. Neither replaces the other.

| | First iteration (packages `src` / `app`) | Second iteration (from branch `second-path`, on `main` since PR #2, package `second_path`) |
| --- | --- | --- |
| Models | TF-IDF + LogReg, TF-IDF + LinearSVC, two HF zero-shot models, DistilBERT fine-tuning | MiniLM `paraphrase-multilingual-MiniLM-L12-v2`: frozen, fine-tuned, centroid, majority; then ablations B0 and R0–R4 |
| Own test | Financial sentences, n = 1,270. Best macro-F1: **LinearSVC 0.7202** | Kenpache DE/EN, n = 1,787. Fine-tuned **0.7187** vs. frozen 0.6656 |
| Serving | API default `tfidf_linearsvc` | v2 selects **R3_deploy** (BF16, source-CV macro-F1 0.6729, about 241 MiB) |
| Training | CPU for TF-IDF. Transformers optional | CUDA, BF16. A CPU fallback is refused |
| API | `uvicorn app.main:app` | `uvicorn second_path.api:app` |
| MLflow experiment | `article-sentiment`, tag `iteration=1-first-path` | `article-sentiment-second-path`, tag `iteration=2-second-path` |

The fair comparison of both approaches on the **same** 600 SEntFiN headlines is in [`reports/development/branch_comparison.md`](reports/development/branch_comparison.md): reconstructed LinearSVC macro-F1 EN 0.4524, second-path v1 0.7360, v2 0.7293. The earlier test scores 0.7202 and 0.7187 come from different splits and are not ranked against each other. The executive pitch is in [`docs/executive-pitch/`](docs/executive-pitch/).

Second-iteration artefacts: data under `data/second_path/processed/`, weights under `models/second_path/`. The first iteration stays in `data/processed/` and `models/`.

## Problem & stakeholders

| | |
| --- | --- |
| Business problem | Sort incoming articles by sentiment without reading every text by hand |
| Decision | What is urgently negative, what is a neutral report, what is demonstrably positive |
| Stakeholders | Newsrooms, PR & Communications, Analysts |
| Languages | German and English |
| Errors | A false alarm and a missed negative text cost different amounts. The costs are still open, so the selection metric is macro-F1 |

The idea was agreed with the lecturer. He expects a comparison of several models and our own fine-tuning.

## Prediction target

| | |
| --- | --- |
| Target | Sentiment of a text |
| Classes | `negative`, `neutral`, `positive` |
| Five levels in zero-shot | Very Negative and Negative → `negative`, Neutral → `neutral`, Positive and Very Positive → `positive` |
| Selection | Highest macro-F1 on the shared test split; if close, inference time and model size |

The mapping uses the label names (`src/labels.py`).

## Dataset

One row is one sentence from a financial news item. The collection is usable for academic purposes and stays local: `python -m src.prepare_data` downloads it, `.gitignore` keeps the texts out of Git. The dummy file `data/sample/sentiment_sample.csv` (18 hand-written sentences, `origin=dummy`) is only for tests and CI (`SENTIMENT_DATA=dummy`).

| Language | Source | Subset | Licence | Rows after filtering |
| --- | --- | --- | --- | --- |
| English | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | `Sentences_75Agree.txt` (≥ 75 % annotator agreement) | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | 3,440 |
| German | [Kenpache/multilingual-financial-sentiment](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment) | `language=de` only | academic / non-commercial; the card also lists Apache-2.0, we apply the stricter reading | 5,023 |

Phrasebank labels take an investor's view: would the sentence rather raise, lower or not move the share price. Kenpache contains shorter sentences from German-language financial media; its labels are usable but coarser (glossary rows, occasionally merged reports). Both sources already provide the three class names. 8,463 sentences in total.

**Split** (only in `src/prepare_data.py`, seed 42, strata `label|language`): first 15 % test, from the rest validation as 15 % of the total (`0.15/0.85`), train is the remainder.

| Split | Rows |
| --- | --- |
| Train | 5,923 |
| Val | 1,270 |
| Test | 1,270 |

Validation stays untouched. All five models see the same test split. There is no article ID, so sentences from the same report can end up in both train and test. Provenance, label mapping and limits are in `data/README.md`.

## Quick start

Python 3.12, project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell): `.venv\Scripts\Activate.ps1`, then the same pip call.

Zero-shot and fine-tuning additionally need PyTorch (CPU wheel) and Transformers. CI does not install this file:

```bash
python -m pip install -r requirements-transformer.txt
```

Fetch and split the data (downloads the two Hugging Face files, writes `data/processed/`, does not commit it):

```bash
python -m src.prepare_data
```

Train both TF-IDF models and log them to MLflow:

```bash
python -m src.train
```

Our own fine-tuning. Without a GPU, the recorded run in this repo is the CPU subset: one epoch, 2,000 stratified training sentences, context 128. Evaluation uses the full test split. On a small CPU this takes a few minutes once `distilbert-base-multilingual-cased` is cached.

```bash
python -m src.finetune --epochs 1 --lr 2e-5 --batch-size 8 --max-length 128 --max-samples 2000
```

Full run on a GPU (Google Colab, Runtime → GPU): `notebooks/02_finetune_colab.ipynb`. Three epochs there, the whole train split, context 256, CUDA PyTorch instead of the CPU wheel. Afterwards put the `models/finetuned` folder in the same place locally.

All five models on the same test split, one table under `reports/`:

```bash
python -m src.train --all
```

`python -m src.compare` does the same. If `models/finetuned` is missing, the comparison runs without the fine-tuned model and says so in the table.

### MLflow: both iterations in one store

Windows, Git Bash, project root. One store, `./mlruns`. The UI is `mlflow ui --backend-store-uri ./mlruns`.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Iteration 1 live on CPU (TF-IDF, dummy or, after `python -m src.prepare_data`, the real splits). Every run gets the tag `iteration=1-first-path`:

```bash
python -m src.train
```

The published metrics of **both** iterations, without a GPU and without retraining, from the committed reports. This is the CPU route for the second iteration, whose training requires CUDA and refuses a CPU fallback:

```bash
python -m src.import_recorded
```

UI:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Linux and macOS: `source .venv/bin/activate`, same `python` calls.

Open <http://127.0.0.1:5000> in the browser.

| Experiment | Contents | Tag |
| --- | --- | --- |
| `article-sentiment` | Live TF-IDF and the imported table from `reports/model_comparison.md` | `iteration=1-first-path` |
| `article-sentiment-second-path` | Imported v1 tests, source CV R0–R4 and the external audit | `iteration=2-second-path` (the reconstructed LinearSVC in the audit: `iteration=1-first-path`) |
| `article-sentiment-comparison` | The same imported rows side by side | both iteration tags |

To compare in the UI, open the experiment `article-sentiment-comparison` and filter by `comparable_group`. `sentfin-external` is the only block where the first and second iteration are scored on the same texts. `first-path-own-test`, `second-path-own-test` and `second-path-source-group-cv` are separate measurements. Column `f1_macro` is set in all imported runs.

A fresh live run of the second iteration (NVIDIA GPU only, CUDA 12.8, lots of disk) writes to the same `./mlruns` and the same experiment `article-sentiment-second-path`, tag `iteration=2-second-path`:

```bash
python -m pip install -r second_path/requirements.txt -r requirements-cuda.txt
python -m second_path.prepare
python -m second_path.train frozen
python -m second_path.train finetuned
python -m second_path.train centroid
python -m second_path.train majority
python -m second_path.evaluate
```

Ablations and release, also CUDA and BF16: `python -m second_path.development run`, then `summarize`, `release`, `activate`. Historical logs stay under `reports/development/`. For fresh repetitions run `export SENTIMENT_DEV_REPORTS=reports/reproduction` first so the committed reports are not overwritten. Second-iteration API: `uvicorn second_path.api:app --host 127.0.0.1 --port 8000`. The first iteration stays `uvicorn app.main:app`.

A live run in the experiment `article-sentiment` contains accuracy, macro-F1, per-class F1, `inference_ms_per_text` and the artefact `confusion_matrix.csv`.

API, default model `tfidf_linearsvc`:

```bash
uvicorn app.main:app --reload
```

- Health: <http://127.0.0.1:8000/health>
- Docs: <http://127.0.0.1:8000/docs>

Other model: `SENTIMENT_MODEL=tfidf_logreg`, `tabularisai`, `cardiffnlp` or `finetuned`. Transformers and the fine-tuned model need `requirements-transformer.txt` and the weights on disk.

### Example calls

Health after `python -m src.train`:

```json
{"status":"ok","model_loaded":true,"model_name":"tfidf_linearsvc","model_version":"tfidf-linearsvc-v1"}
```

German, raw text ("The share lost significant value after the profit warning."):

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Die Aktie verlor nach der Gewinnwarnung deutlich an Wert."}'
```

```json
{"label":"negative","scores":{"negative":0.7077840993250278,"neutral":0.14219203378102802,"positive":0.15002386689394404},"model_version":"tfidf-linearsvc-v1"}
```

English, raw text:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Samsung expects a nine-fold surge in quarterly profits, driven by demand for AI memory chips."}'
```

```json
{"label":"positive","scores":{"negative":0.2651965190241689,"neutral":0.2732371307859462,"positive":0.4615663501898849},"model_version":"tfidf-linearsvc-v1"}
```

Article URL (exactly one, fetched on 8 October 2026). BBC: "AI chip boom pushes Samsung profits to record $80bn", <https://www.bbc.com/news/articles/c687z8127302o>.

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.bbc.com/news/articles/c687z8127302o"}'
```

```json
{"label":"positive","scores":{"negative":0.20056848069748298,"neutral":0.20885450235096512,"positive":0.590577016951552},"model_version":"tfidf-linearsvc-v1"}
```

The LinearSVC scores are a softmax of the decision function, not a calibrated probability. Text and URL together, or neither, returns HTTP 422.

| Situation | HTTP |
| --- | --- |
| Text or URL missing, both set, empty text, wrong type | 422 |
| Page without readable article text | 422 |
| `robots.txt` forbids fetching | 403 |
| Timeout, HTTP error, redirect, host cannot be resolved | 502 |
| Model file missing | 503 |

`src/scraping.py` fetches exactly one http(s) URL. It follows no links and no redirects. Timeout 15 seconds. User agent `AIOpsSentimentBot/0.1`. If `robots.txt` is missing (404), the single fetch is allowed; a read error aborts. Local and non-public addresses are not fetched. Still read the site's terms of use.

## Model comparison

The measured table, the trade-off discussion and the recommendation are in [`reports/model_comparison.md`](reports/model_comparison.md). In short, CPU, shared test set of 1,270 sentences:

| Model | macro-F1 | ms per text | Size |
| --- | --- | --- | --- |
| TF-IDF + LinearSVC | 0.720 | 0.02 | 4 MB |
| TF-IDF + LogisticRegression | 0.656 | 0.03 | 4 MB |
| Fine-tuning, CPU subset (2,000 sentences, 1 epoch, context 128) | 0.614 | 8.3 | 516 MB |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | 0.486 | 19.0 | 1,061 MB |
| tabularisai/multilingual-sentiment-analysis (5→3) | 0.364 | 8.8 | 516 MB |

**Recommendation and API default: `tfidf_linearsvc`.** Highest macro-F1, small, fast, runs with `requirements.txt`. The full GPU run (3 epochs, all 5,923 training sentences, context 256) is prepared but not measured here. CI sets `SENTIMENT_MODEL=tfidf_logreg` and `SENTIMENT_DATA=dummy`, installs only `requirements.txt` and makes no network calls.

Every model has `predict_proba(texts)` (`src/interface.py`). TF-IDF sees the whole text. Transformers split it into chunks of at most 512 tokens and average the distributions. Fine-tuning truncates to `--max-length` during training and chunks with that length at inference.

## Tests

```bash
SENTIMENT_DATA=dummy SENTIMENT_MODEL=tfidf_logreg python -m src.train
pytest
```

CI does exactly this in the `test` job. The `second-path` job installs `second_path/requirements.txt` (without CUDA torch) and runs `pytest tests/secondpath`. The tokenizer test is skipped in the light job as long as `transformers` is missing.

Exploration of the real splits, with saved output: `notebooks/01_exploration.ipynb`.

## Repository structure

```text
.
├── README.md
├── requirements.txt                 # first iteration, CI job test
├── requirements-transformer.txt     # zero-shot and DistilBERT
├── requirements-cuda.txt            # second iteration, CUDA torch
├── requirements-evaluation.txt
├── requirements-lock.txt
├── second_path/                     # MiniLM, ablations, second API
│   └── requirements.txt
├── src/                             # first iteration
├── app/                             # FastAPI of the first iteration
├── data/README.md
├── data/sample/                     # dummy, tests and CI
├── notebooks/
├── reports/model_comparison.md      # own test, first iteration
├── reports/second_path_model_comparison.md
├── reports/development/             # ablations and external comparison
├── docs/executive-pitch/            # pitch, TeX and PDF
├── tests/secondpath/
└── .github/workflows/ci.yml         # pytest for both paths
```

## Milestones

### M1 – Problem, data, baseline

- [x] Business idea agreed (article sentiment, DE/EN)
- [x] Prediction target: negative / neutral / positive
- [x] Labelled news sentences chosen, provenance and licence recorded
- [x] Split rule recorded (70/15/15, label and language, seed 42; no article ID)
- [x] TF-IDF baseline on the real train split

### M2 – Comparison and selection

- [x] Five models, same metrics, one MLflow experiment
- [x] Comparison on the real test split, including CPU fine-tuning
- [x] Selection: `tfidf_linearsvc` (macro-F1, latency, size)
- [x] Trade-off in `reports/model_comparison.md`

### M3 – Artefact, contract, local deployment

- [x] Training and inference separated, shared `predict_proba`
- [x] Contract: exactly text or URL, response with label, scores, model version
- [x] FastAPI locally, valid and invalid cases
- [x] README reflects the measured state
- [x] Executive pitch under `docs/executive-pitch/`

## Open points

- Error costs (false alarm vs. missed negative text) are still open.
- Judgemental or loaded language remains a later goal, without labels or a model.
- The GPU run from the Colab notebook (DistilBERT, three epochs, full train split) is described but not measured in this state.
- The second iteration is measured and lives under `second_path/`. Its training needs CUDA. Without a GPU, load the already recorded runs into MLflow with `python -m src.import_recorded`.
- R2/R4 improve the HTML fixtures but are not the default chosen by the time rule (R3).
- The training domain is financial sentences. A general news article can be off.
- 0.7202 and 0.7187 come from different tests. The shared comparison is `reports/development/branch_comparison.md`.

## Team

| Name | Contact | Role |
| --- | --- | --- |
| Teofil Wetzel | twetzel@stud.hs-heilbronn.de | AI Operations, HHN |
| Nick Fronius | nfronius@stud.hs-heilbronn.de | AI Operations, HHN |
