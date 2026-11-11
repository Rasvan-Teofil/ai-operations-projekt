# Post-pitch plan (draft, 2026-11-10)

Status: draft for Teo and Nick. Dates come from the Canva roadmap (`DAHZaEaLHcQ`) and are not yet confirmed by the lecturer. Pitch submission status is still open (see `pitch-submission-checklist.md`).

## Roadmap items from the deck

| Deck item | Date (deck) | What exists today | Gap |
| --- | --- | --- | --- |
| General news data + error costs | next | `docs/error-costs.md` draft; threshold script `src/second_path/threshold_curve.py` | no labelled general-news set; curve not run; cost ratio unconfirmed |
| Mock deployment + monitoring | 14 Jan | FastAPI `/predict`, Dockerfile with baked TF-IDF LinearSVC (PR #3), CI with 2 jobs | `docker build` unverified; monitoring only as prototype log (`app/monitoring.py`, on `daily-refinement`); MLOps phases "registry" and "monitoring" open |
| Final presentation + bias score (optional) | 21 Jan | Canva deck, LaTeX pitch | final format and date not confirmed |

## Proposed work packages

1. **General-news evaluation set.** Small hand-labelled set of non-financial news (DE + EN), labelled by both team members, agreement reported. Used only for evaluation, not training. Size and sources to agree with Nick.
2. **Error costs.** Run `python -m src.second_path.threshold_curve` where the fine-tuned model lives, pick the margin on validation only, report test once. Confirm the cost ratio with one newsroom volume example.
3. **Mock deployment.** One verified `docker build` / `docker run` plus `/health` and `/predict` screenshots. Model version visible in every response (already returned as `model_version`).
4. **Monitoring (minimal).** Prototype on `daily-refinement` since 2026-11-11 (`PREDICTION_LOG_PATH`, metadata only). Still open: daily report and drift check. Log each request's label, scores, `model_version`, text length and latency to a local file; one small report on label distribution and latency per day. Drift check: compare the daily label distribution against the test split. Keep it local, no external services.
5. **Final presentation.** Update deck numbers only from in-repo sources; keep run 1 vs. run 2 story and MLflow evidence as the backbone.

## Open questions for Teo

- Are 14 Jan and 21 Jan the real dates from the lecture?
- Who labels the general-news set, and how many texts are realistic?
- Is the optional bias score still in scope?
