# Executive Pitch – submission checklist (as of 2026-11-06)

Pitch week: 2026-11-02 … 2026-11-08 (ended Sun 8 Nov). As of 2026-11-09: submission status unconfirmed (was it submitted, which format?).

## Ready (repo / LaTeX)

- [x] English artifacts; stakeholders = Newsrooms, PR & Communications, Analysts
- [x] Improvement story: run 1 → run 2 on shared SEntFiN (EN 0.4524 → 0.7293)
- [x] MLflow evidence + screenshots in `docs/mlflow/`, figure in pitch PDF
- [x] Error-cost draft in `docs/error-costs.md` and in pitch “Next steps”
- [x] Docker bake on `main` (PR #3): TF-IDF LinearSVC baked into the API image; local `docker build` still unverified on the box
- [x] Official layout on `main` (PR #4): second iteration lives under `src/second_path` (API `app/second_path.py`)
- [x] Demo serving model on `main` (PR #5): `tfidf-linearsvc-v2` (balanced + opinion supplement); financial-test macro-F1 **0.7330** (README). Recorded five-model table and pitch still cite v1 **0.7202** — leave until Teo asks to refresh pitch numbers from in-repo sources
- [x] `docs/executive-pitch/executive-pitch.pdf` rebuilt (5 pages)
- [x] Canva deck `DAHZaEaLHcQ` (verified read-only 2026-11-06): “84 tests” / “2 CI jobs” and error-rate wording (~28 % negatives missed, ~35 % false alerts) — no longer “Error costs: not measured”

## Blocked on Teo

- [ ] Confirm whether the pitch was submitted / presented, and in which format (PDF, Canva link, both). Pitch week ended Sun 8 Nov
- [ ] Roadmap “14 Jan / 21 Jan” on the Canva deck — keep or adjust (do not create a second deck)
- [ ] OK before any further Canva edits (test-count + error-cost text already done)

## Nice-to-have before submit (not blockers)

- [ ] Run `python -m src.second_path.threshold_curve` where the fine-tuned model lives; commit CSV
- [ ] One local `docker build` / `docker run` and `/health` screenshot (image bake is on `main`; box has no Docker daemon today)
- [ ] Cost-ratio assumption confirmed with a newsroom volume example
- [ ] Optional: refresh pitch / five-model table to v2 0.7330 only from README / PR #5 sources (do not invent)

## Note on test counts (2026-11-06)

- Canva now shows **84 tests / 2 CI jobs** (matches the pre-PR #5 local count used in earlier daily notes).
- PR #5 reports **93 passed, 1 skipped** under `SENTIMENT_DATA=dummy SENTIMENT_MODEL=tfidf_logreg pytest` (new opinion-supplement / excerpt coverage). Deck still says 84; update only if Teo wants.

## Sources of truth

- LaTeX pitch: `docs/executive-pitch/bericht/document.tex` → PDF beside it
- Canva (one deck only): https://canva.link/zi6m0tz17s5ejl0 (`DAHZaEaLHcQ`)
- Branch for prep work: `daily-refinement` (do not merge without asking)
- `main` tip after rebase: `8668255` (PRs #3–#5)
