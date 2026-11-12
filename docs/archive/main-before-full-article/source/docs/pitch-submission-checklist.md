# Executive Pitch – submission checklist (as of 2026-11-05)

Pitch week: 2026-11-02 … 2026-11-08. Exact day and format still open.

## Ready (repo / LaTeX)

- [x] English artifacts; stakeholders = Newsrooms, PR & Communications, Analysts
- [x] Improvement story: run 1 → run 2 on shared SEntFiN (EN 0.4524 → 0.7293)
- [x] MLflow evidence + screenshots in `docs/mlflow/`, figure in pitch PDF
- [x] Error-cost draft in `docs/error-costs.md` and in pitch “Next steps”
- [x] Minimal `Dockerfile` for first-iteration API (artefacts not in git; build unverified)
- [x] `docs/executive-pitch/executive-pitch.pdf` rebuilt (5 pages)

## Blocked on Teo

- [ ] Confirm submission day + channel/format (PDF only? Canva link? both?)
- [ ] Canva deck `DAHZaEaLHcQ` three text fixes (do not create a second deck):
  - “51 tests” → “84 tests, 2 CI jobs” (two places)
  - “Error costs: not measured” → draft line (≈26 % of negatives missed)
  - Roadmap “14 Jan / 21 Jan” — keep or adjust
- [ ] OK to apply those deck edits without a new design copy

## Nice-to-have before submit (not blockers)

- [ ] Run `python -m src.second_path.threshold_curve` where the fine-tuned model lives; commit CSV
- [ ] One local `docker build` / `docker run` and `/health` screenshot
- [ ] Cost-ratio assumption confirmed with a newsroom volume example

## Sources of truth

- LaTeX pitch: `docs/executive-pitch/bericht/document.tex` → PDF beside it
- Canva (one deck only): https://canva.link/zi6m0tz17s5ejl0 (`DAHZaEaLHcQ`)
- Branch for prep work: `daily-refinement` (do not merge without asking)
