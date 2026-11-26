# Translation Checklist for `main` (German → English)

Status: draft, 2026-11-24. Based on `origin/main` at `2c612d8`.
Nothing translated yet. Waiting for Teo's OK (and coordination with Nick, who owns most of these files).

Rule: translate originals in place, no parallel English copies. German example sentences for model tests, German article titles/quotes in data, and official institution names stay.

Method: files ranked by count of common German function words (heuristic, `git show origin/main:<file>`). Counts in brackets.

## A. Prose docs (translate by hand)

Priority 1, examiner reads these first:
- [ ] `README.md` (43) — draft ready: `docs/drafts/README.en-draft.md`
- [ ] `docs/project-overview.md` (31) — draft ready: `docs/drafts/project-overview.en-draft.md`
- [ ] `docs/presentation-guide.md` (25)
- [ ] `docs/reproduce-full-article.md` (55)
- [ ] `docs/next-stage.md` (171)
- [ ] `docs/repository-structure.md` (39)
- [ ] `docs/README.md` (12)

Priority 2:
- [ ] `CONTRIBUTING.md` (28)
- [ ] `docs/public-hosting.md` (32)
- [ ] `docs/languages/japanese.md` (31)
- [ ] `data/README.md` (18)
- [ ] `src/README.md` (10)
- [ ] `reports/README.md` (8)
- [ ] `reports/next_stage/README.md` (4)
- [ ] `requirements/README.md` (6)

Archive (low priority, could stay German with a one-line English note on top):
- [ ] `docs/archive/next-stage-before-final-release.md` (66)
- [ ] `docs/archive/project-history-through-v5.md` (61)
- [ ] `docs/archive/main-before-full-article/README.md` (14)

## B. Generated files (translate the generator strings, then regenerate)

Do not edit the output by hand, it would be overwritten.

| Generator | German outputs |
|---|---|
| `scripts/build_source_catalog.py` (37) | `docs/data/README.md`, `docs/data/{version}-{language}.md` (e.g. `v4-de.md` 164, `test-de.md` 144, `v3-de.md` 89, `v5-de.md` 68, `train-de.md` 45, `excluded-test.md` 50, `project-review.md` 24; the `-en`/`-ja` pages share the same German template text) |
| `src/development_report.py` (28) | `reports/development/branch_comparison.md` (20), `reports/development/development.md` (8) |
| `src/next_stage_report.py` (11) | `reports/next_stage/development-summary.md` (11) |
| `src/next_stage_test_report.py` (8) | `reports/next_stage/comparison-v1.md` (10), parts of `error-analysis-v1.json` |
| unclear, check | `reports/next_stage/model-cards-v1.md` (15), `reports/model_comparison.md` (7) |

Check after regeneration: metrics and hashes unchanged (`git diff --stat` only text lines), tests pass (`pytest`).

## C. Probably keep (data, not documentation)

- `reports/*/source_manifest.json`, `reports/next_stage/final-test-v1/group-review.json`, `reports/next_stage/pilot/group-review-v1.json`: German article titles, incident names, review notes. Titles stay. Free-text review notes: decide with Nick (changing them may break provenance hashes).
- `licenses/notices/wikinews-full-article-v1-ATTRIBUTION.txt`: licence/attribution text of German sources, keep.
- `docs/archive/main-before-full-article/source/**`: frozen archive code, keep.

## Open questions

- Who translates what (Nick wrote most of main)?
- One PR per block (A1, A2, B) is easier to review than one large PR.
