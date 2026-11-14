# Course completion audit

**Scope:** AI Operations, current project state on `main`, reviewed on
14 November 2026. This document distinguishes confirmed course requirements,
implemented evidence, and work that still depends on information or local
artifacts outside Git.

## Confirmed course requirements

The archived original project plan records that the lecturer expects a
comparison of several models and an own fine-tuning. The lecture layout and the
original milestones additionally require labelled data with provenance, a
recorded split, a baseline, comparable metrics, MLflow, a versioned FastAPI
contract, tests, and a documented project hand-off.

| Requirement | Status | Evidence on `main` |
|---|---|---|
| Business problem and target classes | Complete | `README.md`, `docs/project-overview.md` |
| Data provenance, revisions, rights and exclusions | Complete | `docs/data/README.md`, `licenses/`, `reports/next_stage/final-test-v1/test-manifest.json` |
| Project-side human review | Complete with a stated limitation | `docs/data/project-review.md`; no blind per-row second annotation or agreement statistic is claimed |
| Several models on one common final test | Complete | `reports/next_stage/comparison-v1.md` |
| Own fine-tuning | Complete | Full MiniLM fine-tuning and Qwen LoRA in `src/next_stage_experiment.py` |
| Pre-test model selection | Complete | `reports/next_stage/final-candidates-v1.json` |
| MLflow tracking and immutable run identities | Complete | Run IDs in `reports/next_stage/comparison-v1.json`; tracking code in `src/tracking.py` and `src/next_stage_protocol.py` |
| FastAPI contract and browser interface | Complete locally | `app/main.py`, `app/schemas.py`, `app/static/` |
| Automated verification | Complete for the Git-visible state | 189 offline tests plus layout, integration and source-catalog checks |
| Reproduction instructions | Complete with local-artifact prerequisites | `docs/reproduce-full-article.md` |
| Public deployment | Not confirmed as a course requirement and not implemented | Current constraints in `docs/public-hosting.md` |
| Production monitoring | Not confirmed as a course requirement and not implemented | MLflow covers experiment tracking, not production monitoring |

## Result and language audit

The current comparison is suitable as the final course comparison, provided it
is presented as evidence for this fixed source population rather than as a
general production-quality guarantee.

- All four candidates use the same 240 Train articles and the same 691-article
  final test.
- The final test contains native German, English and Japanese articles. It does
  not use translated German or English test texts as Japanese evidence.
- The three output classes and their order are identical for all languages:
  `negative`, `neutral`, `positive`.
- The selection metric was fixed before the final test: the mean of German and
  English three-class macro-F1. Japanese was declared as a separate extension
  and was not added to the primary metric after seeing its results.
- Qwen LoRA was selected on Train-internal out-of-fold development results, not
  on the final test. It reaches 0.8218 on the fixed DE/EN primary metric.
- Japanese is reported honestly: Qwen Zero is marginally above Qwen LoRA there,
  so no universal per-language LoRA improvement is claimed.
- English has only 118 final-test articles, including 14 neutral references.
  Macro-F1 is the right metric for the imbalance, but the small class count
  still limits precision and transfer claims.

The human review is documented as a project-owner confirmation after the frozen
measurements. It improves the project hand-off, but it is not equivalent to an
independent blinded gold-standard annotation. The frozen files correctly retain
their original `AI-labelled · Codex-reviewed` provenance.

## Verified technical baseline

The following checks were repeated from a fresh clone on 14 November 2026:

- `python scripts/check_layout.py`: passed.
- `python scripts/check_main_integration.py`: passed.
- `python scripts/build_source_catalog.py --check`: passed for 931 selected
  articles, 165 exclusions and 15 metadata-only catalog files.
- `python -m pytest -q -p no:cacheprovider`: 189 passed.
- FastAPI started without local model weights. `/`, `/docs`, `/health`,
  `/models`, `/comparison/full-article` and `/project/data-review` were
  available. `/predict` correctly returned HTTP 503 because the ignored local
  model artifact was absent.

This proves the Git-visible API, contracts, reports and failure behaviour. A
live prediction with the selected model still requires the original local
Qwen base weights, LoRA adapter and selection metadata.

## Inputs still required for an unconditional final sign-off

1. **The lecturer's current assignment or grading rubric.** The repository
   contains the lecture layout and the team's recorded milestones, but no
   separate current rubric. Public deployment, containerisation, monitoring,
   observability or a final written report must not be invented as mandatory
   without that source.
2. **Access to the original training/serving machine or an exact artifact
   transfer.** The raw data, Qwen base weights, LoRA adapter, selected-model
   metadata and MLflow database are intentionally excluded from Git.
3. **The intended scope of the human review.** If every selected Train and Test
   row was checked individually, record that scope explicitly. The current
   evidence only supports a project-level acceptance statement.
4. **The required deployment target.** A local FastAPI demonstration is ready
   in code. A public zero-cost GPU service is not currently available, and a
   different CPU model would be a disclosed deployment variant rather than the
   selected research model.

## Definition of done after those inputs arrive

1. Map every official rubric item to a file, command and verification result.
2. On the original machine, verify the hashes of the Qwen base and adapter,
   start the API offline, and run only synthetic or new demonstration inputs.
3. Show the selected MLflow runs and their immutable run IDs without rerunning
   the final test.
4. Record a final live API smoke check for text, URL, invalid input, unavailable
   model, and model switching.
5. Decide explicitly whether public deployment and monitoring are required. If
   they are, define a separate deployment artifact and acceptance test without
   changing the frozen research result.
6. Re-run all offline checks and merge only documentation or implementation
   changes that preserve the frozen data, predictions and reports.

