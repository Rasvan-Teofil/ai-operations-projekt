# Data and model licences

Reviewed **7 November 2026**. The website's [Data & licences page](http://127.0.0.1:8000/licenses) reads [inventory.json](inventory.json) through `/licenses/inventory`. This is the shared record for currently used sources and proposed replacements; it is not a blanket commercial release statement.

The current Qwen zero-shot weights have an Apache-2.0 licence. Historical MiniLM and reconstructed LinearSVC use training data with non-commercial or unresolved permissions. We retain them as research artifacts pending clarification. A model's base licence does not resolve a separate training-data question. This is our conservative release policy, not a claim that every dataset condition automatically transfers to weights.

The shortlist and comparison plan are in [the next-stage review](../docs/data-sources-next-stage.md). **Wikinews now supplies a completed AI-labelled pilot**, with [new measured results](../reports/common_data_v3/comparison.md). GerNTSA and SENTiVENT are **not used yet**. No full source inputs or weights are included here; the source manifest retains bibliographic titles, attribution URLs and hashes. Publishing aggregate results, distributing source texts, using data commercially and distributing trained weights require separate scope checks.

The follow-up [expanded study](../reports/expanded_news_v4/comparison.md) adds448 training texts, a fresh227-item test and18 Train-only CV runs. Its [attribution notice](notices/wikinews-ai-v4-ATTRIBUTION.txt) and inventory entries retain the same Wikinews/annotation scope. All four expanded models are experimental selectable options. Earlier notices and restricted-model records remain preserved.

The [matched-task LLM follow-up](../reports/llm_adaptation_v5/comparison.md) adds three experimental Qwen profiles: news rubric, six Train examples and own LoRA. It reuses the same 692 Train inputs and has a different fresh Test. The [v5 attribution notice](notices/wikinews-llm-v5-ATTRIBUTION.txt) retains base, Train/Test source and annotation terms. Sharing base weights does not make an adapter a standalone LLM.

## Retained notices and evidence

| File | Scope |
|---|---|
| [qwen-Apache-2.0.txt](notices/qwen-Apache-2.0.txt) | Exact Qwen licence, including Alibaba Cloud copyright notice |
| [Apache-2.0.txt](notices/Apache-2.0.txt) | Standard Apache text; MiniLM card declares this licence |
| [CC-BY-4.0.txt](notices/CC-BY-4.0.txt) | Standard legal text copied verbatim from pinned GerNTSA; also explains declared OPUS-MT, Wikinews EN and SENTiVENT annotation terms, without extending their scope |
| [CC-BY-2.5.txt](notices/CC-BY-2.5.txt) | Official CC legal-code text extracted from HTML; relevant Wikinews text |
| [CC-BY-NC-SA-3.0.txt](notices/CC-BY-NC-SA-3.0.txt) | Official CC legal-code text extracted from HTML; Financial PhraseBank declaration |
| [sentfin-MIT.txt](notices/sentfin-MIT.txt) | Exact SEntFiN repository licence; publisher text clearance not established |
| [sentivent-scope.txt](notices/sentivent-scope.txt), [NOTICE](notices/sentivent-NOTICE.md) | Exact upstream scope and credits; third-party articles expressly excluded |
| [kenpache-usage.txt](notices/kenpache-usage.txt), [wikinews-terms.txt](notices/wikinews-terms.txt) | Project-authored summaries linking the authoritative policies; not substitute grants |
| [wikinews-ai-v3-ATTRIBUTION.txt](notices/wikinews-ai-v3-ATTRIBUTION.txt) | Actual new corpus credits, transformations and annotation provenance; per-page revisions in the source manifest |
| [CC0-1.0.txt](notices/CC0-1.0.txt) | New AI annotation metadata offered under CC0 to the extent project rights apply; underlying Wikinews text retains its own CC BY terms |

[notice_manifest.json](notice_manifest.json) records upstream URLs, retrieval date, transformations and SHA-256 hashes of retained files; extracted CC texts also record original response hashes. [source_checks.json](source_checks.json) records pinned model/dataset-card URLs and hashes. [gerntsa_audit.json](gerntsa_audit.json) contains only structural counts and a source hash; reviewed snippets stay in ignored `data/raw/license_review/`.

Attribution names, exact model/data revisions, restrictions and intended uses are in the inventory. For future corpus releases, retain **per-row** page revision, author/history credit, licence, original URL and transformations. Preserve applicable notices and indicate filtering, annotation and translation. Images, external linked articles and unverified third-party quotations are not covered by a surrounding page's text licence.

## Scope and maintenance

This directory covers dataset/model sources, not a complete dependency SBOM. Installed dependencies keep their own licences. No licence is assigned here to this repository's own code or the colleague's original supplement. Public GitHub access alone does not provide such a grant.

When replacing data or a serving artifact, update the inventory, notices/evidence, README and corresponding MLflow provenance together. Keep earlier version records in Git. Only update the website's measured comparison and default after the new protocol's final evaluation. Re-run `python -m pytest -q`; the licence tests require no network, data download or model training.
