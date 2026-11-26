<!-- English draft of docs/project-overview.md on main (state 2c612d8). Drop-in replacement: same links, same structure. -->
# Project Overview for the Presentation

**AI Operations · Hochschule Heilbronn · Winter Semester 2026/27**

The project supports media monitoring: enter the full text of a news article
or an article URL and classify the reported development as negative, neutral
or positive. Target groups are Newsrooms, PR & Communications, and Analysts.
A model score is not a calibrated confidence and not an automated decision
about people.

## What the current main branch contains

- Native German, English and **Japanese** full articles; three shared classes.
- Four methods compared on the same test set: frozen MiniLM with a head,
  our own MiniLM fine-tuning, a Qwen news rubric, our own Qwen LoRA.
- 240 training articles, 27 train-internal development evaluations, four fixed
  full training fits and a fresh test set of 691 articles.
- Qwen LoRA was selected before the test set was collected; unchanged references
  and recipes, no model selection after the fact.
- FastAPI, browser UI, model comparison, sentence highlighting, source and
  license pages, reproduction and offline contract tests.

## Presentation path without switching branches

1. [README](../README.md): problem, status and language overview;
   trace [data and original sources](data/README.md) directly.
2. [Japanese / 日本語](languages/japanese.md): 91 train, 300 test and separately
   measured results; no translated test texts.
3. [Results](../reports/next_stage/comparison-v1.md): main metric, uncertainty,
   latency, memory and limitations. Do not mix different historical test sets.
4. [Structure](repository-structure.md): data/models local, code and HTTP contract
   separated, measurement evidence under `reports/`, methods under `docs/`.
5. [Reproduction](reproduce-full-article.md) and [licenses](../licenses/README.md).
6. [Demo guide](presentation-guide.md) for the locally running application.

## History preserved

The first iteration and the early second iteration of the former main branch
are kept in full in the [main archive](archive/main-before-full-article/README.md),
including code, README, pitch, template files and the CI of that time.
Its 168 versioned files are checked unchanged via hashes.
The later v1–v5 and full-article evidence is kept in the regular report folders.
The Git histories of both development paths remain part of the integration.

`second-path` in old logs and in the MLflow experiment is a historical
identifier. The current application and its evidence now live on `main`.

## What is still open

Public, free hosting without the private tower has not happened yet.
[Hosting status](public-hosting.md). The data has since been [reviewed by the project](data/project-review.md);
240 training articles are still not a proven optimum.
A freely accessible service also needs suitable operating limits and
source/license notices. The data migration is not a blanket legal guarantee.
