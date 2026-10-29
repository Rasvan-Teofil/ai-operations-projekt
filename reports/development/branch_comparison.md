# Final comparison of the first and second iteration

**Run only after the ablations, the selection and the full release training were finished.** The external evaluation changed neither the selection nor the models.

| Solution | macro-F1 EN native | macro-F1 DE translated | macro-F1 combined (secondary) | ms/text | MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| colleague_main_default | 0.4524 | 0.4293 | 0.4410 | 0.250 | 4.08 |
| second_path_v1 | 0.7360 | 0.6458 | 0.6909 | 2.335 | 465.14 |
| second_path_v2 | 0.7293 | 0.6557 | 0.6925 | 2.326 | 240.73 |

`colleague_main_default` is our first iteration (TF-IDF + LinearSVC, commit `a840bb7`), reconstructed. The name is the key in the JSON/CSV files.

![Shared external comparison](branch_comparison.png)

On the 600 native English headlines, v2 is +0.2770 macro-F1 ahead of our reconstructed first iteration (LinearSVC). A paired bootstrap over these 600 already predicted headlines (1,000 resamples, seed 20261019) gives a 95 % percentile interval for the difference of [0.2257, 0.3249]. No additional model predictions or model selection via bootstrap.

**No general accuracy gain of v2 over v1 is claimed:** English moves from 0.7360 to 0.7293; translated German from 0.6458 to 0.6557. Artefact size moves from 465.1 to 240.7 MiB. The English v1/v2 difference is not called significant here.

Our first iteration (LinearSVC) is considerably smaller and faster on CPU. Our encoder uses CUDA and, despite BF16, is larger and slower. For the given quality goal this external benchmark favours the encoder; for tight CPU/memory budgets LinearSVC remains a clear trade-off. This is not a general ranking for other news domains.

## Provenance, filters, limits

[SEntFiN](https://github.com/pyRis/SEntFiN), revision `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`, original repo under [MIT](https://github.com/pyRis/SEntFiN/blob/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE); Sinha, Kedas, Kumar, Malo. Of 10,753 English headlines, 9,512 remain with one consistent sentiment for all mentioned entities and a valid length. After removing normalised duplicates and checking against both training corpora, 9,453 remain; no exact normalised training overlap was found. 200 negative, 200 neutral and 200 positive headlines are drawn without model predictions, seed 20261019.

The English labels come from human entity annotation. They are transferred to the whole headline only when all entity labels agree; this remains an operationalisation, not a new annotation. The 600 German texts are translated from exactly these headlines with [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de), CC-BY-4.0, revision `6183067f769a302e3861815543b9f312c71b0ca4` (beam 4, no sampling, context 256, max. 128 new tokens). No English source text was truncated by the context. German labels are carried over: translation errors and shifted tone were not checked independently. **DE is therefore an MT transfer check, not a native German gold test.**

The benchmark is class-balanced, consists of headlines and has 600 distinct headline pairs. It represents neither natural news frequencies nor fully labelled articles. Semantic event overlap and pre-training contamination are unknown. The old test score 0.720 and our old own score 0.7187 are not used for ranking.

## Unchanged reference, no retrofitted run on main

The reference code lives only in a separate detached checkout at `a840bb7bc8f1656f1700ccabaa89837ce347ca2b`. The adapter imports `new_linearsvc` and the original `prepare()` function. It copies no model implementation into second-path and never calls `src.train` or the original MLflow experiment. Reference training data: 5,923 rows from Phrasebank EN and Kenpache DE; pandas 3.0.6, scikit-learn 1.9.1 and joblib 1.6.0 as in its requirements. Our models train on 8,335 Kenpache DE/EN rows. The comparison therefore does not separate the effect of the architecture from the choice of training data.

The trained weights of the first iteration were not available. The result is a reconstruction of the delivered default procedure with unchanged code and a re-created training split; bit-identity with the original artefact or test split is not claimed. Code, data and artefact hashes are in `peer_provenance.json`, the reference dependencies in `reference_requirements.txt`.

All three new shared evaluations live only in the experiment `article-sentiment-second-path`, tag `phase=final-branch-comparison`. The main-branch reference and its MLflow directory stay unchanged.

## Decisions by topic

| Topic | Observation / decision |
| --- | --- |
| Quality of evidence | Source-separated train CV plus an external audit instead of another look at the old test. No evidence of general article quality. |
| Training weights | Language×class changes the CV scores only slightly; no robust large accuracy gain. |
| Article processing | R2/R4 pass six instead of two HTML fixtures; a technical improvement, not evidence from human-labelled articles. Not part of the R3 default chosen by the latency tie rule. |
| Operations | R3/BF16 takes about 241 instead of 465 MiB; bounded waiting and a finite-score check prevent two concrete operational failures. Small latency differences are not a significance claim. |
| Combination / selection | Highest CV point estimate: R4_combined with 0.6740. Chosen by the pre-declared tie/latency rule: R3_deploy with 0.6729 and 2.381 ms. No after-the-fact redefinition to ship everything combined. |
| First vs. second iteration | On this shared native EN benchmark, better encoder quality vs. a much smaller/faster CPU default. Other data, costs and language domains can change the decision. |

## Reproduce from scratch (only after the v2 selection)

In a fresh clone, after the v2 commands in the README. The v1 training call only produces the comparison state, not an old test evaluation. On the current tower the artefacts already exist and are not overwritten. New reports stay separate from the historical ones.

```bash
export SENTIMENT_DEV_REPORTS=reports/reproduction
python -m pip install -r requirements-evaluation.txt
python -m second_path.train finetuned
git clone --no-checkout https://github.com/Rasvan-Teofil/ai-operations-projekt.git ../colleague-reference
git -C ../colleague-reference checkout --detach a840bb7bc8f1656f1700ccabaa89837ce347ca2b
python3.12 -m venv ../colleague-reference/.venv
../colleague-reference/.venv/bin/python -m pip install -r reports/development/reference_requirements.txt
../colleague-reference/.venv/bin/python second_path/peer_reference.py prepare --checkout ../colleague-reference --report reports/reproduction/peer_provenance.json
python -m second_path.branch_benchmark prepare
python -m second_path.branch_benchmark audit
python -m second_path.development_report
```

The adapter pins reference data to the logged HF snapshots; the original filter/split/model implementation stays untouched. The benchmark has an exclusive local audit lock. Original test/val files are not used as evaluation data by our development.
