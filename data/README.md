# Data

Training and test texts for article sentiment. One row is one sentence from a financial news item, not a whole article. German and English, three classes: `negative`, `neutral`, `positive`.

The raw texts and the prepared splits are **not** in Git. The repository keeps this description, the dummy sample in `sample/` and the split logic in `src/prepare_data.py`.

## Chosen datasets

| | English | German |
| --- | --- | --- |
| Hugging Face | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | [Kenpache/multilingual-financial-sentiment](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment) |
| Subset | `Sentences_75Agree.txt` (at least 75 % annotator agreement) | `language=de` only |
| Rows before filtering | 3,453 | 5,023 |
| Licence | CC BY-NC-SA 3.0 | academic / non-commercial (see below) |
| Labels | already `negative` / `neutral` / `positive` | already the same three names |

Phrasebank counts in the 75 % file: neutral 2,146, positive 887, negative 420. German in Kenpache: neutral 2,425, negative 1,392, positive 1,206.

After filtering (prepare run on snapshots `8d3fe0c` and `9d43950`): English 3,440 sentences (8 too short, 5 duplicate texts), German 5,023, together 8,463. No text appeared in both sources.

| Split | Rows | de neg / neu / pos | en neg / neu / pos |
| --- | --- | --- | --- |
| Train | 5,923 | 974 / 1,697 / 844 | 294 / 1,493 / 621 |
| Val | 1,270 | 209 / 364 / 181 | 63 / 320 / 133 |
| Test | 1,270 | 209 / 364 / 181 | 63 / 320 / 133 |

The same numbers are stored locally in `data/processed/metadata.json`. A later prepare run with the same seed writes the same proportions as long as the upstream files stay the same.

### English: Financial PhraseBank

Source of the sentences: English financial news about OMX Helsinki companies (LexisNexis). 16 annotators with a finance background labelled them from an investor's point of view: would the sentence rather move the share price up, down or not at all? Anything without economic impact is `neutral`. That is not the same question as "does the article sound friendly".

We use the file with at least 75 % agreement, not the smaller all-agree file and not the noisier 50 % file. `datasets.load_dataset("takala/financial_phrasebank", "sentences_75agree")` does not work at this revision: the dataset repo still contains `financial_phrasebank.py`, and current `datasets` versions no longer execute such scripts. `src/prepare_data.py` therefore downloads the published zip `data/FinancialPhraseBank-v1.0.zip` and reads `Sentences_75Agree.txt` (Latin-1 encoding, label after the last `@`).

Licence on the dataset card and in `License.txt` inside the zip: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Academic, non-commercial use is allowed. Commercial use: contact Pekka Malo and Ankur Sinha. The sentences are not redistributed here (ShareAlike would apply to redistributing the collection). Citation: Malo, Sinha, Korhonen, Wallenius, Takala (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, Journal of the Association for Information Science and Technology.

### German: Kenpache, multilingual financial sentiment

Short sentences from German-language financial media (Börse.de, ntv, FAZ, Handelsblatt, Süddeutsche, Tagesschau, WiWo and others, field `source`). Columns: `sentence`, `label`, `source`, `language`. Labels are already the three class names, no index mapping needed.

The dataset card says two things. At the top: academic, non-commercial research only, text and data mining under Art. 3 of the EU DSM Directive, copyright stays with the publishers, commercial use needs licences from the sources. At the bottom and in the badge: Apache-2.0. This project follows the stricter reading: academic use, no texts in Git, no commercial use. We do not add the English Kenpache subset (6,887 rows). Phrasebank is the more cleanly licensed and better annotated English source; two English labelling guidelines in one pool would weaken the comparison.

Spot checks show usable but not expert-reviewed labels: glossary rows are `neutral`, some rows are two news items glued together. This is a real weakness of the German half.

## Label space

Both sources already deliver `negative`, `neutral`, `positive`. The five-to-three mapping in `src/labels.py` (Very Negative + Negative → `negative`, and so on) is kept for the zero-shot models, not for these files.

## What the prepare step does

```bash
python -m src.prepare_data
```

1. Download the Phrasebank zip and the Kenpache CSV via the Hugging Face Hub API (cached, not committed).
2. Filter German. Drop empty labels. Drop texts under 20 characters. Drop duplicate texts within one source and then across both sources (first hit is kept).
3. Split, only here, seed `42`, strata `label|language`:
   - Test = 15 % (`train_test_split`, `stratify`).
   - From the rest, Val with a share of `0.15/0.85`, stratified again, same seed.
   - Train ≈ 70 %.
4. Write `data/processed/train.csv`, `val.csv`, `test.csv`, `metadata.json`. Columns: `text`, `label`, `language`, `source`, `dataset`.

Training, comparison and fine-tuning read these files via `load_work_splits()` and do not reshuffle them. Val is not used for model selection during fine-tuning; the comparison table is the test split. The same test set for all five models.

`SENTIMENT_DATA=dummy` (tests and CI) ignores `data/processed` and uses `data/sample/sentiment_sample.csv` with the old 80/20 split by label only.

## Limits

- The domain is financial news, not general news. Phrasebank `positive` means "good for the share price", not "friendly tone".
- One row is one sentence. The API later receives whole articles. TF-IDF sees the full text, transformers chunk at 512 tokens.
- There is no article ID. Sentences from the same piece can end up in both train and test. A split by outlet or date is not cleanly possible with these files.
- Neutral is the largest class in both sources. It is not downsampled. The selection metric is macro-F1.
- Licences: academic / non-commercial. Do not commit raw texts.
- Kenpache labels are noisier than Phrasebank. The English and German halves do not follow the same annotation guideline.

## What belongs in Git

- this file, `data/.gitkeep`, `data/sample/**`
- not: `data/processed/`, zips, crawls, full texts
