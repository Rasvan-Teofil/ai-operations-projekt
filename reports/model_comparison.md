# Model comparison

Shared test split of the prepared financial news sentences (seed 42, stratified by `label|language`): **n_train = 5923**, **n_val = 1270**, **n_test = 1270**. Validation was not used for tuning. Label space: `negative`, `neutral`, `positive`. Metrics come from one MLflow experiment, `article-sentiment`.

Sources: [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) `sentences_75agree` (English, CC BY-NC-SA 3.0) and [Kenpache/multilingual-financial-sentiment](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment), German only (academic / non-commercial). Details in `data/README.md`.

## Metrics

Sorted by macro-F1 descending, then inference time. Measured on CPU, no GPU.

| model | model_version | role | accuracy | f1_macro | f1_negative | f1_neutral | f1_positive | inference_ms_per_text | size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf_linearsvc | tfidf-linearsvc-v1 | classic | 0.7575 | 0.7202 | 0.6758 | 0.8260 | 0.6587 | 0.0221 | 4.0826 |
| tfidf_logreg | tfidf-logreg-v1 | baseline | 0.7189 | 0.6562 | 0.6017 | 0.8062 | 0.5608 | 0.0264 | 4.0828 |
| finetuned | finetuned:distilbert-base-multilingual-cased | finetuned | 0.6819 | 0.6142 | 0.5946 | 0.7962 | 0.4517 | 8.3083 | 516.2315 |
| cardiffnlp | cardiffnlp/twitter-xlm-roberta-base-sentiment | pretrained_zero_shot | 0.6039 | 0.4859 | 0.4175 | 0.7165 | 0.3237 | 19.0046 | 1060.6611 |
| tabularisai | tabularisai/multilingual-sentiment-analysis | pretrained_zero_shot | 0.5409 | 0.3636 | 0.2864 | 0.6851 | 0.1193 | 8.8217 | 516.2373 |

## What ran where

- **sklearn (both):** full train split, 5,923 sentences, CPU.
- **Zero-shot (both):** no adaptation, full test split, CPU. `tabularisai` folds five classes into three (Very Negative + Negative, Neutral, Positive + Very Positive). `cardiffnlp` already has three classes. One test sentence is longer than 126 content tokens (maximum 137); all others fit in one window. At 512 tokens none is too long.
- **Fine-tuning, this run:** `distilbert-base-multilingual-cased`, CPU, **1 epoch**, learning rate 2e-5, batch 8, context **128**, **2,000 stratified training sentences** out of 5,923 (`--max-samples 2000`). Evaluated on the full test split. One test sentence (137 tokens) was chunked at inference because the training window is 128. Training time about 201 seconds.
- **Not run:** three epochs on all 5,923 sentences with context 256. That is the GPU run in `notebooks/02_finetune_colab.ipynb`. Its score is not listed because we did not measure it.

The LinearSVC scores are a softmax of the decision function, not calibrated probabilities. The argmax equals `LinearSVC.predict`. macro-F1 depends only on the label.

## Trade-off

On this test set **TF-IDF + LinearSVC** wins clearly: macro-F1 0.72 vs. 0.66 (LogReg), 0.61 (fine-tuning subset), 0.49 (Twitter-XLM-RoBERTa) and 0.36 (tabularisai). It is also the fastest and smallest model, about 0.02 ms per sentence and 4 MB.

The gap to the baseline is not a tie where latency would have to decide. LinearSVC leads in every class, including `positive`, where the transformers are weakest (tabularisai 0.12, cardiffnlp 0.32, fine-tuning subset 0.45). Neutral is easiest everywhere, which follows from the class distribution.

The zero-shot models never see the training labels. CardiffNLP is trained on tweets, not on financial sentences from an investor's view. tabularisai folds five general sentiment levels; "positive for the share price" is not the same thing. Both explain the gap better than model size alone.

Our own fine-tuning scores above both zero-shot models and below both TF-IDF models. It saw only a third of the training sentences, one epoch and the shorter context. A full GPU run may improve that. It cannot promise to close the 0.11 macro-F1 gap to LinearSVC. Until that number is measured, LinearSVC remains the evidenced choice.

For the API it also matters that LinearSVC needs no PyTorch, starts with `requirements.txt` and still answers quickly on a whole article, because TF-IDF sees the full text in one vector. The transformers chunk at 512 tokens.

## Recommendation

**Model: `tfidf_linearsvc`.** This is the API default (`SENTIMENT_MODEL` switches it). CI stays on `tfidf_logreg` so the workflow needs no extra model file and no torch; both TF-IDF models are created by `python -m src.train`.

LogReg is the alternative when better-calibrated scores matter more than the label. Re-check fine-tuning only once the Colab run has been measured on the same test split.
