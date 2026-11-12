# Why repeat the comparison with an adapted LLM?

The data-expansion study answered whether more training coverage helped the small
classifiers. Its unchanged Qwen comparator still had a financial-news prompt and
no access to demonstrations or task-specific weight updates. That was a useful
zero-shot deployment baseline, but insufficient evidence for a general claim that
the small architecture was better. The user asked us to address this mismatch.

The next experiment gives Qwen the same outcome definitions and access to the
same training labels, including its own supervised adaptation. An LLM win is a
hypothesis, not a requirement. Every prespecified condition is retained.

Neither approach trains a foundation model from scratch: MiniLM adapts a pretrained encoder; Qwen LoRA adapts a pretrained generative LLM. The comparison concerns task adaptation and deployment trade-offs.

## Why begin with a small trained classifier?

A compact multilingual encoder was a reasonable first hypothesis for a fixed
three-class endpoint: cheap batch-one inference, small local artifacts, explicit
class weighting and a reproducible supervised baseline. It also let us test the
course requirements—data provenance, a split, own GPU training, MLflow and a
versioned API—within a bounded experiment. These are engineering reasons to try
it, not evidence that it must outperform an LLM. Both approaches can be hosted
locally; availability alone does not favor one.

The historical method remains useful even if the adapted LLM wins: it measures
the latency/storage/label-use trade-off and records a negative or positive result
without erasing the path. The missing step in v4 was task-matched LLM adaptation.
This follow-up supplies that step rather than inventing a justification after
choosing a preferred architecture.

## What each condition isolates

| Pair | Change | What stays fixed |
|---|---|---|
| Original Qwen → news-rubric Qwen | Whole task instruction adapted to general news, including explicit outcome/neutral/sports/discovery rules | Same pinned Qwen weights, precision, article policy and constrained A/B/C scoring |
| News-rubric → six-example Qwen | Six seeded demonstrations: one per language/class, from fit rows only | Same base weights and news rubric |
| News-rubric → LoRA Qwen | Own supervised rank-8 q/v adapter using every fit row for three epochs | Same base snapshot, news rubric and inference article policy |
| Adapted Qwen → existing MiniLM/SVC | Different approaches with access to the same 692 training texts/labels | Identical fresh final Test and reported metric definition |

The first pair changes more than the word “financial”. It does not establish that
removing that one word alone causes any improvement. Neither the Qwen nor MiniLM
pretraining corpus, parameter count, optimizer settings or compute budgets are
equalized. This is a practical adaptation comparison, not a pure architecture
experiment. Few-shot sees six explicit examples per request; LoRA and MiniLM fit
the full Train. These differences and their costs remain visible.

## Data and reference labels

The 692 v4 Train inputs/labels are byte-identical. All previous Val and Test inputs
remain outside fitting and example selection. Fresh Test collection uses seed
20261109, randomly selecting 120 pages per language from the same recent-1,500
published-category pool without sentiment hints or class balancing. All 1,080
previously collected Wikinews inputs and conservative linked-page/shared-source/
near-duplicate groups are protected. The first source-group audit leaves 239 new
inputs; twelve broken or substantially ambiguous annotation cases are excluded
before model results, leaving 227: 132 negative, 58 neutral, 37 positive; 109 DE,
118 EN. Its SHA-256 starts `723673ee`; the coincidentally 227-row v4 Test starts
`6f02c3a6` and is not reused.

The same assistant read each exact title+lead input, wrote a label and rationale,
and reviewed consistency before candidate results. **AI-labelled · Codex-reviewed**
is the display status. One AI annotator can introduce systematic bias; this is not
human-gold accuracy or a claim of superiority to human annotation. Exact assistant
serving revision is unavailable. Site/annotation licences, per-page history links,
transformations and hashes remain in the source/annotation records. Source-page
translation history and public-model pretraining overlap are unknown. No project
translations, external linked publisher articles or images are included.

## Development and final evaluation

The three full-size v4 source-group Train folds are reused by exact membership
(original fold seed 20261108, language:label stratification). Four fixed Qwen
conditions per fold give twelve new scored MLflow runs. Example seed is
20261109+fold, and every example ID is checked against fit/dev membership. Full-Train
examples use seed 20261109. Historical small-model full-fold FP32 scores are
clearly marked as reused references with their original run IDs; no old run is
updated. Final small-model deployment uses the existing BF16 representation of
the same trained fit.

LoRA uses training seed 20261109, rank 8 / alpha 16 / dropout 0.05 on q_proj/v_proj,
microbatch 1 with accumulation 16, AdamW LR 1e-4 / weight decay 0.01, warmup 10%, gradient
clip 1, non-reentrant gradient checkpointing, BF16 frozen base and FP32 adapter/
optimizer states. Its loss is globally inverse-class weighted cross entropy on
the next-token A/B/C logits. This is task classification adaptation, not full
free-text generation SFT. No quantization, training truncation, early stopping,
prompt sweep, hyperparameter search or epoch selection is used. Each fit row is
seen in all three epochs. The discarded longest-Train-input capacity probe
(533 tokens, 2,949,120 trainable parameters) is engineering evidence, not a scored
model run. Its measured peak allocation was 8,343,538,176 bytes.

The exact protocol and prompt were committed before the adaptation comparison.
After CV, one new LoRA uses the full Train, six final demonstrations use only
Train, and all seven final artifacts/dependencies are hash-pinned before any
fresh Test prediction. Each condition predicts each Test input once, with an
exclusive completion lock and per-item interruption checkpoints. No variant is
changed after seeing fresh Test results. Historical tests/results remain intact.

The primary descriptive ranking is mean(DE macro-F1, EN macro-F1). Within 0.005,
warm inference latency then full standalone size decide. We also report Accuracy,
pooled/per-class F1, positive/negative recall, confusion matrices, training cost,
latency, standalone size and incremental shared storage. Exploratory paired
source-group bootstrap intervals use saved predictions, not reruns, and do not
cover annotation error or adjust for multiple comparisons.

## Cost, API and reproducibility

The API is paused during GPU measurements. One RTX 5060 Ti worker uses four CPU
threads, twenty separate synthetic warmups and synchronized batch-one timing.
Tokenization and few-shot context cost are included; loading/network are excluded.
Qwen pretraining cost is excluded for all variants, not claimed to be zero.
MiniLM/SVC historical fitting costs and new LoRA fitting cost are distinguished.
An adapter’s small file size is not presented as a tiny standalone LLM: complete
size includes its pinned base dependency.

Serving only reads local model artifacts. Prompt variants and LoRA share one base
allocation. Adapter state changes are serialized; unadapted calls explicitly
disable it. Offline unit tests and before/after synthetic legacy regression checks
verify isolation without reusing held-out Test texts. `/predict` retains exactly
one text or URL, three label scores and actual model_version; highlighting and
422/403/502/503 behavior remain covered. New profiles are experimental choices;
the existing AI-only serving-default policy is retained.

Only second-path is changed. A separately named reproduction writes new ignored
models/reports and new MLflow runs; it never replaces measured evidence. Raw news,
demonstration text, model weights, venv and MLflow database are local, not Git.
The README contains Tower commands and measured results; the frozen protocol,
labels, source/split/membership hashes, run IDs and aggregate evidence are in Git.

Method sources: [official Qwen card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[PEFT LoRA](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/lora),
[PEFT adapter precision/loading guidance](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/troubleshooting).
These support implementation choices; they do not supply the project's measured
scores. See [the actual comparison](../reports/llm_adaptation_v5/comparison.md).
