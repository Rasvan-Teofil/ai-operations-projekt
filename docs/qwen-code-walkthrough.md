# Qwen and best-model code walkthrough

This is the recommended English walkthrough for explaining the selected model,
the Qwen alternatives and their FastAPI integration. It focuses on design and
data flow rather than explaining every line.

## One-minute summary

> We use one pinned Qwen3 4B Instruct base model in two conditions. Qwen Zero
> uses a fixed news-classification rubric without task-specific weight updates.
> Qwen LoRA uses the same base, prompt and input policy, but trains small
> low-rank adapters on the 240 Train articles. Both conditions turn Qwen into a
> constrained three-class classifier by reading the next-token logits for the
> single tokens A, B and C. Qwen LoRA was selected on Train-internal
> development results before the final test and is therefore our research
> default.

The label mapping is defined in `src/config.py`:

| Token | Project label |
|---|---|
| `A` | `negative` |
| `B` | `neutral` |
| `C` | `positive` |

The three returned values are a softmax over only these three token logits.
They are useful normalized model scores, but they are not calibrated
probabilities.

## Recommended presentation route

For a short weekly code presentation, use this order:

1. State the two Qwen conditions and the selected model without opening code.
2. Open `src/next_stage_experiment.py` and show `PROMPT`, `lora()`,
   `fit_lora()` and `qwen_profile()`.
3. Open `src/next_stage_protocol.py` and explain that candidate selection uses
   Train-internal out-of-fold results, not the final test.
4. Open `src/local_llm.py` and follow one request through `messages()` and
   `predict_proba()` to the constrained A/B/C logits.
5. Open `app/main.py` and show how the common predictor interface reaches
   `/predict` and `/analyze`.
6. Finish with the result table and the language-specific limitation: the LoRA
   gain is a DE/EN result, not a gain in every language.

This route explains the architecture and the important design decisions without
walking through every helper or every training loop statement.

## Code map

| Question | File and main symbols |
|---|---|
| What task does Qwen receive? | `src/next_stage_experiment.py`: `PROMPT`, `qwen_tokens()` |
| How is LoRA configured? | `src/next_stage_experiment.py`: `lora()` |
| How is the adapter trained? | `src/next_stage_experiment.py`: `fit_lora()` |
| How are Zero and LoRA artifacts described? | `src/next_stage_experiment.py`: `qwen_profile()` |
| How was the candidate chosen? | `src/next_stage_protocol.py`: `primary_score()`, `select_candidate()` |
| How were candidates frozen and tested? | `src/next_stage_final.py`: `declaration()`, `train()`, `one_test_pass()`, `evaluate()` |
| How is Qwen loaded for inference? | `src/local_llm.py`: `LocalEngine`, `shared_engine()`, `LocalLLMPredictor` |
| How does the API select an artifact? | `app/main.py`: `candidate_paths()`, `load_predictor()`, `available_predictor()` |
| What is the HTTP contract? | `app/schemas.py`: `PredictRequest`, `Prediction`, `Analysis` |

## Training path: Qwen LoRA

1. `qwen_tokens()` constructs a system message containing the frozen
   classification rubric and a user message containing the complete article.
2. Articles above 4096 article tokens are rejected. The project does not remove
   the middle of a long article for the current full-article study. The complete
   prompt must remain within 4608 sequence tokens.
3. `lora()` loads the pinned local Qwen base in BF16 and attaches rank-8 LoRA
   adapters to the attention `q_proj` and `v_proj` modules. The base weights
   remain frozen.
4. `fit_lora()` trains the adapter for three epochs with learning rate `1e-4`,
   weight decay `0.01`, class-weighted cross-entropy and effective batches of
   16 articles. Only the A/B/C next-token logits enter the classification loss.
5. `qwen_profile()` writes a small serving profile. It points to the shared
   local base and, for the LoRA condition, to the trained adapter. The base
   weights are not duplicated.

The final adapter contains 2,949,120 trainable parameters. The recorded full
standalone size still includes the required Qwen base, because a fair deployment
comparison must count everything needed to run the model.

## Zero-shot path

Qwen Zero uses the same pinned base, the same prompt, the same 4096-token
article policy and the same A/B/C scoring method. It does not load an adapter
and performs no task-specific training. This makes it the clean control for the
effect of LoRA adaptation.

In `LocalEngine.logits()`, an unadapted request explicitly disables any loaded
adapter. This matters because the API shares one base model in memory. A lock
serializes adapter state changes so concurrent requests cannot accidentally use
the wrong model condition.

## Inference path

`LocalLLMPredictor` in `src/local_llm.py` performs the serving work:

1. It requires CUDA with BF16 and refuses a silent CPU fallback.
2. It loads only local files; the API does not download weights or train.
3. `messages()` creates the exact chat prompt and enforces the article policy.
4. `predict_proba()` renders the chat template, runs one forward pass and takes
   the logits of A, B and C at the next-token position.
5. A softmax converts those three logits into the score vector returned to the
   common serving layer.

The common API layer validates that every model returns exactly three finite,
non-negative scores summing to one. It then maps the largest score to the public
label.

## Why Qwen LoRA is the selected model

Selection happened before the final test. Inside the 240 Train articles, the
project used grouped three-fold out-of-fold development predictions. The
primary metric is:

```text
(German three-class macro-F1 + English three-class macro-F1) / 2
```

Candidates within 0.005 of the best development score would be resolved by
lower warm inference latency and then lower complete artifact size. The
predeclared stability rule added a second LoRA seed because the top development
conditions were close. The candidate declaration then froze Qwen LoRA before
fresh test collection and inference.

On the fixed final test:

| Model | DE/EN macro-F1 | DE | EN | JA | Warm ms/article | Complete MiB |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Zero | 0.7796 | 0.7508 | 0.8085 | 0.8143 | 172.33 | 7687.49 |
| Qwen LoRA | 0.8218 | 0.7612 | 0.8825 | 0.8132 | 184.26 | 7698.77 |

The correct conclusion is not that LoRA improves every language. It improves
the predeclared DE/EN primary metric, mainly through English, while Japanese is
essentially unchanged and marginally lower.

## FastAPI path

`app/main.py` keeps the HTTP layer separate from model training:

1. `load_predictor()` reads the local selected-artifact metadata.
2. `available_predictor()` lazily loads an explicitly requested model and
   protects model loading with the same bounded inference semaphore.
3. `POST /predict` accepts exactly one of `text` or `url` and returns the article
   label, three scores, the exact model version and the input policy.
4. `POST /analyze` additionally classifies sentence spans for the visual
   highlighting. These sentence labels are a separate feature, not a causal
   explanation of the article prediction.
5. `/health` and `/models` distinguish unavailable local weights from a loaded
   predictor. Missing weights produce an explicit HTTP 503 instead of a silent
   alternative model.

`app/schemas.py` enforces the public contract with Pydantic. URL fetching and
article extraction are model-independent and apply the same safety and content
policy before any selected model receives the text.

## Questions to be ready for

**Why not generate a textual answer?**  
Generation would add parsing variability. Restricting the next token to A/B/C
creates one deterministic three-class interface shared with the smaller models.

**Why LoRA instead of full Qwen fine-tuning?**  
LoRA trains a small adapter while retaining the frozen base. It fits the local
GPU budget and creates a separately versioned artifact without duplicating the
base model.

**Why use macro-F1?**  
The classes and languages are imbalanced. Macro-F1 gives each class equal weight
within a language; the primary score then gives German and English equal weight.

**Why is Japanese not part of the primary score?**  
The DE/EN metric was frozen first. Japanese was added as a separately reported
native-language extension, so adding it to the main metric after observing the
results would change the selection rule retrospectively.

**Can a clean Git clone run the selected model?**  
Not by itself. Git contains code, contracts and evidence, while raw text,
weights, adapters and the MLflow database remain local. With the verified local
artifacts present, the same FastAPI application serves the selected model.

**Are the scores confidence values?**  
No. They are normalized A/B/C logits and are explicitly documented as
uncalibrated model scores.
