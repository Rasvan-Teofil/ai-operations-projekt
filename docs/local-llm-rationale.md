# Why fine-tune first, and why test a local LLM now?

We started with supervised fine-tuning of a pretrained multilingual MiniLM encoder, not training a language model from scratch. The decision is a fixed three-class classification task for media monitoring and PR. A compact encoder lets us learn the project's labels directly, measure class-specific errors, and serve a small artifact with low latency. The original measured comparison supported fine-tuning over our frozen-encoder baseline on its own test (macro-F1 0.7187 versus 0.6656). Later source-group cross-validation selected the BF16 R3 release under our predefined quality/latency rule. Those results justified that development step; they did not establish that fine-tuning beats a prompted instruction LLM.

## Reasons for the initial approach

- **A defined learning target.** We have labelled examples and three stable outcomes. Supervised training aligns a classifier to those annotations. It still inherits their financial-domain bias and annotation mistakes.
- **Measured improvement over a real baseline.** Training, separate evaluation, class-level F1, confusion matrices and MLflow make the learning process inspectable. This also addresses the course requirement to demonstrate our own fine-tuning. Course compliance alone is not a production model-selection argument.
- **A small operating footprint.** The selected historical MiniLM artifact is about 241 MiB. Its original latency measurements were a few milliseconds per headline on this tower. A general 4B language model has a much larger weight footprint. The new shared comparison measures the actual trade-off; historical times are not silently reused.
- **A narrow response contract.** A classifier returns three scores directly. An LLM needs a fixed prompt and output constraints. We implement those constraints instead of treating generated prose or self-reported percentages as reliable evidence.
- **Control and reproducibility.** Seeds, data revisions, splits, artifact hashes and environment versions are recorded. This is possible for both approaches; it is not exclusive to fine-tuning.

## The professor's alternative is valid

A capable instruction LLM may classify new wording and unseen topics better without a new training run. It can also be hosted locally. Therefore data privacy, independence from a paid inference API and offline inference do **not** distinguish our small classifier from a local LLM. They distinguish local deployment from a remote inference service. We should not claim the LLM is worse, more expensive per useful decision, or unnecessary before measuring it.

The experiment uses [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), a text-only 4B instruction model with multilingual capabilities, Apache-2.0 licensing and no thinking mode. It is compatible with the installed CUDA PyTorch and Transformers stack. Its unquantized BF16 weights are a plausible fit for the tower's 16 GB GPU, leaving room for our bounded input. This choice avoids a quantization dependency and a gated initial download. It is a representative local-LLM experiment, not a claim to have searched all LLMs. [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3) is another valid multilingual local alternative; image support is not needed for this text task.

We use one fixed zero-shot sentiment prompt, with the article in a separate user message. A/B/C map to negative/neutral/positive. The next-token logits for those three choices are normalized to produce the API scores; we do not ask the LLM to invent confidence values. This is constrained LLM inference, not a separately trained classification head. The scores are uncalibrated. Article instructions are treated as data, but prompt separation is not a guarantee against every adversarial input.

## How the comparison answers the question

The frozen protocol is `reports/local_llm/protocol.json`. We compare the unchanged MiniLM v2 release, the pinned current-main LinearSVC v2 recipe reconstructed on its training data plus the 46-sentence supplement, and the fixed-prompt Qwen model. We do not re-evaluate the old project tests or optimize the prompt on the new benchmark.

A fresh sample from the pinned SEntFiN corpus excludes the previous 600 benchmark headlines and normalized overlaps with the available original project splits and supplement. All three candidates see the exact same new 600 native English headlines and their 600 German translations. Primary selection uses native English macro-F1 because those labels have human provenance. German translations are a transfer diagnostic, not native German gold. Paired bootstrap intervals use cached predictions, not additional model calls. Different training data and unknown public-corpus pretraining exposure remain confounders.

The selection rule is frozen before inference: highest native English macro-F1; within 0.005 absolute of the best, prefer lower warm batch-one latency, then smaller artifact. Accuracy, per-class F1, negative precision/recall, confusion matrices, GPU allocation and model size remain visible in MLflow. Results do not establish performance on long general-news articles or a real newsroom's class prevalence.

## What local availability means

After the one-time pinned download, serving loads only files under `models/` with `local_files_only=True`. Text classification needs no cloud inference service or account. Fetching a user-supplied article URL still needs the network. GPU failures, lack of memory, power loss, disk loss and process crashes can still make either local system unavailable; the API reports 503 rather than promising it is always available. Keep local weight backups and a reproducible setup. A system service/restart policy is separate operational work.

## A defensible answer at the presentation

“We first fine-tuned a small pretrained model because our task had labelled examples, a fixed three-class target and a need for low-latency, reproducible inference. It improved over our measured baseline. That was a justified first experiment, not proof that a general LLM would be worse. We have now added a fixed-prompt local LLM to the same evaluation protocol and choose the deployed model using measured classification quality, latency and size.”
