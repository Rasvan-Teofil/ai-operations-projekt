"""Transformer models. Torch and Transformers are imported only on load.

Checked label maps (Hugging Face config, without loading the weights):

- ``tabularisai/multilingual-sentiment-analysis``: 0 Very Negative, 1 Negative,
  2 Neutral, 3 Positive, 4 Very Positive. Folded into three classes.
- ``cardiffnlp/twitter-xlm-roberta-base-sentiment``: 0 negative, 1 neutral,
  2 positive. Stays in the three-class space.

Long texts are split into chunks of at most 512 tokens (tokenizer,
including special tokens). The chunk distributions are averaged and then
mapped to three classes.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from src.config import (
    CARDIFFNLP_MODEL_ID,
    FINETUNED_DIR,
    HF_MAX_TOKENS,
    LABELS_3,
    TABULARISAI_MODEL_ID,
)
from src.interface import SentimentModel
from src.labels import collapse_probabilities, complete_scores, content_windows

PRETRAINED_SPECS = {
    "tabularisai": {
        "model_id": TABULARISAI_MODEL_ID,
        "version": TABULARISAI_MODEL_ID,
        "role": "pretrained_zero_shot",
        "max_length": HF_MAX_TOKENS,
        "notes": "Multilingual DistilBERT, five classes folded into three, chunked at 512 tokens.",
    },
    "cardiffnlp": {
        "model_id": CARDIFFNLP_MODEL_ID,
        "version": CARDIFFNLP_MODEL_ID,
        "role": "pretrained_zero_shot",
        "max_length": HF_MAX_TOKENS,
        "notes": "XLM-RoBERTa trained on tweets, already three classes. Domain is Twitter, not news.",
    },
}


def _missing_dependency_message() -> str:
    return (
        "PyTorch or Transformers is missing. "
        "Please run `pip install -r requirements-transformer.txt`."
    )


def _id2label_names(id2label: Mapping, width: int) -> list[str]:
    names = []
    for index in range(width):
        if index in id2label:
            names.append(str(id2label[index]))
        elif str(index) in id2label:
            names.append(str(id2label[str(index)]))
        else:
            raise ValueError(f"id2label has no entry for class {index}.")
    return names


def effective_max_length(tokenizer, requested: int) -> int:
    """Upper bound per chunk: requested value, tokenizer limit and 512."""
    model_max = getattr(tokenizer, "model_max_length", requested)
    if not isinstance(model_max, int) or model_max > 100_000:
        model_max = requested
    return max(8, min(int(requested), int(model_max), HF_MAX_TOKENS))


class HuggingFaceSentimentModel(SentimentModel):
    def __init__(self, name: str, model_version: str, model_id: str, max_length: int, role: str, notes: str):
        self.name = name
        self.model_version = model_version
        self.role = role
        self.notes = notes
        self.model_id = model_id
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def load(self) -> "HuggingFaceSentimentModel":
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(_missing_dependency_message()) from exc
        import os

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()
        return self

    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        if self._model is None or self._tokenizer is None:
            self.load()
        cleaned = [str(text).strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Empty text.")
        if not cleaned:
            return []
        assert self._tokenizer is not None
        max_length = effective_max_length(self._tokenizer, self.max_length)
        special = int(self._tokenizer.num_special_tokens_to_add(pair=False))
        budget = max_length - special
        raw_ids = self._tokenizer(
            cleaned,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
        long_indexes = [index for index, ids in enumerate(raw_ids) if len(ids) > budget]
        if not long_indexes:
            return self._scores_for_short_batch(cleaned, max_length)
        scores: list[dict[str, float] | None] = [None] * len(cleaned)
        short_indexes = [index for index in range(len(cleaned)) if index not in set(long_indexes)]
        if short_indexes:
            short_scores = self._scores_for_short_batch(
                [cleaned[index] for index in short_indexes],
                max_length,
            )
            for index, row in zip(short_indexes, short_scores, strict=True):
                scores[index] = row
        for index in long_indexes:
            scores[index] = self._scores_for_text(cleaned[index])
        if any(row is None for row in scores):
            raise RuntimeError("Not every text has a prediction.")
        return [row for row in scores if row is not None]

    def _scores_for_short_batch(self, texts: list[str], max_length: int, batch_size: int = 16) -> list[dict[str, float]]:
        """One window per text. Same softmax as the single path, just batched."""
        import torch

        assert self._tokenizer is not None and self._model is not None
        names = None
        rows: list[dict[str, float]] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            chunk = texts[start : start + batch_size]
            batch = self._tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            allowed = ("input_ids", "attention_mask", "token_type_ids")
            inputs = {key: batch[key] for key in allowed if key in batch}
            with torch.no_grad():
                probabilities = torch.softmax(self._model(**inputs).logits, dim=-1).detach().cpu()
            if names is None:
                names = _id2label_names(self._model.config.id2label, int(probabilities.shape[-1]))
            for row in probabilities.tolist():
                rows.append(complete_scores(collapse_probabilities(names, row)))
            done = min(start + batch_size, total)
            if total > batch_size and (start == 0 or done == total or start % (batch_size * 20) == 0):
                print(f"{self.name}: {done}/{total}", flush=True)
        return rows

    def _scores_for_text(self, text: str) -> dict[str, float]:
        import torch

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Empty text.")
        assert self._tokenizer is not None and self._model is not None
        max_length = effective_max_length(self._tokenizer, self.max_length)
        raw_ids = self._tokenizer(cleaned, add_special_tokens=False, truncation=False)["input_ids"]
        special = int(self._tokenizer.num_special_tokens_to_add(pair=False))
        pieces = content_windows(raw_ids, max_length, special)
        batch = self._encode_windows(pieces, max_length)
        input_ids = batch["input_ids"]
        if int(input_ids.shape[-1]) > max_length:
            raise RuntimeError(f"Chunk is longer than {max_length} tokens.")
        allowed = ("input_ids", "attention_mask", "token_type_ids")
        inputs = {key: batch[key] for key in allowed if key in batch}
        with torch.no_grad():
            logits = self._model(**inputs).logits
            chunk_mean = torch.softmax(logits, dim=-1).mean(dim=0)
        names = _id2label_names(self._model.config.id2label, int(chunk_mean.shape[0]))
        collapsed = collapse_probabilities(names, chunk_mean.detach().cpu().tolist())
        return complete_scores(collapsed)

    def _encode_windows(self, pieces: list[list[int]], max_length: int):
        """Adds CLS/SEP itself. Transformers 5 no longer has ``prepare_for_model`` on every tokenizer."""
        assert self._tokenizer is not None
        cls_id = self._tokenizer.cls_token_id
        sep_id = self._tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            raise RuntimeError("Tokenizer has no CLS and SEP id, chunking is not possible.")
        needs_types = "token_type_ids" in getattr(self._tokenizer, "model_input_names", [])
        encoded = []
        for piece in pieces:
            ids = [int(cls_id), *[int(token) for token in piece], int(sep_id)]
            if len(ids) > max_length:
                ids = ids[: max_length - 1] + [int(sep_id)]
            row = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            if needs_types:
                row["token_type_ids"] = [0] * len(ids)
            encoded.append(row)
        return self._tokenizer.pad(encoded, padding=True, return_tensors="pt")

    def tracking_params(self) -> dict[str, str | int | float | bool]:
        return {
            "model_id": self.model_id,
            "max_length": self.max_length,
            "aggregation": "mean_of_chunk_probabilities",
            "label_mapping": "name_to_negative_neutral_positive",
        }

    def size_mb(self) -> float | None:
        if self._model is None:
            return None
        parameter_count = sum(parameter.numel() for parameter in self._model.parameters())
        return parameter_count * 4 / (1024 * 1024)


def load_pretrained(name: str) -> HuggingFaceSentimentModel:
    spec = PRETRAINED_SPECS[name]
    model = HuggingFaceSentimentModel(
        name=name,
        model_version=spec["version"],
        model_id=spec["model_id"],
        max_length=spec["max_length"],
        role=spec["role"],
        notes=spec["notes"],
    )
    return model.load()


def load_finetuned(directory: Path = FINETUNED_DIR) -> HuggingFaceSentimentModel:
    if not (directory / "config.json").exists():
        raise FileNotFoundError(
            "No fine-tuned model in models/finetuned. "
            "Please run `python -m src.finetune` (GPU) or copy the Colab folder here."
        )
    version = "finetuned"
    max_length = HF_MAX_TOKENS
    trained_on = None
    cap = None
    meta_path = directory / "training_meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_model = meta.get("base_model", "unknown")
        version = f"finetuned:{base_model}"
        if meta.get("smoke"):
            version += ":smoke"
        if meta.get("max_length"):
            max_length = min(HF_MAX_TOKENS, int(meta["max_length"]))
        trained_on = meta.get("n_train")
        cap = meta.get("max_samples")
    notes = "Our own fine-tuning on the training split. Inference chunks long articles."
    if cap:
        notes += (
            f" Trained on {trained_on or cap} stratified training sentences"
            " (CPU subset), evaluated on the full test split."
        )
    model = HuggingFaceSentimentModel(
        name="finetuned",
        model_version=version,
        model_id=str(directory),
        max_length=max_length,
        role="finetuned",
        notes=notes,
    )
    return model.load()


def finetuned_is_ready(directory: Path = FINETUNED_DIR) -> bool:
    return (directory / "config.json").exists()


def iter_transformer_models() -> tuple[list[HuggingFaceSentimentModel], list[str]]:
    """Loads both zero-shot models. The fine-tuned model only if its folder exists."""
    loaded = [load_pretrained(name) for name in ("tabularisai", "cardiffnlp")]
    skipped: list[str] = []
    if finetuned_is_ready():
        loaded.append(load_finetuned())
    else:
        skipped.append("finetuned")
    return loaded, skipped


def label_maps_for_docs() -> dict[str, list[str]]:
    """The checked names, in index order. Documentation and tests only."""
    return {
        TABULARISAI_MODEL_ID: [
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive",
        ],
        CARDIFFNLP_MODEL_ID: list(LABELS_3),
    }
