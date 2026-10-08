"""Transformer-Modelle. Torch und Transformers werden erst beim Laden importiert.

Geprüfte Labelkarten (Hugging-Face-config, ohne die Gewichte zu laden):

- ``tabularisai/multilingual-sentiment-analysis``: 0 Very Negative, 1 Negative,
  2 Neutral, 3 Positive, 4 Very Positive. Wird auf drei Klassen gefaltet.
- ``cardiffnlp/twitter-xlm-roberta-base-sentiment``: 0 negative, 1 neutral,
  2 positive. Bleibt im Drei-Klassen-Raum.

Lange Texte werden in Stücke mit höchstens 512 Tokens zerlegt (Tokenizer
inklusive Sonderzeichen). Die Chunk-Verteilungen werden gemittelt, danach
auf drei Klassen abgebildet.
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
        "notes": "Mehrsprachiger DistilBERT, fünf Klassen auf drei gefaltet, auf 512 Tokens gechunked.",
    },
    "cardiffnlp": {
        "model_id": CARDIFFNLP_MODEL_ID,
        "version": CARDIFFNLP_MODEL_ID,
        "role": "pretrained_zero_shot",
        "max_length": HF_MAX_TOKENS,
        "notes": "XLM-RoBERTa auf Tweets, bereits drei Klassen. Domäne ist Twitter, nicht Nachrichten.",
    },
}


def _missing_dependency_message() -> str:
    return (
        "PyTorch oder Transformers fehlen. "
        "Bitte `pip install -r requirements-transformer.txt` ausführen."
    )


def _id2label_names(id2label: Mapping, width: int) -> list[str]:
    names = []
    for index in range(width):
        if index in id2label:
            names.append(str(id2label[index]))
        elif str(index) in id2label:
            names.append(str(id2label[str(index)]))
        else:
            raise ValueError(f"id2label enthält keinen Eintrag für Klasse {index}.")
    return names


def effective_max_length(tokenizer, requested: int) -> int:
    """Obergrenze je Chunk: Wunsch, Tokenizer-Limit und 512."""
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
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()
        return self

    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        if self._model is None or self._tokenizer is None:
            self.load()
        return [self._scores_for_text(str(text)) for text in texts]

    def _scores_for_text(self, text: str) -> dict[str, float]:
        import torch

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Leerer Text.")
        assert self._tokenizer is not None and self._model is not None
        max_length = effective_max_length(self._tokenizer, self.max_length)
        raw_ids = self._tokenizer(cleaned, add_special_tokens=False, truncation=False)["input_ids"]
        special = int(self._tokenizer.num_special_tokens_to_add(pair=False))
        pieces = content_windows(raw_ids, max_length, special)
        encoded_chunks = [
            self._tokenizer.prepare_for_model(
                piece,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            for piece in pieces
        ]
        batch = self._tokenizer.pad(encoded_chunks, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        if int(input_ids.shape[-1]) > max_length:
            raise RuntimeError(f"Chunk ist länger als {max_length} Tokens.")
        allowed = ("input_ids", "attention_mask", "token_type_ids")
        inputs = {key: batch[key] for key in allowed if key in batch}
        with torch.no_grad():
            logits = self._model(**inputs).logits
            chunk_mean = torch.softmax(logits, dim=-1).mean(dim=0)
        names = _id2label_names(self._model.config.id2label, int(chunk_mean.shape[0]))
        collapsed = collapse_probabilities(names, chunk_mean.detach().cpu().tolist())
        return complete_scores(collapsed)

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
            "Kein Fine-Tuning unter models/finetuned. "
            "Bitte `python -m src.finetune` ausführen (GPU) oder den Colab-Ordner hierher kopieren."
        )
    version = "finetuned"
    max_length = HF_MAX_TOKENS
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
    model = HuggingFaceSentimentModel(
        name="finetuned",
        model_version=version,
        model_id=str(directory),
        max_length=max_length,
        role="finetuned",
        notes="Eigenes Fine-Tuning auf dem Trainings-Split. Inferenz chunked lange Artikel.",
    )
    return model.load()


def finetuned_is_ready(directory: Path = FINETUNED_DIR) -> bool:
    return (directory / "config.json").exists()


def iter_transformer_models() -> tuple[list[HuggingFaceSentimentModel], list[str]]:
    """Lädt die beiden Zero-Shot-Modelle. Fine-Tuning nur, wenn der Ordner existiert."""
    loaded = [load_pretrained(name) for name in ("tabularisai", "cardiffnlp")]
    skipped: list[str] = []
    if finetuned_is_ready():
        loaded.append(load_finetuned())
    else:
        skipped.append("finetuned")
    return loaded, skipped


def label_maps_for_docs() -> dict[str, list[str]]:
    """Die geprüften Namen, in Indexreihenfolge. Nur Dokumentation und Tests."""
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
