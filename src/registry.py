"""Lädt das Modell, das die API ausliefern soll.

Sklearn-Artefakte kommen von der Platte. Transformer werden erst importiert,
wenn der Name eines solchen Modells gewählt ist.
"""

from src.config import DEFAULT_MODEL_NAME, selected_model_name
from src.interface import SentimentModel
from src.sklearn_models import MODEL_SPECS, load_sklearn_model

TRANSFORMER_NAMES = frozenset({"tabularisai", "cardiffnlp", "finetuned"})


def known_model_names() -> list[str]:
    return [*MODEL_SPECS.keys(), *sorted(TRANSFORMER_NAMES)]


def load_model(name: str | None = None) -> SentimentModel:
    """Baut das gewählte Modell. Unbekannte Namen und fehlende Dateien fallen laut auf."""
    chosen = (name or selected_model_name()).strip().lower()
    if chosen in MODEL_SPECS:
        return load_sklearn_model(chosen)
    if chosen not in TRANSFORMER_NAMES:
        known = ", ".join(known_model_names())
        raise ValueError(f"Unbekanntes Modell '{chosen}'. Bekannt: {known}")
    from src.hf_models import load_finetuned, load_pretrained

    if chosen == "finetuned":
        return load_finetuned()
    return load_pretrained(chosen)


def default_model_name() -> str:
    return DEFAULT_MODEL_NAME
