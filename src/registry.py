"""Loads the model the API should serve.

Sklearn artefacts are read from disk. Transformers are imported only
when the name of such a model is selected.
"""

from src.config import DEFAULT_MODEL_NAME, selected_model_name
from src.interface import SentimentModel
from src.sklearn_models import MODEL_SPECS, load_sklearn_model

TRANSFORMER_NAMES = frozenset({"tabularisai", "cardiffnlp", "finetuned"})


def known_model_names() -> list[str]:
    return [*MODEL_SPECS.keys(), *sorted(TRANSFORMER_NAMES)]


def load_model(name: str | None = None) -> SentimentModel:
    """Builds the selected model. Unknown names and missing files fail loudly."""
    chosen = (name or selected_model_name()).strip().lower()
    if chosen in MODEL_SPECS:
        return load_sklearn_model(chosen)
    if chosen not in TRANSFORMER_NAMES:
        known = ", ".join(known_model_names())
        raise ValueError(f"Unknown model '{chosen}'. Known: {known}")
    from src.hf_models import load_finetuned, load_pretrained

    if chosen == "finetuned":
        return load_finetuned()
    return load_pretrained(chosen)


def default_model_name() -> str:
    return DEFAULT_MODEL_NAME
