"""Shared inference interface for all sentiment models.

Every model provides ``predict_proba(texts)``: a distribution over
negative, neutral and positive. The API and the comparison know no
model-specific calls.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.labels import label_from_scores


class SentimentModel(ABC):
    name: str
    model_version: str
    role: str

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Probabilities per text; the keys are the three classes."""

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [label_from_scores(scores) for scores in self.predict_proba(texts)]

    def tracking_params(self) -> dict[str, str | int | float | bool]:
        return {}

    def size_mb(self) -> float | None:
        return None
