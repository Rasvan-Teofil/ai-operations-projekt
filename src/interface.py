"""Gemeinsame Inferenz-Schnittstelle für alle Sentiment-Modelle.

Jedes Modell liefert ``predict_proba(texts)``: eine Verteilung über
negative, neutral und positive. Die API und der Vergleich kennen
keine modellspezifischen Aufrufe.
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
        """Wahrscheinlichkeiten je Text, Schlüssel sind die drei Klassen."""

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [label_from_scores(scores) for scores in self.predict_proba(texts)]

    def tracking_params(self) -> dict[str, str | int | float | bool]:
        return {}

    def size_mb(self) -> float | None:
        return None
