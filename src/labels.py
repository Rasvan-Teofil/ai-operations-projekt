"""Abbildung auf den gemeinsamen Drei-Klassen-Raum.

Zero-Shot-Modelle liefern je nach Checkpoint drei oder fünf Labels.
Die Namen werden normalisiert und dann addiert, nicht nach Index geraten.
"""

from collections.abc import Mapping, Sequence

from src.config import HF_MAX_TOKENS, LABELS_3

# Kleinbuchstaben, Leerzeichen statt _ oder -.
_TO_CANONICAL = {
    "very negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very positive": "positive",
}


def canonical_label(name: str) -> str:
    """Mappt einen Modell-Labelnamen auf negative, neutral oder positive."""
    key = " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())
    try:
        return _TO_CANONICAL[key]
    except KeyError as exc:
        raise ValueError(f"Unbekanntes Sentiment-Label: {name}") from exc


def collapse_probabilities(
    label_names: Sequence[str],
    probabilities: Sequence[float],
) -> dict[str, float]:
    """Faltet Klassenwahrscheinlichkeiten auf den Drei-Klassen-Raum.

    Fünf Klassen (Very Negative … Very Positive) werden addiert:
    die beiden negativen Pole nach ``negative``, die beiden positiven
    nach ``positive``. Drei Klassen bleiben erhalten. Die Summe bleibt
    erhalten, wenn die Eingabe eine Verteilung war.
    """
    if len(label_names) != len(probabilities):
        raise ValueError("Labelnamen und Wahrscheinlichkeiten sind unterschiedlich lang.")
    scores = {label: 0.0 for label in LABELS_3}
    for name, probability in zip(label_names, probabilities, strict=True):
        scores[canonical_label(str(name))] += float(probability)
    return scores


def complete_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Sorgt für alle drei Schlüssel, in fester Reihenfolge."""
    return {label: float(scores.get(label, 0.0)) for label in LABELS_3}


def label_from_scores(scores: Mapping[str, float]) -> str:
    """Argmax. Bei Gleichstand gewinnt die frühere Klasse in LABELS_3."""
    return max(LABELS_3, key=lambda label: float(scores.get(label, float("-inf"))))


def mean_probability_rows(rows: Sequence[Sequence[float]]) -> list[float]:
    """Mittelwert über Chunk-Verteilungen, Spalte für Spalte."""
    if not rows:
        raise ValueError("Keine Chunk-Wahrscheinlichkeiten zum Mitteln.")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("Chunk-Verteilungen haben unterschiedliche Breite.")
    count = float(len(rows))
    return [sum(float(row[index]) for row in rows) / count for index in range(width)]


def chunk_token_ids(
    token_ids: Sequence[int],
    max_length: int = HF_MAX_TOKENS,
) -> list[list[int]]:
    """Teilt Token-IDs in Stücke mit höchstens ``max_length`` Einträgen."""
    if max_length < 1:
        raise ValueError("max_length muss mindestens 1 sein.")
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        return []
    return [ids[start : start + max_length] for start in range(0, len(ids), max_length)]


def content_windows(
    token_ids: Sequence[int],
    max_length: int,
    special_tokens: int,
) -> list[list[int]]:
    """Inhaltstücke, sodass Inhalt plus Sonderzeichen in ``max_length`` passen."""
    if special_tokens < 0:
        raise ValueError("special_tokens darf nicht negativ sein.")
    window = max_length - special_tokens
    if window < 1:
        raise ValueError("max_length ist kleiner als die Sonderzeichen.")
    return chunk_token_ids(token_ids, window) or [[]]
