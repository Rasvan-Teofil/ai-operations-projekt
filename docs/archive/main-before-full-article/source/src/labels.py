"""Mapping to the shared three-class space.

Depending on the checkpoint, zero-shot models return three or five labels.
The names are normalised and then summed, not guessed by index.
"""

from collections.abc import Mapping, Sequence

from src.config import HF_MAX_TOKENS, LABELS_3

# Lower case, spaces instead of _ or -.
_TO_CANONICAL = {
    "very negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very positive": "positive",
}


def canonical_label(name: str) -> str:
    """Maps a model label name to negative, neutral or positive."""
    key = " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())
    try:
        return _TO_CANONICAL[key]
    except KeyError as exc:
        raise ValueError(f"Unknown sentiment label: {name}") from exc


def collapse_probabilities(
    label_names: Sequence[str],
    probabilities: Sequence[float],
) -> dict[str, float]:
    """Folds class probabilities into the three-class space.

    Five classes (Very Negative … Very Positive) are summed: the two
    negative poles into ``negative``, the two positive ones into
    ``positive``. Three classes are kept as they are. The sum is
    preserved if the input was a distribution.
    """
    if len(label_names) != len(probabilities):
        raise ValueError("Label names and probabilities have different lengths.")
    scores = {label: 0.0 for label in LABELS_3}
    for name, probability in zip(label_names, probabilities, strict=True):
        scores[canonical_label(str(name))] += float(probability)
    return scores


def complete_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Ensures all three keys, in a fixed order."""
    return {label: float(scores.get(label, 0.0)) for label in LABELS_3}


def label_from_scores(scores: Mapping[str, float]) -> str:
    """Argmax. On a tie, the earlier class in LABELS_3 wins."""
    return max(LABELS_3, key=lambda label: float(scores.get(label, float("-inf"))))


def mean_probability_rows(rows: Sequence[Sequence[float]]) -> list[float]:
    """Mean over chunk distributions, column by column."""
    if not rows:
        raise ValueError("No chunk probabilities to average.")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("Chunk distributions have different widths.")
    count = float(len(rows))
    return [sum(float(row[index]) for row in rows) / count for index in range(width)]


def chunk_token_ids(
    token_ids: Sequence[int],
    max_length: int = HF_MAX_TOKENS,
) -> list[list[int]]:
    """Splits token IDs into chunks of at most ``max_length`` entries."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1.")
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        return []
    return [ids[start : start + max_length] for start in range(0, len(ids), max_length)]


def content_windows(
    token_ids: Sequence[int],
    max_length: int,
    special_tokens: int,
) -> list[list[int]]:
    """Content chunks such that content plus special tokens fit into ``max_length``."""
    if special_tokens < 0:
        raise ValueError("special_tokens must not be negative.")
    window = max_length - special_tokens
    if window < 1:
        raise ValueError("max_length is smaller than the number of special tokens.")
    return chunk_token_ids(token_ids, window) or [[]]
