"""Short preview of a scraped article.

URL predictions classify the full fetched text and return only the
opening, so the demo can show what was actually scraped. Raw-text
requests do not use this helper.
"""

import re

MAX_EXCERPT_SENTENCES = 3
MAX_EXCERPT_CHARS = 400

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def article_excerpt(
    text: str,
    *,
    max_sentences: int = MAX_EXCERPT_SENTENCES,
    max_chars: int = MAX_EXCERPT_CHARS,
) -> str:
    """First sentences of ``text``, capped by count and characters.

    Whitespace is collapsed. When the opening is shorter than the
    article, the result ends with an ellipsis and still fits in
    ``max_chars``.
    """
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1.")
    if max_chars < 8:
        raise ValueError("max_chars must be at least 8.")
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return ""
    sentences = [part.strip() for part in _SENTENCE_BREAK.split(cleaned) if part.strip()]
    if not sentences:
        sentences = [cleaned]
    chosen: list[str] = []
    for sentence in sentences:
        if len(chosen) >= max_sentences:
            break
        candidate = " ".join([*chosen, sentence])
        if len(candidate) <= max_chars:
            chosen.append(sentence)
            continue
        if not chosen:
            chosen.append(_clip_words(sentence, max_chars))
        break
    excerpt = " ".join(chosen).strip()
    if excerpt == cleaned:
        return excerpt
    return _with_ellipsis(excerpt, max_chars)


def _clip_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    if len(cut) < max(1, limit // 2):
        cut = text[:limit].rstrip()
    return cut


def _with_ellipsis(excerpt: str, max_chars: int) -> str:
    body = excerpt.rstrip()
    if len(body) + 1 <= max_chars:
        return body + "…"
    clipped = _clip_words(body, max_chars - 1).rstrip()
    return clipped + "…"
