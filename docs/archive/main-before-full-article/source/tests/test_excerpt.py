"""Preview length for scraped articles."""

from src.excerpt import article_excerpt


def test_excerpt_keeps_the_first_sentences():
    text = (
        "The company reported a sharp drop in quarterly revenue and warned of further losses. "
        "Analysts said the warning covers the next two quarters. "
        "The board meets on Thursday to discuss the outlook. "
        "This closing sentence must stay out of the preview."
    )
    excerpt = article_excerpt(text)
    assert excerpt.startswith("The company reported a sharp drop")
    assert "next two quarters" in excerpt
    assert "Thursday" in excerpt
    assert "closing sentence" not in excerpt
    assert excerpt.endswith("…")
    assert len(excerpt) <= 400


def test_excerpt_returns_a_short_text_unchanged():
    text = "Die Aktie verlor nach der Gewinnwarnung deutlich an Wert."
    assert article_excerpt(text) == text


def test_excerpt_clips_a_long_unbroken_paragraph():
    text = "Revenue " + ("fell sharply " * 80)
    excerpt = article_excerpt(text)
    assert excerpt.startswith("Revenue")
    assert excerpt.endswith("…")
    assert len(excerpt) <= 400
    assert "fell sharply" in excerpt
