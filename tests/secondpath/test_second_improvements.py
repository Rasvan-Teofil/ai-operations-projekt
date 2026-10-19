import numpy as np
import pytest
from second_path.aggregation import aggregate
from second_path.fetch import extract_article


def test_tiny_tail_cannot_count_as_much_as_full_chunk():
    probs = [[.9,.05,.05],[.05,.9,.05]]
    chunks = [{'attention_mask':[1]*128},{'attention_mask':[1]*3}]
    out = aggregate(probs,chunks,'token_weighted')
    assert out[0] > .89
    assert out.sum() == pytest.approx(1)
    assert aggregate(probs,chunks,'uniform')[0] == pytest.approx(.475)


def test_article_chooses_body_retains_title_removes_hidden_and_repeats():
    headline = 'Firma meldet schweren Verlust'
    paragraph = 'Die Firma hat im Quartal einen Verlust erlitten und kündigt einen Stellenabbau an.'
    html = f'<body><article><p>Recommended</p></article><article><h1>{headline}</h1><p>{paragraph}</p><p>{paragraph}</p><p hidden>Hidden advertisement</p><p aria-hidden="true">Hidden text</p></article></body>'
    text = extract_article(html,improved=True)
    assert text == headline+' '+paragraph


def test_default_extraction_stays_compatible():
    paragraph = 'The company announced its quarterly reporting date and location this morning.'
    html = f'<article><h1>Heading</h1><p>{paragraph}</p></article>'
    assert extract_article(html) == paragraph
