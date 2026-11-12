"""Offline collector policies; no remote content or model calls."""
from src.article import extract_full_article
from src.next_stage_collect import flags


def test_wikinews_keeps_lists_and_table_facts_before_sources_only():
    html = '<div class="mw-parser-output"><p>The library has reopened after its entrance was repaired and inspected.</p><ul><li>Entrance restored<ul><li>Wheelchair access restored</li></ul></li><li>Opening hours unchanged</li></ul><table><tr><td>Branches</td><td>3</td></tr></table><h2>Sources</h2><p>Publisher reference must not become an article paragraph.</p></div>'
    text = extract_full_article(html)
    assert text.count('Wheelchair access restored') == 1
    assert 'Opening hours unchanged' in text and 'Branches 3' in text
    assert 'Sources' not in text and 'Publisher reference' not in text


def test_sources_heading_in_regular_article_is_not_a_wikinews_footer():
    html = '<article><p>The report describes several types of electricity generation in detail.</p><h2>Sources</h2><p>Wind and solar capacity increased.</p></article>'
    assert 'Wind and solar capacity increased.' in extract_full_article(html)


def test_explicit_translation_flag_does_not_mean_a_translator_was_used_by_project():
    assert flags('この記事は 英語版ウィキニュース からの翻訳です。')[0]
    assert flags('This article was translated from English.')[0]
    assert not flags('Die Schule eröffnet am Montag wieder ihre Bibliothek.')[0]


def test_wiki_navigation_tooltips_and_related_news_do_not_pollute_body():
    html = '''<div class="mw-parser-output">
    <div style="float:right; width:200px"><a title="カテゴリ:鉄道">鉄道</a><ul>
    <li><a>Unrelated fire</a></li><li><a>Unrelated crash</a></li><li><a>Unrelated delay</a></li></ul></div>
    <p>The railway has opened its completed bridge and the first scheduled train crossed it today.</p>
    <p>The engineer said “It works.”<sup><span class="lang-quote">((ja))<span class="transquote">隠れた重複</span></span></sup></p>
    <table><tr><td>Services restored</td><td>12</td></tr></table>
    <p><span style="display: none">Hidden unrelated news</span>All platforms are accessible.</p>
    <h2>Related news</h2><ul><li>Another accident</li></ul><h2>Sources</h2></div>'''
    text = extract_full_article(html)
    assert 'first scheduled train crossed' in text
    assert 'Services restored 12' in text and 'All platforms are accessible.' in text
    assert not any(s in text for s in ('Unrelated', '隠れた', 'Hidden unrelated', 'Another accident', 'Related news'))


def test_german_comment_footer_with_multiple_links_is_not_article_text():
    html = '<div class="mw-parser-output"><p>The library reopened after workers restored its damaged main entrance yesterday.</p><p>Kommentar abgeben / Kommentare ansehen</p><p>Unrelated footer</p></div>'
    assert 'Kommentar' not in extract_full_article(html)
    assert 'Unrelated footer' not in extract_full_article(html)


def test_quote_length_frame_is_class_blind_and_language_specific():
    from src.next_stage_collect import long_quotation
    assert long_quotation('He said "' + 'a' * 80 + '".', 'en')
    assert not long_quotation('He called it "safe and effective".', 'en')
    assert long_quotation('「' + 'あ' * 40 + '」', 'ja')
    assert not long_quotation('「図書館」', 'ja')


def test_nested_hidden_styles_do_not_crash_after_parent_removal():
    html = '<article><p>The repaired bridge is open and the first scheduled train has crossed it today.</p><div style="display:none"><span style="color:red">Hidden repeated text</span></div></article>'
    assert 'Hidden' not in extract_full_article(html)
