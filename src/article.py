"""Versioned full-article extraction shared by the API and the next data pilot.

This is a bounded text extractor, not a guarantee of publisher/layout support.
Historical lead and improved=False/True extractors remain in their own modules.
"""
import re
import unicodedata
from datetime import datetime

from bs4 import BeautifulSoup

ARTICLE_POLICY = 'full-article-v5'
MAX_CHARACTERS = 20_000


class ArticleError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(detail)


def substantive(text, language):
    """Do not require whitespace-separated words in native Japanese text."""
    if language not in ('de', 'en', 'ja'):
        raise ValueError('Supported pilot languages: de, en, ja')
    letters = sum(unicodedata.category(c).startswith(('L', 'N')) for c in text)
    if language == 'ja':
        return letters >= 40 and bool(re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text))
    return letters >= 60 and len(text.split()) >= 15


def extract_full_article(html, title=None):
    """Extract all visible paragraphs in one body, in order, without truncation.

    Inline quotations remain text; a separate per-item rights review is required.
    The optional title is for revision-pinned data sources, not a model input hint.
    """
    soup = BeautifulSoup(html, 'html.parser')
    headings = [node.get_text(' ', strip=True) for node in soup.select('title, h1')]
    gate_titles = {
        'access denied', 'forbidden', '403 forbidden', '403 access denied',
        'just a moment...', 'just a moment…', 'attention required! | cloudflare',
        'zugriff verweigert', 'verify you are human', 'page not found', '404 not found',
    }
    if any(h.casefold().strip() in gate_titles for h in headings) or soup.select_one('#challenge-form, #cf-challenge-running'):
        raise ArticleError('access_page', 'The page is an access challenge or error page, not a readable article')
    if soup.select_one('input[type="password"]') and not soup.select_one('article, [itemprop="articleBody"]'):
        raise ArticleError('access_page', 'The page requires sign-in; paste the article text instead')
    for node in soup.select('script, style, nav, footer, header, aside, form, noscript, '
                            '[hidden], [aria-hidden="true"], .mw-editsection, .metadata, '
                            '.noprint, .thumb, .gallery, .infobox, .reflist, .references, .toc, sup.reference, figure, .lang-quote'):
        node.decompose()
    for node in list(soup.select('[style]')):
        if node.attrs is None:
            continue
        if re.search(r'(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:!important\s*)?(?:;|$)', node.get('style', ''), re.I):
            node.decompose()
    # Prefer explicitly marked article bodies; retain generic body fallback for
    # simple layouts, subject to the paragraph/length checks below.
    containers = (soup.select('[itemprop="articleBody"]') or soup.select('article')
                  or soup.select('.mw-parser-output') or soup.select('main') or [soup.body])
    container = max((c for c in containers if c is not None),
                    key=lambda c: sum(len(p.get_text(' ', strip=True)) for p in c.find_all('p')),
                    default=None)
    if container is None:
        raise ArticleError('extraction_failed', 'No readable article body was found')
    paragraphs = [p.get_text(' ', strip=True) for p in container.find_all('p')]
    if sum(len(p) for p in paragraphs) < 40:
        raise ArticleError('extraction_failed', 'No readable article paragraphs were found')
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    wiki = 'mw-parser-output' in container.get('class', [])
    if wiki:
        # Old Japanese topic boxes have no class. Restrict the fallback to
        # floating sidebars containing category links and a linked-news list.
        for box in list(container.find_all('div', style=True)):
            if box.parent is None:
                continue
            if (re.search(r'float\s*:\s*right', box['style'], re.I)
                    and box.select_one('ul') and len(box.select('li a')) >= 3
                    and any('カテゴリ:' in a.get('title', '') for a in box.find_all('a'))):
                box.decompose()
        for table in list(container.find_all('table')):
            value = table.get_text(' ', strip=True)
            if 'Bitte keine inhaltlichen Veränderungen vornehmen.' in value and value.startswith('Veröffentlicht'):
                table.decompose()
    footers = {'sources', 'quellen', 'information sources', '情報源', '出典',
               'see also', 'external links', 'links', 'weblinks', '関連ニュース', '関連記事',
               'themenverwandte artikel', 'originäre berichterstattung',
               'related news', 'related articles', 'sister links', 'gallery', '注釈'}
    # Preserve declared image alternatives in data tables (e.g. gain/loss
    # arrows). Do not infer a corrected sign from the image filename.
    for image in list(container.select('table img[alt]')):
        alternative = image.get('alt', '').strip()
        if alternative:
            image.replace_with('[' + alternative + ']')
    for element in container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                       'p', 'li', 'tr', 'dt', 'dd', 'blockquote', 'caption']):
        if wiki and ' '.join(element.get_text(' ', strip=True).casefold().removesuffix('[編集]').removesuffix(':').split()) in footers:
            break
        if element.find_parent(['li', 'tr', 'dt', 'dd', 'blockquote', 'caption']) is not None:
            continue
        value = element.get_text(' ', strip=True)
        if wiki and (value.startswith('Kommentar abgeben') or value == 'Diesen Artikel teilen'):
            break
        # Only adjacent exact duplicates: repeated distant passages may matter.
        if value and (not parts or value != parts[-1]):
            parts.append(value)
    text = '\n\n'.join(parts)
    if len(text) > MAX_CHARACTERS:
        raise ArticleError('article_too_long', 'Article exceeds 20000 characters; no text was truncated')
    return text


def visible_publication_date(text, language):
    """Date displayed by the pinned article, not independently verified truth."""
    intro = text.split('\n\n', 1)[-1][:220]
    if language == 'de':
        match = re.search(r'\b(\d{2})\.(\d{2})\.(\d{4})\b', intro)
        if match:
            day, month, year = map(int, match.groups())
            try:
                return datetime(year, month, day).date().isoformat()
            except ValueError:
                return None
    elif language == 'ja':
        match = re.search(r'【(\d{4})年(\d{1,2})月(\d{1,2})日】', intro)
        if match:
            try:
                return datetime(*map(int, match.groups())).date().isoformat()
            except ValueError:
                return None
    elif language == 'en':
        match = re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b', intro)
        if match:
            return datetime.strptime(match.group(), '%B %d, %Y').date().isoformat()
    return None

