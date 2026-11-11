"""Deterministic DE/EN/JA spans. Colours are predictions, not causal attribution."""
import re

MAX_SENTENCES = 80
ABBREVIATIONS = {'dr.', 'prof.', 'mr.', 'mrs.', 'ms.', 'inc.', 'ltd.', 'corp.',
                 'e.g.', 'i.e.', 'u.s.', 'u.k.', 'z.b.', 'bzw.', 'ca.', 'nr.', 'mio.', 'mrd.'}
BOUNDARY = re.compile(r'[。！？]+[」』）"”’»)]*|[.!?]+["”’»)]*(?=\s+|$)|\n\s*\n')


def sentence_spans(text):
    """Keep offsets in the exact original text; whitespace is never invented or lost."""
    spans, start = [], 0
    for match in BOUNDARY.finditer(text):
        token = text[start:match.end()].split()[-1].lower() if text[start:match.end()].split() else ''
        if token in ABBREVIATIONS or re.fullmatch(r'[a-zäöü]\.', token):
            continue
        end = match.end()
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = end
    while start < len(text) and text[start].isspace():
        start += 1
    end = len(text.rstrip())
    if start < end:
        spans.append((start, end))
    return spans
