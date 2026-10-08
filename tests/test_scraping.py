"""Abruf eines einzelnen Artikels, ohne das Netz zu berühren."""

import socket

import pytest
import requests

from src.scraping import EmptyArticleError, FetchError, RobotsDenied, fetch_article_text

ARTICLE_HTML = """
<html><head><title>Beispiel</title></head>
<body><article>
<h1>Bibliothek</h1>
<p>Freiwillige haben genug Geld gesammelt um die Bibliothek wieder zu oeffnen
und die Kinder wieder lesen zu lassen.</p>
</article></body></html>
"""


class DummyResponse:
    def __init__(self, status_code: int, text: str, redirect: bool = False):
        self.status_code = status_code
        self.text = text
        self.is_redirect = redirect
        self.is_permanent_redirect = False

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


@pytest.fixture
def public_dns(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr("src.scraping.socket.getaddrinfo", fake_getaddrinfo)


def test_localhost_is_rejected_without_a_request(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("kein Netz")

    monkeypatch.setattr("src.scraping.requests.get", boom)
    with pytest.raises(FetchError):
        fetch_article_text("http://localhost/news/1")


def test_robots_disallow_does_not_fetch_the_article(monkeypatch, public_dns):
    calls = []

    def fake_get(url, timeout, headers, allow_redirects):
        calls.append(url)
        assert timeout == 3
        if str(url).endswith("/robots.txt"):
            return DummyResponse(200, "User-agent: *\nDisallow: /\n")
        raise AssertionError("Artikel darf bei Disallow nicht geladen werden")

    monkeypatch.setattr("src.scraping.requests.get", fake_get)
    with pytest.raises(RobotsDenied):
        fetch_article_text("https://example.com/news/1", timeout=3)
    assert calls == ["https://example.com/robots.txt"]


def test_single_article_is_fetched_after_robots(monkeypatch, public_dns):
    calls = []

    def fake_get(url, timeout, headers, allow_redirects):
        calls.append((url, timeout, allow_redirects))
        if str(url).endswith("/robots.txt"):
            return DummyResponse(200, "User-agent: *\nDisallow:\n")
        return DummyResponse(200, ARTICLE_HTML)

    monkeypatch.setattr("src.scraping.requests.get", fake_get)
    text = fetch_article_text("https://example.com/news/bibliothek", timeout=4)
    assert "Bibliothek" in text
    assert [url for url, _, _ in calls] == [
        "https://example.com/robots.txt",
        "https://example.com/news/bibliothek",
    ]
    assert all(timeout == 4 and allow_redirects is False for _, timeout, allow_redirects in calls)


def test_timeout_becomes_fetch_error(monkeypatch, public_dns):
    def fake_get(url, timeout, headers, allow_redirects):
        if str(url).endswith("/robots.txt"):
            return DummyResponse(200, "User-agent: *\nDisallow:\n")
        raise requests.Timeout("langsam")

    monkeypatch.setattr("src.scraping.requests.get", fake_get)
    with pytest.raises(FetchError):
        fetch_article_text("https://example.com/news/1")


def test_empty_page_is_an_empty_article(monkeypatch, public_dns):
    def fake_get(url, timeout, headers, allow_redirects):
        if str(url).endswith("/robots.txt"):
            return DummyResponse(404, "")
        return DummyResponse(200, "<html><body></body></html>")

    monkeypatch.setattr("src.scraping.requests.get", fake_get)
    with pytest.raises(EmptyArticleError):
        fetch_article_text("https://example.com/empty")
