"""Fetches the text of exactly one article.

No crawling: links on the page are not followed, and neither are
redirects. ``robots.txt`` is respected. Hosts that resolve to loopback,
private or link-local addresses are rejected.
"""

import ipaddress
import socket
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from src.config import REQUEST_TIMEOUT_SECONDS, USER_AGENT

try:
    import trafilatura
except ImportError:  # pragma: no cover - dependency is listed in requirements.txt
    trafilatura = None


class ScrapeError(Exception):
    """Base class for expected fetch problems."""


class FetchError(ScrapeError):
    """Network, status code or disallowed URL."""


class RobotsDenied(ScrapeError):
    """robots.txt forbids this fetch for our user agent."""


class EmptyArticleError(ScrapeError):
    """The page returned no readable main text."""


def fetch_article_text(url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
    """Downloads an http(s) URL and returns the article text."""
    parsed = _require_public_http_url(url)
    if not _robots_allow(parsed, timeout):
        raise RobotsDenied("robots.txt forbids fetching this URL.")
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=False,
        )
    except requests.Timeout as exc:
        raise FetchError("Fetching the URL timed out.") from exc
    except requests.RequestException as exc:
        raise FetchError("Fetching the URL failed.") from exc
    if response.is_redirect or response.is_permanent_redirect:
        raise FetchError(
            "Redirects are not followed. Please provide the direct article URL."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        raise FetchError(f"Fetching the URL failed (HTTP {status}).") from exc
    text = extract_main_text(response.text)
    if not text:
        raise EmptyArticleError("No article text could be read from the page.")
    return text


def extract_main_text(html: str) -> str:
    """Trafilatura; if that returns nothing, the visible text of article/main/body."""
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(html)
        except Exception:
            extracted = None
        if extracted and extracted.strip():
            return " ".join(extracted.split())
    return _beautifulsoup_text(html)


def _beautifulsoup_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    node = soup.find("article") or soup.find("main") or soup.body
    if node is None:
        return ""
    return " ".join(node.get_text(separator=" ", strip=True).split())


def _require_public_http_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FetchError("Only single http or https URLs are allowed.")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal")):
        raise FetchError("Local addresses are not fetched.")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not _is_public(literal):
            raise FetchError("Non-public addresses are not fetched.")
        return parsed
    _reject_nonpublic_resolution(host)
    return parsed


def _reject_nonpublic_resolution(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise FetchError("Host could not be resolved.") from exc
    if not infos:
        raise FetchError("Host could not be resolved.")
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not _is_public(address):
            raise FetchError("The URL points to a non-public address.")


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _robots_allow(parsed, timeout: float) -> bool:
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = requests.get(
            robots_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise FetchError("robots.txt could not be read.") from exc
    if response.status_code == 404:
        return True
    if response.status_code >= 400 or response.is_redirect:
        raise FetchError("robots.txt could not be read.")
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, parsed.geturl())
