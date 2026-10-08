"""Holt den Text genau eines Artikels.

Kein Crawlen: Links auf der Seite werden nicht verfolgt, Weiterleitungen
auch nicht. ``robots.txt`` wird beachtet. Hosts, die auf loopback, private
oder link-lokale Adressen zeigen, werden abgelehnt.
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
except ImportError:  # pragma: no cover - Abhängigkeit steht in requirements.txt
    trafilatura = None


class ScrapeError(Exception):
    """Basisklasse für erwartbare Abrufprobleme."""


class FetchError(ScrapeError):
    """Netzwerk, Statuscode oder unzulässige URL."""


class RobotsDenied(ScrapeError):
    """robots.txt verbietet diesen Abruf für unseren User-Agent."""


class EmptyArticleError(ScrapeError):
    """Die Seite lieferte keinen lesbaren Haupttext."""


def fetch_article_text(url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
    """Lädt eine http(s)-URL und gibt den Artikeltext zurück."""
    parsed = _require_public_http_url(url)
    if not _robots_allow(parsed, timeout):
        raise RobotsDenied("robots.txt verbietet den Abruf dieser URL.")
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=False,
        )
    except requests.Timeout as exc:
        raise FetchError("Abruf der URL hat das Zeitlimit überschritten.") from exc
    except requests.RequestException as exc:
        raise FetchError("Abruf der URL ist fehlgeschlagen.") from exc
    if response.is_redirect or response.is_permanent_redirect:
        raise FetchError(
            "Weiterleitungen werden nicht verfolgt. Bitte die direkte Artikel-URL angeben."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        raise FetchError(f"Abruf der URL ist fehlgeschlagen (HTTP {status}).") from exc
    text = extract_main_text(response.text)
    if not text:
        raise EmptyArticleError("Aus der Seite ließ sich kein Artikeltext lesen.")
    return text


def extract_main_text(html: str) -> str:
    """Trafilatura, und wenn das nichts liefert, der sichtbare Text von article/main/body."""
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
        raise FetchError("Nur einzelne http- oder https-URLs sind erlaubt.")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal")):
        raise FetchError("Lokale Adressen werden nicht abgerufen.")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not _is_public(literal):
            raise FetchError("Nicht-öffentliche Adressen werden nicht abgerufen.")
        return parsed
    _reject_nonpublic_resolution(host)
    return parsed


def _reject_nonpublic_resolution(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise FetchError("Host konnte nicht aufgelöst werden.") from exc
    if not infos:
        raise FetchError("Host konnte nicht aufgelöst werden.")
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not _is_public(address):
            raise FetchError("Die URL zeigt auf eine nicht-öffentliche Adresse.")


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
        raise FetchError("robots.txt konnte nicht gelesen werden.") from exc
    if response.status_code == 404:
        return True
    if response.status_code >= 400 or response.is_redirect:
        raise FetchError("robots.txt konnte nicht gelesen werden.")
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, parsed.geturl())
