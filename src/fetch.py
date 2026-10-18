"""One public HTTP(S) article, robots first; pinned DNS, bounded body, no redirects."""
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit, urlunsplit, quote
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

AGENT = 'AIOpsSecondPathBot/1.0'
TIMEOUT = 15
LIMIT = 2_000_000


class FetchError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


def public_target(url):
    try:
        u = urlsplit(url)
        if (u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password
                or any(c.isspace() for c in url) or u.port not in (None, 80, 443)):
            raise FetchError(422, 'Exactly one public http(s) URL is required')
        host = u.hostname.encode('idna').decode('ascii')
        port = u.port or (443 if u.scheme == 'https' else 80)
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = sorted({answer[4][0] for answer in answers})
        if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
            raise FetchError(422, 'Non-public address is not allowed')
        return u, host, port, addresses[0]
    except FetchError:
        raise
    except (ValueError, UnicodeError) as e:
        raise FetchError(422, 'Invalid URL') from e
    except OSError as e:
        raise FetchError(502, 'Host cannot be resolved') from e


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, address):
        super().__init__(host, port, timeout=TIMEOUT, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, self.port), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


def request(u, host, port, address, path):
    # Connection goes to the validated IP; hostname is retained for Host and TLS verification.
    conn = (PinnedHTTPS(host, port, address) if u.scheme == 'https'
            else http.client.HTTPConnection(address, port, timeout=TIMEOUT))
    host_header = f'[{host}]' if ':' in host else host
    if port != (443 if u.scheme == 'https' else 80):
        host_header += f':{port}'
    try:
        conn.request('GET', path, headers={'Host': host_header, 'User-Agent': AGENT,
                                         'Accept': 'text/html,text/plain', 'Accept-Encoding': 'identity'})
        res = conn.getresponse()
        if 300 <= res.status < 400:
            raise FetchError(502, 'Redirects are not followed')
        body = res.read(LIMIT+1)
        if len(body) > LIMIT:
            raise FetchError(502, 'Upstream response is too large')
        charset = res.headers.get_content_charset() or 'utf-8'
        try:
            text = body.decode(charset, errors='replace')
        except LookupError:
            text = body.decode('utf-8', errors='replace')
        return res.status, res.headers.get_content_type(), text
    except FetchError:
        raise
    except (OSError, http.client.HTTPException) as e:
        raise FetchError(502, 'Upstream request failed or timed out') from e
    finally:
        conn.close()


def article_text(url):
    u, host, port, ip = public_target(url)
    status, _, robots = request(u, host, port, ip, '/robots.txt')
    if status == 200:
        parser = RobotFileParser()
        parser.parse(robots.splitlines())
        if not parser.can_fetch(AGENT, url):
            raise FetchError(403, 'robots.txt forbids this article')
    elif status != 404:
        raise FetchError(502, 'robots.txt cannot be read')
    path = urlunsplit(('', '', quote(u.path or '/', safe="/%:@!$&'()*+,;=-._~"),
                         quote(u.query, safe="/?%:@!$&'()*+,;=+-._~"), ''))
    status, content_type, html = request(u, host, port, ip, path)
    if status != 200:
        raise FetchError(502, f'Article returned HTTP {status}')
    if content_type not in ('text/html', 'application/xhtml+xml'):
        raise FetchError(422, 'No readable HTML article')
    soup = BeautifulSoup(html, 'html.parser')
    for node in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        node.decompose()
    container = soup.find('article') or soup.find('main') or soup.body
    text = ' '.join(p.get_text(' ', strip=True) for p in container.find_all('p')) if container else ''
    if len(text.strip()) < 40:
        raise FetchError(422, 'No readable article text')
    if len(text) > 20000:
        raise FetchError(422, 'Article exceeds 20000 characters')
    return text
