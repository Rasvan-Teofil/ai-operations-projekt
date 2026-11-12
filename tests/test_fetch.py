import socket
import pytest
import src.fetch as fetch


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('real network forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def resolve(monkeypatch, ips):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a,**k:
        [(socket.AF_INET,socket.SOCK_STREAM,6,'',(ip,443)) for ip in ips])


@pytest.mark.parametrize('ip', ['127.0.0.1','10.0.0.1','169.254.169.254','::1','::ffff:127.0.0.1','192.168.1.1'])
def test_private_block(monkeypatch, ip):
    resolve(monkeypatch, [ip])
    with pytest.raises(fetch.FetchError) as e:
        fetch.public_target('https://example.com')
    assert e.value.status == 422


def test_mixed_dns_block(monkeypatch):
    resolve(monkeypatch, ['1.1.1.1','127.0.0.1'])
    with pytest.raises(fetch.FetchError):
        fetch.public_target('https://example.com')


def test_dns_failure(monkeypatch):
    def fail(*a,**k):
        raise socket.gaierror('missing')
    monkeypatch.setattr(socket, 'getaddrinfo', fail)
    with pytest.raises(fetch.FetchError) as e:
        fetch.public_target('https://example.com')
    assert e.value.status == 502


@pytest.mark.parametrize('robots,article,expected', [
    ((200,'text/plain','User-agent: *\nDisallow: /'), None,403),
    ((500,'text/plain','oops'), None,502),
    ((404,'text/plain',''), (200,'text/html','<body><p>short</p></body>'),422),
    ((404,'text/plain',''), (500,'text/html','error'),502),
    ((404,'text/plain',''), (200,'application/pdf','binary'),422),
    ((200,'text/plain','User-agent: *\nAllow: /'),
     (200,'text/html','<article><p>Profits increased sharply, exceeding analyst expectations this quarter.</p></article>'),200)])
def test_robots_and_extraction(monkeypatch, robots, article, expected):
    resolve(monkeypatch, ['1.1.1.1'])
    replies = iter([robots, article])
    calls = []
    def request(u,host,port,ip,path):
        calls.append((ip,path))
        return next(replies)
    monkeypatch.setattr(fetch,'request',request)
    if expected == 200:
        assert 'Profits increased' in fetch.article_text('https://example.com/news')
        assert calls == [('1.1.1.1','/robots.txt'),('1.1.1.1','/news')]
    else:
        with pytest.raises(fetch.FetchError) as e:
            fetch.article_text('https://example.com/news')
        assert e.value.status == expected


@pytest.mark.parametrize('mode', ['redirect','timeout','oversize'])
def test_transport_failures(monkeypatch, mode):
    class Response:
        status = 302 if mode == 'redirect' else 200
        def read(self, limit):
            return b'x'*(fetch.LIMIT+1)
    class Conn:
        def __init__(self,*a,**k): pass
        def request(self,*a,**k):
            if mode == 'timeout': raise TimeoutError()
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(fetch.http.client, 'HTTPConnection', Conn)
    u = fetch.urlsplit('http://example.com/news')
    with pytest.raises(fetch.FetchError) as e:
        fetch.request(u,'example.com',80,'1.1.1.1','/news')
    assert e.value.status == 502


def test_unicode_path_is_encoded(monkeypatch):
    resolve(monkeypatch, ['1.1.1.1'])
    paths = []
    def response(u,host,port,ip,path):
        paths.append(path)
        if path == '/robots.txt': return 404,'text/plain',''
        return 200,'text/html','<article><p>Eine lesbare Wirtschaftsmeldung mit ausreichend vielen Zeichen.</p></article>'
    monkeypatch.setattr(fetch,'request',response)
    fetch.article_text('https://example.com/börse?q=höher')
    assert paths[1] == '/b%C3%B6rse?q=h%C3%B6her'


def test_https_pins_ip_but_verifies_hostname(monkeypatch):
    calls = []
    class Sock:
        def close(self): pass
    def connect(addr,timeout):
        calls.append(('connect',addr,timeout))
        return Sock()
    class Context:
        def wrap_socket(self,sock,server_hostname):
            calls.append(('tls',server_hostname))
            return sock
    monkeypatch.setattr(socket,'create_connection',connect)
    monkeypatch.setattr(fetch.ssl,'create_default_context',lambda: Context())
    # Bypass the stdlib initializer's context properties; exercise the actual connect implementation.
    conn = object.__new__(fetch.PinnedHTTPS)
    conn.address,conn.port,conn.timeout,conn.host = '1.1.1.1',443,15,'example.com'
    conn._context = Context()
    conn.connect()
    assert calls == [('connect',('1.1.1.1',443),15),('tls','example.com')]
