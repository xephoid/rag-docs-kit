"""Transport hardening for the optional remote backends (issue 5). No network used."""

from __future__ import annotations

import email.message
import io
import urllib.error
import urllib.request
import urllib.response

import pytest
from pydantic import SecretStr, ValidationError

from ragkit import remote
from ragkit.remote import RemoteConfig, RemoteError, remote_config_from_env, urllib_transport

KEY = "sk-very-secret-value"


@pytest.mark.parametrize(
    "url",
    ["http://api.example.com/v1", "ftp://api.example.com", "http://10.0.0.5/v1", "https://"],
)
def test_plaintext_and_non_https_urls_are_rejected(url):
    env = {remote.BASE_URL_ENV: url, remote.API_KEY_ENV: KEY}
    with pytest.raises(RemoteError):
        remote_config_from_env(env)
    with pytest.raises(ValidationError):
        RemoteConfig(base_url=url, api_key=KEY)
    with pytest.raises(RemoteError):
        urllib_transport(url + "/chat/completions", {}, SecretStr(KEY))


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1",
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_https_and_explicit_loopback_are_allowed(url):
    env = {remote.BASE_URL_ENV: url, remote.API_KEY_ENV: KEY}
    assert remote_config_from_env(env).base_url == url


def test_api_key_is_redacted_everywhere_but_the_request():
    config = remote_config_from_env(
        {remote.BASE_URL_ENV: "https://api.example.com/v1", remote.API_KEY_ENV: KEY}
    )
    for text in (repr(config), str(config), config.model_dump_json(), repr(config.model_dump())):
        assert KEY not in text
    assert config.api_key.get_secret_value() == KEY


class _RedirectingHTTPS(urllib.request.HTTPSHandler):
    """Answers every HTTPS request with a 302 to another origin, without sockets."""

    handler_order = 100  # run before the real HTTPS handler

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[urllib.request.Request] = []

    def https_open(self, req):
        self.requests.append(req)
        headers = email.message.Message()
        headers["Location"] = "https://evil.example/steal"
        resp = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, 302)
        resp.msg = "Found"
        return resp


def test_redirects_are_not_followed(monkeypatch):
    handler = _RedirectingHTTPS()
    original = remote._build_opener

    def opener(direct=False):
        built = original(direct)
        built.add_handler(handler)
        return built

    monkeypatch.setattr(remote, "_build_opener", opener)
    with pytest.raises(RemoteError, match="redirect"):
        urllib_transport("https://api.example.com/v1/chat/completions", {}, SecretStr(KEY))
    [request] = handler.requests
    assert request.host == "api.example.com"
    assert request.unredirected_hdrs["Authorization"] == f"Bearer {KEY}"
    assert "Authorization" not in request.headers


# --- round 3, issue 3: loopback HTTP never goes through an environment proxy --


class _RecordingHTTP(urllib.request.HTTPHandler):
    """Records where each plain-HTTP request would be sent, without sockets."""

    handler_order = 200  # after ProxyHandler (100), before the real HTTPHandler (500)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[urllib.request.Request] = []

    def http_open(self, req):
        self.requests.append(req)
        resp = urllib.response.addinfourl(
            io.BytesIO(b'{"ok": true}'), email.message.Message(), req.full_url, 200
        )
        resp.msg = "OK"
        return resp


@pytest.fixture
def proxy_env(monkeypatch):
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:3128")


def _record(monkeypatch, handler):
    original = remote._build_opener

    def opener(direct=False):
        built = original(direct)
        built.add_handler(handler)
        return built

    monkeypatch.setattr(remote, "_build_opener", opener)


def test_configured_http_proxy_cannot_receive_loopback_request(monkeypatch, proxy_env):
    # Sanity check: without the fix the env proxy would be used for this request.
    leaky = remote._build_opener(direct=False)
    leaky_handler = _RecordingHTTP()
    leaky.add_handler(leaky_handler)
    leaky.open(urllib.request.Request("http://127.0.0.1:8000/v1/x", data=b"{}"))
    assert leaky_handler.requests[0].host == "proxy.invalid:3128"

    handler = _RecordingHTTP()
    _record(monkeypatch, handler)
    for url in ("http://127.0.0.1:8000/v1/chat/completions", "http://localhost:8000/v1/x"):
        assert urllib_transport(url, {"doc": "payload"}, SecretStr(KEY)) == {"ok": True}
    assert [r.host for r in handler.requests] == ["127.0.0.1:8000", "localhost:8000"]
    assert all(not r.has_proxy() for r in handler.requests)


# --- round 3, issue 4: no userinfo/query/fragment; errors never echo URLs ------


@pytest.mark.parametrize(
    "url",
    [
        f"https://user:{KEY}@api.example.com/v1",
        f"https://{KEY}@api.example.com/v1",
        f"http://user:{KEY}@localhost:8000/v1",
        f"https://api.example.com/v1?api_key={KEY}",
        "https://api.example.com/v1?",
        f"https://api.example.com/v1#{KEY}",
        f"https://api.example.com:{KEY}/v1",
        "https://api.example.com:99999/v1",
        "https://api.example.com:/v1",
        "https://bad_host.example/v1",
        "https://[not-ipv6]/v1",
        "https://[::1]junk/v1",
        f"https://api.example.com /{KEY}",
    ],
)
def test_unsafe_or_malformed_urls_are_rejected_without_echoing_them(url):
    env = {remote.BASE_URL_ENV: url, remote.API_KEY_ENV: "k"}
    for call in (
        lambda: remote_config_from_env(env),
        lambda: urllib_transport(url, {}, SecretStr("k")),
    ):
        with pytest.raises(RemoteError) as info:
            call()
        assert KEY not in str(info.value) and url not in str(info.value)
    with pytest.raises(ValidationError) as info:
        RemoteConfig(base_url=url, api_key="k")
    assert KEY not in str(info.value)


def test_redirect_error_omits_the_destination(monkeypatch):
    handler = _RedirectingHTTPS()
    _record(monkeypatch, handler)
    with pytest.raises(RemoteError) as info:
        urllib_transport("https://api.example.com/v1/chat/completions", {}, SecretStr(KEY))
    assert "evil.example" not in str(info.value) and "steal" not in str(info.value)
    assert "api.example.com" not in str(info.value)


class _ErrorHTTPS(urllib.request.HTTPSHandler):
    handler_order = 100

    def https_open(self, req):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", email.message.Message(), None)


class _UnreachableHTTPS(urllib.request.HTTPSHandler):
    handler_order = 100

    def https_open(self, req):
        raise urllib.error.URLError(f"cannot reach {req.full_url}")


ERROR_URL = "https://api.example.com/v1/secret-path/chat/completions"


def test_http_error_omits_the_url(monkeypatch):
    _record(monkeypatch, _ErrorHTTPS())
    with pytest.raises(RemoteError, match="HTTP 500") as info:
        urllib_transport(ERROR_URL, {}, SecretStr(KEY))
    assert "secret-path" not in str(info.value)


def test_connection_error_omits_the_url(monkeypatch):
    url = ERROR_URL
    _record(monkeypatch, _UnreachableHTTPS())
    with pytest.raises(RemoteError, match="Could not reach") as info:
        urllib_transport(url, {}, SecretStr(KEY))
    assert "secret-path" not in str(info.value)


# --- round 4, issue 2: malformed redirect Location and overlong hostnames -------


class _MalformedRedirectHTTPS(urllib.request.HTTPSHandler):
    handler_order = 100

    def __init__(self, location: str) -> None:
        super().__init__()
        self.location = location

    def https_open(self, req):
        headers = email.message.Message()
        headers["Location"] = self.location
        resp = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, 302)
        resp.msg = "Found"
        return resp


@pytest.mark.parametrize(
    "location",
    [
        f"https://user:{KEY}@ex\uff03ample.com/steal",  # NFKC netloc check echoes it
        f"https://user:{KEY}@[bad/steal",
        f"https://user:{KEY}@evil.example:notaport/steal",
    ],
)
def test_malformed_redirect_location_is_not_echoed(monkeypatch, location):
    _record(monkeypatch, _MalformedRedirectHTTPS(location))
    with pytest.raises(RemoteError, match="redirect") as info:
        urllib_transport("https://api.example.com/v1/chat/completions", {}, SecretStr(KEY))
    text = str(info.value)
    assert KEY not in text and "user" not in text and "steal" not in text
    assert info.value.__context__ is None or info.value.__suppress_context__


def test_transport_value_errors_are_not_echoed(monkeypatch):
    class _Boom(urllib.request.HTTPSHandler):
        handler_order = 100

        def https_open(self, req):
            raise ValueError(f"bad netloc user:{KEY}@evil.example")

    _record(monkeypatch, _Boom())
    with pytest.raises(RemoteError) as info:
        urllib_transport("https://api.example.com/v1/x", {}, SecretStr(KEY))
    assert KEY not in str(info.value) and "evil" not in str(info.value)


@pytest.mark.parametrize(
    "host",
    [
        "a" * 64,  # single label over 63
        "a" * 252,
        "ok." + "a" * 64 + ".example",
        ".".join(["a" * 63] * 4),  # 255 chars total, every label legal
        ".".join(["a" * 63] * 3) + "." + "a" * 62,  # 254 chars
    ],
)
def test_overlong_hostnames_are_rejected(host):
    url = f"https://{host}/v1"
    with pytest.raises(RemoteError, match="malformed"):
        remote.check_url(url)
    with pytest.raises(ValidationError):
        RemoteConfig(base_url=url, api_key="k")


@pytest.mark.parametrize(
    "host",
    ["a" * 63 + ".example", ".".join(["a" * 63] * 3) + "." + "a" * 61],  # 253 chars
)
def test_hostnames_at_the_dns_limits_are_allowed(host):
    for suffix in ("", "."):
        url = f"https://{host}{suffix}/v1"
        assert remote.check_url(url) == url
