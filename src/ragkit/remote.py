"""Configuration and HTTP transport for optional OpenAI-compatible backends.

Nothing in this module runs unless a remote backend is explicitly configured:
``OPENAI_COMPATIBLE_BASE_URL`` and ``API_KEY`` must both be set. The default
(offline) code path never imports ``urllib.request`` or opens a socket.

The base URL must use HTTPS, except for explicit loopback hosts (``localhost``,
``127.0.0.0/8``, ``::1``) where plain HTTP is allowed for local servers.
The URL may not carry userinfo (``user:pass@``), a query or a fragment, and
its host and port must be well formed. Redirects are never followed and, for
plain-HTTP loopback URLs, environment proxies are ignored, so the bearer token
is only ever sent to the configured origin. Error messages never echo the URL
(it may embed credentials) or a redirect destination. The API key is held as a
``SecretStr`` and unwrapped only when the request is built.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import SplitResult, urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

if TYPE_CHECKING:
    import urllib.request

BASE_URL_ENV = "OPENAI_COMPATIBLE_BASE_URL"
API_KEY_ENV = "API_KEY"
LLM_MODEL_ENV = "RAGKIT_LLM_MODEL"
EMBED_MODEL_ENV = "RAGKIT_EMBED_MODEL"
EMBEDDINGS_ENV = "RAGKIT_EMBEDDINGS"

DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
TIMEOUT_SECONDS = 60

# (url, payload, api_key) -> parsed JSON response
Transport = Callable[[str, dict[str, Any], SecretStr], dict[str, Any]]


class RemoteError(RuntimeError):
    """A remote backend is misconfigured or a request to it failed."""


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_MAX_HOSTNAME = 253
_HOSTNAME_RE = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$")


def _valid_host(parts: SplitResult) -> bool:
    """True if the netloc is ``host``, ``host:port``, ``[ipv6]`` or ``[ipv6]:port``."""
    netloc = parts.netloc
    if netloc.startswith("["):
        literal, _, rest = netloc[1:].partition("]")
        try:
            ipaddress.IPv6Address(literal)
        except ValueError:
            return False
    else:
        literal, sep, port = netloc.partition(":")
        # RFC 1035/1123: labels <= 63 chars (regex), whole name <= 253 (sans root dot).
        if len(literal.removesuffix(".")) > _MAX_HOSTNAME or not _HOSTNAME_RE.match(literal):
            return False
        rest = sep + port
    # Whatever follows the host must be empty or a non-empty numeric port.
    return rest == "" or (rest.startswith(":") and rest[1:].isdigit())


def _is_plain_loopback(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "http" and _is_loopback(parts.hostname or "")


def check_url(url: str) -> str:
    """Return ``url`` if it is a safe HTTPS (or loopback HTTP) URL, else raise RemoteError.

    Error messages deliberately do not include the URL: it may embed credentials.
    """
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError for a malformed or out-of-range port
    except ValueError:
        raise RemoteError("Remote URL is malformed (bad host or port).") from None
    if any(ch.isspace() or not ch.isprintable() for ch in url):
        raise RemoteError("Remote URL is malformed (contains whitespace or control characters).")
    if parts.scheme not in ("http", "https"):
        raise RemoteError("Refusing remote URL: use https://.")
    if "@" in parts.netloc or parts.username is not None or parts.password is not None:
        raise RemoteError(
            "Refusing remote URL with embedded credentials (user:pass@); set API_KEY instead."
        )
    if parts.query or parts.fragment or "?" in url or "#" in url:
        raise RemoteError("Refusing remote URL with a query string or fragment.")
    if not parts.hostname:
        raise RemoteError("Remote URL has no host.")
    if not _valid_host(parts) or port == 0:
        raise RemoteError("Remote URL is malformed (bad host or port).")
    if parts.scheme == "https":
        return url
    if _is_loopback(parts.hostname):
        return url
    raise RemoteError(
        "Refusing remote URL: use https:// (plain http is only allowed for "
        "localhost / loopback addresses)."
    )


class RemoteConfig(BaseModel):
    # Validation errors must not echo the base URL (it may embed credentials).
    model_config = ConfigDict(hide_input_in_errors=True)

    base_url: str
    api_key: SecretStr
    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL

    @field_validator("base_url")
    @classmethod
    def _secure_base_url(cls, value: str) -> str:
        try:
            return check_url(value)
        except RemoteError as exc:
            raise ValueError(str(exc)) from None

    def endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


def remote_config_from_env(env: dict[str, str] | None = None) -> RemoteConfig | None:
    """Return a config when both the base URL and API key are set, else None.

    Raises RemoteError if they are set but the base URL is not allowed.
    """
    env = dict(os.environ) if env is None else env
    base_url = env.get(BASE_URL_ENV, "").strip()
    api_key = env.get(API_KEY_ENV, "").strip()
    if not base_url or not api_key:
        return None
    check_url(base_url)
    return RemoteConfig(
        base_url=base_url,
        api_key=SecretStr(api_key),
        llm_model=env.get(LLM_MODEL_ENV) or DEFAULT_LLM_MODEL,
        embed_model=env.get(EMBED_MODEL_ENV) or DEFAULT_EMBED_MODEL,
    )


def _no_redirect_handler() -> urllib.request.HTTPRedirectHandler:
    import urllib.request

    def refuse(code: int) -> RemoteError:
        return RemoteError(
            f"Remote server answered HTTP {code} with a redirect; redirects are "
            "not followed so the API key is never sent to another origin."
        )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        # Refuse before urllib parses ``Location``: its parser raises ValueErrors that
        # echo the (possibly credential-bearing) header for malformed authorities.
        def http_error_302(self, req, fp, code, msg, headers):  # noqa: ANN001
            fp.close()
            raise refuse(code)

        http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
            raise refuse(code)

    return NoRedirect()


def _build_opener(direct: bool = False) -> urllib.request.OpenerDirector:
    """Opener that refuses redirects. ``direct`` also disables environment proxies,
    so a plaintext loopback request (token + payload) never goes to an HTTP proxy."""
    import urllib.request

    handlers: list[urllib.request.BaseHandler] = [_no_redirect_handler()]
    if direct:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def urllib_transport(url: str, payload: dict[str, Any], api_key: SecretStr) -> dict[str, Any]:
    """POST JSON with a bearer token to an allowed URL and return the decoded JSON."""
    import http.client
    import urllib.error
    import urllib.request

    check_url(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Unredirected headers are never copied onto a follow-up request.
    request.add_unredirected_header("Authorization", f"Bearer {api_key.get_secret_value()}")
    try:
        opener = _build_opener(direct=_is_plain_loopback(url))
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RemoteError(f"Remote server returned HTTP {exc.code}.") from None
    except urllib.error.URLError as exc:
        # Only an OS-level description; never str(exc), which may contain the URL.
        reason = getattr(exc.reason, "strerror", None) or type(exc.reason).__name__
        raise RemoteError(f"Could not reach the remote server ({reason}).") from None
    except (ValueError, http.client.HTTPException):
        # e.g. a malformed redirect ``Location``; never echo the message (it may
        # contain the header value, including credentials).
        raise RemoteError("Remote server sent an invalid redirect or response.") from None
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError:
        raise RemoteError("Remote server did not return JSON.") from None
