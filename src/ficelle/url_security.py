from __future__ import annotations

from urllib.parse import urlparse


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "*"})


def connectable_host(bind_host: str | None) -> str:
    """Turn a *bind* address into one a client can actually connect to.

    ``0.0.0.0`` means "listen on every interface"; it is not a destination, and the router's
    ``Host`` allowlist deliberately refuses it (a browser can reach ``http://0.0.0.0:<port>``
    on some platforms, so accepting it as a Host would reopen the rebinding hole the allowlist
    closes). Callers that build a URL *to* the router must therefore resolve a wildcard bind to
    loopback rather than echoing it back.
    """
    host = str(bind_host or "").strip()
    if not host or host in {"0.0.0.0", "*"}:
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def uses_secure_http_transport(url: str) -> bool:
    """Whether credentials may be sent to ``url`` without crossing cleartext networking."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme == "https" or (
        parsed.scheme == "http"
        and (parsed.hostname or "") in _LOOPBACK_HOSTS
    )
