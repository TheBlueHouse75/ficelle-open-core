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


def connectable_http_url(bind_host: str | None, port: int, path: str = "") -> str:
    """Build an HTTP URL to a router bind address, including valid IPv6 brackets."""
    host = connectable_host(bind_host)
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    normalized_path = f"/{path.lstrip('/')}" if path else ""
    return f"http://{url_host}:{port}{normalized_path}"


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
