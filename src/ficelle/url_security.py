from __future__ import annotations

from urllib.parse import urlparse


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def uses_secure_http_transport(url: str) -> bool:
    """Whether credentials may be sent to ``url`` without crossing cleartext networking."""
    parsed = urlparse(url)
    return parsed.scheme == "https" or (
        parsed.scheme == "http"
        and (parsed.hostname or "") in _LOOPBACK_HOSTS
    )
