from __future__ import annotations

import hmac
from urllib.parse import urlparse


def is_loopback_origin(origin: str) -> bool:
    try:
        parsed = urlparse(origin)
    except Exception:
        return False
    return parsed.scheme == "http" and (parsed.hostname or "") in {"127.0.0.1", "localhost", "::1"}


def admin_origin_allowed(origin: str | None, server_port: int) -> bool:
    if origin is None:
        return True
    if not is_loopback_origin(origin):
        return False
    try:
        origin_port = urlparse(origin).port
    except Exception:
        return False
    return origin_port == server_port


def admin_token_matches(presented: str | None, expected: str) -> bool:
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8", "ignore"), expected.encode("utf-8"))
