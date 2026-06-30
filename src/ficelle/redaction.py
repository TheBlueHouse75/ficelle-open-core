from __future__ import annotations

import re
from typing import Any


SENSITIVE_ERROR_PATTERNS = [
    re.compile(r"(?i)[\"']\bauthorization[\"']\s*[:=]\s*[\"']?[^\"'}]+"),
    re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:\S+\s+)?\S+"),
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(
        r"(?i)(?:[\"']?\b[A-Z0-9_]*api[_-]?key[\"']?|[\"']?\baccess[_-]?token[\"']?|[\"']?\brefresh[_-]?token[\"']?)\s*[:=]\s*[\"']?[^\"'\s,}]+"
    ),
    re.compile(r"(?i)[\"']\btoken[\"']\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{8,}"),
    re.compile(r"(?i)(?<!plain text )(?<![A-Z0-9_])token\s*[:=]\s*[A-Za-z0-9_./+=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
]
SENSITIVE_JSON_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
}


def is_sensitive_json_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return (
        normalized in SENSITIVE_JSON_KEYS
        or normalized.endswith("_api_key")
        or normalized.endswith("_access_token")
        or normalized.endswith("_refresh_token")
    )


def redact_sensitive_text(value: Any, limit: int | None = None, *, collapse: bool = False) -> str | None:
    text = str(value or "")
    if collapse:
        text = " ".join(text.split())
    if not text:
        return None
    for pattern in SENSITIVE_ERROR_PATTERNS:
        text = pattern.sub("[redacted]", text)
    if limit is not None:
        return text[:limit]
    return text


def redact_sensitive_json(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value) or ""
    if isinstance(value, list):
        return [redact_sensitive_json(item) for item in value]
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = redact_sensitive_text(key) or str(key)
            if is_sensitive_json_key(key) and item not in (None, ""):
                safe[safe_key] = "[redacted]"
            else:
                safe[safe_key] = redact_sensitive_json(item)
        return safe
    return value


def sanitize_error_detail(value: Any, limit: int = 180) -> str | None:
    return redact_sensitive_text(value, limit, collapse=True)
