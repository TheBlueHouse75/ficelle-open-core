from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping


MIN_RETRY_SECONDS = 1
MAX_RETRY_SECONDS = 86_400
_DURATION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*$", re.IGNORECASE)
_BODY_KEYS = frozenset({"retrydelay", "retry_after", "retryafter"})


@dataclass(frozen=True)
class RetryHint:
    seconds: int
    source: str


def _bounded_seconds(value: float) -> int | None:
    if not math.isfinite(value) or value < 0:
        return None
    return max(MIN_RETRY_SECONDS, min(MAX_RETRY_SECONDS, math.ceil(value)))


def _duration_seconds(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _bounded_seconds(float(value))
    match = _DURATION_PATTERN.fullmatch(str(value or ""))
    if match is None:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    multiplier = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return _bounded_seconds(amount * multiplier)


def _header_value(headers: Mapping[str, Any], name: str) -> Any:
    needle = name.lower()
    for key, value in headers.items():
        if str(key).lower() == needle:
            return value
    return None


def _standard_retry_after(value: Any, now: datetime) -> int | None:
    text = str(value or "").strip()
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None:
        return _bounded_seconds(numeric)
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return _bounded_seconds((parsed - now).total_seconds())


def _body_retry_hint(value: Any, *, depth: int = 0) -> int | None:
    if depth > 8:
        return None
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _BODY_KEYS:
                seconds = _duration_seconds(child)
                if seconds is not None:
                    return seconds
            seconds = _body_retry_hint(child, depth=depth + 1)
            if seconds is not None:
                return seconds
    elif isinstance(value, list):
        for child in value[:32]:
            seconds = _body_retry_hint(child, depth=depth + 1)
            if seconds is not None:
                return seconds
    return None


def retry_hint(
    headers: Mapping[str, Any] | None,
    body: str | bytes | Mapping[str, Any] | list[Any] | None,
    *,
    now: datetime | None = None,
) -> RetryHint | None:
    """Return a bounded provider retry delay without retaining raw upstream metadata."""
    safe_headers = headers or {}
    for name in ("retry-after-ms", "x-ms-retry-after-ms"):
        value = _header_value(safe_headers, name)
        if value is None:
            continue
        try:
            milliseconds = float(value)
        except (TypeError, ValueError):
            continue
        seconds = _bounded_seconds(milliseconds / 1000.0)
        if seconds is not None:
            return RetryHint(seconds=seconds, source=name)

    value = _header_value(safe_headers, "retry-after")
    if value is not None:
        seconds = _standard_retry_after(value, now or datetime.now(timezone.utc))
        if seconds is not None:
            return RetryHint(seconds=seconds, source="retry-after")

    parsed_body: Any = body
    if isinstance(body, bytes):
        try:
            parsed_body = body.decode("utf-8")
        except UnicodeDecodeError:
            parsed_body = None
    if isinstance(parsed_body, str):
        try:
            parsed_body = json.loads(parsed_body)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            parsed_body = None
    seconds = _body_retry_hint(parsed_body)
    return RetryHint(seconds=seconds, source="body") if seconds is not None else None
