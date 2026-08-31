from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


ADMISSION_WINDOW_SECONDS = 60.0
ADMISSION_RATE_HEADROOM = 0.9


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    retry_after_seconds: int | None = None
    limit: int | None = None
    remaining: int | None = None
    source: str = "unknown_limit"


class ProviderAdmissionRefused(RuntimeError):
    def __init__(self, source: str, retry_after_seconds: int) -> None:
        super().__init__(f"local provider admission delayed {source} for {retry_after_seconds}s")
        self.source = source
        self.retry_after_seconds = retry_after_seconds


@dataclass
class ObservedRequestLimit:
    limit: int
    remaining: int
    reset_at: float


_DURATION_PART = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)", re.IGNORECASE)


def _reset_seconds(value: Any) -> float | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None and math.isfinite(numeric) and 0 < numeric <= 86_400:
        return numeric
    factors = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    parts = _DURATION_PART.findall(text)
    if not parts:
        return None
    seconds = sum(float(amount) * factors[unit.lower()] for amount, unit in parts)
    return min(seconds, 86_400.0) if seconds > 0 else None


@dataclass
class ProviderAdmissionLedger:
    """One-process rolling RPM guard for provider limits Ficelle actually knows.

    Unknown or malformed limits fail open. Reservations happen immediately before dispatch and
    remain consumed regardless of the upstream outcome, because providers commonly count failed
    requests too. Live traffic never queues here and can fail over; probes skip without producing
    model-quality evidence when live traffic has consumed the known allowance.
    """

    monotonic: Callable[[], float] = time.monotonic
    _requests: dict[tuple[str, str], deque[float]] = field(default_factory=dict)
    _observed: dict[tuple[str, str], ObservedRequestLimit] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _limit(provider_cfg: Any) -> int | None:
        raw = provider_cfg.get("rate_limit_rpm") if isinstance(provider_cfg, dict) else None
        try:
            rpm = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(rpm) or rpm <= 0:
            return None
        return max(1, int(math.floor(rpm * ADMISSION_RATE_HEADROOM)))

    def reserve(
        self,
        source: str,
        provider_cfg: Any,
        *,
        namespace: str = "",
    ) -> AdmissionDecision:
        key = (str(namespace or ""), str(source or ""))
        now = self.monotonic()
        with self._lock:
            observed = self._observed.get(key)
            if observed is not None:
                if observed.reset_at <= now:
                    self._observed.pop(key, None)
                elif observed.remaining <= 0:
                    return AdmissionDecision(
                        allowed=False,
                        retry_after_seconds=max(1, int(math.ceil(observed.reset_at - now))),
                        limit=observed.limit,
                        remaining=0,
                        source="provider_headers",
                    )
                else:
                    observed.remaining -= 1
                    return AdmissionDecision(
                        allowed=True,
                        limit=observed.limit,
                        remaining=observed.remaining,
                        source="provider_headers",
                    )
            limit = self._limit(provider_cfg)
            if limit is None:
                return AdmissionDecision(allowed=True)
            rows = self._requests.setdefault(key, deque())
            cutoff = now - ADMISSION_WINDOW_SECONDS
            while rows and rows[0] <= cutoff:
                rows.popleft()
            if len(rows) >= limit:
                retry_after = max(1, int(math.ceil(rows[0] + ADMISSION_WINDOW_SECONDS - now)))
                return AdmissionDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                    limit=limit,
                    remaining=0,
                    source="declared_rpm",
                )
            rows.append(now)
            return AdmissionDecision(
                allowed=True,
                limit=limit,
                remaining=max(0, limit - len(rows)),
                source="declared_rpm",
            )

    def observe(
        self,
        source: str,
        headers: Any,
        *,
        namespace: str = "",
    ) -> None:
        if not hasattr(headers, "items"):
            return
        normalized = {str(key).lower(): value for key, value in headers.items()}
        try:
            limit = int(float(normalized.get("x-ratelimit-limit-requests")))
            remaining = int(float(normalized.get("x-ratelimit-remaining-requests")))
        except (TypeError, ValueError):
            return
        reset_seconds = _reset_seconds(
            normalized.get("x-ratelimit-reset-requests")
            or normalized.get("retry-after")
        )
        if limit <= 0 or remaining < 0 or reset_seconds is None:
            return
        key = (str(namespace or ""), str(source or ""))
        with self._lock:
            self._observed[key] = ObservedRequestLimit(
                limit=limit,
                remaining=min(limit, remaining),
                reset_at=self.monotonic() + reset_seconds,
            )

PROVIDER_ADMISSION_LEDGER = ProviderAdmissionLedger()
