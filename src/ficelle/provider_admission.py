from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


ADMISSION_WINDOW_SECONDS = 60.0
ADMISSION_RATE_HEADROOM = 0.9

# A header observation is an estimate of what the provider had left when it answered. Past this
# age (or once its own reset time has elapsed) it is shown as stale and never as a live gauge.
OBSERVATION_STALE_SECONDS = 900.0

# A reset horizon up to two minutes is a per-minute bucket; anything longer is a daily one. The
# label only keeps the two buckets apart: admission trusts the provider's own remaining/reset.
MINUTE_WINDOW_MAX_RESET_SECONDS = 120.0
# Two observations of one bucket agree on its reset instant up to header rounding and latency.
SAME_RESET_TOLERANCE_SECONDS = 5.0

METRIC_REQUESTS = "requests"
METRIC_TOKENS = "tokens"
WINDOW_MINUTE = "minute"
WINDOW_DAY = "day"
# Only request buckets gate admission. Token buckets are observation-only until a conservative
# prompt-token estimate exists (provider-quota-intelligence R5/R6 follow-up).
_OBSERVED_METRICS = (METRIC_REQUESTS, METRIC_TOKENS)


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    retry_after_seconds: int | None = None
    limit: int | None = None
    remaining: int | None = None
    source: str = "unknown_limit"


class ProviderAdmissionRefused(RuntimeError):
    def __init__(self, source: str, retry_after_seconds: int, *, model_scoped: bool = False) -> None:
        super().__init__(f"local provider admission delayed {source} for {retry_after_seconds}s")
        self.source = source
        self.retry_after_seconds = retry_after_seconds
        # True when the refusing bucket belongs to this one upstream model, so its siblings on the
        # same provider are still admissible.
        self.model_scoped = model_scoped


@dataclass
class ObservedLimit:
    limit: int
    remaining: int
    reset_at: float
    observed_at: float
    observed_epoch: float


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


def _header_count(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) else None


def _window_kind(reset_seconds: float) -> str:
    return WINDOW_MINUTE if reset_seconds <= MINUTE_WINDOW_MAX_RESET_SECONDS else WINDOW_DAY


def _declared_rpm(provider_cfg: Any) -> float | None:
    raw = provider_cfg.get("rate_limit_rpm") if isinstance(provider_cfg, dict) else None
    try:
        rpm = float(raw)
    except (TypeError, ValueError):
        return None
    return rpm if math.isfinite(rpm) and rpm > 0 else None


def admission_scope_keys(provider_cfg: Any, upstream_id: str) -> tuple[str, str]:
    """``(declared_scope, observed_scope)`` bucket keys for one upstream model.

    The declared RPM stays provider-wide unless the provider says ``rate_limit_scope: model``.
    Response headers describe the bucket of the model that answered, so they are model-scoped
    unless the provider explicitly says ``rate_limit_scope: provider``: one model's exhausted
    bucket must not hold out its siblings.
    """
    scope = str(provider_cfg.get("rate_limit_scope") or "") if isinstance(provider_cfg, dict) else ""
    model = str(upstream_id or "")
    return (model if scope == "model" else ""), ("" if scope == "provider" else model)


@dataclass
class ProviderAdmissionLedger:
    """One-process rolling RPM guard for provider limits Ficelle actually knows.

    Unknown or malformed limits fail open. Reservations happen immediately before dispatch and
    remain consumed regardless of the upstream outcome, because providers commonly count failed
    requests too. Live traffic never queues here and can fail over; probes skip without producing
    model-quality evidence when live traffic has consumed the known allowance.

    The declared RPM is always enforced. Provider response headers add one observed bucket per
    (scope, metric, window kind) on top of it: they can only make admission stricter, never lift
    the declared guard. Token buckets are recorded for the admin gauge but never gate a request.
    All state is process-local and resets on restart.
    """

    monotonic: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    _requests: dict[tuple[str, str, str], deque[float]] = field(default_factory=dict)
    _observed: dict[tuple[str, str, str, str, str], ObservedLimit] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _limit(provider_cfg: Any) -> int | None:
        rpm = _declared_rpm(provider_cfg)
        return None if rpm is None else max(1, int(math.floor(rpm * ADMISSION_RATE_HEADROOM)))

    def reserve(
        self,
        source: str,
        provider_cfg: Any,
        *,
        namespace: str = "",
        scope_key: str = "",
        observed_scope_key: str | None = None,
    ) -> AdmissionDecision:
        namespace, source = str(namespace or ""), str(source or "")
        key = (namespace, source, str(scope_key or ""))
        observed_scope = str(scope_key or "") if observed_scope_key is None else str(observed_scope_key)
        now = self.monotonic()
        with self._lock:
            live: list[ObservedLimit] = []
            exhausted: list[ObservedLimit] = []
            for window in (WINDOW_MINUTE, WINDOW_DAY):
                observed = self._observed.get((namespace, source, observed_scope, METRIC_REQUESTS, window))
                # An elapsed bucket is kept for the admin gauge (as stale) but no longer gates.
                if observed is None or observed.reset_at <= now:
                    continue
                (exhausted if observed.remaining <= 0 else live).append(observed)
            if exhausted:
                blocking = max(exhausted, key=lambda item: item.reset_at)
                return AdmissionDecision(
                    allowed=False,
                    retry_after_seconds=max(1, int(math.ceil(blocking.reset_at - now))),
                    limit=blocking.limit,
                    remaining=0,
                    source="provider_headers",
                )
            limit = self._limit(provider_cfg)
            # `rows` exists exactly when a declared limit does.
            rows = self._requests.setdefault(key, deque()) if limit is not None else None
            if rows is not None:
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
            # Both guards passed: consume from each only now, so a refusal by one never spends the other.
            for observed in live:
                observed.remaining -= 1
            candidates: list[tuple[int, int, str]] = [
                (observed.remaining, observed.limit, "provider_headers") for observed in live
            ]
            if rows is not None:
                rows.append(now)
                candidates.append((max(0, limit - len(rows)), limit, "declared_rpm"))
            if not candidates:
                return AdmissionDecision(allowed=True)
            remaining, binding_limit, binding_source = min(candidates, key=lambda item: item[0])
            return AdmissionDecision(
                allowed=True,
                limit=binding_limit,
                remaining=remaining,
                source=binding_source,
            )

    def observe(
        self,
        source: str,
        headers: Any,
        *,
        namespace: str = "",
        scope_key: str = "",
    ) -> None:
        if not hasattr(headers, "items"):
            return
        normalized = {str(key).lower(): value for key, value in headers.items()}
        now = self.monotonic()
        epoch = self.wall()
        with self._lock:
            # Bounded memory: a bucket this far past its reset neither gates nor shows as live.
            expired = [
                key for key, item in self._observed.items() if item.reset_at + OBSERVATION_STALE_SECONDS < now
            ]
            for key in expired:
                del self._observed[key]
        for metric in _OBSERVED_METRICS:
            limit = _header_count(normalized.get(f"x-ratelimit-limit-{metric}"))
            remaining = _header_count(normalized.get(f"x-ratelimit-remaining-{metric}"))
            if limit is None or remaining is None:
                continue
            reset_value = normalized.get(f"x-ratelimit-reset-{metric}")
            if metric == METRIC_REQUESTS:
                # Retry-After on a 429 is the request bucket's own wait; it says nothing about tokens.
                reset_value = reset_value or normalized.get("retry-after")
            reset_seconds = _reset_seconds(reset_value)
            if limit <= 0 or remaining < 0 or reset_seconds is None:
                continue
            window = _window_kind(reset_seconds)
            key = (str(namespace or ""), str(source or ""), str(scope_key or ""), metric, window)
            sibling_key = (*key[:4], WINDOW_DAY if window == WINDOW_MINUTE else WINDOW_MINUTE)
            with self._lock:
                # One long window draining through the 120s boundary changes kind, not bucket: a
                # sibling that resets at the same instant is this bucket's older copy, updated
                # under its own key so the bucket that does hold this key is left alone.
                sibling = self._observed.get(sibling_key)
                if sibling is not None and abs(sibling.reset_at - (now + reset_seconds)) <= SAME_RESET_TOLERANCE_SECONDS:
                    key = sibling_key
                self._observed[key] = ObservedLimit(
                    limit=limit,
                    remaining=min(limit, remaining),
                    reset_at=now + reset_seconds,
                    observed_at=now,
                    observed_epoch=epoch,
                )

    def snapshot(self, providers: Mapping[str, Any], *, namespace: str = "") -> dict[str, dict[str, Any]]:
        """Read-only, redacted view of what this process knows about each provider's allowance.

        One row per configured source plus any source with observations. Process-local: a restart
        empties the observations, and the row then says so (``unknown``/``declared_only``) rather
        than inventing a number. Only fields derived from limit/remaining/reset headers leave here.
        """
        namespace = str(namespace or "")
        now = self.monotonic()
        with self._lock:
            observed_by_source: dict[str, list[tuple[str, str, str, ObservedLimit]]] = {}
            for (ns, source, scope, metric, window), observed in self._observed.items():
                if ns == namespace:
                    observed_by_source.setdefault(source, []).append(
                        (scope, metric, window, ObservedLimit(**vars(observed)))
                    )
        sources = {str(name) for name in providers} | set(observed_by_source)
        return {
            source: self._snapshot_row(source, providers.get(source), observed_by_source.get(source, []), now)
            for source in sorted(sources)
        }

    def _snapshot_row(
        self,
        source: str,
        provider_cfg: Any,
        observed: list[tuple[str, str, str, ObservedLimit]],
        now: float,
    ) -> dict[str, Any]:
        def is_fresh(metric: str, item: ObservedLimit) -> bool:
            # An exhausted request bucket gates `reserve()` until its reset, and a refused model is
            # never re-observed: it stays live for that whole time rather than aging into "stale".
            gating = metric == METRIC_REQUESTS and item.remaining <= 0
            return item.reset_at > now and (gating or now - item.observed_at <= OBSERVATION_STALE_SECONDS)

        def bucket_row(
            limit: int | float,
            remaining: int | None,
            reset_in_seconds: int | None,
            source: str,
            observed_at: str | None,
            scope: str,
            fresh: bool,
            window: str,
            **extra: Any,
        ) -> dict[str, Any]:
            # One shape for observed and declared-only buckets, which the dashboard reads alike.
            return {
                "limit": limit,
                "remaining": remaining,
                "reset_in_seconds": reset_in_seconds,
                "source": source,
                "observed_at": observed_at,
                "scope": scope,
                "fresh": fresh,
                "window": window,
                **extra,
            }

        def metric_row(metric: str) -> dict[str, Any] | None:
            rows = [(scope, window, item) for scope, name, window, item in observed if name == metric]
            if not rows:
                return None
            fresh = [row for row in rows if is_fresh(metric, row[2])]
            if fresh:
                # The tightest bucket is the one that will bind the next request.
                scope, window, item = min(fresh, key=lambda row: row[2].remaining / row[2].limit)
            else:
                scope, window, item = max(rows, key=lambda row: row[2].observed_at)
            return bucket_row(
                item.limit,
                item.remaining,
                max(0, int(math.ceil(item.reset_at - now))) if item.reset_at > now else None,
                "provider_headers",
                datetime.fromtimestamp(item.observed_epoch, timezone.utc).isoformat(timespec="seconds"),
                "model" if scope else "provider",
                bool(fresh),
                window,
                age_seconds=max(0, int(now - item.observed_at)),
                **({"model": scope} if scope else {}),
            )

        requests_row = metric_row(METRIC_REQUESTS)
        tokens_row = metric_row(METRIC_TOKENS)
        rpm = _declared_rpm(provider_cfg)
        declared = None
        if rpm is not None:
            declared = {
                "rpm": int(rpm) if rpm == int(rpm) else rpm,
                "admission_limit": self._limit(provider_cfg),
                "headroom_percent": round((1 - ADMISSION_RATE_HEADROOM) * 100),
                "scope": "model" if str(provider_cfg.get("rate_limit_scope") or "") == "model" else "provider",
            }
            if requests_row is None:
                requests_row = bucket_row(
                    declared["rpm"], None, None, "declared", None, declared["scope"], False, WINDOW_MINUTE
                )
        if any(row and row["fresh"] for row in (requests_row, tokens_row)):
            state = "observed"
        elif observed:
            state = "stale"
        elif declared is not None:
            state = "declared_only"
        else:
            state = "unknown"
        return {
            "state": state,
            "requests": requests_row,
            "tokens": tokens_row,
            "declared": declared,
            "stale_after_seconds": int(OBSERVATION_STALE_SECONDS),
            "process_local": True,
        }


PROVIDER_ADMISSION_LEDGER = ProviderAdmissionLedger()
