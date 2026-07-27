from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ficelle.redaction import sanitize_error_detail


FailureReason = Literal[
    "auth_or_credit",
    "billing_or_paid",
    "model_not_found",
    "no_free_quota",
    "quota_exhausted",
    "rate_limited",
    "server_error",
    "unavailable",
]

FALSE_FREE_TEXT_MARKERS = (
    "payment required",
    "requires payment",
    "billing",
    "credit",
    "credits",
    "insufficient balance",
    "quota exceeded",
    "not enough balance",
    "free promotion has ended",
)

QUOTA_EXHAUSTED_TEXT_MARKERS = (
    "quota exceeded",
    "quota exhausted",
    "credit exhausted",
    "credits exhausted",
    "free tier exhausted",
    "free-tier exhausted",
    "free tier quota",
    "free-tier quota",
    "trial quota",
    "usage limit exceeded",
    "monthly usage limit",
)

# Structural zero free-tier allocation markers. A positive match quarantines a
# quota-free model instead of treating it as transient quota exhaustion.
FREE_TIER_ZERO_ALLOCATION_MARKERS = (
    "limit: 0",
    "limit:0",
)

# Structural "this upstream model id is not serveable" markers. A positive
# match quarantines a model instead of cooling it for repeated re-probes.
MODEL_NOT_FOUND_TEXT_MARKERS = (
    "not found",
    "does not exist",
    "no longer available",
    "decommissioned",
    "unknown model",
    "model_not_found",
    "gone",
)


@dataclass(frozen=True)
class FailureMarkers:
    false_free: tuple[str, ...] = FALSE_FREE_TEXT_MARKERS
    quota_exhausted: tuple[str, ...] = QUOTA_EXHAUSTED_TEXT_MARKERS
    free_tier_zero_allocation: tuple[str, ...] = FREE_TIER_ZERO_ALLOCATION_MARKERS
    model_not_found: tuple[str, ...] = MODEL_NOT_FOUND_TEXT_MARKERS

    def with_extra(
        self,
        *,
        false_free: tuple[str, ...] = (),
        quota_exhausted: tuple[str, ...] = (),
        free_tier_zero_allocation: tuple[str, ...] = (),
        model_not_found: tuple[str, ...] = (),
    ) -> "FailureMarkers":
        return FailureMarkers(
            false_free=self.false_free + false_free,
            quota_exhausted=self.quota_exhausted + quota_exhausted,
            free_tier_zero_allocation=self.free_tier_zero_allocation + free_tier_zero_allocation,
            model_not_found=self.model_not_found + model_not_found,
        )


DEFAULT_FAILURE_MARKERS = FailureMarkers()
PROVIDER_SCOPED_COOLDOWN_REASONS = {"rate_limited", "auth_or_credit"}
PROVIDER_ERROR_REASONS = PROVIDER_SCOPED_COOLDOWN_REASONS | {"quota_exhausted", "no_free_quota"}
BENCHMARK_ROUTE_BLOCKING_REASONS = {
    "billing_or_paid",
    "no_free_quota",
    "quota_exhausted",
    "auth_or_credit",
    "model_not_found",
}


@dataclass(frozen=True)
class CooldownQuarantinePolicy:
    reason: str
    source: str
    fallback_note: str


@dataclass(frozen=True)
class CooldownPolicy:
    record_provider_error: bool
    provider_cooldown: bool = False
    provider_cooldown_source: str = ""
    quota_cooldown: bool = False
    quarantine: CooldownQuarantinePolicy | None = None
    model_cooldown: bool = True


def cooldown_policy_for_reason(reason: str, *, source: str = "") -> CooldownPolicy:
    """Return the state-write policy for a classified provider failure."""
    provider_source = str(source or "").strip()
    if reason == "no_free_quota":
        return CooldownPolicy(
            record_provider_error=True,
            quarantine=CooldownQuarantinePolicy(
                reason="no_free_quota",
                source="no_free_quota_guard",
                fallback_note="runtime reported a zero free-tier allocation for a quota-free model",
            ),
            model_cooldown=False,
        )
    if reason == "model_not_found":
        return CooldownPolicy(
            record_provider_error=False,
            quarantine=CooldownQuarantinePolicy(
                reason="model_not_found",
                source="model_not_found_guard",
                fallback_note="runtime reported 404/410 for this model id (catalog entry not deployed for this account)",
            ),
            model_cooldown=False,
        )
    if reason == "quota_exhausted":
        return CooldownPolicy(
            record_provider_error=True,
            quota_cooldown=True,
            model_cooldown=False,
        )
    if reason in PROVIDER_SCOPED_COOLDOWN_REASONS:
        return CooldownPolicy(
            record_provider_error=True,
            provider_cooldown=True,
            provider_cooldown_source=provider_source,
        )
    if reason == "billing_or_paid":
        return CooldownPolicy(
            record_provider_error=False,
            quarantine=CooldownQuarantinePolicy(
                reason="billing_or_paid",
                source="anti_false_free_guard",
                fallback_note="runtime reported billing/payment/credit for a catalog-free model",
            ),
        )
    return CooldownPolicy(record_provider_error=False)


def classify_failure(
    status_code: int,
    text: str,
    *,
    normalized_free_access: dict[str, Any] | None = None,
    markers: FailureMarkers | None = None,
) -> FailureReason:
    """Classify a provider failure using already-normalized free-access metadata."""
    marker_set = markers or DEFAULT_FAILURE_MARKERS
    lower = text.lower()
    has_quota_marker = any(marker in lower for marker in marker_set.quota_exhausted)
    access = normalized_free_access if isinstance(normalized_free_access, dict) else {}
    if has_quota_marker and access.get("eligible") is True and access.get("mode") == "quota_free":
        if status_code in {402, 429}:
            if any(marker in lower for marker in marker_set.free_tier_zero_allocation):
                return "no_free_quota"
            return "quota_exhausted"
        if status_code in {401, 403}:
            return "auth_or_credit"
        if status_code >= 500:
            return "server_error"
        return "unavailable"
    if status_code == 429:
        return "rate_limited"
    if status_code == 402 or any(marker in lower for marker in marker_set.false_free):
        return "billing_or_paid"
    if status_code in {401, 403}:
        return "auth_or_credit"
    if status_code in {404, 410} and any(marker in lower for marker in marker_set.model_not_found):
        return "model_not_found"
    if status_code >= 500:
        return "server_error"
    return "unavailable"


def error_body(
    exc: Exception,
    *,
    error_type: str,
    fallback_message: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "message": sanitize_error_detail(exc) or fallback_message,
        "type": error_type,
    }
    if request_id:
        error["request_id"] = request_id
    return {"error": error}


def safe_error_body(exc: Exception, *, request_id: str | None = None) -> dict[str, Any]:
    return error_body(exc, error_type=type(exc).__name__, fallback_message="internal error", request_id=request_id)


def bad_request_body(exc: Exception, *, request_id: str | None = None) -> dict[str, Any]:
    return error_body(exc, error_type="bad_request", fallback_message="bad request", request_id=request_id)


def upstream_failure_actions(reason_counts: dict[str, int]) -> list[str]:
    actions: list[str] = []
    if reason_counts.get("auth_or_credit"):
        actions.append("Check provider credentials/credits, then clear the provider cooldown after fixing it.")
    if reason_counts.get("rate_limited"):
        actions.append("Wait for the provider cooldown or switch this profile to another healthy provider.")
    if reason_counts.get("billing_or_paid"):
        actions.append("Inspect the anti false-free guard result; refresh the catalog before re-enabling the model.")
    if reason_counts.get("no_free_quota"):
        actions.append("Provider reported a zero free-tier allocation (limit: 0) for this model; it is quarantined and will stop being proposed. Use a free-tier model or upgrade the account.")
    if reason_counts.get("model_not_found"):
        actions.append("Provider returned 404/410 for this model id (listed in the catalog but not deployed for this account); it is quarantined and will stop being benchmarked. Clear the quarantine to retry if the provider deploys it later.")
    if reason_counts.get("server_error"):
        actions.append("Retry after the short model cooldown or quarantine the unstable upstream.")
    if reason_counts.get("timeout"):
        actions.append("Ficelle timed out a slow upstream and tried the next candidate; reduce this profile's timeout or quarantine repeat offenders.")
    if reason_counts.get("empty_assistant_message") or reason_counts.get("invalid_success_json"):
        actions.append("Inspect route logs and verified capabilities; this upstream returned HTTP 200 without a usable assistant message.")
    if not actions:
        actions.append("Inspect ~/.ficelle/logs/routes.jsonl with the request_id, then clear cooldowns only after the upstream issue is understood.")
    return actions


def build_upstream_failure_error(
    requested_model: str,
    request_id: str,
    candidate_count: int,
    attempts: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    safe_details: list[dict[str, Any]] = []
    for row in errors:
        if row.get("reason"):
            reason = str(row["reason"])
        elif row.get("error"):
            reason = "exception"
        else:
            reason = "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        safe_row = {
            "model": row.get("model"),
            "source": row.get("source"),
            "upstream": row.get("upstream"),
            "status": row.get("status"),
            "reason": reason,
        }
        detail = sanitize_error_detail(row.get("detail") or row.get("error"))
        if detail:
            safe_row["detail"] = detail
        if "stream_started" in row:
            safe_row["stream_started"] = bool(row.get("stream_started"))
        safe_details.append({key: value for key, value in safe_row.items() if value is not None})
    reason_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items())) or "unknown"
    safe_requested_model = sanitize_error_detail(requested_model, 250) or "[redacted]"
    return {
        "error": {
            "message": f"all Ficelle candidates failed for {safe_requested_model} ({len(attempts)}/{candidate_count} attempted; {reason_summary})",
            "type": "upstream_failure",
            "request_id": request_id,
            "requested_model": safe_requested_model,
            "candidate_count": candidate_count,
            "attempt_count": len(attempts),
            "reasons": reason_counts,
            "last_error": safe_details[-1] if safe_details else None,
            "details": safe_details,
            "actions": upstream_failure_actions(reason_counts),
        }
    }
