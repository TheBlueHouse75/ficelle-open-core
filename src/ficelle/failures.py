from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from ficelle.redaction import sanitize_error_detail


# Provider error messages routinely end with a link to an upgrade or settings page, and the
# false-free markers below are bare substrings — so "see https://…/settings/credits" appended to an
# unrelated rejection would read as a payment demand and quarantine a healthy model. URLs are
# stripped before those markers are matched; a real payment demand states it in prose.
#
# The class stops at JSON/markdown delimiters rather than at whitespace: `text` is the raw response
# body, and providers emit compact JSON, so a whitespace-bounded match would run past the closing
# quote and swallow every field after the link — including the very marker we need to see, as in
# `{"message":"see https://p.test/e","type":"billing_error"}`.
_URL_PATTERN = re.compile(r"""https?://[^\s"'<>)\]},;]+""")


FailureReason = Literal[
    "auth_or_credit",
    "bad_upstream_request",
    "billing_or_paid",
    "model_not_found",
    "no_free_quota",
    "quota_exhausted",
    "rate_limited",
    "rate_limited_upstream",
    "request_too_large",
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

# Structural "the pool behind THIS model id is saturated" markers. An aggregator serves one model id
# from a shared upstream pool, so its 429 says nothing about the account: every sibling model of the
# same provider is still answering. Mapping it to `rate_limited` benches the whole provider for one
# saturated model.
UPSTREAM_RATE_LIMIT_TEXT_MARKERS = (
    "rate-limited upstream",
    "rate limited upstream",
)

# Stronger account-scope signals win over the upstream-pool wording. Some first-party providers name
# both the model and the organization/API key whose RPM budget was consumed; the model id alone must
# not weaken that provider-wide protection.
ACCOUNT_RATE_LIMIT_TEXT_MARKERS = (
    "account",
    "organization",
    "api key",
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
    upstream_rate_limit: tuple[str, ...] = UPSTREAM_RATE_LIMIT_TEXT_MARKERS

    def with_extra(
        self,
        *,
        false_free: tuple[str, ...] = (),
        quota_exhausted: tuple[str, ...] = (),
        free_tier_zero_allocation: tuple[str, ...] = (),
        model_not_found: tuple[str, ...] = (),
        upstream_rate_limit: tuple[str, ...] = (),
    ) -> "FailureMarkers":
        return FailureMarkers(
            false_free=self.false_free + false_free,
            quota_exhausted=self.quota_exhausted + quota_exhausted,
            free_tier_zero_allocation=self.free_tier_zero_allocation + free_tier_zero_allocation,
            model_not_found=self.model_not_found + model_not_found,
            upstream_rate_limit=self.upstream_rate_limit + upstream_rate_limit,
        )


DEFAULT_FAILURE_MARKERS = FailureMarkers()

# The provider rejected the REQUEST we sent, not the account: a 400/422 is by definition about this
# body. Shared with `capability_discovery`, which reads the same statuses as a per-profile capability
# verdict — one name, one meaning, and the two paths' differing consequences stay deliberate.
REQUEST_REJECTION_STATUSES = frozenset({400, 422})

# Failures the CALLER caused, not the model. Two consequences, both flowing from that single fact:
# `cooldown_policy_for_reason` withholds the cooldown, and the consecutive-failure streak skips them.
# The streak's penalty is cumulative (12 points each, `model_scoring`), so without the exemption a
# client looping on a too-small max_tokens would progressively demote every candidate it touches,
# reordering a whole profile's pool over a limit that says nothing about the models. They are still
# counted and shown — a model that only ever truncates must stay visible rather than keep a clean
# record.
CALLER_CAUSED_FAILURE_REASONS = frozenset(
    {
        "truncated_before_content",
        # The upstream refused the request body itself. A malformed tool_call or an unsupported
        # field says nothing about the model's health, and every candidate would reject the same
        # payload — so no retry (see NON_RETRYABLE_FAILURE_REASONS), no cooldown, and the caller
        # gets the upstream's own 400.
        "bad_upstream_request",
        # Writing to the caller's socket failed: the client hung up mid-stream. The upstream was
        # answering fine; cooling it would punish a healthy model for a client-side abort.
        "client_disconnected",
    }
)

# Failures no other candidate can do better on, because the candidate was never the problem. Trying
# the next one replays the same rejection and burns a healthy model's turn. Distinct from
# CALLER_CAUSED_FAILURE_REASONS, which is about *state writes*: a truncated response is the caller's
# fault too, yet a model with a larger budget may well answer, so it keeps its failover.
NON_RETRYABLE_FAILURE_REASONS = frozenset({"bad_upstream_request"})

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
    if reason in CALLER_CAUSED_FAILURE_REASONS:
        # The caller, not the model, produced the failure: a token budget that ran out before any
        # content, a request body the upstream refuses, a client that hung up mid-stream. Record it
        # so the model stays scored and visible in the admin, but never cool it — one client sending
        # max_tokens=20 or a malformed tool_call would otherwise empty a whole profile's pool for
        # the cooldown window, which is exactly how a bad client takes the router down.
        return CooldownPolicy(record_provider_error=False, model_cooldown=False)
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
    upstream_model_id: str | None = None,
) -> FailureReason:
    """Classify a provider failure using already-normalized free-access metadata."""
    marker_set = markers or DEFAULT_FAILURE_MARKERS
    lower = text.lower()
    lower_without_urls = _URL_PATTERN.sub(" ", lower)
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
        # Deliberately not `bad_upstream_request`: the message names an exhausted quota, so the
        # request body is not what this 400 is about, and the model does belong on cooldown.
        return "unavailable"
    if status_code == 429:
        # A 429 that names the shared pool behind one model id is MODEL-scoped: the account is fine
        # and every sibling model keeps answering, so it must not cool the provider. Matched on the
        # URL-stripped text like the false-free markers, since these messages link to a settings page.
        # The marker alone is not enough: an account/provider-level message may also mention an
        # upstream limit. Fail closed to the provider-scoped policy unless the body names the exact
        # model Ficelle called.
        normalized_model_id = str(upstream_model_id or "").strip().lower()
        names_model = len(normalized_model_id) >= 4 and normalized_model_id in lower_without_urls
        names_account_scope = any(marker in lower_without_urls for marker in ACCOUNT_RATE_LIMIT_TEXT_MARKERS)
        if (
            names_model
            and not names_account_scope
            and any(marker in lower_without_urls for marker in marker_set.upstream_rate_limit)
        ):
            return "rate_limited_upstream"
        return "rate_limited"
    # A tokens-per-minute rejection is a transient, MODEL-scoped throughput limit that recharges on
    # its own — deliberately neither `rate_limited` (provider-scoped, would cool every model of the
    # provider) nor `billing_or_paid` (a 24h quarantine). Evaluated before the false-free markers
    # because these messages often point at an upgrade page; URL stripping already covers the link
    # itself, but the surrounding prose can name the plan too.
    if status_code == 413:
        return "request_too_large"
    if status_code == 402 or any(marker in lower_without_urls for marker in marker_set.false_free):
        return "billing_or_paid"
    if status_code in {401, 403}:
        return "auth_or_credit"
    if status_code in {404, 410} and any(marker in lower for marker in marker_set.model_not_found):
        return "model_not_found"
    if status_code >= 500:
        return "server_error"
    # Last, so a 400 that actually states a billing/quota/auth problem keeps its stronger reading
    # above: what is left is the upstream rejecting the request body. Retrying it on another
    # candidate replays the same rejection, and cooling the model blames it for the caller's payload.
    if status_code in REQUEST_REJECTION_STATUSES:
        return "bad_upstream_request"
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
        actions.append("Wait for the provider cooldown or switch this virtual model to another healthy provider.")
    if reason_counts.get("rate_limited_upstream"):
        actions.append("The shared upstream pool behind this model id is saturated, not your account: the model is cooled briefly and its provider keeps serving. Nothing to fix; add your own upstream key if you need dedicated limits.")
    if reason_counts.get("billing_or_paid"):
        actions.append("Inspect the anti false-free guard result; refresh the catalog before re-enabling the model.")
    if reason_counts.get("no_free_quota"):
        actions.append("Provider reported a zero free-tier allocation (limit: 0) for this model; it is quarantined and will stop being proposed. Use a free-tier model or upgrade the account.")
    if reason_counts.get("model_not_found"):
        actions.append("Provider returned 404/410 for this model id (listed in the catalog but not deployed for this account); it is quarantined and will stop being benchmarked. Clear the quarantine to retry if the provider deploys it later.")
    if reason_counts.get("server_error"):
        actions.append("Retry after the short model cooldown or quarantine the unstable upstream.")
    if reason_counts.get("timeout"):
        actions.append("Ficelle timed out a slow upstream and tried the next candidate; reduce this virtual model's timeout or quarantine repeat offenders.")
    if reason_counts.get("bad_upstream_request"):
        actions.append("The upstream rejected the request body itself (HTTP 400/422) — inspect the payload the client sent, typically a malformed tool_call or an unsupported field. No model was cooled and no other candidate was tried: every one of them would reject the same body.")
    if reason_counts.get("client_disconnected"):
        actions.append("The client closed the connection while the answer was streaming; the upstream was healthy and was not cooled. Look at the client's timeout or cancel behaviour, not at the model.")
    if reason_counts.get("truncated_before_content"):
        actions.append("The completion token budget ran out before the model emitted any content; reasoning models spend it on reasoning tokens first. Raise max_tokens on the request. No model was cooled: this is a request-side limit, not an upstream failure.")
    if reason_counts.get("empty_assistant_message") or reason_counts.get("invalid_success_json"):
        actions.append("Inspect route logs and verified capabilities; this upstream returned HTTP 200 without a usable assistant message.")
    if not actions:
        actions.append("Inspect ~/.ficelle/logs/routes.jsonl with the request_id, then clear cooldowns only after the upstream issue is understood.")
    return actions


def caller_rejected_request(errors: list[dict[str, Any]]) -> bool:
    """True when every attempt failed because the upstream refused the request body itself.

    The payload is the problem, so no candidate could have done better: the caller gets the
    upstream's own 400 instead of a 502 that reads as "Ficelle is broken".
    """
    return bool(errors) and all(row.get("reason") == "bad_upstream_request" for row in errors)


def upstream_failure_status(errors: list[dict[str, Any]]) -> int:
    """HTTP status for a run where no candidate delivered.

    Paired with `build_upstream_failure_error`, which reads the same verdict off the same list:
    every caller of one must use the other, or the status and the payload's `type` disagree.
    """
    return 400 if caller_rejected_request(errors) else 502


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
    if list(reason_counts) == ["bad_upstream_request"]:  # `caller_rejected_request`, already counted
        upstream_detail = safe_details[-1].get("detail") if safe_details else ""
        message = "upstream rejected this request as invalid" + (f": {upstream_detail}" if upstream_detail else "")
        error_type = "invalid_request_error"
    else:
        message = f"all Ficelle candidates failed for {safe_requested_model} ({len(attempts)}/{candidate_count} attempted; {reason_summary})"
        error_type = "upstream_failure"
    return {
        "error": {
            "message": message,
            "type": error_type,
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
