from __future__ import annotations

import json

from ficelle.failures import (
    BENCHMARK_ROUTE_BLOCKING_REASONS,
    CALLER_CAUSED_FAILURE_REASONS,
    FailureMarkers,
    PROVIDER_ERROR_REASONS,
    PROVIDER_SCOPED_COOLDOWN_REASONS,
    bad_request_body,
    build_upstream_failure_error,
    caller_rejected_request,
    classify_failure,
    cooldown_policy_for_reason,
    safe_error_body,
    upstream_failure_actions,
)


def quota_free_access(scope: str = "provider"):
    return {"eligible": True, "mode": "quota_free", "scope": scope}


def catalog_free_access():
    return {"eligible": True, "mode": "catalog_free", "scope": "model"}


def test_quota_free_failures_classify_quota_reasons():
    exhausted = classify_failure(429, "quota exceeded: credits exhausted", normalized_free_access=quota_free_access())
    zero_allocation = classify_failure(402, "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0", normalized_free_access=quota_free_access("model"))
    upstream_error = classify_failure(500, "upstream error while quota exceeded", normalized_free_access=quota_free_access())

    assert exhausted == "quota_exhausted"
    assert zero_allocation == "no_free_quota"
    assert upstream_error == "server_error"


def test_false_free_and_not_found_failures_return_blocking_reasons():
    ended_free = classify_failure(401, "Free promotion has ended for Qwen3.6 Plus Free.", normalized_free_access=catalog_free_access())
    not_found = classify_failure(404, "Function 'abc': Not found for account 'xyz'", normalized_free_access=quota_free_access("model"))

    assert ended_free == "billing_or_paid"
    assert not_found == "model_not_found"


def test_provider_markers_can_extend_failure_classification():
    markers = FailureMarkers().with_extra(false_free=("paid-plan-only",))

    assert classify_failure(400, "paid-plan-only upstream refusal", markers=markers) == "billing_or_paid"
    assert classify_failure(400, "insufficient balance", markers=markers) == "billing_or_paid"


def test_transient_failures_return_transient_reasons():
    # Says nothing about WHICH resource is limited, so it keeps the provider-wide meaning. A body
    # that names the model's shared pool instead is `rate_limited_upstream` — see
    # `test_saturated_model_pool_is_model_scoped_not_a_provider_outage`.
    rate_limit = classify_failure(429, "too many requests, slow down", normalized_free_access=quota_free_access())
    auth = classify_failure(403, "invalid api key", normalized_free_access=catalog_free_access())
    unavailable = classify_failure(404, "upstream temporarily overloaded", normalized_free_access=quota_free_access())

    assert rate_limit == "rate_limited"
    assert auth == "auth_or_credit"
    assert unavailable == "unavailable"


def test_failure_reason_sets_match_routing_contracts():
    assert PROVIDER_SCOPED_COOLDOWN_REASONS == {"rate_limited", "auth_or_credit"}
    assert PROVIDER_ERROR_REASONS == {"rate_limited", "auth_or_credit", "quota_exhausted", "no_free_quota"}
    assert BENCHMARK_ROUTE_BLOCKING_REASONS == {
        "billing_or_paid",
        "no_free_quota",
        "quota_exhausted",
        "auth_or_credit",
        "model_not_found",
    }


def test_saturated_model_pool_is_model_scoped_not_a_provider_outage():
    """An aggregator's 429 about ONE model id must not bench the provider's other models.

    Observed live: OpenRouter answered `google/gemma-4-31b-it:free is temporarily rate-limited
    upstream` — Google's shared free pool was full, the account was fine. Classified as
    `rate_limited`, that single model cooled all 14 usable OpenRouter models for 15 minutes, and the
    discovery cycle re-probed it every pass, so 54 of the provider's 55 rate-limit failures came from
    that one id. Same reasoning as `request_too_large`: a limit scoped to one model gets a model
    cooldown, and the branch is deliberately evaluated on the URL-stripped text.
    """
    saturated = (
        '{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":'
        '"google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly, '
        'or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations"}}}'
    )
    model_id = "google/gemma-4-31b-it:free"
    assert classify_failure(429, saturated, upstream_model_id=model_id) == "rate_limited_upstream"
    # Safety default: the phrase alone cannot remove the provider-wide guard. The caller must prove
    # that the body names the exact model it invoked.
    assert classify_failure(429, saturated) == "rate_limited"
    assert (
        classify_failure(
            429,
            f"{model_id} is temporarily rate-limited upstream for your account",
            upstream_model_id=model_id,
        )
        == "rate_limited"
    )
    # The account-level 429 keeps its provider-wide meaning.
    assert classify_failure(429, "Rate limit exceeded for your account") == "rate_limited"

    # Quota-free exhaustion remains the higher-priority structural signal even if the provider also
    # names the model and calls its serving pool upstream.
    quota_text = f"{model_id} is rate-limited upstream: free tier quota exhausted"
    assert (
        classify_failure(
            429,
            quota_text,
            normalized_free_access=quota_free_access(),
            upstream_model_id=model_id,
        )
        == "quota_exhausted"
    )

    assert "rate_limited_upstream" not in PROVIDER_SCOPED_COOLDOWN_REASONS
    assert "rate_limited_upstream" not in PROVIDER_ERROR_REASONS
    policy = cooldown_policy_for_reason("rate_limited_upstream", source="openrouter")
    assert policy.model_cooldown is True
    assert policy.provider_cooldown is False and policy.quarantine is None
    # No provider error either: the provider card must not read "last error: rate limited" for an
    # account that is answering fine on every other model.
    assert policy.record_provider_error is False


def test_quota_exhausted_policy_sets_recoverable_quota_cooldown_only():
    policy = cooldown_policy_for_reason("quota_exhausted", source="nvidia")

    assert policy.record_provider_error is True
    assert policy.quota_cooldown is True
    assert policy.provider_cooldown is False
    assert policy.quarantine is None
    assert policy.model_cooldown is False


def test_tokens_per_minute_rejection_is_model_scoped_not_a_paid_signal():
    """HTTP 413 is a transient per-model throughput limit, not a payment demand or a provider fault.

    `docs/components/router.md` described this classification long before any code implemented it,
    and the admin Settings page shipped a cooldown field bound to a key that existed nowhere in
    Python — so the field silently discarded whatever an operator typed, and a real TPM rejection
    fell through to `unavailable` and its 600s bench instead of the documented 120s.
    """
    tpm = "Request too large for model llama-3.3-70b on tokens per minute (TPM): Limit 8000, Requested 95722"
    assert classify_failure(413, tpm) == "request_too_large"
    # Mapping it to rate_limited would cool every model of the provider; billing_or_paid would
    # quarantine it for 24h. Neither is right for a limit that recharges on its own.
    assert "request_too_large" not in PROVIDER_SCOPED_COOLDOWN_REASONS
    policy = cooldown_policy_for_reason("request_too_large", source="groq")
    assert policy.model_cooldown is True
    assert policy.provider_cooldown is False and policy.quarantine is None

    # The upgrade link these messages carry must not turn the rejection into a paid signal.
    assert classify_failure(413, f"{tpm}, see https://console.groq.com/settings/billing") == "request_too_large"


def test_false_free_markers_ignore_urls():
    """A link to a settings page is not a payment demand — but prose asking for credits still is.

    Provider errors routinely append "see https://…/settings/credits", and the markers are bare
    substrings, so without stripping URLs an unrelated 404 would classify as `billing_or_paid` and
    quarantine a healthy model with no TTL. Strict-zero still has to hold: a real payment demand
    states it outside a URL, and those must keep quarantining.
    """
    assert classify_failure(404, "No audio endpoint. See https://openrouter.ai/settings/credits") == "unavailable"
    assert classify_failure(400, "invalid image credit line format") == "billing_or_paid"  # prose wins
    assert classify_failure(402, "see https://example.test/help") == "billing_or_paid"  # status wins
    for demand in ("add credits to continue", "payment required", "insufficient balance", "billing account needed"):
        assert classify_failure(404, f"{demand} — https://example.test/settings") == "billing_or_paid", demand

    # `text` is the raw response body and providers emit compact JSON, so the match must stop at the
    # closing quote. A whitespace-bounded one would eat every field after the link — including the
    # marker — and silently drop the payment signal.
    compact = '{"error":{"message":"see https://p.test/e","type":"billing_error"}}'
    assert classify_failure(404, compact) == "billing_or_paid"
    nested = '{"error":{"message":"No route. See https://k.test/d","metadata":{"buyCreditsUrl":"https://k.test/credits"}}}'
    assert classify_failure(404, nested) == "billing_or_paid"
    # …while a link inside markdown or angle brackets is still fully stripped.
    assert classify_failure(404, "no endpoint ([docs](https://x.test/credits))") == "unavailable"
    assert classify_failure(404, "no endpoint <https://x.test/billing>") == "unavailable"


def test_a_rejected_request_body_is_not_an_unavailable_model():
    """HTTP 400/422 means "your request is wrong", not "this model is down".

    Reading it as `unavailable` cooled a healthy model for 600s and sent the same doomed body to
    the next candidate — one client with a malformed tool_call could empty a whole profile. It is
    the last branch, so a 400 whose prose names a stronger reading still gets it.
    """
    assert classify_failure(400, "This request is not valid. Additional info: Provider returned error") == "bad_upstream_request"
    assert classify_failure(422, "messages.1: tool_call_id not found") == "bad_upstream_request"

    assert classify_failure(400, "add credits to continue") == "billing_or_paid"  # prose wins
    # …and a body Ficelle could not even read stays the generic unavailable.
    assert classify_failure(418, "<unreadable response body: RuntimeError>") == "unavailable"


def test_caller_caused_policies_record_the_failure_without_cooling():
    """No caller-caused failure may take a healthy model out of the pool.

    Recording still happens — `set_cooldown` runs `update_failure_stats` and
    `record_model_error_in_state` before consulting the policy — so a model that truncates on every
    request stays scored and visible instead of silently keeping a perfect record. Driven off the
    set itself, so a fourth member added later cannot quietly skip the contract.
    """
    for reason in sorted(CALLER_CAUSED_FAILURE_REASONS):
        policy = cooldown_policy_for_reason(reason, source="nous")

        assert policy.model_cooldown is False, reason
        assert policy.provider_cooldown is False, reason
        assert policy.quota_cooldown is False, reason
        assert policy.quarantine is None, reason
        assert policy.record_provider_error is False, reason  # not the provider's fault either


def test_a_wholly_rejected_request_answers_with_the_upstream_message():
    """When every attempt died on the body, the caller needs the upstream's own words.

    "all Ficelle candidates failed" reads as a router outage and invites a retry; the actual
    provider message is what points at the malformed field.
    """
    errors = [
        {
            "model": "ficelle/nous/stepfun/step-3.7-flash:free",
            "reason": "bad_upstream_request",
            "status": 400,
            "detail": '{"status":400,"message":"This request is not valid."}',
        }
    ]

    error = build_upstream_failure_error("ficelle/auto-fast", "req-1", 2, [{"model": "x"}], errors)["error"]

    assert error["type"] == "invalid_request_error"
    assert error["message"].startswith("upstream rejected this request as invalid: ")
    assert "This request is not valid." in error["message"]
    assert any("no other candidate was tried" in action for action in error["actions"])

    # One genuine upstream failure in the mix and it is a gateway problem again.
    mixed = [*errors, {"model": "ficelle/nous/tencent/hy3:free", "reason": "server_error", "status": 503}]
    mixed_error = build_upstream_failure_error("ficelle/auto-fast", "req-2", 2, [{"model": "x"}], mixed)["error"]
    assert mixed_error["type"] == "upstream_failure"
    assert mixed_error["message"].startswith("all Ficelle candidates failed")


def test_caller_rejected_request_needs_every_attempt_to_agree():
    assert caller_rejected_request([{"reason": "bad_upstream_request"}]) is True
    assert caller_rejected_request([{"reason": "bad_upstream_request"}, {"reason": "rate_limited"}]) is False
    assert caller_rejected_request([{"reason": "server_error"}]) is False
    # No attempt reached an upstream at all: nothing says the body is what failed.
    assert caller_rejected_request([]) is False


def test_caller_caused_failures_are_the_only_ones_exempt_from_the_streak():
    """The consecutive-failure streak is reserved for what the model is answerable for.

    Its penalty is cumulative (12 points each), so counting a caller's too-small max_tokens, its
    malformed request body, or its own mid-stream hangup would let one misbehaving client
    progressively demote every candidate it touches. The failure is still counted in
    `requests`/`failures` and still shown — only the streak is left alone.
    """
    assert CALLER_CAUSED_FAILURE_REASONS == {
        "truncated_before_content",
        "bad_upstream_request",
        "client_disconnected",
    }
    for upstream_fault in ("unavailable", "server_error", "timeout", "empty_assistant_message"):
        assert upstream_fault not in CALLER_CAUSED_FAILURE_REASONS


def test_provider_scoped_policy_keeps_model_and_provider_cooldowns():
    policy = cooldown_policy_for_reason("rate_limited", source="openrouter")

    assert policy.record_provider_error is True
    assert policy.provider_cooldown is True
    assert policy.provider_cooldown_source == "openrouter"
    assert policy.quota_cooldown is False
    assert policy.quarantine is None
    assert policy.model_cooldown is True


def test_hard_quarantine_policies_skip_recoverable_cooldowns():
    no_quota = cooldown_policy_for_reason("no_free_quota", source="nvidia")
    not_found = cooldown_policy_for_reason("model_not_found", source="nvidia")

    assert no_quota.record_provider_error is True
    assert no_quota.quarantine is not None
    assert no_quota.quarantine.reason == "no_free_quota"
    assert no_quota.model_cooldown is False
    assert not_found.record_provider_error is False
    assert not_found.quarantine is not None
    assert not_found.quarantine.reason == "model_not_found"
    assert not_found.model_cooldown is False


def test_false_free_policy_quarantines_and_keeps_model_cooldown():
    policy = cooldown_policy_for_reason("billing_or_paid", source="openrouter")

    assert policy.record_provider_error is False
    assert policy.provider_cooldown is False
    assert policy.quarantine is not None
    assert policy.quarantine.reason == "billing_or_paid"
    assert policy.model_cooldown is True


def test_transient_policy_is_model_scoped_only():
    policy = cooldown_policy_for_reason("server_error", source="openrouter")

    assert policy.record_provider_error is False
    assert policy.provider_cooldown is False
    assert policy.quota_cooldown is False
    assert policy.quarantine is None
    assert policy.model_cooldown is True


def test_error_payload_helpers_redact_and_omit_trace():
    payload = safe_error_body(RuntimeError("Authorization: Bearer sk-testsecret123456789"), request_id="req-1")
    bad_request = bad_request_body(ValueError("api_key=plainsecret12345"), request_id="req-2")
    serialized = json.dumps({"payload": payload, "bad_request": bad_request}, sort_keys=True)

    assert "sk-testsecret" not in serialized
    assert "plainsecret" not in serialized
    assert "trace" not in serialized
    assert payload["error"]["request_id"] == "req-1"
    assert payload["error"]["type"] == "RuntimeError"
    assert bad_request["error"]["request_id"] == "req-2"
    assert bad_request["error"]["type"] == "bad_request"


def test_upstream_failure_error_is_actionable_and_safe():
    attempts = [
        {"model": "ficelle/openrouter/a", "status": 500, "reason": "server_error"},
        {"model": "ficelle/openrouter/b", "status": 200, "reason": "empty_assistant_message"},
    ]
    errors = [
        {
            "model": "ficelle/openrouter/a",
            "upstream": "a",
            "source": "openrouter",
            "status": 500,
            "reason": "server_error",
            "detail": "HTTP 500: Bearer sk-testsecret123456789 exploded",
        },
        {
            "model": "ficelle/openrouter/b",
            "upstream": "b",
            "source": "openrouter",
            "status": 200,
            "reason": "empty_assistant_message",
            "stream_started": True,
        },
    ]

    payload = build_upstream_failure_error("api_key=plainsecret12345", "req-1", 3, attempts, errors)
    error = payload["error"]
    serialized = json.dumps(payload)

    assert error["type"] == "upstream_failure"
    assert error["request_id"] == "req-1"
    assert error["candidate_count"] == 3
    assert error["attempt_count"] == 2
    assert error["reasons"] == {"server_error": 1, "empty_assistant_message": 1}
    assert "empty_assistant_message=1" in error["message"]
    assert error["requested_model"] == "[redacted]"
    assert error["last_error"]["stream_started"] is True
    assert any("verified capabilities" in action for action in error["actions"])
    assert "sk-testsecret" not in serialized
    assert "plainsecret" not in serialized
    assert "[redacted]" in serialized


def test_upstream_failure_actions_include_default_operator_guidance():
    assert upstream_failure_actions({}) == [
        "Inspect ~/.ficelle/logs/routes.jsonl with the request_id, then clear cooldowns only after the upstream issue is understood."
    ]
    assert any("provider cooldown" in action for action in upstream_failure_actions({"rate_limited": 1}))


def test_upstream_failure_error_preserves_requested_model_detail_limit():
    requested_model = "ficelle/" + ("model-" * 40)

    payload = build_upstream_failure_error(requested_model, "req-1", 0, [], [])

    assert payload["error"]["requested_model"] == requested_model
