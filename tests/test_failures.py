from __future__ import annotations

import json

from ficelle.failures import (
    BENCHMARK_ROUTE_BLOCKING_REASONS,
    FailureMarkers,
    PROVIDER_ERROR_REASONS,
    PROVIDER_SCOPED_COOLDOWN_REASONS,
    bad_request_body,
    build_upstream_failure_error,
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
    rate_limit = classify_failure(429, "model is temporarily rate-limited upstream", normalized_free_access=quota_free_access())
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


def test_quota_exhausted_policy_sets_recoverable_quota_cooldown_only():
    policy = cooldown_policy_for_reason("quota_exhausted", source="nvidia")

    assert policy.record_provider_error is True
    assert policy.quota_cooldown is True
    assert policy.provider_cooldown is False
    assert policy.quarantine is None
    assert policy.model_cooldown is False


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
