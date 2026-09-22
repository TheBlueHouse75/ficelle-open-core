from __future__ import annotations

import json

import pytest

from ficelle.failures import (
    CONTEXT_LENGTH_EXCEEDED_MARKERS,
    ACCOUNT_RATE_LIMIT_TEXT_MARKERS,
    BENCHMARK_ROUTE_BLOCKING_REASONS,
    NO_MODEL_FAULT_FAILURE_REASONS,
    PRODUCTION_PROFILE_FAILURE_REASONS,
    CONTEXT_DIVERTING_FAILURE_REASONS,
    FALSE_FREE_PAYMENT_DEMAND_MARKERS,
    ERROR_CODE_STATUSES,
    FALSE_FREE_TEXT_MARKERS,
    FailureMarkers,
    false_free_pattern,
    provider_error_codes,
    PROVIDER_ERROR_REASONS,
    PROVIDER_SCOPED_COOLDOWN_REASONS,
    SOURCE_DIVERTING_FAILURE_REASONS,
    TERMINAL_ENDINGS,
    bad_request_body,
    build_upstream_failure_error,
    caller_rejected_request,
    classify_failure,
    cooldown_policy_for_reason,
    exception_is_tls_failure,
    model_not_found_body,
    rejected_request_tokens,
    request_exceeds_context,
    route_log_failure_status,
    safe_error_body,
    status_for_error_codes,
    upstream_failure_actions,
    upstream_failure_status,
)


def quota_free_access(scope: str = "provider"):
    return {"eligible": True, "mode": "quota_free", "scope": scope}


def catalog_free_access():
    return {"eligible": True, "mode": "catalog_free", "scope": "model"}


def test_quota_free_failures_classify_quota_reasons():
    exhausted = classify_failure(429, "quota exceeded: credits exhausted", normalized_free_access=quota_free_access())
    structurally_exhausted = classify_failure(
        429,
        '{"error":{"code":"insufficient_quota","message":"Rate limit exceeded."}}',
        normalized_free_access=quota_free_access("model"),
    )
    structurally_unallocated = classify_failure(
        429,
        '{"error":{"code":"insufficient_quota","message":"Quota exceeded, limit: 0"}}',
        normalized_free_access=quota_free_access("model"),
    )
    zero_allocation = classify_failure(402, "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0", normalized_free_access=quota_free_access("model"))
    upstream_error = classify_failure(500, "upstream error while quota exceeded", normalized_free_access=quota_free_access())

    assert exhausted == "quota_exhausted"
    assert structurally_exhausted == "quota_exhausted"
    assert structurally_unallocated == "no_free_quota"
    assert zero_allocation == "no_free_quota"
    assert upstream_error == "server_error"


def test_insufficient_quota_in_caller_prose_is_not_a_quota_verdict():
    echoed = json.dumps(
        {
            "error": {
                "code": "invalid_request_error",
                "message": "Invalid schema for function 'insufficient_quota'",
            }
        }
    )

    assert classify_failure(400, echoed, normalized_free_access=quota_free_access("model")) == "bad_upstream_request"


def test_false_free_and_not_found_failures_return_blocking_reasons():
    ended_free = classify_failure(401, "Free promotion has ended for Qwen3.6 Plus Free.", normalized_free_access=catalog_free_access())
    not_found = classify_failure(404, "Function 'abc': Not found for account 'xyz'", normalized_free_access=quota_free_access("model"))

    assert ended_free == "billing_or_paid"
    assert not_found == "model_not_found"


def test_provider_markers_can_extend_failure_classification():
    markers = FailureMarkers().with_extra(false_free=("paid-plan-only",))

    # Both on a 400, where the bare markers no longer apply: an adapter's extra is a deliberate,
    # provider-specific phrasing rather than a generic substring, so it stays trusted on every status.
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
    assert PROVIDER_SCOPED_COOLDOWN_REASONS == {"rate_limited", "auth_or_credit", "tls_error"}
    assert PROVIDER_ERROR_REASONS == {
        "rate_limited",
        "auth_or_credit",
        "tls_error",
        "quota_exhausted",
        "no_free_quota",
    }
    assert BENCHMARK_ROUTE_BLOCKING_REASONS == {
        "billing_or_paid",
        "no_free_quota",
        "quota_exhausted",
        "auth_or_credit",
        "model_not_found",
    }


def test_what_the_attempt_window_diverts_on_is_what_the_cooldown_policy_writes():
    """The attempt loop drops part of the pool on two unrelated facts, and only one is a list.

    Everything that blocks *other* candidates says so in the state it writes, so the loop reads the
    write (`AppliedCooldown`) and no set has to be kept in step with `cooldown_policy_for_reason`.
    This test pins that mapping from the policy's side: which reasons block beyond the model that
    earned them, and which deliberately do not.
    """
    for reason in PROVIDER_SCOPED_COOLDOWN_REASONS:
        assert cooldown_policy_for_reason(reason, source="groq").provider_cooldown
    # A quota pool is blocked at whichever scope its provider declares, which is why the loop keys
    # this one and never guesses a source from it.
    assert cooldown_policy_for_reason("quota_exhausted", source="groq").quota_cooldown

    # Blocks nothing wider than the model that earned it, so the planned order is left alone:
    # `rate_limited_upstream` is a saturated pool behind one model id, `no_free_quota` quarantines
    # that model, and `request_too_large` is the caller's own body.
    for reason in ("rate_limited_upstream", "no_free_quota", "request_too_large"):
        policy = cooldown_policy_for_reason(reason, source="groq")
        assert not policy.provider_cooldown and not policy.quota_cooldown

    # `bad_upstream_contract` is the exception that has to be listed: it writes nothing at all — no
    # cooldown, no quarantine — and is a verdict on the source held for one request only.
    for reason in SOURCE_DIVERTING_FAILURE_REASONS:
        policy = cooldown_policy_for_reason(reason, source="groq")
        assert not policy.provider_cooldown and not policy.quota_cooldown and policy.quarantine is None


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


def test_structured_upstream_rate_limit_marker_requires_exact_code_and_type():
    markers = FailureMarkers().with_extra(
        upstream_rate_limit_error_codes=(("3505", "backend_out_of_capacity"),)
    )
    exact = (
        '{"object":"error","message":"Not enough capacity available for this request, '
        'please retry later.","type":"backend_out_of_capacity","param":null,'
        '"code":"3505","raw_status_code":429}'
    )

    assert classify_failure(429, exact, markers=markers) == "rate_limited_upstream"
    assert (
        classify_failure(
            429,
            '{"error":{"type":"backend_out_of_capacity","code":"3505"}}',
            markers=markers,
        )
        == "rate_limited_upstream"
    )
    policy = cooldown_policy_for_reason("rate_limited_upstream", source="mistral")
    assert policy.model_cooldown is True
    assert policy.provider_cooldown is False
    assert policy.record_provider_error is False

    # The structural pair is deliberately exact: an account-level 429 and a quota verdict retain
    # their existing provider/quota scope instead of inheriting this adapter-specific rule.
    assert classify_failure(429, "rate limit exceeded for your account", markers=markers) == "rate_limited"
    assert (
        classify_failure(
            429,
            '{"error":{"type":"backend_out_of_capacity","code":"3505",'
            '"message":"capacity unavailable for your account"}}',
            markers=markers,
        )
        == "rate_limited"
    )
    assert (
        classify_failure(
            429,
            "quota exceeded: credits exhausted",
            normalized_free_access=quota_free_access(),
            markers=markers,
        )
        == "quota_exhausted"
    )
    assert (
        classify_failure(
            429,
            '{"error":{"type":"backend_out_of_capacity","code":"3506"}}',
            markers=markers,
        )
        == "rate_limited"
    )


def test_a_403_naming_a_model_that_must_be_switched_on_does_not_cool_the_provider():
    """A per-model entitlement is not a rejected key, and must not bench a working account.

    Observed live on Mistral: discovery probed `labs-leanstral-1-5`, got 403, and `401/403 ->
    auth_or_credit` cooled ALL of Mistral for an hour — while `/v1/models` answered 200 on the same
    key and `mistral-large-latest` was serving. The probe returned when the hour expired and did it
    again; the state file carries the same failure on three Labs ids, dated 04/08 and 07/08.

    Same fail-closed shape as the 429 branch above: the marker alone must not disarm the
    provider-wide guard, because a dead key also returns 403. The discriminator is that a dead key
    never names a model.
    """
    labs = (
        '{"object":"error","message":"Model labs-leanstral-2603 is a Labs model. To use Labs models, '
        'an admin must enable them in your organization settings at https://admin.mistral.ai/"}'
    )
    model_id = "labs-leanstral-2603"
    assert classify_failure(403, labs, upstream_model_id=model_id) == "model_not_found"

    # The three ways it must stay provider-scoped.
    assert classify_failure(403, labs) == "auth_or_credit"                              # no id proven
    assert classify_failure(403, labs, upstream_model_id="other-model") == "auth_or_credit"
    assert classify_failure(403, '{"message":"Unauthorized"}', upstream_model_id=model_id) == "auth_or_credit"

    # Real-world 403s that name a model AND read like an entitlement, but are account-wide. Every
    # one of these came from attacking the branch: the region case actually got through an earlier
    # version, which would have quarantined one model while leaving a provider that fails every
    # request uncooled — the expensive direction to be wrong in.
    for body in (
        "Access denied for gpt-4o-mini: your API key has been revoked",
        "You do not have access to gpt-4o-mini from your region",
        "gpt-4o-mini: your account is suspended and not enabled",
        "Your organization must be enabled to use gpt-4o-mini",
        "Your account is not enabled for gpt-4o-mini",
        "You are not authorized to use gpt-4o-mini",
        # Account-wide gates phrased per-model. These carry the entitlement marker in the ACTIVE
        # voice — "must enable" — exactly as Mistral's real message does, so only naming the gate
        # itself tells them apart. Worth the extra markers because this branch quarantines, and a
        # quarantine waits for a human where the old provider cooldown expired by itself.
        "Your organization must enable SSO to use gpt-4o-mini",
        "An admin must enable two-factor auth before you can use gpt-4o-mini",
        "Your workspace must enable this integration to use gpt-4o-mini",
    ):
        assert classify_failure(403, body, upstream_model_id="gpt-4o-mini") == "auth_or_credit", body
    # 401 is never a per-model verdict: nothing about a key is model-scoped.
    assert classify_failure(401, labs, upstream_model_id=model_id) == "auth_or_credit"

    # The account-scope guard of the 429 branch is deliberately NOT reused here: this very message
    # trips it while meaning the opposite — "organization" locates the switch, not the fault. Read
    # off the real constant so a later tidy-up that "unifies" the two branches turns this red.
    assert any(marker in labs.lower() for marker in ACCOUNT_RATE_LIMIT_TEXT_MARKERS)

    # The remedy — quarantine, no provider error, no model cooldown, so the hourly re-probe stops —
    # is owned by test_hard_quarantine_policies_skip_recoverable_cooldowns, which pins all of it
    # for this reason. Not restated here.


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


# --- L4-R? nex-agi 266,645-token incident: the request itself was too big for the model ---------


def test_a_context_length_rejection_is_retryable_on_a_larger_candidate_without_cooling():
    """The exact incident: a 266,645-token request sent to a 262,144-context model. The upstream's
    own words are prose, not a structured code, so this has to be a marker match — narrow enough
    that it never fires on an unrelated 400 (see the two tests below)."""
    incident = (
        "The request is 266645 tokens long and exceeds this model's context length of 262144 tokens."
    )

    assert classify_failure(400, incident) == "context_length_exceeded"
    assert rejected_request_tokens(incident) == 266645

    policy = cooldown_policy_for_reason("context_length_exceeded")
    assert policy.model_cooldown is False
    assert policy.record_provider_error is False
    assert "context_length_exceeded" in NO_MODEL_FAULT_FAILURE_REASONS
    # Retryable, unlike `bad_upstream_request`: a candidate with a bigger context window can
    # still answer this exact body.
    from ficelle.failures import NON_RETRYABLE_FAILURE_REASONS

    assert "context_length_exceeded" not in NON_RETRYABLE_FAILURE_REASONS
    assert CONTEXT_DIVERTING_FAILURE_REASONS == frozenset({"context_length_exceeded"})


def test_a_context_length_rejection_also_classifies_on_413_and_on_422():
    """Some gateways answer a context-length rejection with 413 or 422 rather than 400; the marker
    match has to apply on every request-rejection status a provider might choose, not just 400."""
    assert classify_failure(413, "Your prompt is too long for this model's context window.") == (
        "context_length_exceeded"
    )
    assert classify_failure(422, "maximum context length exceeded for this model") == "context_length_exceeded"


def test_context_length_markers_never_widen_an_unrelated_request_rejection():
    """An ordinary malformed-body 400 must keep classifying as `bad_upstream_request`: the marker
    set is whole phrases ("context length", "too many tokens", …), not bare words like "token" or
    "large", so a ordinary validation error never gets misread as a context-size problem."""
    assert classify_failure(400, "This request is not valid. Additional info: Provider returned error") == (
        "bad_upstream_request"
    )
    assert classify_failure(422, "messages.1: tool_call_id not found") == "bad_upstream_request"
    # A TPM rejection ("too large", not "too many tokens") keeps its own, differently-scoped reason.
    tpm = "Request too large for model llama-3.3-70b on tokens per minute (TPM): Limit 8000, Requested 95722"
    assert classify_failure(413, tpm) == "request_too_large"


@pytest.mark.parametrize("marker", CONTEXT_LENGTH_EXCEEDED_MARKERS)
def test_every_context_length_marker_classifies_on_its_own(marker):
    # Each phrase must earn its place: a 400 carrying only that phrase is a context rejection.
    assert classify_failure(400, f"rejected: {marker}.") == "context_length_exceeded"


def test_rejected_request_tokens_returns_none_without_a_stated_size():
    assert rejected_request_tokens("upstream rejected the request") is None
    assert rejected_request_tokens("") is None


def test_rejected_request_tokens_is_the_larger_figure_whatever_the_phrasing():
    # OpenAI names the limit first and the request second; the request is always the larger.
    openai_style = (
        "This model's maximum context length is 8192 tokens. "
        "However, your messages resulted in 10,000 tokens."
    )
    assert rejected_request_tokens(openai_style) == 10000


def test_request_exceeds_context_names_a_run_every_attempt_of_which_was_too_big():
    """The pool-narrowing counterpart to `caller_rejected_request`: a run where every candidate was
    excluded or rejected for context length gets a 400 naming the request's size, not a 502 that
    reads as an outage."""
    errors = [{"reason": "context_length_exceeded", "detail": "request needs ~266645 tokens, model context is 262144"}]

    assert request_exceeds_context(errors) is True
    assert upstream_failure_status(errors) == 400
    payload = build_upstream_failure_error("ficelle/auto-long", "req-ctx", 1, [], errors)
    assert payload["error"]["type"] == "invalid_request_error"
    assert "266645" in payload["error"]["message"]

    # A mixed run (one attempt failed for another reason) is not this verdict.
    mixed = errors + [{"reason": "server_error"}]
    assert request_exceeds_context(mixed) is False


def test_false_free_markers_ignore_urls():
    """A link to a settings page is not a payment demand — but prose asking for credits still is.

    Provider errors routinely append "see https://…/settings/credits", where the marker sits between
    two `/` — so without stripping URLs an unrelated 404 would classify as `billing_or_paid` and
    quarantine a healthy model with no TTL. Strict-zero still has to hold: a real payment demand
    states it outside a URL, and those must keep quarantining.

    The compact/nested JSON cases below now pass through the *fields*, not the prose: since the
    identifier rule landed, `billing_error` and `buyCreditsUrl` are signals because they are the
    provider's own naming — see `test_a_provider_verdict_field_is_read_structurally_not_grepped`.
    """
    assert classify_failure(404, "No audio endpoint. See https://openrouter.ai/settings/credits") == "unavailable"
    # Prose wins over the stripping — outside 400/422, where a bare substring is not enough; see
    # `test_bare_false_free_substrings_do_not_quarantine_on_a_request_rejection`.
    assert classify_failure(404, "invalid image credit line format") == "billing_or_paid"
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


def test_an_identifier_echoed_from_the_request_is_never_a_paid_signal():
    """`_` is a word character, so a marker welded into an identifier is not a payment demand.

    A provider quotes the caller's own vocabulary — a tool name, a field — and URL stripping cannot
    help, because the identifier sits in the prose. `classify_failure(400, "tool_call_id
    call_credits_lookup not found")` returned `billing_or_paid`, which `cooldown_policy_for_reason`
    turns into a 24h `anti_false_free_guard` quarantine: a healthy model taken out of the pool by its
    caller's naming. Word boundaries end that on EVERY status, not only the request rejections.
    """
    for text in (
        "tool_call_id call_credits_lookup not found upstream",
        "Invalid schema for function 'billing_report': missing 'properties'",
        "unknown field credit_score in tool arguments",
        "creditsLookup is not a supported tool",
        # `\b` alone would only cover `_`, and kebab-case tool names are at least as common.
        "tool call credits-lookup failed",
        "unknown tool tools/credits",
        "field credit.balance is not allowed here",
    ):
        for status in (400, 401, 403, 404, 422, 500, 503):
            assert classify_failure(status, text) != "billing_or_paid", (status, text)

    # The provider's OWN identifiers stay signals: they are known names, so they are listed.
    assert classify_failure(500, '{"error":{"type":"billing_error"}}') == "billing_or_paid"


def test_bare_false_free_substrings_do_not_quarantine_on_a_request_rejection():
    """A 400/422 quotes the caller's request, so even a well-separated bare marker is not a signal.

    Word boundaries alone do not cover this: a tool named plainly `credits` is echoed between quotes,
    which *are* boundaries. The status is the remaining evidence — on a request rejection the message
    is about the body we sent — so those two statuses drop the bare words entirely.
    """
    for text in (
        "Invalid schema for function 'credits'",
        "unrecognized request argument supplied: billing",
        "expected one of [text, image], got 'credit'",
        # The provider's own error-code names are dropped here too. They are listed so the identifier
        # class does not hide the provider's verdict, but a caller can name a tool `payment_required`
        # just as easily — and on a request rejection the message is about what the caller sent.
        "Invalid schema for function 'payment_required'",
        "unknown field insufficient_credits in tool arguments",
    ):
        # `bad_upstream_request`, not a payment demand: the provider rejected the body we sent.
        assert classify_failure(400, text) == "bad_upstream_request", text
        assert classify_failure(422, text) == "bad_upstream_request", text

    # Only 400/422 narrows. Every other status keeps the bare markers, which is what catches the
    # providers that answer a payment demand with a status of their own choosing.
    assert classify_failure(404, "unrecognized request argument supplied: billing") == "billing_or_paid"


def test_explicit_payment_demands_still_quarantine_on_a_request_rejection():
    """Strict-zero holds: a 400 that really demands payment must still trip the false-free guard."""
    for demand in (
        "Payment required: this model is no longer free",
        "Your credit balance is too low to access this model",
        "add credits to continue",
        "insufficient balance for this request",
        "This request requires more credits, or fewer max_tokens",
        "billing account not configured for this project",
        "Free promotion has ended for this model",
    ):
        assert classify_failure(400, demand) == "billing_or_paid", demand
        assert classify_failure(422, demand) == "billing_or_paid", demand


def test_an_unknown_status_narrows_the_markers_like_a_request_rejection():
    """No status to read corroborates nothing, so the bare markers must not decide alone."""
    assert classify_failure(None, "tool_call_id call_credits_lookup not found") == "unavailable"
    assert classify_failure(None, "add credits to continue") == "billing_or_paid"


def test_a_provider_error_code_resolves_to_the_status_it_names():
    """A named code is a status in another alphabet; anything else stays unknown."""
    for code, status in ERROR_CODE_STATUSES.items():
        assert status_for_error_codes(code.upper()) == status, code
    assert status_for_error_codes("tool_use_failed") is None
    assert status_for_error_codes(402) is None  # already a status, parsed by the caller
    assert status_for_error_codes(None) is None

    # A payment condition resolves to 402 on the word alone. This field is the provider's vocabulary,
    # never the caller's, so a bare substring is safe here — which is what covers the spellings no
    # list of exact codes anticipates.
    for code in ("billing_error", "insufficient_credit", "credit_limit_exceeded", "creditBalance"):
        assert status_for_error_codes(code) == 402, code
    # …but `balance`/`funds` only in full, or a load balancer would read as a payment demand.
    assert status_for_error_codes("insufficient_balance") == 402
    assert status_for_error_codes("load_balancer_error") is None


def test_a_named_request_rejection_narrows_whatever_status_wraps_it():
    """A gateway relaying an upstream 400 under its own 502 is still telling us it is our request.

    The narrowing keyed on the HTTP status alone, so `{"code": "invalid_request_error"}` wrapped in a
    502 still read a tool named `credits` as a payment demand and quarantined a healthy model for 24h
    — the defect this whole area exists to prevent, on the one door the status pairing cannot see.
    """
    echoed = json.dumps({"error": {"code": "invalid_request_error", "message": "Invalid schema for function 'credits'"}})
    for status in (400, 404, 422, 500, 502, 503):
        assert classify_failure(status, echoed) != "billing_or_paid", status

    # It narrows the markers; it does not overrule the status itself.
    assert classify_failure(503, echoed) == "server_error"
    # And a real demand alongside the same code still classifies.
    demanded = json.dumps({"error": {"code": "invalid_request_error", "message": "add credits to continue"}})
    assert classify_failure(503, demanded) == "billing_or_paid"


def test_a_payment_word_must_be_a_whole_word_of_the_error_code():
    """The field is the provider's vocabulary, which rules out the caller's echo — not a longer word.

    `error.code` is matched without the identifier rule that guards the prose, so a bare substring
    read `discredit` and `prepayment_ok` as payment demands. Splitting the code into words first is
    what keeps that from quarantining a model, and it also removes the need to spell `balance` and
    `funds` in full to dodge `load_balancer_error`.
    """
    for code in ("accreditation_failed", "discredit", "subcredit", "billinghistory", "load_balancer_error"):
        assert status_for_error_codes(code) is None, code
    for code in ("insufficient_credit", "credit_limit_exceeded", "creditBalance", "CREDIT_LIMIT", "insufficient_funds"):
        assert status_for_error_codes(code) == 402, code


def test_a_named_status_outranks_a_payment_word_in_another_field():
    """A provider saying "this is about your request" must not lose to a word inside `error.code`.

    `code` is read before `type`, so once a payment word alone resolved to 402, a tool named
    `billing_report` outranked `type: invalid_request_error` and quarantined the model for 24h — the
    original defect, re-entered through the field the fallback was meant to make safe.
    """
    assert status_for_error_codes("billing_report", "invalid_request_error") == 400
    # With no canonical name anywhere, the payment word still decides.
    assert status_for_error_codes("insufficient_credit", "some_unknown_type") == 402


def test_a_provider_verdict_field_is_read_structurally_not_grepped():
    """`error.code`/`error.type`/`error.metadata` keys are the provider's own naming, so they count.

    They used to be covered by a hand-listed set of exact spellings grepped out of the flattened
    body, which could not tell `{"type":"billing_error"}` (a verdict) from
    `function 'billing_report'` (the caller's tool). Reading the fields removes both the list and the
    ambiguity — and the same names reach the classifier whether the body arrived raw or parsed.
    """
    assert provider_error_codes('{"error":{"type":"billing_error","message":"upstream"}}') == ("billing_error",)
    assert classify_failure(503, '{"error":{"type":"billing_error","message":"upstream"}}') == "billing_or_paid"
    assert classify_failure(500, '{"error":{"code":"credit_limit_exceeded","message":"x"}}') == "billing_or_paid"

    # A body long enough to have been cut by the old 1 KB slice still parses, which is the whole
    # point of handing `classify_failure` the untruncated text.
    long_body = '{"error":{"code":"insufficient_credit","message":"%s"}}' % ("upstream rejected it. " * 60)
    assert len(long_body) > 1000 and classify_failure(500, long_body) == "billing_or_paid"

    # `metadata` is NOT read — it is a free-form bag, and an accounting key on an unrelated failure
    # would quarantine a healthy model for 24h.
    assert provider_error_codes('{"error":{"message":"x","metadata":{"credits_used":0.0}}}') == ()
    assert classify_failure(500, '{"error":{"message":"upstream timeout","metadata":{"credits_used":0}}}') == "server_error"

    # The caller's vocabulary never reaches those fields, so an echo in the prose still does not count
    # — and a canonical name in a sibling field still outranks a payment word in `code`.
    echo = '{"error":{"code":"tool_use_failed","message":"tool_call_id call_credits_lookup not found"}}'
    assert classify_failure(500, echo) == "server_error"
    contradicted = '{"error":{"type":"invalid_request_error","code":"billing_report","message":"Invalid schema"}}'
    assert classify_failure(400, contradicted) == "bad_upstream_request"

    # Anything not shaped like a provider error body falls back to the prose rules.
    assert provider_error_codes("plain text upstream failure") == ()
    assert provider_error_codes('{"error":{"message":"truncated mid-ob') == ()
    assert provider_error_codes('{"error":"not an object"}') == ()
    # …including a body that blows the JSON parser's recursion limit while staying under the size
    # one. `RecursionError` is not a `ValueError`, and letting it out turns a failover into a 500:
    # the chat path calls `classify_failure` outside the try/except that wraps `invoke_model`.
    # Whether the parser actually gives up at this depth is a CPython implementation limit that
    # moves between versions — 3.11 raises where 3.14 parses the same body — so pin the invariant
    # that holds on both rather than the symptom: the call returns a verdict instead of letting
    # RecursionError out, and the classification stays server_error either way.
    deeply_nested = '{"error":{"code":"x","m":%s}}' % ("[" * 25_000 + "]" * 25_000)
    assert len(deeply_nested) < 64_000
    assert isinstance(provider_error_codes(deeply_nested), tuple)
    assert classify_failure(500, deeply_nested) == "server_error"


def test_every_false_free_marker_still_matches_its_own_wording():
    """A marker that starts or ends on an identifier character can never match anything.

    Trivially true for a well-formed word, which is the point: it guards the entries someone adds
    later with leading or trailing punctuation, where the pattern would silently read as coverage
    while matching nothing.
    """
    for markers in (FALSE_FREE_TEXT_MARKERS, FALSE_FREE_PAYMENT_DEMAND_MARKERS):
        pattern = false_free_pattern(markers)
        for marker in markers:
            assert pattern.search(marker), marker


def test_an_empty_marker_set_matches_nothing():
    """`re.compile("")` matches every string — an adapter clearing the markers must disable them."""
    assert false_free_pattern(()).search("add credits to continue") is None


def test_request_rejection_markers_only_narrow_the_default_set():
    """The 400/422 list never classifies a message the other statuses would let through.

    Half of it is structural — it is derived from `FALSE_FREE_TEXT_MARKERS` — but the explicit
    phrasings appended to it are hand-written, and one that no bare marker covers would make a 400
    quarantine where a 404 does not.
    """
    for marker in FALSE_FREE_PAYMENT_DEMAND_MARKERS:
        assert any(bare in marker for bare in FALSE_FREE_TEXT_MARKERS), marker


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


def test_a_model_specific_sampling_rejection_fails_over_without_cooling():
    """Mistral may accept the body generally but disable one option on one model.

    Jan sends ``top_k`` to the OpenAI-compatible endpoint. Mistral's 3051 rejection explicitly
    says the option is disabled *for this model*, so another candidate can answer the same request.
    Treating it as a generic bad body stopped the chain and leaked the 400 back to Jan.
    """
    mistral_400 = (
        '{"object":"error","message":"top_k sampling is not enabled for this model",'
        '"type":"invalid_request_invalid_args","param":null,"code":"3051"}'
    )

    assert classify_failure(400, mistral_400) == "bad_upstream_contract"
    policy = cooldown_policy_for_reason("bad_upstream_contract", source="mistral")
    assert policy.model_cooldown is False
    assert policy.provider_cooldown is False
    assert policy.quarantine is None


def test_a_forbidden_private_assistant_field_fails_over_without_cooling():
    """One provider may reject reasoning metadata another candidate needs and accepts.

    Ficelle can restore a previous provider's reasoning trace onto a replayed tool call. Mistral's
    request schema rejects that private assistant field before inference, but the unchanged history
    remains valid for the provider that emitted it. That is a contract mismatch, not a malformed
    request across the whole pool.
    """
    mistral_422 = json.dumps(
        {
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 2, "assistant", "reasoning_content"],
                    "msg": "Extra inputs are not permitted",
                    "input": "The venue says payment required at the door.",
                }
            ]
        }
    )

    assert classify_failure(422, mistral_422) == "bad_upstream_contract"
    policy = cooldown_policy_for_reason("bad_upstream_contract", source="mistral")
    assert policy.model_cooldown is False
    assert policy.provider_cooldown is False
    assert policy.quarantine is None


def test_only_structured_private_assistant_field_rejections_get_contract_failover():
    """Keep the matcher fail-closed for real body defects and spoofed prose."""
    cases = [
        {
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 2, "assistant", "content"],
                    "msg": "Extra inputs are not permitted",
                }
            ]
        },
        {
            "detail": [
                {
                    "type": "string_type",
                    "loc": ["body", "messages", 2, "assistant", "reasoning_content"],
                    "msg": "Input should be a valid string",
                }
            ]
        },
        {"error": {"message": "extra_forbidden reasoning_content: Extra inputs are not permitted"}},
        # A negative for the client-metadata reading too: `message_id` is only excused, an
        # ordinary field like `content` on any role stays a genuinely malformed body.
        {
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 3, "user", "content"],
                    "msg": "Extra inputs are not permitted",
                }
            ]
        },
    ]

    for payload in cases:
        assert classify_failure(422, json.dumps(payload)) == "bad_upstream_request"


def test_a_forbidden_client_message_id_fails_over_without_cooling():
    """Hermes stamps `message_id` on every user message; Mistral's schema rejects it as extra.

    That is a provider-strictness mismatch, not a malformed body: most providers ignore the field
    outright, so a sibling candidate remains answerable with the unchanged history. Real body from
    request b784a0ab3f624260adc72b84e4ebddad (2026-09-11 17:06:27).
    """
    mistral_422 = json.dumps(
        {
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 10, "user", "message_id"],
                    "msg": "Extra inputs are not permitted",
                    "input": "1547981489512120481",
                }
            ]
        }
    )

    assert classify_failure(422, mistral_422) == "bad_upstream_contract"
    policy = cooldown_policy_for_reason("bad_upstream_contract", source="mistral")
    assert policy.model_cooldown is False
    assert policy.provider_cooldown is False
    assert policy.quarantine is None


def test_a_forbidden_client_message_id_without_a_role_segment_fails_over_without_cooling():
    """Some providers report the rejection without a role in the `loc` path at all: `location[-2]`
    is then the message index, not a role. `message_id` must still be excused in that shape.
    """
    upstream_422 = json.dumps(
        {
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 10, "message_id"],
                    "msg": "Extra inputs are not permitted",
                }
            ]
        }
    )

    assert classify_failure(422, upstream_422) == "bad_upstream_contract"

    # The negative lock: the same roleless shape, but for an ordinary field, stays a genuinely
    # malformed body — the roleless allowance is specific to CLIENT_MESSAGE_METADATA_FIELDS.
    upstream_422_content = json.dumps(
        {
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 10, "content"],
                    "msg": "Extra inputs are not permitted",
                }
            ]
        }
    )

    assert classify_failure(422, upstream_422_content) == "bad_upstream_request"


def test_multiple_forbidden_client_message_ids_fail_over_without_cooling():
    """The same rejection can list one `extra_forbidden` entry per offending message.

    Real body from the same day (2026-09-11 14:02:58): Mistral names both message 1 and message 30.
    """
    mistral_422 = json.dumps(
        {
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 1, "user", "message_id"],
                    "msg": "Extra inputs are not permitted",
                    "input": "1547981489512120400",
                },
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "messages", 30, "user", "message_id"],
                    "msg": "Extra inputs are not permitted",
                    "input": "1547981489512120481",
                },
            ]
        }
    )

    assert classify_failure(422, mistral_422) == "bad_upstream_contract"
    policy = cooldown_policy_for_reason("bad_upstream_contract", source="mistral")
    assert policy.model_cooldown is False
    assert policy.provider_cooldown is False
    assert policy.quarantine is None


def test_caller_caused_policies_record_the_failure_without_cooling():
    """No caller-caused failure may take a healthy model out of the pool.

    Recording still happens — `set_cooldown` runs `update_failure_stats` and
    `record_model_error_in_state` before consulting the policy — so a model that truncates on every
    request stays scored and visible instead of silently keeping a perfect record. Driven off the
    set itself, so a member added later cannot quietly skip the contract.
    """
    for reason in sorted(NO_MODEL_FAULT_FAILURE_REASONS):
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


def test_a_restart_names_itself_in_both_the_status_and_the_payload():
    """The run's ending is carried, not re-derived, and the two halves cannot drift.

    The row that records the abandonment is `attempted: False`, so every scan over the error list
    skips it: the status was 502 `upstream_failure` for a run whose route log already said
    `service_restarting`, blaming a pool that was never asked.
    """
    errors = [
        {"model": "ficelle/nous/a", "reason": "server_error", "status": 503},
        {"model": "ficelle/groq/b", "reason": "service_restarting", "attempted": False},
    ]

    assert upstream_failure_status(errors) == 502
    assert upstream_failure_status(errors, terminal_reason="service_restarting") == 503

    error = build_upstream_failure_error(
        "ficelle/auto-fast", "req-stop", 2, [{"model": "x"}], errors, "service_restarting"
    )["error"]
    assert error["type"] == "service_restarting"
    assert error["message"].startswith("Ficelle stopped taking new fallback attempts")
    assert any("service restart or update" in action for action in error["actions"])

    # Ficelle's own deadline deliberately does not restate the status: the attempted failures are
    # what the caller needs to see, and a 504 over them would hide the 503 that actually happened.
    assert upstream_failure_status(errors, terminal_reason="request_deadline_exceeded") == 502


def test_the_restart_does_not_outrank_what_was_wrong_with_the_request():
    """A SIGTERM changes when the run stopped, never what the caller sent.

    Every attempt said the upstream refused the body itself, so the truthful answer is that 400
    with the provider's own words — a 503 would tell the client to retry a request that will be
    rejected identically every time.
    """
    errors = [
        {"model": "ficelle/nous/a", "reason": "bad_upstream_request", "status": 400, "detail": "bad tool_call"},
        {"model": "ficelle/groq/b", "reason": "service_restarting", "attempted": False},
    ]

    assert upstream_failure_status(errors, terminal_reason="service_restarting") == 400

    error = build_upstream_failure_error(
        "ficelle/auto-fast", "req-stop", 2, [{"model": "x"}], errors, "service_restarting"
    )["error"]
    assert error["type"] == "invalid_request_error"
    assert "bad tool_call" in error["message"]


def test_a_dead_caller_is_logged_with_a_status_nobody_was_sent():
    """`client_disconnected` writes no body at all, so the row must not claim a status.

    The route log used to carry the 502/504/429 the run would have answered, which reads on the
    Requests page as a failure the client saw. 499 says what happened: the caller closed first.
    """
    errors = [
        {"model": "ficelle/nous/a", "reason": "timeout", "status": "timeout"},
        {"model": "ficelle/groq/b", "reason": "client_disconnected", "attempted": False},
    ]

    assert upstream_failure_status(errors, terminal_reason="client_disconnected") == 504
    assert route_log_failure_status(errors, terminal_reason="client_disconnected") == 499
    # Every other ending is shown with the status the client actually received.
    assert route_log_failure_status(errors, terminal_reason="service_restarting") == 503
    assert route_log_failure_status(errors, terminal_reason="request_deadline_exceeded") == 504
    assert not TERMINAL_ENDINGS["client_disconnected"].writes_body


def test_redaction_bounds_its_input_before_the_patterns_run(monkeypatch):
    """A provider body cannot hand the redaction regexes an unbounded string.

    Every input here is provider- or client-controlled and the seven patterns scan whatever they
    are given, for a value that is then cut to a couple of hundred characters. The bound lives in
    `redact_sensitive_text` so it holds for every caller, not only the one that remembered it.
    """
    from ficelle import redaction

    # The real patterns still redact what is inside the bound.
    assert redaction.sanitize_error_detail("boom Authorization: Bearer abc123def", 180) == (
        "boom [redacted]"
    )

    seen: list[int] = []

    class RecordingPattern:
        def sub(self, _replacement, text):
            seen.append(len(text))
            return text

    monkeypatch.setattr(redaction, "SENSITIVE_ERROR_PATTERNS", [RecordingPattern()])
    detail = redaction.sanitize_error_detail("x" * 200_000, 180)

    assert seen == [redaction.REDACTION_INPUT_LIMIT]
    assert len(detail) == 180


def test_a_tls_failure_is_recognized_through_its_wrappers():
    """A handshake failure reaches the router wrapped in whatever the layer above made of it."""

    class SSLFakeError(Exception):
        pass

    assert exception_is_tls_failure(SSLFakeError("certificate verify failed"))
    assert exception_is_tls_failure(ConnectionError("wrapped", SSLFakeError("certificate verify failed")))
    assert not exception_is_tls_failure(ConnectionError("connection refused"))


def test_model_not_found_body_is_one_wording_for_every_route():
    """The lookup route and the chat refusal state the same fact, so they state it identically."""
    body = model_not_found_body("ficelle/does-not-exist")

    assert body == {
        "error": {
            "message": "model ficelle/does-not-exist is not served by this Ficelle",
            "type": "model_not_found",
        }
    }


def test_caller_rejected_request_needs_every_attempt_to_agree():
    assert caller_rejected_request([{"reason": "bad_upstream_request"}]) is True
    assert caller_rejected_request([{"reason": "bad_upstream_request"}, {"reason": "rate_limited"}]) is False
    assert caller_rejected_request([{"reason": "server_error"}]) is False
    # No attempt reached an upstream at all: nothing says the body is what failed.
    assert caller_rejected_request([]) is False


def test_no_model_fault_failures_are_the_only_ones_exempt_from_the_streak():
    """The consecutive-failure streak is reserved for what the model is answerable for.

    Its penalty is cumulative (12 points each), so counting a caller's too-small max_tokens, its
    malformed request body, or its own mid-stream hangup would let one misbehaving client
    progressively demote every candidate it touches. The failure is still counted in
    `requests`/`failures` and still shown — only the streak is left alone.

    `bad_upstream_contract` is the one member no caller caused: the exemption is there because the
    turn that fails is the one Ficelle learns the model's trace on, and the next one succeeds.
    Cooling or demoting a model over it would bench a working candidate for being fixed. The
    deadline is Ficelle's own doing, which is why the set is named for the absence of a model fault
    rather than for the caller: on a deadline the earlier attempts are what spent the budget.
    """
    assert NO_MODEL_FAULT_FAILURE_REASONS == {
        "truncated_before_content",
        "bad_upstream_request",
        "bad_upstream_contract",
        "client_disconnected",
        "request_deadline_exceeded",
        "context_length_exceeded",
    }
    # `service_restarting` is not a member and needs no exemption: the loop records it as
    # `attempted: False` and breaks, so it never reaches a cooldown or a scoring streak. What it
    # does mean lives in `TERMINAL_ENDINGS`.
    assert "service_restarting" not in NO_MODEL_FAULT_FAILURE_REASONS
    for upstream_fault in ("unavailable", "server_error", "timeout", "empty_assistant_message"):
        assert upstream_fault not in NO_MODEL_FAULT_FAILURE_REASONS


def test_provider_scoped_policy_uses_the_provider_cooldown_as_sole_blocking_scope():
    # L2-R4 (synthetic-health remediation): provider-wide rate_limited/auth_or_credit
    # write ONE blocking cooldown — the provider's. The model keeps its recorded attempt,
    # error, and stats for diagnosis, but no model cooldown that could outlive recovery.
    for reason in ("rate_limited", "auth_or_credit"):
        policy = cooldown_policy_for_reason(reason, source="openrouter")

        assert policy.record_provider_error is True
        assert policy.provider_cooldown is True
        assert policy.provider_cooldown_source == "openrouter"
        assert policy.quota_cooldown is False
        assert policy.quarantine is None
        assert policy.model_cooldown is False


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


def test_false_free_policy_quarantines_without_provider_cooldown():
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


def test_runaway_output_cooldown_is_model_scoped_only():
    # The model answered -- it just kept going past the profile's completion budget. That is
    # answerable by the model, so it is cooled like any other transient model-side failure, but
    # never quarantined or promoted to a provider-wide cooldown: a sibling model behind the same
    # provider is not implicated by this one running away.
    policy = cooldown_policy_for_reason("runaway_output", source="openrouter")

    assert policy.record_provider_error is False
    assert policy.provider_cooldown is False
    assert policy.quota_cooldown is False
    assert policy.quarantine is None
    assert policy.model_cooldown is True


def test_runaway_output_is_not_exempt_from_production_profile_evidence():
    assert "runaway_output" not in NO_MODEL_FAULT_FAILURE_REASONS
    assert "runaway_output" in PRODUCTION_PROFILE_FAILURE_REASONS


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


# --- Lot 4 (synthetic-health remediation) L4-R1: capacity dimensions ------------------------


def test_capacity_dimension_names_the_real_limit_without_changing_classification():
    from ficelle.failures import capacity_dimension, classify_failure

    groq_tpm = "Request too large for model. TPM Limit 12000, Requested 41215. Please reduce your message size."
    assert capacity_dimension(413, groq_tpm) == "capacity.tpm"
    # The classification itself stays request_too_large, model-scoped (SHR-011).
    assert classify_failure(413, groq_tpm) == "request_too_large"

    assert capacity_dimension(429, "requests per minute exceeded for this key") == "capacity.rpm"
    assert capacity_dimension(429, "free tier quota exhausted until tomorrow") == "capacity.free_quota"
    assert capacity_dimension(429, "too many concurrent requests") == "capacity.concurrent"
    # The seventh baseline Groq case: a generic 413 without any stated dimension.
    assert capacity_dimension(413, "Request Entity Too Large") == "capacity.context_or_body"
    assert capacity_dimension(413, "nope") == "capacity.unknown"
    # Non-capacity statuses never get a dimension.
    assert capacity_dimension(500, "tokens per minute") is None
    assert capacity_dimension(None, "tpm") is None


# --- Upstream treatment of rejected sampling parameters -------------------------------------


def test_rejected_sampling_parameters_names_only_refused_present_knobs():
    from ficelle.failures import rejected_sampling_parameters

    mistral = '{"message":"top_k sampling is not enabled for this model","type":"invalid_request_invalid_args","code":"3051"}'
    body = {"model": "m", "messages": [], "top_k": 1, "temperature": 0}

    assert rejected_sampling_parameters(mistral, body) == ("top_k",)
    # Not carried by the request: nothing to drop, whatever the message says.
    assert rejected_sampling_parameters(mistral, {"model": "m", "messages": []}) == ()
    # A message that merely echoes a value is not a rejection verdict.
    assert rejected_sampling_parameters('{"message":"received top_k=1"}', body) == ()
    # Intent-carrying fields are never droppable, even when named as unsupported.
    tools_rejection = '{"message":"tools is not supported by this model"}'
    assert rejected_sampling_parameters(tools_rejection, {"tools": [], "messages": []}) == ()
    schema_rejection = '{"message":"response_format is not supported for this model"}'
    assert rejected_sampling_parameters(schema_rejection, {"response_format": {}, "messages": []}) == ()
    # Several knobs at once, deduplicated to what the body carries.
    multi = '{"message":"top_k and min_p are not supported by this model"}'
    assert set(rejected_sampling_parameters(multi, {"top_k": 1, "min_p": 0.1, "top_p": 0.9})) == {"top_k", "min_p"}


# --- Incident 2026-09-11: OpenRouter agentic-harness gate must quarantine, not cool -----------


def test_an_openrouter_agentic_harness_gate_quarantines_the_model_not_the_provider():
    """Live incident, 2026-09-11 11:13 UTC: `thinkingmachines/inkling-small:free` answered this 403
    while every other OpenRouter model kept serving requests. The generic 403 fallback
    (`auth_or_credit`) cooled all of OpenRouter for an hour; the provider-specific marker must win
    instead and quarantine only the gated model.
    """
    body = (
        '{"error":{"message":"thinkingmachines/inkling-small:free is only available on agentic '
        'harnesses. Try plugging it into a coding agent or productivity app listed on '
        'https://openrouter.ai/apps","code":403,"metadata":{"routing_funnel":[{"step":"Initial '
        'Endpoints","endpoint_count":1}],"failed_routing_step":"Gate Free Endpoints by Agentic '
        'Harness"}}}'
    )
    markers = FailureMarkers().with_extra(model_not_entitled=("only available on agentic harnesses",))

    reason = classify_failure(
        403,
        body,
        markers=markers,
        upstream_model_id="thinkingmachines/inkling-small:free",
    )

    assert reason == "model_not_found"
    policy = cooldown_policy_for_reason(reason, source="openrouter")
    assert policy.provider_cooldown is False
    assert policy.quarantine is not None
    assert policy.quarantine.reason == "model_not_found"


def test_an_agentic_harness_403_that_does_not_name_the_model_stays_auth_or_credit():
    """The same marker, on a body that never names the model asked for, is not evidence of an
    entitlement gate for THIS request — a dead key must still cool the whole provider."""
    body = '{"error":{"message":"only available on agentic harnesses","code":403}}'
    markers = FailureMarkers().with_extra(model_not_entitled=("only available on agentic harnesses",))

    reason = classify_failure(
        403,
        body,
        markers=markers,
        upstream_model_id="thinkingmachines/inkling-small:free",
    )

    assert reason == "auth_or_credit"


# --- Incident 2026-09-11: Groq's unknown message property is a contract mismatch, not a bad body -


_GROQ_UNKNOWN_MESSAGE_PROPERTY_BODY = (
    "{\"error\":{\"message\":\"'messages.29' : for 'role:user' the following must be satisfied"
    "[('messages.29' : property 'message_id' is unsupported)]\",\"type\":\"invalid_request_error\"}}"
)


def test_groq_rejecting_an_unknown_message_property_diverts_without_cooling():
    """Groq validates message keys strictly and rejects `message_id`, a field Hermes adds; most
    providers ignore unknown message keys. That is the provider's strictness, not a malformed
    request, so a sibling source must get its turn instead of the run ending early."""
    reason = classify_failure(400, _GROQ_UNKNOWN_MESSAGE_PROPERTY_BODY)

    assert reason == "bad_upstream_contract"
    policy = cooldown_policy_for_reason(reason, source="groq")
    assert policy.model_cooldown is False
    assert "bad_upstream_contract" in SOURCE_DIVERTING_FAILURE_REASONS


def test_an_unrelated_400_still_classifies_as_a_terminal_bad_request():
    """The narrow `property '<x>' is unsupported` pattern must not widen to generic 400 prose."""
    reason = classify_failure(400, '{"error":{"message":"invalid JSON body","type":"invalid_request_error"}}')

    assert reason == "bad_upstream_request"


def test_an_unrelated_unsupported_property_is_a_terminal_bad_request():
    body = '{"error":{"message":"property \'foo\' is unsupported"}}'

    assert classify_failure(400, body) == "bad_upstream_request"
