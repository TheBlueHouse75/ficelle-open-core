from __future__ import annotations

import json

from ficelle.failures import (
    ACCOUNT_RATE_LIMIT_TEXT_MARKERS,
    BENCHMARK_ROUTE_BLOCKING_REASONS,
    CALLER_CAUSED_FAILURE_REASONS,
    FALSE_FREE_PAYMENT_DEMAND_MARKERS,
    ERROR_CODE_STATUSES,
    FALSE_FREE_TEXT_MARKERS,
    FailureMarkers,
    false_free_pattern,
    provider_error_codes,
    PROVIDER_ERROR_REASONS,
    PROVIDER_SCOPED_COOLDOWN_REASONS,
    bad_request_body,
    build_upstream_failure_error,
    caller_rejected_request,
    classify_failure,
    cooldown_policy_for_reason,
    safe_error_body,
    status_for_error_codes,
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

    `bad_upstream_contract` is the one member no caller caused: the exemption is there because the
    turn that fails is the one Ficelle learns the model's trace on, and the next one succeeds.
    Cooling or demoting a model over it would bench a working candidate for being fixed.
    """
    assert CALLER_CAUSED_FAILURE_REASONS == {
        "truncated_before_content",
        "bad_upstream_request",
        "bad_upstream_contract",
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
