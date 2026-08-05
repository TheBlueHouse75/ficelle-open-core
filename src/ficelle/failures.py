from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from ficelle.redaction import sanitize_error_detail


# Provider error messages routinely end with a link to an upgrade or settings page, and a URL can
# put `credits` somewhere the identifier rule below still accepts it (`?plan=credits`, `#credits`) —
# so a link appended to an unrelated rejection would read as a payment demand and quarantine a
# healthy model. URLs are stripped first; a real payment demand states it in prose.
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

# Provider error-code spellings, kept as text markers for the bodies `error_object_codes` cannot
# read: a non-JSON body, an unusual shape, an upstream payload nested inside a string. Restricted to
# identifiers a caller would not plausibly name a tool — `payment_required` is deliberately absent,
# since the structural read covers it and a tool could carry that name.
FALSE_FREE_ERROR_CODE_MARKERS = (
    "billing_error",
    "billing_hard_limit_reached",
    "buycreditsurl",
    "insufficient_credits",
)

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
) + FALSE_FREE_ERROR_CODE_MARKERS

# Every marker that is a single token rather than prose. `false_free_pattern` already keeps these
# from matching inside an identifier, but a token can also be echoed with real boundaries around it —
# a tool named plainly `credits` comes back as `Invalid schema for function 'credits'`, and quotes
# *are* boundaries. HTTP 400/422 is where that is the norm, since the message is about the body we
# just sent, so those two statuses drop the single tokens entirely and keep only explicit prose,
# pairing status and markers the way the 404/410 branch already does for `model_not_found`. A
# provider that really demands payment on a 400 states it in prose, and 402 stands on its own.
SINGLE_TOKEN_FALSE_FREE_MARKERS = ("billing", "credit", "credits") + FALSE_FREE_ERROR_CODE_MARKERS

# Derived, not re-typed: a marker added to FALSE_FREE_TEXT_MARKERS must keep applying on 400/422, or
# the narrowing would silently start dropping real payment demands. The extras below are the
# phrasings the single tokens used to cover on their own.
FALSE_FREE_PAYMENT_DEMAND_MARKERS = tuple(
    marker for marker in FALSE_FREE_TEXT_MARKERS if marker not in SINGLE_TOKEN_FALSE_FREE_MARKERS
) + (
    "insufficient credit",
    "requires more credits",
    "add credits",
    "buy credits",
    "purchase credits",
    "out of credits",
    "credits have run out",
    "credits remaining",
    "credit balance",
    "credit limit",
    "billing account",
    "billing issue",
    "billing required",
    "enable billing",
)

# Characters an identifier is built from. `\b` would only cover `_`, leaving `credits-lookup`,
# `tools/credits` and `credit.balance` matching — and kebab-case tool names are at least as common as
# snake_case. The trade-off is that a single token followed by a period ("out of credit.") stops
# counting on its own; explicit prose and HTTP 402 still carry those.
_IDENTIFIER_CHARS = r"[\w./-]"


@lru_cache(maxsize=None)
def false_free_pattern(markers: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a false-free marker set so single tokens cannot match inside an identifier.

    A provider message routinely quotes an identifier the caller sent — `call_credits_lookup`,
    `billing_report`, `credits-lookup` — and a plain substring match reads those as a payment demand,
    which quarantines a healthy model for 24h. Requiring that no identifier character sits on either
    side ends that class on *every* status, where the narrowed marker set only covers the request
    rejections.

    Phrases stay plain substrings: a space cannot occur inside an identifier, so they are not exposed
    to the echo, and substring matching is what makes them deliberately cover their own inflections
    ("insufficient credit" catching "insufficient credits").
    """
    parts = [
        re.escape(marker)
        if " " in marker
        else rf"(?<!{_IDENTIFIER_CHARS}){re.escape(marker)}(?!{_IDENTIFIER_CHARS})"
        for marker in markers
    ]
    # An adapter that cleared its markers must disable them: `re.compile("")` matches *everything*
    # and would turn every failure into a payment demand.
    return re.compile("|".join(parts) or r"(?!)")


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

# A provider that wraps its rejection in an HTTP 200 payload puts the status in `error.code` or
# `error.type` — sometimes as a number, sometimes as OpenAI's/Anthropic's name for it. A name is not
# "no status": dropping it classifies a dead API key as a plain model failure and a spent quota as a
# healthy model, where the same message carried by its real HTTP status classifies as
# `auth_or_credit` / `rate_limited`. Translating the canonical names keeps both doors in agreement;
# anything else (`"tool_use_failed"`, `"invalid_value"`) really is unknown and stays `None`.
ERROR_CODE_STATUSES = {
    "api_error": 500,
    "authentication_error": 401,
    "insufficient_quota": 429,
    "invalid_api_key": 401,
    "invalid_request_error": 400,
    "model_not_found": 404,
    "not_found_error": 404,
    "overloaded_error": 503,
    "permission_error": 403,
    "rate_limit_error": 429,
}


# Words that make a provider's own error code a payment condition, whatever it spelled it — matched
# against the code's WORDS, not as substrings of it. `error.code`/`error.type` is the PROVIDER's
# vocabulary rather than the caller's, which is why the message-prose identifier rule does not apply
# here; but that only rules out an echo, so `discredit` and `prepayment_ok` would still have read as
# payment demands. Splitting first is what covers the spellings no list anticipates —
# `insufficient_credit` (singular), `credit_limit_exceeded`, `creditBalance` — without them.
PAYMENT_CODE_WORDS = frozenset({"billing", "credit", "credits", "payment", "balance", "funds"})

# `_`, `-`, `.` and a camelCase hump all separate words in a provider's code. Split on the ORIGINAL
# casing — lowercasing first would weld `creditBalance` into one word.
_CODE_WORD_BOUNDARY = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def error_code_words(code: str) -> set[str]:
    """Split a provider error code into its words, so a match cannot land inside a longer one."""
    return {word.lower() for word in _CODE_WORD_BOUNDARY.split(code) if word}


# The prose markers read the head of the body only — a gateway that echoes its upstream's payload or
# a stack trace can answer with far more than anyone states a payment demand in. The structural read
# below parses further, because `error.code` can sit behind a long `message`, but still refuses a
# body no error payload would plausibly reach.
PROSE_SCAN_LIMIT = 1000
ERROR_BODY_PARSE_LIMIT = 64_000


def error_object_codes(error: dict[str, Any]) -> tuple[str, ...]:
    """The names a provider chose for an error: its `code` and `type`.

    This is the structural discriminator the prose markers cannot have. These two fields are the
    PROVIDER's verdict — the caller's vocabulary lands in `message` and `param` — so a payment word
    inside them counts, where the same word in the prose first has to clear the identifier rule
    (`false_free_pattern`). It is what lets `{"type":"billing_error"}` classify while
    `Invalid schema for function 'billing_report'` does not, with no list of spellings to maintain.

    `metadata` is deliberately NOT read: it is a free-form bag, and a key as ordinary as
    `credits_used` on a timeout would quarantine a healthy model for 24h.
    """
    return tuple(error[key] for key in ("code", "type") if isinstance(error.get(key), str))


def provider_error_codes(text: str) -> tuple[str, ...]:
    """`error_object_codes` for a raw response body, so every path reads the same fields.

    Returns `()` for a body that is not JSON, is shaped differently, or is too large to be an error
    payload — all of which fall back to matching the prose.
    """
    if len(text) > ERROR_BODY_PARSE_LIMIT:
        return ()
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError):
        # RecursionError is not a ValueError: `json.loads` raises it on a deeply nested body, which
        # fits well under the size limit. Letting it out turns a failover into a 500, since the chat
        # path calls `classify_failure` outside the try/except that wraps `invoke_model`.
        return ()
    error = payload.get("error") if isinstance(payload, dict) else None
    return error_object_codes(error) if isinstance(error, dict) else ()


def status_for_error_codes(*values: Any) -> int | None:
    """Map a provider's textual error code/type to the HTTP status it stands for.

    A canonical name wins over the payment-word fallback across ALL the fields, not field by field: a
    provider answering `type: invalid_request_error` alongside `code: billing_report` is telling us
    the message is about the request we sent, and a `code` that merely *contains* a payment word must
    not outrank that — `billing_report` is the caller's tool.
    """
    codes = [value.strip() for value in values if isinstance(value, str)]
    for code in codes:
        mapped = ERROR_CODE_STATUSES.get(code.lower())
        if mapped is not None:
            return mapped
    # A code naming a payment condition is that provider's spelling of 402, which stands on its own.
    for code in codes:
        if error_code_words(code) & PAYMENT_CODE_WORDS:
            return 402
    return None


@dataclass(frozen=True)
class FailureMarkers:
    false_free: tuple[str, ...] = FALSE_FREE_TEXT_MARKERS
    false_free_payment_demand: tuple[str, ...] = FALSE_FREE_PAYMENT_DEMAND_MARKERS
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
            # A provider adapter's extra markers are deliberate, provider-specific phrasings rather
            # than the generic substrings the 400/422 narrowing protects against, so they stay
            # trusted on every status.
            false_free_payment_demand=self.false_free_payment_demand + false_free,
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
    status_code: int | None,
    text: str,
    *,
    error_codes: tuple[str, ...] = (),
    normalized_free_access: dict[str, Any] | None = None,
    markers: FailureMarkers | None = None,
    upstream_model_id: str | None = None,
) -> FailureReason:
    """Classify a provider failure using already-normalized free-access metadata.

    ``status_code`` is ``None`` when there is no status to read at all — a provider that wrapped its
    rejection in an HTTP 200 payload whose error code is not a status (`"invalid_request_error"`).

    Pass ``text`` UNTRUNCATED: the prose scan bounds itself (`PROSE_SCAN_LIMIT`), while the structural
    read needs the whole object to parse. Slicing it first is what silently disables the latter, since
    a body cut mid-JSON no longer loads. ``error_codes`` short-circuits that read for a caller that
    already parsed the payload.
    """
    marker_set = markers or DEFAULT_FAILURE_MARKERS
    # An unknown status corroborates nothing, so it narrows the false-free markers exactly like a
    # request rejection does: a message that carries no status is just as likely to be echoing the
    # caller's own request. Every other branch below sees the neutral 200 this used to be called with.
    unknown_status = status_code is None
    if status_code is None:
        status_code = 200
    lower = text[:PROSE_SCAN_LIMIT].lower()
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
    # The provider's own code fields skip both marker rules: they carry its verdict, not the caller's
    # vocabulary, so `status_for_error_codes` reads them directly — and its own precedence keeps a
    # canonical name (`type: invalid_request_error`) ahead of a payment word in a sibling field.
    named_status = status_for_error_codes(*(error_codes or provider_error_codes(text)))
    # A provider that NAMES the error a request rejection is saying the message is about the body we
    # sent, whatever status it wrapped that in — a gateway relaying an upstream 400 under its own 502
    # is the common case. That is the same evidence as the status itself, so it narrows the markers
    # too; without it, `{"code": "invalid_request_error"}` on a 502 still read a tool named `credits`
    # as a payment demand. See FALSE_FREE_PAYMENT_DEMAND_MARKERS.
    quotes_caller_request = unknown_status or status_code in {400, 422} or named_status in {400, 422}
    false_free = marker_set.false_free_payment_demand if quotes_caller_request else marker_set.false_free
    if status_code == 402 or named_status == 402 or false_free_pattern(false_free).search(lower_without_urls):
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
