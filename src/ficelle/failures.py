from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, Literal

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
    "bad_upstream_contract",
    "bad_upstream_request",
    "billing_or_paid",
    "context_length_exceeded",
    "model_not_found",
    "no_free_quota",
    "quota_exhausted",
    "rate_limited",
    "rate_limited_upstream",
    "request_too_large",
    "runaway_output",
    "server_error",
    "tls_error",
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
# Account-scope evidence *for a rate limit only*. These words are not a general scope signal:
# a 403 saying "an admin must enable them in your organization settings" is model-scoped and
# carries "organization" as a location, not as the thing at fault. The 403 branch deliberately
# does not consult this tuple — see it for why.
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

# The other half of "not serveable": the id IS deployed, but this account may not call it — a
# preview/Labs tier an admin has to switch on. Mistral says "Model <id> is a Labs model. To use
# Labs models, an admin must enable them in your organization settings". Deliberately narrow
# phrasings: they must not fire on a plain dead-key body, which is what the fail-closed default
# below still has to catch.
# Words that place a 403 at the ACCOUNT level even when the body names a model: a region block, a
# revoked key, a billing switch. Found by attacking the branch below with real-world 403 bodies —
# "You do not have access to <model> from your region" satisfied both of its other conditions and
# would have quarantined one model while leaving a provider that fails every request uncooled.
# Distinct from ACCOUNT_RATE_LIMIT_TEXT_MARKERS, which is 429-only and would misfire here.
ACCOUNT_SCOPE_403_MARKERS = (
    "region",
    "country",
    "billing",
    "revoked",
    "suspended",
    "api key",
    # Account-wide gates that a provider may phrase per-model — "an admin must enable SSO before
    # you can use <model>" carries the entitlement marker and names the model, so only naming the
    # gate itself separates it from "an admin must enable them" about a model tier. Worth closing
    # even though it looks narrow: this branch leads to a QUARANTINE, which a human has to clear,
    # where the old provider cooldown expired on its own. Misreading an account gate would
    # therefore quarantine the provider's whole catalogue, one model per probe, permanently.
    "sso",
    "two-factor",
    "2fa",
    "mfa",
    "workspace",
    "integration",
)

MODEL_NOT_ENTITLED_TEXT_MARKERS = (
    "must enable",
    "request access",
)

# What "not serveable" means, in one place. Three readers used to carry their own copy of this
# sentence and the widening from 404/410-only reached two of them, leaving a 502 body and a
# quarantine note describing a status the failure never had.
MODEL_NOT_SERVEABLE_NOTE = (
    "the provider will not serve this model id to this account — a 404/410 (listed in its catalog "
    "but not deployed), or a 403 naming it as a tier an admin has to switch on"
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
    if not isinstance(payload, dict):
        return ()
    nested_error = payload.get("error")
    # OpenAI-compatible providers use both shapes in practice: the canonical
    # ``{"error": {...}}`` envelope and a bare serialized error object. Mistral's
    # SDK exposes the latter for HTTP failures, including its stable capacity
    # code/type pair. Read the same two structural fields from either shape;
    # arbitrary nested metadata remains deliberately out of scope.
    error = nested_error if isinstance(nested_error, dict) else payload
    return error_object_codes(error)


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


def walk_exception_chain(exc: Any) -> Iterator[BaseException]:
    """Yield an exception and everything it wraps, nearest cause first, each one once.

    ``__cause__``/``__context__`` are the explicit links; ``args`` carries the rest, because
    the layers that matter here wrap by argument — `requests` hands a `urllib3` error to
    `ConnectionError(...)`, which itself carries the original `OSError`. Breadth-first so the
    innermost socket error is reached last and the caller's first match is the nearest one.
    """
    pending: deque[BaseException] = deque([exc] if isinstance(exc, BaseException) else [])
    seen: set[int] = set()
    while pending:
        current = pending.popleft()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        for linked in (current.__cause__, current.__context__, *getattr(current, "args", ())):
            if isinstance(linked, BaseException):
                pending.append(linked)


def first_exception_errno(exc: Any) -> int | None:
    """The first non-zero ``errno`` found walking an exception and its wrapped causes.

    Transport failures reach the router several layers away from the socket that produced
    them: `requests` wraps a `urllib3` error, which wraps the original `OSError`. The errno
    is the only field that separates a provider that answered slowly from a keep-alive
    connection the network dropped (``ETIMEDOUT``, ``ECONNRESET``, ``EPIPE``), so it is worth
    recovering from the chain instead of reading only the outermost type.
    """
    for current in walk_exception_chain(exc):
        candidate = getattr(current, "errno", None)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate:
            return candidate
    return None


def exception_is_tls_failure(exc: BaseException) -> bool:
    """Whether this transport exception is, or wraps, a TLS/certificate failure.

    Read off the class hierarchy rather than by importing `requests`/`ssl`: the same fault surfaces
    as `requests.exceptions.SSLError`, `ssl.SSLError` or urllib3's own subclass depending on which
    layer raised it, and `requests.exceptions.SSLError` subclasses `ConnectionError`, so the name is
    the only thing that separates it from an ordinary connection drop. The whole chain is walked for
    the same reason `first_exception_errno` walks it: a handshake failure reaches the router wrapped
    in whatever the layer above turned it into.
    """
    return any(
        base.__name__.startswith("SSL")
        for current in walk_exception_chain(exc)
        for base in type(current).__mro__
    )


@dataclass(frozen=True)
class FailureMarkers:
    false_free: tuple[str, ...] = FALSE_FREE_TEXT_MARKERS
    false_free_payment_demand: tuple[str, ...] = FALSE_FREE_PAYMENT_DEMAND_MARKERS
    quota_exhausted: tuple[str, ...] = QUOTA_EXHAUSTED_TEXT_MARKERS
    free_tier_zero_allocation: tuple[str, ...] = FREE_TIER_ZERO_ALLOCATION_MARKERS
    model_not_found: tuple[str, ...] = MODEL_NOT_FOUND_TEXT_MARKERS
    model_not_entitled: tuple[str, ...] = MODEL_NOT_ENTITLED_TEXT_MARKERS
    upstream_rate_limit: tuple[str, ...] = UPSTREAM_RATE_LIMIT_TEXT_MARKERS
    # Exact ``error.code``/``error.type`` pairs for a provider-specific model-pool saturation
    # verdict. Each pair is ordered as (code, type), matching ``error_object_codes``.
    upstream_rate_limit_error_codes: tuple[tuple[str, str], ...] = ()

    def with_extra(
        self,
        *,
        false_free: tuple[str, ...] = (),
        quota_exhausted: tuple[str, ...] = (),
        free_tier_zero_allocation: tuple[str, ...] = (),
        model_not_found: tuple[str, ...] = (),
        model_not_entitled: tuple[str, ...] = (),
        upstream_rate_limit: tuple[str, ...] = (),
        upstream_rate_limit_error_codes: tuple[tuple[str, str], ...] = (),
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
            model_not_entitled=self.model_not_entitled + model_not_entitled,
            upstream_rate_limit=self.upstream_rate_limit + upstream_rate_limit,
            upstream_rate_limit_error_codes=(
                self.upstream_rate_limit_error_codes + upstream_rate_limit_error_codes
            ),
        )


DEFAULT_FAILURE_MARKERS = FailureMarkers()

# The provider rejected the REQUEST we sent, not the account: a 400/422 is by definition about this
# body. Shared with `capability_discovery`, which reads the same statuses as a per-profile capability
# verdict — one name, one meaning, and the two paths' differing consequences stay deliberate.
REQUEST_REJECTION_STATUSES = frozenset({400, 422})

# Failures the MODEL is not answerable for — most of them the caller's doing, two of them
# Ficelle's own. Two consequences, both flowing from that single fact: `cooldown_policy_for_reason`
# withholds the cooldown, and the consecutive-failure streak skips them.
# The streak's penalty is cumulative (12 points each, `model_scoring`), so without the exemption a
# client looping on a too-small max_tokens would progressively demote every candidate it touches,
# reordering a whole profile's pool over a limit that says nothing about the models. They are still
# counted and shown — a model that only ever truncates must stay visible rather than keep a clean
# record.
NO_MODEL_FAULT_FAILURE_REASONS = frozenset(
    {
        "truncated_before_content",
        # The upstream refused the request body itself. A malformed tool_call or an unsupported
        # field says nothing about the model's health, and every candidate would reject the same
        # payload — so no retry (see NON_RETRYABLE_FAILURE_REASONS), no cooldown, and the caller
        # gets the upstream's own 400.
        "bad_upstream_request",
        # This model's request contract differs from another candidate's: it either wants a private
        # field the OpenAI schema has no room for, or disables an otherwise valid option. Neither
        # case says the model is unhealthy, so this set carries the right consequences: no cooldown
        # and no scoring streak.
        "bad_upstream_contract",
        # Writing to the caller's socket failed: the client hung up mid-stream. The upstream was
        # answering fine; cooling it would punish a healthy model for a client-side abort.
        "client_disconnected",
        # Ficelle's own wall-clock request budget ran out (L2-R2, and earlier attempts may have
        # spent most of it). The candidate proved nothing wrong — often it was never even asked —
        # so blaming it would demote a healthy pool on every slow request.
        # `service_restarting` is deliberately absent: it is recorded as `attempted: False` and the
        # loop breaks immediately, so it reaches no state writer at all and needs no exemption. Its
        # consequences live in `TERMINAL_ENDINGS` below.
        "request_deadline_exceeded",
        # The request does not fit this candidate's context window. Every candidate with the same
        # (or smaller) context would reject the identical body, which says nothing about this
        # model's health — so no cooldown and no scoring streak. It stays retryable (see
        # CONTEXT_DIVERTING_FAILURE_REASONS) because a larger-context candidate can still answer.
        "context_length_exceeded",
    }
)


@dataclass(frozen=True)
class TerminalEnding:
    """What an ending Ficelle itself produced means for the answer and for the route log.

    `http_status` `None` means the ending does not decide the status — the attempted failures do.
    `route_log_status` `None` means the Requests page shows the status that was sent; a value
    there is a status the client never received and that exists only on the row.
    """

    http_status: int | None = None
    error_type: str | None = None
    writes_body: bool = True
    route_log_status: int | None = None


# The three endings the attempt loop can decide by itself, and everything that follows from each.
# One table because the same fact used to be re-derived by four separate string comparisons — the
# HTTP status, the payload `type`, the route-log reason and "is a body written at all" — which is
# how a run the route log called `client_disconnected` still carried a `final_status` of 502 that
# was never sent to anyone.
TERMINAL_ENDINGS: dict[str, TerminalEnding] = {
    # The caller hung up. Nothing is written back (there is no socket left to write to), so the
    # route log carries 499 — nginx's "client closed request" — rather than a status that was
    # never on the wire.
    "client_disconnected": TerminalEnding(
        error_type="client_disconnected",
        writes_body=False,
        route_log_status=499,
    ),
    # Ficelle is draining a SIGTERM: the caller is still there and gets a 503 telling it to retry,
    # never a verdict on the pool. It does not outrank a run every attempt of which said the
    # request body itself was invalid — that 400 is the truthful answer whatever ended the run.
    "service_restarting": TerminalEnding(http_status=503, error_type="service_restarting"),
    # Ficelle's own request budget expired. Deliberately no status of its own: the attempted
    # failures are the real cause, and answering 504 over a run that died on provider 500s would
    # hide it.
    "request_deadline_exceeded": TerminalEnding(),
}

# Failures no other candidate can do better on, because the candidate was never the problem. Trying
# the next one replays the same rejection and burns a healthy model's turn. Distinct from
# NO_MODEL_FAULT_FAILURE_REASONS, which is about *state writes*: a truncated response is no model's
# fault either, yet a model with a larger budget may well answer, so it keeps its failover.
#
# `bad_upstream_contract` is deliberately absent: a candidate-specific rejection is not a verdict
# on the body across the pool. The evidence includes a private reasoning field being required or
# forbidden, and Mistral disabling `top_k` on selected models while another candidate can accept
# the same request.
NON_RETRYABLE_FAILURE_REASONS = frozenset({"bad_upstream_request"})

# ...but staying retryable does not make the *next* candidate a free choice. A contract rejection is
# the provider's request validation answering, not the model: its siblings sit behind the same
# endpoint and the same checks, so the attempt after one of these buys far more from another source.
# Live evidence, run 20260810T221353Z: `top_k sampling is not enabled for this model` cost six
# Mistral models in a row on one request before any other provider was tried, and the caller got a
# 502 with 56 candidates in the pool. Read only by the attempt loop, and only for *this* request:
# nothing is cooled, nothing is scored, and a pool with no other source left still spends its window
# here — some providers do disable an option on selected models only.
#
# One member, and the only reason that has to be *listed* at all. Every other fact that takes the
# window off part of the pool writes state saying so — a provider cooldown naming its source, a quota
# cooldown naming its pool at the scope that provider declares — and the attempt loop diverts on what
# the write reports (`AppliedCooldown`), so it needs no list to stay in step with the cooldown policy.
# A contract rejection writes nothing at all: no cooldown, no scoring streak, nothing for a later
# request to read. It is a verdict on the source held for the length of one request, which is exactly
# why it lives here as a reason rather than as a key.
SOURCE_DIVERTING_FAILURE_REASONS = frozenset({"bad_upstream_contract"})

# Mirrors SOURCE_DIVERTING_FAILURE_REASONS for the other fact that invalidates part of the
# remaining attempt window without cooling anything: the request's own size, not the source, rules
# candidates out. The attempt loop reads this to re-plan the window against every candidate whose
# known `context_length` is at least as large as the (possibly upstream-corrected) request size,
# rather than against a source name.
CONTEXT_DIVERTING_FAILURE_REASONS = frozenset({"context_length_exceeded"})

# Attempt reasons that judge the MODEL on the profile it was asked to serve, rather than the
# provider or the transport: the upstream answered, and what came back was unusable as an answer.
# A real request ending this way writes `failed` capability evidence (`record_production_profile_failure`),
# which closes a gated profile's route gate for that model — so a reason the model is not answerable
# for must never be in here: a client looping on `max_tokens=20` would otherwise close the gate on
# every candidate it touched. Written as a subtraction rather than by hand so that stays true when
# either set moves.
PRODUCTION_PROFILE_FAILURE_REASONS = (
    frozenset({"empty_assistant_message", "upstream_finish_error", "runaway_output"})
    - NO_MODEL_FAULT_FAILURE_REASONS
)
assert PRODUCTION_PROFILE_FAILURE_REASONS, "no model-answerable reason left to write production evidence from"

# Fields providers disagree on because the OpenAI chat-completions schema does not define them. One
# upstream may demand a field while another rejects the same field as extra; neither verdict applies
# to every candidate in a heterogeneous pool. `reasoning_replay` exists because of exactly that gap.
# Reserved for the `assistant` role: a replayed reasoning trace only ever rides on assistant turns.
NON_STANDARD_REQUEST_FIELDS = ("reasoning_content", "reasoning_details", "thinking_blocks")

# Client-injected metadata Hermes stamps on messages of ANY role, not just `assistant` — unlike
# NON_STANDARD_REQUEST_FIELDS above, which a rejection only excuses on that one role. Most providers
# ignore it; Mistral and Groq reject it outright (Mistral structurally, on a `user` message; Groq via
# the prose matched by `_rejects_unknown_message_property`). Kept to the exact field observed so an
# unrelated `extra_forbidden` on, say, `content` still reads as a genuinely malformed body.
CLIENT_MESSAGE_METADATA_FIELDS = ("message_id",)

# ...but naming the field is not enough: `reasoning_content must be a string` is a verdict on a value
# that WAS sent, and the next candidate would reject it just the same. The retryable reading needs
# the upstream to be asking for something it did not get, so a requirement marker has to appear too.
# Fails closed, like the upstream-rate-limit narrowing: an unrecognised phrasing keeps the terminal
# `bad_upstream_request` reading rather than earning a failover it may not deserve.
MISSING_FIELD_MARKERS = (
    "passed back",
    "must be provided",
    "is required",
    "are required",
    "required field",
    "missing",
    "not provided",
    "must be present",
    "must be included",
)

# ...in the same breath. A body can be wrong in two ways at once — `reasoning_content must be a
# string; required field: messages` names our field and demands a different one — and proximity
# cannot tell the two apart, because the decoy sits *closer* to the field name than the real
# message's own marker does. What separates them is the sentence: the genuine rejection makes one
# statement, the decoy makes two. Splitting on ordinary clause punctuation (which JSON field
# separators also fall under) keeps each statement's field and demand together.
_CLAUSE_SPLIT_PATTERN = re.compile(r"[;.,\n]")


def _demands_missing_field(lower: str) -> bool:
    """True when one clause both names a non-standard field and asks for it."""
    for clause in _CLAUSE_SPLIT_PATTERN.split(lower):
        if any(field in clause for field in NON_STANDARD_REQUEST_FIELDS) and any(
            marker in clause for marker in MISSING_FIELD_MARKERS
        ):
            return True
    return False


def _forbids_nonstandard_message_field(text: str) -> bool:
    """True for a structured schema rejection of known provider-private message metadata.

    Pydantic/FastAPI identifies an unknown field with ``type: extra_forbidden`` and a typed
    ``loc`` path. Both are required: matching prose would let a caller merely mention these words
    and turn a genuinely malformed body into a failover. A wrong value type stays terminal too.

    Two families of field are recognised, deliberately not merged into one: `NON_STANDARD_REQUEST_FIELDS`
    only excuses a rejection on the `assistant` role, while `CLIENT_MESSAGE_METADATA_FIELDS` is
    tolerated on any role's message — including a `loc` that omits the role segment entirely
    (``["messages", <int>, "message_id"]``, where `location[-2]` is the message index, not a role).
    Naming an unlisted field, on any role, keeps the terminal reading — this must never widen into
    "any unknown key is a contract mismatch".
    """
    if len(text) > ERROR_BODY_PARSE_LIMIT:
        return False
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError):
        return False
    details = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(details, list):
        return False
    for detail in details:
        if not isinstance(detail, dict) or detail.get("type") != "extra_forbidden":
            continue
        location = detail.get("loc")
        if not isinstance(location, list) or len(location) < 4:
            continue
        field, role = location[-1], location[-2]
        is_assistant_only_field = role == "assistant" and field in NON_STANDARD_REQUEST_FIELDS
        is_any_role_metadata_field = field in CLIENT_MESSAGE_METADATA_FIELDS
        if not (is_assistant_only_field or is_any_role_metadata_field):
            continue
        try:
            messages_index = location.index("messages")
        except ValueError:
            continue
        # The role-bearing shape is [..., <int>, <role>, field]: the int sits before location[-2].
        # The roleless metadata shape is [..., <int>, field]: location[-2] IS that int, so the
        # slice has to reach one place further to still see it.
        end = -2 if is_assistant_only_field else -1
        if any(type(part) is int for part in location[messages_index + 1 : end]):
            return True
    return False


# A request option can be valid in the provider's OpenAI-compatible API while one particular model
# disables it. That is a model contract mismatch, not a malformed body: a sibling candidate may
# support the same option and should get its turn. Keep this evidence deliberately narrow to the
# exact option and wording observed from Mistral; an unfamiliar 400 remains terminal rather than
# spending quota by replaying a genuinely invalid body.
MODEL_SPECIFIC_OPTION_REJECTION_MARKERS = (
    "not enabled for this model",
    "not supported by this model",
    "not supported for this model",
)
MODEL_SPECIFIC_REQUEST_OPTIONS = ("top_k",)

# Sampling knobs that shape *how* a model draws tokens, never *what* is asked of it. Only these
# may be dropped and retried when a provider rejects them by name: removing one changes the
# sampling distribution, not the request's meaning. Messages, tools, tool_choice, schemas,
# response_format, max tokens and every field carrying caller intent are deliberately absent —
# dropping any of those would silently answer a different question.
DROPPABLE_SAMPLING_PARAMETERS = (
    "top_k",
    "top_p",
    "min_p",
    "top_a",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "seed",
)


def _rejects_model_specific_request_option(lower: str) -> bool:
    return any(option in lower for option in MODEL_SPECIFIC_REQUEST_OPTIONS) and any(
        marker in lower for marker in MODEL_SPECIFIC_OPTION_REJECTION_MARKERS
    )


# A provider can validate message keys strictly (Groq: `property 'message_id' is unsupported`)
# while every sibling ignores the same extra field Hermes adds. That is a provider-strictness
# mismatch, not a malformed body, so it belongs with the other `bad_upstream_contract` readings
# rather than the terminal `bad_upstream_request`. Keep both the sanctioned Hermes field and its
# message path in the match: a caller typo in a top-level property remains a terminal bad request.
_UNKNOWN_MESSAGE_PROPERTY_PATTERN = re.compile(
    r"messages(?:\.\d+|\[\d+\])[^\n]{0,256}property 'message_id' is unsupported"
)


def _rejects_unknown_message_property(lower: str) -> bool:
    return bool(_UNKNOWN_MESSAGE_PROPERTY_PATTERN.search(lower))


def estimated_tokens_for_chars(char_count: Any) -> int:
    """The chars/4 token heuristic every size estimate in Ficelle shares (0 for nothing)."""
    try:
        chars = int(char_count)
    except (TypeError, ValueError):
        return 0
    if chars <= 0:
        return 0
    return max(1, math.ceil(chars / 4))


# Prose an upstream uses to reject a request because it does not fit the model's context window.
# Unlike `request_too_large` (a per-minute throughput limit that recharges on its own), a context
# rejection is permanent for THIS candidate — but not for the pool: a candidate with a larger
# context window can still answer the same request. Kept deliberately narrow (whole phrases, not
# bare "token" or "large") so an unrelated 400/422/413 keeps its existing classification.
CONTEXT_LENGTH_EXCEEDED_MARKERS = (
    "context length",
    "context window",
    "maximum context",
    "context_length_exceeded",
    "too many tokens",
    "exceeds the model's maximum",
    "prompt is too long",
    "input is too long",
)


def _mentions_context_length_exceeded(lower: str) -> bool:
    return any(marker in lower for marker in CONTEXT_LENGTH_EXCEEDED_MARKERS)


# Every "N tokens" figure a rejection states. Providers order them both ways ("request is 266645
# tokens ... context length of 262144 tokens" vs "maximum context length is 8192 tokens. However,
# your messages resulted in 10000 tokens"), and in a context rejection the request's size is
# always the larger of the two, so the maximum is the request size whatever the phrasing.
_REJECTED_REQUEST_TOKENS_PATTERN = re.compile(r"(\d[\d,]*)\s*tokens?\b", re.IGNORECASE)


def rejected_request_tokens(text: str) -> int | None:
    """The request size an upstream's own context-length rejection states, if any.

    Lets the attempt loop learn the request's real size from the provider's own words instead of
    trusting only Ficelle's char/4 estimate, and narrow the remaining pool to candidates whose
    known context is large enough.
    """
    sizes: list[int] = []
    for figure in _REJECTED_REQUEST_TOKENS_PATTERN.findall(str(text or "")[:PROSE_SCAN_LIMIT]):
        try:
            sizes.append(int(figure.replace(",", "")))
        except ValueError:
            continue
    return max(sizes) if sizes else None


def rejected_sampling_parameters(text: str, body: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Sampling parameters this rejection names, and that the request actually carries.

    The provider must name the parameter AND reject it as unsupported — a message merely
    echoing a value is not a verdict. Intersecting with the request body keeps a rejection
    quoting an unrelated field from dropping something the caller never sent (L3-R4's rule:
    never silently weaken a request; here, never drop what was not refused)."""
    lower = str(text or "")[:PROSE_SCAN_LIMIT].lower()
    if not any(marker in lower for marker in MODEL_SPECIFIC_OPTION_REJECTION_MARKERS):
        return ()
    present = set(body or {})
    return tuple(
        parameter
        for parameter in DROPPABLE_SAMPLING_PARAMETERS
        if parameter in lower and (not body or parameter in present)
    )


# How much of an upstream's own error text survives into the failure we hand back. The default 180
# cut the sentence that named the problem — "The `reasoning_content` in the thinking mode must be
# passed back to the API" arrived as "... [invalid_request_error] The " — which cost a live
# reproduction to recover something the provider had already said. Relays nest their messages
# (gateway wrapper, then the real upstream's), so the useful half is at the end.
UPSTREAM_DETAIL_LIMIT = 400

# Provider-wide by construction, not by classification: a TLS handshake that fails describes the
# transport to the provider's host, never one model id, so `tls_error` joins the two reasons the
# body can state. It is raised by the invocation path (`evaluate_invocation_exception`), never by
# `classify_failure` — an upstream that answered HTTP already completed its handshake.
PROVIDER_SCOPED_COOLDOWN_REASONS = {"rate_limited", "auth_or_credit", "tls_error"}
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
                fallback_note=f"runtime reported that {MODEL_NOT_SERVEABLE_NOTE}",
            ),
            model_cooldown=False,
        )
    if reason in NO_MODEL_FAULT_FAILURE_REASONS:
        # Something other than the model produced the failure: a token budget that ran out before
        # any content, a request body the upstream refuses, a client that hung up mid-stream, or
        # Ficelle's own request deadline. Record it so the model stays scored and visible in the
        # admin, but never cool it — one client sending max_tokens=20 or a malformed tool_call would
        # otherwise empty a whole profile's pool for the cooldown window, which is exactly how a bad
        # client takes the router down.
        return CooldownPolicy(record_provider_error=False, model_cooldown=False)
    if reason == "quota_exhausted":
        return CooldownPolicy(
            record_provider_error=True,
            quota_cooldown=True,
            model_cooldown=False,
        )
    if reason in PROVIDER_SCOPED_COOLDOWN_REASONS:
        # Provider-wide failures use the provider cooldown as the SOLE selection-blocking
        # cooldown (L2-R4): the model attempt, error, and stats stay recorded for diagnosis,
        # but no model cooldown is written — a redundant one outlives provider recovery and
        # then reports two independent guards for one incident. A model-named upstream-pool
        # 429 classifies as `rate_limited_upstream` and stays model-scoped.
        return CooldownPolicy(
            record_provider_error=True,
            provider_cooldown=True,
            provider_cooldown_source=provider_source,
            model_cooldown=False,
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


def _body_names_our_model(lower_without_urls: str, upstream_model_id: str | None) -> bool:
    """Does the upstream's own text name the exact model we asked for?

    The evidence every model-scoped verdict rests on: a provider-wide failure — a dead key, an
    account out of credit — describes the account, never one id. Both callers are fail-closed on
    it, in opposite directions (an over-match leaves a dead provider uncooled on the 429 path, and
    quarantines a healthy model on the 403 path), so the rule lives here once rather than in two
    copies that can drift apart. The four-character floor keeps a short id from matching prose.
    """
    normalized = str(upstream_model_id or "").strip().lower()
    return len(normalized) >= 4 and normalized in lower_without_urls


def _matches_structured_error_code(
    error_codes: tuple[str, ...], configured_pairs: tuple[tuple[str, str], ...]
) -> bool:
    """Match an adapter's exact ``error.code``/``error.type`` verdict.

    ``error_object_codes`` keeps the two fields in a stable ``(code, type)`` order. Requiring both
    values, in their respective positions, prevents an unrelated 429/account or quota payload from
    inheriting a provider-specific model-pool classification merely because one word overlaps.
    """
    if len(error_codes) < 2:
        return False
    return error_codes[:2] in configured_pairs


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
    # This structural verdict has to precede every prose scan below: ``detail[].input`` repeats the
    # rejected reasoning trace, whose ordinary content may itself say "payment required" or "quota
    # exceeded". Those are caller/model words, not the provider's diagnosis. A contradictory named
    # status remains stronger; only an absent or request-rejection code lets the typed location win.
    resolved_error_codes = error_codes or provider_error_codes(text)
    named_status = status_for_error_codes(*resolved_error_codes)
    access = normalized_free_access if isinstance(normalized_free_access, dict) else {}
    lower = text[:PROSE_SCAN_LIMIT].lower()
    # `insufficient_quota` is the provider's structured verdict, not prose that may echo a caller's
    # tool or field name. On a quota-free route it therefore means the declared quota pool is empty,
    # at whatever model/provider/account scope the provider configured. This also handles gateways
    # that wrap the named 429 inside another HTTP status without widening text-marker matching.
    if (
        "insufficient_quota" in resolved_error_codes
        and named_status == 429
        and access.get("eligible") is True
        and access.get("mode") == "quota_free"
    ):
        if any(marker in lower for marker in marker_set.free_tier_zero_allocation):
            return "no_free_quota"
        return "quota_exhausted"
    # A free-only virtual route can exhaust while the account key remains valid. The provider's
    # stable code is stronger than the otherwise generic HTTP 403 auth verdict, and creates a
    # recoverable quota cooldown without trusting mutable prose.
    if status_code == 403 and "free_quota_exhausted" in resolved_error_codes:
        return "quota_exhausted"
    if (
        status_code in REQUEST_REJECTION_STATUSES
        and (named_status is None or named_status in REQUEST_REJECTION_STATUSES)
        and _forbids_nonstandard_message_field(text)
    ):
        return "bad_upstream_contract"
    lower_without_urls = _URL_PATTERN.sub(" ", lower)
    has_quota_marker = any(marker in lower for marker in marker_set.quota_exhausted)
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
        names_account_scope = any(
            marker in lower_without_urls for marker in ACCOUNT_RATE_LIMIT_TEXT_MARKERS
        )
        if not names_account_scope and _matches_structured_error_code(
            resolved_error_codes,
            marker_set.upstream_rate_limit_error_codes,
        ):
            return "rate_limited_upstream"
        # A 429 that names the shared pool behind one model id is MODEL-scoped: the account is fine
        # and every sibling model keeps answering, so it must not cool the provider. Matched on the
        # URL-stripped text like the false-free markers, since these messages link to a settings page.
        # The marker alone is not enough: an account/provider-level message may also mention an
        # upstream limit. Fail closed to the provider-scoped policy unless the body names the exact
        # model Ficelle called.
        names_model = _body_names_our_model(lower_without_urls, upstream_model_id)
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
        # A provider can also answer 413 when the body itself is too big for the model's context
        # window, not its per-minute throughput — checked first so that reading stays available
        # for the pool-narrowing retry (see CONTEXT_DIVERTING_FAILURE_REASONS) rather than the
        # short, model-scoped `request_too_large` cooldown a TPM limit earns.
        if _mentions_context_length_exceeded(lower):
            return "context_length_exceeded"
        return "request_too_large"
    # The provider's own code fields skip both marker rules: they carry its verdict, not the caller's
    # vocabulary, so `status_for_error_codes` reads them directly — and its own precedence keeps a
    # canonical name (`type: invalid_request_error`) ahead of a payment word in a sibling field.
    # A provider that NAMES the error a request rejection is saying the message is about the body we
    # sent, whatever status it wrapped that in — a gateway relaying an upstream 400 under its own 502
    # is the common case. That is the same evidence as the status itself, so it narrows the markers
    # too; without it, `{"code": "invalid_request_error"}` on a 502 still read a tool named `credits`
    # as a payment demand. See FALSE_FREE_PAYMENT_DEMAND_MARKERS.
    quotes_caller_request = unknown_status or status_code in {400, 422} or named_status in {400, 422}
    false_free = marker_set.false_free_payment_demand if quotes_caller_request else marker_set.false_free
    if status_code == 402 or named_status == 402 or false_free_pattern(false_free).search(lower_without_urls):
        return "billing_or_paid"
    if status_code == 403:
        # A 403 that names the model Ficelle asked for AND says it has to be switched on is a
        # per-MODEL entitlement, not a rejected key: every sibling keeps answering, so cooling the
        # provider benches a working account. Observed on Mistral, whose Labs tier 403s three ids
        # while `/v1/models` and `mistral-large-latest` answer 200 on the same key — each probe
        # cooled all of Mistral for an hour, and the probe came back an hour later.
        #
        # Fail-closed on three conditions, the same shape the 429 branch uses, and each one earns
        # its place: a dead key returns 403 too and the marker alone would read it as one bad model
        # (a dead key never names a model), while a region block manages both — it names the model
        # *and* reads like an entitlement, which is how an earlier two-condition version let one
        # through. That branch's account-scope constant is NOT reused: its markers include
        # "organization", which this very message carries while meaning the opposite ("enable them
        # in your organization settings" locates the switch, not the fault).
        names_model = _body_names_our_model(lower_without_urls, upstream_model_id)
        names_account_scope = any(marker in lower_without_urls for marker in ACCOUNT_SCOPE_403_MARKERS)
        if (
            names_model
            and not names_account_scope
            and any(marker in lower_without_urls for marker in marker_set.model_not_entitled)
        ):
            # `model_not_found` is the policy for "this id is not serveable to us", which is what
            # this is — it quarantines rather than cooling, so the hourly re-probe stops too.
            return "model_not_found"
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
        # Unless the rejection exposes this candidate's narrower contract: it requires private
        # assistant metadata, or it disables an option other candidates accept. The structured
        # mirror image — forbidding that metadata — returned before the prose scans above so the
        # rejected field's echoed value could not masquerade as a billing or quota verdict.
        if (
            _demands_missing_field(lower)
            or _rejects_model_specific_request_option(lower)
            or _rejects_unknown_message_property(lower)
        ):
            return "bad_upstream_contract"
        # Last of the specific readings: the body itself would fit another candidate's larger
        # context window, so it is not the generic, non-retryable `bad_upstream_request`.
        if _mentions_context_length_exceeded(lower):
            return "context_length_exceeded"
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


def model_not_found_body(safe_id: str) -> dict[str, Any]:
    """404 body for a model id this Ficelle does not serve, whichever route was asked.

    `GET /v1/models/{id}` and a chat completion naming an unknown model state the same fact, so
    they state it with the same words: a client that reads one message and then the other must not
    have to work out that they mean the same thing. Takes an already-redacted id — the caller owns
    the sanitization, since the chat path has one on hand and the lookup route builds one.
    """
    return {"error": {"message": f"model {safe_id} is not served by this Ficelle", "type": "model_not_found"}}


def upstream_failure_actions(reason_counts: dict[str, int]) -> list[str]:
    actions: list[str] = []
    if reason_counts.get("auth_or_credit"):
        actions.append("Check provider credentials/credits, then clear the provider cooldown after fixing it. Ficelle cooled that provider and sent the rest of this request's attempts to another one when the pool had another to offer, so a single failed attempt here is expected rather than a truncated failover.")
    if reason_counts.get("rate_limited"):
        actions.append("Wait for the provider cooldown or switch this virtual model to another healthy provider. Ficelle cooled that provider and sent the rest of this request's attempts to another one when the pool had another to offer, so a single failed attempt here is expected rather than a truncated failover.")
    if reason_counts.get("rate_limited_upstream"):
        actions.append("The shared upstream pool behind this model id is saturated, not your account: the model is cooled briefly and its provider keeps serving. Nothing to fix; add your own upstream key if you need dedicated limits.")
    if reason_counts.get("billing_or_paid"):
        actions.append("Inspect the anti false-free guard result; refresh the catalog before re-enabling the model.")
    if reason_counts.get("no_free_quota"):
        actions.append("Provider reported a zero free-tier allocation (limit: 0) for this model; it is quarantined and will stop being proposed. Use a free-tier model or upgrade the account.")
    if reason_counts.get("model_not_found"):
        actions.append(
            f"Not serveable: {MODEL_NOT_SERVEABLE_NOTE}. It is quarantined and will stop being "
            "benchmarked. Clear the quarantine to retry once the provider deploys it, or once an "
            "admin enables the tier."
        )
    if reason_counts.get("server_error"):
        actions.append("Retry after the short model cooldown or quarantine the unstable upstream.")
    if reason_counts.get("timeout"):
        actions.append("Ficelle timed out a slow upstream and tried the next candidate; reduce this virtual model's timeout or quarantine repeat offenders.")
    if reason_counts.get("bad_upstream_request"):
        actions.append("The upstream rejected the request body itself (HTTP 400/422) — inspect the payload the client sent, typically a malformed tool_call or an unsupported field. No model was cooled and no other candidate was tried: every one of them would reject the same body.")
    if reason_counts.get("bad_upstream_contract"):
        actions.append("One upstream rejected a request option or provider-private field that another candidate may handle. Ficelle kept the request unchanged and preferred a different provider for the next attempt when one was available. No model was cooled.")
    if reason_counts.get("context_length_exceeded"):
        actions.append("The request does not fit this candidate's context window. Ficelle raises the required context from the upstream's own stated size and retries on a larger-context candidate automatically; no model was cooled. If every attempt shows this reason, no candidate in the pool has a large enough context for this request — add one or shorten the request.")
    if reason_counts.get("client_disconnected"):
        actions.append("The client closed the connection while the answer was streaming; the upstream was healthy and was not cooled. Look at the client's timeout or cancel behaviour, not at the model.")
    if reason_counts.get("service_restarting"):
        actions.append("Ficelle was stopping (service restart or update) and did not start further fallback attempts for this request. No model was cooled: nothing upstream failed. Retry once the service is back up.")
    if reason_counts.get("truncated_before_content"):
        actions.append("The completion token budget ran out before the model emitted any content; reasoning models spend it on reasoning tokens first. Raise max_tokens on the request. No model was cooled: this is a request-side limit, not an upstream failure.")
    if reason_counts.get("empty_assistant_message") or reason_counts.get("invalid_success_json"):
        actions.append("Inspect route logs and verified capabilities; this upstream returned HTTP 200 without a usable assistant message.")
    if reason_counts.get("runaway_output"):
        actions.append("The model generated past the profile's completion character budget, so Ficelle stopped the answer and cooled the model. A buffered response can fail over to another candidate; an already-committed stream ends with a terminal error because fallback is no longer safe. If this is a legitimate answer shape for the profile, raise its entry in max_completion_chars_by_profile.")
    if not actions:
        actions.append("Inspect ~/.ficelle/logs/routes.jsonl with the request_id, then clear cooldowns only after the upstream issue is understood.")
    return actions


# Capacity dimensions a 413/429 body may name (L4-R1). Read for reporting only — they never
# change the failure classification or its scope; unknown limits stay unknown.
CAPACITY_DIMENSION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tpm", ("tokens per minute", "tpm")),
    ("rpm", ("requests per minute", "rpm")),
    ("tpd", ("tokens per day", "tpd")),
    ("rpd", ("requests per day", "rpd", "daily request")),
    ("concurrent", ("concurrent request", "concurrent session", "simultaneous request")),
    ("free_quota", ("free quota", "free tier", "free-tier", "trial quota")),
    ("context_or_body", ("context length", "maximum context", "request entity too large", "request too large", "body too large")),
)


def capacity_dimension(status_code: int | None, text: str) -> str | None:
    """The real dimension of a provider capacity rejection, or None for non-capacity errors.

    `capacity.unknown` covers a 413/429 whose body names no dimension (the seventh baseline
    Groq case): capacity evidence without an invented unit."""
    if status_code not in {413, 429}:
        return None
    lower = _URL_PATTERN.sub(" ", str(text or "")).lower()
    for dimension, markers in CAPACITY_DIMENSION_MARKERS:
        if any(marker in lower for marker in markers):
            return f"capacity.{dimension}"
    return "capacity.unknown"


def every_attempt_failed_with(errors: list[dict[str, Any]], reason: str) -> bool:
    """True when the run had attempts and every one of them ended with `reason`.

    The verdicts below turn a whole-run failure into a precise client status (400/422) instead of
    a 502 that reads as "Ficelle is broken" when the request itself is the problem.
    """
    return bool(errors) and all(row.get("reason") == reason for row in errors)


def caller_rejected_request(errors: list[dict[str, Any]]) -> bool:
    """Every attempt: the upstream refused the request body itself."""
    return every_attempt_failed_with(errors, "bad_upstream_request")


def request_feature_incompatible(errors: list[dict[str, Any]]) -> bool:
    """Every candidate was excluded for an unsupported tool-schema feature (L3-R4), so nothing
    was sent upstream and the 422 names which request feature to change."""
    return every_attempt_failed_with(errors, "unsupported_tool_schema")


def request_exceeds_context(errors: list[dict[str, Any]]) -> bool:
    """Every attempt: the request does not fit the candidate's context window, whether the
    static pre-filter excluded it or the upstream rejected it live."""
    return every_attempt_failed_with(errors, "context_length_exceeded")


def upstream_failure_status(errors: list[dict[str, Any]], *, terminal_reason: str | None = None) -> int:
    """HTTP status for a run where no candidate delivered.

    Paired with `build_upstream_failure_error`, which reads the same verdict off the same inputs:
    every caller of one must use the other, or the status and the payload's `type` disagree.

    `terminal_reason` is why the loop stopped, carried explicitly rather than re-inferred from the
    error list: the row that records an abandonment has `attempted: False`, so the scan below never
    sees it and used to answer 502 `upstream_failure` for a run the route log called
    `service_restarting`. What each ending means is read off `TERMINAL_ENDINGS`, and only after the
    two verdicts about the request itself: a run every attempt of which said the body was invalid
    answers 400 whether or not a SIGTERM happened to cut it short — the restart changed when the
    run stopped, not what was wrong with the request.
    """
    # Both verdicts are about the request the caller sent, so they read the rows where it was
    # actually sent. The abandonment row is `attempted: False`, and counting it would let a SIGTERM
    # turn a run every attempt of which rejected the body into a 503 that says nothing true.
    attempted_errors = [error for error in errors if error.get("attempted") is not False]
    if request_feature_incompatible(attempted_errors):
        return 422
    if caller_rejected_request(attempted_errors) or request_exceeds_context(attempted_errors):
        return 400
    ending = TERMINAL_ENDINGS.get(terminal_reason or "")
    if ending is not None and ending.http_status is not None:
        return ending.http_status
    reasons = {str(error.get("reason") or "") for error in attempted_errors}
    if reasons and reasons <= {"rate_limited", "rate_limited_upstream", "quota_exhausted"}:
        return 429
    if reasons and reasons <= {"timeout", "request_deadline_exceeded"}:
        return 504
    return 502


def route_log_failure_status(errors: list[dict[str, Any]], *, terminal_reason: str | None = None) -> int:
    """The status the Requests page shows for a run where no candidate delivered.

    The same status the client got, except for an ending that writes no body at all: there the row
    carries the ending's own `route_log_status` (499 for a caller that hung up) instead of a 502 or
    504 nobody ever received.
    """
    ending = TERMINAL_ENDINGS.get(terminal_reason or "")
    if ending is not None and ending.route_log_status is not None:
        return ending.route_log_status
    return upstream_failure_status(errors, terminal_reason=terminal_reason)


def upstream_retry_after_seconds(errors: list[dict[str, Any]], status: int) -> int | None:
    """Return a truthful client retry delay for a homogeneous capacity failure."""
    if status != 429:
        return None
    delays = [
        int(error["retry_after_seconds"])
        for error in errors
        if error.get("attempted") is not False
        and isinstance(error.get("retry_after_seconds"), (int, float))
        and not isinstance(error.get("retry_after_seconds"), bool)
        and float(error["retry_after_seconds"]) > 0
    ]
    return min(delays) if delays else None


def build_upstream_failure_error(
    requested_model: str,
    request_id: str,
    candidate_count: int,
    attempts: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    terminal_reason: str | None = None,
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
        detail = sanitize_error_detail(row.get("detail") or row.get("error"), UPSTREAM_DETAIL_LIMIT)
        if detail:
            safe_row["detail"] = detail
        if "stream_started" in row:
            safe_row["stream_started"] = bool(row.get("stream_started"))
        safe_details.append({key: value for key, value in safe_row.items() if value is not None})
    reason_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items())) or "unknown"
    safe_requested_model = sanitize_error_detail(requested_model, 250) or "[redacted]"
    # Same predicates, same order and the same attempted-only rows as `upstream_failure_status`:
    # the status and this payload's `type` are one verdict written twice, and they must not drift.
    attempted_errors = [error for error in errors if error.get("attempted") is not False]
    # The row a rejection message quotes is the last one that was actually sent: an abandonment row
    # is last in the list but carries no upstream words, so quoting it would drop the provider
    # sentence naming the malformed field.
    attempted_details = [
        detail for row, detail in zip(errors, safe_details) if row.get("attempted") is not False
    ]
    ending = TERMINAL_ENDINGS.get(terminal_reason or "")
    if request_feature_incompatible(attempted_errors):
        feature_detail = attempted_details[-1].get("detail") if attempted_details else ""
        message = "the requested model does not support a tool-schema feature in this request" + (
            f": {feature_detail}" if feature_detail else ""
        )
        error_type = "invalid_request_error"
    elif caller_rejected_request(attempted_errors):
        upstream_detail = attempted_details[-1].get("detail") if attempted_details else ""
        message = "upstream rejected this request as invalid" + (f": {upstream_detail}" if upstream_detail else "")
        error_type = "invalid_request_error"
    elif request_exceeds_context(attempted_errors):
        context_detail = attempted_details[-1].get("detail") if attempted_details else ""
        message = "this request does not fit any available model's context window" + (
            f": {context_detail}" if context_detail else ""
        )
        error_type = "invalid_request_error"
    elif ending is not None and ending.error_type is not None:
        # The run was cut short by an ending Ficelle owns, so the payload must not read as a
        # verdict on the pool.
        message = (
            f"Ficelle stopped taking new fallback attempts for {safe_requested_model} "
            f"({ending.error_type}; {len(attempts)}/{candidate_count} attempted; {reason_summary})"
        )
        error_type = ending.error_type
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
