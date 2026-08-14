from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

try:
    from ficelle_pro.compression import (
        DEFAULT_COMPRESSION_CONFIG,
        compression_marker,
        compress_block,
        normalize_compression_settings,
        plan_chat_compression,
        put_original,
    )
except ImportError:  # pragma: no cover - exercised only in core-only installs
    # Native compression is a closed Pro engine. When it is absent the free tier
    # runs with compression mode "off": prepare_compression_route_body short-circuits
    # before it would ever call the primitives below, so they are safe as None.
    # See docs/prds/open-core-extraction-prd.md (Lot 1).
    from ficelle.compression_fallback import (
        DEFAULT_COMPRESSION_CONFIG,
        normalize_compression_settings,
    )

    compression_marker = None
    compress_block = None
    plan_chat_compression = None
    put_original = None
from ficelle.failures import (
    rejected_sampling_parameters,
    CALLER_CAUSED_FAILURE_REASONS,
    NON_RETRYABLE_FAILURE_REASONS,
    SOURCE_DIVERTING_FAILURE_REASONS,
    UPSTREAM_DETAIL_LIMIT,
    caller_rejected_request,
    status_for_error_codes,
    upstream_failure_status,
)
from ficelle.domain_models import SelectionResult
from ficelle.use_cases.cooldowns import AppliedCooldown
from ficelle.redaction import redact_sensitive_json, sanitize_error_detail
from ficelle.use_cases.benchmark import finish_reason_is_truncation


DEFAULT_CHAT_COMPLETION_MODEL = "ficelle/auto-tools"


@dataclass(frozen=True)
class ChatCompletionRequest:
    requested_model: str
    safe_requested_model: str


@dataclass(frozen=True)
class ChatCompletionResponse:
    status: int
    payload: dict[str, Any]
    headers: dict[str, str]


@dataclass(frozen=True)
class ChatCompletionRawResponse:
    status: int
    content_type: str
    headers: dict[str, str]
    content: bytes


@dataclass(frozen=True)
class ChatCompletionStreamStart:
    status: int
    content_type: str
    headers: dict[str, str]


@dataclass(frozen=True)
class ChatCompletionStart:
    request: ChatCompletionRequest
    catalog: dict[str, Any]
    candidates: list[dict[str, Any]]
    is_fusion_request: bool
    response: ChatCompletionResponse | None = None


@dataclass(frozen=True)
class ChatCompletionAttemptPlan:
    candidates: list[dict[str, Any]]
    candidate_count: int
    routed_body: dict[str, Any]
    compression_metadata: dict[str, Any] | None
    requested_model_is_virtual: bool
    # The scored pool `candidates` was drawn from, kept so the attempt loop can re-plan what is left
    # of the window once a failure reason rules a whole source out. Empty means "the window is all
    # that is known", which is what a hand-built plan gets.
    candidate_pool: list[dict[str, Any]] = field(default_factory=list)
    # Error rows for candidates ruled out before the window was cut (L3-R4), seeded into the
    # attempt loop's `errors`. Carried only when the filter left nothing to try: they are then the
    # whole reason the request fails, and the failure response reads them to answer with the
    # precise 422 naming the feature. An exclusion a run could route around costs that run nothing
    # and is not one of its failure reasons.
    excluded_errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ChatCompletionLastRouteRecord:
    safe_requested_model: str
    status: str
    reason: str
    request_id: str
    candidate_count: int
    attempt_count: int
    duration_seconds: float
    selected_model: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] | None = None
    competence: str | None = None
    compression: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatCompletionRouteTelemetry:
    last_route: ChatCompletionLastRouteRecord
    route_log: dict[str, Any]


@dataclass(frozen=True)
class ChatCompletionAttemptRunResult:
    outcome: Literal["non_streaming_success", "streaming_complete", "json_failure"]
    raw_response: ChatCompletionRawResponse | None = None
    json_response: ChatCompletionResponse | None = None


@dataclass(frozen=True)
class ChatCompletionHandleResult:
    start: ChatCompletionStart
    attempt_plan: ChatCompletionAttemptPlan | None = None
    attempt_result: ChatCompletionAttemptRunResult | None = None


@dataclass(frozen=True)
class CompressionRoutePlan:
    body: dict[str, Any]
    metadata: dict[str, Any] | None


CooldownStatus = int | str


@dataclass(frozen=True)
class NonStreamingAttemptDecision:
    outcome: Literal["success", "retryable_failure", "terminal_failure"]
    attempt_update: dict[str, Any]
    error: dict[str, Any] | None = None
    cooldown_reason: str | None = None
    cooldown_detail: str | None = None
    cooldown_status: CooldownStatus | None = None
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class StreamingAttemptDecision:
    outcome: Literal["success", "retryable_failure", "terminal_failure", "mid_stream_failure"]
    attempt_update: dict[str, Any]
    error: dict[str, Any] | None = None
    cooldown_reason: str | None = None
    cooldown_detail: str | None = None
    cooldown_status: CooldownStatus | None = None
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class UpstreamFailureDecision:
    outcome: Literal["retryable_failure", "terminal_failure"]
    attempt_update: dict[str, Any]
    error: dict[str, Any]
    cooldown_reason: str
    cooldown_detail: str
    cooldown_status: CooldownStatus


@dataclass(frozen=True)
class InvocationExceptionDecision:
    attempt_update: dict[str, Any]
    error: dict[str, Any]
    cooldown_reason: str
    cooldown_detail: str
    cooldown_status: CooldownStatus


AttemptCooldownDecision = (
    NonStreamingAttemptDecision | StreamingAttemptDecision | UpstreamFailureDecision | InvocationExceptionDecision
)


@dataclass(frozen=True)
class SuccessRouteLogInput:
    request_id: str
    safe_requested_model: str
    selected_model: dict[str, Any]
    competence: str
    final_status: int
    candidate_count: int
    attempt_count: int
    attempts: list[dict[str, Any]]
    duration_seconds: float
    stream: bool
    compression: dict[str, Any] | None = None
    stream_started: bool | None = None
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class FailureRouteLogInput:
    request_id: str
    safe_requested_model: str
    candidate_count: int
    attempt_count: int
    attempts: list[dict[str, Any]]
    duration_seconds: float
    stream: bool
    # The errors the run collected, so the log records the status and reason the caller actually
    # received. Hardcoding 502/`upstream_failure` made the Requests page contradict the response
    # whenever every attempt died on the request body.
    errors: list[dict[str, Any]]
    compression: dict[str, Any] | None = None


@dataclass(frozen=True)
class MidStreamFailureRouteLogInput:
    request_id: str
    safe_requested_model: str
    selected_model: dict[str, Any]
    final_status: int
    candidate_count: int
    attempt_count: int
    attempts: list[dict[str, Any]]
    duration_seconds: float
    # `client_disconnected` when the caller hung up, `mid_stream_failure` when the upstream broke.
    # Both end the request the same way, but only one of them is the model's fault — so this is
    # required rather than defaulted: a caller that forgets it would silently log a client abort
    # as the model's failure, which is the exact bug this field exists to prevent.
    reason: str
    compression: dict[str, Any] | None = None


@dataclass(frozen=True)
class UpstreamFailureResponseInput:
    requested_model: str
    safe_requested_model: str
    request_id: str
    candidate_count: int
    attempts: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    compression: dict[str, Any] | None = None


@dataclass(frozen=True)
class AttemptFailureRecordInput:
    attempt: dict[str, Any]
    attempt_update: dict[str, Any]
    attempts: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class AttemptResultRecordInput:
    attempt: dict[str, Any]
    attempt_update: dict[str, Any]
    attempts: list[dict[str, Any]]


@dataclass(frozen=True)
class AttemptCooldownRequest:
    model: dict[str, Any]
    reason: str
    detail: str | None
    status: CooldownStatus


@dataclass(frozen=True)
class NonStreamingSuccessResponseInput:
    request_id: str
    safe_requested_model: str
    selected_model: dict[str, Any]
    attempt_count: int
    response_status: int
    response_headers: Mapping[str, Any]
    response_content: bytes
    compression: dict[str, Any] | None = None
    # Sampling knobs removed from this request because the serving model rejected them by
    # name. Announced to the caller — a dropped parameter must never be silent.
    dropped_parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamingResponseStartInput:
    request_id: str
    safe_requested_model: str
    selected_model: dict[str, Any]
    attempt_count: int
    response_status: int
    response_headers: Mapping[str, Any]
    compression: dict[str, Any] | None = None
    dropped_parameters: tuple[str, ...] = ()


CatalogLoader = Callable[[dict[str, Any]], dict[str, Any]]
CandidateSelector = Callable[[str, dict[str, Any], dict[str, Any]], list[dict[str, Any]]]
# Returns the typed SelectionResult (candidates + per-model exclusion reasons) so a refusal
# can name what actually blocked the request instead of one generic message (L1-R4).
ResultSelector = Callable[[str, dict[str, Any], dict[str, Any]], "SelectionResult"]

# Error types Ficelle emits for its own selection refusals, before any upstream attempt.
# Exported so the synthetic-health harness recognizes them structurally instead of keeping
# a hand-synced copy. `invalid_request_error` is deliberately absent: the router also uses
# it when every upstream rejected the body, and that verdict belongs to the providers.
SELECTION_REFUSAL_ERROR_TYPES = frozenset({
    "no_available_model",
    "model_unavailable",
    "model_not_found",
})

# Safe response scope for each selection-exclusion reason. Anything unknown stays "model":
# the narrowest claim that is always true of a blocked concrete model. A shared-account
# quota block is reported as `quota_pool` here on purpose: the response should not reveal
# cross-source account topology to an arbitrary client — the harness's eligibility
# records, which are local evidence, do distinguish `shared_account`.
REFUSAL_SCOPE_BY_BLOCK_REASON = {
    "cooldown": "model",
    "quarantined": "model",
    "provider_cooldown": "provider",
    "quota_cooldown": "quota_pool",
    "not_invokable": "credential",
    "stale_catalog": "catalog",
    "profile_requirements": "profile",
    "competence_gate": "profile",
}
FusionPredicate = Callable[[str], bool]
VirtualModelPredicate = Callable[[str], bool]
ResponseHeadersBuilder = Callable[[str, str], dict[str, str]]
# (request_id, safe_requested_model, selected_model, attempt_count, compression, dropped_parameters)
SuccessResponseHeadersBuilder = Callable[..., dict[str, str]]
FailureResponseHeadersBuilder = Callable[[str, str, int, dict[str, Any] | None], dict[str, str]]
UpstreamFailureErrorBuilder = Callable[[str, str, int, list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
LastRouteRecorder = Callable[[str, str, str, str, int, int, float], None]
RouteLogWriter = Callable[[dict[str, Any]], None]
Clock = Callable[[], float]
MaxAttemptsCalculator = Callable[[str, dict[str, Any], int], int]
CompressionPlanner = Callable[[dict[str, Any], dict[str, Any]], CompressionRoutePlan]
CompressionBlockCompressor = Callable[[Any, dict[str, Any]], Any]
OriginalWriter = Callable[..., str]
SuccessErrorDetector = Callable[[Any, dict[str, Any] | None], tuple[str, str, int | None] | None]
DeliverablePredicate = Callable[[Any], bool]
FailureClassifier = Callable[[int, str, dict[str, Any] | None], str]
# (model, body, config, *, remaining_budget_seconds=None, deadline_monotonic=None) — the
# keyword budget arguments are how the request deadline constrains an in-flight attempt.
ModelInvoker = Callable[..., Any]
# Returns what the write blocked beyond this candidate, which is what the attempt loop diverts on.
# `None` stays accepted so a wiring that only needs the side effect keeps working — it then reports
# nothing, and the loop behaves as it did before any of this existed.
CooldownApplier = Callable[[AttemptCooldownRequest], "AppliedCooldown | None"]
# `quota_cooldown_matches_model`: does this key block that candidate? A quota cooldown is keyed at
# whichever scope the provider declares, so only this predicate can say — comparing sources would
# over-divert a `model:` key and under-divert a `shared_account:` one.
QuotaCooldownMatcher = Callable[[str, dict[str, Any]], bool]
SuccessRecorder = Callable[[dict[str, Any], float], None]
CompetenceResolver = Callable[[str, dict[str, Any]], str]
TelemetryRecorder = Callable[[ChatCompletionRouteTelemetry], None]
# (response, model, attempt_count, *, deadline_monotonic=None)
StreamResponseHandler = Callable[..., dict[str, Any]]
TimeoutDetector = Callable[[Exception], bool]
AttemptPortsFactory = Callable[[ChatCompletionAttemptPlan], "ChatCompletionAttemptPorts"]


@dataclass(frozen=True)
class ChatCompletionAttemptPorts:
    invoke_model: ModelInvoker
    apply_cooldown: CooldownApplier
    record_success: SuccessRecorder
    resolve_competence: CompetenceResolver
    record_telemetry: TelemetryRecorder
    # One state write for the success and its route telemetry. Kept alongside the two separate
    # ports because the failure paths still record telemetry on its own.
    record_success_with_telemetry: Callable[..., None]
    stream_response: StreamResponseHandler
    detect_success_error: SuccessErrorDetector
    has_deliverable: DeliverablePredicate
    classify_failure: FailureClassifier
    is_timeout_exception: TimeoutDetector
    build_success_headers: SuccessResponseHeadersBuilder
    build_failure_error: UpstreamFailureErrorBuilder
    build_failure_headers: FailureResponseHeadersBuilder
    # Recognizes Ficelle's own request-deadline expiry (L2-R2), which must be reported as
    # `request_deadline_exceeded`, distinct from a provider `timeout`. Optional so existing
    # wirings and fixtures keep their behavior.
    is_deadline_exception: TimeoutDetector | None = None
    # Learned sampling-parameter incompatibilities: read before sending (proactive), written
    # after a provider rejects a knob by name (reactive). Default to no-ops so a wiring that
    # does not persist them still gets the reactive drop-and-retry.
    # Reads a quota cooldown key at its own scope. Defaults to "matches nothing", so a wiring that
    # does not provide it simply never diverts on a quota key — the behaviour before this port.
    quota_cooldown_matches_model: QuotaCooldownMatcher = lambda _key, _model: False
    unsupported_parameters_for: Callable[[dict[str, Any], dict[str, Any]], tuple[str, ...]] = (
        lambda _model, _body: ()
    )
    learn_unsupported_parameters: Callable[[dict[str, Any], tuple[str, ...]], None] = (
        lambda _model, _parameters: None
    )


COMPRESSION_PENDING_MARKER = "<<ficelle:compressed:pending>>"


def detect_tool_schema_features(body: dict[str, Any]) -> set[str]:
    """The advanced JSON-Schema dialect features this request's tool schemas use (L3-R4).

    Read-only and conservative: only features a provider is known to reject are named, so a
    candidate is excluded for a feature it declared unsupported — never for schema style.
    """
    features: set[str] = set()
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    if isinstance(body.get("tool_choice"), dict):
        features.add("forced_tool_choice")

    def walk(schema: Any, *, depth: int) -> None:
        if not isinstance(schema, dict):
            return
        type_value = schema.get("type")
        if isinstance(type_value, list):
            features.add("nullable_type_array" if "null" in type_value else "type_array")
        if isinstance(schema.get("anyOf"), list):
            if any(isinstance(alt, dict) and alt.get("type") == "null" for alt in schema["anyOf"]):
                features.add("anyof_nullability")
        if schema.get("additionalProperties") is False:
            features.add("additional_properties_false")
        if isinstance(schema.get("enum"), list):
            features.add("enums")
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for child in properties.values():
            if isinstance(child, dict):
                if child.get("type") == "object" and depth >= 1:
                    features.add("nested_objects_arrays")
                walk(child, depth=depth + 1)
        items = schema.get("items")
        if isinstance(items, dict):
            if items.get("type") == "object":
                features.add("nested_objects_arrays")
            walk(items, depth=depth + 1)

    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        parameters = function.get("parameters") if isinstance(function, dict) else None
        walk(parameters, depth=0)
    return features


def declared_unsupported_schema_features(source: Any, config: dict[str, Any]) -> set[str]:
    """The tool-schema features a provider declares it cannot take (L3-R4).

    Declaration lives in provider config (curated pack or operator) under
    `unsupported_tool_schema_features`, and is made per provider rather than per model; absent
    declaration means nothing is excluded — incompatibility is asserted, never guessed."""
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    provider_cfg = providers.get(str(source or ""))
    declared = provider_cfg.get("unsupported_tool_schema_features") if isinstance(provider_cfg, dict) else None
    if not isinstance(declared, list) or not declared:
        return set()
    return {str(item) for item in declared}


def exclude_incompatible_candidates(
    body: dict[str, Any],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a scored pool into what this request's tool schemas can be sent to, and error rows
    for what they cannot (L3-R4).

    The declaration is per provider, so the verdict is identical for every candidate of a source
    and is a pure function of the request, the source and the config — knowable before the attempt
    window is even cut. Evaluated per candidate inside the attempt loop, it cost one window slot
    per model a declaring provider owns: a window of six mostly from one provider ended as one real
    attempt and a 502 with the rest of the pool untried. Nothing is sent upstream either way, so
    what filtering statically buys is a window that can actually be spent, not saved quota.

    The rows are returned rather than dropped: they are what still answers a request whose every
    candidate is incompatible with the precise 422 naming the feature, instead of a pool that
    silently reads as empty.
    """
    # The two halves are independent, and neither is per candidate: what a provider refuses is
    # keyed by source, what this request uses is keyed by nothing at all. So the config is read
    # once per distinct source, and the tool schemas are walked once for the whole pool — and not
    # at all when no source in it declares anything, which is the ordinary case.
    declared_by_source = {
        source: declared_unsupported_schema_features(source, config)
        for source in {str(candidate.get("source") or "") for candidate in candidates}
    }
    if not any(declared_by_source.values()):
        return list(candidates), []
    features = detect_tool_schema_features(body)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        incompatible = features & declared_by_source[str(candidate.get("source") or "")]
        if not incompatible:
            eligible.append(candidate)
            continue
        excluded.append({
            "model": candidate.get("id"),
            "upstream": candidate.get("upstream_id"),
            "source": candidate.get("source"),
            "reason": "unsupported_tool_schema",
            "detail": "unsupported tool-schema features: " + ", ".join(sorted(incompatible)),
        })
    return eligible, excluded


def drop_sampling_parameters(body: dict[str, Any], parameters: tuple[str, ...] | set[str]) -> dict[str, Any]:
    """A copy of the request without the named sampling knobs. Never mutates the caller's body."""
    dropped = set(parameters)
    return {key: value for key, value in body.items() if key not in dropped}


def normalize_chat_completion_request(body: Any) -> ChatCompletionRequest:
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    requested_model = str(body.get("model") or DEFAULT_CHAT_COMPLETION_MODEL)
    return ChatCompletionRequest(
        requested_model=requested_model,
        safe_requested_model=sanitize_error_detail(requested_model, 250) or "[redacted]",
    )


def malformed_tool_call_detail(body: Any) -> str | None:
    """Locate the first replayed tool call with no function name, or return None.

    The one request-body defect worth catching before the network. OpenAI requires
    ``tool_calls[].function.name``; a client that replays an empty one gets the upstream's own
    400 ("historical tool function name is invalid") — and since `bad_upstream_request` is
    non-retryable, that costs a full round trip to learn something knowable locally. On Fusion
    it costs one per panelist. Ficelle stays a passthrough: the body is never rewritten, only
    refused with a message naming the offending index. One deliberate exception exists, and it
    does not weaken the rule — ``reasoning_replay`` restores a thinking model's own
    ``reasoning_content`` on the way back out (see its module docstring for why the upstream
    demands it and the schema gives the client no way to return it). It adds a field Ficelle
    itself relayed outward; it never edits what the caller wrote, which is what this check
    refuses to do.

    Deliberately narrow, because the provider remains the authority on everything else in the
    body: only a ``function`` object that IS present with a blank or non-string ``name`` is
    rejected. A ``type: "custom"`` tool call carries no ``function`` at all and must still reach
    the provider, and anything that is not shaped like a tool call is left alone rather than
    guessed at.
    """
    if not isinstance(body, dict):
        return None
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                return f"messages[{message_index}].tool_calls[{call_index}].function.name is empty"
    return None


def request_deadline(config: dict[str, Any] | None, request_started: float) -> float | None:
    """Wall-clock instant past which no NEW attempt is started, or None when disabled.

    Without this the worst case was `max_attempts_per_request` × the profile timeout — four
    attempts at the 120 s default is over eight minutes with nothing to stop it. The budget is
    deliberately generous: it exists to bound a request that is failing, not to trim the tail of
    one that is working, and the default was sized against this install's own route log so that it
    cuts none of the successes actually observed.
    """
    raw = (config or {}).get("request_deadline_seconds")
    if raw is None:
        return None
    seconds = _safe_float(raw, 0.0)
    if seconds <= 0:
        return None
    return request_started + seconds


def attempts_with_source_diversity(
    candidates: list[dict[str, Any]],
    max_attempt_count: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Take the attempt window in score order, but not all from one provider.

    A provider-wide outage classifies correctly as a model-scoped `unavailable` per candidate —
    that scoping rule is deliberate and unchanged here. The side effect was that when the top of
    the pool belonged to one provider, all four attempts were spent inside the same outage while
    healthy candidates sat below the window, and the caller got a 502 with a working pool.

    Order is otherwise preserved: this only defers a candidate past its provider's quota, it never
    promotes a worse-scoring model ahead of a better one from a provider that still has room.
    """
    if max_attempt_count <= 0 or not candidates:
        return []
    raw_cap = (config or {}).get("max_attempts_per_source")
    cap = _safe_int(raw_cap, 0) if raw_cap is not None else 0
    if cap <= 0:
        # Default: leave at least one attempt for a different provider once more than one is in
        # play, and never constrain a single-provider pool.
        cap = max(1, max_attempt_count - 1)
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    for candidate in candidates:
        if len(selected) >= max_attempt_count:
            break
        source = str(candidate.get("source") or "")
        if used.get(source, 0) >= cap:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        used[source] = used.get(source, 0) + 1
    # A pool that is entirely one provider must still fill its window rather than answer with
    # fewer attempts than the operator configured.
    for candidate in deferred:
        if len(selected) >= max_attempt_count:
            break
        selected.append(candidate)
    return selected


def attempts_after_source_rejection(
    pool: list[dict[str, Any]],
    *,
    considered_ids: set[str],
    is_ruled_out: Callable[[dict[str, Any]], bool],
    remaining_attempts: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """What is left of the attempt window once part of the pool has stopped being worth it.

    The window is chosen before the first attempt, when no failure reason exists yet, and this
    request can then learn two kinds of fact that invalidate the rest of it: the provider's request
    validation rejected this body, so its other models validate it the same way; or a cooldown it
    just wrote now blocks candidates the window is still about to call. Re-planned from the whole
    pool rather than merely filtered, so the freed attempts go to the best remaining candidates
    instead of shrinking the window: the observed 502 had spent six of seven attempts inside one
    provider with 56 candidates available.

    `is_ruled_out` decides, rather than a set of source names, because the two facts do not block
    the same shape: a contract rejection and a provider cooldown block a source, while a quota
    cooldown blocks whichever scope its provider declares — one model, or an account several
    sources share.

    Score order and the per-source cap are whatever `attempts_with_source_diversity` makes of the
    survivors — this only removes candidates from its input, it never reorders. An empty result means
    the pool has nothing else to offer, and the caller keeps the window it already had: a
    single-provider pool spends every attempt the operator configured, here as anywhere else.
    """
    survivors = [
        candidate
        for candidate in pool
        if not is_ruled_out(candidate) and str(candidate.get("id") or "") not in considered_ids
    ]
    return attempts_with_source_diversity(survivors, remaining_attempts, config)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_state_key(value: Any) -> str:
    return sanitize_error_detail(value) or "[redacted]"


def _increment_count(counts: dict[str, int], key: Any) -> None:
    name = _safe_state_key(key)
    counts[name] = counts.get(name, 0) + 1


def live_zone_final_candidate(original: str, candidate: str) -> str:
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:24]
    return candidate.replace(COMPRESSION_PENDING_MARKER, compression_marker(digest))


def compression_gate_accepts(original_chars: int, compressed_chars: int, compression_config: dict[str, Any]) -> bool:
    if original_chars <= 0 or compressed_chars >= original_chars:
        return False
    saved_ratio = (original_chars - compressed_chars) / original_chars
    return saved_ratio >= _safe_float(
        compression_config.get("min_savings_ratio"), DEFAULT_COMPRESSION_CONFIG["min_savings_ratio"]
    )


def apply_live_zone_compression(
    body: dict[str, Any],
    original_body: dict[str, Any],
    block: Any,
    result: Any,
    final_candidate_text: str,
    compression_config: dict[str, Any],
    *,
    store_path: Any,
    write_original: OriginalWriter = put_original,
) -> dict[str, Any]:
    if body is original_body:
        body = copy.deepcopy(original_body)
    write_original(
        result.strategy,
        block.content,
        final_candidate_text,
        store_path=store_path,
        ttl_seconds=_safe_int(compression_config.get("store_ttl_seconds"), DEFAULT_COMPRESSION_CONFIG["store_ttl_seconds"]),
        max_entries=_safe_int(compression_config.get("store_max_entries"), DEFAULT_COMPRESSION_CONFIG["store_max_entries"]),
    )
    messages = body.get("messages")
    if isinstance(messages, list) and 0 <= block.message_index < len(messages) and isinstance(messages[block.message_index], dict):
        messages[block.message_index]["content"] = final_candidate_text
    return body


def prepare_compression_route_body(
    body: dict[str, Any],
    config: dict[str, Any],
    *,
    store_path: Any,
    compress: CompressionBlockCompressor = compress_block,
    write_original: OriginalWriter = put_original,
) -> CompressionRoutePlan:
    """Prepare compression metadata and the request body that should be routed upstream."""
    compression_config = normalize_compression_settings(config.get("compression"), strict=False)
    mode = str(compression_config.get("mode") or "off")
    if mode == "off":
        return CompressionRoutePlan(body=body, metadata=None)

    strategies: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    metadata: dict[str, Any] = {
        "mode": mode,
        "block_count": 0,
        "compressed_block_count": 0,
        "estimated_original_chars": 0,
        "estimated_compressed_chars": 0,
        "estimated_saved_chars": 0,
        "savings_ratio": 0.0,
        "strategies": strategies,
        "outcomes": outcomes,
        "needs_original_storage_count": 0,
    }
    try:
        plan = plan_chat_compression(body, compression_config)
        metadata["block_count"] = len(plan.blocks)
        for outcome in plan.outcomes:
            _increment_count(outcomes, outcome.reason)
        transformed_body = body
        for block in plan.blocks:
            _increment_count(strategies, block.strategy)
            result = compress(block, compression_config)
            if result is None:
                _increment_count(outcomes, "unsupported_strategy")
                continue
            if result.status != "compressed":
                _increment_count(outcomes, result.reason)
                continue
            compressed_chars = result.compressed_chars
            live_candidate_text = result.candidate_text
            if mode == "live_zone":
                live_candidate_text = live_zone_final_candidate(block.content, result.candidate_text)
                compressed_chars = len(live_candidate_text)
                if not compression_gate_accepts(result.original_chars, compressed_chars, compression_config):
                    _increment_count(outcomes, "not_smaller_after_marker")
                    continue
            metadata["compressed_block_count"] += 1
            metadata["estimated_original_chars"] += result.original_chars
            metadata["estimated_compressed_chars"] += compressed_chars
            if result.needs_original_storage:
                metadata["needs_original_storage_count"] += 1
            if mode == "live_zone":
                transformed_body = apply_live_zone_compression(
                    transformed_body,
                    body,
                    block,
                    result,
                    live_candidate_text,
                    compression_config,
                    store_path=store_path,
                    write_original=write_original,
                )
            _increment_count(outcomes, "compressed")
        saved = max(0, metadata["estimated_original_chars"] - metadata["estimated_compressed_chars"])
        metadata["estimated_saved_chars"] = saved
        original = metadata["estimated_original_chars"]
        metadata["savings_ratio"] = round(saved / original, 4) if original > 0 else 0.0
        if metadata["compressed_block_count"] > 0:
            metadata["status"] = "compressed" if mode == "live_zone" else "dry_run"
        elif metadata["block_count"] > 0:
            metadata["status"] = "rejected_not_smaller"
        elif "streaming_bypass" in outcomes:
            metadata["status"] = "streaming_bypass"
        else:
            metadata["status"] = "no_eligible_blocks"
        return CompressionRoutePlan(body=transformed_body, metadata=redact_sensitive_json(metadata))
    except Exception as exc:
        metadata["status"] = "error_original_forwarded"
        metadata["error_type"] = sanitize_error_detail(type(exc).__name__)
        metadata["outcomes"] = {"error": 1}
        return CompressionRoutePlan(body=body, metadata=redact_sensitive_json(metadata))


def evaluate_non_streaming_success_response(
    response: Any,
    model: dict[str, Any],
    *,
    latency_seconds: float,
    requested_model_is_virtual: bool,
    detect_success_error: SuccessErrorDetector,
    has_deliverable: DeliverablePredicate,
) -> NonStreamingAttemptDecision:
    status_code = int(response.status_code)
    latency = round(latency_seconds, 4)
    try:
        payload = response.json()
    except ValueError as exc:
        return _non_streaming_failure(
            model,
            status=status_code,
            reason="invalid_success_json",
            latency_seconds=latency,
            cooldown_reason="unavailable",
            cooldown_detail=f"invalid JSON success response: {exc}",
            retry=requested_model_is_virtual,
        )

    success_error = detect_success_error(payload, model)
    if success_error is not None:
        reason, detail, status = success_error
        failure_status = status or status_code
        return _non_streaming_failure(
            model,
            status=failure_status,
            reason=reason,
            latency_seconds=latency,
            cooldown_reason=reason,
            cooldown_detail=detail,
            retry=_should_retry(reason, requested_model_is_virtual),
        )

    if not has_deliverable(payload):
        # A reasoning model can burn the whole completion budget on reasoning tokens and stop before
        # emitting any assistant content. That is the caller's max_tokens being too small, not a
        # broken upstream, so `truncated_before_content` carries a no-cooldown policy: the failure is
        # still recorded and scored, but the model keeps its slot in the pool.
        reason, cooldown_reason, cooldown_detail = (
            (
                "truncated_before_content",
                "truncated_before_content",
                "response hit the completion token budget before emitting assistant content",
            )
            if finish_reason_is_truncation(payload)
            else (
                "empty_assistant_message",
                "unavailable",
                "success response had no assistant content or tool calls",
            )
        )
        return _non_streaming_failure(
            model,
            status=status_code,
            reason=reason,
            latency_seconds=latency,
            cooldown_reason=cooldown_reason,
            cooldown_detail=cooldown_detail,
            retry=requested_model_is_virtual,
        )

    return NonStreamingAttemptDecision(
        outcome="success",
        attempt_update={"status": status_code, "reason": "ok", "latency_seconds": latency},
        usage=normalized_token_usage(payload.get("usage")),
    )


def normalized_token_usage(raw: Any) -> dict[str, int] | None:
    """Prompt/completion token counts from an OpenAI-style usage block, or None.

    Token *counts* are the one usage fact recorded — they are not content, and they are
    what the admin "estimated saved" figure needs. Anything malformed reads as absent.
    """
    if not isinstance(raw, dict):
        return None
    prompt = _safe_int(raw.get("prompt_tokens"), None)
    completion = _safe_int(raw.get("completion_tokens"), None)
    if prompt is None and completion is None:
        return None
    return {"prompt_tokens": max(0, prompt or 0), "completion_tokens": max(0, completion or 0)}


def _should_retry(reason: str, requested_model_is_virtual: bool) -> bool:
    """Whether the next candidate is worth trying after this failure.

    A rejected request body is terminal even on a virtual profile: the next candidate receives the
    very same body and answers the very same 400, so failing over only burns healthy candidates and
    delays an error the caller has to fix anyway. It reaches here by two routes — an HTTP 400, and a
    200 whose payload carries `error.code: 400` — and both have to stop.
    """
    return requested_model_is_virtual and reason not in NON_RETRYABLE_FAILURE_REASONS


def _non_streaming_failure(
    model: dict[str, Any],
    *,
    status: int,
    reason: str,
    latency_seconds: float,
    cooldown_reason: str,
    cooldown_detail: str,
    retry: bool,
) -> NonStreamingAttemptDecision:
    return NonStreamingAttemptDecision(
        outcome="retryable_failure" if retry else "terminal_failure",
        attempt_update={"status": status, "reason": reason, "latency_seconds": latency_seconds},
        error={
            "model": model.get("id"),
            "upstream": model.get("upstream_id"),
            "source": model.get("source"),
            "status": status,
            "reason": reason,
        },
        cooldown_reason=cooldown_reason,
        cooldown_detail=cooldown_detail,
        cooldown_status=status,
    )


def evaluate_streaming_result(
    stream_result: dict[str, Any],
    model: dict[str, Any],
    *,
    response_status: int,
    latency_seconds: float,
    requested_model_is_virtual: bool,
    classify: FailureClassifier | None = None,
) -> StreamingAttemptDecision:
    reason = str(stream_result.get("reason") or "stream_failure")
    stream_started = bool(stream_result.get("stream_started"))
    classified_status: int | None = None
    if (
        stream_result.get("status") == "error"
        and not stream_started
        and reason == "pre_stream_failure"
        and classify is not None
    ):
        # A provider may put an OpenAI-style error object in the first SSE event while keeping the
        # HTTP status at 200. The stream reader has not committed that event to the client and keeps
        # its canonical `type`/`code` as `error_type`; recover the logical status here so this path
        # earns the same auth/rate-limit/request-contract verdict as a non-streaming wrapped error.
        error_code = str(stream_result.get("error_type") or "").strip()
        classified_status = status_for_error_codes(error_code)
        if classified_status is None:
            try:
                numeric_status = int(error_code)
            except ValueError:
                numeric_status = 0
            if 100 <= numeric_status <= 599:
                classified_status = numeric_status
        if classified_status is not None:
            message = str(stream_result.get("message") or "")
            reason = classify(classified_status, f"{error_code}: {message}".strip(), model)
    attempt_status = classified_status or response_status
    attempt_update = {
        "status": attempt_status,
        "reason": reason,
        "latency_seconds": round(latency_seconds, 4),
        "stream_started": stream_started,
        "stream_chunk_count": _safe_int(stream_result.get("chunk_count"), 0),
        "stream_bytes_sent": _safe_int(stream_result.get("bytes_sent"), 0),
    }
    if stream_result.get("status") == "ok":
        return StreamingAttemptDecision(
            outcome="success",
            attempt_update=attempt_update,
            usage=normalized_token_usage(stream_result.get("usage")),
        )

    error = {
        "model": model.get("id"),
        "upstream": model.get("upstream_id"),
        "source": model.get("source"),
        "status": attempt_status,
        "reason": reason,
        "stream_started": stream_started,
    }
    if stream_started:
        outcome: Literal["retryable_failure", "terminal_failure", "mid_stream_failure"] = "mid_stream_failure"
    elif _should_retry(reason, requested_model_is_virtual):
        outcome = "retryable_failure"
    else:
        outcome = "terminal_failure"
    return StreamingAttemptDecision(
        outcome=outcome,
        attempt_update=attempt_update,
        error=error,
        # A classified error object keeps the same cooldown scope as the equivalent HTTP/non-stream
        # failure. `client_disconnected` independently carries its own no-cooldown policy: the write
        # to the caller's socket failed, so the model keeps its slot. Other transport failures stay
        # the generic model-scoped `unavailable` verdict.
        cooldown_reason=(
            reason
            if classified_status is not None or reason in CALLER_CAUSED_FAILURE_REASONS
            else "unavailable"
        ),
        cooldown_detail=f"{reason}: {stream_result.get('error_type') or ''} {stream_result.get('message') or ''}".strip(),
        cooldown_status=classified_status or "stream_error",
    )


def evaluate_upstream_failure_response(
    response: Any,
    model: dict[str, Any],
    *,
    latency_seconds: float,
    requested_model_is_virtual: bool,
    classify: FailureClassifier,
) -> UpstreamFailureDecision:
    status_code = int(response.status_code)
    # On a streaming request the body is not buffered, so reading `.text` pulls the rest of the
    # connection and raises if the upstream drops it mid-transfer. That exception used to escape
    # `run_attempts` — whose only try/except wraps `invoke_model` — turning a failover into a 500.
    # An unreadable body is simply an unclassifiable one: keep the status, let `classify` judge it.
    try:
        text = str(response.text)
    except Exception as exc:
        text = f"<unreadable response body: {type(exc).__name__}>"
    reason = classify(status_code, text, model)
    return UpstreamFailureDecision(
        outcome="retryable_failure" if _should_retry(reason, requested_model_is_virtual) else "terminal_failure",
        attempt_update={"status": status_code, "reason": reason, "latency_seconds": round(latency_seconds, 4)},
        error={
            "model": model.get("id"),
            "upstream": model.get("upstream_id"),
            "source": model.get("source"),
            "status": status_code,
            "reason": reason,
            # Wider than the cooldown detail below on purpose: this one is what the caller reads.
            # Relays nest their wrappers ahead of the real message, so the sentence that names the
            # problem sits at the end — a 250 cut on the OpenCode Zen rejection landed five
            # characters short of "...passed back to the API.". The cooldown detail stays bounded
            # because it is written to state, once per cooled model, and only ever read for triage.
            "detail": text[:UPSTREAM_DETAIL_LIMIT],
        },
        cooldown_reason=reason,
        cooldown_detail=f"HTTP {status_code}: {text[:250]}",
        cooldown_status=status_code,
    )


def evaluate_invocation_exception(
    exc: Exception,
    model: dict[str, Any],
    *,
    latency_seconds: float,
    timeout: bool,
    deadline_exceeded: bool = False,
) -> InvocationExceptionDecision:
    error_type = type(exc).__name__
    detail = f"{error_type}: {exc}"
    if deadline_exceeded:
        # Ficelle's own request budget ended the attempt (L2-R2): named distinctly so the
        # route row states which budget ended the request, and never blamed on the model.
        reason = status = error_reason = "request_deadline_exceeded"
    else:
        reason = "timeout" if timeout else "unavailable"
        status = "timeout" if timeout else "exception"
        error_reason = "timeout" if timeout else "exception"
    return InvocationExceptionDecision(
        attempt_update={
            "status": status,
            "reason": reason,
            "error_type": error_type,
            "latency_seconds": round(latency_seconds, 4),
        },
        error={
            "model": model.get("id"),
            "upstream": model.get("upstream_id"),
            "source": model.get("source"),
            "reason": error_reason,
            "error": detail,
        },
        cooldown_reason=reason,
        cooldown_detail=detail,
        cooldown_status=status,
    )


def build_success_route_log(row: SuccessRouteLogInput) -> dict[str, Any]:
    route_log = {
        "request_id": row.request_id,
        "requested_model": row.safe_requested_model,
        "selected_model": row.selected_model.get("id"),
        "selected_upstream": row.selected_model.get("upstream_id"),
        "selected_source": row.selected_model.get("source"),
        "competence": row.competence,
        "final_status": row.final_status,
        "final_reason": "ok",
        "candidate_count": row.candidate_count,
        "attempt_count": row.attempt_count,
        "attempts": row.attempts,
        "duration_seconds": round(row.duration_seconds, 4),
        "stream": row.stream,
        "compression": row.compression,
    }
    if row.stream_started is not None:
        route_log["stream_started"] = row.stream_started
    if row.usage is not None:
        route_log["usage"] = row.usage
        # Recorded with the request, not joined at read time: the model's catalog row
        # can change or disappear later, and a savings estimate must be immutable
        # history — never inflatable by a later price change.
        reference = row.selected_model.get("reference_pricing")
        if isinstance(reference, dict):
            route_log["reference_pricing"] = {
                "prompt": reference.get("prompt"),
                "completion": reference.get("completion"),
            }
    return route_log


def build_success_route_telemetry(row: SuccessRouteLogInput) -> ChatCompletionRouteTelemetry:
    return ChatCompletionRouteTelemetry(
        last_route=ChatCompletionLastRouteRecord(
            safe_requested_model=row.safe_requested_model,
            status="ok",
            reason="ok",
            request_id=row.request_id,
            candidate_count=row.candidate_count,
            attempt_count=row.attempt_count,
            duration_seconds=row.duration_seconds,
            selected_model=row.selected_model,
            attempts=row.attempts,
            competence=row.competence,
            compression=row.compression,
        ),
        route_log=build_success_route_log(row),
    )


def failure_route_reason(errors: list[dict[str, Any]]) -> str:
    return "bad_upstream_request" if caller_rejected_request(errors) else "upstream_failure"


def build_failure_route_log(row: FailureRouteLogInput) -> dict[str, Any]:
    return {
        "request_id": row.request_id,
        "requested_model": row.safe_requested_model,
        "final_status": upstream_failure_status(row.errors),
        "final_reason": failure_route_reason(row.errors),
        "candidate_count": row.candidate_count,
        "attempt_count": row.attempt_count,
        "attempts": row.attempts,
        "duration_seconds": round(row.duration_seconds, 4),
        "stream": row.stream,
        "compression": row.compression,
    }


def build_failure_route_telemetry(row: FailureRouteLogInput) -> ChatCompletionRouteTelemetry:
    return ChatCompletionRouteTelemetry(
        last_route=ChatCompletionLastRouteRecord(
            safe_requested_model=row.safe_requested_model,
            status="fail",
            reason=failure_route_reason(row.errors),
            request_id=row.request_id,
            candidate_count=row.candidate_count,
            attempt_count=row.attempt_count,
            duration_seconds=row.duration_seconds,
            attempts=row.attempts,
            compression=row.compression,
        ),
        route_log=build_failure_route_log(row),
    )


def build_mid_stream_failure_route_log(row: MidStreamFailureRouteLogInput) -> dict[str, Any]:
    return {
        "request_id": row.request_id,
        "requested_model": row.safe_requested_model,
        "selected_model": row.selected_model.get("id"),
        "selected_upstream": row.selected_model.get("upstream_id"),
        "selected_source": row.selected_model.get("source"),
        "final_status": row.final_status,
        "final_reason": row.reason,
        "candidate_count": row.candidate_count,
        "attempt_count": row.attempt_count,
        "attempts": row.attempts,
        "duration_seconds": round(row.duration_seconds, 4),
        "stream": True,
        "stream_started": True,
        "compression": row.compression,
    }


def build_mid_stream_failure_route_telemetry(row: MidStreamFailureRouteLogInput) -> ChatCompletionRouteTelemetry:
    return ChatCompletionRouteTelemetry(
        last_route=ChatCompletionLastRouteRecord(
            safe_requested_model=row.safe_requested_model,
            status="fail",
            reason=row.reason,
            request_id=row.request_id,
            candidate_count=row.candidate_count,
            attempt_count=row.attempt_count,
            duration_seconds=row.duration_seconds,
            selected_model=row.selected_model,
            attempts=row.attempts,
            compression=row.compression,
        ),
        route_log=build_mid_stream_failure_route_log(row),
    )


def build_upstream_failure_response(
    row: UpstreamFailureResponseInput,
    *,
    build_error: UpstreamFailureErrorBuilder,
    build_headers: FailureResponseHeadersBuilder,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        status=upstream_failure_status(row.errors),
        payload=build_error(row.requested_model, row.request_id, row.candidate_count, row.attempts, row.errors),
        headers=build_headers(row.request_id, row.safe_requested_model, len(row.attempts), row.compression),
    )


def record_attempt_failure(row: AttemptFailureRecordInput) -> None:
    row.attempt.update(row.attempt_update)
    row.attempts.append(row.attempt)
    if row.error is not None:
        row.errors.append(row.error)


def record_attempt_result(row: AttemptResultRecordInput) -> None:
    row.attempt.update(row.attempt_update)
    row.attempts.append(row.attempt)


def build_attempt_cooldown_request(
    model: dict[str, Any],
    decision: AttemptCooldownDecision,
    *,
    fallback_reason: str | None = None,
    fallback_detail: object | None = None,
    fallback_status: CooldownStatus | None = None,
) -> AttemptCooldownRequest:
    reason = decision.cooldown_reason or fallback_reason
    status = decision.cooldown_status or fallback_status
    if reason is None or status is None:
        raise ValueError("cooldown reason and status are required")
    detail = decision.cooldown_detail
    if detail is None and fallback_detail is not None:
        detail = str(fallback_detail)
    return AttemptCooldownRequest(model=model, reason=str(reason), detail=detail, status=status)


def build_non_streaming_success_response(
    row: NonStreamingSuccessResponseInput,
    *,
    build_headers: SuccessResponseHeadersBuilder,
) -> ChatCompletionRawResponse:
    return ChatCompletionRawResponse(
        status=row.response_status,
        content_type=str(row.response_headers.get("Content-Type") or "application/json"),
        headers=build_headers(
            row.request_id,
            row.safe_requested_model,
            row.selected_model,
            row.attempt_count,
            row.compression,
            row.dropped_parameters,
        ),
        content=row.response_content,
    )


def build_streaming_response_start(
    row: StreamingResponseStartInput,
    *,
    build_headers: SuccessResponseHeadersBuilder,
) -> ChatCompletionStreamStart:
    return ChatCompletionStreamStart(
        status=row.response_status,
        content_type=str(row.response_headers.get("Content-Type") or "text/event-stream"),
        headers=build_headers(
            row.request_id,
            row.safe_requested_model,
            row.selected_model,
            row.attempt_count,
            row.compression,
            row.dropped_parameters,
        ),
    )


class ChatCompletionRouter:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        load_catalog: CatalogLoader,
        select_candidates: CandidateSelector,
        is_fusion_profile: FusionPredicate,
        is_virtual_model: VirtualModelPredicate,
        response_headers: ResponseHeadersBuilder,
        record_last_route: LastRouteRecorder,
        write_route_log: RouteLogWriter,
        max_attempts_for_request: MaxAttemptsCalculator,
        prepare_compression_route_body: CompressionPlanner,
        now: Clock,
        select_result: ResultSelector | None = None,
    ) -> None:
        self.config = config
        self.load_catalog = load_catalog
        self.select_candidates = select_candidates
        self.is_fusion_profile = is_fusion_profile
        self.is_virtual_model = is_virtual_model
        self.response_headers = response_headers
        self.record_last_route = record_last_route
        self.write_route_log = write_route_log
        self.max_attempts_for_request = max_attempts_for_request
        self.prepare_compression_route_body = prepare_compression_route_body
        self.now = now
        self.select_result = select_result

    def start(
        self,
        body: Any,
        *,
        request_id: str,
        request_started: float,
        request: ChatCompletionRequest | None = None,
    ) -> ChatCompletionStart:
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        request = request or normalize_chat_completion_request(body)
        # Before the catalog, and before Fusion is even considered: a body no provider can accept
        # is invalid whatever the routing would have been, and refusing it here is what turns a
        # billed round trip (N of them on a Fusion panel) into a local 400. `catalog` is left
        # empty on that path — the only reader, the Fusion branch in `router.py`, is unreachable
        # once `response` is set.
        malformed_tool_call = malformed_tool_call_detail(body)
        if malformed_tool_call is not None:
            return ChatCompletionStart(
                request=request,
                catalog={},
                candidates=[],
                is_fusion_request=False,
                response=self._malformed_tool_call_response(
                    body, request, request_id, request_started, malformed_tool_call
                ),
            )
        catalog = self.load_catalog(self.config)
        is_fusion_request = self.is_fusion_profile(request.requested_model)
        candidates = []
        selection = None
        if not is_fusion_request:
            if self.select_result is not None:
                selection = self.select_result(request.requested_model, catalog, self.config)
                candidates = selection.as_legacy_models()
            else:
                candidates = self.select_candidates(request.requested_model, catalog, self.config)
        response = None
        if not is_fusion_request and not candidates:
            if selection is not None:
                response = self._selection_refusal_response(
                    body, request, request_id, request_started, catalog, selection
                )
            else:
                response = self._no_available_model_response(body, request, request_id, request_started)
        return ChatCompletionStart(
            request=request,
            catalog=catalog,
            candidates=candidates,
            is_fusion_request=is_fusion_request,
            response=response,
        )

    def handle(
        self,
        body: Any,
        *,
        request_id: str,
        request_started: float,
        ports_factory: AttemptPortsFactory,
        request: ChatCompletionRequest | None = None,
    ) -> ChatCompletionHandleResult:
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        start = self.start(body, request_id=request_id, request_started=request_started, request=request)
        if start.is_fusion_request or start.response is not None:
            return ChatCompletionHandleResult(start=start)
        attempt_plan = self.plan_attempts(body, start.request, start.candidates)
        attempt_result = self.run_attempts(
            body,
            start.request,
            attempt_plan,
            request_id=request_id,
            request_started=request_started,
            ports=ports_factory(attempt_plan),
        )
        return ChatCompletionHandleResult(
            start=start,
            attempt_plan=attempt_plan,
            attempt_result=attempt_result,
        )

    def plan_attempts(
        self,
        body: dict[str, Any],
        request: ChatCompletionRequest,
        candidates: list[dict[str, Any]],
    ) -> ChatCompletionAttemptPlan:
        candidate_count = len(candidates)
        max_attempt_count = self.max_attempts_for_request(request.requested_model, self.config, candidate_count)
        compression_plan = self.prepare_compression_route_body(body, self.config)
        # Filtered before the diversity cap sees the pool, so the window is cut from candidates
        # that can actually be tried. The pool goes through the same filter or the loop's dynamic
        # re-planning would re-offer a candidate the plan just excluded.
        eligible, excluded = exclude_incompatible_candidates(compression_plan.body, candidates, self.config)
        attempt_candidates = attempts_with_source_diversity(eligible, max_attempt_count, self.config)
        return ChatCompletionAttemptPlan(
            candidates=attempt_candidates,
            candidate_count=candidate_count,
            routed_body=compression_plan.body,
            compression_metadata=compression_plan.metadata,
            requested_model_is_virtual=self.is_virtual_model(request.requested_model),
            candidate_pool=eligible,
            excluded_errors=excluded if not eligible else [],
        )

    def run_attempts(
        self,
        body: dict[str, Any],
        request: ChatCompletionRequest,
        plan: ChatCompletionAttemptPlan,
        *,
        request_id: str,
        request_started: float,
        ports: ChatCompletionAttemptPorts,
    ) -> ChatCompletionAttemptRunResult:
        # Seeded with what the plan ruled out before any attempt (L3-R4): copied, so appending a
        # failure here never writes back into the plan.
        errors: list[dict[str, Any]] = list(plan.excluded_errors)
        attempts: list[dict[str, Any]] = []
        deadline = request_deadline(self.config, request_started)
        # The body actually sent, and the sampling knobs removed from it. Both are scoped to
        # the CURRENT candidate: a knob one model refuses says nothing about the next one, so
        # every candidate starts from the caller's full request minus only what it is itself
        # known to reject. Caller intent is preserved as widely as it can be.
        attempt_body = plan.routed_body
        dropped_parameters: set[str] = set()
        # The plan's window was chosen before any reason was known, so what is left of it is the
        # loop's own state: `pending` shrinks by one per iteration and is re-planned when a failure
        # reason rules a source out. `considered` is the window position; `considered_ids` is what a
        # re-plan must not offer again.
        candidate_pool = plan.candidate_pool or plan.candidates
        pending = list(plan.candidates)
        considered_ids: set[str] = set()
        # What this request has ruled out for the rest of its own window, kept as the two shapes a
        # block actually takes: a source, and a quota key at whatever scope its provider declared.
        # Names cannot stand in for the second — a `model:` key blocks one candidate of a source and
        # a `shared_account:` key blocks candidates of several.
        ruled_out_sources: set[str] = set()
        ruled_out_quota_keys: set[str] = set()
        # Raised when a new fact invalidates the remaining window, so it is re-planned once per
        # useful diversion rather than once per attempt.
        divert_pending = False

        def is_ruled_out(candidate: dict[str, Any]) -> bool:
            # Read per candidate on each re-plan. Each attempt contributes at most one key, so the
            # `any` is bounded by the window and never by the pool — a few hundred string
            # comparisons on the worst request, against one HTTP round trip per attempt.
            return str(candidate.get("source") or "") in ruled_out_sources or any(
                ports.quota_cooldown_matches_model(key, candidate) for key in ruled_out_quota_keys
            )

        def rule_out(applied: AppliedCooldown | None) -> None:
            """Remember what a cooldown this request just wrote blocks beyond the candidate."""
            nonlocal divert_pending
            if applied is None:
                return
            if applied.provider_source and applied.provider_source not in ruled_out_sources:
                ruled_out_sources.add(applied.provider_source)
                divert_pending = True
            if applied.quota_key and applied.quota_key not in ruled_out_quota_keys:
                ruled_out_quota_keys.add(applied.quota_key)
                # A `model:` key normally matches only the candidate just considered, and the default
                # matcher deliberately matches nothing at all. Neither invalidates the remaining
                # window, and re-planning anyway is not neutral: it re-cuts the window from the pool
                # under a diversity cap derived from a smaller remaining-attempt count, which can
                # promote a source the plan had left out. So divert only when this key actually
                # blocks a candidate the request could still offer. Tested against `applied.quota_key`
                # rather than through `is_ruled_out`, which answers for every fact gathered so far and
                # would keep re-planning on candidates an earlier divert already excluded.
                # The provider branch above needs no such guard: it rules out a whole source, which
                # is a real exclusion even when the pool holds no other model of it.
                divert_pending = divert_pending or any(
                    str(candidate.get("id") or "") not in considered_ids
                    and ports.quota_cooldown_matches_model(applied.quota_key, candidate)
                    for candidate in candidate_pool
                )

        considered = 0
        # Set when a provider named a sampling knob it refuses: the same model is asked again,
        # once, with that knob removed. It reuses its window slot rather than taking a new one.
        retry_same_model: dict[str, Any] | None = None
        while pending or retry_same_model is not None:
            is_retry = retry_same_model is not None
            if is_retry:
                model = retry_same_model
                retry_same_model = None
            else:
                # Two kinds of fact rule part of the pool out of the rest of the window, and the loop
                # treats them alike because the remedy is the same. Most of them are read off the
                # cooldown the attempt just wrote (`rule_out`), which is what makes this exact: a
                # provider cooldown blocks a source, a quota cooldown blocks its declared scope, and
                # a model cooldown or a quarantine blocks nothing the window would offer again — so
                # the model-scoped reasons (timeout, server_error, unavailable, rate_limited_upstream)
                # leave the planned order alone without being listed anywhere.
                # `bad_upstream_contract` is the one fact that writes no state at all: the provider's
                # request validation refused this body, its siblings behind the same endpoint would
                # refuse it identically, and that is a verdict on the source read from the reason.
                # It is a per-REQUEST decision even though the cooldowns are persisted: the loop rules
                # out only what its own failures blocked and deliberately does not re-read cooldown
                # state. A mid-request re-read would let a concurrent request's writes and a refreshed
                # catalog move the pool under a window already half spent, where honouring the write we
                # just made keeps the re-plan derived from evidence this request owns — and keeps the
                # attempt budget (`considered + pending <= planned`) reasoning about a set that cannot
                # move.
                last_attempt = attempts[-1] if attempts else {}
                rejected_source = str(last_attempt.get("source") or "")
                if (
                    str(last_attempt.get("reason") or "") in SOURCE_DIVERTING_FAILURE_REASONS
                    and rejected_source not in ruled_out_sources
                ):
                    ruled_out_sources.add(rejected_source)
                    divert_pending = True
                # Once per new fact: an iteration that records no attempt row of its own leaves the
                # reason standing on the last one, and re-planning twice on it would re-cut the window
                # under a smaller cap.
                if divert_pending:
                    divert_pending = False
                    replanned = attempts_after_source_rejection(
                        candidate_pool,
                        considered_ids=considered_ids,
                        is_ruled_out=is_ruled_out,
                        remaining_attempts=len(plan.candidates) - considered,
                        config=self.config,
                    )
                    # Empty means the pool has no other source to offer. Keeping the window as planned
                    # is the lesser evil, for both facts above: some providers disable an option on
                    # selected models only, a classification can be wrong, and stranding the request
                    # after one attempt would be worse than either. The cooldown is what stops the
                    # NEXT request from coming back here; this one still spends what it was given.
                    pending = replanned or pending
                model = pending.pop(0)
                # New candidate: restore the caller's body, then apply only this model's
                # learned incompatibilities BEFORE the first send (the proactive half —
                # after one rejection, this model never sees that knob again).
                dropped_parameters = set(ports.unsupported_parameters_for(model, plan.routed_body))
                attempt_body = (
                    drop_sampling_parameters(plan.routed_body, dropped_parameters)
                    if dropped_parameters
                    else plan.routed_body
                )
            # Checked between attempts, never mid-attempt: an answer already being produced is
            # never abandoned. The first candidate always runs, so a deadline can shorten a
            # failing request but can never turn a request into zero attempts.
            if considered and deadline is not None and self.now() >= deadline:
                errors.append({
                    "model": model.get("id"),
                    "upstream": model.get("upstream_id"),
                    "source": model.get("source"),
                    "reason": "request_deadline_exceeded",
                })
                break
            if not is_retry:
                # A re-ask of the same model with a refused knob removed reuses its slot: the
                # window counts candidates, not round trips. Counting it here would also let a
                # drop-and-retry eat the budget a source rejection needs to re-plan with.
                considered += 1
                considered_ids.add(str(model.get("id") or ""))
            started = self.now()
            attempt = {
                "model": model.get("id"),
                "upstream": model.get("upstream_id"),
                "source": model.get("source"),
            }
            # The remaining request budget constrains the in-flight attempt (L2-R2), not
            # only the decision to start one: the invocation derives its read timeout from
            # it and the streaming consumer stops at the absolute deadline.
            remaining_budget = deadline - self.now() if deadline is not None else None
            try:
                response = ports.invoke_model(
                    model,
                    attempt_body,
                    self.config,
                    remaining_budget_seconds=remaining_budget,
                    deadline_monotonic=deadline,
                )
            except Exception as exc:
                latency = self.now() - started
                exception_decision = evaluate_invocation_exception(
                    exc,
                    model,
                    latency_seconds=latency,
                    timeout=ports.is_timeout_exception(exc),
                    deadline_exceeded=bool(
                        ports.is_deadline_exception is not None and ports.is_deadline_exception(exc)
                    ),
                )
                rule_out(ports.apply_cooldown(build_attempt_cooldown_request(model, exception_decision)))
                record_attempt_failure(
                    AttemptFailureRecordInput(
                        attempt=attempt,
                        attempt_update=exception_decision.attempt_update,
                        attempts=attempts,
                        errors=errors,
                        error=exception_decision.error,
                    )
                )
                continue

            latency = self.now() - started
            if 200 <= response.status_code < 300:
                if not body.get("stream"):
                    success_decision = evaluate_non_streaming_success_response(
                        response,
                        model,
                        latency_seconds=latency,
                        requested_model_is_virtual=plan.requested_model_is_virtual,
                        detect_success_error=ports.detect_success_error,
                        has_deliverable=ports.has_deliverable,
                    )
                    if success_decision.outcome != "success":
                        rule_out(
                            ports.apply_cooldown(
                                build_attempt_cooldown_request(
                                    model,
                                    success_decision,
                                    fallback_reason="unavailable",
                                    fallback_detail=success_decision.attempt_update.get("reason"),
                                    fallback_status=response.status_code,
                                )
                            )
                        )
                        record_attempt_failure(
                            AttemptFailureRecordInput(
                                attempt=attempt,
                                attempt_update=success_decision.attempt_update,
                                attempts=attempts,
                                errors=errors,
                                error=success_decision.error,
                            )
                        )
                        if success_decision.outcome == "terminal_failure":
                            break
                        continue
                    record_attempt_result(
                        AttemptResultRecordInput(
                            attempt=attempt,
                            attempt_update=success_decision.attempt_update,
                            attempts=attempts,
                        )
                    )
                    # Competence is derived from verified_capabilities, never from the `successes`
                    # this is about to write, so resolving it first costs nothing and lets the
                    # success and its telemetry share a single state write instead of two.
                    competence = ports.resolve_competence(request.requested_model, model)
                    ports.record_success_with_telemetry(
                        model,
                        latency,
                        build_success_route_telemetry(
                            SuccessRouteLogInput(
                                request_id=request_id,
                                safe_requested_model=request.safe_requested_model,
                                selected_model=model,
                                competence=competence,
                                final_status=response.status_code,
                                candidate_count=plan.candidate_count,
                                attempt_count=len(attempts),
                                attempts=attempts,
                                duration_seconds=self.now() - request_started,
                                stream=False,
                                compression=plan.compression_metadata,
                                usage=success_decision.usage,
                            )
                        )
                    )
                    return ChatCompletionAttemptRunResult(
                        outcome="non_streaming_success",
                        raw_response=build_non_streaming_success_response(
                            NonStreamingSuccessResponseInput(
                                request_id=request_id,
                                safe_requested_model=request.safe_requested_model,
                                selected_model=model,
                                attempt_count=len(attempts),
                                response_status=response.status_code,
                                response_headers=response.headers,
                                response_content=response.content,
                                compression=plan.compression_metadata,
                                dropped_parameters=tuple(sorted(dropped_parameters)),
                            ),
                            build_headers=ports.build_success_headers,
                        ),
                    )

                stream_result = ports.stream_response(
                    response, model, len(attempts) + 1, deadline_monotonic=deadline
                )
                latency = self.now() - started
                stream_decision = evaluate_streaming_result(
                    stream_result,
                    model,
                    response_status=response.status_code,
                    latency_seconds=latency,
                    requested_model_is_virtual=plan.requested_model_is_virtual,
                    classify=ports.classify_failure,
                )
                record_attempt_result(
                    AttemptResultRecordInput(
                        attempt=attempt,
                        attempt_update=stream_decision.attempt_update,
                        attempts=attempts,
                    )
                )
                if stream_decision.outcome == "success":
                    competence = ports.resolve_competence(request.requested_model, model)
                    ports.record_success_with_telemetry(
                        model,
                        latency,
                        build_success_route_telemetry(
                            SuccessRouteLogInput(
                                request_id=request_id,
                                safe_requested_model=request.safe_requested_model,
                                selected_model=model,
                                competence=competence,
                                final_status=response.status_code,
                                candidate_count=plan.candidate_count,
                                attempt_count=len(attempts),
                                attempts=attempts,
                                duration_seconds=self.now() - request_started,
                                stream=True,
                                stream_started=True,
                                compression=plan.compression_metadata,
                                usage=stream_decision.usage,
                            )
                        )
                    )
                    return ChatCompletionAttemptRunResult(outcome="streaming_complete")

                rule_out(
                    ports.apply_cooldown(
                        build_attempt_cooldown_request(
                            model,
                            stream_decision,
                            fallback_reason="unavailable",
                            fallback_detail=stream_decision.attempt_update.get("reason"),
                            fallback_status="stream_error",
                        )
                    )
                )
                if stream_decision.error is not None:
                    errors.append(stream_decision.error)
                if stream_decision.outcome == "mid_stream_failure":
                    ports.record_telemetry(
                        build_mid_stream_failure_route_telemetry(
                            MidStreamFailureRouteLogInput(
                                request_id=request_id,
                                safe_requested_model=request.safe_requested_model,
                                selected_model=model,
                                final_status=response.status_code,
                                candidate_count=plan.candidate_count,
                                attempt_count=len(attempts),
                                attempts=attempts,
                                duration_seconds=self.now() - request_started,
                                compression=plan.compression_metadata,
                                reason=stream_decision.attempt_update["reason"],
                            )
                        )
                    )
                    return ChatCompletionAttemptRunResult(outcome="streaming_complete")
                if stream_decision.outcome == "terminal_failure":
                    break
                continue

            upstream_failure = evaluate_upstream_failure_response(
                response,
                model,
                latency_seconds=latency,
                requested_model_is_virtual=plan.requested_model_is_virtual,
                classify=ports.classify_failure,
            )
            # A provider that rejects a sampling knob by name is telling us how to succeed:
            # drop that knob and ask the SAME model again, once. Only distribution-shaping
            # parameters qualify (never messages, tools, schemas or budgets), the retry is
            # announced to the caller, and one retry per attempt keeps the loop bounded.
            rejected_knobs = rejected_sampling_parameters(
                str(upstream_failure.error.get("detail") or ""), attempt_body
            ) if upstream_failure.error else ()
            retryable_knobs = tuple(knob for knob in rejected_knobs if knob not in dropped_parameters)
            if retryable_knobs:
                dropped_parameters.update(retryable_knobs)
                ports.learn_unsupported_parameters(model, retryable_knobs)
                attempt_body = drop_sampling_parameters(attempt_body, retryable_knobs)
                # The attempt keeps its true failure reason — the provider did reject the
                # body — with the consequence recorded alongside it. Overwriting the reason
                # would erase the evidence that a 400 happened at all.
                record_attempt_result(
                    AttemptResultRecordInput(
                        attempt=attempt,
                        attempt_update={
                            **upstream_failure.attempt_update,
                            "retried_without_parameters": list(retryable_knobs),
                        },
                        attempts=attempts,
                    )
                )
                retry_same_model = model
                continue
            rule_out(ports.apply_cooldown(build_attempt_cooldown_request(model, upstream_failure)))
            record_attempt_failure(
                AttemptFailureRecordInput(
                    attempt=attempt,
                    attempt_update=upstream_failure.attempt_update,
                    attempts=attempts,
                    errors=errors,
                    error=upstream_failure.error,
                )
            )
            if upstream_failure.outcome == "terminal_failure":
                break

        ports.record_telemetry(
            build_failure_route_telemetry(
                FailureRouteLogInput(
                    request_id=request_id,
                    safe_requested_model=request.safe_requested_model,
                    candidate_count=plan.candidate_count,
                    attempt_count=len(attempts),
                    attempts=attempts,
                    duration_seconds=self.now() - request_started,
                    stream=bool(body.get("stream")),
                    errors=errors,
                    compression=plan.compression_metadata,
                )
            )
        )
        return ChatCompletionAttemptRunResult(
            outcome="json_failure",
            json_response=build_upstream_failure_response(
                UpstreamFailureResponseInput(
                    requested_model=request.requested_model,
                    safe_requested_model=request.safe_requested_model,
                    request_id=request_id,
                    candidate_count=plan.candidate_count,
                    attempts=attempts,
                    errors=errors,
                    compression=plan.compression_metadata,
                ),
                build_error=ports.build_failure_error,
                build_headers=ports.build_failure_headers,
            ),
        )

    def _refusal_response(
        self,
        body: dict[str, Any],
        request: ChatCompletionRequest,
        request_id: str,
        request_started: float,
        *,
        status: int,
        reason: str,
        error_type: str,
        message: str,
        refusal_details: dict[str, Any] | None = None,
    ) -> ChatCompletionResponse:
        """A refusal Ficelle owns, logged like any other route so the Requests page shows it.

        `candidate_count` and `attempts` are 0/empty and mean it literally: no model was picked,
        none was called, none was cooled. One shape for every such refusal, so the route-log row
        the index reads cannot drift between them. `refusal_details` carries the same safe
        scope/reason facts into both the response payload and the route-log row, so telemetry
        and the client can never disagree about why the request was refused (L1-R4).
        """
        duration = self.now() - request_started
        self.record_last_route(request.safe_requested_model, "fail", reason, request_id, 0, 0, duration)
        route_row = {
            "request_id": request_id,
            "requested_model": request.safe_requested_model,
            "final_status": status,
            "final_reason": reason,
            "candidate_count": 0,
            "attempts": [],
            "duration_seconds": round(duration, 4),
            "stream": bool(body.get("stream")),
        }
        error_payload: dict[str, Any] = {"message": message, "type": error_type, "request_id": request_id}
        if refusal_details:
            route_row["refusal"] = dict(refusal_details)
            error_payload.update(refusal_details)
        self.write_route_log(route_row)
        return ChatCompletionResponse(
            status=status,
            payload={"error": error_payload},
            headers=self.response_headers(request_id, request.safe_requested_model),
        )

    def _malformed_tool_call_response(
        self,
        body: dict[str, Any],
        request: ChatCompletionRequest,
        request_id: str,
        request_started: float,
        detail: str,
    ) -> ChatCompletionResponse:
        """Its own reason class rather than `bad_upstream_request`: that one records a provider's
        verdict on a body Ficelle did forward, this one records Ficelle declining to ask."""
        return self._refusal_response(
            body,
            request,
            request_id,
            request_started,
            status=400,
            reason="malformed_tool_call",
            error_type="invalid_request_error",
            message=(
                f"invalid request body: {detail}. Every tool call replayed in the conversation "
                "history must name the function it called; fix the client that built this "
                "history, or start a new conversation. No provider was called: they all reject "
                "this body."
            ),
        )

    def _no_available_model_response(
        self,
        body: dict[str, Any],
        request: ChatCompletionRequest,
        request_id: str,
        request_started: float,
    ) -> ChatCompletionResponse:
        return self._refusal_response(
            body,
            request,
            request_id,
            request_started,
            status=503,
            reason="no_available_model",
            error_type="no_available_model",
            message=f"no invokable free tool-capable model for {request.safe_requested_model}",
        )

    def _selection_refusal_response(
        self,
        body: dict[str, Any],
        request: ChatCompletionRequest,
        request_id: str,
        request_started: float,
        catalog: dict[str, Any],
        selection: SelectionResult,
    ) -> ChatCompletionResponse:
        """Truthful zero-candidate refusals (L1-R4): a concrete model absent from the active
        catalog is 404 `model_not_found`; one that is present but blocked is 503
        `model_unavailable` with its safe scope and reason; an exhausted virtual pool is 503
        `no_available_model` with bounded exclusion counts. The message reflects the actual
        request — it never claims "tool-capable" for a plain text request."""
        requested = request.requested_model
        safe_requested = request.safe_requested_model
        if not self.is_virtual_model(requested):
            row = next(
                (
                    model
                    for model in catalog.get("models", [])
                    if isinstance(model, dict) and requested in {model.get("id"), model.get("upstream_id")}
                ),
                None,
            )
            if row is None:
                return self._refusal_response(
                    body,
                    request,
                    request_id,
                    request_started,
                    status=404,
                    reason="model_not_found",
                    error_type="model_not_found",
                    message=f"model {safe_requested} is not in the active catalog",
                )
            block_reason = selection.excluded_reasons.get(str(row.get("id") or "")) or "not_selectable"
            scope = REFUSAL_SCOPE_BY_BLOCK_REASON.get(block_reason, "model")
            return self._refusal_response(
                body,
                request,
                request_id,
                request_started,
                status=503,
                reason="model_unavailable",
                error_type="model_unavailable",
                message=(
                    f"model {safe_requested} is in the active catalog but temporarily "
                    f"not selectable ({block_reason}, scope: {scope})"
                ),
                refusal_details={"scope": scope, "block_reason": block_reason},
            )
        excluded_counts = Counter(selection.excluded_reasons.values())
        bounded = dict(sorted(excluded_counts.items(), key=lambda item: (-item[1], item[0]))[:8])
        capability = "tool-capable model" if body.get("tools") else "model"
        return self._refusal_response(
            body,
            request,
            request_id,
            request_started,
            status=503,
            reason="no_available_model",
            error_type="no_available_model",
            message=f"no invokable free {capability} available for {safe_requested}",
            refusal_details={"excluded": bounded} if bounded else None,
        )
