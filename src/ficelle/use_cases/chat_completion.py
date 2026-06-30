from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
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
from ficelle.redaction import redact_sensitive_json, sanitize_error_detail


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


@dataclass(frozen=True)
class StreamingAttemptDecision:
    outcome: Literal["success", "retryable_failure", "terminal_failure", "mid_stream_failure"]
    attempt_update: dict[str, Any]
    error: dict[str, Any] | None = None
    cooldown_reason: str | None = None
    cooldown_detail: str | None = None
    cooldown_status: CooldownStatus | None = None


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


@dataclass(frozen=True)
class FailureRouteLogInput:
    request_id: str
    safe_requested_model: str
    candidate_count: int
    attempt_count: int
    attempts: list[dict[str, Any]]
    duration_seconds: float
    stream: bool
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


@dataclass(frozen=True)
class StreamingResponseStartInput:
    request_id: str
    safe_requested_model: str
    selected_model: dict[str, Any]
    attempt_count: int
    response_status: int
    response_headers: Mapping[str, Any]
    compression: dict[str, Any] | None = None


CatalogLoader = Callable[[dict[str, Any]], dict[str, Any]]
CandidateSelector = Callable[[str, dict[str, Any], dict[str, Any]], list[dict[str, Any]]]
FusionPredicate = Callable[[str], bool]
VirtualModelPredicate = Callable[[str], bool]
ResponseHeadersBuilder = Callable[[str, str], dict[str, str]]
SuccessResponseHeadersBuilder = Callable[[str, str, dict[str, Any], int, dict[str, Any] | None], dict[str, str]]
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
ModelInvoker = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any]
CooldownApplier = Callable[[AttemptCooldownRequest], None]
SuccessRecorder = Callable[[dict[str, Any], float], None]
CompetenceResolver = Callable[[str, dict[str, Any]], str]
TelemetryRecorder = Callable[[ChatCompletionRouteTelemetry], None]
StreamResponseHandler = Callable[[Any, dict[str, Any], int], dict[str, Any]]
TimeoutDetector = Callable[[Exception], bool]
AttemptPortsFactory = Callable[[ChatCompletionAttemptPlan], "ChatCompletionAttemptPorts"]


@dataclass(frozen=True)
class ChatCompletionAttemptPorts:
    invoke_model: ModelInvoker
    apply_cooldown: CooldownApplier
    record_success: SuccessRecorder
    resolve_competence: CompetenceResolver
    record_telemetry: TelemetryRecorder
    stream_response: StreamResponseHandler
    detect_success_error: SuccessErrorDetector
    has_deliverable: DeliverablePredicate
    classify_failure: FailureClassifier
    is_timeout_exception: TimeoutDetector
    build_success_headers: SuccessResponseHeadersBuilder
    build_failure_error: UpstreamFailureErrorBuilder
    build_failure_headers: FailureResponseHeadersBuilder


COMPRESSION_PENDING_MARKER = "<<ficelle:compressed:pending>>"


def normalize_chat_completion_request(body: Any) -> ChatCompletionRequest:
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    requested_model = str(body.get("model") or DEFAULT_CHAT_COMPLETION_MODEL)
    return ChatCompletionRequest(
        requested_model=requested_model,
        safe_requested_model=sanitize_error_detail(requested_model, 250) or "[redacted]",
    )


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
            retry=requested_model_is_virtual,
        )

    if not has_deliverable(payload):
        return _non_streaming_failure(
            model,
            status=status_code,
            reason="empty_assistant_message",
            latency_seconds=latency,
            cooldown_reason="unavailable",
            cooldown_detail="success response had no assistant content or tool calls",
            retry=requested_model_is_virtual,
        )

    return NonStreamingAttemptDecision(
        outcome="success",
        attempt_update={"status": status_code, "reason": "ok", "latency_seconds": latency},
    )


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
) -> StreamingAttemptDecision:
    reason = str(stream_result.get("reason") or "stream_failure")
    stream_started = bool(stream_result.get("stream_started"))
    attempt_update = {
        "status": response_status,
        "reason": reason,
        "latency_seconds": round(latency_seconds, 4),
        "stream_started": stream_started,
        "stream_chunk_count": _safe_int(stream_result.get("chunk_count"), 0),
        "stream_bytes_sent": _safe_int(stream_result.get("bytes_sent"), 0),
    }
    if stream_result.get("status") == "ok":
        return StreamingAttemptDecision(outcome="success", attempt_update=attempt_update)

    error = {
        "model": model.get("id"),
        "upstream": model.get("upstream_id"),
        "source": model.get("source"),
        "status": response_status,
        "reason": reason,
        "stream_started": stream_started,
    }
    if stream_started:
        outcome: Literal["retryable_failure", "terminal_failure", "mid_stream_failure"] = "mid_stream_failure"
    elif requested_model_is_virtual:
        outcome = "retryable_failure"
    else:
        outcome = "terminal_failure"
    return StreamingAttemptDecision(
        outcome=outcome,
        attempt_update=attempt_update,
        error=error,
        cooldown_reason="unavailable",
        cooldown_detail=f"{reason}: {stream_result.get('error_type') or ''} {stream_result.get('message') or ''}".strip(),
        cooldown_status="stream_error",
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
    text = str(response.text)[:1000]
    reason = classify(status_code, text, model)
    return UpstreamFailureDecision(
        outcome="retryable_failure" if requested_model_is_virtual else "terminal_failure",
        attempt_update={"status": status_code, "reason": reason, "latency_seconds": round(latency_seconds, 4)},
        error={
            "model": model.get("id"),
            "upstream": model.get("upstream_id"),
            "source": model.get("source"),
            "status": status_code,
            "reason": reason,
            "detail": text[:250],
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
) -> InvocationExceptionDecision:
    error_type = type(exc).__name__
    detail = f"{error_type}: {exc}"
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


def build_failure_route_log(row: FailureRouteLogInput) -> dict[str, Any]:
    return {
        "request_id": row.request_id,
        "requested_model": row.safe_requested_model,
        "final_status": 502,
        "final_reason": "upstream_failure",
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
            reason="upstream_failure",
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
        "final_reason": "mid_stream_failure",
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
            reason="mid_stream_failure",
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
        status=502,
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
        catalog = self.load_catalog(self.config)
        is_fusion_request = self.is_fusion_profile(request.requested_model)
        candidates = []
        if not is_fusion_request:
            candidates = self.select_candidates(request.requested_model, catalog, self.config)
        response = None
        if not is_fusion_request and not candidates:
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
        attempt_candidates = candidates[:max_attempt_count]
        compression_plan = self.prepare_compression_route_body(body, self.config)
        return ChatCompletionAttemptPlan(
            candidates=attempt_candidates,
            candidate_count=candidate_count,
            routed_body=compression_plan.body,
            compression_metadata=compression_plan.metadata,
            requested_model_is_virtual=self.is_virtual_model(request.requested_model),
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
        errors: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        for model in plan.candidates:
            started = self.now()
            attempt = {
                "model": model.get("id"),
                "upstream": model.get("upstream_id"),
                "source": model.get("source"),
            }
            try:
                response = ports.invoke_model(model, plan.routed_body, self.config)
            except Exception as exc:
                latency = self.now() - started
                exception_decision = evaluate_invocation_exception(
                    exc,
                    model,
                    latency_seconds=latency,
                    timeout=ports.is_timeout_exception(exc),
                )
                ports.apply_cooldown(build_attempt_cooldown_request(model, exception_decision))
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
                        ports.apply_cooldown(
                            build_attempt_cooldown_request(
                                model,
                                success_decision,
                                fallback_reason="unavailable",
                                fallback_detail=success_decision.attempt_update.get("reason"),
                                fallback_status=response.status_code,
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
                    ports.record_success(model, latency)
                    record_attempt_result(
                        AttemptResultRecordInput(
                            attempt=attempt,
                            attempt_update=success_decision.attempt_update,
                            attempts=attempts,
                        )
                    )
                    competence = ports.resolve_competence(request.requested_model, model)
                    ports.record_telemetry(
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
                            ),
                            build_headers=ports.build_success_headers,
                        ),
                    )

                stream_result = ports.stream_response(response, model, len(attempts) + 1)
                latency = self.now() - started
                stream_decision = evaluate_streaming_result(
                    stream_result,
                    model,
                    response_status=response.status_code,
                    latency_seconds=latency,
                    requested_model_is_virtual=plan.requested_model_is_virtual,
                )
                record_attempt_result(
                    AttemptResultRecordInput(
                        attempt=attempt,
                        attempt_update=stream_decision.attempt_update,
                        attempts=attempts,
                    )
                )
                if stream_decision.outcome == "success":
                    ports.record_success(model, latency)
                    competence = ports.resolve_competence(request.requested_model, model)
                    ports.record_telemetry(
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
                            )
                        )
                    )
                    return ChatCompletionAttemptRunResult(outcome="streaming_complete")

                ports.apply_cooldown(
                    build_attempt_cooldown_request(
                        model,
                        stream_decision,
                        fallback_reason="unavailable",
                        fallback_detail=stream_decision.attempt_update.get("reason"),
                        fallback_status="stream_error",
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
            ports.apply_cooldown(build_attempt_cooldown_request(model, upstream_failure))
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

    def _no_available_model_response(
        self,
        body: dict[str, Any],
        request: ChatCompletionRequest,
        request_id: str,
        request_started: float,
    ) -> ChatCompletionResponse:
        duration = self.now() - request_started
        self.record_last_route(
            request.safe_requested_model,
            "fail",
            "no_available_model",
            request_id,
            0,
            0,
            duration,
        )
        self.write_route_log(
            {
                "request_id": request_id,
                "requested_model": request.safe_requested_model,
                "final_status": 503,
                "final_reason": "no_available_model",
                "candidate_count": 0,
                "attempts": [],
                "duration_seconds": round(duration, 4),
                "stream": bool(body.get("stream")),
            }
        )
        return ChatCompletionResponse(
            status=503,
            payload={
                "error": {
                    "message": f"no invokable free tool-capable model for {request.safe_requested_model}",
                    "type": "no_available_model",
                    "request_id": request_id,
                }
            },
            headers=self.response_headers(request_id, request.safe_requested_model),
        )
