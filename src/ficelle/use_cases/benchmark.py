from __future__ import annotations

import base64
import json
import mimetypes
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ficelle.use_cases.capability_discovery import (
    ROUTE_REJECTION_VERDICT,
    VERDICT_BASIS_KEY,
    CapabilityDiscoveryJob,
)


class BenchmarkMediaError(RuntimeError):
    """Raised when a media-specific benchmark lacks a usable local fixture."""


@dataclass(frozen=True)
class BenchmarkMediaPorts:
    safe_int: Callable[[Any, int], int]
    env_get: Callable[[str], str | None]
    default_media_path: Callable[[str], Path]
    load_json: Callable[[Path, Any], Any]
    vision_fixtures_path: Path
    now_seconds: Callable[[], float]


@dataclass(frozen=True)
class BenchmarkEvidencePorts:
    expected_test_type: Callable[[str], str]
    active_ttl_seconds: Callable[[], int]
    evidence_timestamp_value: Callable[[Any], str | None]
    parse_timestamp: Callable[[str | None], float | None]
    now_seconds: Callable[[], float]


def benchmark_media_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    settings = (config or {}).get("benchmark_media") if isinstance((config or {}).get("benchmark_media"), dict) else {}
    return settings if isinstance(settings, dict) else {}


def configured_benchmark_media_path(
    config: dict[str, Any] | None,
    kind: str,
    *,
    ports: BenchmarkMediaPorts,
) -> Path | None:
    env_key = f"FICELLE_BENCHMARK_{kind.upper()}_PATH"
    raw = ports.env_get(env_key) or benchmark_media_settings(config).get(f"{kind}_path")
    if not isinstance(raw, str) or not raw.strip():
        default = ports.default_media_path(kind)
        return default if default.exists() else None
    return Path(raw).expanduser()


def max_benchmark_media_bytes(config: dict[str, Any] | None, *, ports: BenchmarkMediaPorts) -> int:
    return max(1, ports.safe_int(benchmark_media_settings(config).get("max_bytes"), 5_000_000))


def benchmark_file_data_url(
    config: dict[str, Any] | None,
    kind: str,
    default_mime: str,
    *,
    ports: BenchmarkMediaPorts,
) -> str:
    path = configured_benchmark_media_path(config, kind, ports=ports)
    if path is None:
        raise BenchmarkMediaError(f"benchmark_media.{kind}_path is not configured")
    if not path.exists() or not path.is_file():
        raise BenchmarkMediaError(f"benchmark media file not found for {kind}: {path}")
    size = path.stat().st_size
    max_bytes = max_benchmark_media_bytes(config, ports=ports)
    if size <= 0:
        raise BenchmarkMediaError(f"benchmark media file is empty for {kind}: {path}")
    if size > max_bytes:
        raise BenchmarkMediaError(f"benchmark media file is too large for {kind}: {size} bytes > {max_bytes}")
    mime = mimetypes.guess_type(path.name)[0] or default_mime
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def benchmark_audio_part(config: dict[str, Any] | None, *, ports: BenchmarkMediaPorts) -> dict[str, Any]:
    path = configured_benchmark_media_path(config, "audio", ports=ports)
    if path is None:
        raise BenchmarkMediaError("benchmark_media.audio_path is not configured")
    data_url = benchmark_file_data_url(config, "audio", "audio/mpeg", ports=ports)
    _prefix, _separator, encoded = data_url.partition(",")
    audio_format = path.suffix.lower().lstrip(".") or "mp3"
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}


def benchmark_media_answer(config: dict[str, Any] | None, kind: str, default: str) -> str:
    raw = benchmark_media_settings(config).get(f"{kind}_answer")
    return raw.strip() if isinstance(raw, str) and raw.strip() else default


def vision_benchmark_fixtures(*, ports: BenchmarkMediaPorts) -> dict[str, str]:
    data = ports.load_json(ports.vision_fixtures_path, {})
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str) and v.startswith("data:image/")} if isinstance(data, dict) else {}


def pick_vision_fixture(index: int | None = None, *, ports: BenchmarkMediaPorts) -> tuple[str, str]:
    fixtures = sorted(vision_benchmark_fixtures(ports=ports).items())
    if not fixtures:
        raise BenchmarkMediaError("no vision benchmark fixtures available")
    if index is None:
        index = int(ports.now_seconds())
    number, url = fixtures[index % len(fixtures)]
    return number, url


# How long a verdict reached by route rejection stays in force. The verified-capability TTL runs for
# days because "the model answered and got it wrong" is a stable property; a 404 is not — it can be
# an account-level setting or a provider hiccup, and the model would otherwise stay gated on that
# profile long after the cause is fixed. One hour is under the default discovery interval, so a
# recovered route is re-probed on the very next cycle.
ROUTE_REJECTION_VERDICT_TTL_SECONDS = 3600


def benchmark_result_is_aged(
    row: dict[str, Any],
    *,
    ports: BenchmarkEvidencePorts,
    ttl_seconds: float | None = None,
    now_ts: float | None = None,
) -> bool:
    ttl = ports.active_ttl_seconds() if ttl_seconds is None else ttl_seconds
    # Only ever shortens — `None` and `<= 0` keep their "never ages" meaning below.
    if ttl is not None and isinstance(row, dict) and row.get(VERDICT_BASIS_KEY) == ROUTE_REJECTION_VERDICT:
        ttl = min(ttl, ROUTE_REJECTION_VERDICT_TTL_SECONDS)
    if ttl is None or ttl <= 0:
        return False
    stamp = ports.parse_timestamp(ports.evidence_timestamp_value(row))
    if stamp is None:
        return False
    current = ports.now_seconds() if now_ts is None else now_ts
    return (current - stamp) > ttl


def benchmark_result_test_type_matches(
    profile_id: str,
    row: Any,
    *,
    ports: BenchmarkEvidencePorts,
) -> bool:
    if not isinstance(row, dict) or not row:
        return False
    expected = ports.expected_test_type(profile_id)
    test_type = row.get("test_type")
    if test_type is None:
        return expected == "exact_text"
    return str(test_type or "") == expected


def benchmark_result_matches_current_test(
    profile_id: str,
    row: Any,
    *,
    ports: BenchmarkEvidencePorts,
    now_ts: float | None = None,
    ttl_seconds: float | None = None,
) -> bool:
    if not benchmark_result_test_type_matches(profile_id, row, ports=ports):
        return False
    return not benchmark_result_is_aged(row, ports=ports, ttl_seconds=ttl_seconds, now_ts=now_ts)


def stale_score_decay_factor(
    profile_id: str,
    row: Any,
    *,
    ports: BenchmarkEvidencePorts,
    now_ts: float | None = None,
    ttl_seconds: float | None = None,
) -> float:
    if not benchmark_result_test_type_matches(profile_id, row, ports=ports):
        return 0.0
    ttl = ports.active_ttl_seconds() if ttl_seconds is None else ttl_seconds
    if ttl is None or ttl <= 0:
        return 0.0
    stamp = ports.parse_timestamp(ports.evidence_timestamp_value(row))
    if stamp is None:
        return 0.0
    current = ports.now_seconds() if now_ts is None else now_ts
    age = current - stamp
    if age <= ttl:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((age - ttl) / ttl)))


def mark_stale_benchmark_row(
    profile_id: str,
    row: dict[str, Any],
    *,
    ports: BenchmarkEvidencePorts,
) -> dict[str, Any]:
    if benchmark_result_matches_current_test(profile_id, row, ports=ports):
        return row
    expected = ports.expected_test_type(profile_id)
    test_type = row.get("test_type")
    type_matches = benchmark_result_test_type_matches(profile_id, row, ports=ports)
    stale = dict(row)
    stale["status"] = "stale"
    stale["original_status"] = row.get("status")
    stale["expected_test_type"] = expected
    if type_matches:
        stale["message"] = "stale benchmark result: verdict older than the verified-capability TTL; re-benchmark to refresh"
    else:
        stale["message"] = f"stale benchmark result: expected {expected}, stored {test_type or 'unknown'}"
    return stale


def redacted_benchmark_row(
    profile_id: str,
    row: dict[str, Any],
    *,
    ports: BenchmarkEvidencePorts,
    redact_sensitive_json: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    safe = redact_sensitive_json(mark_stale_benchmark_row(profile_id, row, ports=ports))
    return safe if isinstance(safe, dict) else {}


def record_benchmark_result_in_state(
    state: dict[str, Any],
    profile_id: str,
    model: dict[str, Any],
    result: dict[str, Any],
    *,
    cooldown_key: Callable[[dict[str, Any]], str],
    redact_sensitive_json: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    key = cooldown_key(model)
    benchmark_results = state.setdefault("benchmark_results", {})
    if not isinstance(benchmark_results, dict):
        benchmark_results = {}
        state["benchmark_results"] = benchmark_results
    by_profile = benchmark_results.setdefault(key, {})
    if not isinstance(by_profile, dict):
        by_profile = {}
        benchmark_results[key] = by_profile
    by_profile[profile_id] = redact_sensitive_json(result)


def record_benchmark_failure_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    *,
    detail: str | None,
    profile_id: str | None,
    update_failure_stats: Callable[[dict[str, Any], dict[str, Any], str], None],
    record_model_error: Callable[..., None],
) -> None:
    update_failure_stats(state, model, reason)
    record_model_error(state, model, reason, detail, profile_id=profile_id)


def canary_status_from_benchmark_result(
    result: dict[str, Any],
    *,
    safe_int: Callable[[Any, int], int],
) -> str:
    return "pass" if safe_int((result.get("summary") or {}).get("failed"), 0) == 0 else "fail"


def canary_state_payload(result: dict[str, Any], status: str) -> dict[str, Any]:
    result_rows = [row for row in result.get("results", []) if isinstance(row, dict)]
    result_profiles = [
        str(row.get("profile_id"))
        for row in result_rows
        if row.get("profile_id")
    ]
    return {
        "status": status,
        "last_run_at": result.get("ran_at"),
        "profiles": result_profiles,
        "summary": result.get("summary"),
        "failed_profiles": [row.get("profile_id") for row in result_rows if row.get("status") != "pass"],
    }


def apply_canary_result_to_state(state: dict[str, Any], result: dict[str, Any], status: str) -> None:
    state["canary"] = canary_state_payload(result, status)


def profile_ids_from_admin_body(body: dict[str, Any]) -> list[str]:
    raw_profiles = body.get("profiles", body.get("profile_ids", []))
    if raw_profiles is None:
        return []
    if isinstance(raw_profiles, str):
        return [raw_profiles]
    if isinstance(raw_profiles, list):
        if not all(isinstance(item, str) for item in raw_profiles):
            raise ValueError("profiles must contain only strings")
        return raw_profiles
    raise ValueError("profiles must be a string or list")


def run_benchmark_profiles(
    profile_ids: list[str],
    config: dict[str, Any],
    *,
    benchmark_profile: Callable[..., dict[str, Any]],
    known_profile_ids: set[str],
    default_profile_ids: list[str],
    safe_int: Callable[[Any, int], int],
    now_iso: Callable[[], str],
    all_candidates: bool = False,
    only_model_id: str | None = None,
) -> dict[str, Any]:
    selected = [profile_id for profile_id in profile_ids if profile_id in known_profile_ids]
    if not selected:
        selected = list(default_profile_ids)
    results = [
        benchmark_profile(
            profile_id,
            config,
            all_candidates=all_candidates,
            only_model_id=only_model_id,
        )
        for profile_id in selected
    ]
    candidate_summary = {
        "tested": sum(safe_int(result.get("tested_count"), 0) for result in results),
        "passed": sum(safe_int(result.get("passed_count"), 0) for result in results),
        "failed": sum(safe_int(result.get("failed_count"), 0) for result in results),
    } if all_candidates else None
    return {
        "ran_at": now_iso(),
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for result in results if result.get("status") == "pass"),
            "failed": sum(1 for result in results if result.get("status") != "pass"),
            **({"candidates": candidate_summary} if candidate_summary is not None else {}),
        },
    }


def _bounded_int(
    value: Any,
    *,
    safe_int: Callable[[Any, int], int],
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    return min(maximum, max(minimum, safe_int(value, default)))


def build_compression_benchmark_source(
    config: dict[str, Any] | None,
    *,
    safe_int: Callable[[Any, int], int],
    default_chars: int,
    min_chars: int,
    max_chars: int,
) -> str:
    target = default_chars
    if isinstance(config, dict):
        target = _bounded_int(
            config.get("benchmark_compression_chars"),
            safe_int=safe_int,
            default=target,
            minimum=min_chars,
            maximum=max_chars,
        )
    filler = (
        "The team reviewed routine logistics, meeting cadence, and status notes, "
        "with no decisions taken. "
    )
    facts = [
        "Key fact: the project budget is 4200 euros.",
        "Key fact: the lead engineer is Lena Park.",
        "Key fact: the tracking ticket is KX-19.",
    ]
    parts: list[str] = []
    chars = 0
    inserted = [False, False, False]
    thresholds = (0.1, 0.5, 0.9)
    while chars < target:
        parts.append(filler)
        chars += len(filler)
        for index, threshold in enumerate(thresholds):
            if not inserted[index] and chars >= target * threshold:
                parts.append(facts[index] + " ")
                inserted[index] = True
    for index, fact in enumerate(facts):
        if not inserted[index]:
            parts.append(fact + " ")
    body = "".join(parts)
    return (
        "Conversation history to compact:\n\n" + body
        + "\n\nSummarize the history above in at most 40 words. Preserve every key fact "
        "(numbers, names, identifiers) exactly."
    )


def build_long_context_benchmark_source(
    config: dict[str, Any] | None,
    *,
    safe_int: Callable[[Any, int], int],
    default_tokens: int,
    min_tokens: int,
    max_tokens: int,
    chars_per_token: int,
) -> str:
    target_tokens = default_tokens
    if isinstance(config, dict):
        target_tokens = _bounded_int(
            config.get("benchmark_long_context_tokens"),
            safe_int=safe_int,
            default=target_tokens,
            minimum=min_tokens,
            maximum=max_tokens,
        )
    target_chars = target_tokens * chars_per_token
    needles = [
        "long-needle-alpha-4731",
        "long-needle-bravo-8052",
        "long-needle-charlie-2649",
    ]
    filler = "This sentence is neutral background context with no instructions and no secrets. "
    depths = (0.15, 0.5, 0.85)
    parts: list[str] = []
    chars = 0
    inserted = [False, False, False]
    while chars < target_chars:
        parts.append(filler)
        chars += len(filler)
        for index, depth in enumerate(depths):
            if not inserted[index] and chars >= target_chars * depth:
                parts.append(f"Secret token {index + 1} is {needles[index]}. ")
                inserted[index] = True
    for index, needle in enumerate(needles):
        if not inserted[index]:
            parts.append(f"Secret token {index + 1} is {needle}. ")
    body = "".join(parts)
    return (
        "Read the following long document carefully.\n\n" + body
        + "\n\nThe document contains three secret tokens shaped like long-needle-WORD-NNNN. "
        "Reply with all three secret tokens."
    )


@dataclass(frozen=True)
class ProbeRegistry:
    compression_max_words: int = 60

    def validate_response(
        self,
        test_type: str,
        expected: str,
        text: str,
        payload: Any | None = None,
    ) -> tuple[bool, str]:
        if test_type in {"json", "fast"}:
            return self._validate_json_response(test_type, expected, text)
        if test_type == "exact_text":
            if text.strip() == expected:
                return True, "exact text ok"
            return False, f"expected exact text {expected!r}, got {text[:80]!r}"
        if test_type in {"audio", "video", "vision"}:
            if expected.lower() in text.lower():
                return True, f"{test_type} ok"
            return False, f"expected {expected!r} in {test_type} answer, got {text[:80]!r}"
        if test_type == "tool_call":
            return self._validate_tool_call(expected, payload)
        if test_type == "orchestration":
            return self._validate_orchestration(expected, payload)
        if test_type == "compression":
            return self._validate_compression(expected, text)
        if test_type == "reasoning":
            if not extract_reasoning_text(payload):
                return False, "no reasoning trace returned (parameter accepted but not honored)"
            if expected.lower() in text.lower():
                return True, "reasoning ok"
            return False, f"reasoning returned but answer marker {expected!r} missing"
        if test_type == "long_context":
            needles = expected.split("|")
            missing = [needle for needle in needles if needle.lower() not in text.lower()]
            if missing:
                return False, f"missed {len(missing)} of {len(needles)} needles: {missing}"
            return True, "long context ok"
        if expected.lower() in text.lower():
            return True, f"{test_type} ok"
        return False, f"expected marker not found: {expected}"

    def _validate_json_response(self, test_type: str, expected: str, text: str) -> tuple[bool, str]:
        try:
            parsed = json.loads(text)
        except Exception as exc:
            return False, f"invalid JSON: {exc}"
        if not isinstance(parsed, dict):
            return False, "JSON response is not an object"
        if not expected.startswith("{"):
            if test_type == "fast":
                return (True, "fast ok") if parsed.get(expected) is True else (False, f"missing {expected}=true")
            return (True, "json ok") if parsed.get(expected) == "ok" else (False, f"missing {expected}=ok")
        try:
            want = json.loads(expected)
        except Exception as exc:
            return False, f"benchmark expectation not JSON: {exc}"
        ok, reason = json_fields_match(want, parsed)
        return (True, "json ok") if ok else (False, reason)

    def _validate_tool_call(self, expected: str, payload: Any | None) -> tuple[bool, str]:
        want_tool, _, want_status = expected.partition(":")
        if not want_status:
            want_tool, want_status = "ficelle_probe", expected
        for call in extract_tool_calls(payload):
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            if function.get("name") != want_tool:
                continue
            if str(tool_call_arguments(call).get("status") or "") == want_status:
                return True, "tool call ok"
        return False, f"expected {want_tool} tool call with status={want_status!r} not found"

    def _validate_orchestration(self, expected: str, payload: Any | None) -> tuple[bool, str]:
        parts = expected.split(":")
        want_tool = parts[0] if parts else ""
        want_category = parts[1] if len(parts) > 1 else ""
        want_ticket = parts[2] if len(parts) > 2 else ""
        for call in extract_tool_calls(payload):
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            if function.get("name") != want_tool:
                continue
            args = tool_call_arguments(call)
            category = str(args.get("category") or "").strip().lower()
            ticket = str(args.get("ticket_id") or "").strip().lstrip("#").lower()
            if category == want_category.lower() and ticket == want_ticket.lower():
                return True, "orchestration ok"
        return False, f"expected {want_tool} call with category={want_category!r} ticket={want_ticket!r}"

    def _validate_compression(self, expected: str, text: str) -> tuple[bool, str]:
        word_count = len(text.split())
        if word_count > self.compression_max_words:
            return False, f"summary not bounded: {word_count} words (> {self.compression_max_words})"
        missing = [fact for fact in expected.split("|") if fact.lower() not in text.lower()]
        if missing:
            return False, f"summary dropped key facts: {missing}"
        return True, "compression ok"


@dataclass(frozen=True)
class BenchmarkProbeBuilder:
    canonical_virtual_model_id: Callable[[str], str]
    compression_source: Callable[[dict[str, Any] | None], str]
    long_context_source: Callable[[dict[str, Any] | None], str]
    pick_vision_fixture: Callable[[], tuple[str, str]]
    benchmark_file_data_url: Callable[[dict[str, Any] | None, str, str], str]
    benchmark_audio_part: Callable[[dict[str, Any] | None], dict[str, Any]]
    benchmark_media_answer: Callable[[dict[str, Any] | None, str, str], str]
    video_default_answer: str
    audio_default_answer: str

    def build(self, profile_id: str, config: dict[str, Any] | None = None) -> tuple[dict[str, Any], str, str]:
        canonical_profile_id = self.canonical_virtual_model_id(profile_id)
        if canonical_profile_id == "ficelle/auto-orchestrator":
            return self._orchestrator_probe(profile_id)
        if canonical_profile_id == "ficelle/auto-tools":
            return self._tool_probe(profile_id)
        if canonical_profile_id == "ficelle/auto-json":
            return self._json_probe(profile_id)
        if canonical_profile_id == "ficelle/auto-vision":
            return self._vision_probe(profile_id)
        if canonical_profile_id == "ficelle/auto-video":
            return self._video_probe(profile_id, config)
        if canonical_profile_id == "ficelle/auto-audio":
            return self._audio_probe(profile_id, config)
        if canonical_profile_id == "ficelle/auto-reasoning":
            return self._reasoning_probe(profile_id)
        if canonical_profile_id == "ficelle/auto-compression":
            return self._compression_probe(profile_id, config)
        if canonical_profile_id == "ficelle/auto-fast":
            return self._fast_probe(profile_id)
        if canonical_profile_id == "ficelle/auto-long":
            return self._long_context_probe(profile_id, config)
        marker = {
            "ficelle/auto-multimodal": "multimodal-router-ok",
        }.get(canonical_profile_id, "router-ok")
        return self._exact_text_probe(profile_id, marker)

    @staticmethod
    def _orchestrator_probe(profile_id: str) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Ficelle's task orchestrator. Decide the next action and call "
                            "exactly one tool. Do not answer in prose."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Ticket #A7: summarize the attached report into bullet points.",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_probe_1",
                                "type": "function",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": "{\"query\": \"is the report already loaded?\"}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_probe_1",
                        "content": "The report text is already loaded. No search needed; proceed to dispatch.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Good. Now dispatch the task: call route_task with the matching "
                            "category and the ticket id."
                        ),
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_docs",
                            "description": "Search the knowledge base for a query string.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "route_task",
                            "description": "Dispatch a task to a worker by category, echoing its ticket id.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "category": {
                                        "type": "string",
                                        "enum": ["summarize", "code", "extract", "chat"],
                                    },
                                    "ticket_id": {"type": "string"},
                                },
                                "required": ["category", "ticket_id"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Return the final answer to the user.",
                            "parameters": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                        },
                    },
                ],
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": 160,
            },
            "orchestration",
            "route_task:summarize:A7",
        )

    @staticmethod
    def _tool_probe(profile_id: str) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "Call the single most appropriate tool. Do not answer in prose.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Ignore the weather tool. Report the Ficelle status: if 7 is greater "
                            "than 3 use alpha-ok, otherwise beta-no."
                        ),
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "ficelle_probe",
                            "description": "Report a Ficelle routing status token.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string", "enum": ["alpha-ok", "beta-no"]}
                                },
                                "required": ["status"],
                                "additionalProperties": False,
                            },
                        },
                    },
                ],
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": 80,
            },
            "tool_call",
            "ficelle_probe:alpha-ok",
        )

    @staticmethod
    def _json_probe(profile_id: str) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Extract fields from the note and return ONLY a JSON object with keys: "
                            "name (string), age (number), active (boolean), tags (array of strings), "
                            "phone (string or null). If a field is absent, use null. Add no other keys.\n\n"
                            "Note: Maya Lopez (34) - status: active. Interests: hiking, chess. "
                            "No phone on file. [ref#9931 internal]"
                        ),
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                # 768, not 160: reasoning models spend the token budget on their hidden trace BEFORE
                # emitting `content`, so a tight cap returns an empty assistant message that the
                # validator reads as "invalid JSON: Expecting value (char 0)" — a false capability
                # failure. Measured on a free reasoning panel the trace alone runs 0-490 tokens;
                # nvidia/nemotron-nano-9b-v2 needs ~460-530 completion tokens (4 runs) to finish.
                # 768 covers the worst legitimate case with ~45% margin while well-behaved models
                # still stop early (80-211 tokens), so the larger cap costs them nothing. Mirrors
                # auto-reasoning's max_tokens:1024, which already sizes for the trace.
                "max_tokens": 768,
            },
            "json",
            '{"name": "Maya Lopez", "age": 34, "active": true, "tags": ["hiking", "chess"], "phone": null}',
        )

    def _vision_probe(self, profile_id: str) -> tuple[dict[str, Any], str, str]:
        number, image_url = self.pick_vision_fixture()
        return (
            {
                "model": profile_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Read the number shown in the image. Reply with only that number.",
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
            },
            "vision",
            number,
        )

    def _video_probe(
        self,
        profile_id: str,
        config: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "A short video shows a number. Reply with only that number.",
                            },
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": self.benchmark_file_data_url(config, "video", "video/mp4")
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
            },
            "video",
            self.benchmark_media_answer(config, "video", self.video_default_answer),
        )

    def _audio_probe(
        self,
        profile_id: str,
        config: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Reply with only the single word spoken in the audio, lowercase.",
                            },
                            self.benchmark_audio_part(config),
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
            },
            "audio",
            self.benchmark_media_answer(config, "audio", self.audio_default_answer),
        )

    @staticmethod
    def _reasoning_probe(profile_id: str) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Three friends - Alice, Bob, and Carol - sit in a row of three seats "
                            "(left, middle, right). Clue 1: Alice is not on either end. Clue 2: "
                            "Bob sits immediately to the left of Alice. Reason step by step, then "
                            "on the final line reply with ONLY this token: reasoning-<NAME>-ok, "
                            "where <NAME> is the lowercase first name of the person on the right end."
                        ),
                    },
                ],
                "reasoning": {"effort": "low"},
                "include_reasoning": True,
                "temperature": 0,
                "max_tokens": 1024,
            },
            "reasoning",
            "reasoning-carol-ok",
        )

    def _compression_probe(
        self,
        profile_id: str,
        config: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [{"role": "user", "content": self.compression_source(config)}],
                "temperature": 0,
                "max_tokens": 200,
            },
            "compression",
            "4200|Lena Park|KX-19",
        )

    @staticmethod
    def _fast_probe(profile_id: str) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [{"role": "user", "content": 'Return only this JSON object: {"fast_ok": true}'}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 40,
            },
            "fast",
            "fast_ok",
        )

    def _long_context_probe(
        self,
        profile_id: str,
        config: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [{"role": "user", "content": self.long_context_source(config)}],
                "temperature": 0,
                "max_tokens": 120,
            },
            "long_context",
            "long-needle-alpha-4731|long-needle-bravo-8052|long-needle-charlie-2649",
        )

    @staticmethod
    def _exact_text_probe(profile_id: str, marker: str) -> tuple[dict[str, Any], str, str]:
        return (
            {
                "model": profile_id,
                "messages": [{"role": "user", "content": f"Reply exactly: {marker}"}],
                "temperature": 0,
                "max_tokens": 16,
            },
            "exact_text",
            marker,
        )


@dataclass(frozen=True)
class BenchmarkRunner:
    load_or_refresh_catalog: Callable[[dict[str, Any]], dict[str, Any]]
    select_models: Callable[..., list[dict[str, Any]]]
    order_benchmark_candidates: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
    benchmark_body: Callable[[str, dict[str, Any] | None], tuple[dict[str, Any], str, str]]
    invoke_model: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any]
    classify_failure: Callable[[int, str, dict[str, Any]], str]
    set_cooldown: Callable[..., None]
    record_benchmark_failure: Callable[..., None]
    record_benchmark_result: Callable[[str, dict[str, Any], dict[str, Any]], None]
    record_success: Callable[[dict[str, Any], float], None]
    record_verified_capability: Callable[[str, dict[str, Any], dict[str, Any]], None]
    record_capability_discrepancy: Callable[[str, dict[str, Any], bool], None]
    extract_message_text: Callable[[Any], str]
    safe_detail: Callable[..., str]
    redact_sensitive_json: Callable[[dict[str, Any]], dict[str, Any]]
    now_iso: Callable[[], str]
    route_blocking_reasons: frozenset[str]
    media_error_types: tuple[type[BaseException], ...] = ()
    registry: ProbeRegistry = ProbeRegistry()

    def run_benchmarks(
        self,
        profile_ids: list[str],
        config: dict[str, Any],
        *,
        known_profile_ids: set[str],
        default_profile_ids: list[str],
        safe_int: Callable[[Any, int], int],
        all_candidates: bool = False,
        only_model_id: str | None = None,
    ) -> dict[str, Any]:
        return run_benchmark_profiles(
            profile_ids,
            config,
            benchmark_profile=self.benchmark_profile,
            known_profile_ids=known_profile_ids,
            default_profile_ids=default_profile_ids,
            safe_int=safe_int,
            now_iso=self.now_iso,
            all_candidates=all_candidates,
            only_model_id=only_model_id,
        )

    def benchmark_profile(
        self,
        profile_id: str,
        config: dict[str, Any],
        *,
        all_candidates: bool = False,
        only_model_id: str | None = None,
    ) -> dict[str, Any]:
        catalog = self.load_or_refresh_catalog(config)
        selected_models = self.select_models(profile_id, catalog, config, purpose="benchmark")
        candidates = self.order_benchmark_candidates(profile_id, selected_models)
        if only_model_id:
            candidates = [model for model in candidates if model.get("id") == only_model_id]
            if not candidates:
                return {
                    "profile_id": profile_id,
                    "status": "fail",
                    "test_type": "selection",
                    "ran_at": self.now_iso(),
                    "message": (
                        f"{only_model_id} is not an eligible benchmark candidate for {profile_id} "
                        "(excluded, cooling down, disabled, or not matching the profile requirements)"
                    ),
                }
        if not candidates:
            return {
                "profile_id": profile_id,
                "status": "fail",
                "test_type": "selection",
                "ran_at": self.now_iso(),
                "message": "no eligible invokable model",
            }

        try:
            body, test_type, expected = self.benchmark_body(profile_id, config)
        except Exception as exc:
            if not isinstance(exc, self.media_error_types):
                raise
            return {
                "profile_id": profile_id,
                "status": "fail",
                "test_type": "media_fixture",
                "ran_at": self.now_iso(),
                "message": self.safe_detail(exc),
            }

        last_result: dict[str, Any] | None = None
        candidate_results: list[dict[str, Any]] = []
        for model in candidates:
            result = self._benchmark_candidate(profile_id, model, body, test_type, expected, config)
            if all_candidates:
                candidate_results.append(self.redact_sensitive_json(dict(result)))
                last_result = result
                continue
            if result.get("status") == "pass":
                return result
            last_result = result

        if all_candidates:
            passed_count = sum(1 for result in candidate_results if result.get("status") == "pass")
            return {
                "profile_id": profile_id,
                "status": "pass" if passed_count else "fail",
                "test_type": test_type,
                "ran_at": self.now_iso(),
                "message": f"{passed_count}/{len(candidate_results)} candidates passed",
                "candidate_count": len(candidates),
                "tested_count": len(candidate_results),
                "passed_count": passed_count,
                "failed_count": len(candidate_results) - passed_count,
                "candidate_results": candidate_results,
            }

        if last_result is not None:
            last_result = dict(last_result)
            last_result["message"] = self.safe_detail(
                f"all candidates failed; last failure: {last_result.get('message')}"
            )
            return self.redact_sensitive_json(last_result)
        return {
            "profile_id": profile_id,
            "status": "fail",
            "test_type": test_type,
            "ran_at": self.now_iso(),
            "message": "no benchmark candidate completed",
        }

    def _benchmark_candidate(
        self,
        profile_id: str,
        model: dict[str, Any],
        body: dict[str, Any],
        test_type: str,
        expected: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.time()
        result: dict[str, Any] = {
            "profile_id": profile_id,
            "model_id": model.get("id"),
            "upstream_id": model.get("upstream_id"),
            "source": model.get("source"),
            "test_type": test_type,
            "ran_at": self.now_iso(),
        }
        try:
            response = self.invoke_model(model, body, config)
            latency = time.time() - started
            result["latency_seconds"] = round(latency, 3)
            if not (200 <= response.status_code < 300):
                return self._record_http_failure(profile_id, model, response, config, result)
            payload = response.json()
            text = self.extract_message_text(payload)
            passed, message = self.registry.validate_response(test_type, expected, text, payload)
            result.update({
                "status": "pass" if passed else "fail",
                "message": self.safe_detail(message),
                "text_preview": self.safe_detail(text, 160),
            })
            if passed:
                self.record_success(model, latency)
                self.record_benchmark_result(profile_id, model, result)
                self.record_verified_capability(profile_id, model, result)
                self.record_capability_discrepancy(profile_id, model, True)
                return result
            self.record_benchmark_failure(model, "benchmark_failed", profile_id=profile_id, detail=message)
            self.record_benchmark_result(profile_id, model, result)
            self.record_verified_capability(profile_id, model, result)
            self.record_capability_discrepancy(profile_id, model, False)
            return result
        except Exception as exc:
            self.set_cooldown(
                model,
                "unavailable",
                config,
                detail=f"benchmark {type(exc).__name__}: {exc}",
                profile_id=profile_id,
            )
            result.update({"status": "fail", "message": self.safe_detail(f"{type(exc).__name__}: {exc}")})
            self.record_benchmark_result(profile_id, model, result)
            return result

    def _record_http_failure(
        self,
        profile_id: str,
        model: dict[str, Any],
        response: Any,
        config: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        reason = self.classify_failure(response.status_code, response.text[:1000], model)
        detail = f"benchmark HTTP {response.status_code}: {response.text[:250]}"
        if reason in self.route_blocking_reasons:
            self.set_cooldown(model, reason, config, detail=detail, profile_id=profile_id)
        else:
            self.record_benchmark_failure(model, reason, profile_id=profile_id, detail=detail)
        result.update({
            "status": "fail",
            "message": f"HTTP {response.status_code}: {reason}",
            "text_preview": self.safe_detail(response.text, 160),
        })
        self.record_benchmark_result(profile_id, model, result)
        return result


def expected_probe_test_type(
    profile_id: str,
    benchmark_test_types_by_profile: Mapping[str, str],
    *,
    canonicalize: Callable[[str], str] | None = None,
) -> str:
    lookup_id = canonicalize(profile_id) if canonicalize else profile_id
    return benchmark_test_types_by_profile.get(lookup_id, "exact_text")


def validate_probe_response(
    test_type: str,
    expected: str,
    text: str,
    payload: Any | None = None,
    *,
    compression_max_words: int = 60,
) -> tuple[bool, str]:
    return ProbeRegistry(compression_max_words=compression_max_words).validate_response(
        test_type,
        expected,
        text,
        payload,
    )


def extract_message_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def extract_reasoning_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = [item["text"] for item in value if isinstance(item, dict) and isinstance(item.get("text"), str)]
            joined = "\n".join(parts).strip()
            if joined:
                return joined
    return ""


def extract_tool_calls(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    return [item for item in tool_calls if isinstance(item, dict)] if isinstance(tool_calls, list) else []


def has_deliverable_message(payload: Any) -> bool:
    return bool(extract_message_text(payload) or extract_tool_calls(payload))


# One concept, spelled differently per provider: OpenAI/OpenRouter `length`, Mistral `model_length`,
# Anthropic `max_tokens`, Gemini's native API `MAX_TOKENS`. Matched case- and space-insensitively
# because gateway providers pass their upstream's spelling through unmapped.
TRUNCATION_FINISH_REASONS = frozenset({"length", "model_length", "max_tokens", "max_output_tokens", "max_completion_tokens"})


def finish_reason_is_truncation(payload: Any) -> bool:
    """True when the first choice stopped because the completion token budget ran out."""
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    return str(choices[0].get("finish_reason") or "").strip().lower() in TRUNCATION_FINISH_REASONS


def tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


_MISSING_FIELD = object()


def json_fields_match(want: dict[str, Any], got: dict[str, Any]) -> tuple[bool, str]:
    for key, want_value in want.items():
        actual = got.get(key, _MISSING_FIELD)
        if want_value is None:
            if actual is not _MISSING_FIELD and actual is not None:
                return False, f"field {key!r}: expected null/absent, got {actual!r}"
            continue
        if actual is _MISSING_FIELD:
            return False, f"field {key!r} missing"
        if isinstance(want_value, bool):
            if actual is not want_value:
                return False, f"field {key!r}: expected boolean {want_value}, got {actual!r}"
        elif isinstance(want_value, (int, float)):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or float(actual) != float(want_value):
                return False, f"field {key!r}: expected number {want_value}, got {actual!r}"
        elif isinstance(want_value, list):
            if not isinstance(actual, list):
                return False, f"field {key!r}: expected array, got {actual!r}"
            if {str(x).strip().lower() for x in actual} != {str(x).strip().lower() for x in want_value}:
                return False, f"field {key!r}: expected items {want_value}, got {actual}"
        else:
            if str(actual).strip().lower() != str(want_value).strip().lower():
                return False, f"field {key!r}: expected {want_value!r}, got {actual!r}"
    return True, "fields match"
