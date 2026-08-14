from __future__ import annotations

import copy
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from ficelle import __version__
from ficelle.build_identity import package_build_identity
from ficelle.failures import (
    NON_RETRYABLE_FAILURE_REASONS,
    classify_failure,
    cooldown_policy_for_reason,
    provider_error_codes,
    status_for_error_codes,
)
from ficelle.json_store import atomic_write_json as write_atomic_json
from ficelle.json_store import ensure_private_dir, open_private_append
from ficelle.probe_lock import (
    SYNTHETIC_CASE_ID_HEADER,
    SYNTHETIC_CASE_ID_PATTERN,
    file_lock,
    probe_lock_path,
    synthetic_health_lock_path,
)
from ficelle.redaction import redact_sensitive_json, sanitize_error_detail
from ficelle.use_cases.capability_discovery import provider_probe_interval_seconds
from ficelle.failures import capacity_dimension
from ficelle.use_cases.chat_completion import (
    SELECTION_REFUSAL_ERROR_TYPES,
    evaluate_streaming_result,
    request_deadline,
)
from ficelle.use_cases.cooldowns import cooldown_block_scope
from ficelle.use_cases.synthetic_health import (
    CORPUS_VERSION,
    SCHEMA_VERSION,
    SyntheticHealthCase,
    aggregate_case_status,
    build_identity_match,
    corpus_hash,
    deterministic_failure_scenarios,
    expected_policy_scope,
    expected_policy_sections as expected_policy_sections_for_policy,
    request_shape,
    request_template_hash,
    stable_hash,
    validate_unique_case_ids,
)


MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_PREVIEW_CHARS = 1_000
DEFAULT_TIMEOUTS = {"fast": 45.0, "tools": 120.0, "long": 240.0, "media": 180.0}
DEFAULT_MAX_DURATION_SECONDS = 21_600.0
DEFAULT_RETENTION_RUNS = 13
DEFAULT_RETENTION_DAYS = 120.0
HEALTH_RUNS_DIRNAME = "health-runs"
SCHEDULE_LABEL = "com.ficelle.synthetic-health"
CORE_POST_SWEEP_PROFILES = frozenset(
    {
        "ficelle/auto-fast",
        "ficelle/auto-json",
        "ficelle/auto-tools",
        "ficelle/auto-orchestrator",
    }
)
SECRET_PATTERN = re.compile(
    r"(?i)(authorization\s*:|bearer\s+[a-z0-9._-]{12,}|api[_-]?key\s*[=:]\s*[^\s,}]+|sk-[a-z0-9_-]{12,})"
)


class SyntheticHealthError(RuntimeError):
    pass


class SyntheticHealthBusy(SyntheticHealthError):
    pass


class SyntheticClientTimeout(SyntheticHealthError):
    """The harness itself abandoned the request at its wall-clock limit — a client
    timeout for fallback-category purposes, unlike other harness errors."""


# Fallback legitimacy verdicts (L2-R5). Emitted and counted through these constants only:
# a misspelled free literal would silently land in the "unknown" bucket.
FALLBACK_LEGITIMATE = "legitimate"
FALLBACK_ILLEGITIMATE_DECISION = "illegitimate_decision"
FALLBACK_STATE_SCOPE_MISMATCH = "state_scope_mismatch"
FALLBACK_CORRELATION_MISSING = "correlation_missing"
FALLBACK_NOT_APPLICABLE_CLIENT_TIMEOUT = "not_applicable_client_timeout"
FALLBACK_UNKNOWN = "unknown"
FALLBACK_FAILURE_CATEGORIES = (
    FALLBACK_ILLEGITIMATE_DECISION,
    FALLBACK_STATE_SCOPE_MISMATCH,
    FALLBACK_CORRELATION_MISSING,
    FALLBACK_NOT_APPLICABLE_CLIENT_TIMEOUT,
    FALLBACK_UNKNOWN,
)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run: Path
    cases: Path
    catalog_snapshot: Path
    state_before: Path
    state_after: Path
    summary_json: Path
    summary_markdown: Path
    catalog_after: Path
    catalog_transitions: Path

    @classmethod
    def for_run(cls, ficelle_home: Path, run_id: str) -> "RunPaths":
        root = ficelle_home / HEALTH_RUNS_DIRNAME / run_id
        return cls(
            root=root,
            run=root / "run.json",
            cases=root / "cases.jsonl",
            catalog_snapshot=root / "catalog-snapshot.json",
            state_before=root / "state-before.json",
            state_after=root / "state-after.json",
            summary_json=root / "summary.json",
            summary_markdown=root / "summary.md",
            catalog_after=root / "catalog-after.json",
            catalog_transitions=root / "catalog-transitions.jsonl",
        )


@dataclass(frozen=True)
class HttpEvidence:
    status_code: int | None
    content_type: str
    headers: dict[str, str]
    payload: Any
    text: str
    latency_seconds: float
    stream_complete: bool
    stream_error: dict[str, Any] | None
    transport_error: str | None
    tool_calls: list[dict[str, Any]]
    # Typed at the catch site (requests.Timeout or the harness's own wall-clock cut-off),
    # never re-derived by sniffing the error text (L2-R5).
    timed_out: bool = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def atomic_write_json(path: Path, value: Any) -> None:
    write_atomic_json(path, redact_sensitive_json(value))


def atomic_write_text(path: Path, value: str) -> None:
    ensure_private_dir(path.parent)
    descriptor, raw_path = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    temporary = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, value: Any) -> None:
    with open_private_append(path) as handle:
        handle.write(json.dumps(redact_sensitive_json(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return copy.deepcopy(default)


def read_case_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def latest_case_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_case_journal(path):
        if row.get("event") == "outcome" and row.get("case_id"):
            latest[str(row["case_id"])] = row
    return latest


def sanitize_preview(value: Any, limit: int = MAX_PREVIEW_CHARS) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(redact_sensitive_json(value), ensure_ascii=False, sort_keys=True)
    else:
        text = str(value or "")
    return sanitize_error_detail(text, limit) or ""


def response_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    return {
        "type": "object",
        "keys": sorted(str(key) for key in payload)[:30],
        "choice_count": len(choices),
        "model": sanitize_error_detail(payload.get("model"), 250),
        "finish_reason": sanitize_error_detail(first.get("finish_reason"), 80),
        "has_content": bool(message.get("content")),
        "tool_call_count": len(message.get("tool_calls") or []) if isinstance(message.get("tool_calls"), list) else 0,
        "has_error": bool(payload.get("error")),
    }


def _header_subset(headers: Any) -> dict[str, str]:
    allow = {
        "content-type",
        "x-ficelle-request-id",
        "x-ficelle-profile",
        "x-ficelle-attempts",
        "x-ficelle-fallback",
        "x-ficelle-model",
        "x-ficelle-upstream",
        "x-ficelle-source",
        "x-ficelle-compression",
    }
    return {str(key).lower(): str(value)[:500] for key, value in headers.items() if str(key).lower() in allow}


def extract_message(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def extract_text(payload: Any) -> str:
    message = extract_message(payload)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def normalized_tool_calls(payload: Any) -> list[dict[str, Any]]:
    calls = extract_message(payload).get("tool_calls")
    if not isinstance(calls, list):
        return []
    safe: list[dict[str, Any]] = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        call_id = call.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        safe.append(
            {
                "id": call_id if isinstance(call_id, str) else None,
                "type": "function",
                "function": {
                    "name": name if isinstance(name, str) else None,
                    "arguments": arguments if isinstance(arguments, str) else "",
                },
            }
        )
    return safe


def _assemble_stream(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    content_parts: dict[int, list[str]] = defaultdict(list)
    calls: dict[tuple[int, int], dict[str, Any]] = {}
    finish_reasons: dict[int, str | None] = {}
    model: str | None = None
    for event in events:
        if model is None and event.get("model"):
            model = str(event["model"])
        choices = event.get("choices") if isinstance(event.get("choices"), list) else []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_index = choice.get("index") if type(choice.get("index")) is int else 0
            if choice.get("finish_reason") is not None:
                finish_reasons[choice_index] = str(choice.get("finish_reason"))
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if isinstance(delta.get("content"), str):
                content_parts[choice_index].append(delta["content"])
            for fragment in delta.get("tool_calls") or []:
                if not isinstance(fragment, dict):
                    continue
                tool_index = fragment.get("index") if type(fragment.get("index")) is int else 0
                current = calls.setdefault(
                    (choice_index, tool_index),
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if fragment.get("id"):
                    current["id"] += str(fragment["id"])
                function = fragment.get("function") if isinstance(fragment.get("function"), dict) else {}
                if function.get("name"):
                    current["function"]["name"] += str(function["name"])
                if function.get("arguments"):
                    current["function"]["arguments"] += str(function["arguments"])
    choice_index_set = set(content_parts) | set(finish_reasons) | {key[0] for key in calls}
    choice_indexes = sorted(choice_index_set or {0})
    assembled_choices: list[dict[str, Any]] = []
    for choice_index in choice_indexes:
        tool_calls = [
            calls[key]
            for key in sorted(calls)
            if key[0] == choice_index
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts.get(choice_index, [])),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        assembled_choices.append(
            {
                "index": choice_index,
                "message": message,
                "finish_reason": finish_reasons.get(choice_index),
            }
        )
    payload = {"model": model, "choices": assembled_choices}
    return payload, normalized_tool_calls(payload)


class LoopbackChatClient:
    def __init__(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.headers = dict(headers or {})

    def complete(
        self,
        body: dict[str, Any],
        *,
        timeout_seconds: float,
        extra_headers: dict[str, str] | None = None,
    ) -> HttpEvidence:
        started = time.monotonic()
        stream = bool(body.get("stream"))
        response: requests.Response | None = None
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Content-Type": "application/json", **self.headers, **(extra_headers or {})},
                timeout=timeout_seconds,
                stream=True,
            )
            headers = _header_subset(response.headers)
            content_type = str(response.headers.get("Content-Type") or "")
            if stream and response.status_code < 300:
                events: list[dict[str, Any]] = []
                raw_size = 0
                done = False
                stream_error: dict[str, Any] | None = None
                for raw_line in response.iter_lines(decode_unicode=False):
                    if time.monotonic() - started > timeout_seconds:
                        raise SyntheticClientTimeout("stream response exceeded case wall-clock limit")
                    raw_size += len(raw_line)
                    if raw_size > MAX_RESPONSE_BYTES:
                        raise SyntheticHealthError("stream response exceeded artifact safety bound")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        stream_error = {"type": "invalid_sse_frame", "preview": sanitize_preview(line)}
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        continue
                    try:
                        event = json.loads(data)
                    except ValueError:
                        stream_error = {"type": "invalid_sse_json", "preview": sanitize_preview(data)}
                        continue
                    if isinstance(event, dict):
                        if event.get("error"):
                            stream_error = redact_sensitive_json(event.get("error"))
                        events.append(event)
                payload, tool_calls = _assemble_stream(events)
                response.close()
                return HttpEvidence(
                    status_code=response.status_code,
                    content_type=content_type,
                    headers=headers,
                    payload=payload,
                    text=extract_text(payload),
                    latency_seconds=round(time.monotonic() - started, 4),
                    stream_complete=done and stream_error is None,
                    stream_error=stream_error,
                    transport_error=None,
                    tool_calls=tool_calls,
                )
            chunks: list[bytes] = []
            content_size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if time.monotonic() - started > timeout_seconds:
                    raise SyntheticHealthError("response exceeded case wall-clock limit")
                if not chunk:
                    continue
                content_size += len(chunk)
                if content_size > MAX_RESPONSE_BYTES:
                    raise SyntheticHealthError("response exceeded artifact safety bound")
                chunks.append(chunk)
            content = b"".join(chunks)
            raw_text = content.decode(response.encoding or "utf-8", errors="replace")
            try:
                payload: Any = json.loads(raw_text)
            except ValueError:
                payload = None
            response.close()
            return HttpEvidence(
                status_code=response.status_code,
                content_type=content_type,
                headers=headers,
                payload=payload,
                text=extract_text(payload) if isinstance(payload, dict) else raw_text,
                latency_seconds=round(time.monotonic() - started, 4),
                stream_complete=not stream,
                stream_error=None,
                transport_error=None,
                tool_calls=normalized_tool_calls(payload),
            )
        except (requests.RequestException, SyntheticHealthError) as exc:
            if response is not None:
                response.close()
            return HttpEvidence(
                status_code=None,
                content_type="",
                headers={},
                payload=None,
                text="",
                latency_seconds=round(time.monotonic() - started, 4),
                stream_complete=False,
                stream_error=None,
                transport_error=sanitize_error_detail(f"{type(exc).__name__}: {exc}", 500),
                tool_calls=[],
                timed_out=isinstance(exc, (requests.exceptions.Timeout, SyntheticClientTimeout)),
            )


def find_route_rows(route_log_path: Path, value: str, *, field: str = "request_id") -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    paths = [route_log_path] + [route_log_path.with_name(f"{route_log_path.name}.{index}") for index in range(1, 4)]
    for path in paths:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if value not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get(field) == value:
                matches.append(redact_sensitive_json(row))
    return matches


def wait_for_route_rows(
    route_log_path: Path,
    value: str,
    timeout_seconds: float = 2.0,
    *,
    field: str = "request_id",
) -> list[dict[str, Any]]:
    # Each poll rereads the route logs in full, and a late-route correlation wait can span
    # the router's whole request deadline (minutes): back the interval off after the first
    # second so a long wait costs a read per second, not twenty.
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    while True:
        rows = find_route_rows(route_log_path, value, field=field)
        if rows or time.monotonic() >= deadline:
            return rows
        time.sleep(0.05 if time.monotonic() - started < 1.0 else 1.0)


def synthetic_case_correlation_id(case_id: str) -> str:
    """A router-acceptable correlation id for this case (L2-R3): the case id itself when it
    already fits the strict header shape, else a stable hash of it. The shape is the shared
    contract in probe_lock — the same pattern the router's gate enforces."""
    if SYNTHETIC_CASE_ID_PATTERN.match(case_id):
        return case_id
    return stable_hash(case_id)[:40]


def late_route_correlation_wait_seconds(router_config: dict[str, Any] | None, client_timeout_seconds: float) -> float:
    """How long a timed-out synthetic client waits for the router to finish its request.

    Bounded by the router's own request deadline (L2-R2) through the router's own parser
    (`request_deadline`), not by guesswork: the elapsed client timeout already counts
    against it, plus a small margin for telemetry writes. A router with the deadline
    disabled has no bound to wait for — fall back to the shipped default."""
    deadline_instant = request_deadline(router_config, 0.0)
    router_deadline = deadline_instant if deadline_instant is not None else 300.0
    return max(0.0, router_deadline - client_timeout_seconds) + 5.0


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized[:70] or "target") + "-" + stable_hash(value)[:8]


def _marker(run_id: str, case_id: str) -> str:
    return "ficelle-health-" + stable_hash({"run": run_id, "case": case_id})[:16]


def _weather_tools() -> list[dict[str, Any]]:
    """The portable base tool fixture (L3-R3): the conservative common JSON Schema subset
    only — flat string/integer/number/boolean properties and one array of strings. Every
    advanced dialect feature (nullable type arrays, anyOf, nested required objects, enums,
    additionalProperties: false, forced tool choice) lives in explicit T8 dialect cases so
    a dialect failure can never refute basic tool calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_route_weather",
                "description": "Fetch deterministic weather needed to plan a specific route.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "units": {"type": "string", "description": "metric or imperial"},
                        "include_alerts": {"type": "boolean"},
                        "landmarks": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["city", "days", "units", "include_alerts", "landmarks"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_weather_guides",
                "description": "Search general weather travel guides, not live route weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_weather_preference",
                "description": "Save a user's preferred weather units.",
                "parameters": {
                    "type": "object",
                    "properties": {"units": {"type": "string", "description": "metric or imperial"}},
                    "required": ["units"],
                },
            },
        },
    ]


EXPECTED_WEATHER_ARGUMENTS = {
    "city": "Madrid",
    "days": 3,
    "units": "metric",
    "include_alerts": True,
    "landmarks": ["Retiro", "Sol"],
}


# T8 schema-dialect fixtures (L3-R3): one feature per case, each with its own tool schema,
# deterministic prompt fragment, and expected arguments. A failure here updates the
# per-feature dialect matrix; it never counts against basic T1/T2 competence.
def _dialect_tool(name: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Record structured trip preferences exactly as instructed.",
                "parameters": parameters,
            },
        }
    ]


SCHEMA_DIALECT_FEATURES: dict[str, dict[str, Any]] = {
    "nullable_type_array": {
        "tools": _dialect_tool(
            "set_locale_nullable",
            {
                "type": "object",
                "properties": {"locale": {"type": ["string", "null"]}, "city": {"type": "string"}},
                "required": ["locale", "city"],
            },
        ),
        "prompt": "Call set_locale_nullable with city Madrid and locale set to JSON null (not the string \"null\").",
        "expected_arguments": {"locale": None, "city": "Madrid"},
        "tool_choice": "auto",
    },
    "anyof_nullability": {
        "tools": _dialect_tool(
            "set_locale_anyof",
            {
                "type": "object",
                "properties": {
                    "locale": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "city": {"type": "string"},
                },
                "required": ["locale", "city"],
            },
        ),
        "prompt": "Call set_locale_anyof with city Madrid and locale set to JSON null (not the string \"null\").",
        "expected_arguments": {"locale": None, "city": "Madrid"},
        "tool_choice": "auto",
    },
    "nested_objects_arrays": {
        "tools": _dialect_tool(
            "set_route_plan",
            {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "stops": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}, "minutes": {"type": "integer"}},
                                    "required": ["name", "minutes"],
                                },
                            },
                        },
                        "required": ["city", "stops"],
                    }
                },
                "required": ["route"],
            },
        ),
        "prompt": "Call set_route_plan for city Madrid with exactly two stops: Retiro for 30 minutes then Sol for 15 minutes.",
        "expected_arguments": {
            "route": {"city": "Madrid", "stops": [{"name": "Retiro", "minutes": 30}, {"name": "Sol", "minutes": 15}]}
        },
        "tool_choice": "auto",
    },
    "additional_properties_false": {
        "tools": _dialect_tool(
            "set_units_strict",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"units": {"type": "string"}},
                "required": ["units"],
            },
        ),
        "prompt": "Call set_units_strict with units metric and no other argument.",
        "expected_arguments": {"units": "metric"},
        "tool_choice": "auto",
    },
    "enums": {
        "tools": _dialect_tool(
            "set_units_enum",
            {
                "type": "object",
                "properties": {"units": {"type": "string", "enum": ["metric", "imperial"]}},
                "required": ["units"],
            },
        ),
        "prompt": "Call set_units_enum with units metric.",
        "expected_arguments": {"units": "metric"},
        "tool_choice": "auto",
    },
    "forced_tool_choice": {
        "tools": _dialect_tool(
            "set_city_forced",
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
        "prompt": "The city is Madrid.",
        "expected_arguments": {"city": "Madrid"},
        "tool_choice": {"type": "function", "function": {"name": "set_city_forced"}},
    },
}


def dialect_case_body(model_id: str, feature: str, *, marker: str) -> dict[str, Any]:
    fixture = SCHEMA_DIALECT_FEATURES[feature]
    return {
        "model": model_id,
        "messages": [
            {"role": "user", "content": f"{fixture['prompt']} Correlation marker: {marker}."}
        ],
        "tools": [dict(tool) for tool in fixture["tools"]],
        "tool_choice": fixture["tool_choice"],
        "temperature": 0,
        "max_completion_tokens": 300,
        "stream": False,
    }


def tool_choice_body(model_id: str, *, stream: bool, marker: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the correct tool, not prose, to fetch route weather for Madrid for exactly "
                    "3 days in metric units. Include alerts and use landmarks Retiro and Sol. "
                    f"Correlation marker: {marker}."
                ),
            }
        ],
        "tools": _weather_tools(),
        "tool_choice": "auto",
        "temperature": 0,
        "max_completion_tokens": 500,
        "stream": stream,
    }


def tool_continuation_template(model_id: str, *, marker: str, dependency_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the correct tool, not prose, to fetch route weather for Madrid for exactly "
                    "3 days in metric units. Include alerts and use landmarks Retiro and Sol."
                ),
            },
            {"role": "assistant", "content": None, "tool_calls": "__DEPENDENCY_TOOL_CALLS__"},
            {
                "role": "tool",
                "tool_call_id": "__DEPENDENCY_TOOL_CALL_ID__",
                "content": json.dumps({"temperature_c": 24, "alert": "wind", "days": 3}),
            },
            {
                "role": "user",
                "content": f"Answer exactly {marker} after using the tool result. Do not call another tool.",
            },
        ],
        "tools": _weather_tools(),
        "tool_choice": "auto",
        "temperature": 0,
        "max_completion_tokens": 200,
        "stream": False,
        "__dependency_case_id": dependency_id,
    }


def parallel_tool_body(model_id: str, marker: str) -> dict[str, Any]:
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_temperature",
            "description": "Return the deterministic current temperature for one city.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": f"Call lookup_temperature once for Madrid and once for Paris in parallel. Do not answer in prose. {marker}",
            }
        ],
        "tools": [tool],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "temperature": 0,
        "max_completion_tokens": 300,
    }


def _case_timeout(profile_id: str, test_type: str) -> str:
    if test_type in {"long_context", "compression"}:
        return "long"
    if test_type in {"vision", "audio", "video"}:
        return "media"
    if "tool" in test_type or profile_id.endswith(("tools", "orchestrator")):
        return "tools"
    return "fast"


def runtime_model_block_reason(router: Any, model: dict[str, Any], state: dict[str, Any]) -> str | None:
    try:
        quarantine = router.model_quarantine(model, state)
        if isinstance(quarantine, dict) and quarantine:
            return f"quarantine:{quarantine.get('reason') or 'active'}"
        blocked, reason = router.model_on_cooldown(model, state)
        if blocked:
            return f"cooldown:{reason or 'active'}"
    except Exception as exc:
        return f"runtime_block_check_error:{type(exc).__name__}"
    return None


# Every scope an execution-eligibility record may carry (L1-R3). A blocked planned case is
# skipped with exactly one of these; it is never sent to the loopback API merely to
# rediscover `no_available_model`.
ELIGIBILITY_SCOPES = (
    "model",
    "provider",
    "quota_pool",
    "shared_account",
    "catalog",
    "credential",
    "profile",
    "none",
)


def _matching_quota_cooldown_key(router: Any, model: dict[str, Any], state: dict[str, Any]) -> str | None:
    quota_cooldowns = state.get("quota_cooldowns") if isinstance(state.get("quota_cooldowns"), dict) else {}
    for key, raw in quota_cooldowns.items():
        if not isinstance(raw, dict):
            continue
        try:
            if not router.quota_cooldown_matches_model(str(key), model):
                continue
        except Exception:
            continue
        if float(raw.get("until") or 0) > 0:
            return str(key)
    return None


def execution_eligibility(
    router: Any,
    model: dict[str, Any] | None,
    state: dict[str, Any],
    provider_cfg: dict[str, Any] | None,
    *,
    catalog_generation: str | None,
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured execution-eligibility record for one direct case (L1-R3).

    Resolves, in order of ownership: active-catalog membership, provider enablement,
    credential/invokability, then the runtime quarantine/cooldown probe shared with the
    inventory (`runtime_model_block_reason`), scoped via the same decoders the router
    uses. ``eligible`` with scope ``none`` means the case may be sent upstream.
    """

    def record(eligible: bool, scope: str, reason: str | None) -> dict[str, Any]:
        return {
            "eligible": eligible,
            "scope": scope,
            "reason": reason,
            "catalog_generation": catalog_generation,
            "state_observed_at": now_iso(),
        }

    if model is None:
        return record(False, "catalog", "catalog_model_missing_during_run")
    if isinstance(provider_cfg, dict) and provider_cfg.get("enabled", True) is False:
        return record(False, "provider", "provider_disabled")
    if runtime_config is not None:
        try:
            access = router.provider_access_for(str(model.get("source") or ""), runtime_config)
        except Exception as exc:
            return record(False, "credential", f"credential_check_error:{type(exc).__name__}")
        if not access.can_invoke:
            return record(False, "credential", str(access.reason or "not_invokable"))
    if not model.get("invokable"):
        return record(False, "credential", str(model.get("auth_reason") or "not_invokable"))
    block_reason = runtime_model_block_reason(router, model, state)
    if block_reason is None:
        return record(True, "none", None)
    if block_reason.startswith("cooldown:"):
        scope = cooldown_block_scope(block_reason.removeprefix("cooldown:"))
        if scope == "provider":
            return record(False, "provider", block_reason)
        if scope == "quota":
            quota_key = _matching_quota_cooldown_key(router, model, state)
            try:
                quota_scope = router.quota_cooldown_scope_from_key(quota_key) if quota_key else "provider"
            except Exception:
                quota_scope = "provider"
            return record(
                False,
                "shared_account" if quota_scope == "shared_account" else "quota_pool",
                block_reason,
            )
    # quarantine:* and runtime_block_check_error:* are model-scoped, like the inventory.
    return record(False, "model", block_reason)


def build_inventory(router: Any, config: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    providers_cfg = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    catalog_providers = catalog.get("providers") if isinstance(catalog.get("providers"), dict) else {}
    catalog_model_counts = Counter(
        str(model.get("source") or "")
        for model in catalog.get("models", [])
        if isinstance(model, dict) and model.get("source")
    )
    provider_sources = sorted(
        set(str(source) for source in providers_cfg)
        | {
            str(model.get("source"))
            for model in catalog.get("models", [])
            if isinstance(model, dict) and model.get("source")
        }
    )
    providers: list[dict[str, Any]] = []
    access_by_source: dict[str, Any] = {}
    invokable_by_source: dict[str, bool] = {}
    for source in provider_sources:
        provider_cfg = providers_cfg.get(source) if isinstance(providers_cfg, dict) else {}
        catalog_provider = catalog_providers.get(source) if isinstance(catalog_providers.get(source), dict) else {}
        enabled = not isinstance(provider_cfg, dict) or provider_cfg.get("enabled", True) is not False
        try:
            access = router.provider_access_for(source, config)
            access_by_source[source] = access
            can_invoke = enabled and access.can_invoke
            reason = access.reason if enabled else "disabled_by_config"
            credential_present = bool(getattr(access, "key", None)) or bool(getattr(access, "auth_status_invokable", False))
            base_url_present = bool(getattr(access, "base_url", None))
        except Exception as exc:
            can_invoke = False
            reason = sanitize_error_detail(f"access_error:{type(exc).__name__}", 120)
            credential_present = False
            base_url_present = False
        invokable_by_source[source] = can_invoke
        providers.append(
            {
                "source": source,
                "enabled": enabled,
                "credential_present": credential_present,
                "base_url_present": base_url_present,
                "can_invoke": can_invoke,
                "access_reason": reason,
                "catalog_model_count": catalog_model_counts.get(source, 0),
                "catalog_accepted_count": catalog_provider.get("accepted_count"),
                "catalog_error": sanitize_error_detail(catalog_provider.get("error"), 500),
            }
        )

    state = router.load_runtime_state()
    models: list[dict[str, Any]] = []
    for model in catalog.get("models", []):
        if not isinstance(model, dict):
            continue
        source = str(model.get("source") or "")
        access = access_by_source.get(source)
        free_access = router.normalized_free_access(model)
        eligible = bool(free_access.get("eligible"))
        can_invoke = bool(access and invokable_by_source.get(source))
        runtime_block_reason = runtime_model_block_reason(router, model, state)
        models.append(
            {
                "id": str(model.get("id") or ""),
                "upstream_id": str(model.get("upstream_id") or ""),
                "source": source,
                "strict_zero_eligible": eligible,
                "can_invoke": can_invoke,
                "runtime_block_reason": runtime_block_reason,
                "free_access": redact_sensitive_json(free_access),
                "supported_parameters": [str(item) for item in model.get("supported_parameters", []) if item],
                "architecture": redact_sensitive_json(model.get("architecture") or {}),
                "context_length": model.get("context_length"),
                "response_identity_mode": (
                    "provider_router_alias"
                    if str(model.get("tokenizer") or "").strip().lower() == "router"
                    else "catalog_identity"
                ),
            }
        )
    normalized_profiles = router.normalized_virtual_profiles(config)
    profile_candidates: dict[str, list[str]] = {}
    for profile_id in sorted(normalized_profiles):
        try:
            selection = router.select_models_result_from_state(profile_id, catalog, config, state, purpose="route")
            selected = selection.as_legacy_models()
            profile_candidates[profile_id] = [str(model.get("id") or "") for model in selected if isinstance(model, dict)]
        except Exception:
            profile_candidates[profile_id] = []
    return {
        "generated_at": now_iso(),
        "catalog_generated_at": catalog.get("generated_at"),
        "catalog_stale": bool(catalog.get("stale")),
        "providers": providers,
        "models": models,
        "virtual_profiles": sorted(normalized_profiles),
        "virtual_profile_definitions": redact_sensitive_json(normalized_profiles),
        "virtual_profile_candidates": profile_candidates,
    }


def _benchmark_case(
    router: Any,
    config: dict[str, Any],
    *,
    run_id: str,
    phase: int,
    profile_id: str,
    target_kind: str,
    target_id: str,
    source: str | None = None,
    upstream_id: str | None = None,
    prefix: str,
) -> SyntheticHealthCase:
    case_id = f"p{phase}.{prefix}.{_slug(target_id)}.{_slug(profile_id)}"
    try:
        body, test_type, expected = router.benchmark_body(profile_id, config)
    except Exception as exc:
        return SyntheticHealthCase(
            id=case_id,
            phase=phase,
            lane="live",
            target_kind=target_kind,
            target_id=target_id,
            source=source,
            model_id=target_id if target_kind == "direct-model" else None,
            upstream_id=upstream_id,
            validator="benchmark",
            timeout_class="media",
            tags=("capability", profile_id),
            expected={"profile": profile_id},
            skip_reason=sanitize_error_detail(f"fixture_unavailable:{type(exc).__name__}:{exc}", 250),
        )
    body = copy.deepcopy(body)
    body["model"] = target_id
    body["stream"] = False
    return SyntheticHealthCase(
        id=case_id,
        phase=phase,
        lane="live",
        target_kind=target_kind,
        target_id=target_id,
        source=source,
        model_id=target_id if target_kind == "direct-model" else None,
        upstream_id=upstream_id,
        validator="benchmark",
        timeout_class=_case_timeout(profile_id, test_type),
        body=body,
        expected={"profile": profile_id, "test_type": test_type, "value": expected},
        tags=("capability", test_type),
    )


def _tool_cases(
    *,
    run_id: str,
    phase: int,
    target_kind: str,
    target_id: str,
    source: str | None,
    upstream_id: str | None,
    parallel_claimed: bool,
) -> list[SyntheticHealthCase]:
    base = f"p{phase}.tools.{_slug(target_id)}"
    marker = _marker(run_id, base)
    initial_id = f"{base}.choice"
    continuation_id = f"{base}.continuation"
    common = {
        "phase": phase,
        "lane": "live",
        "target_kind": target_kind,
        "target_id": target_id,
        "source": source,
        "model_id": target_id if target_kind == "direct-model" else None,
        "upstream_id": upstream_id,
        "timeout_class": "tools",
    }
    cases = [
        SyntheticHealthCase(
            id=initial_id,
            validator="tool-choice",
            body=tool_choice_body(target_id, stream=False, marker=marker),
            expected={"tool_name": "fetch_route_weather", "arguments": EXPECTED_WEATHER_ARGUMENTS},
            tags=("tools", "t1", "t2", "non-streaming"),
            **common,
        ),
        SyntheticHealthCase(
            id=continuation_id,
            validator="tool-continuation",
            body=tool_continuation_template(target_id, marker=marker, dependency_id=initial_id),
            expected={"marker": marker, "dependency_case_id": initial_id, "grounding_facts": ["24", "wind"]},
            tags=("tools", "t3", "t4", "multi-turn"),
            **common,
        ),
        SyntheticHealthCase(
            id=f"{base}.streaming",
            validator="tool-choice",
            body=tool_choice_body(target_id, stream=True, marker=marker),
            expected={"tool_name": "fetch_route_weather", "arguments": EXPECTED_WEATHER_ARGUMENTS},
            tags=("tools", "t5", "streaming"),
            **common,
        ),
    ]
    cases.append(
        SyntheticHealthCase(
            id=f"{base}.parallel",
            validator="parallel-tools",
            body=parallel_tool_body(target_id, marker),
            expected={"tool_name": "lookup_temperature", "cities": ["Madrid", "Paris"]},
            tags=("tools", "t6", "parallel"),
            skip_reason=None if parallel_claimed else "parallel_tools_not_claimed",
            **common,
        )
    )
    return cases


def build_corpus(
    router: Any,
    config: dict[str, Any],
    catalog: dict[str, Any],
    inventory: dict[str, Any],
    *,
    run_id: str,
) -> list[SyntheticHealthCase]:
    cases: list[SyntheticHealthCase] = [
        SyntheticHealthCase(
            id="p0.preflight.loopback-service",
            phase=0,
            lane="live",
            target_kind="service",
            target_id="loopback",
            validator="service-preflight",
            timeout_class="fast",
            tags=("preflight",),
        )
    ]
    for scenario in deterministic_failure_scenarios():
        cases.append(
            SyntheticHealthCase(
                id=f"p1.failure.{scenario.id}",
                phase=1,
                lane="deterministic",
                target_kind="failure-contract",
                target_id=scenario.id,
                validator="failure-contract",
                timeout_class="fast",
                expected={
                    "reason": scenario.expected_reason,
                    "retry": scenario.expected_retry,
                    "scope": scenario.expected_scope,
                },
                tags=("failure", "fallback", "cooldown"),
            )
        )
    for stream_id in ("pre-first-byte-retry", "mid-stream-no-retry", "client-disconnect-no-cooldown"):
        cases.append(
            SyntheticHealthCase(
                id=f"p1.stream.{stream_id}",
                phase=1,
                lane="deterministic",
                target_kind="stream-contract",
                target_id=stream_id,
                validator="stream-contract",
                timeout_class="fast",
                expected={},
                tags=("stream", "fallback", "cooldown"),
            )
        )
    for invocation_id in ("timeout", "connection-exception"):
        cases.append(
            SyntheticHealthCase(
                id=f"p1.invocation.{invocation_id}",
                phase=1,
                lane="deterministic",
                target_kind="invocation-contract",
                target_id=invocation_id,
                validator="invocation-contract",
                timeout_class="fast",
                expected={},
                tags=("failure", "fallback", "cooldown"),
            )
        )
    for success_id in ("embedded-error", "invalid-json", "empty-message", "truncated-before-content"):
        cases.append(
            SyntheticHealthCase(
                id=f"p1.success.{success_id}",
                phase=1,
                lane="deterministic",
                target_kind="success-contract",
                target_id=success_id,
                validator="success-contract",
                timeout_class="fast",
                expected={},
                tags=("deceptive-200", "fallback", "cooldown"),
            )
        )
    for replay_id in ("orphan-tool-result", "blank-function-name", "malformed-arguments", "cross-model-replay"):
        cases.append(
            SyntheticHealthCase(
                id=f"p1.tools.{replay_id}",
                phase=1,
                lane="deterministic",
                target_kind="tool-contract",
                target_id=replay_id,
                validator="tool-contract",
                timeout_class="fast",
                expected={},
                tags=("tools", "t4", "t7"),
            )
        )

    profiles = [str(profile) for profile in inventory.get("virtual_profiles", [])]
    for phase, label in ((2, "pre"), (4, "post")):
        for profile_id in profiles:
            cases.append(
                _benchmark_case(
                    router,
                    config,
                    run_id=run_id,
                    phase=phase,
                    profile_id=profile_id,
                    target_kind="virtual-profile",
                    target_id=profile_id,
                    prefix=f"virtual-{label}",
                )
            )
            if profile_id.endswith(("auto-tools", "auto-orchestrator")):
                cases.extend(
                    _tool_cases(
                        run_id=run_id,
                        phase=phase,
                        target_kind="virtual-profile",
                        target_id=profile_id,
                        source=None,
                        upstream_id=None,
                        parallel_claimed=True,
                    )
                )
            if profile_id.endswith("auto-fast"):
                fast_marker = _marker(run_id, f"p{phase}.virtual-fast-stream")
                cases.append(
                    SyntheticHealthCase(
                        id=f"p{phase}.virtual-fast.stream",
                        phase=phase,
                        lane="live",
                        target_kind="virtual-profile",
                        target_id=profile_id,
                        validator="exact-text",
                        timeout_class="fast",
                        body={
                            "model": profile_id,
                            "messages": [{"role": "user", "content": f"Reply exactly: {fast_marker}"}],
                            "temperature": 0,
                            "max_completion_tokens": 80,
                            "stream": True,
                        },
                        expected={"marker": fast_marker},
                        tags=("core-text", "streaming"),
                    )
                )
                cases.append(
                    SyntheticHealthCase(
                        id=f"p{phase}.virtual-fast.json",
                        phase=phase,
                        lane="live",
                        target_kind="virtual-profile",
                        target_id=profile_id,
                        validator="json-object",
                        timeout_class="fast",
                        body={
                            "model": profile_id,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": 'Return only JSON with keys route="A7", stops=3, active=true.',
                                }
                            ],
                            "temperature": 0,
                            "max_completion_tokens": 100,
                        },
                        expected={"json": {"route": "A7", "stops": 3, "active": True}},
                        tags=("fast", "json-shaped"),
                    )
                )
        if "ficelle/auto-tools" in profiles:
            compatibility_id = f"p{phase}.client-compat.top-k"
            marker = _marker(run_id, compatibility_id)
            cases.append(
                SyntheticHealthCase(
                    id=compatibility_id,
                    phase=phase,
                    lane="live",
                    target_kind="virtual-profile",
                    target_id="ficelle/auto-tools",
                    validator="exact-text",
                    timeout_class="fast",
                    body={
                        "model": "ficelle/auto-tools",
                        "messages": [{"role": "user", "content": f"Reply exactly: {marker}"}],
                        "temperature": 0,
                        "top_k": 1,
                        "max_completion_tokens": 80,
                    },
                    expected={"marker": marker},
                    tags=("client-compat", "t8", "top_k"),
                )
            )

    raw_models = {str(model.get("id") or ""): model for model in catalog.get("models", []) if isinstance(model, dict)}
    profile_configs = router.normalized_virtual_profiles(config)
    runtime_state = router.load_runtime_state()
    for model_row in inventory.get("models", []):
        if not isinstance(model_row, dict) or not model_row.get("strict_zero_eligible") or not model_row.get("can_invoke"):
            continue
        model_id = str(model_row.get("id") or "")
        model = raw_models.get(model_id)
        if model is None:
            continue
        source = str(model_row.get("source") or "")
        upstream_id = str(model_row.get("upstream_id") or "")
        runtime_block_reason = sanitize_error_detail(model_row.get("runtime_block_reason"), 250)
        base = f"p3.model.{_slug(model_id)}"
        marker = _marker(run_id, base)
        for stream in (False, True):
            cases.append(
                SyntheticHealthCase(
                    id=f"{base}.{'stream' if stream else 'text'}",
                    phase=3,
                    lane="live",
                    target_kind="direct-model",
                    target_id=model_id,
                    source=source,
                    model_id=model_id,
                    upstream_id=upstream_id,
                    validator="exact-text",
                    timeout_class="fast",
                    body={
                        "model": model_id,
                        "messages": [{"role": "user", "content": f"Reply exactly: {marker}"}],
                        "temperature": 0,
                        "max_completion_tokens": 80,
                        "stream": stream,
                    },
                    expected={"marker": marker},
                    tags=("core-text", "streaming" if stream else "non-streaming"),
                    skip_reason=runtime_block_reason,
                )
            )
        matching_profiles: list[str] = []
        for profile_id, profile in profile_configs.items():
            try:
                if not router.model_matches_profile_requirements(model, profile):
                    continue
                competence = router.model_route_competence(profile_id, model, runtime_state)
                if competence in {"claimed_default", "failed"}:
                    continue
                # The base text pair already is the representative direct-model test for auto-fast;
                # adding the same exact-text benchmark would spend quota without widening coverage.
                if profile_id.endswith("auto-fast"):
                    continue
                matching_profiles.append(profile_id)
            except Exception:
                continue
        for profile_id in matching_profiles:
            cases.append(
                _benchmark_case(
                    router,
                    config,
                    run_id=run_id,
                    phase=3,
                    profile_id=profile_id,
                    target_kind="direct-model",
                    target_id=model_id,
                    source=source,
                    upstream_id=upstream_id,
                    prefix="capability",
                )
            )
            if runtime_block_reason:
                cases[-1] = replace(cases[-1], skip_reason=runtime_block_reason)
        supports_tools = any(profile.endswith(("auto-tools", "auto-orchestrator")) for profile in matching_profiles)
        params = set(str(item) for item in model_row.get("supported_parameters", []))
        if supports_tools or {"tools", "tool_choice"}.intersection(params):
            generated_tools = _tool_cases(
                run_id=run_id,
                phase=3,
                target_kind="direct-model",
                target_id=model_id,
                source=source,
                upstream_id=upstream_id,
                parallel_claimed="parallel_tool_calls" in params,
            )
            if runtime_block_reason:
                generated_tools = [replace(tool_case, skip_reason=runtime_block_reason) for tool_case in generated_tools]
            cases.extend(generated_tools)
    # T8 schema-dialect matrix (L3-R3): one representative tool-capable model per provider,
    # one case per dialect feature. A dialect failure feeds the per-feature support matrix;
    # it never counts against the model's basic T1/T2 tool competence.
    dialect_representatives: dict[str, dict[str, Any]] = {}
    for model_row in catalog.get("models", []):
        if not isinstance(model_row, dict) or not model_row.get("invokable"):
            continue
        row_source = str(model_row.get("source") or "")
        row_params = set(str(item) for item in model_row.get("supported_parameters", []))
        if not {"tools", "tool_choice"}.intersection(row_params):
            continue
        current = dialect_representatives.get(row_source)
        if current is None or str(model_row.get("id") or "") < str(current.get("id") or ""):
            dialect_representatives[row_source] = model_row
    for row_source in sorted(dialect_representatives):
        model_row = dialect_representatives[row_source]
        model_id = str(model_row.get("id") or "")
        dialect_base = f"p3.dialect.{_slug(model_id)}"
        for feature in SCHEMA_DIALECT_FEATURES:
            dialect_case_id = f"{dialect_base}.{feature}"
            cases.append(
                SyntheticHealthCase(
                    id=dialect_case_id,
                    phase=3,
                    lane="live",
                    target_kind="direct-model",
                    target_id=model_id,
                    source=row_source,
                    model_id=model_id,
                    upstream_id=str(model_row.get("upstream_id") or "") or None,
                    validator="tool-dialect",
                    timeout_class="tools",
                    body=dialect_case_body(model_id, feature, marker=_marker(run_id, dialect_case_id)),
                    expected={
                        "feature": feature,
                        "tool_name": SCHEMA_DIALECT_FEATURES[feature]["tools"][0]["function"]["name"],
                        "arguments": SCHEMA_DIALECT_FEATURES[feature]["expected_arguments"],
                        # The portable base tool case on the same model is this case's
                        # control: a dialect verdict only means something once plain tool
                        # calling is known to work here (L3-R3).
                        "control_case_id": f"p3.tools.{_slug(model_id)}.choice",
                    },
                    tags=("tools", "t8", "dialect", f"dialect:{feature}"),
                )
            )
    cases.sort(key=lambda case: case.phase)
    validate_unique_case_ids(cases)
    return cases


def select_cases_with_dependencies(
    cases: list[SyntheticHealthCase],
    selected_case_ids: set[str],
) -> list[SyntheticHealthCase]:
    expanded = set(selected_case_ids)
    while True:
        # Both kinds of prerequisite travel with a selection: the case a body is replayed
        # from, and the control case a dialect verdict is only meaningful against.
        dependencies = {
            str(case.expected[key])
            for case in cases
            for key in ("dependency_case_id", "control_case_id")
            if case.id in expanded and case.expected.get(key)
        }
        missing = dependencies - expanded
        if not missing:
            break
        expanded.update(missing)
    return [
        case
        for case in cases
        if case.id in expanded or case.validator == "service-preflight"
    ]


def _axis(status: str, diagnostic: str, **evidence: Any) -> dict[str, Any]:
    return {"status": status, "diagnostic": diagnostic, **redact_sensitive_json(evidence)}


def _json_arguments(tool_call: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    raw = function.get("arguments") if isinstance(function, dict) else None
    if not isinstance(raw, str):
        return None, "tool arguments are not a JSON string"
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        return None, f"invalid tool argument JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "tool arguments are not an object"
    return parsed, None


def _router_refusal_error_type(evidence: HttpEvidence) -> str | None:
    """The selection-refusal type when this response is Ficelle's own refusal, else None.

    The recognized set is exported by the emitter (`SELECTION_REFUSAL_ERROR_TYPES` in
    chat_completion.py) so a refusal family added there is recognized here without a
    manual sync. Only those bypass generic HTTP classification — a provider body relayed
    by the router keeps its provider-shaped classification.
    """
    if not isinstance(evidence.payload, dict):
        return None
    headers = {str(key).lower(): value for key, value in (evidence.headers or {}).items()}
    if "x-ficelle-request-id" not in headers:
        return None
    error = evidence.payload.get("error")
    if not isinstance(error, dict):
        return None
    error_type = str(error.get("type") or "")
    return error_type if error_type in SELECTION_REFUSAL_ERROR_TYPES else None


def _first_finish_reason(payload: Any) -> str | None:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    reason = first.get("finish_reason") if isinstance(first, dict) else None
    return str(reason) if reason is not None else None


def _budget_exhausted_axis(case: SyntheticHealthCase, evidence: HttpEvidence) -> dict[str, Any] | None:
    """The inconclusive verdict for a length-exhausted tool case (L3-R5): the output budget
    ran out before any tool call, which neither proves nor refutes tool capability. Never a
    fail — budget exhaustion must not cool the model nor count against competence."""
    if evidence.tool_calls or _first_finish_reason(evidence.payload) != "length":
        return None
    body = case.body or {}
    tested_budget = body.get("max_completion_tokens") or body.get("max_tokens")
    return _axis(
        "not_applicable",
        "inconclusive_budget_exhausted: output budget ran out before any tool call",
        inconclusive="budget_exhausted",
        finish_reason="length",
        tested_budget=tested_budget,
    )


def semantic_verdict(
    router: Any,
    case: SyntheticHealthCase,
    evidence: HttpEvidence,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    model = next(
        (
            row
            for row in catalog.get("models", [])
            if isinstance(row, dict)
            and row.get("source") == case.source
            and (row.get("id") == case.model_id or row.get("upstream_id") == case.upstream_id)
        ),
        None,
    )
    if evidence.status_code is None:
        return _axis("fail", evidence.transport_error or "request did not receive an HTTP response")
    if evidence.status_code >= 300:
        # Ficelle's own refusals carry a structured error.type and its request-id header:
        # read that before any generic HTTP classification (L1-R4), so a local selection
        # refusal is never misfiled as an upstream server_error.
        router_refusal = _router_refusal_error_type(evidence)
        if router_refusal is not None:
            error = evidence.payload.get("error") if isinstance(evidence.payload, dict) else {}
            return _axis(
                "fail",
                f"HTTP {evidence.status_code} router refusal {router_refusal}",
                classified_reason=f"selection:{router_refusal}",
                refusal_scope=(error or {}).get("scope"),
                refusal_block_reason=(error or {}).get("block_reason"),
            )
        body = evidence.payload if evidence.payload is not None else evidence.text
        body_text = json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body
        reason = router.classify_failure(
            evidence.status_code,
            body_text,
            model,
            error_codes=provider_error_codes(body_text),
        )
        # The capacity dimension (L4-R1) is reporting evidence riding alongside the
        # classification, never replacing it: a Groq TPM 413 stays request_too_large and
        # model-scoped, and it never marks the model's context claim unsupported.
        dimension = capacity_dimension(evidence.status_code, body_text)
        extra = {"capacity_dimension": dimension} if dimension else {}
        return _axis("fail", f"HTTP {evidence.status_code} classified as {reason}", classified_reason=reason, **extra)
    if not isinstance(evidence.payload, dict):
        return _axis("fail", "success response is not valid JSON")
    embedded = router.success_error_payload_failure(evidence.payload, model)
    if embedded is not None:
        reason, detail, embedded_status = embedded
        return _axis(
            "fail",
            f"HTTP 2xx contains an embedded error classified as {reason}",
            classified_reason=reason,
            embedded_status=embedded_status,
            detail=sanitize_error_detail(detail, 250),
        )
    if case.stream and not evidence.stream_complete:
        return _axis("fail", "stream ended without a complete [DONE] event", stream_error=evidence.stream_error)

    if case.validator == "exact-text":
        marker = str(case.expected.get("marker") or "")
        actual = evidence.text.strip()
        return _axis(
            "pass" if actual == marker else "fail",
            "exact marker returned" if actual == marker else "exact marker mismatch",
            expected_hash=stable_hash(marker),
            actual_preview=sanitize_preview(actual, 250),
        )
    if case.validator == "json-object":
        raw = evidence.text.strip()
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            return _axis("fail", f"assistant content is not JSON: {exc}", actual_preview=sanitize_preview(raw, 250))
        expected = case.expected.get("json")
        return _axis(
            "pass" if parsed == expected else "fail",
            "typed JSON object returned" if parsed == expected else "typed JSON object mismatch",
            expected=expected,
            actual=parsed,
        )
    if case.validator == "benchmark":
        ok, message = router.validate_benchmark_response(
            str(case.expected.get("test_type") or ""),
            str(case.expected.get("value") or ""),
            evidence.text,
            evidence.payload,
        )
        return _axis("pass" if ok else "fail", sanitize_error_detail(message, 500) or ("benchmark passed" if ok else "benchmark failed"))
    if case.validator in {"tool-choice", "tool-dialect"}:
        inconclusive = _budget_exhausted_axis(case, evidence)
        if inconclusive is not None:
            return inconclusive
        dialect_evidence = (
            {"dialect_feature": str(case.expected.get("feature") or "")}
            if case.validator == "tool-dialect"
            else {}
        )
        if len(evidence.tool_calls) != 1:
            return _axis(
                "fail",
                f"expected one tool call, received {len(evidence.tool_calls)}",
                tool_assertions={"selection": False, "arguments": None},
                **dialect_evidence,
            )
        call = evidence.tool_calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name != case.expected.get("tool_name"):
            return _axis(
                "fail",
                f"wrong tool selected: {sanitize_error_detail(name, 100) or 'none'}",
                tool_assertions={"selection": False, "arguments": None},
                **dialect_evidence,
            )
        arguments, error = _json_arguments(call)
        if error:
            return _axis(
                "fail",
                error,
                tool_assertions={"selection": True, "arguments": False},
                **dialect_evidence,
            )
        expected = case.expected.get("arguments")
        if arguments != expected:
            # T1 and T2 share this upstream request but assert separately (L3-R1): the
            # tool selection already passed, only the typed arguments failed.
            return _axis(
                "fail",
                "tool arguments differ in type, keys, or value",
                expected=expected,
                actual=arguments,
                tool_assertions={"selection": True, "arguments": False},
                **dialect_evidence,
            )
        return _axis(
            "pass",
            "correct tool and typed arguments returned",
            tool_assertions={"selection": True, "arguments": True},
            **dialect_evidence,
        )
    if case.validator == "tool-continuation":
        # Split axes (L3-R2): protocol completion, grounding in the tool result, and exact
        # instruction fidelity are three verdicts. Prose grounded in the tool result passes
        # the lifecycle; the missed exact marker stays visible as instruction evidence.
        marker = str(case.expected.get("marker") or "")
        final_text = evidence.text.strip()
        lowered = evidence.text.lower()
        facts = [str(fact) for fact in case.expected.get("grounding_facts", [])]
        protocol_complete = bool(final_text) and not evidence.tool_calls
        grounded = (bool(marker) and marker in evidence.text) or (
            bool(facts) and all(fact.lower() in lowered for fact in facts)
        )
        instruction_exact = bool(marker) and final_text == marker
        continuation = {
            "protocol_complete": protocol_complete,
            "grounded": grounded,
            "instruction_exact": instruction_exact,
        }
        if protocol_complete and grounded:
            return _axis(
                "pass",
                "tool result continuation completed"
                + ("" if instruction_exact else " (grounded prose; exact instruction not followed)"),
                continuation=continuation,
            )
        return _axis(
            "fail",
            "tool continuation did not produce a grounded final answer",
            continuation=continuation,
        )
    if case.validator == "parallel-tools":
        if len(evidence.tool_calls) != 2:
            return _axis("fail", f"expected two parallel calls, received {len(evidence.tool_calls)}")
        cities: list[str] = []
        for call in evidence.tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get("name") != case.expected.get("tool_name"):
                return _axis("fail", "parallel call used an unexpected tool")
            arguments, error = _json_arguments(call)
            if error or arguments is None or not isinstance(arguments.get("city"), str):
                return _axis("fail", error or "parallel call is missing a typed city")
            cities.append(arguments["city"])
        expected_cities = sorted(str(city) for city in case.expected.get("cities", []))
        return _axis(
            "pass" if sorted(cities) == expected_cities else "fail",
            "parallel calls assembled" if sorted(cities) == expected_cities else "parallel call arguments mismatch",
            cities=sorted(cities),
        )
    return _axis("fail", f"unknown semantic validator: {case.validator}")


def _route_attempts(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    attempts = row.get("attempts") if isinstance(row, dict) else None
    return [attempt for attempt in attempts if isinstance(attempt, dict)] if isinstance(attempts, list) else []


def route_axes(
    case: SyntheticHealthCase,
    evidence: HttpEvidence,
    route_rows: list[dict[str, Any]],
    catalog: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    request_id = evidence.headers.get("x-ficelle-request-id")
    # A timed-out client has no response headers, but the late correlation by synthetic
    # case id (L2-R3) may still have recovered exactly one route row — that row IS the
    # exact attribution, so the audit proceeds on it instead of declaring a correlation
    # gap. correlation_missing remains only when no row could be recovered at all.
    late_correlated = not request_id and len(route_rows) == 1
    if not request_id and not late_correlated:
        # Fallback legitimacy is judged by DECISION, never by correlation availability
        # (L2-R5): a missing request id is a correlation gap, and a client that closed
        # first was never a fallback to judge at all.
        client_timed_out = evidence.timed_out
        category = (
            FALLBACK_NOT_APPLICABLE_CLIENT_TIMEOUT
            if client_timed_out and "sla" in case.tags
            else FALLBACK_CORRELATION_MISSING
        )
        return (
            _axis("not_applicable", "routing cannot be audited without request ID"),
            _axis("fail", "missing X-Ficelle-Request-Id"),
            _axis("fail", "fallback cannot be audited without request ID", fallback_category=category),
            False,
        )
    if request_id and len(route_rows) != 1:
        return (
            _axis("not_applicable", "routing cannot be audited without one exact route row"),
            _axis("fail", f"expected one route row for request ID, found {len(route_rows)}"),
            _axis(
                "fail",
                "fallback cannot be audited with missing or duplicate telemetry",
                fallback_category=FALLBACK_CORRELATION_MISSING,
            ),
            False,
        )
    row = route_rows[0]
    attempts = _route_attempts(row)
    header_attempts_raw = evidence.headers.get("x-ficelle-attempts", "")
    try:
        header_attempts = int(header_attempts_raw)
    except ValueError:
        header_attempts = -1
    logged_attempts = int(row.get("attempt_count") or len(attempts) or 0)
    fallback_header = evidence.headers.get("x-ficelle-fallback", "false").lower() == "true"
    fallback_observed = logged_attempts > 1
    telemetry_issues: list[str] = []
    routing_issues: list[str] = []
    # Header-vs-row consistency checks need response headers; a late-correlated case never
    # received them, and their absence is the client timeout, not a telemetry defect.
    if not late_correlated:
        if header_attempts != logged_attempts:
            telemetry_issues.append(f"header attempts {header_attempts} != logged {logged_attempts}")
        if fallback_header != fallback_observed:
            telemetry_issues.append("fallback header contradicts attempt count")
    mappings = (
        ("x-ficelle-model", "selected_model"),
        ("x-ficelle-upstream", "selected_upstream"),
        ("x-ficelle-source", "selected_source"),
    )
    for header, field in mappings:
        value = evidence.headers.get(header)
        logged = row.get(field)
        if value and logged and value != str(logged):
            telemetry_issues.append(f"{header} contradicts {field}")
    logged_status = row.get("final_status")
    if evidence.status_code is not None and logged_status is not None:
        try:
            if int(logged_status) != evidence.status_code:
                telemetry_issues.append("HTTP status contradicts logged final status")
        except (TypeError, ValueError):
            telemetry_issues.append("logged final status is invalid")
    response_model = evidence.payload.get("model") if isinstance(evidence.payload, dict) else None
    response_identity = "absent" if evidence.status_code is not None and evidence.status_code < 300 else "not_applicable"
    if evidence.status_code is not None and evidence.status_code < 300 and isinstance(response_model, str) and response_model:
        selected_model = str(evidence.headers.get("x-ficelle-model") or row.get("selected_model") or "")
        selected_upstream = str(evidence.headers.get("x-ficelle-upstream") or row.get("selected_upstream") or "")
        allowed_response_models = {
            value
            for value in (selected_model, selected_upstream, case.model_id, case.upstream_id)
            if value
        }
        if response_model in allowed_response_models:
            response_identity = "catalog_or_selected_alias"
        else:
            selected_source = str(evidence.headers.get("x-ficelle-source") or row.get("selected_source") or "")
            catalog_lookup = _safe_model_lookup(catalog or {})
            selected_catalog_model = catalog_lookup.get((selected_source, selected_model))
            if selected_catalog_model is None:
                selected_catalog_model = catalog_lookup.get((selected_source, selected_upstream))
            tokenizer = str((selected_catalog_model or {}).get("tokenizer") or "").strip().lower()
            if tokenizer == "router":
                response_identity = "provider_router_alias"
            else:
                response_identity = "unexplained_drift"
                routing_issues.append("response body model differs from selected/catalog identities")
    if case.target_kind == "direct-model":
        if fallback_observed:
            routing_issues.append("direct-model case used fallback")
        selected = evidence.headers.get("x-ficelle-model") or row.get("selected_model")
        upstream = evidence.headers.get("x-ficelle-upstream") or row.get("selected_upstream")
        if selected and selected != case.model_id and upstream != case.upstream_id:
            routing_issues.append("direct-model response identity differs from catalog identity")
    routing = _axis(
        "fail" if routing_issues else "pass",
        "; ".join(routing_issues) if routing_issues else "selected route is consistent with the requested target",
        response_model=sanitize_error_detail(response_model, 250),
        response_model_identity=response_identity,
    )
    telemetry = _axis(
        "fail" if telemetry_issues else "pass",
        "; ".join(telemetry_issues)
        if telemetry_issues
        else (
            "route row correlated by synthetic case id after client timeout"
            if late_correlated
            else "response headers match exact route row"
        ),
        request_id=request_id or row.get("request_id"),
        late_route_correlated=late_correlated,
    )

    policy_issues: list[str] = []
    retry_decisions: list[dict[str, Any]] = []
    try:
        candidate_count = int(row.get("candidate_count") or 0)
    except (TypeError, ValueError):
        candidate_count = 0
    for index, attempt in enumerate(attempts):
        reason = str(attempt.get("reason") or "")
        has_next_attempt = index < len(attempts) - 1
        expected_retry = (
            case.target_kind == "virtual-profile"
            and bool(reason)
            and reason != "ok"
            and reason not in NON_RETRYABLE_FAILURE_REASONS
            and not bool(attempt.get("stream_started"))
        )
        retry_decisions.append(
            {
                "attempt": index + 1,
                "reason": reason,
                "expected_retry": expected_retry,
                "observed_retry": has_next_attempt,
            }
        )
        if not reason:
            policy_issues.append("failed attempt has no classified reason")
        elif has_next_attempt and not expected_retry:
            policy_issues.append(f"unexpected fallback after {reason}")
        elif (
            not has_next_attempt
            and expected_retry
            and candidate_count > len(attempts)
        ):
            policy_issues.append(f"missing fallback after retryable {reason}")
    if policy_issues:
        # These are real retry-decision violations — the only failures that may be called
        # an illegitimate fallback decision (L2-R5).
        policy = _axis(
            "fail",
            "; ".join(policy_issues),
            decisions=retry_decisions,
            fallback_category=FALLBACK_ILLEGITIMATE_DECISION,
        )
    else:
        policy = _axis(
            "pass",
            "fallback was retryable" if fallback_observed else "no fallback required",
            decisions=retry_decisions,
        )
    return routing, telemetry, policy, fallback_observed


def _safe_model_lookup(catalog: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model in catalog.get("models", []):
        if not isinstance(model, dict):
            continue
        for key in (str(model.get("id") or ""), str(model.get("upstream_id") or "")):
            if key:
                result[(str(model.get("source") or ""), key)] = model
    return result


def safety_axis(
    router: Any,
    evidence: HttpEvidence,
    catalog: dict[str, Any],
    route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lookup = _safe_model_lookup(catalog)
    for attempt in _route_attempts(route):
        source = str(attempt.get("source") or "")
        model_id = str(attempt.get("model") or "")
        upstream_id = str(attempt.get("upstream") or "")
        model = lookup.get((source, model_id)) or lookup.get((source, upstream_id))
        if model is None:
            return _axis(
                "fail",
                "route attempt is absent from the initial strict-zero catalog",
                source=source,
                model=model_id,
            )
        if not router.normalized_free_access(model).get("eligible"):
            return _axis(
                "fail",
                "route attempt used a model that is not strict-zero eligible",
                source=source,
                model=model_id,
            )
    if evidence.status_code is None or evidence.status_code >= 300:
        return _axis("pass", "all attempted upstreams were strict-zero eligible")
    source = evidence.headers.get("x-ficelle-source", "")
    model_id = evidence.headers.get("x-ficelle-model", "")
    upstream_id = evidence.headers.get("x-ficelle-upstream", "")
    model = lookup.get((source, model_id)) or lookup.get((source, upstream_id))
    if model is None:
        return _axis("fail", "delivered model is absent from the initial strict-zero catalog")
    free_access = router.normalized_free_access(model)
    if not free_access.get("eligible"):
        return _axis("fail", "delivered model is not strict-zero eligible", source=source, model=model_id)
    return _axis("pass", "delivered model is strict-zero eligible", source=source, model=model_id)


def _compact_state_section(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"count": len(value), "hash": stable_hash(redact_sensitive_json(value))}
    if isinstance(value, list):
        return {"count": len(value), "hash": stable_hash(redact_sensitive_json(value))}
    return {"count": 0, "hash": stable_hash(value)}


def state_change_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    interesting = (
        "cooldowns",
        "provider_cooldowns",
        "quota_cooldowns",
        "quarantine",
        "provider_errors",
        "model_errors",
        "provider_budget_spend",
        "provider_budgets",
        "provider_stats",
        "stats",
        "successes",
        "verified_capabilities",
    )
    changes: dict[str, Any] = {}
    for key in interesting:
        before_summary = _compact_state_section(before.get(key))
        after_summary = _compact_state_section(after.get(key))
        if before_summary != after_summary:
            changes[key] = {"before": before_summary, "after": after_summary}
    return changes


def state_policy_verdict(
    route: dict[str, Any] | None,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    late_extended_window: bool = False,
) -> dict[str, Any]:
    section_for_scope = {
        "model": "cooldowns",
        "provider": "provider_cooldowns",
        "quota": "quota_cooldowns",
        "quarantine": "quarantine",
    }
    changes = state_change_summary(before, after)
    missing: list[str] = []
    checked: list[dict[str, str]] = []
    expected_policy_sections: set[str] = set()
    attempts = _route_attempts(route)
    failed_attempts = [attempt for attempt in attempts if str(attempt.get("reason") or "") not in {"", "ok"}]
    has_success = any(str(attempt.get("reason") or "") == "ok" for attempt in attempts)
    for attempt in failed_attempts:
        reason = str(attempt.get("reason") or "")
        policy = cooldown_policy_for_reason(reason, source=str(attempt.get("source") or ""))
        scope = expected_policy_scope(policy)
        checked.append({"reason": reason, "expected_scope": scope})
        # Expect every section the policy actually writes, not only the primary scope:
        # billing_or_paid legitimately writes quarantine AND a 24h model cooldown.
        expected_policy_sections.update(expected_policy_sections_for_policy(policy))
        if getattr(policy, "record_provider_error", False):
            expected_policy_sections.add("provider_errors")
    for section in sorted(expected_policy_sections):
        if section not in changes:
            missing.append(f"expected state change in {section}")
    if failed_attempts and not {"stats", "model_errors"}.intersection(changes):
        missing.append("failed attempts did not update model failure evidence")
    if has_success and "successes" not in changes:
        missing.append("successful attempt did not update success evidence")
    if "verified_capabilities" in changes:
        missing.append("ordinary chat request unexpectedly changed verified capabilities")
    policy_sections = set(section_for_scope.values()) | {"provider_errors"}
    expected_model_keys = {
        f"{attempt.get('source')}::{attempt.get('upstream')}"
        for attempt in attempts
        if attempt.get("source") and attempt.get("upstream")
    }
    expected_source_keys = {str(attempt.get("source") or "") for attempt in attempts if attempt.get("source")}

    def _changed_keys(section: str) -> set[str]:
        before_section = before.get(section) if isinstance(before.get(section), dict) else {}
        after_section = after.get(section) if isinstance(after.get(section), dict) else {}
        return {
            str(key)
            for key in set(before_section) | set(after_section)
            if before_section.get(key) != after_section.get(key)
        }

    foreign_late_writes: list[str] = []
    if failed_attempts and not has_success:
        for section in sorted(policy_sections.intersection(changes) - expected_policy_sections):
            changed = _changed_keys(section)
            own_keys = expected_source_keys if section in {"provider_cooldowns", "provider_errors"} else expected_model_keys
            own_changed = changed & own_keys
            foreign_changed = changed - own_keys
            if own_changed or not changed:
                # A write in an unexpected scope for THIS case's own models/providers is a
                # real scope violation (the L2-R4 contract this checker exists to enforce).
                missing.append(f"unexpected state change in {section}")
            elif foreign_changed and not late_extended_window:
                missing.append(f"unexpected state change in {section}")
            else:
                # Extended correlation window (the client timed out and the harness waited
                # for the router's deadline): a foreign-key write is the previous case's
                # router finishing late — an attribution note, not this case's violation.
                foreign_late_writes.append(section)
    if missing:
        # A state write in an unexpected scope is a scope mismatch, not an illegitimate
        # retry decision (L2-R5) — the two must never share one bucket.
        return _axis(
            "fail",
            "; ".join(missing),
            checked=checked,
            fallback_category=FALLBACK_STATE_SCOPE_MISMATCH,
        )
    return _axis(
        "pass",
        "observed runtime state consequences match classified scopes"
        + (
            f" (late foreign writes outside this case's scope: {', '.join(foreign_late_writes)})"
            if foreign_late_writes
            else ""
        ),
        checked=checked,
        late_foreign_write_sections=foreign_late_writes or None,
    )


def resolve_case_body(case: SyntheticHealthCase, outcomes: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    body = copy.deepcopy(case.body or {})
    dependency_id = body.pop("__dependency_case_id", None)
    if not dependency_id:
        return body, None
    dependency = outcomes.get(str(dependency_id))
    calls = dependency.get("tool_calls") if isinstance(dependency, dict) else None
    if not isinstance(calls, list) or not calls:
        return None, f"dependency {dependency_id} did not produce a usable tool call"
    first_id = calls[0].get("id") if isinstance(calls[0], dict) else None
    if not first_id:
        return None, f"dependency {dependency_id} tool call has no id"
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        if message.get("tool_calls") == "__DEPENDENCY_TOOL_CALLS__":
            message["tool_calls"] = calls
        if message.get("tool_call_id") == "__DEPENDENCY_TOOL_CALL_ID__":
            message["tool_call_id"] = first_id
    return body, None


def execute_deterministic_case(case: SyntheticHealthCase) -> dict[str, Any]:
    started = now_iso()
    axes = {name: _axis("not_applicable", "not applicable to deterministic case") for name in ("transport", "protocol", "semantic", "routing", "policy", "telemetry", "safety")}
    diagnostics: dict[str, Any] = {}
    passed = False
    if case.validator == "failure-contract":
        scenario = next((item for item in deterministic_failure_scenarios() if item.id == case.target_id), None)
        if scenario is None:
            diagnostics["error"] = "unknown deterministic scenario"
        else:
            error_codes = provider_error_codes(scenario.body)
            effective_status = scenario.status_code
            if effective_status is None:
                effective_status = status_for_error_codes(*error_codes)
            reason = classify_failure(
                effective_status,
                scenario.body,
                error_codes=error_codes,
                normalized_free_access=scenario.free_access,
                upstream_model_id=scenario.upstream_model_id,
            )
            policy = cooldown_policy_for_reason(reason, source="synthetic-provider")
            scope = expected_policy_scope(policy)
            retry = reason not in NON_RETRYABLE_FAILURE_REASONS
            passed = reason == scenario.expected_reason and scope == scenario.expected_scope and retry == scenario.expected_retry
            diagnostics = {
                "observed_reason": reason,
                "observed_scope": scope,
                "observed_retry": retry,
                "expected_reason": scenario.expected_reason,
                "expected_scope": scenario.expected_scope,
                "expected_retry": scenario.expected_retry,
            }
            from ficelle.use_cases.chat_completion import FailureRouteLogInput, SuccessRouteLogInput, build_failure_route_log, build_success_route_log

            first_attempt = {
                "model": "synthetic/model-a",
                "upstream": scenario.upstream_model_id,
                "source": "synthetic-provider",
                "status": effective_status,
                "reason": reason,
            }
            if retry:
                attempts = [first_attempt, {"model": "synthetic/model-b", "upstream": "model-b", "source": "synthetic-provider-b", "status": 200, "reason": "ok"}]
                route_log = build_success_route_log(
                    SuccessRouteLogInput(
                        request_id="deterministic-request",
                        safe_requested_model="ficelle/auto-tools",
                        selected_model={"id": "synthetic/model-b", "upstream_id": "model-b", "source": "synthetic-provider-b"},
                        competence="fallback",
                        final_status=200,
                        candidate_count=2,
                        attempt_count=2,
                        attempts=attempts,
                        duration_seconds=0.2,
                        stream=False,
                    )
                )
            else:
                attempts = [first_attempt]
                route_log = build_failure_route_log(
                    FailureRouteLogInput(
                        request_id="deterministic-request",
                        safe_requested_model="ficelle/auto-tools",
                        candidate_count=2,
                        attempt_count=1,
                        attempts=attempts,
                        errors=[first_attempt],
                        duration_seconds=0.1,
                        stream=False,
                    )
                )
            route_shape_ok = route_log.get("attempt_count") == len(attempts) and route_log.get("request_id") == "deterministic-request"
            diagnostics["route_log"] = route_log
            axes["semantic"] = _axis("pass" if passed else "fail", "shared classifier and cooldown policy match" if passed else "shared failure contract mismatch", **diagnostics)
            axes["policy"] = axes["semantic"]
            axes["routing"] = _axis("pass" if retry == scenario.expected_retry else "fail", "fallback boundary matches shared reason")
            axes["telemetry"] = _axis("pass" if route_shape_ok else "fail", "shared route-log builder recorded ordered attempts")
            axes["safety"] = _axis("pass", "ephemeral deterministic lane performed no state write")
    elif case.validator == "stream-contract":
        streams = {
            "pre-first-byte-retry": ({"status": "fail", "reason": "pre_stream_failure", "stream_started": False}, "retryable_failure", "model"),
            "mid-stream-no-retry": ({"status": "fail", "reason": "mid_stream_failure", "stream_started": True}, "mid_stream_failure", "model"),
            "client-disconnect-no-cooldown": ({"status": "fail", "reason": "client_disconnected", "stream_started": True}, "mid_stream_failure", "none"),
        }
        definition = streams.get(case.target_id)
        if definition is not None:
            stream_result, expected_outcome, expected_scope = definition
            decision = evaluate_streaming_result(
                stream_result,
                {"id": "synthetic/model-a", "upstream_id": "model-a", "source": "synthetic-provider"},
                response_status=200,
                latency_seconds=0.1,
                requested_model_is_virtual=True,
            )
            policy = cooldown_policy_for_reason(str(decision.cooldown_reason or "unavailable"), source="synthetic-provider")
            scope = expected_policy_scope(policy)
            passed = decision.outcome == expected_outcome and scope == expected_scope
            diagnostics = {"outcome": decision.outcome, "cooldown_scope": scope}
            axes["semantic"] = _axis("pass" if passed else "fail", "stream boundary matches shared decision" if passed else "stream boundary mismatch", **diagnostics)
            axes["routing"] = axes["semantic"]
            axes["policy"] = axes["semantic"]
            axes["telemetry"] = _axis("pass", "stream attempt fields are present in the shared decision")
            axes["safety"] = _axis("pass", "ephemeral deterministic lane performed no state write")
    elif case.validator == "invocation-contract":
        from ficelle.use_cases.chat_completion import evaluate_invocation_exception

        timeout = case.target_id == "timeout"
        exc: Exception = TimeoutError("synthetic timeout") if timeout else ConnectionError("synthetic connection failure")
        decision = evaluate_invocation_exception(
            exc,
            {"id": "synthetic/model-a", "upstream_id": "model-a", "source": "synthetic-provider"},
            latency_seconds=0.1,
            timeout=timeout,
        )
        policy = cooldown_policy_for_reason(decision.cooldown_reason, source="synthetic-provider")
        scope = expected_policy_scope(policy)
        passed = decision.cooldown_reason == ("timeout" if timeout else "unavailable") and scope == "model"
        diagnostics = {"reason": decision.cooldown_reason, "cooldown_scope": scope, "attempt": decision.attempt_update}
        axes["semantic"] = _axis("pass" if passed else "fail", "invocation exception maps to retryable failure", **diagnostics)
        axes["routing"] = _axis("pass" if passed else "fail", "virtual request may retry the next candidate")
        axes["policy"] = axes["semantic"]
        axes["telemetry"] = _axis("pass" if decision.attempt_update.get("error_type") else "fail", "exception type is recorded")
        axes["safety"] = _axis("pass", "ephemeral deterministic lane performed no state write")
        diagnostics["observed_retry"] = True
    elif case.validator == "success-contract":
        from ficelle.use_cases.benchmark import has_deliverable_message
        from ficelle.use_cases.chat_completion import evaluate_non_streaming_success_response

        class Response:
            status_code = 200

            def json(self) -> Any:
                if case.target_id == "invalid-json":
                    raise ValueError("synthetic invalid JSON")
                if case.target_id == "embedded-error":
                    return {"error": {"type": "authentication_error", "message": "invalid api key"}}
                finish_reason = "length" if case.target_id == "truncated-before-content" else "stop"
                return {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": finish_reason}]}

        def detect(payload: Any, _model: dict[str, Any]) -> tuple[str, str, int | None] | None:
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                return "auth_or_credit", "embedded authentication error", 401
            return None

        decision = evaluate_non_streaming_success_response(
            Response(),
            {"id": "synthetic/model-a", "upstream_id": "model-a", "source": "synthetic-provider"},
            latency_seconds=0.1,
            requested_model_is_virtual=True,
            detect_success_error=detect,
            has_deliverable=has_deliverable_message,
        )
        expected_reason = {
            "embedded-error": "auth_or_credit",
            "invalid-json": "invalid_success_json",
            "empty-message": "empty_assistant_message",
            "truncated-before-content": "truncated_before_content",
        }[case.target_id]
        reason = str(decision.attempt_update.get("reason") or "")
        policy = cooldown_policy_for_reason(str(decision.cooldown_reason or "unavailable"), source="synthetic-provider")
        scope = expected_policy_scope(policy)
        expected_scope = "provider" if case.target_id == "embedded-error" else ("none" if case.target_id == "truncated-before-content" else "model")
        passed = decision.outcome == "retryable_failure" and reason == expected_reason and scope == expected_scope
        diagnostics = {"observed_retry": True, "reason": reason, "cooldown_scope": scope, "attempt": decision.attempt_update}
        axes["semantic"] = _axis("pass" if passed else "fail", "deceptive HTTP 200 is rejected" if passed else "deceptive HTTP 200 contract mismatch", **diagnostics)
        axes["protocol"] = axes["semantic"]
        axes["routing"] = _axis("pass" if passed else "fail", "fallback remains legal before delivery")
        axes["policy"] = axes["semantic"]
        axes["telemetry"] = _axis("pass" if decision.attempt_update.get("reason") else "fail", "attempt reason is explicit")
        axes["safety"] = _axis("pass", "ephemeral deterministic lane performed no state write")
    elif case.validator == "tool-contract":
        from ficelle.reasoning_replay import ReasoningReplayStore
        from ficelle.use_cases.chat_completion import malformed_tool_call_detail

        if case.target_id == "blank-function-name":
            detail = malformed_tool_call_detail({"messages": [{"role": "assistant", "tool_calls": [{"id": "call-a", "type": "function", "function": {"name": "", "arguments": "{}"}}]}]})
            passed = bool(detail)
            diagnostics = {"local_rejection": bool(detail), "detail": detail}
        elif case.target_id == "malformed-arguments":
            reason = classify_failure(400, '{"error":{"message":"malformed tool arguments"}}')
            policy = cooldown_policy_for_reason(reason)
            passed = reason == "bad_upstream_request" and expected_policy_scope(policy) == "none"
            diagnostics = {"reason": reason, "cooldown_scope": expected_policy_scope(policy)}
        elif case.target_id == "orphan-tool-result":
            orphan = {"role": "tool", "tool_call_id": "missing-call", "content": "{}"}
            passed = orphan["tool_call_id"] == "missing-call"
            diagnostics = {"classification": "caller-caused-terminal", "cooldown_scope": "none"}
        else:
            store = ReasoningReplayStore()
            calls = [{"id": "call-cross", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
            store.observe("model-a", "I need the lookup result.", ["call-cross"], ["lookup({})"], now=1.0)
            body = {"messages": [{"role": "assistant", "content": None, "tool_calls": calls}]}
            replayed = store.replay_into_body("model-b", body, now=2.0)
            passed = replayed["messages"][0].get("reasoning_content") == "I need the lookup result."
            diagnostics = {"cross_model_trace_replayed": passed}
        axes["semantic"] = _axis("pass" if passed else "fail", "tool replay contract holds" if passed else "tool replay contract mismatch", **diagnostics)
        axes["policy"] = _axis("pass" if passed else "fail", "caller/tool history policy is deterministic")
        axes["safety"] = _axis("pass", "deterministic tools execute no external action")
    fallback_observed = bool(diagnostics.get("observed_retry")) or case.target_id == "pre-first-byte-retry"
    status = aggregate_case_status(axes, fallback_observed=fallback_observed)
    return {
        "schema_version": SCHEMA_VERSION,
        "event": "outcome",
        "case_id": case.id,
        "phase": case.phase,
        "lane": case.lane,
        "target": {"kind": case.target_kind, "id": case.target_id},
        "source": case.source,
        "tags": list(case.tags),
        "started_at": started,
        "ended_at": now_iso(),
        "status": status,
        "axes": axes,
        "diagnostic": diagnostics,
    }


def _planned_row(run_id: str, case: SyntheticHealthCase) -> dict[str, Any]:
    plan = case.public_plan()
    return {
        "schema_version": SCHEMA_VERSION,
        "event": "planned",
        "run_id": run_id,
        "case_id": case.id,
        "phase": case.phase,
        "lane": case.lane,
        "planned_at": now_iso(),
        "target": plan["target"],
        "source": case.source,
        "validator": case.validator,
        "tags": list(case.tags),
        "request_template_hash": plan["request_template_hash"],
        "request_shape": plan["request_shape"],
        "skip_reason": case.skip_reason,
    }


def _skipped_outcome(
    case: SyntheticHealthCase,
    reason: str,
    *,
    eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    axes = {name: _axis("not_applicable", reason) for name in ("transport", "protocol", "semantic", "routing", "policy", "telemetry", "safety")}
    outcome = {
        "schema_version": SCHEMA_VERSION,
        "event": "outcome",
        "case_id": case.id,
        "phase": case.phase,
        "lane": case.lane,
        "target": {"kind": case.target_kind, "id": case.target_id},
        "source": case.source,
        "tags": list(case.tags),
        "started_at": now_iso(),
        "ended_at": now_iso(),
        "status": "skipped",
        "skip_reason": reason,
        "axes": axes,
    }
    if eligibility is not None:
        outcome["eligibility"] = dict(eligibility)
        # A model that vanished from the active catalog is a surface change, not a model
        # capability failure: it stays in the planned denominator under its own marker.
        if eligibility.get("scope") == "catalog":
            outcome["surface_changed"] = str(eligibility.get("reason") or "catalog_model_missing_during_run")
    return outcome


def _pre_sweep_case_id(case_id: str) -> str | None:
    if not case_id.startswith("p4."):
        return None
    return "p2." + case_id[3:].replace("virtual-post", "virtual-pre", 1)


def post_sweep_skip_reason(
    case: SyntheticHealthCase,
    outcomes: dict[str, dict[str, Any]],
) -> str | None:
    pre_case_id = _pre_sweep_case_id(case.id)
    if not pre_case_id or case.target_id in CORE_POST_SWEEP_PROFILES:
        return None
    pre_outcome = outcomes.get(pre_case_id)
    if pre_outcome and pre_outcome.get("status") not in {"pass", "pass_with_fallback", "degraded"}:
        return f"pre_sweep_case_not_successful:{pre_case_id}"
    return None


# A control outcome that neither passed nor was skipped means plain tool calling is broken
# on this model, so no dialect verdict can be attributed to a schema feature.
CONTROL_PASSING_STATUSES = {"pass", "pass_with_fallback", "degraded"}


def dialect_control_verdict(
    case: SyntheticHealthCase,
    outcomes: dict[str, dict[str, Any]],
) -> str | None:
    """Why this dialect case cannot produce a feature verdict, or None to run it.

    A model that refuses plain tool calling (Groq's compound systems document exactly
    that: "custom user-provided tools are not supported") would otherwise fail all six
    dialect cases and be recorded as six feature failures — evidence that blames the
    schema for a missing capability. The control case decides instead, and the dialect
    cases are not even sent (L3-R3)."""
    control_id = str(case.expected.get("control_case_id") or "")
    if not control_id:
        return None
    control = outcomes.get(control_id)
    if control is None:
        return None
    status = str(control.get("status") or "")
    if status in CONTROL_PASSING_STATUSES:
        return None
    if status == "skipped":
        # The control never ran (cooldown, quarantine, missing model): unknown, not failed.
        return f"dialect_control_not_run:{control_id}"
    axes = control.get("axes") if isinstance(control.get("axes"), dict) else {}
    semantic = axes.get("semantic") if isinstance(axes.get("semantic"), dict) else {}
    classified = str(semantic.get("classified_reason") or "")
    if classified in {"bad_upstream_request", "unsupported_tool_schema"}:
        # The provider refused the portable fixture itself: this model does not accept
        # user-provided tools at all.
        return f"tool_support_absent:{control_id}"
    return f"dialect_control_failed:{control_id}"


def _evidence_record(body: dict[str, Any], evidence: HttpEvidence) -> dict[str, Any]:
    return {
        "request_template_hash": request_template_hash(body),
        "request_shape": request_shape(body),
        "latency_seconds": evidence.latency_seconds,
        "http_status": evidence.status_code,
        "content_type": evidence.content_type[:250],
        "headers": evidence.headers,
        "response_shape": response_shape(evidence.payload),
        "response_preview": sanitize_preview(evidence.payload if evidence.payload is not None else evidence.text),
        "stream_complete": evidence.stream_complete,
        "stream_error": redact_sensitive_json(evidence.stream_error),
        "transport_error": evidence.transport_error,
    }


def staged_verdict_record(
    case: SyntheticHealthCase,
    evidence: HttpEvidence,
    route: dict[str, Any] | None,
    axes: dict[str, dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    """The five-rung verdict every live tool-tagged outcome exposes (L3-R1). A rung past
    the first False is None: it was never reached, and a report must not read it as
    failure."""
    candidate_count = int(route.get("candidate_count") or 0) if isinstance(route, dict) else None
    selection_reachable = candidate_count > 0 if candidate_count is not None else evidence.status_code is not None and evidence.status_code < 500
    attempts = _route_attempts(route)
    upstream_reached = bool(attempts) if route is not None else None
    protocol_valid = axes.get("protocol", {}).get("status") == "pass" if selection_reachable else None
    semantic_axis = axes.get("semantic", {}) if isinstance(axes.get("semantic"), dict) else {}
    tool_semantic_valid: bool | None
    if not selection_reachable or not upstream_reached or protocol_valid is not True:
        tool_semantic_valid = None
    elif semantic_axis.get("inconclusive"):
        tool_semantic_valid = None
    else:
        tool_semantic_valid = semantic_axis.get("status") == "pass"
    return {
        "selection_reachable": selection_reachable,
        "upstream_reached": upstream_reached,
        "protocol_valid": protocol_valid,
        "tool_semantic_valid": tool_semantic_valid,
        "end_user_success": status in {"pass", "pass_with_fallback"},
    }


def primary_failure_layer(row: dict[str, Any]) -> str | None:
    """Exactly one primary layer per failure row (artifact contract). Priority order walks
    from the outermost surface inward; secondary dimensions stay visible in the axes."""
    status = str(row.get("status") or "")
    if status in {"pass", "pass_with_fallback", "skipped", "planned"}:
        return None
    if status == "safety_breach":
        return "safety"
    if status == "harness_error":
        return "harness"
    tags = set(row.get("tags") or [])
    if "preflight" in tags:
        return "preflight"
    axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
    semantic = axes.get("semantic") if isinstance(axes.get("semantic"), dict) else {}
    classified = str(semantic.get("classified_reason") or "")
    route = row.get("route") if isinstance(row.get("route"), dict) else {}
    candidate_count = route.get("candidate_count")
    transport = axes.get("transport") if isinstance(axes.get("transport"), dict) else {}
    if transport.get("status") == "fail" and str(transport.get("diagnostic") or "").strip():
        # No HTTP response at all: the request died between client and router.
        if row.get("route") is None and not classified:
            return "downstream_transport"
    if classified.startswith("selection:"):
        return "selection"
    if candidate_count == 0:
        return "catalog"
    if classified in {"bad_upstream_request", "unsupported_tool_schema"}:
        return "request_compatibility"
    if classified in {"rate_limited", "rate_limited_upstream", "request_too_large", "quota_exhausted", "no_free_quota", "auth_or_credit", "billing_or_paid"}:
        return "provider_capacity"
    if classified in {"timeout", "server_error", "unavailable", "truncated_before_content", "empty_message", "mid_stream_failure"}:
        return "upstream_transport"
    if classified == "request_deadline_exceeded":
        return "router_transport"
    protocol = axes.get("protocol") if isinstance(axes.get("protocol"), dict) else {}
    if protocol.get("status") == "fail":
        return "protocol"
    if semantic.get("status") == "fail":
        return "tool_semantics" if "tools" in tags else "model_semantics"
    if transport.get("status") == "fail":
        return "downstream_transport"
    return "harness"


def execute_live_case(
    router: Any,
    case: SyntheticHealthCase,
    *,
    client: LoopbackChatClient,
    route_log_path: Path,
    catalog: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    started = now_iso()
    if case.skip_reason:
        return _skipped_outcome(case, case.skip_reason)
    post_sweep_skip = post_sweep_skip_reason(case, outcomes)
    if post_sweep_skip:
        return _skipped_outcome(case, post_sweep_skip)
    control_skip = dialect_control_verdict(case, outcomes)
    if control_skip:
        return _skipped_outcome(case, control_skip)
    body, dependency_error = resolve_case_body(case, outcomes)
    if body is None:
        return _skipped_outcome(case, dependency_error or "case dependency unavailable")

    state_before = router.load_runtime_state()
    correlation_id = synthetic_case_correlation_id(case.id)
    evidence = client.complete(
        body,
        timeout_seconds=timeout_seconds,
        extra_headers={SYNTHETIC_CASE_ID_HEADER: correlation_id},
    )
    request_id = evidence.headers.get("x-ficelle-request-id", "")
    route_rows = wait_for_route_rows(route_log_path, request_id) if request_id else []
    late_route_correlated = False
    if not request_id and evidence.transport_error:
        # The client gave up first (L2-R3): wait only for the router's bounded request
        # deadline, then correlate by the synthetic case id. Exactly one late row means a
        # late completion; none means the request was still in flight or never entered.
        wait_seconds = late_route_correlation_wait_seconds(
            router.load_runtime_config(), timeout_seconds
        )
        late_rows = wait_for_route_rows(
            route_log_path, correlation_id, wait_seconds, field="synthetic_case_id"
        )
        if late_rows:
            route_rows = late_rows
            late_route_correlated = True
    state_after = router.load_runtime_state()

    transport_ok = evidence.status_code is not None and (not case.stream or evidence.status_code >= 300 or evidence.stream_complete)
    protocol_ok = evidence.status_code is not None and (
        (case.stream and evidence.status_code < 300 and evidence.stream_complete)
        or (not case.stream and isinstance(evidence.payload, dict))
        or (evidence.status_code >= 300 and isinstance(evidence.payload, dict))
    )
    semantic = semantic_verdict(router, case, evidence, catalog)
    routing, telemetry, policy, fallback_observed = route_axes(case, evidence, route_rows, catalog)
    route = route_rows[0] if len(route_rows) == 1 else None
    safety = safety_axis(router, evidence, catalog, route)
    state_policy = state_policy_verdict(
        route, state_before, state_after, late_extended_window=late_route_correlated
    )
    if state_policy["status"] == "fail":
        policy = state_policy
    axes = {
        "transport": _axis("pass" if transport_ok else "fail", "HTTP exchange completed" if transport_ok else evidence.transport_error or "stream did not complete"),
        "protocol": _axis("pass" if protocol_ok else "fail", "OpenAI response envelope is parseable" if protocol_ok else "invalid or incomplete OpenAI response envelope"),
        "semantic": semantic,
        "routing": routing,
        "policy": policy,
        "telemetry": telemetry,
        "safety": safety,
    }
    failure_signatures: list[dict[str, Any]] = []
    for attempt in _route_attempts(route):
        reason = str(attempt.get("reason") or "")
        if not reason or reason == "ok":
            continue
        signature_basis = {
            "source": attempt.get("source") or case.source,
            "model": attempt.get("model"),
            "status": attempt.get("status"),
            "reason": reason,
            "error_type": attempt.get("error_type"),
        }
        failure_signatures.append({**signature_basis, "signature": stable_hash(signature_basis)[:20]})
    changes = state_change_summary(state_before, state_after)
    tool_transition: dict[str, Any] | None = None
    if case.validator == "tool-continuation":
        dependency_id = str(case.expected.get("dependency_case_id") or "")
        dependency = outcomes.get(dependency_id)
        dependency_route = dependency.get("route") if isinstance(dependency, dict) else None
        if isinstance(dependency_route, dict) and isinstance(route, dict):
            previous_identity = {
                "source": dependency_route.get("selected_source"),
                "model": dependency_route.get("selected_model"),
                "upstream": dependency_route.get("selected_upstream"),
            }
            current_identity = {
                "source": route.get("selected_source"),
                "model": route.get("selected_model"),
                "upstream": route.get("selected_upstream"),
            }
            tool_transition = {
                "dependency_case_id": dependency_id,
                "from": previous_identity,
                "to": current_identity,
                "switched": previous_identity != current_identity,
            }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event": "outcome",
        "case_id": case.id,
        "phase": case.phase,
        "lane": case.lane,
        "target": {"kind": case.target_kind, "id": case.target_id},
        "source": case.source,
        "tags": list(case.tags),
        "started_at": started,
        "ended_at": now_iso(),
        "status": aggregate_case_status(axes, fallback_observed=fallback_observed),
        "axes": axes,
        "request": _evidence_record(body, evidence),
        "route": route,
        "route_match_count": len(route_rows),
        "synthetic_case_id": correlation_id,
        "late_route_correlated": late_route_correlated,
        "tool_calls": evidence.tool_calls,
        "failure_signatures": failure_signatures,
        "state_changes": changes,
        "state_change_attribution": "observed_during_case" if changes else "none_observed",
        "tool_continuation_transition": tool_transition,
    }
    if "tools" in case.tags:
        record["staged_verdict"] = staged_verdict_record(case, evidence, route, axes, str(record["status"]))
    layer = primary_failure_layer(record)
    if layer is not None:
        record["primary_failure_layer"] = layer
    serialized = json.dumps(redact_sensitive_json(record), ensure_ascii=False, sort_keys=True)
    if SECRET_PATTERN.search(serialized):
        record["axes"]["safety"] = _axis("fail", "secret-shaped content remained after redaction")
        record["status"] = "safety_breach"
    return record


def _finding_rows(outcomes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for row in outcomes:
        status = str(row.get("status") or "")
        if status in {"pass", "pass_with_fallback"}:
            continue
        if status == "skipped" and row.get("skip_reason") == "parallel_tools_not_claimed":
            continue
        axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
        failed_axes = [name for name, axis in axes.items() if isinstance(axis, dict) and axis.get("status") == "fail"]
        diagnostic = next(
            (
                str(axes[name].get("diagnostic") or "")
                for name in ("safety", "transport", "protocol", "semantic", "routing", "policy", "telemetry")
                if name in axes and isinstance(axes[name], dict) and axes[name].get("status") == "fail"
            ),
            str(row.get("skip_reason") or status),
        )
        finding_id = stable_hash({"case": row.get("case_id"), "axes": failed_axes, "diagnostic": diagnostic})[:20]
        findings.append(
            {
                "id": finding_id,
                "case_id": row.get("case_id"),
                "status": status,
                "failed_axes": failed_axes,
                "diagnostic": sanitize_error_detail(diagnostic, 500),
                "source": row.get("source"),
            }
        )
    return findings


def _previous_comparable_summary(paths: RunPaths, run_metadata: dict[str, Any]) -> dict[str, Any] | None:
    root = paths.root.parent
    candidates: list[tuple[float, dict[str, Any]]] = []
    current_build = run_metadata.get("build_identity")
    if not isinstance(current_build, dict) or current_build.get("complete") is not True:
        return None
    if not root.exists():
        return None
    for summary_path in root.glob("*/summary.json"):
        if summary_path == paths.summary_json:
            continue
        summary = load_json(summary_path, {})
        identity = summary.get("identity") if isinstance(summary, dict) else None
        if not isinstance(identity, dict):
            continue
        previous_build = identity.get("build_identity")
        if not isinstance(previous_build, dict) or previous_build.get("complete") is not True:
            continue
        harness_health = (summary.get("health") or {}).get("harness") or {}
        if summary.get("parent_run_id") or harness_health.get("incomplete"):
            continue
        if identity.get("corpus_version") != run_metadata.get("corpus_version"):
            continue
        try:
            candidates.append((summary_path.stat().st_mtime, summary))
        except OSError:
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _test_surface(inventory: dict[str, Any], cases: list[SyntheticHealthCase]) -> dict[str, Any]:
    capability_tags: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        if case.target_kind != "direct-model":
            continue
        for tag in case.tags:
            if tag in {"capability", "tools", "streaming", "json-shaped"} or tag.startswith("t"):
                capability_tags[case.target_id].add(tag)
    providers = [row for row in inventory.get("providers", []) if isinstance(row, dict)]
    models = [row for row in inventory.get("models", []) if isinstance(row, dict)]
    return {
        "providers": sorted(str(row.get("source") or "") for row in providers if row.get("source")),
        "invokable_providers": sorted(
            str(row.get("source") or "") for row in providers if row.get("source") and row.get("can_invoke")
        ),
        "eligible_models": sorted(
            str(row.get("id") or "") for row in models if row.get("id") and row.get("strict_zero_eligible")
        ),
        "virtual_profiles": sorted(str(profile) for profile in inventory.get("virtual_profiles", [])),
        "model_capabilities": {
            model_id: sorted(tags)
            for model_id, tags in sorted(capability_tags.items())
        },
    }


def _surface_changes(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for key in ("providers", "invokable_providers", "eligible_models", "virtual_profiles"):
        current_values = {str(item) for item in current.get(key, [])}
        previous_values = {str(item) for item in previous.get(key, [])}
        changes[key] = {
            "added": sorted(current_values - previous_values),
            "removed": sorted(previous_values - current_values),
        }
    current_capabilities = current.get("model_capabilities") if isinstance(current.get("model_capabilities"), dict) else {}
    previous_capabilities = previous.get("model_capabilities") if isinstance(previous.get("model_capabilities"), dict) else {}
    capability_changes: list[dict[str, Any]] = []
    for model_id in sorted(set(current_capabilities) & set(previous_capabilities)):
        current_values = {str(item) for item in current_capabilities.get(model_id, [])}
        previous_values = {str(item) for item in previous_capabilities.get(model_id, [])}
        if current_values != previous_values:
            capability_changes.append(
                {
                    "model_id": model_id,
                    "added": sorted(current_values - previous_values),
                    "removed": sorted(previous_values - current_values),
                }
            )
    changes["capability_claims"] = capability_changes
    return changes


def _tool_metrics(cases: list[SyntheticHealthCase], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    planned = [case for case in cases if "tools" in case.tags]
    outcomes_by_id = {str(row.get("case_id") or ""): row for row in outcomes}
    tagged = [outcomes_by_id[case.id] for case in planned if case.id in outcomes_by_id]
    direct_models = sorted({case.target_id for case in planned if case.target_kind == "direct-model"})
    covered_models = 0
    blocked_models = 0
    for model_id in direct_models:
        model_cases = [case for case in planned if case.target_kind == "direct-model" and case.target_id == model_id]
        required = [case for case in model_cases if not case.skip_reason]
        if not required:
            blocked_models += 1
        elif all(
            case.id in outcomes_by_id and outcomes_by_id[case.id].get("status") != "skipped"
            for case in required
        ):
            covered_models += 1
    metrics: dict[str, Any] = {
        "planned": len(planned),
        "completed": len(tagged),
        "passed": sum(1 for row in tagged if row.get("status") in {"pass", "pass_with_fallback"}),
        "failed": sum(1 for row in tagged if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"}),
        "skipped": sum(1 for row in tagged if row.get("status") == "skipped"),
        "models": {
            "denominator": len(direct_models),
            "covered": covered_models,
            "blocked": blocked_models,
            "not_fully_covered": len(direct_models) - covered_models - blocked_models,
        },
        "live_continuations": sum(
            1
            for row in tagged
            if isinstance(row.get("tool_continuation_transition"), dict)
        ),
        "live_cross_model_continuations": sum(
            1
            for row in tagged
            if isinstance(row.get("tool_continuation_transition"), dict)
            and row["tool_continuation_transition"].get("switched") is True
        ),
    }
    for tag in ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"):
        planned_rows = [case for case in cases if tag in case.tags]
        rows = [outcomes_by_id[case.id] for case in planned_rows if case.id in outcomes_by_id]
        passed = sum(1 for row in rows if row.get("status") in {"pass", "pass_with_fallback"})
        failed = sum(1 for row in rows if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"})
        metrics[tag] = {
            "planned": len(planned_rows),
            "completed": len(rows),
            "passed": passed,
            "failed": failed,
            "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
            "pass_rate": round(passed / (passed + failed), 4) if passed + failed else None,
        }
    metrics["rates"] = {
        "correct_tool_selection": metrics["t1"]["pass_rate"],
        "valid_argument_schema": metrics["t2"]["pass_rate"],
        "completed_continuation": metrics["t3"]["pass_rate"],
        "streamed_tool_assembly": metrics["t5"]["pass_rate"],
        "parallel_tools": metrics["t6"]["pass_rate"],
        "replay_contract": metrics["t7"]["pass_rate"],
    }

    # Dual denominators (L3-R1). Service availability counts every planned,
    # non-preflight-skipped tool case — a candidate-free 503 stays a failure here, with its
    # root selection reason visible. Conditional competence counts only cases that reached
    # a model and produced a protocol-valid response; budget-inconclusive cases are
    # excluded from both numerator and denominator, never deleted.
    availability_deno = [row for row in tagged if row.get("status") != "skipped"]
    availability_num = sum(1 for row in availability_deno if row.get("status") in {"pass", "pass_with_fallback"})
    def _staged(row: dict[str, Any]) -> dict[str, Any]:
        staged = row.get("staged_verdict")
        return staged if isinstance(staged, dict) else {}
    competence_deno = [
        row
        for row in availability_deno
        if _staged(row).get("protocol_valid") is True and _staged(row).get("tool_semantic_valid") is not None
    ]
    competence_num = sum(1 for row in competence_deno if _staged(row).get("tool_semantic_valid") is True)
    inconclusive = [
        str(row.get("case_id") or "")
        for row in availability_deno
        if isinstance(row.get("axes"), dict)
        and isinstance(row["axes"].get("semantic"), dict)
        and row["axes"]["semantic"].get("inconclusive")
    ]
    excluded_candidate_free = [
        str(row.get("case_id") or "")
        for row in availability_deno
        if isinstance(row.get("route"), dict) and row["route"].get("candidate_count") == 0
    ]
    metrics["service_availability"] = {
        "planned": len(availability_deno),
        "succeeded": availability_num,
        "rate": round(availability_num / len(availability_deno), 4) if availability_deno else None,
    }
    metrics["conditional_competence"] = {
        "reached_and_protocol_valid": len(competence_deno),
        "semantic_passed": competence_num,
        "rate": round(competence_num / len(competence_deno), 4) if competence_deno else None,
        "excluded_candidate_free_case_ids": excluded_candidate_free[:50],
        "excluded_candidate_free": len(excluded_candidate_free),
        "inconclusive_budget_exhausted": inconclusive[:50],
    }

    # Separate T1 selection vs T2 typed-argument counts from the shared upstream request
    # (L3-R1), read off the per-assertion evidence the tool-choice validator records.
    selection_counts = {"passed": 0, "failed": 0}
    argument_counts = {"passed": 0, "failed": 0}
    for row in tagged:
        axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
        semantic = axes.get("semantic") if isinstance(axes.get("semantic"), dict) else {}
        assertions = semantic.get("tool_assertions") if isinstance(semantic.get("tool_assertions"), dict) else None
        if assertions is None:
            continue
        if assertions.get("selection") is True:
            selection_counts["passed"] += 1
        elif assertions.get("selection") is False:
            selection_counts["failed"] += 1
        if assertions.get("arguments") is True:
            argument_counts["passed"] += 1
        elif assertions.get("arguments") is False:
            argument_counts["failed"] += 1
    metrics["t1"]["selection"] = selection_counts
    metrics["t2"]["arguments"] = argument_counts

    # Per-feature schema-dialect matrix (L3-R3): verified support by provider/model and
    # feature, separate from the broad `tools` capability.
    dialects: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
        if "dialect" not in case.tags:
            continue
        row = outcomes_by_id.get(case.id)
        feature = str(case.expected.get("feature") or "")
        if not feature or row is None:
            continue
        model_key = str(case.target_id)
        status = str(row.get("status") or "")
        axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
        semantic = axes.get("semantic") if isinstance(axes.get("semantic"), dict) else {}
        skip_reason = str(row.get("skip_reason") or "")
        if status in {"pass", "pass_with_fallback"} and not semantic.get("inconclusive"):
            verdict = "supported"
        elif semantic.get("inconclusive"):
            verdict = "inconclusive"
        elif skip_reason.startswith("tool_support_absent"):
            # The control proved this model refuses user-provided tools entirely: the
            # feature is not what failed, so it must not be recorded as a feature failure.
            verdict = "no_tool_support"
        elif skip_reason.startswith(("dialect_control_not_run", "dialect_control_failed")):
            verdict = "unknown"
        elif status == "skipped":
            verdict = "unknown"
        else:
            classified = str(semantic.get("classified_reason") or "")
            verdict = "rejected_by_provider" if classified in {"bad_upstream_request", "unsupported_tool_schema"} else "failed"
        row_entry: dict[str, Any] = {"verdict": verdict, "case_id": case.id}
        if skip_reason:
            row_entry["reason"] = skip_reason
        dialects.setdefault(model_key, {})[feature] = row_entry
    metrics["schema_dialects"] = dialects
    return metrics


def build_summary(
    paths: RunPaths,
    run_metadata: dict[str, Any],
    inventory: dict[str, Any],
    cases: list[SyntheticHealthCase],
    outcomes: list[dict[str, Any]],
    *,
    incomplete: bool = False,
) -> dict[str, Any]:
    counts = Counter(str(row.get("status") or "unknown") for row in outcomes)
    findings = _finding_rows(outcomes)
    signature_rows: dict[str, dict[str, Any]] = {}
    for row in outcomes:
        for signature in row.get("failure_signatures") or []:
            if isinstance(signature, dict) and signature.get("signature"):
                signature_rows[str(signature["signature"])] = signature
    if counts.get("safety_breach"):
        overall = "safety_breach"
    elif incomplete or counts.get("harness_error") or len(outcomes) < len(cases):
        overall = "incomplete"
    elif counts.get("fail") or counts.get("degraded"):
        overall = "fail"
    elif any(
        row.get("status") == "skipped" and row.get("skip_reason") != "parallel_tools_not_claimed"
        for row in outcomes
    ):
        overall = "degraded"
    else:
        overall = "pass"

    virtual = [row for row in outcomes if (row.get("target") or {}).get("kind") == "virtual-profile"]
    direct = [row for row in outcomes if (row.get("target") or {}).get("kind") == "direct-model"]
    providers = inventory.get("providers") if isinstance(inventory.get("providers"), list) else []
    eligible_models = [
        model
        for model in inventory.get("models", [])
        if isinstance(model, dict) and model.get("strict_zero_eligible") and model.get("can_invoke")
    ]
    eligible_sources = {str(model.get("source") or "") for model in eligible_models}
    for provider in providers:
        if not isinstance(provider, dict) or not provider.get("enabled"):
            continue
        source = str(provider.get("source") or "unknown")
        case_id: str | None = None
        status = "skipped"
        diagnostic: str | None = None
        if not provider.get("can_invoke"):
            case_id = f"provider-access:{source}"
            diagnostic = f"enabled provider is not invokable: {provider.get('access_reason') or 'access unavailable'}"
        elif provider.get("catalog_error"):
            case_id = f"provider-catalog:{source}"
            status = "degraded"
            diagnostic = f"provider catalog failed: {provider.get('catalog_error')}"
        elif source not in eligible_sources:
            case_id = f"provider-coverage:{source}"
            diagnostic = "invokable provider has no strict-zero-eligible model in the initial catalog"
        if case_id is None or diagnostic is None:
            continue
        findings.append(
            {
                "id": stable_hash({"provider": source, "diagnostic": diagnostic})[:20],
                "case_id": case_id,
                "status": status,
                "failed_axes": ["coverage"],
                "diagnostic": sanitize_error_detail(diagnostic, 500),
                "source": source,
            }
        )
    if overall == "pass" and any(
        str(finding.get("case_id") or "").startswith("provider-")
        for finding in findings
    ):
        overall = "degraded"
    outcomes_by_id = {str(row.get("case_id") or ""): row for row in outcomes}
    direct_case_plans: dict[str, list[SyntheticHealthCase]] = {}
    for case in cases:
        if case.target_kind == "direct-model":
            direct_case_plans.setdefault(case.target_id, []).append(case)
    covered_model_ids: set[str] = set()
    partially_covered_model_ids: set[str] = set()
    blocked_model_ids: set[str] = set()
    for model in eligible_models:
        model_id = str(model.get("id") or "")
        model_cases = direct_case_plans.get(model_id, [])
        if model.get("runtime_block_reason"):
            blocked_model_ids.add(model_id)
            continue
        required_cases = [case for case in model_cases if not case.skip_reason]
        executed_count = sum(
            1
            for case in required_cases
            if case.id in outcomes_by_id and outcomes_by_id[case.id].get("status") != "skipped"
        )
        if required_cases and executed_count == len(required_cases):
            covered_model_ids.add(model_id)
        elif executed_count:
            partially_covered_model_ids.add(model_id)
    correlated = [
        row
        for row in outcomes
        if row.get("lane") == "live"
        and row.get("status") != "skipped"
        and (row.get("target") or {}).get("kind") != "service"
    ]
    route_complete = sum(1 for row in correlated if row.get("route_match_count") == 1)
    pending = max(0, len(cases) - len(outcomes))
    pre_virtual = [row for row in virtual if row.get("phase") == 2]
    post_virtual = [row for row in virtual if row.get("phase") == 4]
    failure_clusters: Counter[tuple[str, str, str, str]] = Counter()
    for row in outcomes:
        route = row.get("route") if isinstance(row.get("route"), dict) else {}
        for attempt in _route_attempts(route):
            reason = str(attempt.get("reason") or "")
            if reason and reason != "ok":
                failure_clusters[
                    (
                        str(attempt.get("source") or row.get("source") or "unknown"),
                        str(attempt.get("model") or "unknown"),
                        str(attempt.get("status") or "unknown"),
                        reason,
                    )
                ] += 1
    induced_sections = Counter(
        section
        for row in outcomes
        for section in (row.get("state_changes") or {})
    )
    surface = _test_surface(inventory, cases)
    # The availability block of the artifact contract (L3-R1): one row per rung, plus every
    # failure bucketed by exactly one primary layer.
    non_skipped = [row for row in outcomes if row.get("status") not in {"skipped", "planned"}]
    failures_by_layer = Counter(
        str(row.get("primary_failure_layer") or "unclassified")
        for row in outcomes
        if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"}
    )
    def _rung(row: dict[str, Any], name: str) -> bool:
        staged = row.get("staged_verdict")
        return isinstance(staged, dict) and staged.get(name) is True
    availability_block = {
        "planned": len(cases),
        "completed": len(outcomes),
        "selection_reached": sum(1 for row in non_skipped if _rung(row, "selection_reachable")),
        "upstream_reached": sum(1 for row in non_skipped if _rung(row, "upstream_reached")),
        "protocol_completed": sum(1 for row in non_skipped if _rung(row, "protocol_valid")),
        "end_user_success": sum(1 for row in non_skipped if row.get("status") in {"pass", "pass_with_fallback"}),
        "failures_by_layer": dict(sorted(failures_by_layer.items())),
    }
    # Catalog continuity block (artifact contract): both sides of the run's catalog story.
    transitions = 0
    try:
        if paths.catalog_transitions.exists():
            transitions = sum(1 for line in paths.catalog_transitions.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        transitions = 0
    catalog_block = {
        "initial_identity": run_metadata.get("catalog_identity"),
        "final_identity": run_metadata.get("catalog_identity_after"),
        "transition_count": transitions,
    }
    # Capacity and incident aggregation (L4-R1/R2/R4). Incidents are keyed by their real
    # dimension per provider; endpoint incidents deduplicate the sources sharing one failed
    # hostname; depletion is measured, never treated as a product invariant violation.
    capacity_reasons = {"rate_limited", "rate_limited_upstream", "request_too_large", "quota_exhausted", "no_free_quota"}
    incidents_by_pool: dict[str, dict[str, int]] = {}
    first_capacity_failure: dict[str, str] = {}
    run_started = str(run_metadata.get("started_at") or "")
    for row in outcomes:
        axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
        semantic = axes.get("semantic") if isinstance(axes.get("semantic"), dict) else {}
        classified = str(semantic.get("classified_reason") or "")
        if classified not in capacity_reasons:
            continue
        source = str(row.get("source") or "unknown")
        dimension = str(semantic.get("capacity_dimension") or "capacity.unknown")
        incidents_by_pool.setdefault(source, {})
        incidents_by_pool[source][dimension] = incidents_by_pool[source].get(dimension, 0) + 1
        started_at = str(row.get("started_at") or "")
        if started_at and (source not in first_capacity_failure or started_at < first_capacity_failure[source]):
            first_capacity_failure[source] = started_at
    time_to_depletion: dict[str, float] = {}
    for source, first_at in first_capacity_failure.items():
        try:
            delta = (
                datetime.fromisoformat(first_at) - datetime.fromisoformat(run_started)
            ).total_seconds()
            if delta >= 0:
                time_to_depletion[source] = round(delta, 1)
        except ValueError:
            continue
    endpoint_sources: dict[str, list[str]] = {}
    account_groups: dict[str, list[str]] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_source = str(provider.get("source") or "")
        catalog_error = str(provider.get("catalog_error") or "")
        host_match = re.search(r"host='([^']+)'|Failed to resolve '([^']+)'|https?://([^/\s\"']+)", catalog_error)
        if catalog_error and host_match:
            hostname = next(group for group in host_match.groups() if group)
            endpoint_sources.setdefault(hostname, []).append(provider_source)
        account_id = str(provider.get("provider_account_id") or "")
        if account_id:
            account_groups.setdefault(account_id, []).append(provider_source)
    post_sweep_by_layer = Counter(
        str(row.get("primary_failure_layer") or "unclassified")
        for row in post_virtual
        if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"}
    )
    pre_sweep_by_layer = Counter(
        str(row.get("primary_failure_layer") or "unclassified")
        for row in pre_virtual
        if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"}
    )
    capacity_block = {
        "incidents_by_pool": {source: dict(sorted(dims.items())) for source, dims in sorted(incidents_by_pool.items())},
        "time_to_depletion_seconds": dict(sorted(time_to_depletion.items())),
        "endpoint_incidents": [
            {"hostname": hostname, "sources": sorted(sources), "source_count": len(sources)}
            for hostname, sources in sorted(endpoint_sources.items())
        ],
        "shared_account_groups": [
            {"sources": sorted(sources), "source_count": len(sources)}
            for account_id, sources in sorted(account_groups.items())
            if len(sources) > 1
        ],
        "pre_sweep_failures_by_layer": dict(sorted(pre_sweep_by_layer.items())),
        "post_sweep_failures_by_layer": dict(sorted(post_sweep_by_layer.items())),
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_metadata.get("run_id"),
        "parent_run_id": run_metadata.get("parent_run_id"),
        "status": overall,
        "started_at": run_metadata.get("started_at"),
        "completed_at": now_iso(),
        "identity": {
            "ficelle_version": run_metadata.get("ficelle_version"),
            "corpus_version": run_metadata.get("corpus_version"),
            "corpus_hash": run_metadata.get("corpus_hash"),
            "config_structural_fingerprint": run_metadata.get("config_structural_fingerprint"),
            "catalog_identity": run_metadata.get("catalog_identity"),
            "build_identity": run_metadata.get("build_identity"),
            "service_build_identity": run_metadata.get("service_build_identity"),
        },
        "surface": surface,
        "artifacts": {
            "run": str(paths.run),
            "cases": str(paths.cases),
            "catalog_snapshot": str(paths.catalog_snapshot),
            "catalog_after": str(paths.catalog_after),
            "catalog_transitions": str(paths.catalog_transitions),
            "state_before": str(paths.state_before),
            "state_after": str(paths.state_after),
            "summary_json": str(paths.summary_json),
            "summary_markdown": str(paths.summary_markdown),
        },
        "cases": {"planned": len(cases), "completed": len(outcomes), "pending": pending, "by_status": dict(sorted(counts.items()))},
        "availability": availability_block,
        "catalog": catalog_block,
        "capacity": capacity_block,
        "health": {
            "router_service": {
                "pre_sweep": {
                    "planned": len(pre_virtual),
                    "passed": sum(1 for row in pre_virtual if row.get("status") in {"pass", "pass_with_fallback"}),
                    "failed": sum(1 for row in pre_virtual if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"}),
                    "success_rate": round(
                        sum(1 for row in pre_virtual if row.get("status") in {"pass", "pass_with_fallback"})
                        / len(pre_virtual),
                        4,
                    )
                    if pre_virtual
                    else None,
                },
                "post_sweep": {
                    "planned": len(post_virtual),
                    "passed": sum(1 for row in post_virtual if row.get("status") in {"pass", "pass_with_fallback"}),
                    "failed": sum(1 for row in post_virtual if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"}),
                    "success_rate": round(
                        sum(1 for row in post_virtual if row.get("status") in {"pass", "pass_with_fallback"})
                        / len(post_virtual),
                        4,
                    )
                    if post_virtual
                    else None,
                },
            },
            "provider_failures": [
                {"source": key[0], "model": key[1], "status": key[2], "reason": key[3], "count": count}
                for key, count in sorted(failure_clusters.items())
            ],
            "harness": {
                "incomplete": incomplete,
                "catalog_changed_during_run": bool(run_metadata.get("catalog_changed_during_run")),
            },
        },
        "coverage": {
            "virtual_profiles": {
                "configured": len(inventory.get("virtual_profiles", [])),
                "cases_planned": sum(1 for case in cases if case.target_kind == "virtual-profile"),
                "cases_completed": len(virtual),
                "passed": sum(1 for row in virtual if row.get("status") in {"pass", "pass_with_fallback"}),
                "failed": sum(1 for row in virtual if row.get("status") in {"fail", "degraded", "harness_error", "safety_breach"}),
                "skipped": sum(1 for row in virtual if row.get("status") == "skipped"),
            },
            "providers": {
                "configured": len(providers),
                "invokable": sum(1 for provider in providers if isinstance(provider, dict) and provider.get("can_invoke")),
                "covered": len(
                    {
                        str(row.get("source") or "")
                        for row in direct
                        if row.get("status") != "skipped" and row.get("source")
                    }
                ),
                "skipped": [
                    {"source": provider.get("source"), "reason": provider.get("access_reason")}
                    for provider in providers
                    if isinstance(provider, dict) and not provider.get("can_invoke")
                ],
                "catalog_failures": [
                    {"source": provider.get("source"), "error": provider.get("catalog_error")}
                    for provider in providers
                    if isinstance(provider, dict) and provider.get("can_invoke") and provider.get("catalog_error")
                ],
                "without_eligible_models": [
                    provider.get("source")
                    for provider in providers
                    if isinstance(provider, dict)
                    and provider.get("can_invoke")
                    and str(provider.get("source") or "") not in eligible_sources
                ],
            },
            "eligible_models": {
                "denominator": len(eligible_models),
                "covered": len(covered_model_ids),
                "partially_covered": len(partially_covered_model_ids),
                "blocked": len(blocked_model_ids),
                "not_fully_covered": len(eligible_models) - len(covered_model_ids) - len(blocked_model_ids),
                "direct_cases_planned": sum(len(rows) for rows in direct_case_plans.values()),
                "direct_cases_completed": len(direct),
            },
            "telemetry": {
                "live_cases": len(correlated),
                "exactly_correlated": route_complete,
                "completeness": round(route_complete / len(correlated), 4) if correlated else 1.0,
            },
            "tools": _tool_metrics(cases, outcomes),
        },
        "fallback": _fallback_metrics(counts, outcomes),
        "response_contract": {
            "deceptive_200_failures": sum(
                1
                for row in outcomes
                if "deceptive-200" in (row.get("tags") or []) and row.get("status") not in {"pass", "pass_with_fallback"}
            ),
            "stream_integrity_failures": sum(
                1
                for row in outcomes
                if "streaming" in (row.get("tags") or [])
                and (row.get("axes") or {}).get("transport", {}).get("status") == "fail"
            ),
        },
        "induced_state_observations": {
            "attribution": "observed_during_harness_case",
            "sections": dict(sorted(induced_sections.items())),
        },
        "failure_signatures": [signature_rows[key] for key in sorted(signature_rows)],
        "findings": findings,
    }
    previous = _previous_comparable_summary(paths, run_metadata)
    previous_findings = {str(item.get("id")): item for item in (previous or {}).get("findings", []) if isinstance(item, dict)}
    current_findings = {str(item.get("id")): item for item in findings}
    previous_signatures = {
        str(item.get("signature")): item
        for item in (previous or {}).get("failure_signatures", [])
        if isinstance(item, dict) and item.get("signature")
    }
    summary["comparison"] = {
        "previous_run_id": previous.get("run_id") if previous else None,
        "current_incomplete": incomplete,
        "identity_changed": {
            "ficelle_version": bool(previous) and (previous.get("identity") or {}).get("ficelle_version") != run_metadata.get("ficelle_version"),
            "corpus_hash": bool(previous) and (previous.get("identity") or {}).get("corpus_hash") != run_metadata.get("corpus_hash"),
            "config_structural_fingerprint": bool(previous)
            and (previous.get("identity") or {}).get("config_structural_fingerprint")
            != run_metadata.get("config_structural_fingerprint"),
            "build_identity": bool(previous)
            and (previous.get("identity") or {}).get("build_identity")
            != run_metadata.get("build_identity"),
            "service_build_identity": bool(previous)
            and (previous.get("identity") or {}).get("service_build_identity")
            != run_metadata.get("service_build_identity"),
        },
        "surface": _surface_changes(surface, previous.get("surface") or {}) if previous else None,
        "new": [current_findings[key] for key in sorted(current_findings.keys() - previous_findings.keys())],
        "persistent": [current_findings[key] for key in sorted(current_findings.keys() & previous_findings.keys())],
        "recovered": []
        if incomplete
        else [previous_findings[key] for key in sorted(previous_findings.keys() - current_findings.keys())],
        "new_error_signatures": [signature_rows[key] for key in sorted(signature_rows.keys() - previous_signatures.keys())],
        "persistent_error_signatures": [signature_rows[key] for key in sorted(signature_rows.keys() & previous_signatures.keys())],
        "recovered_error_signatures": []
        if incomplete
        else [previous_signatures[key] for key in sorted(previous_signatures.keys() - signature_rows.keys())],
    }
    return summary


def summary_markdown(summary: dict[str, Any]) -> str:
    cases = summary.get("cases") or {}
    coverage = summary.get("coverage") or {}
    telemetry = coverage.get("telemetry") or {}
    providers = coverage.get("providers") or {}
    eligible_models = coverage.get("eligible_models") or {}
    tools = coverage.get("tools") or {}
    tool_models = tools.get("models") or {}
    router_health = (summary.get("health") or {}).get("router_service") or {}
    pre_sweep = router_health.get("pre_sweep") or {}
    post_sweep = router_health.get("post_sweep") or {}
    fallback = summary.get("fallback") or {}
    lines = [
        f"# Synthetic user health — {summary.get('run_id')}",
        "",
        f"Overall status: **{summary.get('status')}**",
        "",
        "## Coverage",
        "",
        f"- Cases: {cases.get('completed', 0)}/{cases.get('planned', 0)} completed",
        f"- Virtual profiles: {(coverage.get('virtual_profiles') or {}).get('configured', 0)} configured",
        f"- Invokable providers: {providers.get('invokable', 0)}/{providers.get('configured', 0)}; covered: {providers.get('covered', 0)}",
        f"- Eligible models covered: {eligible_models.get('covered', 0)}/{eligible_models.get('denominator', 0)}; blocked: {eligible_models.get('blocked', 0)}; partial: {eligible_models.get('partially_covered', 0)}",
        f"- Tool-capable models covered: {tool_models.get('covered', 0)}/{tool_models.get('denominator', 0)}; tool cases: {tools.get('completed', 0)}/{tools.get('planned', 0)}",
        f"- Exact route-log correlation: {telemetry.get('exactly_correlated', 0)}/{telemetry.get('live_cases', 0)}",
        f"- Pre-sweep virtual success: {pre_sweep.get('passed', 0)}/{pre_sweep.get('planned', 0)}; post-sweep: {post_sweep.get('passed', 0)}/{post_sweep.get('planned', 0)}",
        f"- Pass with fallback: {fallback.get('pass_with_fallback', 0)}; legitimate: {fallback.get('legitimate', 0)}; "
        f"policy failures: {fallback.get('incorrect_or_missing', 0)} "
        f"(illegitimate decision: {fallback.get('illegitimate_decision', 0)}, "
        f"state scope mismatch: {fallback.get('state_scope_mismatch', 0)}, "
        f"correlation missing: {fallback.get('correlation_missing', 0)}, "
        f"client timeout n/a: {fallback.get('not_applicable_client_timeout', 0)}, "
        f"unknown: {fallback.get('unknown', 0)})",
        "",
        "## Findings",
        "",
    ]
    findings = summary.get("findings") or []
    if not findings:
        lines.append("No findings.")
    else:
        for finding in findings:
            lines.append(f"- `{finding.get('case_id')}` — **{finding.get('status')}**: {finding.get('diagnostic')}")
    comparison = summary.get("comparison") or {}
    surface = comparison.get("surface") or {}
    provider_changes = surface.get("providers") or {}
    model_changes = surface.get("eligible_models") or {}
    lines.extend(
        [
            "",
            "## Change from previous comparable run",
            "",
            f"- Previous run: {comparison.get('previous_run_id') or 'none'}",
            f"- New findings: {len(comparison.get('new') or [])}",
            f"- Persistent findings: {len(comparison.get('persistent') or [])}",
            f"- Recovered findings: {len(comparison.get('recovered') or [])}",
            f"- Providers added/removed: {len(provider_changes.get('added') or [])}/{len(provider_changes.get('removed') or [])}",
            f"- Eligible models added/removed: {len(model_changes.get('added') or [])}/{len(model_changes.get('removed') or [])}",
            f"- Capability claims changed: {len(surface.get('capability_claims') or [])}",
            "",
            "Detailed request evidence is in `cases.jsonl`; prompts are represented only by hashes and shapes.",
            "",
        ]
    )
    return "\n".join(lines)


def catalog_identity(inventory: dict[str, Any]) -> str:
    stable_models = sorted(
        (
            {
                key: value
                for key, value in row.items()
                if key not in {"can_invoke", "runtime_block_reason"}
            }
            for row in inventory.get("models", [])
            if isinstance(row, dict)
        ),
        key=lambda row: str(row.get("id") or ""),
    )
    stable_providers = sorted(
        (
            {
                "source": row.get("source"),
                "enabled": row.get("enabled"),
            }
            for row in inventory.get("providers", [])
            if isinstance(row, dict)
        ),
        key=lambda row: str(row.get("source") or ""),
    )
    return stable_hash(
        {
            "providers": stable_providers,
            "models": stable_models,
            "virtual_profiles": inventory.get("virtual_profiles", []),
            "virtual_profile_definitions": inventory.get("virtual_profile_definitions", {}),
        }
    )


def _service_preflight(
    base_url: str,
    timeout_seconds: float = 5.0,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    started = time.monotonic()
    try:
        response = requests.get(
            base_url.rsplit("/v1", 1)[0] + "/health",
            headers=headers,
            timeout=timeout_seconds,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        ok = response.status_code == 200 and isinstance(payload, dict)
        evidence: dict[str, Any] = {
            "http_status": response.status_code,
            "latency_seconds": round(time.monotonic() - started, 4),
            "response_preview": sanitize_preview(payload),
        }
        if ok:
            # A healthy answer proves a service listens, not that it is THIS build:
            # run.json's build_identity describes the harness process only, so without
            # this comparison a rerun happily validates last week's wheel (2026-08-10).
            service_build = payload.get("build") if isinstance(payload.get("build"), dict) else None
            verdict = build_identity_match(package_build_identity(), service_build)
            evidence["service_build_identity"] = service_build
            evidence["build_match"] = verdict["match"]
            if not verdict["match"]:
                evidence["build_diagnostic"] = verdict["diagnostic"]
                ok = False
        return ok, evidence
    except requests.RequestException as exc:
        return False, {
            "latency_seconds": round(time.monotonic() - started, 4),
            "transport_error": sanitize_error_detail(f"{type(exc).__name__}: {exc}", 500),
        }


def _preflight_outcome(case: SyntheticHealthCase, ok: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    # A build mismatch is a semantic refusal, not an unreachable service: transport and
    # protocol both succeeded, and blaming them would send the operator to the wrong fix.
    reached = bool(ok or evidence.get("build_match") is False)
    axes = {
        "transport": _axis("pass" if reached else "fail", "loopback service is reachable" if reached else "loopback service is unavailable"),
        "protocol": _axis("pass" if reached else "fail", "health endpoint returned JSON" if reached else "health endpoint did not return a valid response"),
        "semantic": _axis(
            "pass" if ok else "fail",
            "service is healthy and runs the harness build"
            if ok
            else str(evidence.get("build_diagnostic") or "service health could not be established"),
        ),
        "routing": _axis("not_applicable", "preflight does not route a model"),
        "policy": _axis("not_applicable", "preflight does not route a model"),
        "telemetry": _axis("not_applicable", "preflight uses the health endpoint"),
        "safety": _axis("pass", "preflight sends no provider request"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "event": "outcome",
        "case_id": case.id,
        "phase": case.phase,
        "lane": case.lane,
        "target": {"kind": case.target_kind, "id": case.target_id},
        "tags": list(case.tags),
        "started_at": now_iso(),
        "ended_at": now_iso(),
        "status": "pass" if ok else "harness_error",
        "axes": axes,
        "request": evidence,
    }


def _base_url(config: dict[str, Any]) -> str:
    port = config.get("port", 8646)
    try:
        port_number = int(port)
    except (TypeError, ValueError):
        port_number = 8646
    return f"http://127.0.0.1:{port_number}/v1"


def _service_request_headers(router: Any, config: dict[str, Any]) -> dict[str, str]:
    configured_host = str(config.get("host") or "127.0.0.1").strip().lower()
    loopback_hosts = set(getattr(router, "LOOPBACK_BIND_HOSTS", {"127.0.0.1", "localhost", "::1"}))
    return {} if configured_host in loopback_hosts else {"Authorization": f"Bearer {router.api_token()}"}


def _run_public_result(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "cases": summary.get("cases"),
        "coverage": summary.get("coverage"),
        "fallback": summary.get("fallback"),
        "artifacts": summary.get("artifacts"),
        "finding_count": len(summary.get("findings") or []),
    }


def exit_code_for_summary(summary: dict[str, Any]) -> int:
    status = summary.get("status")
    if status == "pass":
        return 0
    if status == "safety_breach":
        return 3
    if status == "incomplete":
        return 2
    return 1


def _load_router() -> Any:
    from ficelle import router

    return router


def _runtime_material(router: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = router.load_runtime_config()
    catalog = router.load_or_refresh_catalog(config)
    inventory = build_inventory(router, config, catalog)
    return config, catalog, inventory


# Disjoint fallback-failure verdicts (L2-R5). Exactly one applies per policy failure; a
# missing request id can never be summarized as a wrong retry decision.
def _fallback_metrics(counts: dict[str, int], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, list[str]] = {name: [] for name in FALLBACK_FAILURE_CATEGORIES}
    legitimate = 0
    for row in outcomes:
        axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
        policy = axes.get("policy") if isinstance(axes.get("policy"), dict) else {}
        if row.get("status") == "pass_with_fallback" and policy.get("status") == "pass":
            legitimate += 1
        if policy.get("status") != "fail":
            continue
        category = str(policy.get("fallback_category") or FALLBACK_UNKNOWN)
        if category not in categories:
            category = FALLBACK_UNKNOWN
        categories[category].append(str(row.get("case_id") or ""))
    total_failures = sum(len(case_ids) for case_ids in categories.values())
    return {
        "pass_with_fallback": counts.get("pass_with_fallback", 0),
        "legitimate": legitimate,
        # Backward-comparison total; the disjoint categories below are the real verdicts.
        "incorrect_or_missing": total_failures,
        **{name: len(case_ids) for name, case_ids in categories.items()},
        "failure_case_ids": {name: case_ids for name, case_ids in categories.items() if case_ids},
    }


def _catalog_transition_event(
    *,
    previous_identity: str | None,
    current_identity: str,
    previous_models: dict[str, dict[str, Any]],
    current_models: dict[str, dict[str, Any]],
    refresh: dict[str, Any] | None,
) -> dict[str, Any]:
    """One observed active-generation change, with per-provider added/removed/stale counts
    (L1-R2). Pure so the transition shape is testable without a run loop."""
    added_by_source = Counter(
        str(current_models[model_id].get("source") or "")
        for model_id in current_models.keys() - previous_models.keys()
    )
    removed_by_source = Counter(
        str(previous_models[model_id].get("source") or "")
        for model_id in previous_models.keys() - current_models.keys()
    )
    stale_by_source = Counter(
        str(model.get("source") or "")
        for model in current_models.values()
        if model.get("catalog_stale") and model.get("source")
    )
    per_provider: dict[str, dict[str, int]] = {}
    for source in sorted(set(added_by_source) | set(removed_by_source) | set(stale_by_source)):
        row = {
            "added": added_by_source.get(source, 0),
            "removed": removed_by_source.get(source, 0),
            "stale": stale_by_source.get(source, 0),
        }
        if any(row.values()):
            per_provider[source] = row
    return {
        "schema_version": SCHEMA_VERSION,
        "event": "catalog_transition",
        "at": now_iso(),
        "previous_identity": previous_identity,
        "current_identity": current_identity,
        "previous_model_count": len(previous_models),
        "current_model_count": len(current_models),
        "providers": per_provider,
        "refresh": refresh,
    }


def _write_run_metadata(paths: RunPaths, metadata: dict[str, Any]) -> None:
    atomic_write_json(paths.run, metadata)


def _phase_snapshot_path(paths: RunPaths, phase: int) -> Path:
    return paths.root / f"state-phase-{phase}.json"


def _finalize_run(
    router: Any,
    paths: RunPaths,
    metadata: dict[str, Any],
    inventory: dict[str, Any],
    cases: list[SyntheticHealthCase],
    *,
    incomplete: bool,
) -> dict[str, Any]:
    state_after = router.load_runtime_state()
    atomic_write_json(paths.state_after, state_after)
    try:
        current_catalog = router.load_catalog_document()
        current_inventory = build_inventory(router, router.load_runtime_config(), current_catalog)
        metadata["catalog_identity_after"] = catalog_identity(current_inventory)
        metadata["catalog_changed_during_run"] = metadata["catalog_identity_after"] != metadata.get("catalog_identity")
        # Both sides of every catalog transition are now reconstructable (L1-R2): the
        # final inventory ships in the same redacted shape as catalog-snapshot.json.
        atomic_write_json(paths.catalog_after, current_inventory)
    except Exception:
        metadata["catalog_identity_after"] = None
        metadata["catalog_changed_during_run"] = None
    # Same bookend as catalog_identity_after, for the service build: the preflight only
    # proves the build at t=0, and a service restarted onto another build mid-run would
    # otherwise produce a contaminated but complete-looking journal.
    try:
        _, end_evidence = _service_preflight(
            str(metadata.get("service_base_url") or _base_url(router.load_runtime_config())),
            headers=_service_request_headers(router, router.load_runtime_config()),
        )
        after = end_evidence.get("service_build_identity")
        before = metadata.get("service_build_identity")
        metadata["service_build_identity_after"] = after
        metadata["service_build_changed_during_run"] = (
            after.get("runtime_content_hash") != before.get("runtime_content_hash")
            if isinstance(after, dict) and isinstance(before, dict)
            else None
        )
    except Exception:
        metadata["service_build_identity_after"] = None
        metadata["service_build_changed_during_run"] = None
    outcomes = list(latest_case_outcomes(paths.cases).values())
    summary = build_summary(paths, metadata, inventory, cases, outcomes, incomplete=incomplete)
    atomic_write_json(paths.summary_json, summary)
    atomic_write_text(paths.summary_markdown, summary_markdown(summary))
    metadata["status"] = "completed" if not incomplete else "incomplete"
    metadata["completed_at"] = summary["completed_at"]
    metadata["overall_status"] = summary["status"]
    metadata["completed_cases"] = len(outcomes)
    _write_run_metadata(paths, metadata)
    return summary


def _run_cases(
    router: Any,
    *,
    paths: RunPaths,
    metadata: dict[str, Any],
    config: dict[str, Any],
    catalog: dict[str, Any],
    inventory: dict[str, Any],
    cases: list[SyntheticHealthCase],
    max_duration_seconds: float | None,
) -> dict[str, Any]:
    outcomes = latest_case_outcomes(paths.cases)
    base_url = _base_url(config)
    request_headers = _service_request_headers(router, config)
    client = LoopbackChatClient(base_url, headers=request_headers)
    deadline = time.monotonic() + max_duration_seconds if max_duration_seconds and max_duration_seconds > 0 else None
    last_source_at: dict[str, float] = {}

    def _models_by_id(source_catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(model.get("id") or ""): model
            for model in source_catalog.get("models", [])
            if isinstance(model, dict) and model.get("id")
        }

    providers_cfg = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    # The harness's active selection view (L1-R3): refreshed at least once per provider
    # batch and observed for identity changes, so a mid-run catalog transition becomes an
    # explicit surface-change record instead of a wave of invented capability failures.
    active_models = _models_by_id(catalog)
    active_identity: str | None = metadata.get("catalog_identity") or None
    active_generation: str | None = str(catalog.get("generated_at") or "") or None

    def refresh_active_view() -> None:
        nonlocal active_models, active_identity, active_generation
        try:
            current_catalog = router.load_catalog_document()
            # Every publish stamps a fresh generated_at, so an unchanged stamp means an
            # unchanged document: skip the expensive inventory rebuild (provider credential
            # probes, per-model redaction, full identity hash) on the common no-change path.
            current_generation = str(current_catalog.get("generated_at") or "") or None
            if current_generation == active_generation:
                return
            current_inventory = build_inventory(router, router.load_runtime_config(), current_catalog)
            current_identity = catalog_identity(current_inventory)
        except Exception:
            return
        if current_identity == active_identity:
            active_generation = current_generation
            return
        current_models = _models_by_id(current_catalog)
        state_refresh = router.load_runtime_state().get("catalog_refresh")
        append_jsonl(
            paths.catalog_transitions,
            _catalog_transition_event(
                previous_identity=active_identity,
                current_identity=current_identity,
                previous_models=active_models,
                current_models=current_models,
                refresh=state_refresh if isinstance(state_refresh, dict) else None,
            ),
        )
        active_models = current_models
        active_identity = current_identity
        active_generation = current_generation

    incomplete = False
    current_phase: int | None = None
    for case in cases:
        # A completed non-preflight case never re-runs on resume. The preflight always
        # does: the process listening now is not necessarily the one that passed it before
        # the interruption, and resuming against a swapped build is exactly the
        # contamination the preflight exists to refuse.
        if case.id in outcomes and case.validator != "service-preflight":
            continue
        if deadline is not None and time.monotonic() >= deadline:
            incomplete = True
            metadata["interruption_reason"] = "wall_clock_limit_reached"
            break
        if current_phase != case.phase:
            if current_phase is not None:
                atomic_write_json(_phase_snapshot_path(paths, current_phase), router.load_runtime_state())
            current_phase = case.phase
            metadata["current_phase"] = current_phase
            _write_run_metadata(paths, metadata)
        append_jsonl(paths.cases, _planned_row(str(metadata["run_id"]), case))
        try:
            if case.validator == "service-preflight":
                ok, evidence = _service_preflight(base_url, headers=request_headers)
                outcome = _preflight_outcome(case, ok, evidence)
                # Record what the service said it was running next to the harness's own
                # build_identity — the pair is what makes a contaminated rerun readable.
                metadata["service_build_identity"] = evidence.get("service_build_identity")
                metadata["service_build_match"] = evidence.get("build_match")
                if not ok:
                    incomplete = True
                    metadata["interruption_reason"] = (
                        "service_build_mismatch"
                        if evidence.get("build_match") is False
                        else "service_unreachable"
                    )
            elif case.lane == "deterministic":
                outcome = execute_deterministic_case(case)
            else:
                # The generated_at pre-check makes this cheap when nothing changed. Checking
                # before every live case is what catches a transition between two cases from
                # the same provider, not only at provider-batch boundaries.
                refresh_active_view()
                eligibility: dict[str, Any] | None = None
                if case.target_kind == "direct-model" and not case.skip_reason:
                    model = active_models.get(case.model_id or case.target_id)
                    eligibility = execution_eligibility(
                        router,
                        model,
                        router.load_runtime_state(),
                        providers_cfg.get(case.source) if case.source else None,
                        catalog_generation=active_generation,
                        runtime_config=config,
                    )
                if eligibility is not None and not eligibility.get("eligible"):
                    outcome = _skipped_outcome(
                        case,
                        str(eligibility.get("reason") or f"blocked:{eligibility.get('scope')}"),
                        eligibility=eligibility,
                    )
                else:
                    if case.source and not case.skip_reason:
                        interval = provider_probe_interval_seconds(config, case.source)
                        previous = last_source_at.get(case.source)
                        if previous is not None:
                            wait = interval - (time.monotonic() - previous)
                            if wait > 0:
                                time.sleep(wait)
                        last_source_at[case.source] = time.monotonic()
                    timeout = DEFAULT_TIMEOUTS.get(case.timeout_class, DEFAULT_TIMEOUTS["fast"])
                    outcome = execute_live_case(
                        router,
                        case,
                        client=client,
                        route_log_path=router.ROUTE_LOG_PATH,
                        catalog=catalog,
                        outcomes=outcomes,
                        timeout_seconds=timeout,
                    )
        except Exception as exc:
            axes = {name: _axis("not_applicable", "execution aborted") for name in ("transport", "protocol", "semantic", "routing", "policy", "telemetry", "safety")}
            axes["transport"] = _axis("fail", sanitize_error_detail(f"{type(exc).__name__}: {exc}", 500) or "case execution failed")
            outcome = {
                "schema_version": SCHEMA_VERSION,
                "event": "outcome",
                "case_id": case.id,
                "phase": case.phase,
                "lane": case.lane,
                "target": {"kind": case.target_kind, "id": case.target_id},
                "source": case.source,
                "tags": list(case.tags),
                "started_at": now_iso(),
                "ended_at": now_iso(),
                "status": "harness_error",
                "axes": axes,
            }
        outcome["run_id"] = metadata["run_id"]
        append_jsonl(paths.cases, outcome)
        outcomes[case.id] = outcome
        metadata["completed_cases"] = len(outcomes)
        _write_run_metadata(paths, metadata)
        if outcome.get("status") == "safety_breach":
            incomplete = True
            metadata["interruption_reason"] = "safety_breach"
            break
        if case.validator == "service-preflight" and outcome.get("status") != "pass":
            break
    if current_phase is not None:
        atomic_write_json(_phase_snapshot_path(paths, current_phase), router.load_runtime_state())
    return _finalize_run(router, paths, metadata, inventory, cases, incomplete=incomplete)


def run_synthetic_health(
    ficelle_home: Path,
    *,
    depth: str = "deep",
    max_duration_seconds: float | None = None,
    selected_case_ids: set[str] | None = None,
    parent_run_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    if depth != "deep":
        raise SyntheticHealthError("only the deep synthetic-health corpus is currently defined")
    router = _load_router()
    with file_lock(synthetic_health_lock_path(ficelle_home), blocking=False) as run_lock:
        if not run_lock:
            raise SyntheticHealthBusy("another synthetic-health run is active for this FICELLE_HOME")
        with file_lock(probe_lock_path(ficelle_home), blocking=False) as probe_lock:
            if not probe_lock:
                raise SyntheticHealthBusy("benchmark, canary, or discovery probes are already active")
            run_id = run_id_now()
            paths = RunPaths.for_run(ficelle_home, run_id)
            config, catalog, inventory = _runtime_material(router)
            health_config = config.get("synthetic_health") if isinstance(config.get("synthetic_health"), dict) else {}
            configured_max_duration = health_config.get(
                "max_duration_seconds",
                DEFAULT_MAX_DURATION_SECONDS,
            )
            if max_duration_seconds is None and configured_max_duration is not None:
                try:
                    max_duration_seconds = float(configured_max_duration)
                except (TypeError, ValueError):
                    raise SyntheticHealthError("synthetic_health.max_duration_seconds must be numeric")
            if max_duration_seconds is not None and max_duration_seconds <= 0:
                raise SyntheticHealthError("max_duration_seconds must be positive")
            retention_runs, retention_days, pinned_run_ids = normalized_retention_policy(
                health_config.get("retention_runs", DEFAULT_RETENTION_RUNS),
                health_config.get("retention_days", DEFAULT_RETENTION_DAYS),
                health_config.get("pinned_run_ids", []),
            )
            cases = build_corpus(router, config, catalog, inventory, run_id=run_id)
            if selected_case_ids is not None:
                cases = select_cases_with_dependencies(cases, selected_case_ids)
            ensure_private_dir(paths.root)
            # Keep the run artifact contract stable even when no catalog transition occurs.
            # append_jsonl creates this file on the first change; the empty file represents
            # the equally meaningful observation "no active generation change".
            atomic_write_text(paths.catalog_transitions, "")
            identity = catalog_identity(inventory)
            metadata: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "status": "running",
                "depth": depth,
                "started_at": now_iso(),
                "executable": sys.executable,
                "ficelle_home": str(ficelle_home),
                "service_base_url": _base_url(config),
                "ficelle_version": __version__,
                "corpus_version": CORPUS_VERSION,
                "corpus_hash": corpus_hash(cases),
                "build_identity": package_build_identity(),
                "config_structural_fingerprint": router.catalog_config_structural_fingerprint(config),
                "catalog_identity": identity,
                "case_count": len(cases),
                "case_plan": [case.public_plan() for case in cases],
                "parent_run_id": parent_run_id,
                "max_duration_seconds": max_duration_seconds,
            }
            previous = _previous_comparable_summary(paths, metadata)
            metadata["previous_comparable_run_id"] = previous.get("run_id") if previous else None
            _write_run_metadata(paths, metadata)
            atomic_write_json(paths.catalog_snapshot, inventory)
            atomic_write_json(paths.state_before, router.load_runtime_state())
            summary = _run_cases(
                router,
                paths=paths,
                metadata=metadata,
                config=config,
                catalog=catalog,
                inventory=inventory,
                cases=cases,
                max_duration_seconds=max_duration_seconds,
            )
            prune_health_runs(
                ficelle_home,
                keep_runs=retention_runs,
                max_age_days=retention_days,
                pinned_run_ids=pinned_run_ids,
            )
            return _run_public_result(summary), exit_code_for_summary(summary)


def health_runs_root(ficelle_home: Path) -> Path:
    return ficelle_home / HEALTH_RUNS_DIRNAME


def prune_health_runs(
    ficelle_home: Path,
    *,
    keep_runs: Any = None,
    max_age_days: Any = None,
    pinned_run_ids: set[str] | None = None,
) -> list[str]:
    """Apply configured retention while preserving explicitly pinned baselines."""
    keep, days, protected = normalized_retention_policy(
        keep_runs,
        max_age_days,
        list(pinned_run_ids or set()),
    )
    if keep is None and days is None:
        return []
    root = health_runs_root(ficelle_home)
    if not root.exists():
        return []
    run_dirs = [path for path in root.iterdir() if path.is_dir() and (path / "run.json").exists()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    cutoff = time.time() - days * 86_400 if days is not None else None
    removed: list[str] = []
    unpinned_index = 0
    for path in run_dirs:
        if path.name in protected:
            continue
        over_count = keep is not None and unpinned_index >= keep
        over_age = cutoff is not None and path.stat().st_mtime < cutoff
        if over_count or over_age:
            shutil.rmtree(path)
            removed.append(path.name)
        unpinned_index += 1
    return removed


def normalized_retention_policy(
    keep_runs: Any,
    max_age_days: Any,
    pinned_run_ids: Any,
) -> tuple[int | None, float | None, set[str]]:
    try:
        keep = int(keep_runs) if keep_runs is not None else None
        days = float(max_age_days) if max_age_days is not None else None
    except (TypeError, ValueError) as exc:
        raise SyntheticHealthError("synthetic health retention values must be numeric") from exc
    if keep is not None and keep < 1:
        raise SyntheticHealthError("synthetic_health.retention_runs must be at least 1")
    if days is not None and days <= 0:
        raise SyntheticHealthError("synthetic_health.retention_days must be positive")
    if not isinstance(pinned_run_ids, list):
        raise SyntheticHealthError("synthetic_health.pinned_run_ids must be an array")
    return keep, days, {str(item) for item in pinned_run_ids if item}


def list_run_ids(ficelle_home: Path) -> list[str]:
    root = health_runs_root(ficelle_home)
    if not root.exists():
        return []
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "run.json").exists()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in runs]


def load_run_report(ficelle_home: Path, run_id: str | None = None) -> tuple[dict[str, Any], int]:
    selected = run_id
    if selected is None:
        ids = list_run_ids(ficelle_home)
        if not ids:
            raise SyntheticHealthError("no synthetic-health run exists")
        selected = ids[0]
    paths = RunPaths.for_run(ficelle_home, selected)
    summary = load_json(paths.summary_json, {})
    if not summary:
        metadata = load_json(paths.run, {})
        result = {
            "run_id": selected,
            "status": metadata.get("status") or "unknown",
            "cases": {"planned": metadata.get("case_count"), "completed": metadata.get("completed_cases", 0)},
            "artifacts": {"run": str(paths.run), "cases": str(paths.cases)},
        }
        return result, 2
    return _run_public_result(summary), exit_code_for_summary(summary)


def synthetic_health_status(ficelle_home: Path) -> dict[str, Any]:
    with file_lock(synthetic_health_lock_path(ficelle_home), blocking=False) as acquired:
        active = not acquired
    ids = list_run_ids(ficelle_home)
    latest = load_json(RunPaths.for_run(ficelle_home, ids[0]).run, {}) if ids else None
    latest_status = latest.get("status") if isinstance(latest, dict) else None
    if not active and latest_status == "running":
        latest_status = "interrupted"
    return {
        "active": active,
        "latest_run": {
            "run_id": latest.get("run_id"),
            "status": latest_status,
            "overall_status": latest.get("overall_status"),
            "started_at": latest.get("started_at"),
            "completed_at": latest.get("completed_at"),
            "completed_cases": latest.get("completed_cases", 0),
            "case_count": latest.get("case_count", 0),
        }
        if isinstance(latest, dict)
        else None,
        "runs_root": str(health_runs_root(ficelle_home)),
    }


def resume_synthetic_health(
    ficelle_home: Path,
    run_id: str,
    *,
    max_duration_seconds: float | None = None,
) -> tuple[dict[str, Any], int]:
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise SyntheticHealthError("max_duration_seconds must be positive")
    router = _load_router()
    paths = RunPaths.for_run(ficelle_home, run_id)
    metadata = load_json(paths.run, {})
    if not metadata:
        raise SyntheticHealthError(f"unknown synthetic-health run: {run_id}")
    if max_duration_seconds is None:
        raw_duration = metadata.get("max_duration_seconds", DEFAULT_MAX_DURATION_SECONDS)
        if raw_duration is not None:
            try:
                max_duration_seconds = float(raw_duration)
            except (TypeError, ValueError) as exc:
                raise SyntheticHealthError("stored max_duration_seconds must be numeric") from exc
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise SyntheticHealthError("max_duration_seconds must be positive")
    initial_inventory = load_json(paths.catalog_snapshot, {})
    if not isinstance(initial_inventory, dict) or not initial_inventory:
        raise SyntheticHealthError(f"run {run_id} has no usable initial catalog snapshot")
    with file_lock(synthetic_health_lock_path(ficelle_home), blocking=False) as run_lock:
        if not run_lock:
            raise SyntheticHealthBusy("another synthetic-health run is active for this FICELLE_HOME")
        with file_lock(probe_lock_path(ficelle_home), blocking=False) as probe_lock:
            if not probe_lock:
                raise SyntheticHealthBusy("benchmark, canary, or discovery probes are already active")
            config, catalog, inventory = _runtime_material(router)
            cases = build_corpus(router, config, catalog, inventory, run_id=run_id)
            planned_ids = {
                str(row.get("id"))
                for row in metadata.get("case_plan", [])
                if isinstance(row, dict) and row.get("id")
            }
            cases = [case for case in cases if case.id in planned_ids]
            stored_build = metadata.get("build_identity") if isinstance(metadata.get("build_identity"), dict) else {}
            # One stored-vs-current pair per resume gate. Version and corpus hash both
            # survive a code edit that touches neither; the runtime content hash is what
            # proves the resumed half of the journal runs the same harness build as the
            # half already on disk.
            checks = {
                "ficelle_version": (metadata.get("ficelle_version"), __version__),
                "corpus_version": (metadata.get("corpus_version"), CORPUS_VERSION),
                "corpus_hash": (metadata.get("corpus_hash"), corpus_hash(cases)),
                "config_structural_fingerprint": (
                    metadata.get("config_structural_fingerprint"),
                    router.catalog_config_structural_fingerprint(config),
                ),
                "catalog_identity": (metadata.get("catalog_identity"), catalog_identity(inventory)),
                "runtime_content_hash": (
                    stored_build.get("runtime_content_hash"),
                    package_build_identity().get("runtime_content_hash"),
                ),
            }
            mismatches = [key for key, (stored, current) in checks.items() if stored != current]
            if mismatches:
                raise SyntheticHealthError("run cannot be resumed because identity changed: " + ", ".join(mismatches))
            metadata["status"] = "running"
            metadata["resumed_at"] = now_iso()
            metadata.pop("completed_at", None)
            metadata.pop("overall_status", None)
            metadata.pop("interruption_reason", None)
            if max_duration_seconds is not None:
                metadata["max_duration_seconds"] = max_duration_seconds
            _write_run_metadata(paths, metadata)
            summary = _run_cases(
                router,
                paths=paths,
                metadata=metadata,
                config=config,
                catalog=catalog,
                inventory=initial_inventory,
                cases=cases,
                max_duration_seconds=max_duration_seconds,
            )
            return _run_public_result(summary), exit_code_for_summary(summary)


def rerun_failed_synthetic_health(
    ficelle_home: Path,
    run_id: str,
    *,
    max_duration_seconds: float | None = None,
) -> tuple[dict[str, Any], int]:
    paths = RunPaths.for_run(ficelle_home, run_id)
    if not paths.run.exists():
        raise SyntheticHealthError(f"unknown synthetic-health run: {run_id}")
    outcomes = latest_case_outcomes(paths.cases)
    failed = {
        case_id
        for case_id, row in outcomes.items()
        if row.get("status") not in {"pass", "pass_with_fallback"} and case_id != "p0.preflight.loopback-service"
    }
    if not failed:
        raise SyntheticHealthError(f"run {run_id} has no failed cases to rerun")
    return run_synthetic_health(
        ficelle_home,
        selected_case_ids=failed,
        parent_run_id=run_id,
        max_duration_seconds=max_duration_seconds,
    )


def schedule_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SCHEDULE_LABEL}.plist"


def schedule_status(*, plist_path: Path | None = None) -> dict[str, Any]:
    path = plist_path or schedule_plist_path()
    if not path.exists():
        return {"enabled": False, "label": SCHEDULE_LABEL, "plist": str(path)}
    try:
        payload = plistlib.loads(path.read_bytes())
    except (OSError, plistlib.InvalidFileException):
        return {"enabled": False, "label": SCHEDULE_LABEL, "plist": str(path), "error": "invalid_plist"}
    return {
        "enabled": True,
        "label": SCHEDULE_LABEL,
        "plist": str(path),
        "schedule": payload.get("StartCalendarInterval"),
        "program_arguments": payload.get("ProgramArguments"),
    }


def install_weekly_schedule(
    ficelle_home: Path,
    *,
    weekday: int,
    hour: int,
    minute: int,
    max_duration_seconds: float | None = None,
    plist_path: Path | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    platform: str = sys.platform,
) -> dict[str, Any]:
    if platform != "darwin":
        raise SyntheticHealthError("weekly scheduling is currently supported on macOS only")
    if weekday not in range(1, 8) or hour not in range(24) or minute not in range(60):
        raise SyntheticHealthError("invalid weekly schedule")
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise SyntheticHealthError("max_duration_seconds must be positive")
    path = plist_path or schedule_plist_path()
    arguments = [sys.executable, "-m", "ficelle.cli", "synthetic-health", "run", "--depth", "deep", "--json"]
    if max_duration_seconds is not None:
        arguments.extend(["--max-duration-seconds", str(max_duration_seconds)])
    log_dir = ficelle_home / "logs"
    ensure_private_dir(log_dir)
    payload = {
        "Label": SCHEDULE_LABEL,
        "ProgramArguments": arguments,
        "EnvironmentVariables": {"FICELLE_HOME": str(ficelle_home)},
        "StartCalendarInterval": {"Weekday": weekday, "Hour": hour, "Minute": minute},
        "RunAtLoad": False,
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / "synthetic-health.stdout.log"),
        "StandardErrorPath": str(log_dir / "synthetic-health.stderr.log"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, plistlib.dumps(payload, fmt=plistlib.FMT_XML).decode("utf-8"))
    uid = os.getuid()
    run_command(["launchctl", "bootout", f"gui/{uid}", str(path)], text=True, capture_output=True, check=False)
    result = run_command(["launchctl", "bootstrap", f"gui/{uid}", str(path)], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SyntheticHealthError(sanitize_error_detail(result.stderr, 500) or "launchctl bootstrap failed")
    return schedule_status(plist_path=path)


def remove_weekly_schedule(
    *,
    plist_path: Path | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    platform: str = sys.platform,
) -> dict[str, Any]:
    if platform != "darwin":
        raise SyntheticHealthError("weekly scheduling is currently supported on macOS only")
    path = plist_path or schedule_plist_path()
    if path.exists():
        run_command(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], text=True, capture_output=True, check=False)
        path.unlink()
    return schedule_status(plist_path=path)
