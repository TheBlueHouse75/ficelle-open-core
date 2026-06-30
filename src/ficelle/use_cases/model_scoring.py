from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelScoringPorts:
    cooldown_key: Callable[[dict[str, Any]], str]
    safe_int: Callable[[Any, int], int]


@dataclass(frozen=True)
class ModelEvidencePorts:
    cooldown_key: Callable[[dict[str, Any]], str]
    canonical_virtual_model_id: Callable[[str], str]
    benchmark_result_matches_current_test: Callable[[str, Any], bool]
    benchmark_result_test_type_matches: Callable[[str, Any], bool]
    benchmark_result_is_aged: Callable[[Any], bool]
    runtime_evidence_timestamp_value: Callable[[dict[str, Any]], str | None]
    parse_iso_timestamp: Callable[[Any], float | None]
    stale_score_decay_factor: Callable[[str, dict[str, Any]], float]


@dataclass(frozen=True)
class ModelScoreExplanationPorts:
    scoring: ModelScoringPorts
    evidence: ModelEvidencePorts
    model_has_any: Callable[[dict[str, Any], str, list[str]], bool]


def success_rate_for_model(model: dict[str, Any], state: dict[str, Any], *, ports: ModelScoringPorts) -> float:
    record = ((state.get("stats") or {}).get(ports.cooldown_key(model)) or {}) if isinstance(state, dict) else {}
    successes = ports.safe_int(record.get("successes"), 0)
    failures = ports.safe_int(record.get("failures"), 0)
    total = successes + failures
    if total <= 0:
        return 0.72
    # Bayesian-ish smoothing: avoid over-trusting a single lucky call.
    return (successes + 2.0) / (total + 4.0)


def latency_score_for_model(model: dict[str, Any], state: dict[str, Any], *, ports: ModelScoringPorts) -> float:
    record = ((state.get("stats") or {}).get(ports.cooldown_key(model)) or {}) if isinstance(state, dict) else {}
    latency = record.get("latency_ewma")
    if latency is None:
        legacy = ((state.get("successes") or {}).get(ports.cooldown_key(model)) or {}) if isinstance(state, dict) else {}
        latency = legacy.get("latency_seconds")
    try:
        seconds = float(latency)
    except Exception:
        return 0.5
    # 0s => 1.0, 30s+ => 0.0. Free models are often slow, so don't over-punish.
    return max(0.0, min(1.0, 1.0 - (seconds / 30.0)))


def runtime_profile_row(
    section: str,
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    ports: ModelEvidencePorts,
) -> dict[str, Any]:
    rows = state.get(section) if isinstance(state.get(section), dict) else {}
    by_profile = rows.get(ports.cooldown_key(model)) if isinstance(rows, dict) else None
    row = by_profile.get(ports.canonical_virtual_model_id(profile_id)) if isinstance(by_profile, dict) else None
    return dict(row) if isinstance(row, dict) else {}


def raw_benchmark_row(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    ports: ModelEvidencePorts,
) -> dict[str, Any]:
    return runtime_profile_row("benchmark_results", profile_id, model, state, ports=ports)


def raw_verified_capability_row(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    ports: ModelEvidencePorts,
) -> dict[str, Any]:
    return runtime_profile_row("verified_capabilities", profile_id, model, state, ports=ports)


def score_row_status(profile_id: str, row: dict[str, Any], *, ports: ModelEvidencePorts) -> str:
    if not row:
        return "missing"
    if ports.benchmark_result_matches_current_test(profile_id, row):
        return "fresh"
    return "stale"


def score_row_reason(profile_id: str, row: dict[str, Any], *, ports: ModelEvidencePorts) -> str | None:
    if not row or ports.benchmark_result_matches_current_test(profile_id, row):
        return None
    if not ports.benchmark_result_test_type_matches(profile_id, row):
        return "test_type_mismatch"
    if ports.benchmark_result_is_aged(row):
        return "ttl_expired"
    return "invalid_evidence"


def score_evidence_for_model(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    ports: ModelEvidencePorts,
) -> dict[str, Any]:
    profile_id = ports.canonical_virtual_model_id(profile_id)
    verified_raw = raw_verified_capability_row(profile_id, model, state, ports=ports) if isinstance(state, dict) else {}
    benchmark_raw = raw_benchmark_row(profile_id, model, state, ports=ports) if isinstance(state, dict) else {}
    verified_status = score_row_status(profile_id, verified_raw, ports=ports)
    benchmark_status = score_row_status(profile_id, benchmark_raw, ports=ports)
    verified_reason = score_row_reason(profile_id, verified_raw, ports=ports)
    benchmark_reason = score_row_reason(profile_id, benchmark_raw, ports=ports)
    latest_value: str | None = None
    latest_stamp: float | None = None
    for row in (verified_raw, benchmark_raw):
        value = ports.runtime_evidence_timestamp_value(row)
        stamp = ports.parse_iso_timestamp(value)
        if value and (latest_stamp is None or (stamp is not None and stamp >= latest_stamp)):
            latest_value = value
            latest_stamp = stamp
    if "stale" in {verified_status, benchmark_status}:
        status = "stale"
    elif "fresh" in {verified_status, benchmark_status}:
        status = "fresh"
    else:
        status = "missing"
    return {
        "verified_row": verified_raw if verified_status == "fresh" else {},
        "verified_raw": verified_raw,
        "verified_status": verified_status,
        "verified_reason": verified_reason,
        "benchmark_raw": benchmark_raw,
        "benchmark_status": benchmark_status,
        "benchmark_reason": benchmark_reason,
        "status": status,
        "reason": verified_reason or benchmark_reason,
        "last_evidence_at": latest_value,
        "score_decay_factor": min(
            ports.stale_score_decay_factor(profile_id, verified_raw) if verified_raw else 1.0,
            ports.stale_score_decay_factor(profile_id, benchmark_raw) if benchmark_raw else 1.0,
        ),
    }


def benchmark_adjustment_parts(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    ports: ModelEvidencePorts,
) -> dict[str, Any]:
    profile_id = ports.canonical_virtual_model_id(profile_id)
    evidence = score_evidence_for_model(profile_id, model, state, ports=ports)
    verified_row = evidence["verified_row"]
    verified_raw = evidence["verified_raw"]
    benchmark_raw = evidence["benchmark_raw"]
    verified_status = str(verified_row.get("status") or "")
    verified_bonus = 0.0
    if verified_status == "verified":
        verified_bonus = 12.0
    elif verified_status == "failed":
        verified_bonus = -24.0
    elif verified_raw:
        decay = ports.stale_score_decay_factor(profile_id, verified_raw)
        raw_status = str(verified_raw.get("status") or "")
        if raw_status == "verified":
            verified_bonus = 12.0 * decay
        elif raw_status == "failed":
            verified_bonus = -24.0 * decay

    benchmark_bonus = 0.0
    benchmark_row = dict(benchmark_raw)
    benchmark_decay = 1.0
    if not ports.benchmark_result_matches_current_test(profile_id, benchmark_row):
        benchmark_decay = ports.stale_score_decay_factor(profile_id, benchmark_row)
        if benchmark_decay <= 0:
            benchmark_row = {}
    if benchmark_row:
        if benchmark_row.get("status") == "pass":
            benchmark_bonus += 8.0 * benchmark_decay
            raw_latency = benchmark_row.get("latency_seconds")
            if raw_latency is not None:
                try:
                    latency = float(raw_latency)
                    benchmark_bonus += max(0.0, min(6.0, 6.0 * (1.0 - (latency / 30.0)))) * benchmark_decay
                except Exception:
                    pass
        elif benchmark_row.get("status") == "fail":
            benchmark_bonus = -16.0 * benchmark_decay

    return {
        "verified_bonus": verified_bonus,
        "benchmark_bonus": benchmark_bonus,
        "score_adjustment": verified_bonus + benchmark_bonus,
        "score_decay_factor": evidence["score_decay_factor"],
        "evidence_status": evidence["status"],
        "evidence_reason": evidence["reason"],
        "benchmark_evidence_status": evidence["benchmark_status"],
        "benchmark_evidence_reason": evidence["benchmark_reason"],
        "verified_evidence_status": evidence["verified_status"],
        "verified_evidence_reason": evidence["verified_reason"],
        "last_evidence_at": evidence["last_evidence_at"],
    }


def model_score_explanation(
    requested_model: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    ports: ModelScoreExplanationPorts,
) -> dict[str, Any]:
    requested_model = ports.evidence.canonical_virtual_model_id(requested_model)
    record = ((state.get("stats") or {}).get(ports.scoring.cooldown_key(model)) or {}) if isinstance(state, dict) else {}
    success_rate = success_rate_for_model(model, state, ports=ports.scoring)
    latency_score = latency_score_for_model(model, state, ports=ports.scoring)
    consecutive_failures = ports.scoring.safe_int(record.get("consecutive_failures"), 0)
    context_length = ports.scoring.safe_int(model.get("context_length"), 0)
    context_score = min(1.0, context_length / 1_000_000) if context_length > 0 else 0.0
    structured = bool(model.get("supports_structured_outputs"))
    tool_bonus = 1.0 if model.get("supports_tools") else 0.0
    reasoning = ports.model_has_any(model, "supported_parameters", ["reasoning", "include_reasoning"])
    image_in = ports.model_has_any(model, "input_modalities", ["image"])
    video_in = ports.model_has_any(model, "input_modalities", ["video"])
    audio_in = ports.model_has_any(model, "input_modalities", ["audio"])
    multimodal = image_in or video_in or audio_in

    if requested_model == "ficelle/auto-orchestrator":
        score = (success_rate * 42) + (tool_bonus * 18) + ((1.0 if structured else 0.0) * 18) + (context_score * 14) + (latency_score * 8)
    elif requested_model == "ficelle/auto-fast":
        score = (success_rate * 45) + (latency_score * 35) + (tool_bonus * 10) + (context_score * 10)
    elif requested_model == "ficelle/auto-json":
        score = (success_rate * 40) + ((1.0 if structured else 0.0) * 30) + (tool_bonus * 15) + (latency_score * 10) + (context_score * 5)
    elif requested_model == "ficelle/auto-compression":
        score = (success_rate * 42) + (latency_score * 26) + (context_score * 18) + ((1.0 if structured else 0.0) * 10) + (tool_bonus * 4)
    elif requested_model == "ficelle/auto-long":
        score = (success_rate * 35) + (context_score * 35) + (tool_bonus * 15) + ((1.0 if structured else 0.0) * 10) + (latency_score * 5)
    elif requested_model == "ficelle/auto-reasoning":
        score = (success_rate * 35) + ((1.0 if reasoning else 0.0) * 30) + (tool_bonus * 15) + ((1.0 if structured else 0.0) * 10) + (context_score * 10)
    elif requested_model == "ficelle/auto-multimodal":
        score = (success_rate * 35) + ((1.0 if multimodal else 0.0) * 30) + (tool_bonus * 15) + ((1.0 if structured else 0.0) * 10) + (context_score * 10)
    elif requested_model == "ficelle/auto-vision":
        score = (success_rate * 35) + ((1.0 if image_in else 0.0) * 35) + (tool_bonus * 10) + ((1.0 if structured else 0.0) * 10) + (context_score * 10)
    elif requested_model == "ficelle/auto-video":
        score = (success_rate * 35) + ((1.0 if video_in else 0.0) * 35) + (tool_bonus * 10) + ((1.0 if structured else 0.0) * 10) + (context_score * 10)
    elif requested_model == "ficelle/auto-audio":
        score = (success_rate * 35) + ((1.0 if audio_in else 0.0) * 35) + (tool_bonus * 10) + ((1.0 if structured else 0.0) * 10) + (context_score * 10)
    else:
        score = (success_rate * 45) + (tool_bonus * 20) + ((1.0 if structured else 0.0) * 15) + (latency_score * 10) + (context_score * 10)

    parts = benchmark_adjustment_parts(requested_model, model, state, ports=ports.evidence)
    failure_penalty = consecutive_failures * 12.0
    score_total = max(0.0, score + parts["score_adjustment"] - failure_penalty)
    return {
        "score_base": round(score, 1),
        "score_adjustment": round(parts["score_adjustment"], 1),
        "score_total_raw": score_total,
        "score_total": round(score_total, 1),
        "benchmark_bonus": round(parts["benchmark_bonus"], 1),
        "verified_bonus": round(parts["verified_bonus"], 1),
        "score_decay_factor": round(parts["score_decay_factor"], 4),
        "failure_penalty": round(failure_penalty, 1),
        "success_rate": round(success_rate, 4),
        "latency_score": round(latency_score, 4),
        "context_score": round(context_score, 4),
        "evidence_status": parts["evidence_status"],
        "evidence_reason": parts["evidence_reason"],
        "benchmark_evidence_status": parts["benchmark_evidence_status"],
        "benchmark_evidence_reason": parts["benchmark_evidence_reason"],
        "verified_evidence_status": parts["verified_evidence_status"],
        "verified_evidence_reason": parts["verified_evidence_reason"],
        "last_evidence_at": parts["last_evidence_at"],
    }
