from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ficelle.use_cases.cooldowns import FAILURE_PENALTY_POINTS, failure_penalty_weight, score_decay_factor
from ficelle.use_cases.quality_feedback import (
    quality_feedback_routing_enabled,
    quality_feedback_summary_for_model,
)


@dataclass(frozen=True)
class ModelScoringPorts:
    cooldown_key: Callable[[dict[str, Any]], str]
    safe_int: Callable[[Any, int], int]
    # Read-side clock for the success-rate half-life. Defaults to the wall clock so callers
    # written before the decay moved to read time keep working; tests inject a fixed instant.
    now_epoch: Callable[[], float] = time.time


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
    quality_feedback_summary: Callable[[str, dict[str, Any], dict[str, Any], Callable[[], float]], dict[str, Any]] = (
        quality_feedback_summary_for_model
    )
    quality_feedback_routing_enabled: Callable[[str, dict[str, Any]], bool] = quality_feedback_routing_enabled


# Success rate of a model with no usable history, and the value a decayed one converges back to.
NEUTRAL_SUCCESS_RATE = 0.72
# Below one effective observation the decayed counters no longer support a verdict, so the model
# reads as unknown instead of drifting toward the smoothing prior (0.5) — which would rank a model
# idle for weeks BELOW one nobody has ever called.
MIN_SCORED_OBSERVATIONS = 1.0
# Latency at which a model scores exactly 0.5, and the scale of the whole curve. Free upstreams
# routinely answer in tens of seconds, so it is deliberately generous.
LATENCY_REFERENCE_SECONDS = 15.0


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def latency_score_from_seconds(seconds: float) -> float:
    """Map a latency to (0, 1]: 0s => 1.0, 15s => 0.5, and strictly decreasing after that.

    The former scale was `1 - seconds/30` clamped at zero, so every model slower than 30s scored
    exactly 0 and the 26-35 latency points of the profile weights stopped discriminating in a pool
    where slow is normal: 51s and 247s were indistinguishable, and a 35s model could sit second in
    `auto-fast` on its success rate alone. This form never saturates, so a slow model always ranks
    behind a slower one.
    """
    return LATENCY_REFERENCE_SECONDS / (LATENCY_REFERENCE_SECONDS + max(0.0, seconds))


def success_rate_for_model(model: dict[str, Any], state: dict[str, Any], *, ports: ModelScoringPorts) -> float:
    record = ((state.get("stats") or {}).get(ports.cooldown_key(model)) or {}) if isinstance(state, dict) else {}
    # Decayed counters when present: cumulative lifetime totals let a model coast on wins from
    # weeks ago (on this install, half the models last ran 48 days back) or stay punished for
    # equally old failures. A state written before decay existed has neither field, so fall back
    # to the lifetime totals rather than scoring it as brand new.
    if record.get("scored_at") is not None:
        # The write path ages the counters when it folds in a new observation, so a model that
        # stopped being called kept whichever rate it last wrote — 12 models on this install were
        # last scored more than 10 days ago, one frozen at 0.944 for 17. The same half-life is
        # therefore re-applied here for the time since that write: reading and writing use one
        # clock and one curve, so applying it twice cannot double-count.
        factor = score_decay_factor(ports.now_epoch() - _safe_float(record.get("scored_at")))
        successes = _safe_float(record.get("scored_successes")) * factor
        failures = _safe_float(record.get("scored_failures")) * factor
    else:
        successes = float(ports.safe_int(record.get("successes"), 0))
        failures = float(ports.safe_int(record.get("failures"), 0))
    total = successes + failures
    if total < MIN_SCORED_OBSERVATIONS:
        return NEUTRAL_SUCCESS_RATE
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
        # No usable timing: the reference latency, which is exactly what an unmeasured model is
        # worth against one measured at the reference.
        return 0.5
    return latency_score_from_seconds(seconds)


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


def _failed_verdict_penalty(full_penalty: float, decay: float, *, same_test: bool) -> float:
    """The (positive) penalty a `failed` verdict still carries.

    A `failed` verdict about the CURRENT test does not decay at all: an expired verdict is not a
    neutral one, it is one that is due a re-test, and only a fresh passing probe replaces the row.
    `nvidia/nemotron-3.5-lightning:free` failed the `auto-compression` probe ("summary not
    bounded"); seven days later its decay factor reached 0, its -24 verified penalty and -16
    benchmark penalty both vanished, and it went straight back to rank 1 of 11 candidates — ahead
    of a model holding a FRESH pass — where it carried half the profile's traffic at a 279s
    average. The full penalty is also the whole mechanism: it keeps the candidate behind every
    model with current evidence while leaving it eligible, which is what the anti-empty fallback
    needs.

    A verdict about a different probe shape (`test_type` mismatch) is not evidence about the
    current test at all, so it keeps decaying all the way to zero.
    """
    return full_penalty if same_test else full_penalty * max(0.0, decay)


def stale_failed_profile_evidence(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    ports: ModelEvidencePorts,
) -> bool:
    """True when this (model, profile) pair's only verdict is a `failed` one past its TTL.

    Semantics, deliberately separate from `model_failed_profile_evidence`:

    - the model stays **eligible**. The competence gate reads fresh verdicts only, and the
      anti-empty-pool fallback must keep working, so an expired condemnation cannot empty a pool;
    - it is **demoted** by the score alone: `_failed_verdict_penalty` does not decay, so the pair
      keeps the full -24/-16 until a fresh probe replaces the row. This predicate does not order
      the pool;
    - it **jumps the discovery queue**, so the next probe either rehabilitates it or re-condemns
      it with a fresh verdict rather than leaving it silently promoted for weeks. That is this
      predicate's only remaining reader.

    A fresh passing verdict in either section clears it: the model has been re-tested and won.
    """
    profile_id = ports.canonical_virtual_model_id(profile_id)
    verified_raw = raw_verified_capability_row(profile_id, model, state, ports=ports) if isinstance(state, dict) else {}
    benchmark_raw = raw_benchmark_row(profile_id, model, state, ports=ports) if isinstance(state, dict) else {}
    stale_failed = False
    for row, failed_status, passed_status in (
        (verified_raw, "failed", "verified"),
        (benchmark_raw, "fail", "pass"),
    ):
        if not row:
            continue
        status = str(row.get("status") or "")
        fresh = ports.benchmark_result_matches_current_test(profile_id, row)
        if fresh and status == passed_status:
            return False
        if fresh or status != failed_status:
            continue
        if ports.benchmark_result_test_type_matches(profile_id, row):
            stale_failed = True
    return stale_failed


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
            verified_bonus = -_failed_verdict_penalty(
                24.0,
                decay,
                same_test=ports.benchmark_result_test_type_matches(profile_id, verified_raw),
            )

    benchmark_bonus = 0.0
    benchmark_row = dict(benchmark_raw)
    benchmark_decay = 1.0
    if not ports.benchmark_result_matches_current_test(profile_id, benchmark_row):
        benchmark_decay = ports.stale_score_decay_factor(profile_id, benchmark_row)
        # A fully decayed row is dropped, except a `fail`: that one keeps its full penalty below.
        if benchmark_decay <= 0 and benchmark_row.get("status") != "fail":
            benchmark_row = {}
    if benchmark_row:
        if benchmark_row.get("status") == "pass":
            benchmark_bonus += 8.0 * benchmark_decay
            raw_latency = benchmark_row.get("latency_seconds")
            if raw_latency is not None:
                try:
                    latency = float(raw_latency)
                    benchmark_bonus += 6.0 * latency_score_from_seconds(latency) * benchmark_decay
                except Exception:
                    pass
        elif benchmark_row.get("status") == "fail":
            benchmark_bonus = -_failed_verdict_penalty(
                16.0,
                benchmark_decay,
                same_test=ports.benchmark_result_test_type_matches(profile_id, benchmark_row),
            )

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
        score = (success_rate * 42) + (latency_score * 26) + (context_score * 18) + (tool_bonus * 4)
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
    # Recency-weighted, and answered by a success of the same standing (`cooldowns.failure_ledger`):
    # the streak it replaced was zeroed by any success, a 5-token probe included.
    failure_penalty = failure_penalty_weight(record, ports.scoring.now_epoch()) * FAILURE_PENALTY_POINTS
    score_total_without_quality = max(0.0, score + parts["score_adjustment"] - failure_penalty)
    quality = ports.quality_feedback_summary(requested_model, model, state, ports.scoring.now_epoch)
    try:
        quality_adjustment = float(quality.get("quality_adjustment") or 0.0)
    except (TypeError, ValueError):
        quality_adjustment = 0.0
    if not math.isfinite(quality_adjustment):
        quality_adjustment = 0.0
    score_total_with_quality = max(0.0, score_total_without_quality + quality_adjustment)
    quality_routing = ports.quality_feedback_routing_enabled(requested_model, state)
    # Shadow is the default: evidence stays visible to operators without perturbing the route
    # score or tie-break order. Enabling it is profile-scoped in the ephemeral config view.
    score_total = score_total_with_quality if quality_routing else score_total_without_quality
    return {
        "score_base": round(score, 1),
        "score_adjustment": round(parts["score_adjustment"], 1),
        "score_total_raw": score_total,
        "score_total": round(score_total, 1),
        "score_total_without_quality": score_total_without_quality,
        "score_total_with_quality": score_total_with_quality,
        "quality_adjustment": quality_adjustment,
        "quality_sample_count": int(quality.get("sample_count") or 0),
        "quality_last_reason": quality.get("last_reason"),
        "quality_last_severity": quality.get("last_severity"),
        "quality_last_recorded_at": quality.get("last_recorded_at"),
        "quality_status": quality.get("status") or "none",
        "quality_routing_enabled": quality_routing,
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
