from __future__ import annotations

from typing import Any

from ficelle.use_cases.model_scoring import (
    ModelEvidencePorts,
    ModelScoreExplanationPorts,
    ModelScoringPorts,
    benchmark_adjustment_parts,
    latency_score_for_model,
    model_score_explanation,
    raw_benchmark_row,
    raw_verified_capability_row,
    runtime_profile_row,
    score_evidence_for_model,
    score_row_reason,
    score_row_status,
    success_rate_for_model,
)


def _ports() -> ModelScoringPorts:
    return ModelScoringPorts(
        cooldown_key=lambda model: f"{model.get('source')}::{model.get('upstream_id')}",
        safe_int=lambda value, default: int(value) if value is not None else default,
    )


def _evidence_ports() -> ModelEvidencePorts:
    return ModelEvidencePorts(
        cooldown_key=lambda model: f"{model.get('source')}::{model.get('upstream_id')}",
        canonical_virtual_model_id=lambda profile_id: "ficelle/auto-fast" if profile_id == "ficelle/auto" else profile_id,
        benchmark_result_matches_current_test=lambda _profile_id, row: bool(row.get("fresh")),
        benchmark_result_test_type_matches=lambda _profile_id, row: row.get("reason") != "test_type",
        benchmark_result_is_aged=lambda row: row.get("reason") == "aged",
        runtime_evidence_timestamp_value=lambda row: row.get("recorded_at"),
        parse_iso_timestamp=lambda value: {"older": 1.0, "newer": 2.0}.get(value),
        stale_score_decay_factor=lambda _profile_id, row: float(row.get("decay", 1.0)),
    )


def _score_explanation_ports() -> ModelScoreExplanationPorts:
    return ModelScoreExplanationPorts(
        scoring=_ports(),
        evidence=_evidence_ports(),
        model_has_any=lambda model, key, values: any(value in model.get(key, []) for value in values),
    )


def test_success_rate_uses_neutral_prior_without_history() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}

    assert success_rate_for_model(model, {}, ports=_ports()) == 0.72


def test_success_rate_uses_smoothed_runtime_stats() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {"stats": {"openrouter::free": {"successes": "3", "failures": "1"}}}

    assert success_rate_for_model(model, state, ports=_ports()) == 0.625


def test_latency_score_prefers_ewma_and_clamps_at_zero() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {"stats": {"openrouter::free": {"latency_ewma": "45"}}}

    assert latency_score_for_model(model, state, ports=_ports()) == 0.0


def test_latency_score_falls_back_to_legacy_success_latency() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {"successes": {"openrouter::free": {"latency_seconds": 15}}}

    assert latency_score_for_model(model, state, ports=_ports()) == 0.5


def test_latency_score_returns_neutral_value_for_invalid_latency() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state: dict[str, Any] = {"stats": {"openrouter::free": {"latency_ewma": "not-a-number"}}}

    assert latency_score_for_model(model, state, ports=_ports()) == 0.5


def test_runtime_profile_row_uses_canonical_profile_and_returns_copy() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    row = {"status": "pass", "fresh": True}
    state = {"benchmark_results": {"openrouter::free": {"ficelle/auto-fast": row}}}

    result = runtime_profile_row("benchmark_results", "ficelle/auto", model, state, ports=_evidence_ports())

    assert result == row
    assert result is not row
    assert raw_benchmark_row("ficelle/auto", model, state, ports=_evidence_ports()) == row
    assert raw_verified_capability_row("ficelle/auto", model, state, ports=_evidence_ports()) == {}


def test_score_row_status_and_reason_classify_missing_fresh_and_stale_evidence() -> None:
    ports = _evidence_ports()

    assert score_row_status("ficelle/auto-fast", {}, ports=ports) == "missing"
    assert score_row_reason("ficelle/auto-fast", {}, ports=ports) is None
    assert score_row_status("ficelle/auto-fast", {"fresh": True}, ports=ports) == "fresh"
    assert score_row_reason("ficelle/auto-fast", {"fresh": True}, ports=ports) is None
    assert score_row_status("ficelle/auto-fast", {"fresh": False}, ports=ports) == "stale"
    assert score_row_reason("ficelle/auto-fast", {"fresh": False, "reason": "test_type"}, ports=ports) == "test_type_mismatch"
    assert score_row_reason("ficelle/auto-fast", {"fresh": False, "reason": "aged"}, ports=ports) == "ttl_expired"
    assert score_row_reason("ficelle/auto-fast", {"fresh": False, "reason": "other"}, ports=ports) == "invalid_evidence"


def test_score_evidence_combines_verified_and_benchmark_status_with_latest_timestamp() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {
        "verified_capabilities": {
            "openrouter::free": {
                "ficelle/auto-fast": {"status": "verified", "fresh": True, "recorded_at": "older"}
            }
        },
        "benchmark_results": {
            "openrouter::free": {
                "ficelle/auto-fast": {
                    "status": "pass",
                    "fresh": False,
                    "reason": "aged",
                    "recorded_at": "newer",
                    "decay": 0.5,
                }
            }
        },
    }

    evidence = score_evidence_for_model("ficelle/auto", model, state, ports=_evidence_ports())

    assert evidence["verified_status"] == "fresh"
    assert evidence["benchmark_status"] == "stale"
    assert evidence["status"] == "stale"
    assert evidence["reason"] == "ttl_expired"
    assert evidence["last_evidence_at"] == "newer"
    assert evidence["score_decay_factor"] == 0.5


def test_benchmark_adjustment_parts_applies_verified_bonus_and_benchmark_latency_bonus() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {
        "verified_capabilities": {
            "openrouter::free": {
                "ficelle/auto-fast": {"status": "verified", "fresh": True, "recorded_at": "older"}
            }
        },
        "benchmark_results": {
            "openrouter::free": {
                "ficelle/auto-fast": {
                    "status": "pass",
                    "fresh": True,
                    "latency_seconds": 15,
                    "recorded_at": "newer",
                }
            }
        },
    }

    parts = benchmark_adjustment_parts("ficelle/auto", model, state, ports=_evidence_ports())

    assert parts["verified_bonus"] == 12.0
    assert parts["benchmark_bonus"] == 11.0
    assert parts["score_adjustment"] == 23.0
    assert parts["evidence_status"] == "fresh"


def test_model_score_explanation_combines_profile_weights_evidence_and_failure_penalty() -> None:
    model = {
        "source": "openrouter",
        "upstream_id": "free",
        "context_length": 500_000,
        "supports_tools": True,
        "supports_structured_outputs": True,
        "input_modalities": ["image"],
    }
    state = {
        "stats": {"openrouter::free": {"successes": 3, "failures": 1, "latency_ewma": 15, "consecutive_failures": 1}},
        "verified_capabilities": {
            "openrouter::free": {
                "ficelle/auto-multimodal": {"status": "verified", "fresh": True, "recorded_at": "older"}
            }
        },
        "benchmark_results": {
            "openrouter::free": {
                "ficelle/auto-multimodal": {
                    "status": "pass",
                    "fresh": True,
                    "latency_seconds": 15,
                    "recorded_at": "newer",
                }
            }
        },
    }

    score = model_score_explanation("ficelle/auto-multimodal", model, state, ports=_score_explanation_ports())

    assert score["score_base"] == 81.9
    assert score["score_adjustment"] == 23.0
    assert score["failure_penalty"] == 12.0
    assert score["score_total"] == 92.9
    assert score["score_total_raw"] == 92.875
    assert score["success_rate"] == 0.625
    assert score["latency_score"] == 0.5
    assert score["context_score"] == 0.5
    assert score["evidence_status"] == "fresh"
    assert score["last_evidence_at"] == "newer"


def test_auto_compression_score_does_not_prefer_structured_output_support() -> None:
    plain_model = {
        "source": "openrouter",
        "upstream_id": "plain",
        "context_length": 500_000,
        "supports_tools": True,
        "supports_structured_outputs": False,
    }
    structured_model = {
        **plain_model,
        "upstream_id": "structured",
        "supports_structured_outputs": True,
    }

    plain_score = model_score_explanation(
        "ficelle/auto-compression", plain_model, {}, ports=_score_explanation_ports()
    )
    structured_score = model_score_explanation(
        "ficelle/auto-compression", structured_model, {}, ports=_score_explanation_ports()
    )

    assert plain_score["score_base"] == structured_score["score_base"]
