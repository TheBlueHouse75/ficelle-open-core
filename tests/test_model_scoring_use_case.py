from __future__ import annotations

from typing import Any

from ficelle.use_cases.model_scoring import (
    ModelEvidencePorts,
    ModelScoreExplanationPorts,
    ModelScoringPorts,
    benchmark_adjustment_parts,
    first_delta_score_for_model,
    latency_score_for_model,
    model_score_explanation,
    raw_benchmark_row,
    raw_verified_capability_row,
    runtime_profile_row,
    score_evidence_for_model,
    score_row_reason,
    score_row_status,
    stale_failed_profile_evidence,
    success_rate_for_model,
    throughput_score_for_model,
)
from ficelle.use_cases.quality_feedback import quality_feedback_scoring_state


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


def _quality_score_explanation_ports(*, routing_enabled: bool) -> ModelScoreExplanationPorts:
    return ModelScoreExplanationPorts(
        scoring=_ports(),
        evidence=_evidence_ports(),
        model_has_any=lambda model, key, values: any(value in model.get(key, []) for value in values),
        quality_feedback_summary=lambda _profile_id, _model, _state, _now: {
            "sample_count": 3,
            "quality_adjustment": -12.0,
            "last_reason": "invalid_json",
            "last_severity": "major",
            "last_recorded_at": "newer",
            "status": "fail",
        },
        quality_feedback_routing_enabled=lambda _profile_id, _state: routing_enabled,
    )


def test_success_rate_uses_neutral_prior_without_history() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}

    assert success_rate_for_model(model, {}, ports=_ports()) == 0.72


def test_quality_feedback_is_shadowed_by_default_and_only_applied_when_enabled() -> None:
    model = {
        "source": "openrouter",
        "upstream_id": "free",
        "context_length": 128_000,
        "supports_tools": True,
        "supports_structured_outputs": True,
    }

    shadow = model_score_explanation("ficelle/auto-fast", model, {}, ports=_quality_score_explanation_ports(routing_enabled=False))
    enabled = model_score_explanation("ficelle/auto-fast", model, {}, ports=_quality_score_explanation_ports(routing_enabled=True))

    assert shadow["score_total_raw"] == shadow["score_total_without_quality"]
    assert shadow["score_total_with_quality"] == shadow["score_total_without_quality"] - 12.0
    assert shadow["score_total"] == round(shadow["score_total_without_quality"], 1)
    assert shadow["quality_adjustment"] == -12.0
    assert shadow["quality_sample_count"] == 3
    assert shadow["quality_last_reason"] == "invalid_json"
    assert shadow["quality_last_severity"] == "major"
    assert shadow["quality_last_recorded_at"] == "newer"
    assert shadow["quality_status"] == "fail"
    assert shadow["quality_routing_enabled"] is False

    assert enabled["score_total_raw"] == enabled["score_total_with_quality"]
    assert enabled["score_total"] == round(enabled["score_total_with_quality"], 1)
    assert enabled["quality_routing_enabled"] is True


def test_malformed_quality_feedback_is_neutral_in_shadow_and_enabled_scoring() -> None:
    now = 1_000_000.0
    profile_id = "ficelle/auto-orchestrator"
    model = {
        "source": "openrouter",
        "upstream_id": "free",
        "context_length": 128_000,
        "supports_tools": True,
        "supports_structured_outputs": True,
    }
    state: dict[str, Any] = {
        "quality_feedback": {
            "openrouter::free": {
                profile_id: {
                    "event_slots": {
                        "list-outcome": {
                            "outcome": [],
                            "severity": {},
                            "reason": "invalid_json",
                            "validator": "operator.review",
                            "recorded_at": "1970-01-12T13:46:40+00:00",
                            "expires_at": "2099-01-01T00:00:00+00:00",
                        },
                        "dict-outcome": {
                            "outcome": {},
                            "severity": [],
                            "reason": "invalid_json",
                            "validator": "operator.review",
                            "recorded_at": "1970-01-12T13:46:40+00:00",
                            "expires_at": "2099-01-01T00:00:00+00:00",
                        },
                    }
                }
            }
        }
    }
    ports = ModelScoreExplanationPorts(
        scoring=ModelScoringPorts(
            cooldown_key=lambda candidate: f"{candidate.get('source')}::{candidate.get('upstream_id')}",
            safe_int=lambda value, default: int(value) if value is not None else default,
            now_epoch=lambda: now,
        ),
        evidence=_evidence_ports(),
        model_has_any=lambda candidate, key, values: any(value in candidate.get(key, []) for value in values),
    )
    shadow_state = quality_feedback_scoring_state(state, {}, canonical_profile_id=lambda value: value)
    enabled_state = quality_feedback_scoring_state(
        state,
        {"quality_feedback": {"routing_enabled": True, "enabled_profiles": [profile_id]}},
        canonical_profile_id=lambda value: value,
    )

    shadow = model_score_explanation(profile_id, model, shadow_state, ports=ports)
    enabled = model_score_explanation(profile_id, model, enabled_state, ports=ports)

    for explanation in (shadow, enabled):
        assert explanation["quality_adjustment"] == 0.0
        assert explanation["quality_sample_count"] == 0
        assert explanation["quality_status"] == "none"
        assert explanation["quality_last_reason"] is None
        assert explanation["quality_last_severity"] is None
        assert explanation["quality_last_recorded_at"] is None
        assert explanation["score_total_raw"] == explanation["score_total_without_quality"]
        assert explanation["score_total_with_quality"] == explanation["score_total_without_quality"]
    assert shadow["quality_routing_enabled"] is False
    assert enabled["quality_routing_enabled"] is True


def test_success_rate_uses_smoothed_runtime_stats() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {"stats": {"openrouter::free": {"successes": "3", "failures": "1"}}}

    assert success_rate_for_model(model, state, ports=_ports()) == 0.625


def test_success_rate_keeps_decaying_while_a_model_is_idle() -> None:
    """D5: the half-life is re-applied at read time, not only when a new observation is written.

    Write-time decay alone froze a model that stopped being called at whatever rate it last wrote
    — 12 models on the dogfood install were last scored more than 10 days earlier, one stuck at
    0.944 for 17 days, which is exactly the rank a degraded model needs to keep winning.
    """
    model = {"source": "openrouter", "upstream_id": "free"}
    written_at = 1_000_000.0
    state = {
        "stats": {
            "openrouter::free": {"scored_successes": 17.0, "scored_failures": 1.0, "scored_at": written_at}
        }
    }

    def rate_after(days: float) -> float:
        ports = ModelScoringPorts(
            cooldown_key=lambda candidate: f"{candidate.get('source')}::{candidate.get('upstream_id')}",
            safe_int=lambda value, default: int(value) if value is not None else default,
            now_epoch=lambda: written_at + days * 86_400,
        )
        return success_rate_for_model(model, state, ports=ports)

    # Same state, three read instants: the rate slides back toward the prior as the evidence ages.
    assert rate_after(0) > rate_after(7) > rate_after(21)
    # Past ~5 half-lives fewer than one effective observation is left, so the model reads as
    # unknown rather than drifting below a model nobody has ever called.
    assert rate_after(35) == 0.72
    # A state written before decay existed keeps scoring on its lifetime totals.
    legacy = {"stats": {"openrouter::free": {"successes": 3, "failures": 1}}}
    assert success_rate_for_model(model, legacy, ports=_ports()) == 0.625


def test_latency_score_prefers_ewma_and_never_saturates() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    # 45s used to score exactly 0.0, along with every other latency past 30s. The scale is now
    # `15 / (15 + seconds)`, so 45s keeps a quarter of the latency points and the tail still ranks.
    state = {"stats": {"openrouter::free": {"latency_ewma": "45"}}}

    assert latency_score_for_model(model, state, ports=_ports()) == 0.25


def test_latency_score_separates_every_speed_in_a_slow_pool() -> None:
    def score(seconds: float) -> float:
        return latency_score_for_model(
            {"source": "openrouter", "upstream_id": "free"},
            {"stats": {"openrouter::free": {"latency_ewma": seconds}}},
            ports=_ports(),
        )

    # The live pool that motivated this: ministral-14b answers in 0.5s, dots-3 in 35s, and two
    # compression candidates in 51s and 247s. Under the old clamp the last three all scored 0.
    assert score(0.5) > score(35.0) > score(51.0) > score(247.0) > 0.0
    assert score(0.5) - score(35.0) > 0.6, "a sub-second model must stay far ahead of a 35s one"


def test_latency_score_falls_back_to_legacy_success_latency() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {"successes": {"openrouter::free": {"latency_seconds": 15}}}

    assert latency_score_for_model(model, state, ports=_ports()) == 0.5


def test_latency_score_returns_neutral_value_for_invalid_latency() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state: dict[str, Any] = {"stats": {"openrouter::free": {"latency_ewma": "not-a-number"}}}

    assert latency_score_for_model(model, state, ports=_ports()) == 0.5


def test_first_delta_score_prefers_first_delta_ewma_over_total_latency() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    fast_first_delta = {"stats": {"openrouter::free": {"latency_ewma": 15, "first_delta_ewma": 3}}}
    slow_first_delta = {"stats": {"openrouter::free": {"latency_ewma": 15, "first_delta_ewma": 150}}}

    assert first_delta_score_for_model(model, fast_first_delta, ports=_ports()) > first_delta_score_for_model(
        model, slow_first_delta, ports=_ports()
    )


def test_first_delta_score_stays_neutral_when_only_total_latency_is_known() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    only_latency = {"stats": {"openrouter::free": {"latency_ewma": 60}}}
    legacy_only = {"successes": {"openrouter::free": {"latency_seconds": 60}}}

    # Total latency is a different quantity: a slow-but-unmeasured model is not punished on the
    # first-delta curve until a real first delta has been observed.
    assert first_delta_score_for_model(model, only_latency, ports=_ports()) == 0.5
    assert first_delta_score_for_model(model, legacy_only, ports=_ports()) == 0.5


def test_first_delta_and_throughput_score_are_neutral_without_any_data() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state: dict[str, Any] = {"stats": {"openrouter::free": {}}}

    assert first_delta_score_for_model(model, state, ports=_ports()) == 0.5
    assert throughput_score_for_model(model, state, ports=_ports()) == 0.5


def test_throughput_score_rewards_faster_generation() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    slow = {"stats": {"openrouter::free": {"completion_tokens_per_second": 5}}}
    fast = {"stats": {"openrouter::free": {"completion_tokens_per_second": 60}}}

    assert throughput_score_for_model(model, fast, ports=_ports()) > throughput_score_for_model(
        model, slow, ports=_ports()
    )


def test_throughput_score_is_neutral_for_missing_or_non_positive_values() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    missing = {"stats": {"openrouter::free": {}}}
    zero = {"stats": {"openrouter::free": {"completion_tokens_per_second": 0}}}

    assert throughput_score_for_model(model, missing, ports=_ports()) == 0.5
    assert throughput_score_for_model(model, zero, ports=_ports()) == 0.5


def test_slow_first_delta_loses_the_ranking_despite_equal_total_latency() -> None:
    """This is the bug that motivated the speed score: `nvidia/nemotron-3-ultra-550b` answered
    in over 120s to first token (past Hermes' abort window) while `latency_ewma` alone looked
    identical to a model that streams promptly, so it kept winning auto-compression and auto-fast.
    """
    slow_starter = {"source": "openrouter", "upstream_id": "slow-starter"}
    fast_starter = {"source": "openrouter", "upstream_id": "fast-starter"}
    base_stats = {"successes": 3, "failures": 1, "latency_ewma": 20}
    state = {
        "stats": {
            "openrouter::slow-starter": {**base_stats, "first_delta_ewma": 150},
            "openrouter::fast-starter": {**base_stats, "first_delta_ewma": 3},
        }
    }

    for profile_id in ("ficelle/auto-compression", "ficelle/auto-fast"):
        slow_score = model_score_explanation(profile_id, slow_starter, state, ports=_score_explanation_ports())
        fast_score = model_score_explanation(profile_id, fast_starter, state, ports=_score_explanation_ports())
        assert fast_score["score_total"] > slow_score["score_total"], profile_id


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


def _failed_evidence_state(*, decay: float, fresh: bool = False) -> dict[str, Any]:
    row = {"status": "failed", "fresh": fresh, "recorded_at": "older", "decay": decay}
    return {"verified_capabilities": {"openrouter::free": {"ficelle/auto-compression": row}}}


def test_a_failed_verdict_about_this_test_does_not_decay() -> None:
    """An expired `failed` verdict is due a re-test, not an amnesty — so its penalty is unchanged.

    The demotion is the score and nothing else, which is why it may not fade: at decay 0 the
    penalty used to vanish and the model returned to rank 1 of its pool with a clean sheet.
    """
    model = {"source": "openrouter", "upstream_id": "free"}

    fresh = benchmark_adjustment_parts(
        "ficelle/auto-compression", model, _failed_evidence_state(decay=1.0, fresh=True), ports=_evidence_ports()
    )
    half = benchmark_adjustment_parts(
        "ficelle/auto-compression", model, _failed_evidence_state(decay=0.5), ports=_evidence_ports()
    )
    expired = benchmark_adjustment_parts(
        "ficelle/auto-compression", model, _failed_evidence_state(decay=0.0), ports=_evidence_ports()
    )

    # The full -24 in all three: fresh, half-decayed, and fully expired.
    assert fresh["verified_bonus"] == -24.0
    assert half["verified_bonus"] == -24.0
    assert expired["verified_bonus"] == -24.0


def test_an_expired_failed_benchmark_keeps_its_full_penalty() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {
        "benchmark_results": {
            "openrouter::free": {
                "ficelle/auto-compression": {"status": "fail", "fresh": False, "decay": 0.0, "recorded_at": "older"}
            }
        }
    }

    parts = benchmark_adjustment_parts("ficelle/auto-compression", model, state, ports=_evidence_ports())

    # -16, the fresh benchmark penalty: the row is fully decayed but still the only verdict there is.
    assert parts["benchmark_bonus"] == -16.0


def test_a_verdict_about_another_test_shape_still_decays_to_nothing() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    state = {
        "verified_capabilities": {
            "openrouter::free": {
                # `reason: test_type` makes the fake ports report a test-type mismatch.
                "ficelle/auto-compression": {"status": "failed", "fresh": False, "decay": 0.0, "reason": "test_type"}
            }
        }
    }

    parts = benchmark_adjustment_parts("ficelle/auto-compression", model, state, ports=_evidence_ports())

    assert parts["verified_bonus"] == 0.0, "a verdict about a different probe is not evidence here"


def test_stale_failed_profile_evidence_marks_only_expired_condemnations() -> None:
    model = {"source": "openrouter", "upstream_id": "free"}
    ports = _evidence_ports()

    def verdict(state: dict[str, Any]) -> bool:
        return stale_failed_profile_evidence("ficelle/auto-compression", model, state, ports=ports)

    assert verdict(_failed_evidence_state(decay=0.0)) is True
    assert verdict(_failed_evidence_state(decay=1.0, fresh=True)) is False, "a fresh failure is gated, not demoted"
    assert verdict({}) is False
    rehabilitated = _failed_evidence_state(decay=0.0)
    rehabilitated["benchmark_results"] = {
        "openrouter::free": {"ficelle/auto-compression": {"status": "pass", "fresh": True}}
    }
    assert verdict(rehabilitated) is False, "a fresh pass clears the expired verdict"


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
    # No first_delta_ewma and no throughput recorded: both stay neutral, so speed_score collapses
    # to the latency term. auto-multimodal's own weights do not use speed_score at all, so
    # score_base is unaffected by this change — these fields are asserted for coverage only.
    assert score["first_delta_score"] == 0.5
    assert score["throughput_score"] == 0.5
    assert score["speed_score"] == 0.5
    assert score["context_score"] == 0.5
    assert score["evidence_status"] == "fresh"
    assert score["last_evidence_at"] == "newer"


def test_the_failure_penalty_is_the_ledger_weight_times_the_points() -> None:
    """The per-status arithmetic is proven on the ledger itself (`test_cooldowns_use_case`); this
    is the wiring. The legacy `consecutive_failures` fixture above proves the same penalty
    survives migration."""
    model = {"source": "openrouter", "upstream_id": "free"}
    ports = ModelScoringPorts(
        cooldown_key=lambda candidate: f"{candidate.get('source')}::{candidate.get('upstream_id')}",
        safe_int=lambda value, default: int(value) if value is not None else default,
        now_epoch=lambda: 1_000_000.0,
    )
    explanation_ports = ModelScoreExplanationPorts(
        scoring=ports,
        evidence=_evidence_ports(),
        model_has_any=lambda candidate, key, values: any(value in candidate.get(key, []) for value in values),
    )

    state = {
        "stats": {
            "openrouter::free": {
                "failure_ledger": {
                    "timeout": {"origin": "request", "status": "open", "weight": 2.5, "last_at": 1_000_000.0, "episode": 3}
                }
            }
        }
    }

    score = model_score_explanation("ficelle/auto-fast", model, state, ports=explanation_ports)

    assert score["failure_penalty"] == 30.0


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
