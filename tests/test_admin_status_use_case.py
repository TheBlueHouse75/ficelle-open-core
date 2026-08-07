from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ficelle.use_cases.admin_status import (
    ADMIN_STATUS_TOP_LEVEL_FIELDS,
    AdminStateBuilder,
    AdminStateBuildPorts,
    AdminStatusBuilder,
    AdminStatusBuildPorts,
    FusionAdminStatusBuilder,
    active_admin_jobs,
    candidate_rank_rows,
    active_model_cooldown_rows,
    active_provider_cooldown_rows,
    active_quota_cooldown_rows,
    admin_job_matches,
    build_admin_job_row,
    build_admin_catalog_summary,
    build_catalog_refresh_summary,
    build_admin_health_summary,
    build_admin_model_availability,
    build_admin_performance_history_rows,
    build_admin_profile_row,
    build_admin_profile_rows,
    build_admin_recovery_snapshot,
    build_admin_runtime_section,
    build_admin_runtime_snapshot,
    build_admin_runtime_summary,
    build_admin_status_payload,
    build_admin_status_document,
    cached_catalog_stale_reason,
    catalog_with_auto_scores,
    failed_profile_evidence_row,
    filter_cached_catalog_to_enabled_providers,
    load_cached_catalog_for_admin_status,
    model_history_row,
    quarantine_rows,
    record_admin_job_finish,
    stale_reason_keeps_last_known,
    record_admin_job_start,
    safe_quota_cooldown_row,
    selected_profile_origin,
)


def normalize_fusion_config(raw: Any, *, strict: bool = False) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(source.get("enabled")),
        "budget_policy": source.get("budget_policy") or "strict_zero",
        "raw_output_policy": source.get("raw_output_policy") or "metadata_only",
        "max_total_calls": source.get("max_total_calls") or 9,
    }


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def make_builder() -> FusionAdminStatusBuilder:
    return FusionAdminStatusBuilder(
        normalize_fusion_config=normalize_fusion_config,
        fusion_call_multiplier=lambda fusion: int(fusion.get("max_total_calls") or 0),
        judge_history_limit=50,
        reason_code_pattern=re.compile(r"^[a-z][a-z0-9_.-]{0,63}$"),
    )


def test_fusion_admin_status_filters_and_redacts_last_run_metadata():
    builder = make_builder()

    last_run = builder.safe_last_run(
        {
            "last_fusion_run": {
                "request_id": "req-123",
                "status": "ok",
                "reason_code": "unsafe detail with secret-token-123",
                "stage_latency_seconds": {"panel": 1.2},
                "panel_models": ["ficelle/openrouter/model-a"],
                "prompt": "secret-token-123 prompt",
                "raw_outputs": ["secret-token-123 answer"],
            }
        }
    )

    assert last_run["request_id"] == "req-123"
    assert last_run["reason_code"] == "unsafe_reason_code"
    assert last_run["stage_latency_seconds"] == {"panel": 1.2}
    assert "prompt" not in last_run
    assert "raw_outputs" not in last_run
    assert "secret-token-123" not in repr(last_run)


def test_admin_state_builder_assembles_operator_payload_and_runs_quota_probe():
    calls: list[str] = []
    stale_row_states: list[dict] = []
    catalog = {"models": [{"id": "m1", "invokable": True}]}
    state = {"last_routes": {"ficelle/auto-fast": {"status": "ok"}}}

    builder = AdminStateBuilder(
        AdminStateBuildPorts(
            load_or_refresh_catalog=lambda _config: catalog,
            run_due_quota_probes=lambda _config, _catalog: calls.append("quota")
            or {"status": "ok"},
            load_runtime_state=lambda: state,
            normalized_virtual_profiles=lambda _config: {"ficelle/auto-fast": {"mode": "auto"}},
            build_admin_status=lambda _catalog, _config, _state: {
                "profiles": {"ficelle/auto-fast": {"selected_model": "m1"}},
                "performance_history": {"ficelle/auto-fast": []},
                "compression": {"health_digest": "ok"},
            },
            catalog_with_auto_scores=lambda source, _config, _state: {**source, "scored": True},
            fusion_payload=lambda _config, _state: {"config": {"enabled": False}},
            router_settings_payload=lambda _config: {"config": {"max_attempts_per_request": 3}},
            redact_runtime_state=lambda source: {"last_routes": source.get("last_routes", {})},
            stale_profile_model_rows=lambda _config, _catalog, row_state: stale_row_states.append(row_state)
            or [
                {
                    "profile_id": "ficelle/auto-fast",
                    "model_id": "ficelle/openrouter/retired",
                    "source": "openrouter",
                    "field": "models",
                    "disposition": "retired",
                }
            ],
            safe_int=lambda value, default: int(value) if value is not None else default,
            safe_string_list=lambda value: [str(item) for item in value]
            if isinstance(value, list)
            else [],
        ),
        default_min_context=128000,
        default_canary_profiles=["ficelle/auto-fast"],
    )

    payload = builder.build({"host": "127.0.0.2", "port": 9999, "catalog_ttl_seconds": 60})

    assert calls == ["quota"]
    assert payload["catalog"]["scored"] is True
    assert payload["virtual_profiles"] == {"ficelle/auto-fast": {"mode": "auto"}}
    # Carried into the operator payload beside the profiles it annotates: the dashboard needs
    # both to tell "provider down, kept" from "retired, going away" on the same card.
    assert payload["stale_profile_models"] == [
        {
            "profile_id": "ficelle/auto-fast",
            "model_id": "ficelle/openrouter/retired",
            "source": "openrouter",
            "field": "models",
            "disposition": "retired",
        }
    ]
    # The un-redacted runtime state, not the payload's redacted copy: the previous refresh's
    # per-provider counts live there, and they are what separates "retired" from "cannot tell".
    assert stale_row_states == [state]
    assert payload["profiles"]["ficelle/auto-fast"]["selected_model"] == "m1"
    assert payload["performance_history"] == {"ficelle/auto-fast": []}
    assert payload["compression"] == {"health_digest": "ok"}
    assert payload["fusion"] == {"config": {"enabled": False}}
    assert payload["settings"] == {"config": {"max_attempts_per_request": 3}}
    assert payload["state"] == {"last_routes": state["last_routes"]}
    assert payload["config"] == {
        "min_context_length": 128000,
        "allow_paid_fallback": False,
        "catalog_ttl_seconds": 60,
        "canary_profiles": ["ficelle/auto-fast"],
        "base_url": "http://127.0.0.2:9999/v1",
        "hermes_base_url": "http://127.0.0.2:9999/v1",
    }


def test_catalog_with_auto_scores_enriches_copy_without_mutating_source():
    ttl_calls: list[dict[str, Any]] = []
    catalog = {
        "models": [
            {"id": "m1", "free_access": {"eligible": False}},
            "not-a-model",
        ]
    }
    config = {"virtual_profiles": {"ficelle/auto-fast": {}, "ficelle/auto-json": {}}}

    enriched = catalog_with_auto_scores(
        catalog,
        config,
        {},
        apply_verified_capability_ttl=lambda source: ttl_calls.append(source),
        normalized_virtual_profiles=lambda source: source["virtual_profiles"],
        normalized_free_access=lambda model: {"eligible": model["id"] == "m1", "mode": "test"},
        model_score_explanation=lambda profile_id, model, _state: {
            "score_total": 90.0 if profile_id == "ficelle/auto-fast" else 70.0,
            "model_id": model["id"],
        },
    )

    assert ttl_calls == [config]
    assert enriched is not catalog
    assert enriched["models"][0] is not catalog["models"][0]
    assert catalog["models"][0]["free_access"] == {"eligible": False}
    assert enriched["models"][0]["free_access"] == {"eligible": True, "mode": "test"}
    assert enriched["models"][0]["auto_scores"] == {
        "ficelle/auto-fast": 90.0,
        "ficelle/auto-json": 70.0,
    }
    assert enriched["models"][0]["auto_score_details"]["ficelle/auto-fast"]["model_id"] == "m1"


def test_admin_job_helpers_record_redact_finish_and_filter_active_jobs():
    row = build_admin_job_row(
        "job-1",
        {"token": "plainsecret12345", "profiles": ["ficelle/auto-fast"]},
        now_iso=lambda: "2026-06-22T00:00:00+00:00",
        now_epoch=lambda: 100.0,
    )
    state: dict[str, Any] = {"admin_jobs": "invalid"}

    record_admin_job_start(state, "benchmark", row)
    active = active_admin_jobs(
        state,
        now_ts=160.0,
        max_age_seconds=120.0,
        safe_float=lambda value, default: float(value) if value is not None else default,
        safe_state_key=lambda value: str(value).replace("/", "_"),
    )

    assert admin_job_matches(state, "benchmark", "job-1") is True
    assert active["benchmark"]["id"] == "job-1"
    assert active["benchmark"]["metadata"]["profiles"] == ["ficelle/auto-fast"]
    assert "plainsecret12345" not in repr(active)

    record_admin_job_finish(state, "benchmark", "other-job")
    assert admin_job_matches(state, "benchmark", "job-1") is True

    record_admin_job_finish(state, "benchmark", "job-1")
    assert state["admin_jobs"] == {}


def test_active_admin_jobs_ignores_stale_invalid_and_non_running_rows():
    state = {
        "admin_jobs": {
            "fresh/job": {"id": "fresh", "status": "running", "started_at_epoch": 90.0},
            "stale": {"id": "stale", "status": "running", "started_at_epoch": 1.0},
            "done": {"id": "done", "status": "done", "started_at_epoch": 90.0},
            "bad": "invalid",
        }
    }

    active = active_admin_jobs(
        state,
        now_ts=100.0,
        max_age_seconds=60.0,
        safe_float=lambda value, default: float(value) if value is not None else default,
        safe_state_key=lambda value: str(value).replace("/", "_"),
    )

    assert active == {
        "fresh_job": {"id": "fresh", "status": "running", "started_at_epoch": 90.0}
    }


def test_fusion_admin_status_observability_readiness_and_stats():
    builder = make_builder()

    payload = builder.fusion_payload(
        {"fusion": {"enabled": True, "max_total_calls": 11}},
        {
            "last_fusion_run": {
                "status": "ok",
                "degraded_flags": {"single_panel_degraded": True},
            },
            "fusion_judge_history": ["ok", "invalid_json", "invalid_schema", 123],
            "fusion_quality_signal": {"model-a": {"value": 0.8}, "bad": "skip"},
        },
    )

    assert payload["config"]["enabled"] is True
    observability = payload["observability"]
    assert observability["readiness"] == "degraded"
    assert observability["call_multiplier_estimate"] == 11
    assert observability["free_models_only"] is True
    assert observability["judge_failure_stats"] == {
        "samples": 3,
        "invalid_json": 1,
        "invalid_schema": 1,
        "invalid_total": 2,
        "invalid_rate": 0.6667,
        "window_limit": 50,
    }
    assert observability["peer_ranking_quality_signal"] == {"model-a": {"value": 0.8}}


def test_admin_health_summary_reports_ok_with_default_action():
    summary = build_admin_health_summary(
        invokable_model_count=1,
        profile_rows={"ficelle/auto-fast": {"candidate_count": 1}},
        critical_profile_ids=("ficelle/auto-fast",),
        active_model_cooldowns=[],
        active_provider_cooldowns=[],
        active_quota_cooldowns=[],
        quarantines=[],
        canary_status="ok",
        auto_benchmark={},
        compression={},
    )

    assert summary == {
        "status": "ok",
        "reasons": [],
        "actions": ["Continue dogfood and monitor route logs/canaries."],
    }


def test_admin_health_summary_marks_stale_catalog_degraded():
    summary = build_admin_health_summary(
        invokable_model_count=5,
        profile_rows={"ficelle/auto-fast": {"candidate_count": 1}},
        critical_profile_ids=("ficelle/auto-fast",),
        active_model_cooldowns=[],
        active_provider_cooldowns=[],
        active_quota_cooldowns=[],
        quarantines=[],
        canary_status="ok",
        auto_benchmark={},
        compression={},
        catalog_stale=True,
    )

    # A stale-but-usable catalog (models present) is degraded — not "ok" (over-claim) and
    # not "fail" (a false outage on an idle-but-healthy instance).
    assert summary["status"] == "degraded"
    assert "catalog_stale" in summary["reasons"]


def test_admin_health_summary_stale_catalog_without_models_stays_fail():
    summary = build_admin_health_summary(
        invokable_model_count=0,
        profile_rows={"ficelle/auto-fast": {"candidate_count": 0}},
        critical_profile_ids=("ficelle/auto-fast",),
        active_model_cooldowns=[],
        active_provider_cooldowns=[],
        active_quota_cooldowns=[],
        quarantines=[],
        canary_status="ok",
        auto_benchmark={},
        compression={},
        catalog_stale=True,
    )

    # Staleness must NOT downgrade a real "no models" outage (e.g. a structural change that
    # legitimately empties the catalog) to merely degraded — that stays fail.
    assert summary["status"] == "fail"
    assert "no_invokable_models" in summary["reasons"]
    assert "catalog_stale" in summary["reasons"]


def test_stale_reason_keeps_last_known_is_the_recoverable_set():
    # Only "aging / structure-confirmed-unchanged" reasons keep the last-known catalog;
    # every other reason — including any unrecognised future one — fails safe to emptying it.
    assert stale_reason_keeps_last_known("ttl_expired") is True
    assert stale_reason_keeps_last_known("credential_fingerprint_mismatch") is True
    for emptied in (
        "config_structural_fingerprint_mismatch",
        "config_fingerprint_mismatch",
        "invalid_generated_at",
        "some_future_reason",
        None,
    ):
        assert stale_reason_keeps_last_known(emptied) is False


def test_admin_health_summary_reports_degraded_recovery_actions():
    summary = build_admin_health_summary(
        invokable_model_count=1,
        profile_rows={"ficelle/auto-fast": {"candidate_count": 1}},
        critical_profile_ids=("ficelle/auto-fast",),
        active_model_cooldowns=[{"reason": "server_error"}],
        active_provider_cooldowns=[],
        active_quota_cooldowns=[{"reason": "quota_exhausted"}],
        quarantines=[],
        canary_status="ok",
        auto_benchmark={"last_error": {"type": "RuntimeError"}},
        compression={"health_digest": "compression_errors_repeated"},
    )

    assert summary["status"] == "degraded"
    assert summary["reasons"] == [
        "active_model_cooldowns",
        "active_quota_cooldowns",
        "auto_benchmark_background_error",
        "compression_errors_repeated",
    ]
    assert any("model cooldown" in action for action in summary["actions"])
    assert any("Free-quota cooldown" in action for action in summary["actions"])
    assert any("auto-benchmark" in action for action in summary["actions"])
    assert any("Compression" in action for action in summary["actions"])


def test_admin_health_summary_reports_fail_for_missing_critical_candidates():
    summary = build_admin_health_summary(
        invokable_model_count=0,
        profile_rows={"ficelle/auto-fast": {"candidate_count": 0}},
        critical_profile_ids=("ficelle/auto-fast",),
        active_model_cooldowns=[],
        active_provider_cooldowns=[],
        active_quota_cooldowns=[],
        quarantines=[{"reason": "billing_or_paid"}],
        canary_status="fail",
        auto_benchmark={},
        compression={},
    )

    assert summary["status"] == "fail"
    assert summary["reasons"] == [
        "no_invokable_models",
        "critical_profile_without_candidates:ficelle/auto-fast",
        "active_quarantine",
        "canary_failed",
    ]
    assert summary["actions"][0].startswith("Check provider credentials")
    assert any("quarantined models" in action for action in summary["actions"])
    assert any("Run ficelle canary" in action for action in summary["actions"])


def test_admin_catalog_summary_preserves_contract_fields():
    summary = build_admin_catalog_summary(
        {
            "generated_at": "2026-06-21T10:00:00+00:00",
            "min_context_length": 128000,
            "providers": {"openrouter": {"model_count": 2}},
        },
        model_count=3,
        invokable_count=2,
        available_count=1,
    )

    assert summary == {
        "generated_at": "2026-06-21T10:00:00+00:00",
        "min_context_length": 128000,
        "allow_paid_fallback": False,
        "model_count": 3,
        "invokable_count": 2,
        "available_count": 1,
        "providers": {"openrouter": {"model_count": 2}},
    }


def test_admin_catalog_summary_marks_stale_cached_catalog():
    summary = build_admin_catalog_summary(
        {
            "generated_at": "2026-06-21T10:00:00+00:00",
            "min_context_length": 128000,
            "stale": True,
            "stale_reason": "config_fingerprint_mismatch",
        },
        model_count=0,
        invokable_count=0,
        available_count=0,
    )

    assert summary["stale"] is True
    assert summary["stale_reason"] == "config_fingerprint_mismatch"
    assert summary["model_count"] == 0


def test_load_cached_catalog_for_admin_status_falls_back_for_invalid_cache():
    fallback = load_cached_catalog_for_admin_status(
        {"providers": {"openrouter": {}}},
        {"min_context_length": 32000},
        default_min_context=128000,
    )
    existing = {"models": [{"id": "ficelle/openrouter/free"}], "providers": {}}

    assert fallback == {
        "generated_at": None,
        "min_context_length": 32000,
        "providers": {},
        "models": [],
    }
    assert load_cached_catalog_for_admin_status(existing, {}, default_min_context=128000) is existing


def test_cached_catalog_stale_reason_preserves_fingerprint_and_ttl_contracts():
    config = {"catalog_ttl_seconds": 3600}
    catalog = {
        "generated_at": "2026-06-21T10:00:00+00:00",
        "config_fingerprint": "full",
        "config_structural_fingerprint": "structural",
    }

    fresh = cached_catalog_stale_reason(
        catalog,
        config,
        now_seconds=lambda: timestamp("2026-06-21T10:30:00+00:00"),
        catalog_config_fingerprint=lambda _config: "full",
        catalog_config_structural_fingerprint=lambda _config: "structural",
    )
    structural_changed = cached_catalog_stale_reason(
        {**catalog, "config_structural_fingerprint": "old-structural"},
        config,
        now_seconds=lambda: timestamp("2026-06-21T10:30:00+00:00"),
        catalog_config_fingerprint=lambda _config: "full",
        catalog_config_structural_fingerprint=lambda _config: "structural",
    )
    credentials_changed = cached_catalog_stale_reason(
        {**catalog, "config_fingerprint": "old-full"},
        config,
        now_seconds=lambda: timestamp("2026-06-21T10:30:00+00:00"),
        catalog_config_fingerprint=lambda _config: "full",
        catalog_config_structural_fingerprint=lambda _config: "structural",
    )
    expired = cached_catalog_stale_reason(
        catalog,
        config,
        now_seconds=lambda: timestamp("2026-06-21T11:00:01+00:00"),
        catalog_config_fingerprint=lambda _config: "full",
        catalog_config_structural_fingerprint=lambda _config: "structural",
    )

    assert fresh is None
    assert structural_changed == "config_structural_fingerprint_mismatch"
    assert credentials_changed == "credential_fingerprint_mismatch"
    assert expired == "ttl_expired"


def test_filter_cached_catalog_to_enabled_providers_keeps_only_invokable_sources():
    config = {
        "providers": {
            "openrouter": {"enabled": True},
            "nvidia": {"activation_policy": "configured_credentials"},
            "groq": {"enabled": False},
            "broken": "not-a-provider",
        }
    }
    catalog = {
        "providers": {
            "openrouter": {"accepted_count": 1},
            "nvidia": {"accepted_count": 1},
            "groq": {"accepted_count": 1},
            "missing": {"accepted_count": 1},
        },
        "models": [
            {"id": "ficelle/openrouter/a", "source": "openrouter"},
            {"id": "ficelle/nvidia/a", "source": "nvidia"},
            {"id": "ficelle/groq/a", "source": "groq"},
            {"id": "ficelle/missing/a", "source": "missing"},
        ],
    }

    filtered = filter_cached_catalog_to_enabled_providers(
        catalog,
        config,
        provider_auth_row=lambda source, _config: {"invokable": source == "nvidia"},
    )

    assert filtered["providers"] == {
        "openrouter": {"accepted_count": 1},
        "nvidia": {"accepted_count": 1},
    }
    assert [model["source"] for model in filtered["models"]] == ["openrouter", "nvidia"]


def test_catalog_refresh_summary_preserves_refresh_contract_shape():
    summary = build_catalog_refresh_summary(
        {
            "generated_at": "2026-06-22T00:00:00+00:00",
            "min_context_length": 128000,
            "providers": {"openrouter": {"accepted_count": 1}},
            "models": [
                {"id": "model-a", "invokable": True},
                {"id": "model-b", "invokable": False},
                "bad-row",
            ],
        }
    )

    assert summary == {
        "generated_at": "2026-06-22T00:00:00+00:00",
        "min_context_length": 128000,
        "allow_paid_fallback": False,
        "model_count": 2,
        "invokable_count": 1,
        "providers": {"openrouter": {"accepted_count": 1}},
    }
    assert "available_count" not in summary


def test_admin_runtime_summary_preserves_contract_fields():
    summary = build_admin_runtime_summary(
        active_cooldowns_count=3,
        active_model_cooldowns_count=1,
        active_provider_cooldowns_count=1,
        active_quota_cooldowns_count=1,
        quarantine_count=2,
        stats_records=4,
        provider_stats_records=5,
        provider_error_count=6,
        model_error_count=7,
        last_route_count=8,
        compression_error_count=9,
        compression_last_status="error",
        state_diagnostics={"status": "ok"},
        canary={"status": "pass"},
        auto_benchmark={"status": "idle"},
    )

    assert summary == {
        "active_cooldowns_count": 3,
        "active_model_cooldowns_count": 1,
        "active_provider_cooldowns_count": 1,
        "active_quota_cooldowns_count": 1,
        "quarantine_count": 2,
        "stats_records": 4,
        "provider_stats_records": 5,
        "provider_error_count": 6,
        "model_error_count": 7,
        "last_route_count": 8,
        "compression_error_count": 9,
        "compression_last_status": "error",
        "state_diagnostics": {"status": "ok"},
        "canary": {"status": "pass"},
        "auto_benchmark": {"status": "idle"},
    }


def test_admin_runtime_section_builds_counts_from_snapshots():
    recovery = build_admin_recovery_snapshot(
        state={},
        active_model_cooldown_rows=lambda state: [{"scope": "model"}],
        active_provider_cooldown_rows=lambda state: [{"scope": "provider"}],
        active_quota_cooldown_rows=lambda state: [{"scope": "quota"}],
        quarantine_rows=lambda state: [{"model_id": "model-a"}],
    )
    runtime = build_admin_runtime_snapshot(
        {
            "provider_errors": {"openrouter": {"reason": "rate_limit"}},
            "model_errors": {"model-a": {"reason": "server_error"}},
            "last_routes": {"ficelle/auto-fast": {"status": "ok"}},
            "canary": {"status": "pass"},
            "auto_benchmark": {"status": "idle"},
        }
    )

    section = build_admin_runtime_section(
        state={
            "stats": {"model-a": {}},
            "provider_stats": {"openrouter": {}, "groq": {}},
        },
        recovery=recovery,
        runtime=runtime,
        compression={"error_count": "5", "last_status": "error"},
        state_diagnostics={"status": "ok"},
        safe_int=lambda value, default: int(value) if str(value).isdigit() else default,
    )

    assert section == {
        "active_cooldowns_count": 3,
        "active_model_cooldowns_count": 1,
        "active_provider_cooldowns_count": 1,
        "active_quota_cooldowns_count": 1,
        "quarantine_count": 1,
        "stats_records": 1,
        "provider_stats_records": 2,
        "provider_error_count": 1,
        "model_error_count": 1,
        "last_route_count": 1,
        "compression_error_count": 5,
        "compression_last_status": "error",
        "state_diagnostics": {"status": "ok"},
        "canary": {"status": "pass"},
        "auto_benchmark": {"status": "idle"},
    }


def test_admin_runtime_snapshot_normalizes_missing_sections():
    snapshot = build_admin_runtime_snapshot(
        {
            "last_routes": [],
            "provider_errors": None,
            "model_errors": "bad",
            "quota_probe_results": 42,
            "canary": {"status": "pass"},
        }
    )

    assert snapshot.last_routes == {}
    assert snapshot.provider_errors == {}
    assert snapshot.model_errors == {}
    assert snapshot.quota_probe_results == {}
    assert snapshot.canary == {"status": "pass"}
    assert snapshot.canary_status == "pass"
    assert snapshot.auto_benchmark == {}


def test_admin_runtime_snapshot_redacts_auto_benchmark_error_details():
    snapshot = build_admin_runtime_snapshot(
        {
            "last_routes": {"ficelle/auto-fast": {"status": "ok"}},
            "provider_errors": {"openrouter": {"reason": "rate_limit"}},
            "model_errors": {"model-a": {"reason": "server_error"}},
            "quota_probe_results": {"openrouter": {"status": "ok"}},
            "canary": {},
            "auto_benchmark": {
                "last_error": {
                    "type": "RuntimeError",
                    "message": "token sk-test-123 should not leak",
                    "api_key": "sk-test-123",
                }
            },
        }
    )

    assert snapshot.last_routes == {"ficelle/auto-fast": {"status": "ok"}}
    assert snapshot.provider_errors == {"openrouter": {"reason": "rate_limit"}}
    assert snapshot.model_errors == {"model-a": {"reason": "server_error"}}
    assert snapshot.quota_probe_results == {"openrouter": {"status": "ok"}}
    assert snapshot.canary == {}
    assert snapshot.canary_status == "unknown"
    assert "sk-test-123" not in repr(snapshot.auto_benchmark)
    assert snapshot.auto_benchmark["last_error"]["api_key"] == "[redacted]"


def test_admin_model_availability_filters_non_invokable_cooldown_and_quarantine():
    ready = {"id": "ready", "invokable": True}
    not_invokable = {"id": "not-invokable", "invokable": False}
    cooldown = {"id": "cooldown", "invokable": True}
    quarantined = {"id": "quarantined", "invokable": True}

    availability = build_admin_model_availability(
        catalog={"models": [ready, "bad-row", not_invokable, cooldown, quarantined]},
        state={"marker": True},
        model_on_cooldown=lambda model, state: (model["id"] == "cooldown", "rate_limit"),
        model_is_quarantined=lambda model, state: model["id"] == "quarantined",
    )

    assert availability.models == [ready, not_invokable, cooldown, quarantined]
    assert availability.invokable_models == [ready, cooldown, quarantined]
    assert availability.available_models == [ready]


def test_admin_recovery_snapshot_aggregates_cooldowns_and_quarantines():
    snapshot = build_admin_recovery_snapshot(
        state={"marker": True},
        active_model_cooldown_rows=lambda state: [{"scope": "model"}],
        active_provider_cooldown_rows=lambda state: [{"scope": "provider"}],
        active_quota_cooldown_rows=lambda state: [{"scope": "quota"}],
        quarantine_rows=lambda state: [{"model_id": "model-a"}],
    )

    assert snapshot.active_model_cooldowns == [{"scope": "model"}]
    assert snapshot.active_provider_cooldowns == [{"scope": "provider"}]
    assert snapshot.active_quota_cooldowns == [{"scope": "quota"}]
    assert snapshot.active_cooldowns == [
        {"scope": "model"},
        {"scope": "provider"},
        {"scope": "quota"},
    ]
    assert snapshot.quarantines == [{"model_id": "model-a"}]


def test_active_model_cooldown_rows_filters_expired_and_sorts_by_remaining_time():
    rows = active_model_cooldown_rows(
        {
            "cooldowns": {
                "slow": {"until": 150.0, "reason": "server_error", "detail": "HTTP 500"},
                "fast": {"until": 120.0, "reason": "timeout", "set_at": "now"},
                "expired": {"until": 90.0, "reason": "gone"},
                "bad": {"until": "not-a-number"},
            }
        },
        100.0,
        safe_state_key=str,
        safe_detail=lambda value: None if value is None else str(value),
        safe_int=lambda value, default: int(value) if value is not None else default,
        epoch_to_iso=lambda value: f"iso:{value}",
    )

    assert [row["key"] for row in rows] == ["fast", "slow"]
    assert rows[0]["scope"] == "model"
    assert rows[0]["seconds_remaining"] == 20
    assert rows[0]["until_iso"] == "iso:120.0"
    assert rows[1]["detail"] == "HTTP 500"


def test_active_provider_cooldown_rows_adds_source_and_sorts():
    rows = active_provider_cooldown_rows(
        {
            "provider_cooldowns": {
                "z-provider": {"until": 130.0, "reason": "auth_or_credit"},
                "a-provider": {"until": 130.0, "reason": "rate_limited"},
            }
        },
        100.0,
        safe_state_key=str,
        safe_detail=lambda value: None if value is None else str(value),
        safe_int=lambda value, default: int(value) if value is not None else default,
        epoch_to_iso=lambda value: f"iso:{value}",
    )

    assert [row["key"] for row in rows] == ["a-provider", "z-provider"]
    assert [row["source"] for row in rows] == ["a-provider", "z-provider"]
    assert {row["scope"] for row in rows} == {"provider"}


def test_safe_quota_cooldown_row_formats_probe_metadata_and_active_flag():
    row = safe_quota_cooldown_row(
        "model:nvidia::model-a",
        {
            "source": "nvidia",
            "model_id": "ficelle/nvidia/model-a",
            "upstream_id": "model-a",
            "until": 150.0,
            "next_probe_at": 90.0,
            "probe_interval_seconds": "30",
            "consecutive_probe_failures": "2",
            "detail": "quota exhausted",
        },
        100.0,
        quota_cooldown_scope_from_key=lambda key: key.split(":", 1)[0],
        safe_float=lambda value, default: float(value) if value is not None else default,
        safe_state_key=str,
        safe_detail=lambda value: None if value is None else str(value),
        safe_int=lambda value, default: int(value) if value is not None else default,
        epoch_to_iso=lambda value: f"iso:{value}",
        include_active=True,
    )

    assert row["scope"] == "model"
    assert row["reason"] == "quota_exhausted"
    assert row["seconds_remaining"] == 50
    assert row["next_probe_at_iso"] == "iso:90.0"
    assert row["probe_due"] is True
    assert row["probe_interval_seconds"] == 30
    assert row["consecutive_probe_failures"] == 2
    assert row["active"] is True


def test_active_quota_cooldown_rows_filters_expired_and_sorts():
    rows = active_quota_cooldown_rows(
        {
            "quota_cooldowns": {
                "provider:nvidia": {"until": 160.0, "source": "nvidia"},
                "account:nvidia::acct-a": {"until": 120.0, "source": "nvidia"},
                "expired:nvidia": {"until": 90.0},
                "bad-row": "skip",
            }
        },
        100.0,
        quota_cooldown_scope_from_key=lambda key: key.split(":", 1)[0],
        safe_float=lambda value, default: float(value) if value is not None else default,
        safe_state_key=str,
        safe_detail=lambda value: None if value is None else str(value),
        safe_int=lambda value, default: int(value) if value is not None else default,
        epoch_to_iso=lambda value: f"iso:{value}",
    )

    assert [row["key"] for row in rows] == ["account:nvidia::acct-a", "provider:nvidia"]
    assert [row["seconds_remaining"] for row in rows] == [20, 60]
    assert [row["scope"] for row in rows] == ["account", "provider"]


def test_quarantine_rows_redacts_defaults_and_sorts():
    rows = quarantine_rows(
        {
            "quarantine": {
                "z-key": {
                    "reason": "model_not_found",
                    "note": "missing",
                    "set_at": "2026-06-22T00:00:00+00:00",
                    "source": "model_not_found_guard",
                    "manual": False,
                    "model_id": "model-z",
                    "upstream_id": "upstream-z",
                },
                "a-key": {"model_id": "model-a", "manual": True},
                "bad-row": "skip",
            }
        },
        safe_state_key=lambda value: f"key:{value}",
        safe_detail=lambda value: None if value is None else f"safe:{value}",
    )

    assert [row["key"] for row in rows] == ["key:a-key", "key:z-key"]
    assert rows[0]["reason"] == "quarantined"
    assert rows[0]["source"] == "manual"
    assert rows[0]["manual"] is True
    assert rows[1]["reason"] == "safe:model_not_found"
    assert rows[1]["source"] == "safe:model_not_found_guard"
    assert rows[1]["manual"] is False


def test_failed_profile_evidence_row_prefers_verified_capability_failure():
    model = {"id": "model-a", "upstream_id": "up-a", "source": "test"}

    row = failed_profile_evidence_row(
        "ficelle/auto-json@source",
        model,
        {},
        canonical_virtual_model_id=lambda profile_id: profile_id.split("@", 1)[0],
        verified_capability_row=lambda _profile_id, _model, _state: {
            "status": "failed",
            "message": "expected JSON",
            "test_type": "json",
            "failed_at": "2026-06-22T00:00:00+00:00",
        },
        raw_benchmark_row=lambda *_args: {"status": "fail", "message": "ignored benchmark"},
        benchmark_result_matches_current_test=lambda *_args: True,
        redacted_benchmark_row=lambda _profile_id, row: row,
        safe_detail=lambda value: None if value is None else str(value),
    )

    assert row == {
        "profile_id": "ficelle/auto-json",
        "model_id": "model-a",
        "upstream_id": "up-a",
        "source": "test",
        "reason": "failed_capability_check",
        "message": "expected JSON",
        "test_type": "json",
        "failed_at": "2026-06-22T00:00:00+00:00",
        "evidence_source": "verified_capability",
    }


def test_failed_profile_evidence_row_uses_current_failed_benchmark():
    model = {"id": "model-b", "upstream_id": "up-b", "source": "test"}

    row = failed_profile_evidence_row(
        "ficelle/auto-tools",
        model,
        {},
        canonical_virtual_model_id=lambda profile_id: profile_id,
        verified_capability_row=lambda *_args: {},
        raw_benchmark_row=lambda *_args: {
            "status": "fail",
            "message": "tool call missing",
            "test_type": "tool_call",
            "ran_at": "2026-06-22T00:01:00+00:00",
        },
        benchmark_result_matches_current_test=lambda *_args: True,
        redacted_benchmark_row=lambda _profile_id, row: {**row, "message": "[redacted]"},
        safe_detail=lambda value: None if value is None else str(value),
    )

    assert row["reason"] == "failed_benchmark"
    assert row["message"] == "[redacted]"
    assert row["test_type"] == "tool_call"
    assert row["failed_at"] == "2026-06-22T00:01:00+00:00"
    assert row["evidence_source"] == "benchmark_result"


def test_failed_profile_evidence_row_returns_empty_without_current_failure():
    row = failed_profile_evidence_row(
        "ficelle/auto-tools",
        {"id": "model-c"},
        {},
        canonical_virtual_model_id=lambda profile_id: profile_id,
        verified_capability_row=lambda *_args: {"status": "verified"},
        raw_benchmark_row=lambda *_args: {"status": "fail"},
        benchmark_result_matches_current_test=lambda *_args: False,
        redacted_benchmark_row=lambda _profile_id, source: source,
        safe_detail=lambda value: None if value is None else str(value),
    )

    assert row == {}


def test_model_history_row_assembles_stats_evidence_and_rank_details():
    model = {"id": "model-a", "upstream_id": "up-a", "source": "test"}
    rank_row = {
        "rank": 2,
        "origin": "auto_tail",
        "manual_rank": None,
        "free_access": {"mode": "catalog_free"},
        "score": 82.0,
        "score_base": 80.0,
        "score_adjustment": 2.0,
        "benchmark_bonus": 1.0,
        "verified_bonus": 1.0,
        "score_decay_factor": 0.9,
        "failure_penalty": 0.0,
        "evidence_status": "fresh",
        "evidence_reason": "verified",
        "benchmark_evidence_status": "fresh",
        "benchmark_evidence_reason": "pass",
        "verified_evidence_status": "fresh",
        "verified_evidence_reason": "verified",
        "last_evidence_at": "2026-06-22T00:02:00+00:00",
    }

    row = model_history_row(
        "ficelle/auto-tools",
        model,
        {
            "stats": {
                "test::up-a": {
                    "requests": "3",
                    "successes": "2",
                    "failures": "1",
                    "consecutive_failures": "1",
                    "latency_ewma": 0.4,
                    "last_failure_reason": "server_error",
                }
            }
        },
        rank_row,
        cooldown_key=lambda item: f"{item['source']}::{item['upstream_id']}",
        normalized_free_access=lambda _model: {"mode": "default"},
        raw_benchmark_row=lambda *_args: {
            "status": "fail",
            "message": "tool call missing",
            "test_type": "tool_call",
            "ran_at": "2026-06-22T00:01:00+00:00",
        },
        redacted_benchmark_row=lambda _profile_id, source: {**source, "message": "[redacted]"},
        verified_capability_row=lambda *_args: {
            "status": "failed",
            "capability": "tool_call",
            "failed_at": "2026-06-22T00:03:00+00:00",
        },
        failed_profile_evidence_row=lambda *_args: {
            "reason": "failed_capability_check",
            "message": "expected tool call",
            "failed_at": "2026-06-22T00:03:00+00:00",
        },
        safe_detail=lambda value: None if value is None else str(value),
        safe_int=lambda value, default: int(value) if value is not None else default,
    )

    assert row["rank"] == 2
    assert row["free_access"] == {"mode": "catalog_free"}
    assert row["requests"] == 3
    assert row["successes"] == 2
    assert row["benchmark_status"] == "fail"
    assert row["benchmark_message"] == "[redacted]"
    assert row["verified_status"] == "failed"
    assert row["failed_profile_evidence"] is True
    assert row["failed_profile_evidence_message"] == "expected tool call"


def test_candidate_rank_rows_assembles_manual_origin_and_route_metadata():
    candidates = [
        {
            "id": "model-manual",
            "upstream_id": "manual",
            "source": "test",
            "name": "Manual",
            "context_length": "131072",
            "reference_confidence": "high",
        },
        {"id": "model-tail", "upstream_id": "tail", "source": "test", "context_length": 262144},
    ]

    def score(_profile_id: str, model: dict[str, Any], _state: dict[str, Any]) -> dict[str, Any]:
        base = 90.0 if model["id"] == "model-manual" else 80.0
        return {
            "score_total": base,
            "score_base": base,
            "score_adjustment": 0.0,
            "benchmark_bonus": 0.0,
            "verified_bonus": 0.0,
            "score_decay_factor": 1.0,
            "failure_penalty": 0.0,
            "evidence_status": "fresh",
            "evidence_reason": "verified",
            "benchmark_evidence_status": "missing",
            "benchmark_evidence_reason": "none",
            "verified_evidence_status": "fresh",
            "verified_evidence_reason": "verified",
            "last_evidence_at": "2026-06-22T00:04:00+00:00",
        }

    rows = candidate_rank_rows(
        "ficelle/auto-tools@source",
        candidates,
        {"mode": "manual_order", "models": ["model-manual"]},
        {},
        canonical_virtual_model_id=lambda profile_id: profile_id.split("@", 1)[0],
        model_score_explanation=score,
        verified_capability_row=lambda *_args: {"status": "verified", "capability": "tool_call"},
        failed_profile_evidence_row=lambda _profile_id, model, _state: (
            {"reason": "failed_capability_check", "message": "failed"}
            if model["id"] == "model-tail"
            else {}
        ),
        model_route_competence=lambda _profile_id, model, _state: (
            "claimed_default" if model["id"] == "model-tail" else "verified"
        ),
        gate_capability_by_profile={"ficelle/auto-tools": "tools"},
        model_capability_provenance=lambda _model, capability: f"provenance:{capability}",
        normalized_free_access=lambda _model: {"mode": "catalog_free"},
        safe_int=lambda value, default: int(value) if value is not None else default,
    )

    assert [row["origin"] for row in rows] == ["manual", "auto_tail"]
    assert rows[0]["manual_rank"] == 1
    assert rows[0]["score"] == 90.0
    assert rows[0]["capability_provenance"] == "provenance:tools"
    assert rows[0]["route_eligible"] is True
    assert rows[1]["failed_profile_evidence"] is True
    assert rows[1]["failed_profile_evidence_reason"] == "failed_capability_check"
    assert rows[1]["route_eligible"] is False


def test_admin_profile_row_preserves_selected_contract_fields():
    row = build_admin_profile_row(
        profile={
            "mode": "manual_order",
            "models": ["ficelle/openrouter/manual-a"],
            "auto_tail": False,
            "requirements": {"tools": True},
        },
        candidates=[{"id": "ficelle/openrouter/manual-a"}],
        selected={"id": "ficelle/openrouter/manual-a", "upstream_id": "manual-a"},
        selected_origin="manual",
        selected_score={"score_total": 91.5, "latency": 0.3},
        selected_verified={"status": "verified", "capability": "tool_call"},
        selected_competence="verified",
        top_candidates=[{"model_id": "ficelle/openrouter/manual-a"}],
        failed_candidates=[],
        last_route={"status": "ok", "request_id": "req-1"},
    )

    assert row == {
        "status": "ok",
        "mode": "manual_order",
        "manual_models_count": 1,
        "auto_tail": False,
        "candidate_count": 1,
        "selected_model": "ficelle/openrouter/manual-a",
        "selected_upstream": "manual-a",
        "selected_origin": "manual",
        "selected_score": 91.5,
        "selected_score_details": {"score_total": 91.5, "latency": 0.3},
        "selected_competence": "verified",
        "selected_verified_status": "verified",
        "selected_verified_capability": "tool_call",
        "top_candidates": [{"model_id": "ficelle/openrouter/manual-a"}],
        "failed_profile_candidates": [],
        "last_route": {"status": "ok", "request_id": "req-1"},
        "requirements": {"tools": True},
    }


def test_admin_profile_row_reports_fail_without_candidates():
    row = build_admin_profile_row(
        profile={"mode": "auto"},
        candidates=[],
        selected=None,
        selected_origin=None,
        selected_score={},
        selected_verified={},
        selected_competence=None,
        top_candidates=[],
        failed_candidates=[{"model_id": "ficelle/openrouter/nope"}],
        last_route={},
    )

    assert row["status"] == "fail"
    assert row["manual_models_count"] == 0
    assert row["auto_tail"] is True
    assert row["candidate_count"] == 0
    assert row["selected_model"] is None
    assert row["selected_score"] is None
    assert row["selected_verified_status"] == "unknown"
    assert row["failed_profile_candidates"] == [{"model_id": "ficelle/openrouter/nope"}]


def test_selected_profile_origin_reports_manual_auto_tail_and_auto():
    assert selected_profile_origin({"mode": "auto"}, {"id": "model-a"}) == "auto"
    assert (
        selected_profile_origin(
            {"mode": "manual_order", "models": ["manual-a"]},
            {"id": "manual-a"},
        )
        == "manual"
    )
    assert (
        selected_profile_origin(
            {"mode": "manual_order", "models": ["manual-a"]},
            {"id": "auto-tail-a"},
        )
        == "auto_tail"
    )
    assert selected_profile_origin({"mode": "auto"}, None) is None


def test_admin_profile_rows_assembles_selected_and_failed_evidence():
    profiles = {
        "ficelle/auto-fast": {
            "mode": "manual_order",
            "models": ["ficelle/openrouter/manual-a"],
            "requirements": {"tools": True},
        }
    }
    selected = {
        "id": "ficelle/openrouter/manual-a",
        "upstream_id": "manual-a",
    }
    failed = {
        "id": "ficelle/openrouter/failed-a",
        "upstream_id": "failed-a",
    }

    rows = build_admin_profile_rows(
        profiles=profiles,
        available_models=[selected, failed],
        state={"marker": True},
        last_routes={"ficelle/auto-fast": {"status": "ok", "request_id": "req-1"}},
        candidates_for_profile=lambda profile_id, available, state, profile: available,
        route_competent_candidates=lambda profile_id, candidates, state: [selected],
        candidate_rank_rows=lambda profile_id, candidates, profile, state: [
            {"model_id": model["id"]}
            for model in candidates
        ],
        verified_capability_row=lambda profile_id, model, state: {
            "status": "verified",
            "capability": "tool_call",
        },
        model_score_explanation=lambda profile_id, model, state: {"score_total": 93.0},
        model_route_competence=lambda profile_id, model, state: "verified",
        failed_profile_evidence_row=lambda profile_id, model, state: (
            {"model_id": model["id"], "reason": "semantic_failure"}
            if model["id"].endswith("failed-a")
            else {}
        ),
    )

    row = rows["ficelle/auto-fast"]
    assert row["selected_model"] == "ficelle/openrouter/manual-a"
    assert row["selected_origin"] == "manual"
    assert row["selected_score"] == 93.0
    assert row["selected_verified_status"] == "verified"
    assert row["selected_competence"] == "verified"
    assert row["last_route"] == {"status": "ok", "request_id": "req-1"}
    assert row["top_candidates"] == [
        {"model_id": "ficelle/openrouter/manual-a"},
        {"model_id": "ficelle/openrouter/failed-a"},
    ]
    assert row["failed_profile_candidates"] == [
        {"model_id": "ficelle/openrouter/failed-a", "reason": "semantic_failure"}
    ]


def test_admin_performance_history_rows_preserves_route_and_compression_fields():
    selected = {
        "id": "ficelle/openrouter/model-a",
        "upstream_id": "model-a",
        "source": "openrouter",
    }
    second = {
        "id": "ficelle/openrouter/model-b",
        "upstream_id": "model-b",
        "source": "openrouter",
    }

    rows = build_admin_performance_history_rows(
        profiles={"ficelle/auto-fast": {"mode": "auto"}},
        available_models=[selected, second],
        state={"marker": True},
        last_routes={
            "ficelle/auto-fast": {
                "status": "ok",
                "reason": "selected",
                "model_id": "ficelle/openrouter/model-a",
                "upstream_id": "model-a",
                "request_id": "req-1",
                "seen_at": "2026-06-21T10:00:00+00:00",
                "latency_seconds": 0.42,
                "attempt_count": "2",
                "compression": {
                    "status": "applied",
                    "mode": "live",
                    "estimated_saved_chars": 1200,
                    "estimated_saved_tokens": 300,
                    "strategies": {"trim": 1},
                },
            }
        },
        candidates_for_profile=lambda profile_id, available, state, profile: available,
        candidate_rank_rows=lambda profile_id, candidates, profile, state, limit: [
            {"model_id": model["id"], "rank": index + 1}
            for index, model in enumerate(candidates[:limit])
        ],
        route_competent_candidates=lambda profile_id, candidates, state: [selected],
        model_history_row=lambda profile_id, model, state, rank_row: {
            "model_id": model["id"],
            "rank": rank_row.get("rank"),
        },
        model_auto_score=lambda profile_id, model, state: 91.26,
        model_route_competence=lambda profile_id, model, state: "verified",
        fallback_attempt_rows=lambda last_route: [{"reason": last_route.get("reason")}],
        safe_int=lambda value, default: int(value) if str(value).isdigit() else default,
    )

    row = rows["ficelle/auto-fast"]
    assert row["profile_id"] == "ficelle/auto-fast"
    assert row["candidate_count"] == 2
    assert row["selected_model"] == "ficelle/openrouter/model-a"
    assert row["selected_upstream"] == "model-a"
    assert row["selected_source"] == "openrouter"
    assert row["selected_score"] == 91.3
    assert row["selected_competence"] == "verified"
    assert row["last_route_status"] == "ok"
    assert row["last_route_attempt_count"] == 2
    assert row["compression_status"] == "applied"
    assert row["compression_mode"] == "live"
    assert row["compression_estimated_saved_chars"] == 1200
    assert row["compression_estimated_saved_tokens"] == 300
    assert row["compression_strategy_mix"] == {"trim": 1}
    assert row["fallback_reasons"] == [{"reason": "selected"}]
    assert row["models"] == [
        {"model_id": "ficelle/openrouter/model-a", "rank": 1},
        {"model_id": "ficelle/openrouter/model-b", "rank": 2},
    ]


def test_admin_status_payload_preserves_top_level_contract_fields():
    payload = build_admin_status_payload(
        status="degraded",
        reasons=["active_model_cooldowns"],
        generated_at="2026-06-21T10:00:00+00:00",
        catalog={"model_count": 2},
        profiles={"ficelle/auto-fast": {"status": "ok"}},
        fusion={"config": {"enabled": True}},
        critical_profiles=("ficelle/auto-fast",),
        runtime={"active_cooldowns_count": 1},
        compression={"status": "ok"},
        cooldowns=[{"model_id": "model-a"}],
        model_cooldowns=[{"model_id": "model-a"}],
        provider_cooldowns=[],
        quota_cooldowns=[],
        quota_probe_results={"provider-a": {"status": "ok"}},
        provider_errors={"provider-a": {"reason": "rate_limit"}},
        model_errors={"model-a": {"reason": "server_error"}},
        last_routes={"ficelle/auto-fast": {"status": "ok"}},
        performance_history={"ficelle/auto-fast": {"model_id": "model-a"}},
        quarantine=[{"model_id": "model-b"}],
        actions=["Inspect active model cooldown reasons before clearing or waiting for recovery."],
    )

    assert tuple(payload) == ADMIN_STATUS_TOP_LEVEL_FIELDS
    assert payload == {
        "status": "degraded",
        "reasons": ["active_model_cooldowns"],
        "generated_at": "2026-06-21T10:00:00+00:00",
        "catalog": {"model_count": 2},
        "profiles": {"ficelle/auto-fast": {"status": "ok"}},
        "fusion": {"config": {"enabled": True}},
        "critical_profiles": ("ficelle/auto-fast",),
        "runtime": {"active_cooldowns_count": 1},
        "compression": {"status": "ok"},
        "cooldowns": [{"model_id": "model-a"}],
        "model_cooldowns": [{"model_id": "model-a"}],
        "provider_cooldowns": [],
        "quota_cooldowns": [],
        "quota_probe_results": {"provider-a": {"status": "ok"}},
        "provider_errors": {"provider-a": {"reason": "rate_limit"}},
        "model_errors": {"model-a": {"reason": "server_error"}},
        "last_routes": {"ficelle/auto-fast": {"status": "ok"}},
        "performance_history": {"ficelle/auto-fast": {"model_id": "model-a"}},
        "quarantine": [{"model_id": "model-b"}],
        "actions": ["Inspect active model cooldown reasons before clearing or waiting for recovery."],
    }


def make_admin_status_ports() -> AdminStatusBuildPorts:
    def rank_rows(profile_id, candidates, profile, state, limit=None):
        selected = candidates if limit is None else candidates[:limit]
        return [{"model_id": item["id"], "rank": index + 1} for index, item in enumerate(selected)]

    return AdminStatusBuildPorts(
        model_on_cooldown=lambda item, state: (False, None),
        model_is_quarantined=lambda item, state: False,
        normalized_virtual_profiles=lambda config: {"ficelle/auto-fast": {"mode": "auto", "requirements": {}}},
        redact_runtime_state=lambda state: state,
        candidates_for_profile=lambda profile_id, available, state, profile: available,
        route_competent_candidates=lambda profile_id, candidates, state: candidates,
        candidate_rank_rows=rank_rows,
        verified_capability_row=lambda profile_id, item, state: {"status": "verified", "capability": "text"},
        model_score_explanation=lambda profile_id, item, state: {"score_total": 88.0},
        model_route_competence=lambda profile_id, item, state: "verified",
        failed_profile_evidence_row=lambda profile_id, item, state: {},
        active_model_cooldown_rows=lambda state: [],
        active_provider_cooldown_rows=lambda state: [],
        active_quota_cooldown_rows=lambda state: [],
        quarantine_rows=lambda state: [],
        model_history_row=lambda profile_id, item, state, rank_row: {
            "model_id": item["id"],
            "rank": rank_row.get("rank"),
        },
        model_auto_score=lambda profile_id, item, state: 88.0,
        fallback_attempt_rows=lambda last_route: [],
        compression_observability=lambda config, runtime_state: {"error_count": 0, "last_status": "ok"},
        runtime_state_diagnostics=lambda runtime_state: {"status": "ok"},
        fusion_payload=lambda config, state: {"config": {"enabled": False}},
        now_iso=lambda: "2026-06-21T10:00:00+00:00",
        safe_int=lambda value, default: int(value) if str(value).isdigit() else default,
    )


def admin_status_model() -> dict[str, Any]:
    return {
        "id": "ficelle/openrouter/model-a",
        "upstream_id": "model-a",
        "source": "openrouter",
        "invokable": True,
    }


def test_admin_status_document_assembles_core_sections_from_ports():
    model = admin_status_model()

    payload = build_admin_status_document(
        catalog={"models": [model], "generated_at": "catalog-time", "min_context_length": 128000},
        config={},
        state={
            "last_routes": {"ficelle/auto-fast": {"status": "ok", "attempt_count": "1"}},
            "stats": {"model-a": {}},
            "provider_stats": {"openrouter": {}},
        },
        critical_profile_ids=("ficelle/auto-fast",),
        ports=make_admin_status_ports(),
    )

    assert payload["status"] == "ok"
    assert payload["generated_at"] == "2026-06-21T10:00:00+00:00"
    assert payload["catalog"]["model_count"] == 1
    assert payload["catalog"]["invokable_count"] == 1
    assert payload["catalog"]["available_count"] == 1
    assert payload["profiles"]["ficelle/auto-fast"]["selected_model"] == "ficelle/openrouter/model-a"
    assert payload["runtime"]["stats_records"] == 1
    assert payload["runtime"]["provider_stats_records"] == 1
    assert payload["runtime"]["last_route_count"] == 1
    assert payload["performance_history"]["ficelle/auto-fast"]["models"] == [
        {"model_id": "ficelle/openrouter/model-a", "rank": 1}
    ]
    assert payload["fusion"] == {"config": {"enabled": False}}


def test_admin_status_builder_applies_config_rules_before_document_build():
    calls: list[str] = []

    builder = AdminStatusBuilder(
        ports=make_admin_status_ports(),
        critical_profile_ids=("ficelle/auto-fast",),
        apply_verified_capability_ttl=lambda config: calls.append(f"ttl:{config['marker']}"),
        apply_route_on_capability_reference=lambda config: calls.append(f"reference:{config['marker']}"),
    )

    payload = builder.build(
        catalog={"models": [admin_status_model()], "generated_at": "catalog-time", "min_context_length": 128000},
        config={"marker": "config-1"},
        state={},
    )

    assert calls == ["ttl:config-1", "reference:config-1"]
    assert payload["profiles"]["ficelle/auto-fast"]["selected_model"] == "ficelle/openrouter/model-a"
