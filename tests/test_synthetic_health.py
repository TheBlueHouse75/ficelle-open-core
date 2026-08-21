from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ficelle import build_identity, synthetic_health
from ficelle.probe_lock import file_lock
from ficelle.use_cases.synthetic_health import SyntheticHealthCase, deterministic_failure_scenarios


class FakeRouter:
    ROUTE_LOG_PATH = Path("/unused/routes.jsonl")

    def __init__(self, *, profiles: dict[str, Any] | None = None) -> None:
        self.profiles = profiles or {}
        self.state: dict[str, Any] = {}

    def load_runtime_config(self) -> dict[str, Any]:
        return {"providers": {"future_provider": {"enabled": True, "base_url": "https://example.invalid/v1"}}}

    def load_or_refresh_catalog(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"generated_at": "2026-01-01T00:00:00Z", "models": []}

    def normalized_virtual_profiles(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.profiles

    def provider_access_for(self, source: str, config: dict[str, Any]) -> Any:
        return SimpleNamespace(can_invoke=True, reason="configured")

    def normalized_free_access(self, model: dict[str, Any]) -> dict[str, Any]:
        return model.get("free_access") or {"eligible": True, "mode": "catalog_free"}

    def benchmark_body(self, profile_id: str, config: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        return ({"model": profile_id, "messages": [{"role": "user", "content": "probe"}]}, "exact_text", "ok")

    def model_matches_profile_requirements(self, model: dict[str, Any], profile: dict[str, Any]) -> bool:
        return bool(profile.get("match"))

    def model_route_competence(self, profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> str:
        return "claimed_catalog"

    def catalog_config_structural_fingerprint(self, config: dict[str, Any]) -> str:
        return "config-fingerprint"

    def load_runtime_state(self) -> dict[str, Any]:
        return dict(self.state)

    def model_quarantine(self, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
        quarantines = state.get("quarantine")
        row = quarantines.get(model.get("id")) if isinstance(quarantines, dict) else None
        return row if isinstance(row, dict) else None

    def model_on_cooldown(self, model: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str | None]:
        cooldowns = state.get("cooldowns")
        row = cooldowns.get(model.get("id")) if isinstance(cooldowns, dict) else None
        return (isinstance(row, dict), str(row.get("reason") or "active") if isinstance(row, dict) else None)

    def success_error_payload_failure(self, _payload: dict[str, Any], _model: dict[str, Any] | None) -> None:
        return None


def test_service_base_url_uses_the_configured_ipv6_listener() -> None:
    assert synthetic_health._base_url({"host": "::1", "port": 9123}) == "http://[::1]:9123/v1"
    assert synthetic_health._base_url({"host": "::", "port": 9123}) == "http://[::1]:9123/v1"


def test_deterministic_failure_matrix_uses_shared_classifier_and_policy() -> None:
    statuses: list[str] = []
    for scenario in deterministic_failure_scenarios():
        case = SyntheticHealthCase(
            id=f"failure.{scenario.id}",
            phase=1,
            lane="deterministic",
            target_kind="failure-contract",
            target_id=scenario.id,
            validator="failure-contract",
            timeout_class="fast",
        )
        statuses.append(synthetic_health.execute_deterministic_case(case)["status"])

    assert set(statuses) <= {"pass", "pass_with_fallback"}
    assert "pass_with_fallback" in statuses


def test_deterministic_outcome_preserves_case_identity_and_tags() -> None:
    case = SyntheticHealthCase(
        id="p1.tools.cross-model-replay",
        phase=1,
        lane="deterministic",
        target_kind="tool-contract",
        target_id="cross-model-replay",
        validator="tool-contract",
        timeout_class="fast",
        source="synthetic",
        tags=("tools", "t4", "t7"),
    )

    outcome = synthetic_health.execute_deterministic_case(case)

    assert outcome["target"] == {"kind": "tool-contract", "id": "cross-model-replay"}
    assert outcome["source"] == "synthetic"
    assert outcome["tags"] == ["tools", "t4", "t7"]


def test_full_generated_deterministic_lane_covers_deceptive_stream_and_tool_contracts() -> None:
    router = FakeRouter()
    inventory = synthetic_health.build_inventory(router, router.load_runtime_config(), {"models": []})
    cases = synthetic_health.build_corpus(router, router.load_runtime_config(), {"models": []}, inventory, run_id="run-a")
    deterministic = [case for case in cases if case.lane == "deterministic"]

    results = [synthetic_health.execute_deterministic_case(case) for case in deterministic]

    assert all(result["status"] in {"pass", "pass_with_fallback"} for result in results)
    targets = {case.target_id for case in deterministic}
    assert {"timeout", "connection-exception", "embedded-error", "invalid-json", "empty-message", "truncated-before-content"} <= targets
    assert {"pre-first-byte-retry", "mid-stream-no-retry", "client-disconnect-no-cooldown"} <= targets


def test_stream_assembly_keeps_split_tool_call_fragments() -> None:
    events = [
        {"model": "m", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_", "function": {"name": "fetch_", "arguments": '{"city":"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "1", "function": {"name": "weather", "arguments": 'Madrid"}'}}]}, "finish_reason": "tool_calls"}]},
    ]

    payload, calls = synthetic_health._assemble_stream(events)

    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert calls == [{"id": "call_1", "type": "function", "function": {"name": "fetch_weather", "arguments": '{"city":"Madrid"}'}}]


def test_tool_call_normalization_preserves_exact_replay_values() -> None:
    call_id = "call-" + "x" * 220
    arguments = '{"query":"two  spaces","payload":"' + "y" * 2_100 + '"}'
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "fetch_exact", "arguments": arguments},
                        }
                    ]
                }
            }
        ]
    }

    calls = synthetic_health.normalized_tool_calls(payload)

    assert calls[0]["id"] == call_id
    assert calls[0]["function"]["arguments"] == arguments


def test_stream_assembly_does_not_mix_tool_fragments_between_choices() -> None:
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call-a", "function": {"name": "first", "arguments": "{}"}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
                {
                    "index": 1,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call-b", "function": {"name": "second", "arguments": "{}"}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
            ]
        }
    ]

    payload, calls = synthetic_health._assemble_stream(events)

    assert payload["choices"][0]["message"]["tool_calls"][0]["id"] == "call-a"
    assert payload["choices"][1]["message"]["tool_calls"][0]["id"] == "call-b"
    assert calls[0]["id"] == "call-a"


def test_dynamic_inventory_and_corpus_include_an_unlisted_provider_and_tools() -> None:
    router = FakeRouter(profiles={"ficelle/auto-tools": {"match": True}})
    config = router.load_runtime_config()
    catalog = {
        "models": [
            {
                "id": "future/model-a",
                "upstream_id": "model-a",
                "source": "future_provider",
                "free_access": {"eligible": True, "mode": "catalog_free"},
                "supported_parameters": ["tools", "tool_choice", "parallel_tool_calls"],
            }
        ]
    }

    inventory = synthetic_health.build_inventory(router, config, catalog)
    cases = synthetic_health.build_corpus(router, config, catalog, inventory, run_id="run-a")

    assert [case.phase for case in cases] == sorted(case.phase for case in cases)
    assert any(provider["source"] == "future_provider" and provider["can_invoke"] for provider in inventory["providers"])
    direct = [case for case in cases if case.target_id == "future/model-a"]
    assert any("core-text" in case.tags for case in direct)
    assert {"t1", "t2", "t3", "t4", "t5", "t6"} <= {tag for case in direct for tag in case.tags}
    assert next(case for case in direct if "t6" in case.tags).skip_reason is None


def test_disabled_provider_is_reported_but_never_enters_live_model_scope() -> None:
    router = FakeRouter()
    config = {"providers": {"future_provider": {"enabled": False}}}
    catalog = {
        "models": [
            {
                "id": "future/model-a",
                "upstream_id": "model-a",
                "source": "future_provider",
                "free_access": {"eligible": True, "mode": "catalog_free"},
            }
        ]
    }

    inventory = synthetic_health.build_inventory(router, config, catalog)
    cases = synthetic_health.build_corpus(router, config, catalog, inventory, run_id="run-disabled")

    assert inventory["providers"][0]["access_reason"] == "disabled_by_config"
    assert inventory["providers"][0]["can_invoke"] is False
    assert inventory["models"][0]["can_invoke"] is False
    assert not any(case.target_id == "future/model-a" for case in cases)


def test_runtime_blocked_model_remains_in_denominator_with_explicit_skips() -> None:
    router = FakeRouter(profiles={"ficelle/auto-tools": {"match": True}})
    router.state = {"quarantine": {"future/model-a": {"reason": "billing_or_paid"}}}
    config = router.load_runtime_config()
    catalog = {
        "models": [
            {
                "id": "future/model-a",
                "upstream_id": "model-a",
                "source": "future_provider",
                "free_access": {"eligible": True, "mode": "catalog_free"},
                "supported_parameters": ["tools", "tool_choice"],
            }
        ]
    }

    inventory = synthetic_health.build_inventory(router, config, catalog)
    cases = synthetic_health.build_corpus(router, config, catalog, inventory, run_id="run-blocked")
    direct = [case for case in cases if case.target_id == "future/model-a"]

    assert inventory["models"][0]["runtime_block_reason"] == "quarantine:billing_or_paid"
    assert direct
    assert all(case.skip_reason == "quarantine:billing_or_paid" for case in direct)
    assert any("t6" in case.tags for case in direct)


def test_runtime_block_detected_after_planning_produces_traceable_skip() -> None:
    router = FakeRouter()
    model = {
        "id": "future/model-a",
        "upstream_id": "model-a",
        "source": "future_provider",
    }
    case = SyntheticHealthCase(
        id="p3.direct.future-model-a.core-text",
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="future/model-a",
        source="future_provider",
        model_id="future/model-a",
        upstream_id="model-a",
        validator="exact-text",
        timeout_class="fast",
        tags=("core-text",),
    )
    router.state = {"cooldowns": {"future/model-a": {"reason": "rate_limit"}}}

    reason = synthetic_health.runtime_model_block_reason(router, model, router.load_runtime_state())
    outcome = synthetic_health._skipped_outcome(case, reason or "missing_reason")

    assert reason == "cooldown:rate_limit"
    assert outcome["status"] == "skipped"
    assert outcome["skip_reason"] == "cooldown:rate_limit"
    assert outcome["target"] == {"kind": "direct-model", "id": "future/model-a"}
    assert outcome["source"] == "future_provider"
    assert outcome["tags"] == ["core-text"]


def test_tool_metrics_keep_planned_denominators_before_execution() -> None:
    cases = synthetic_health._tool_cases(
        run_id="run-a",
        phase=3,
        target_kind="direct-model",
        target_id="future/model-a",
        source="future_provider",
        upstream_id="model-a",
        parallel_claimed=True,
    )

    metrics = synthetic_health._tool_metrics(cases, [])

    assert metrics["planned"] == 4
    assert metrics["completed"] == 0
    assert metrics["models"] == {"denominator": 1, "covered": 0, "blocked": 0, "not_fully_covered": 1}
    assert metrics["t6"]["planned"] == 1


def test_tool_metrics_join_deterministic_outcomes_by_case_id() -> None:
    case = SyntheticHealthCase(
        id="p1.tools.cross-model-replay",
        phase=1,
        lane="deterministic",
        target_kind="tool-contract",
        target_id="cross-model-replay",
        validator="tool-contract",
        timeout_class="fast",
        tags=("tools", "t4", "t7"),
    )
    legacy_outcome = {"case_id": case.id, "status": "pass"}

    metrics = synthetic_health._tool_metrics([case], [legacy_outcome])

    assert metrics["completed"] == 1
    assert metrics["t7"] == {
        "planned": 1,
        "completed": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "pass_rate": 1.0,
    }


def test_failed_continuation_rerun_includes_its_tool_choice_dependency() -> None:
    cases = synthetic_health._tool_cases(
        run_id="run-a",
        phase=3,
        target_kind="direct-model",
        target_id="future/model-a",
        source="future_provider",
        upstream_id="model-a",
        parallel_claimed=False,
    )
    continuation = next(case for case in cases if case.validator == "tool-continuation")
    selected = synthetic_health.select_cases_with_dependencies(cases, {continuation.id})

    assert [case.validator for case in selected] == ["tool-choice", "tool-continuation"]


def test_core_post_sweep_runs_even_when_pre_sweep_failed() -> None:
    core = SyntheticHealthCase(
        id="p4.virtual-post.ficelle-auto-fast.ficelle-auto-fast",
        phase=4,
        lane="live",
        target_kind="virtual-profile",
        target_id="ficelle/auto-fast",
        validator="exact-text",
        timeout_class="fast",
    )
    optional = SyntheticHealthCase(
        id="p4.virtual-post.ficelle-auto-audio.ficelle-auto-audio",
        phase=4,
        lane="live",
        target_kind="virtual-profile",
        target_id="ficelle/auto-audio",
        validator="benchmark",
        timeout_class="media",
    )
    outcomes = {
        "p2.virtual-pre.ficelle-auto-fast.ficelle-auto-fast": {"status": "fail"},
        "p2.virtual-pre.ficelle-auto-audio.ficelle-auto-audio": {"status": "fail"},
    }

    assert synthetic_health.post_sweep_skip_reason(core, outcomes) is None
    assert synthetic_health.post_sweep_skip_reason(optional, outcomes) == (
        "pre_sweep_case_not_successful:p2.virtual-pre.ficelle-auto-audio.ficelle-auto-audio"
    )


def test_route_correlation_detects_header_contradictions() -> None:
    case = SyntheticHealthCase(
        id="direct",
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="model-a",
        model_id="model-a",
        upstream_id="upstream-a",
        validator="exact-text",
        timeout_class="fast",
    )
    evidence = synthetic_health.HttpEvidence(
        status_code=200,
        content_type="application/json",
        headers={
            "x-ficelle-request-id": "req-1",
            "x-ficelle-attempts": "2",
            "x-ficelle-fallback": "false",
            "x-ficelle-model": "model-a",
            "x-ficelle-upstream": "upstream-a",
        },
        payload={},
        text="",
        latency_seconds=0.1,
        stream_complete=True,
        stream_error=None,
        transport_error=None,
        tool_calls=[],
    )
    row = {
        "request_id": "req-1",
        "attempt_count": 2,
        "selected_model": "model-a",
        "selected_upstream": "upstream-a",
        "attempts": [{"reason": "server_error"}, {"reason": "ok"}],
    }

    routing, telemetry, policy, fallback = synthetic_health.route_axes(case, evidence, [row])

    assert fallback is True
    assert routing["status"] == "fail"
    assert telemetry["status"] == "fail"
    assert policy["status"] == "fail"


def test_route_policy_detects_missing_retryable_fallback() -> None:
    case = SyntheticHealthCase(
        id="virtual",
        phase=2,
        lane="live",
        target_kind="virtual-profile",
        target_id="ficelle/auto-fast",
        validator="exact-text",
        timeout_class="fast",
    )
    evidence = synthetic_health.HttpEvidence(
        status_code=502,
        content_type="application/json",
        headers={
            "x-ficelle-request-id": "req-missing",
            "x-ficelle-attempts": "1",
            "x-ficelle-fallback": "false",
        },
        payload={"error": {"message": "failed"}},
        text="",
        latency_seconds=0.1,
        stream_complete=True,
        stream_error=None,
        transport_error=None,
        tool_calls=[],
    )
    row = {
        "request_id": "req-missing",
        "candidate_count": 2,
        "attempt_count": 1,
        "final_status": 502,
        "attempts": [{"reason": "server_error"}],
    }

    routing, telemetry, policy, fallback = synthetic_health.route_axes(case, evidence, [row])

    assert fallback is False
    assert routing["status"] == "pass"
    assert telemetry["status"] == "pass"
    assert policy["status"] == "fail"
    assert "missing fallback" in policy["diagnostic"]


def test_state_policy_rejects_cooldown_for_terminal_caller_error() -> None:
    route = {
        "attempts": [
            {
                "source": "future_provider",
                "model": "future/model-a",
                "reason": "bad_upstream_request",
            }
        ]
    }
    before = {"stats": {}, "cooldowns": {}}
    valid_after = {"stats": {"future/model-a": {"failures": 1}}, "cooldowns": {}}
    invalid_after = {
        "stats": {"future/model-a": {"failures": 1}},
        "cooldowns": {"future/model-a": {"reason": "bad_upstream_request"}},
    }

    valid = synthetic_health.state_policy_verdict(route, before, valid_after)
    invalid = synthetic_health.state_policy_verdict(route, before, invalid_after)

    assert valid["status"] == "pass"
    assert invalid["status"] == "fail"
    assert "unexpected state change in cooldowns" in invalid["diagnostic"]


def test_state_policy_flags_model_cooldown_added_to_provider_rate_limit() -> None:
    route = {
        "attempts": [
            {
                "source": "future_provider",
                "model": "future/model-a",
                "reason": "rate_limited",
            }
        ]
    }
    before = {"stats": {}, "model_errors": {}, "provider_errors": {}, "provider_cooldowns": {}, "cooldowns": {}}
    valid_after = {
        "stats": {"future/model-a": {"failures": 1}},
        "model_errors": {"future/model-a": {"reason": "rate_limited"}},
        "provider_errors": {"future_provider": {"reason": "rate_limited"}},
        "provider_cooldowns": {"future_provider": {"reason": "rate_limited"}},
        "cooldowns": {},
    }
    redundant_model_cooldown = {
        **valid_after,
        "cooldowns": {"future/model-a": {"reason": "rate_limited"}},
    }

    valid = synthetic_health.state_policy_verdict(route, before, valid_after)
    invalid = synthetic_health.state_policy_verdict(route, before, redundant_model_cooldown)

    assert valid["status"] == "pass"
    assert invalid["status"] == "fail"
    assert "unexpected state change in cooldowns" in invalid["diagnostic"]


def test_route_correlation_records_response_model_alias_and_drift() -> None:
    case = SyntheticHealthCase(
        id="direct",
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="provider/model-a",
        model_id="provider/model-a",
        upstream_id="model-a",
        validator="exact-text",
        timeout_class="fast",
    )
    base_headers = {
        "x-ficelle-request-id": "req-model",
        "x-ficelle-attempts": "1",
        "x-ficelle-fallback": "false",
        "x-ficelle-model": "provider/model-a",
        "x-ficelle-upstream": "model-a",
    }
    row = {
        "request_id": "req-model",
        "candidate_count": 1,
        "attempt_count": 1,
        "selected_model": "provider/model-a",
        "selected_upstream": "model-a",
        "final_status": 200,
        "attempts": [{"reason": "ok"}],
    }

    def evidence(response_model: str) -> synthetic_health.HttpEvidence:
        return synthetic_health.HttpEvidence(
            status_code=200,
            content_type="application/json",
            headers=base_headers,
            payload={"model": response_model, "choices": []},
            text="",
            latency_seconds=0.1,
            stream_complete=True,
            stream_error=None,
            transport_error=None,
            tool_calls=[],
        )

    alias, _, _, _ = synthetic_health.route_axes(case, evidence("model-a"), [row])
    drift, _, _, _ = synthetic_health.route_axes(case, evidence("unexpected-model"), [row])

    assert alias["status"] == "pass"
    assert alias["response_model_identity"] == "catalog_or_selected_alias"
    assert drift["status"] == "fail"
    assert drift["response_model_identity"] == "unexplained_drift"


def test_route_correlation_accepts_catalog_declared_provider_router_alias() -> None:
    case = SyntheticHealthCase(
        id="virtual-router",
        phase=2,
        lane="live",
        target_kind="virtual-profile",
        target_id="ficelle/auto-vision",
        validator="benchmark",
        timeout_class="media",
    )
    evidence = synthetic_health.HttpEvidence(
        status_code=200,
        content_type="application/json",
        headers={
            "x-ficelle-request-id": "req-router",
            "x-ficelle-attempts": "1",
            "x-ficelle-fallback": "false",
            "x-ficelle-source": "future_provider",
            "x-ficelle-model": "future/router-model",
            "x-ficelle-upstream": "router-model",
        },
        payload={"model": "concrete/model-b", "choices": []},
        text="",
        latency_seconds=0.1,
        stream_complete=True,
        stream_error=None,
        transport_error=None,
        tool_calls=[],
    )
    row = {
        "request_id": "req-router",
        "candidate_count": 1,
        "attempt_count": 1,
        "selected_source": "future_provider",
        "selected_model": "future/router-model",
        "selected_upstream": "router-model",
        "final_status": 200,
        "attempts": [{"reason": "ok"}],
    }
    catalog = {
        "models": [
            {
                "id": "future/router-model",
                "upstream_id": "router-model",
                "source": "future_provider",
                "tokenizer": "Router",
            }
        ]
    }

    routing, telemetry, policy, fallback = synthetic_health.route_axes(case, evidence, [row], catalog)

    assert routing["status"] == "pass"
    assert routing["response_model_identity"] == "provider_router_alias"
    assert telemetry["status"] == "pass"
    assert policy["status"] == "pass"
    assert fallback is False


def test_append_journal_redacts_nested_secrets(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    synthetic_health.append_jsonl(path, {"nested": {"authorization": "Bearer secret-token-value"}})

    content = path.read_text(encoding="utf-8")
    assert "secret-token-value" not in content
    assert "[redacted]" in content


def test_file_lock_is_process_safe_within_nested_handles(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with file_lock(path, blocking=False) as first:
        with file_lock(path, blocking=False) as second:
            assert first is True
            assert second is False


def test_router_benchmark_respects_cross_process_probe_lock(monkeypatch: Any, tmp_path: Path) -> None:
    from ficelle import router

    lock_path = tmp_path / "provider-probes.lock"
    monkeypatch.setattr(router, "PROVIDER_PROBE_LOCK_PATH", lock_path)
    monkeypatch.setattr(router, "_benchmark_profile_locked", lambda *args, **kwargs: {"status": "pass"})

    with file_lock(lock_path, blocking=False) as acquired:
        assert acquired is True
        result = router.benchmark_profile("ficelle/auto-fast", {})

    assert result["status"] == "skipped"
    assert result["reason"] == "provider_probes_already_running"


def test_router_quota_probe_respects_cross_process_probe_lock(monkeypatch: Any, tmp_path: Path) -> None:
    from ficelle import router

    lock_path = tmp_path / "provider-probes.lock"
    monkeypatch.setattr(router, "PROVIDER_PROBE_LOCK_PATH", lock_path)

    with file_lock(lock_path, blocking=False) as acquired:
        assert acquired is True
        result = router.run_due_quota_probes({}, {"models": []})

    assert result == {
        "status": "skipped",
        "reason": "provider_probes_already_running",
        "probed_count": 0,
        "results": [],
    }


def test_weekly_schedule_is_disabled_until_installed_and_uses_same_cli(tmp_path: Path) -> None:
    plist = tmp_path / "com.ficelle.synthetic-health.plist"
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    assert synthetic_health.schedule_status(plist_path=plist)["enabled"] is False
    status = synthetic_health.install_weekly_schedule(
        tmp_path / "home",
        weekday=2,
        hour=3,
        minute=15,
        plist_path=plist,
        run_command=run,
        platform="darwin",
    )

    assert status["enabled"] is True
    assert status["schedule"] == {"Weekday": 2, "Hour": 3, "Minute": 15}
    assert ["synthetic-health", "run", "--depth", "deep", "--json"] == status["program_arguments"][3:]
    assert calls[-1][:2] == ["launchctl", "bootstrap"]


def test_service_down_run_writes_durable_incomplete_artifacts(monkeypatch: Any, tmp_path: Path) -> None:
    router = FakeRouter()
    router.ROUTE_LOG_PATH = tmp_path / "routes.jsonl"
    monkeypatch.setattr(synthetic_health, "_load_router", lambda: router)
    monkeypatch.setattr(
        synthetic_health,
        "_service_preflight",
        lambda base_url, **kwargs: (False, {"transport_error": "connection refused"}),
    )

    payload, exit_code = synthetic_health.run_synthetic_health(tmp_path)

    assert exit_code == 2
    assert payload["status"] == "incomplete"
    artifacts = payload["artifacts"]
    for name in ("run", "cases", "catalog_snapshot", "state_before", "state_after", "summary_json", "summary_markdown"):
        assert Path(artifacts[name]).exists()
    rows = synthetic_health.read_case_journal(Path(artifacts["cases"]))
    assert rows[-1]["case_id"] == "p0.preflight.loopback-service"
    assert rows[-1]["status"] == "harness_error"
    run_metadata = json.loads(Path(artifacts["run"]).read_text(encoding="utf-8"))
    assert run_metadata["max_duration_seconds"] == synthetic_health.DEFAULT_MAX_DURATION_SECONDS
    assert run_metadata["interruption_reason"] == "service_unreachable"


def test_invalid_retention_is_rejected_before_live_cases(monkeypatch: Any, tmp_path: Path) -> None:
    router = FakeRouter()
    router.load_runtime_config = lambda: {  # type: ignore[method-assign]
        "synthetic_health": {"retention_runs": 0},
        "providers": {},
    }
    router.ROUTE_LOG_PATH = tmp_path / "routes.jsonl"
    monkeypatch.setattr(synthetic_health, "_load_router", lambda: router)
    preflight_called = False

    def preflight(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        nonlocal preflight_called
        preflight_called = True
        return True, {}

    monkeypatch.setattr(synthetic_health, "_service_preflight", preflight)

    with pytest.raises(synthetic_health.SyntheticHealthError, match="retention_runs must be at least 1"):
        synthetic_health.run_synthetic_health(tmp_path)

    assert preflight_called is False
    assert not synthetic_health.health_runs_root(tmp_path).exists()


def test_service_down_run_can_resume_after_service_recovers(monkeypatch: Any, tmp_path: Path) -> None:
    router = FakeRouter()
    router.load_runtime_config = lambda: {}  # type: ignore[method-assign]
    router.ROUTE_LOG_PATH = tmp_path / "routes.jsonl"
    monkeypatch.setattr(synthetic_health, "_load_router", lambda: router)
    monkeypatch.setattr(
        synthetic_health,
        "_service_preflight",
        lambda base_url, **kwargs: (False, {"transport_error": "connection refused"}),
    )
    first, _ = synthetic_health.run_synthetic_health(tmp_path)
    monkeypatch.setattr(
        synthetic_health,
        "_service_preflight",
        lambda base_url, **kwargs: (True, {"http_status": 200}),
    )

    resumed, exit_code = synthetic_health.resume_synthetic_health(tmp_path, first["run_id"])

    assert exit_code == 0
    assert resumed["status"] == "pass"
    assert resumed["cases"]["pending"] == 0


def test_retention_keeps_pinned_baseline(tmp_path: Path) -> None:
    root = tmp_path / "health-runs"
    for run_id in ("old", "middle", "new"):
        path = root / run_id
        path.mkdir(parents=True)
        (path / "run.json").write_text("{}", encoding="utf-8")
    timestamps = {"old": 1.0, "middle": 2.0, "new": 3.0}
    for run_id, timestamp in timestamps.items():
        path = root / run_id
        path.touch(exist_ok=True)
        import os

        os.utime(path, (timestamp, timestamp))

    removed = synthetic_health.prune_health_runs(tmp_path, keep_runs=1, pinned_run_ids={"old"})

    assert removed == ["middle"]
    assert (root / "old").exists()
    assert (root / "new").exists()


def test_catalog_identity_ignores_runtime_blocks_and_candidate_availability() -> None:
    base = {
        "providers": [{"source": "future_provider", "can_invoke": True}],
        "models": [{"id": "future/model-a", "strict_zero_eligible": True, "runtime_block_reason": None}],
        "virtual_profiles": ["ficelle/auto-fast"],
        "virtual_profile_definitions": {"ficelle/auto-fast": {"mode": "auto"}},
        "virtual_profile_candidates": {"ficelle/auto-fast": ["future/model-a"]},
    }
    changed_runtime = {
        **base,
        "models": [{"id": "future/model-a", "strict_zero_eligible": True, "runtime_block_reason": "cooldown:timeout"}],
        "virtual_profile_candidates": {"ficelle/auto-fast": []},
    }

    assert synthetic_health.catalog_identity(base) == synthetic_health.catalog_identity(changed_runtime)


def test_summary_surfaces_enabled_provider_access_failure(tmp_path: Path) -> None:
    paths = synthetic_health.RunPaths.for_run(tmp_path, "run-provider-gap")
    inventory = {
        "providers": [
            {
                "source": "future_provider",
                "enabled": True,
                "can_invoke": False,
                "access_reason": "credential_missing",
            }
        ],
        "models": [],
        "virtual_profiles": [],
    }

    summary = synthetic_health.build_summary(
        paths,
        {
            "run_id": "run-provider-gap",
            "corpus_version": "1",
            "corpus_hash": "hash",
            "config_structural_fingerprint": "config",
        },
        inventory,
        [],
        [],
    )

    assert summary["status"] == "degraded"
    assert summary["findings"][0]["case_id"] == "provider-access:future_provider"
    assert summary["coverage"]["providers"]["skipped"][0]["reason"] == "credential_missing"


def test_summary_surfaces_invokable_provider_catalog_failure(tmp_path: Path) -> None:
    paths = synthetic_health.RunPaths.for_run(tmp_path, "run-catalog-gap")
    inventory = {
        "providers": [
            {
                "source": "future_provider",
                "enabled": True,
                "can_invoke": True,
                "catalog_error": "ReadTimeout: catalog unavailable",
            }
        ],
        "models": [],
        "virtual_profiles": [],
    }

    summary = synthetic_health.build_summary(
        paths,
        {
            "run_id": "run-catalog-gap",
            "corpus_version": "1",
            "corpus_hash": "hash",
            "config_structural_fingerprint": "config",
        },
        inventory,
        [],
        [],
    )

    assert summary["status"] == "degraded"
    assert summary["findings"][0]["case_id"] == "provider-catalog:future_provider"
    assert summary["coverage"]["providers"]["catalog_failures"] == [
        {"source": "future_provider", "error": "ReadTimeout: catalog unavailable"}
    ]


def test_surface_diff_reports_new_provider_model_and_capability() -> None:
    previous = {
        "providers": ["provider-a"],
        "invokable_providers": ["provider-a"],
        "eligible_models": ["provider-a/model-a"],
        "virtual_profiles": ["ficelle/auto-fast"],
        "model_capabilities": {"provider-a/model-a": ["streaming"]},
    }
    current = {
        "providers": ["provider-a", "provider-b"],
        "invokable_providers": ["provider-a", "provider-b"],
        "eligible_models": ["provider-a/model-a", "provider-b/model-b"],
        "virtual_profiles": ["ficelle/auto-fast"],
        "model_capabilities": {"provider-a/model-a": ["streaming", "tools"]},
    }

    changes = synthetic_health._surface_changes(current, previous)

    assert changes["providers"]["added"] == ["provider-b"]
    assert changes["eligible_models"]["added"] == ["provider-b/model-b"]
    assert changes["capability_claims"] == [
        {"model_id": "provider-a/model-a", "added": ["tools"], "removed": []}
    ]


def test_safety_axis_rejects_non_eligible_delivered_model() -> None:
    router = FakeRouter()
    evidence = synthetic_health.HttpEvidence(
        status_code=200,
        content_type="application/json",
        headers={
            "x-ficelle-source": "future_provider",
            "x-ficelle-model": "future/paid-model",
            "x-ficelle-upstream": "paid-model",
        },
        payload={"choices": []},
        text="",
        latency_seconds=0.1,
        stream_complete=True,
        stream_error=None,
        transport_error=None,
        tool_calls=[],
    )
    catalog = {
        "models": [
            {
                "id": "future/paid-model",
                "upstream_id": "paid-model",
                "source": "future_provider",
                "free_access": {"eligible": False, "mode": "paid"},
            }
        ]
    }

    verdict = synthetic_health.safety_axis(router, evidence, catalog)

    assert verdict["status"] == "fail"
    assert "not strict-zero eligible" in verdict["diagnostic"]


def test_safety_axis_rejects_non_eligible_failed_attempt() -> None:
    router = FakeRouter()
    evidence = synthetic_health.HttpEvidence(
        status_code=402,
        content_type="application/json",
        headers={},
        payload={"error": {"message": "payment required"}},
        text="",
        latency_seconds=0.1,
        stream_complete=True,
        stream_error=None,
        transport_error=None,
        tool_calls=[],
    )
    catalog = {
        "models": [
            {
                "id": "future/paid-model",
                "upstream_id": "paid-model",
                "source": "future_provider",
                "free_access": {"eligible": False, "mode": "paid"},
            }
        ]
    }
    route = {
        "attempts": [
            {
                "source": "future_provider",
                "model": "future/paid-model",
                "upstream": "paid-model",
                "reason": "billing_or_paid",
            }
        ]
    }

    verdict = synthetic_health.safety_axis(router, evidence, catalog, route)

    assert verdict["status"] == "fail"
    assert "route attempt" in verdict["diagnostic"]


# --- Lot 1 (synthetic-health remediation): eligibility scopes, refusal reading, build id ----


class _EligibilityRouter:
    def __init__(self, *, quarantine=None, cooldown=(False, None), quota_keys=()):
        self._quarantine = quarantine
        self._cooldown = cooldown
        self._quota_keys = set(quota_keys)

    def model_quarantine(self, _model, _state):
        return self._quarantine

    def model_on_cooldown(self, _model, _state):
        return self._cooldown

    def quota_cooldown_matches_model(self, key, _model):
        return key in self._quota_keys

    def quota_cooldown_scope_from_key(self, key):
        return str(key).split(":", 1)[0]


def _eligible_model(**overrides):
    model = {
        "id": "ficelle/alpha/model-a",
        "source": "alpha",
        "upstream_id": "model-a",
        "invokable": True,
    }
    model.update(overrides)
    return model


def test_execution_eligibility_covers_every_scope():
    generation = "identity-1"

    def eligibility(router, model, state=None, provider_cfg=None):
        return synthetic_health.execution_eligibility(
            router, model, state or {}, provider_cfg, catalog_generation=generation
        )

    missing = eligibility(_EligibilityRouter(), None)
    assert (missing["eligible"], missing["scope"], missing["reason"]) == (
        False, "catalog", "catalog_model_missing_during_run"
    )

    disabled = eligibility(_EligibilityRouter(), _eligible_model(), None, {"enabled": False})
    assert (disabled["eligible"], disabled["scope"], disabled["reason"]) == (
        False, "provider", "provider_disabled"
    )

    credential = eligibility(_EligibilityRouter(), _eligible_model(invokable=False, auth_reason="missing credentials"))
    assert (credential["eligible"], credential["scope"], credential["reason"]) == (
        False, "credential", "missing credentials"
    )

    quarantined = eligibility(_EligibilityRouter(quarantine={"reason": "model_not_found_guard"}), _eligible_model())
    assert (quarantined["eligible"], quarantined["scope"], quarantined["reason"]) == (
        False, "model", "quarantine:model_not_found_guard"
    )

    provider_cooled = eligibility(_EligibilityRouter(cooldown=(True, "provider:rate_limited")), _eligible_model())
    assert (provider_cooled["eligible"], provider_cooled["scope"]) == (False, "provider")
    assert provider_cooled["reason"] == "cooldown:provider:rate_limited"

    quota_state = {"quota_cooldowns": {"account:alpha::acct-1": {"until": 4102444800.0}}}
    quota_cooled = synthetic_health.execution_eligibility(
        _EligibilityRouter(cooldown=(True, "quota:quota_exhausted"), quota_keys={"account:alpha::acct-1"}),
        _eligible_model(),
        quota_state,
        None,
        catalog_generation=generation,
    )
    assert (quota_cooled["eligible"], quota_cooled["scope"]) == (False, "quota_pool")

    shared_state = {"quota_cooldowns": {"shared_account:acct-9": {"until": 4102444800.0}}}
    shared = synthetic_health.execution_eligibility(
        _EligibilityRouter(cooldown=(True, "quota:quota_exhausted"), quota_keys={"shared_account:acct-9"}),
        _eligible_model(),
        shared_state,
        None,
        catalog_generation=generation,
    )
    assert (shared["eligible"], shared["scope"]) == (False, "shared_account")

    model_cooled = eligibility(_EligibilityRouter(cooldown=(True, "timeout")), _eligible_model())
    assert (model_cooled["eligible"], model_cooled["scope"], model_cooled["reason"]) == (
        False, "model", "cooldown:timeout"
    )

    clear = eligibility(_EligibilityRouter(), _eligible_model())
    assert (clear["eligible"], clear["scope"], clear["reason"]) == (True, "none", None)
    assert clear["catalog_generation"] == generation
    assert clear["state_observed_at"]
    assert {record["scope"] for record in (missing, disabled, credential, quarantined, provider_cooled, quota_cooled, shared, model_cooled, clear)} <= set(synthetic_health.ELIGIBILITY_SCOPES)


def test_skipped_outcome_marks_catalog_scope_as_surface_change():
    case = SyntheticHealthCase(
        id="p3.model.alpha-model-a.text",
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="ficelle/alpha/model-a",
        source="alpha",
        model_id="ficelle/alpha/model-a",
        upstream_id="model-a",
        validator="exact-text",
        timeout_class="fast",
        tags=("t1",),
    )
    eligibility = {
        "eligible": False,
        "scope": "catalog",
        "reason": "catalog_model_missing_during_run",
        "catalog_generation": "identity-2",
        "state_observed_at": "2026-08-09T10:00:00+00:00",
    }

    outcome = synthetic_health._skipped_outcome(case, str(eligibility["reason"]), eligibility=eligibility)

    assert outcome["status"] == "skipped"
    assert outcome["surface_changed"] == "catalog_model_missing_during_run"
    assert outcome["eligibility"]["scope"] == "catalog"


def _refusal_evidence(status_code: int, headers: dict[str, str], payload: dict) -> synthetic_health.HttpEvidence:
    return synthetic_health.HttpEvidence(
        status_code=status_code,
        content_type="application/json",
        headers=headers,
        payload=payload,
        text="",
        latency_seconds=0.01,
        stream_complete=True,
        stream_error=None,
        transport_error=None,
        tool_calls=[],
    )


def test_router_refusals_bypass_generic_http_classification():
    refusal = _refusal_evidence(
        503,
        {"X-Ficelle-Request-Id": "req-1"},
        {"error": {"type": "no_available_model", "message": "no invokable free model", "excluded": {"cooldown": 2}}},
    )
    assert synthetic_health._router_refusal_error_type(refusal) == "no_available_model"

    # No Ficelle request-id header: not a router refusal, keep provider-shaped classification.
    provider_error = _refusal_evidence(
        503, {}, {"error": {"type": "no_available_model", "message": "provider-side message"}}
    )
    assert synthetic_health._router_refusal_error_type(provider_error) is None

    # invalid_request_error is also the all-upstreams-rejected shape: never a selection refusal.
    upstream_invalid = _refusal_evidence(
        400,
        {"X-Ficelle-Request-Id": "req-2"},
        {"error": {"type": "invalid_request_error", "message": "upstream rejected this request as invalid"}},
    )
    assert synthetic_health._router_refusal_error_type(upstream_invalid) is None


def test_package_build_identity_is_complete_for_this_editable_checkout():
    identity = synthetic_health.package_build_identity()

    assert set(identity) == {
        "ficelle_version",
        "package_dir",
        "install_mode",
        "git_commit",
        "git_dirty",
        "runtime_content_hash",
        "complete",
    }
    assert identity["ficelle_version"]
    assert identity["install_mode"] == "editable"
    assert identity["git_commit"]
    assert identity["git_dirty"] in (True, False)
    assert identity["runtime_content_hash"]
    assert identity["complete"] is True


def test_package_build_identity_is_process_cached_and_returns_a_copy(monkeypatch: Any) -> None:
    collected = {
        "ficelle_version": "test",
        "package_dir": "/test/ficelle",
        "runtime_content_hash": "a" * 64,
        "complete": True,
    }
    calls = 0

    def collect() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return collected

    monkeypatch.setattr(build_identity, "_CACHED_IDENTITY", None)
    monkeypatch.setattr(build_identity, "_collect_identity", collect)

    first = build_identity.package_build_identity()
    first["runtime_content_hash"] = "mutated"
    second = build_identity.package_build_identity()

    assert calls == 1
    assert second["runtime_content_hash"] == "a" * 64


@pytest.mark.parametrize(
    "git_error",
    [FileNotFoundError("git unavailable"), subprocess.TimeoutExpired(cmd="git", timeout=10)],
)
def test_package_build_identity_degrades_when_git_cannot_answer(
    monkeypatch: Any,
    git_error: Exception,
) -> None:
    monkeypatch.setattr(
        build_identity.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(git_error),
    )

    identity = build_identity._collect_identity()

    assert identity["git_commit"] is None
    assert identity["git_dirty"] is None
    assert identity["runtime_content_hash"]
    assert identity["complete"] is False


def test_package_build_identity_degrades_when_runtime_hashing_fails(monkeypatch: Any) -> None:
    git_results = iter([SimpleNamespace(stdout="deadbeef\n"), SimpleNamespace(stdout="")])
    monkeypatch.setattr(build_identity.subprocess, "run", lambda *_args, **_kwargs: next(git_results))
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(OSError("hashing unavailable")),
    )

    identity = build_identity._collect_identity()

    assert identity["runtime_content_hash"] is None
    assert identity["complete"] is False


def test_catalog_transition_event_counts_added_removed_and_stale_rows_per_provider():
    previous_models = {
        "ficelle/alpha/kept": {"id": "ficelle/alpha/kept", "source": "alpha"},
        "ficelle/alpha/removed": {"id": "ficelle/alpha/removed", "source": "alpha"},
        "ficelle/beta/gone": {"id": "ficelle/beta/gone", "source": "beta"},
    }
    current_models = {
        "ficelle/alpha/kept": {"id": "ficelle/alpha/kept", "source": "alpha"},
        "ficelle/alpha/arrived": {"id": "ficelle/alpha/arrived", "source": "alpha"},
        "ficelle/gamma/preserved": {
            "id": "ficelle/gamma/preserved",
            "source": "gamma",
            "catalog_stale": True,
            "stale_since": "2026-08-09T09:00:00+00:00",
        },
    }

    event = synthetic_health._catalog_transition_event(
        previous_identity="identity-1",
        current_identity="identity-2",
        previous_models=previous_models,
        current_models=current_models,
        refresh={"last_success_at": "2026-08-09T10:00:00+00:00"},
    )

    assert event["event"] == "catalog_transition"
    assert event["previous_identity"] == "identity-1"
    assert event["current_identity"] == "identity-2"
    assert event["previous_model_count"] == 3
    assert event["current_model_count"] == 3
    # One transition row carries the per-provider added/removed/stale counts (L1-R2).
    assert event["providers"] == {
        "alpha": {"added": 1, "removed": 1, "stale": 0},
        "beta": {"added": 0, "removed": 1, "stale": 0},
        "gamma": {"added": 1, "removed": 0, "stale": 1},
    }
    assert event["refresh"] == {"last_success_at": "2026-08-09T10:00:00+00:00"}


def test_synthetic_case_correlation_id_fits_the_router_gate():
    # A conforming case id travels as-is.
    assert synthetic_health.synthetic_case_correlation_id("p3.tools.alpha-model.choice") == "p3.tools.alpha-model.choice"
    # Over-long or non-conforming ids become a stable hash that still fits the gate.
    long_id = "p3." + "x" * 120
    hashed = synthetic_health.synthetic_case_correlation_id(long_id)
    assert hashed == synthetic_health.synthetic_case_correlation_id(long_id)
    assert len(hashed) <= 80
    import re as _re

    assert _re.fullmatch(r"[A-Za-z0-9._-]+", hashed)


def test_late_route_correlation_waits_only_for_the_bounded_router_deadline():
    # 300 s router deadline, 45 s already spent client-side: wait the remainder plus margin.
    assert synthetic_health.late_route_correlation_wait_seconds({"request_deadline_seconds": 300}, 45.0) == 260.0
    # A router deadline shorter than the client timeout leaves only the margin.
    assert synthetic_health.late_route_correlation_wait_seconds({"request_deadline_seconds": 30}, 45.0) == 5.0
    # Unconfigured: the 300 s default applies.
    assert synthetic_health.late_route_correlation_wait_seconds({}, 100.0) == 205.0


def test_find_route_rows_correlates_by_synthetic_case_id(tmp_path):
    route_log = tmp_path / "routes.jsonl"
    route_log.write_text(
        "\n".join(
            [
                json.dumps({"request_id": "req-1", "final_reason": "ok"}),
                json.dumps({"request_id": "req-2", "synthetic_case_id": "p3.case.late", "final_reason": "ok"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = synthetic_health.find_route_rows(route_log, "p3.case.late", field="synthetic_case_id")

    assert len(rows) == 1
    assert rows[0]["request_id"] == "req-2"


def test_fallback_failures_report_disjoint_categories_never_one_bucket():
    def outcome(case_id: str, status: str, policy: dict) -> dict:
        return {"case_id": case_id, "status": status, "axes": {"policy": policy}}

    outcomes = [
        outcome("p3.ok", "pass_with_fallback", {"status": "pass"}),
        outcome("p3.scope", "fail", {"status": "fail", "fallback_category": "state_scope_mismatch"}),
        outcome("p3.corr-1", "fail", {"status": "fail", "fallback_category": "correlation_missing"}),
        outcome("p3.corr-2", "fail", {"status": "fail", "fallback_category": "correlation_missing"}),
        outcome("p3.bad-retry", "fail", {"status": "fail", "fallback_category": "illegitimate_decision"}),
        outcome("p3.legacy", "fail", {"status": "fail"}),  # no category: stays visible as unknown
    ]

    metrics = synthetic_health._fallback_metrics({"pass_with_fallback": 1}, outcomes)

    assert metrics["legitimate"] == 1
    assert metrics["state_scope_mismatch"] == 1
    assert metrics["correlation_missing"] == 2
    assert metrics["illegitimate_decision"] == 1
    assert metrics["not_applicable_client_timeout"] == 0
    assert metrics["unknown"] == 1
    # The backward-comparison total equals the sum of the disjoint categories.
    assert metrics["incorrect_or_missing"] == 5
    assert metrics["failure_case_ids"]["correlation_missing"] == ["p3.corr-1", "p3.corr-2"]


def test_missing_request_id_is_correlation_missing_never_an_illegitimate_decision():
    case = SyntheticHealthCase(
        id="p3.model.alpha.text",
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="ficelle/alpha/m",
        validator="exact-text",
        timeout_class="fast",
        source="alpha",
    )
    evidence = synthetic_health.HttpEvidence(
        status_code=None,
        content_type="",
        headers={},
        payload=None,
        text="",
        latency_seconds=45.0,
        stream_complete=False,
        stream_error=None,
        transport_error="ReadTimeout: read timed out (45.0s)",
        tool_calls=[],
    )

    _routing, _telemetry, policy, _fallback = synthetic_health.route_axes(case, evidence, [])

    assert policy["status"] == "fail"
    assert policy["fallback_category"] == "correlation_missing"


# --- Lot 3 (synthetic-health remediation): staged verdicts, dialects, budgets ---------------


def _tool_case(case_id: str, validator: str = "tool-choice", expected: dict | None = None, tags: tuple = ("tools", "t1", "t2")) -> SyntheticHealthCase:
    return SyntheticHealthCase(
        id=case_id,
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="ficelle/alpha/m",
        source="alpha",
        model_id="ficelle/alpha/m",
        upstream_id="m",
        validator=validator,
        timeout_class="tools",
        expected=expected or {"tool_name": "fetch_route_weather", "arguments": {"city": "Madrid"}},
        tags=tags,
    )


def _tool_evidence(*, text: str = "", tool_calls: list | None = None, payload: dict | None = None) -> synthetic_health.HttpEvidence:
    return synthetic_health.HttpEvidence(
        status_code=200,
        content_type="application/json",
        headers={"x-ficelle-request-id": "req"},
        payload=payload if payload is not None else {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
        text=text,
        latency_seconds=0.1,
        stream_complete=True,
        stream_error=None,
        transport_error=None,
        tool_calls=tool_calls or [],
    )


def _call(name: str, arguments: str) -> dict:
    return {"id": "call-1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_continuation_validator_splits_protocol_grounding_and_exact_instruction():
    case = _tool_case(
        "p3.tools.m.continuation",
        validator="tool-continuation",
        expected={"marker": "MARK-42", "grounding_facts": ["24", "wind"]},
        tags=("tools", "t3", "t4"),
    )
    router = FakeRouter()
    catalog = {"models": []}

    exact = synthetic_health.semantic_verdict(router, case, _tool_evidence(text="MARK-42"), catalog)
    assert exact["status"] == "pass"
    assert exact["continuation"] == {"protocol_complete": True, "grounded": True, "instruction_exact": True}

    grounded_prose = synthetic_health.semantic_verdict(
        router, case, _tool_evidence(text="It is 24C with wind alerts on the route."), catalog
    )
    assert grounded_prose["status"] == "pass", "grounded prose passes the lifecycle (L3-R2)"
    assert grounded_prose["continuation"]["instruction_exact"] is False

    hallucinated = synthetic_health.semantic_verdict(
        router, case, _tool_evidence(text="It will be sunny, probably 30 degrees."), catalog
    )
    assert hallucinated["status"] == "fail"
    assert hallucinated["continuation"]["grounded"] is False

    repeated_call = synthetic_health.semantic_verdict(
        router, case, _tool_evidence(text="", tool_calls=[_call("fetch_route_weather", "{}")]), catalog
    )
    assert repeated_call["status"] == "fail"
    assert repeated_call["continuation"]["protocol_complete"] is False

    empty = synthetic_health.semantic_verdict(router, case, _tool_evidence(text="   "), catalog)
    assert empty["status"] == "fail"


def test_tool_choice_records_separate_selection_and_argument_assertions():
    case = _tool_case("p3.tools.m.choice")
    router = FakeRouter()
    catalog = {"models": []}

    wrong_tool = synthetic_health.semantic_verdict(
        router, case, _tool_evidence(tool_calls=[_call("search_weather_guides", "{}")]), catalog
    )
    assert wrong_tool["status"] == "fail"
    assert wrong_tool["tool_assertions"] == {"selection": False, "arguments": None}

    wrong_arguments = synthetic_health.semantic_verdict(
        router, case, _tool_evidence(tool_calls=[_call("fetch_route_weather", '{"city":"Paris"}')]), catalog
    )
    assert wrong_arguments["status"] == "fail"
    assert wrong_arguments["tool_assertions"] == {"selection": True, "arguments": False}

    correct = synthetic_health.semantic_verdict(
        router, case, _tool_evidence(tool_calls=[_call("fetch_route_weather", '{"city":"Madrid"}')]), catalog
    )
    assert correct["status"] == "pass"
    assert correct["tool_assertions"] == {"selection": True, "arguments": True}


def test_length_exhausted_tool_case_is_inconclusive_not_failed():
    case = _tool_case("p3.tools.m.choice")
    router = FakeRouter()

    verdict = synthetic_health.semantic_verdict(
        router,
        case,
        _tool_evidence(payload={"choices": [{"message": {"content": "thinking..."}, "finish_reason": "length"}]}),
        {"models": []},
    )

    assert verdict["status"] == "not_applicable", "budget exhaustion neither proves nor refutes tools (L3-R5)"
    assert verdict["inconclusive"] == "budget_exhausted"
    assert verdict["finish_reason"] == "length"


def test_portable_base_fixture_uses_only_the_conservative_schema_subset():
    from ficelle.use_cases.chat_completion import detect_tool_schema_features

    body = synthetic_health.tool_choice_body("ficelle/alpha/m", stream=False, marker="ficelle-health-0123456789abcdef")

    assert detect_tool_schema_features(body) == set(), "advanced dialect features belong to T8 (L3-R3)"

    for feature, fixture in synthetic_health.SCHEMA_DIALECT_FEATURES.items():
        dialect_body = synthetic_health.dialect_case_body("ficelle/alpha/m", feature, marker="ficelle-health-0123456789abcdef")
        detected = detect_tool_schema_features(dialect_body)
        assert detected, f"dialect case {feature} must exercise at least one advanced feature"


def test_dialect_matrix_aggregates_verdicts_per_model_and_feature():
    case = _tool_case(
        "p3.dialect.alpha-m.nullable_type_array",
        validator="tool-dialect",
        expected={"feature": "nullable_type_array", "tool_name": "set_locale_nullable", "arguments": {"locale": None, "city": "Madrid"}},
        tags=("tools", "t8", "dialect", "dialect:nullable_type_array"),
    )
    passing = {
        "case_id": case.id,
        "status": "pass",
        "tags": list(case.tags),
        "axes": {"semantic": {"status": "pass", "tool_assertions": {"selection": True, "arguments": True}}},
        "staged_verdict": {"selection_reachable": True, "upstream_reached": True, "protocol_valid": True, "tool_semantic_valid": True, "end_user_success": True},
    }

    metrics = synthetic_health._tool_metrics([case], [passing])

    assert metrics["schema_dialects"]["ficelle/alpha/m"]["nullable_type_array"]["verdict"] == "supported"


def test_dual_denominators_exclude_candidate_free_from_competence_only():
    reachable = _tool_case("p3.tools.m.choice")
    unreachable = _tool_case("p3.tools.n.choice")
    reachable_row = {
        "case_id": reachable.id,
        "status": "pass",
        "tags": list(reachable.tags),
        "axes": {"semantic": {"status": "pass", "tool_assertions": {"selection": True, "arguments": True}}},
        "route": {"candidate_count": 1},
        "staged_verdict": {"selection_reachable": True, "upstream_reached": True, "protocol_valid": True, "tool_semantic_valid": True, "end_user_success": True},
    }
    candidate_free_row = {
        "case_id": unreachable.id,
        "status": "fail",
        "tags": list(unreachable.tags),
        "axes": {"semantic": {"status": "fail", "classified_reason": "selection:no_available_model"}},
        "route": {"candidate_count": 0},
        "staged_verdict": {"selection_reachable": False, "upstream_reached": None, "protocol_valid": None, "tool_semantic_valid": None, "end_user_success": False},
    }

    metrics = synthetic_health._tool_metrics([reachable, unreachable], [reachable_row, candidate_free_row])

    # Availability counts the candidate-free failure; competence excludes it with a reason.
    assert metrics["service_availability"] == {"planned": 2, "succeeded": 1, "rate": 0.5}
    assert metrics["conditional_competence"]["reached_and_protocol_valid"] == 1
    assert metrics["conditional_competence"]["semantic_passed"] == 1
    assert metrics["conditional_competence"]["rate"] == 1.0
    assert metrics["conditional_competence"]["excluded_candidate_free"] == 1
    assert metrics["conditional_competence"]["excluded_candidate_free_case_ids"] == [unreachable.id]


def test_late_correlated_case_is_audited_through_its_recovered_route_row():
    case = _tool_case("p3.model.m.text", validator="exact-text", expected={"marker": "M"}, tags=("t1",))
    evidence = synthetic_health.HttpEvidence(
        status_code=None,
        content_type="",
        headers={},
        payload=None,
        text="",
        latency_seconds=45.0,
        stream_complete=False,
        stream_error=None,
        transport_error="ReadTimeout: read timed out",
        tool_calls=[],
        timed_out=True,
    )
    row = {
        "request_id": "req-late",
        "candidate_count": 3,
        "attempt_count": 1,
        "attempts": [{"source": "alpha", "model": "ficelle/alpha/m", "upstream": "m", "reason": "ok", "status": 200}],
        "final_status": 200,
        "selected_model": "ficelle/alpha/m",
        "selected_source": "alpha",
        "selected_upstream": "m",
        "synthetic_case_id": case.id,
    }

    routing, telemetry, policy, fallback_observed = synthetic_health.route_axes(case, evidence, [row])

    # The recovered row restores exact attribution: no correlation_missing anywhere.
    assert telemetry["status"] == "pass"
    assert telemetry["late_route_correlated"] is True
    assert policy.get("fallback_category") is None
    assert fallback_observed is False


def test_unrecovered_timeout_still_reports_correlation_missing():
    case = _tool_case("p3.model.m.text", validator="exact-text", expected={"marker": "M"}, tags=("t1",))
    evidence = synthetic_health.HttpEvidence(
        status_code=None, content_type="", headers={}, payload=None, text="",
        latency_seconds=45.0, stream_complete=False, stream_error=None,
        transport_error="ReadTimeout: read timed out", tool_calls=[], timed_out=True,
    )

    _routing, _telemetry, policy, _fb = synthetic_health.route_axes(case, evidence, [])

    assert policy["fallback_category"] == "correlation_missing"


def test_foreign_late_writes_are_notes_while_own_key_violations_still_fail():
    route = {
        "attempts": [
            {"source": "alpha", "model": "ficelle/alpha/m", "upstream": "m", "reason": "server_error"}
        ]
    }
    before = {"cooldowns": {}, "stats": {}, "model_errors": {}}
    # server_error expects a model cooldown for alpha::m; a foreign write lands too.
    after_foreign = {
        "cooldowns": {"alpha::m": {"reason": "server_error"}, "beta::other": {"reason": "timeout"}},
        "stats": {"alpha::m": {"consecutive_failures": 1}},
        "model_errors": {"alpha::m": {"reason": "server_error"}},
        "quarantine": {"gamma::else": {"reason": "model_not_found"}},
    }

    extended = synthetic_health.state_policy_verdict(route, before, after_foreign, late_extended_window=True)
    assert extended["status"] == "pass", extended
    assert "quarantine" in (extended.get("late_foreign_write_sections") or [])

    normal = synthetic_health.state_policy_verdict(route, before, after_foreign, late_extended_window=False)
    assert normal["status"] == "fail"

    # An own-key write in an unexpected scope stays a violation even in an extended window
    # (the L2-R4 contract): rate_limited must never write a model cooldown.
    route_rl = {"attempts": [{"source": "alpha", "model": "ficelle/alpha/m", "upstream": "m", "reason": "rate_limited"}]}
    after_own = {
        "cooldowns": {"alpha::m": {"reason": "rate_limited"}},
        "provider_cooldowns": {"alpha": {"reason": "rate_limited"}},
        "stats": {"alpha::m": {"consecutive_failures": 1}},
        "model_errors": {"alpha::m": {"reason": "rate_limited"}},
        "provider_errors": {"alpha": {"reason": "rate_limited"}},
    }
    verdict = synthetic_health.state_policy_verdict(route_rl, before, after_own, late_extended_window=True)
    assert verdict["status"] == "fail"
    assert "unexpected state change in cooldowns" in verdict["diagnostic"]


def test_billing_policy_writes_are_expected_in_both_their_sections():
    # billing_or_paid legitimately writes quarantine AND a 24h model cooldown
    # (CooldownPolicy.model_cooldown default): the checker must expect both sections
    # instead of misreading its own contract as a scope mismatch.
    route = {
        "attempts": [
            {"source": "ollama", "model": "ficelle/ollama/kimi-k3", "upstream": "kimi-k3", "reason": "billing_or_paid"}
        ]
    }
    before = {"cooldowns": {}, "quarantine": {}, "stats": {}, "model_errors": {}}
    after = {
        "cooldowns": {"ollama::kimi-k3": {"reason": "billing_or_paid"}},
        "quarantine": {"ollama::kimi-k3": {"reason": "billing_or_paid"}},
        "stats": {"ollama::kimi-k3": {"consecutive_failures": 1}},
        "model_errors": {"ollama::kimi-k3": {"reason": "billing_or_paid"}},
    }

    verdict = synthetic_health.state_policy_verdict(route, before, after)

    assert verdict["status"] == "pass", verdict


def _dialect_case(feature: str, control_id: str = "p3.tools.alpha-m.choice") -> SyntheticHealthCase:
    return SyntheticHealthCase(
        id=f"p3.dialect.alpha-m.{feature}",
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="ficelle/alpha/m",
        source="alpha",
        model_id="ficelle/alpha/m",
        upstream_id="m",
        validator="tool-dialect",
        timeout_class="tools",
        expected={"feature": feature, "tool_name": "t", "arguments": {}, "control_case_id": control_id},
        tags=("tools", "t8", "dialect", f"dialect:{feature}"),
    )


def _control_outcome(status: str, classified: str | None = None) -> dict:
    semantic = {"status": "pass" if status == "pass" else "fail"}
    if classified:
        semantic["classified_reason"] = classified
    return {"case_id": "p3.tools.alpha-m.choice", "status": status, "axes": {"semantic": semantic}}


def test_dialect_cases_are_not_sent_when_the_control_proves_no_tool_support():
    case = _dialect_case("enums")
    # Groq's compound shape: the portable base fixture itself is refused by the provider.
    outcomes = {"p3.tools.alpha-m.choice": _control_outcome("fail", "bad_upstream_request")}

    verdict = synthetic_health.dialect_control_verdict(case, outcomes)

    assert verdict == "tool_support_absent:p3.tools.alpha-m.choice"


def test_dialect_cases_run_when_plain_tool_calling_works():
    case = _dialect_case("enums")

    assert synthetic_health.dialect_control_verdict(case, {"p3.tools.alpha-m.choice": _control_outcome("pass")}) is None
    # No control outcome yet (ordering, resume): run the case rather than invent a verdict.
    assert synthetic_health.dialect_control_verdict(case, {}) is None


def test_a_skipped_or_broken_control_leaves_the_feature_unknown():
    case = _dialect_case("enums")

    skipped = synthetic_health.dialect_control_verdict(case, {"p3.tools.alpha-m.choice": _control_outcome("skipped")})
    assert skipped == "dialect_control_not_run:p3.tools.alpha-m.choice"

    broken = synthetic_health.dialect_control_verdict(case, {"p3.tools.alpha-m.choice": _control_outcome("fail", "server_error")})
    assert broken == "dialect_control_failed:p3.tools.alpha-m.choice"


def test_matrix_records_no_tool_support_instead_of_six_feature_failures():
    cases = [_dialect_case(feature) for feature in ("enums", "nullable_type_array")]
    outcomes = [
        {
            "case_id": case.id,
            "status": "skipped",
            "skip_reason": "tool_support_absent:p3.tools.alpha-m.choice",
            "tags": list(case.tags),
            "axes": {},
        }
        for case in cases
    ]

    matrix = synthetic_health._tool_metrics(cases, outcomes)["schema_dialects"]["ficelle/alpha/m"]

    assert {entry["verdict"] for entry in matrix.values()} == {"no_tool_support"}
    assert matrix["enums"]["reason"] == "tool_support_absent:p3.tools.alpha-m.choice"


def test_a_rerun_of_a_dialect_case_brings_its_control_case_along():
    control = SyntheticHealthCase(
        id="p3.tools.alpha-m.choice",
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="ficelle/alpha/m",
        validator="tool-choice",
        timeout_class="tools",
        tags=("tools", "t1"),
    )
    dialect = _dialect_case("enums")

    selected = synthetic_health.select_cases_with_dependencies([control, dialect], {dialect.id})

    # Without its control, the rerun could not tell "feature unsupported" from
    # "this model refuses tools entirely".
    assert {case.id for case in selected} == {control.id, dialect.id}


def test_dialect_cases_are_planned_after_their_control_case():
    router = FakeRouter(profiles={"ficelle/auto-tools": {"match": True}})
    config = router.load_runtime_config()
    catalog = {
        "generated_at": "2026-01-01T00:00:00Z",
        "models": [
            {
                "id": "ficelle/alpha/m",
                "source": "alpha",
                "upstream_id": "m",
                "invokable": True,
                "supports_tools": True,
                "context_length": 200000,
                "supported_parameters": ["tools", "tool_choice"],
                "free_access": {"eligible": True, "mode": "catalog_free"},
            }
        ],
    }
    inventory = synthetic_health.build_inventory(router, config, catalog)

    cases = synthetic_health.build_corpus(router, config, catalog, inventory, run_id="run-order")
    order = [case.id for case in cases]
    dialect_ids = [cid for cid in order if ".dialect." in cid]

    assert dialect_ids, "the corpus must plan dialect cases for a tool-capable model"
    controls = [cid for cid in order if cid.endswith(".choice") and ".tools." in cid]
    assert controls, "the portable base tool case must exist"
    # Every dialect case is planned after every control case of the same model.
    assert order.index(dialect_ids[0]) > order.index(controls[0])


class FakeHealthResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def stub_health_endpoint(monkeypatch: Any, payload: Any) -> None:
    monkeypatch.setattr(
        synthetic_health.requests,
        "get",
        lambda url, headers=None, timeout=None: FakeHealthResponse(payload),
    )


def matching_health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "generated_at": "2026-01-01T00:00:00Z",
        "model_count": 0,
        "build": synthetic_health.package_build_identity(),
    }


def stale_service_build() -> dict[str, Any]:
    return {
        **synthetic_health.package_build_identity(),
        "runtime_content_hash": "0" * 64,
        "package_dir": "/stale/venv/site-packages/ficelle",
    }


def test_service_preflight_accepts_a_service_running_the_harness_build(monkeypatch: Any) -> None:
    stub_health_endpoint(monkeypatch, matching_health_payload())

    ok, evidence = synthetic_health._service_preflight("http://127.0.0.1:8646/v1")

    assert ok is True
    assert evidence["build_match"] is True
    assert "build_diagnostic" not in evidence


def test_service_preflight_fails_closed_on_build_mismatch(monkeypatch: Any) -> None:
    stub_health_endpoint(monkeypatch, {"status": "ok", "build": stale_service_build()})

    ok, evidence = synthetic_health._service_preflight("http://127.0.0.1:8646/v1")

    assert ok is False
    assert evidence["build_match"] is False
    assert "build mismatch" in evidence["build_diagnostic"]
    assert "/stale/venv/site-packages/ficelle" in evidence["build_diagnostic"]


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "ok", "generated_at": "now", "model_count": 3},
        {"status": "ok", "build": "not-an-object"},
        {"status": "ok", "build": []},
    ],
)
def test_service_preflight_fails_closed_when_service_reports_no_usable_build(
    monkeypatch: Any,
    payload: dict[str, Any],
) -> None:
    stub_health_endpoint(monkeypatch, payload)

    ok, evidence = synthetic_health._service_preflight("http://127.0.0.1:8646/v1")

    assert ok is False
    assert evidence["service_build_identity"] is None
    assert evidence["build_match"] is False
    assert "did not report a build identity" in evidence["build_diagnostic"]


def test_build_mismatch_run_fails_explicitly_without_running_any_case(monkeypatch: Any, tmp_path: Path) -> None:
    router = FakeRouter()
    router.ROUTE_LOG_PATH = tmp_path / "routes.jsonl"
    monkeypatch.setattr(synthetic_health, "_load_router", lambda: router)
    stub_health_endpoint(monkeypatch, {"status": "ok", "build": stale_service_build()})

    payload, exit_code = synthetic_health.run_synthetic_health(tmp_path)

    assert exit_code == 2
    assert payload["status"] == "incomplete"
    rows = synthetic_health.read_case_journal(Path(payload["artifacts"]["cases"]))
    outcome_rows = [row for row in rows if row.get("event") == "outcome"]
    # The refused preflight is the whole journal: nothing ran against the stale build.
    assert [row["case_id"] for row in outcome_rows] == ["p0.preflight.loopback-service"]
    axes = outcome_rows[0]["axes"]
    assert axes["transport"]["status"] == "pass"
    assert axes["protocol"]["status"] == "pass"
    assert axes["semantic"]["status"] == "fail"
    assert "build mismatch" in axes["semantic"]["diagnostic"]
    run_metadata = json.loads(Path(payload["artifacts"]["run"]).read_text(encoding="utf-8"))
    assert run_metadata["service_build_match"] is False
    assert run_metadata["service_build_identity"]["runtime_content_hash"] == "0" * 64
    assert run_metadata["interruption_reason"] == "service_build_mismatch"


def test_matching_build_run_records_the_verified_service_identity(monkeypatch: Any, tmp_path: Path) -> None:
    router = FakeRouter()
    # No providers: the enabled-but-uncovered fake provider would otherwise degrade the
    # summary and hide the pass this test asserts.
    router.load_runtime_config = lambda: {}  # type: ignore[method-assign]
    router.ROUTE_LOG_PATH = tmp_path / "routes.jsonl"
    monkeypatch.setattr(synthetic_health, "_load_router", lambda: router)
    stub_health_endpoint(monkeypatch, matching_health_payload())

    payload, exit_code = synthetic_health.run_synthetic_health(tmp_path)

    assert exit_code == 0
    assert payload["status"] == "pass"
    run_metadata = json.loads(Path(payload["artifacts"]["run"]).read_text(encoding="utf-8"))
    assert run_metadata["service_build_match"] is True
    assert run_metadata["service_build_identity"]["runtime_content_hash"] == run_metadata["build_identity"]["runtime_content_hash"]
    # The finalize bookend re-reads the service build so a mid-run swap cannot hide.
    assert run_metadata["service_build_identity_after"]["runtime_content_hash"] == run_metadata["build_identity"]["runtime_content_hash"]
    assert run_metadata["service_build_changed_during_run"] is False
    summary = json.loads(Path(payload["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert summary["identity"]["service_build_identity"] == run_metadata["service_build_identity"]
    assert "service_build_identity" in summary["comparison"]["identity_changed"]


def test_resume_reverifies_the_service_build(monkeypatch: Any, tmp_path: Path) -> None:
    router = FakeRouter()
    router.ROUTE_LOG_PATH = tmp_path / "routes.jsonl"
    monkeypatch.setattr(synthetic_health, "_load_router", lambda: router)

    def service_down(url: str, headers: Any = None, timeout: Any = None) -> FakeHealthResponse:
        raise synthetic_health.requests.ConnectionError("connection refused")

    monkeypatch.setattr(synthetic_health.requests, "get", service_down)
    first, first_exit = synthetic_health.run_synthetic_health(tmp_path)
    assert first_exit == 2

    # The service that comes back up is not the build under validation: the resumed
    # run must refuse it, not resume the incident.
    stub_health_endpoint(monkeypatch, {"status": "ok", "build": stale_service_build()})

    resumed, exit_code = synthetic_health.resume_synthetic_health(tmp_path, first["run_id"])

    assert exit_code == 2
    assert resumed["status"] == "incomplete"
    run_metadata = json.loads(Path(resumed["artifacts"]["run"]).read_text(encoding="utf-8"))
    assert run_metadata["interruption_reason"] == "service_build_mismatch"


def test_resume_refuses_when_the_harness_build_changed(monkeypatch: Any, tmp_path: Path) -> None:
    router = FakeRouter()
    router.ROUTE_LOG_PATH = tmp_path / "routes.jsonl"
    monkeypatch.setattr(synthetic_health, "_load_router", lambda: router)
    monkeypatch.setattr(
        synthetic_health,
        "_service_preflight",
        lambda base_url, **kwargs: (False, {"transport_error": "connection refused"}),
    )
    first, _ = synthetic_health.run_synthetic_health(tmp_path)

    run_path = Path(first["artifacts"]["run"])
    metadata = json.loads(run_path.read_text(encoding="utf-8"))
    metadata["build_identity"]["runtime_content_hash"] = "0" * 64
    run_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(synthetic_health.SyntheticHealthError, match="runtime_content_hash"):
        synthetic_health.resume_synthetic_health(tmp_path, first["run_id"])
