from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ficelle.use_cases.quota_probe import (
    QuotaProbeModelPorts,
    QuotaProbePorts,
    QuotaProbeRunner,
    QuotaProbeStatePorts,
    find_quota_probe_model,
    quota_probe_body,
    record_quota_probe_result,
    reserve_quota_probe_in_state,
)


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def make_state_ports(state: dict[str, Any], *, now: float = 100.0) -> QuotaProbeStatePorts:
    def update_state(mutator: Callable[[dict[str, Any]], None], reason: str | None = None) -> dict[str, Any]:
        state["update_reason"] = reason
        mutator(state)
        return state

    return QuotaProbeStatePorts(
        safe_detail=lambda value: str(value)[:250] if value is not None else None,
        safe_float=safe_float,
        now_seconds=lambda: now,
        now_iso=lambda: "2026-06-22T00:00:00+00:00",
        update_state=update_state,
    )


def make_model_ports(*, now: float = 100.0) -> QuotaProbeModelPorts:
    return QuotaProbeModelPorts(
        safe_float=safe_float,
        now_seconds=lambda: now,
        model_is_quarantined=lambda model, state: model.get("id") in state.get("quarantined", set()),
        provider_on_cooldown=lambda source, state: (source in state.get("provider_cooldowns", set()), None),
        cooldown_key=lambda model: f"{model.get('source')}::{model.get('upstream_id')}",
        normalized_free_access=lambda model: model.get("free_access", {}),
        quota_cooldown_matches_model=lambda key, model: key == model.get("quota_key"),
    )


def make_runner(state: dict[str, Any], *, now: float = 100.0) -> tuple[QuotaProbeRunner, list[str], list[str]]:
    reserved: list[str] = []
    probed: list[str] = []

    def fresh_runtime_state() -> dict[str, Any]:
        return state

    def normalized_virtual_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return config.get("virtual_profiles", {})

    def canonical_virtual_model_id(model_id: str) -> str:
        return {"ficelle/auto-tools-legacy": "ficelle/auto-tools"}.get(model_id, model_id)

    def model_matches_profile_requirements(model: dict[str, Any], _profile: dict[str, Any]) -> bool:
        return bool(model.get("eligible", True))

    def quota_cooldown_matches_model(key: str, model: dict[str, Any]) -> bool:
        return key == model.get("quota_key")

    def find_quota_probe_model(catalog: dict[str, Any], key: str, _state: dict[str, Any]) -> dict[str, Any] | None:
        return next((model for model in catalog.get("models", []) if model.get("quota_key") == key), None)

    def reserve_quota_probe(_state: dict[str, Any], key: str, _row: dict[str, Any], _config: dict[str, Any]) -> bool:
        reserved.append(key)
        return key != "provider:busy"

    def probe_quota_cooldown(key: str, model: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any]:
        probed.append(key)
        return {"status": "pass", "key": key, "model_id": model["id"]}

    ports = QuotaProbePorts(
        fresh_runtime_state=fresh_runtime_state,
        safe_float=safe_float,
        safe_string_list=safe_string_list,
        normalized_virtual_profiles=normalized_virtual_profiles,
        canonical_virtual_model_id=canonical_virtual_model_id,
        model_matches_profile_requirements=model_matches_profile_requirements,
        quota_cooldown_matches_model=quota_cooldown_matches_model,
        find_quota_probe_model=find_quota_probe_model,
        reserve_quota_probe=reserve_quota_probe,
        probe_quota_cooldown=probe_quota_cooldown,
        virtual_models={"ficelle/auto-tools"},
        now_seconds=lambda: now,
    )
    return QuotaProbeRunner(ports), reserved, probed


def test_quota_probe_body_uses_safe_default_and_removes_stream():
    assert quota_probe_body({"source": "nvidia"}, {}) == {
        "messages": [{"role": "user", "content": "Reply exactly: ficelle-quota-probe-ok"}],
        "temperature": 0,
        "max_tokens": 8,
    }

    body = quota_probe_body(
        {"source": "nvidia"},
        {"providers": {"nvidia": {"quota_probe_body": {"messages": [], "stream": True, "max_tokens": 3}}}},
    )

    assert body == {"messages": [], "max_tokens": 3}


def test_find_quota_probe_model_filters_to_invokable_quota_free_matching_model():
    catalog = {
        "models": [
            {
                "id": "non-invokable",
                "source": "nvidia",
                "upstream_id": "non",
                "invokable": False,
                "quota_key": "provider:nvidia",
                "free_access": {"eligible": True, "mode": "quota_free"},
            },
            {
                "id": "paused",
                "source": "nvidia",
                "upstream_id": "paused",
                "invokable": True,
                "quota_key": "provider:nvidia",
                "free_access": {"eligible": True, "mode": "quota_free"},
            },
            {
                "id": "strict-zero",
                "source": "nvidia",
                "upstream_id": "strict",
                "invokable": True,
                "quota_key": "provider:nvidia",
                "free_access": {"eligible": True, "mode": "strict_zero"},
            },
            {
                "id": "match",
                "source": "nvidia",
                "upstream_id": "match",
                "invokable": True,
                "quota_key": "provider:nvidia",
                "free_access": {"eligible": True, "mode": "quota_free"},
            },
        ]
    }
    state = {"cooldowns": {"nvidia::paused": {"until": 200.0}}}

    match = find_quota_probe_model(catalog, "provider:nvidia", state, ports=make_model_ports())

    assert match is not None
    assert match["id"] == "match"


def test_find_quota_probe_model_skips_quarantined_and_provider_cooldown_candidates():
    catalog = {
        "models": [
            {
                "id": "quarantined",
                "source": "nvidia",
                "upstream_id": "q",
                "invokable": True,
                "quota_key": "provider:nvidia",
                "free_access": {"eligible": True, "mode": "quota_free"},
            },
            {
                "id": "provider-paused",
                "source": "paused",
                "upstream_id": "p",
                "invokable": True,
                "quota_key": "provider:nvidia",
                "free_access": {"eligible": True, "mode": "quota_free"},
            },
        ]
    }
    state = {"quarantined": {"quarantined"}, "provider_cooldowns": {"paused"}}

    assert find_quota_probe_model(catalog, "provider:nvidia", state, ports=make_model_ports()) is None


def test_record_quota_probe_result_preserves_safe_runtime_shape():
    state: dict[str, Any] = {}
    model = {"id": "ficelle/nvidia/free", "source": "nvidia", "upstream_id": "free"}

    record_quota_probe_result(
        state,
        "provider:nvidia",
        model,
        "fail",
        "quota_exhausted",
        "x" * 300,
        ports=make_state_ports(state),
    )

    assert state["quota_probe_results"]["provider:nvidia"] == {
        "status": "fail",
        "reason": "quota_exhausted",
        "detail": "x" * 250,
        "probed_at": "2026-06-22T00:00:00+00:00",
        "source": "nvidia",
        "model_id": "ficelle/nvidia/free",
        "upstream_id": "free",
    }


def test_reserve_quota_probe_in_state_sets_lock_window_and_update_reason():
    state = {"quota_cooldowns": {"provider:nvidia": {"source": "nvidia", "until": 200.0}}}

    reserved = reserve_quota_probe_in_state(
        state,
        "provider:nvidia",
        state["quota_cooldowns"]["provider:nvidia"],
        {"quota_probe_timeout_seconds": 11},
        ports=make_state_ports(state, now=100.0),
    )

    assert reserved is True
    row = state["quota_cooldowns"]["provider:nvidia"]
    assert row["last_probe_started_at"] == "2026-06-22T00:00:00+00:00"
    assert row["next_probe_at"] == 130.0
    assert row["probe_interval_seconds"] == 30
    assert state["update_reason"] == "reserve_quota_probe"


def test_reserve_quota_probe_in_state_refuses_missing_key_without_write():
    state = {"quota_cooldowns": {}}

    reserved = reserve_quota_probe_in_state(
        state,
        "provider:nvidia",
        {"source": "nvidia"},
        {"quota_probe_timeout_seconds": 60},
        ports=make_state_ports(state),
    )

    assert reserved is False
    assert "update_reason" not in state


def test_due_quota_probe_keys_respects_manual_profile_and_auto_tail():
    state = {
        "quota_cooldowns": {
            "provider:manual": {"until": 200, "next_probe_at": 90},
            "provider:tail": {"until": 200, "next_probe_at": 90},
            "provider:excluded": {"until": 200, "next_probe_at": 90},
            "provider:not-due": {"until": 200, "next_probe_at": 150},
        }
    }
    config = {
        "virtual_profiles": {
            "ficelle/auto-tools": {
                "mode": "manual_order",
                "models": ["manual"],
                "auto_tail": True,
            }
        }
    }
    models = [
        {"id": "manual", "quota_key": "provider:manual"},
        {"id": "tail", "quota_key": "provider:tail"},
        {"id": "excluded", "quota_key": "provider:excluded", "eligible": False},
        {"id": "not-due", "quota_key": "provider:not-due"},
    ]
    runner, _, _ = make_runner(state)

    assert runner.due_quota_probe_keys_for_request("ficelle/auto-tools", models, config, state) == [
        "provider:manual",
        "provider:tail",
    ]


def test_due_quota_probe_keys_supports_direct_model_requests():
    state = {
        "quota_cooldowns": {
            "provider:target": {"until": 200, "next_probe_at": 90},
            "provider:other": {"until": 200, "next_probe_at": 90},
        }
    }
    models = [
        {"id": "ficelle/provider/target", "upstream_id": "target", "quota_key": "provider:target"},
        {"id": "ficelle/provider/other", "upstream_id": "other", "quota_key": "provider:other"},
    ]
    runner, _, _ = make_runner(state)

    assert runner.due_quota_probe_keys_for_request("target", models, {}, state) == ["provider:target"]


def test_run_due_quota_probes_filters_source_keys_and_reservation():
    state = {
        "quota_cooldowns": {
            "provider:nvidia": {"source": "nvidia", "until": 200, "next_probe_at": 90},
            "provider:busy": {"source": "nvidia", "until": 200, "next_probe_at": 90},
            "provider:groq": {"source": "groq", "until": 200, "next_probe_at": 90},
            "provider:not-due": {"source": "nvidia", "until": 200, "next_probe_at": 150},
        }
    }
    catalog = {
        "models": [
            {"id": "nvidia-model", "quota_key": "provider:nvidia"},
            {"id": "busy-model", "quota_key": "provider:busy"},
            {"id": "groq-model", "quota_key": "provider:groq"},
        ]
    }
    runner, reserved, probed = make_runner(state)

    result = runner.run_due_quota_probes(
        {},
        catalog,
        source="nvidia",
        keys={"provider:nvidia", "provider:busy", "provider:groq"},
    )

    assert result["probed_count"] == 1
    assert result["results"] == [{"status": "pass", "key": "provider:nvidia", "model_id": "nvidia-model"}]
    assert reserved == ["provider:nvidia", "provider:busy"]
    assert probed == ["provider:nvidia"]
