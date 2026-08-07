from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ficelle.selection import SelectionPolicy
from ficelle.use_cases.model_selection import ModelSelectionPorts, ModelSelectionRunner


def simple_policy() -> SelectionPolicy:
    def model_on_cooldown(model: dict[str, Any], state: dict[str, Any]) -> tuple[bool, None]:
        key = f"{model.get('source')}::{model.get('upstream_id')}"
        cooldowns = state.get("cooldowns") if isinstance(state.get("cooldowns"), dict) else {}
        return (key in cooldowns, None)

    def manual_order_candidates(_requested_model, eligible, _state, profile):
        by_id = {model["id"]: model for model in eligible}
        return [by_id[model_id] for model_id in profile.get("models", []) if model_id in by_id]

    return SelectionPolicy(
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=lambda _model, _state: False,
        normalized_virtual_profiles=lambda config: config.get("virtual_profiles", {}),
        canonical_virtual_model_id=lambda model_id: model_id,
        model_matches_profile_requirements=lambda model, _profile: bool(model.get("eligible", True)),
        manual_order_candidates=manual_order_candidates,
        sort_available_for_virtual_model=lambda _requested, models, _state: list(models),
        route_competence_gate_result=lambda _profile, candidates, _state: (list(candidates), False),
        virtual_models={"ficelle/auto-tools"},
    )


def make_ports(
    states: list[dict[str, Any]],
    *,
    due_keys: list[str] | None = None,
    probe_calls: list[set[str]] | None = None,
    scheduled_calls: list[set[str]] | None = None,
) -> ModelSelectionPorts:
    def fresh_runtime_state() -> dict[str, Any]:
        if len(states) > 1:
            return states.pop(0)
        return states[0]

    def run_due_quota_probes(_config: dict[str, Any], _catalog: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if probe_calls is not None:
            probe_calls.append(set(kwargs.get("keys") or set()))
        return {"status": "ok", "probed_count": len(kwargs.get("keys") or set()), "results": []}

    def schedule_quota_probes(_config: dict[str, Any], _catalog: dict[str, Any], **kwargs: Any) -> None:
        if scheduled_calls is not None:
            scheduled_calls.append(set(kwargs.get("keys") or set()))

    return ModelSelectionPorts(
        load_config=lambda: {},
        fresh_runtime_state=fresh_runtime_state,
        due_quota_probe_keys_for_request=lambda *_args: list(due_keys or []),
        run_due_quota_probes=run_due_quota_probes,
        schedule_quota_probes=schedule_quota_probes,
        safe_int=lambda value, default: int(value) if value is not None else default,
        apply_verified_capability_ttl=lambda _config: None,
        apply_route_on_capability_reference=lambda _config: None,
        selection_policy=simple_policy,
    )


def test_model_selection_runner_runs_limited_due_quota_probes_and_reloads_state():
    initial_state = {"stats": {}}
    refreshed_state = {"cooldowns": {"openrouter::blocked": {"until": 999}}}
    probe_calls: list[set[str]] = []
    ports = make_ports(
        [initial_state, refreshed_state],
        due_keys=["provider:first", "provider:second"],
        probe_calls=probe_calls,
    )
    runner = ModelSelectionRunner(ports)
    catalog = {
        "models": [
            {"id": "ficelle/openrouter/blocked", "source": "openrouter", "upstream_id": "blocked", "invokable": True},
            {"id": "ficelle/openrouter/ok", "source": "openrouter", "upstream_id": "ok", "invokable": True},
        ]
    }

    result = runner.select_models_result(
        "ficelle/auto-tools",
        catalog,
        {"quota_inline_probe_limit": 1},
    )

    assert probe_calls == [{"provider:first"}]
    assert [model.id for model in result.candidates] == ["ficelle/openrouter/ok"]


def test_model_selection_defers_the_probes_the_inline_limit_drops():
    # The inline probe is what puts a recovered model back into THIS request's candidate list,
    # so it stays on the request thread. What used to be lost is everything past the limit: those
    # keys were dropped and waited for some later request to make them due again. They are now
    # handed to the background instead, without changing what this request routes to.
    probe_calls: list[set[str]] = []
    scheduled_calls: list[set[str]] = []
    ports = make_ports(
        [{"stats": {}}, {"stats": {}}],
        due_keys=["provider:first", "provider:second", "provider:third"],
        probe_calls=probe_calls,
        scheduled_calls=scheduled_calls,
    )
    runner = ModelSelectionRunner(ports)
    catalog = {"models": [{"id": "ficelle/openrouter/ok", "source": "openrouter", "upstream_id": "ok", "invokable": True}]}

    runner.select_models_result("ficelle/auto-tools", catalog, {"quota_inline_probe_limit": 1})

    assert probe_calls == [{"provider:first"}]
    assert scheduled_calls == [{"provider:second", "provider:third"}]


def test_model_selection_schedules_nothing_when_every_due_key_ran_inline():
    probe_calls: list[set[str]] = []
    scheduled_calls: list[set[str]] = []
    ports = make_ports(
        [{"stats": {}}, {"stats": {}}],
        due_keys=["provider:only"],
        probe_calls=probe_calls,
        scheduled_calls=scheduled_calls,
    )
    runner = ModelSelectionRunner(ports)
    catalog = {"models": [{"id": "ficelle/openrouter/ok", "source": "openrouter", "upstream_id": "ok", "invokable": True}]}

    runner.select_models_result("ficelle/auto-tools", catalog, {"quota_inline_probe_limit": 1})

    assert probe_calls == [{"provider:only"}]
    assert scheduled_calls == []


def test_model_selection_from_state_does_not_read_runtime_state_or_probe_quota():
    def fail_runtime_state() -> dict[str, Any]:
        raise AssertionError("select_models_result_from_state must use the supplied state")

    ports = make_ports([{}], due_keys=["provider:first"])
    ports = replace(
        ports,
        fresh_runtime_state=fail_runtime_state,
        run_due_quota_probes=lambda *_args, **_kwargs: pytest.fail("quota probes must not run"),
    )
    runner = ModelSelectionRunner(ports)
    catalog = {
        "models": [
            {"id": "ficelle/openrouter/ok", "source": "openrouter", "upstream_id": "ok", "invokable": True},
        ]
    }

    result = runner.select_models_result_from_state("ficelle/auto-tools", catalog, {}, {})

    assert [model.id for model in result.candidates] == ["ficelle/openrouter/ok"]
