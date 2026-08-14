from __future__ import annotations

import copy

from ficelle.selection import (
    SelectionPolicy,
    candidates_for_profile as candidates_for_profile_policy,
    manual_order_candidates as manual_order_candidates_policy,
    select_models_result_from_typed_rows,
    sort_available_for_virtual_model as sort_available_for_virtual_model_policy,
    typed_catalog_rows,
)


def model_row(model_id: str, **overrides):
    row = {
        "id": model_id,
        "source": "openrouter",
        "upstream_id": model_id.removeprefix("ficelle/openrouter/"),
        "context_length": 131_072,
        "supports_tools": True,
        "supports_structured_outputs": True,
        "invokable": True,
        "pricing": {"prompt": "0", "completion": "0"},
        "free_access": {"eligible": True, "mode": "catalog_free", "scope": "model"},
    }
    row.update(overrides)
    return row


def simple_policy(*, competence_blocked_ids: set[str] | None = None) -> SelectionPolicy:
    blocked_ids = competence_blocked_ids or set()

    def model_on_cooldown(model, state):
        cooldowns = state.get("cooldowns") if isinstance(state.get("cooldowns"), dict) else {}
        return str(model.get("id") or "") in cooldowns, None

    def model_is_quarantined(model, state):
        quarantine = state.get("quarantine") if isinstance(state.get("quarantine"), dict) else {}
        return str(model.get("id") or "") in quarantine

    def normalized_virtual_profiles(config):
        return config.get("virtual_profiles") if isinstance(config.get("virtual_profiles"), dict) else {}

    def model_matches_profile_requirements(model, profile):
        requirements = profile.get("requirements") if isinstance(profile.get("requirements"), dict) else {}
        return not requirements.get("tools", True) or bool(model.get("supports_tools"))

    def manual_order_candidates(_requested_model, eligible, _state, profile):
        by_id = {str(model.get("id")): model for model in eligible}
        ordered = [by_id[model_id] for model_id in profile.get("models", []) if model_id in by_id]
        seen = {str(model.get("id")) for model in ordered}
        if profile.get("auto_tail", True):
            ordered.extend(model for model in eligible if str(model.get("id")) not in seen)
        return ordered

    def sort_available_for_virtual_model(_requested_model, available, _state):
        return sorted(available, key=lambda model: str(model.get("id") or ""))

    def route_competence_gate_result(_profile_id, candidates, _state):
        gated = [model for model in candidates if str(model.get("id") or "") not in blocked_ids]
        return (gated, False) if gated else (candidates, bool(candidates))

    return SelectionPolicy(
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=model_is_quarantined,
        normalized_virtual_profiles=normalized_virtual_profiles,
        canonical_virtual_model_id=lambda model_id: model_id,
        model_matches_profile_requirements=model_matches_profile_requirements,
        manual_order_candidates=manual_order_candidates,
        sort_available_for_virtual_model=sort_available_for_virtual_model,
        route_competence_gate_result=route_competence_gate_result,
        virtual_models={"ficelle/auto-tools"},
    )


def test_pure_selection_is_deterministic_and_does_not_mutate_inputs():
    catalog = {
        "models": [
            model_row("ficelle/openrouter/b"),
            model_row("ficelle/openrouter/a"),
            model_row("ficelle/openrouter/no-tools", supports_tools=False),
            model_row("ficelle/openrouter/not-invokable", invokable=False),
        ]
    }
    config = {
        "virtual_profiles": {
            "ficelle/auto-tools": {
                "mode": "auto",
                "requirements": {"tools": True},
            }
        }
    }
    state = {}
    before = (copy.deepcopy(catalog), copy.deepcopy(config), copy.deepcopy(state))
    typed_rows = typed_catalog_rows(catalog)

    first = select_models_result_from_typed_rows("ficelle/auto-tools", typed_rows, config, state, simple_policy())
    second = select_models_result_from_typed_rows("ficelle/auto-tools", typed_rows, config, state, simple_policy())

    assert [candidate.id for candidate in first.candidates] == ["ficelle/openrouter/a", "ficelle/openrouter/b"]
    assert first == second
    assert first.excluded_reasons["ficelle/openrouter/no-tools"] == "profile_requirements"
    assert first.excluded_reasons["ficelle/openrouter/not-invokable"] == "not_invokable"
    assert (catalog, config, state) == before


def test_pure_selection_applies_manual_order_auto_tail_and_state_exclusions():
    catalog = {
        "models": [
            model_row("ficelle/openrouter/tail"),
            model_row("ficelle/openrouter/cooldown"),
            model_row("ficelle/openrouter/pinned"),
            model_row("ficelle/openrouter/quarantined"),
        ]
    }
    config = {
        "virtual_profiles": {
            "ficelle/auto-tools": {
                "mode": "manual_order",
                "models": ["ficelle/openrouter/pinned", "ficelle/openrouter/cooldown"],
                "auto_tail": True,
                "requirements": {"tools": True},
            }
        }
    }
    state = {
        "cooldowns": {"ficelle/openrouter/cooldown": {"reason": "timeout"}},
        "quarantine": {"ficelle/openrouter/quarantined": {"reason": "billing"}},
    }

    result = select_models_result_from_typed_rows(
        "ficelle/auto-tools",
        typed_catalog_rows(catalog),
        config,
        state,
        simple_policy(),
    )

    assert [candidate.id for candidate in result.candidates] == [
        "ficelle/openrouter/pinned",
        "ficelle/openrouter/tail",
    ]
    assert result.excluded_reasons["ficelle/openrouter/cooldown"] == "cooldown"
    assert result.excluded_reasons["ficelle/openrouter/quarantined"] == "quarantined"


def test_pure_selection_keeps_benchmark_pool_and_marks_route_anti_empty_fallback():
    catalog = {
        "models": [
            model_row("ficelle/openrouter/a"),
            model_row("ficelle/openrouter/b"),
        ]
    }
    config = {"virtual_profiles": {"ficelle/auto-tools": {"mode": "auto", "requirements": {"tools": True}}}}
    policy = simple_policy(competence_blocked_ids={"ficelle/openrouter/a", "ficelle/openrouter/b"})

    route_result = select_models_result_from_typed_rows(
        "ficelle/auto-tools",
        typed_catalog_rows(catalog),
        config,
        {},
        policy,
    )
    benchmark_result = select_models_result_from_typed_rows(
        "ficelle/auto-tools",
        typed_catalog_rows(catalog),
        config,
        {},
        policy,
        purpose="benchmark",
    )

    assert {candidate.id for candidate in route_result.candidates} == {
        "ficelle/openrouter/a",
        "ficelle/openrouter/b",
    }
    assert route_result.anti_empty_fallback is True
    assert route_result.excluded_reasons == {}
    assert [candidate.id for candidate in benchmark_result.candidates] == [
        "ficelle/openrouter/a",
        "ficelle/openrouter/b",
    ]
    assert benchmark_result.anti_empty_fallback is False


def test_sort_available_for_virtual_model_uses_score_failures_context_and_id_tiebreakers():
    models = [
        model_row("ficelle/openrouter/high-failures", upstream_id="high-failures", context_length=200_000),
        model_row("ficelle/openrouter/high-clean", upstream_id="high-clean", context_length=100_000),
        model_row("ficelle/openrouter/context-short", upstream_id="context-short", context_length=100_000),
        model_row("ficelle/openrouter/context-long", upstream_id="context-long", context_length=900_000),
        model_row("ficelle/openrouter/same-score-z", upstream_id="same-score-z", context_length=100_000),
        model_row("ficelle/openrouter/same-score-a", upstream_id="same-score-a", context_length=100_000),
    ]
    scores = {
        "ficelle/openrouter/high-failures": 90.0,
        "ficelle/openrouter/high-clean": 90.0,
        "ficelle/openrouter/context-short": 80.0,
        "ficelle/openrouter/context-long": 80.0,
        "ficelle/openrouter/same-score-z": 70.0,
        "ficelle/openrouter/same-score-a": 70.0,
    }
    state = {"stats": {"openrouter::high-failures": {"consecutive_failures": 2}}}

    ordered = sort_available_for_virtual_model_policy(
        "ficelle/auto-tools",
        models,
        state,
        model_auto_score=lambda _requested, model, _state: scores[str(model["id"])],
        safe_int=lambda value, default: int(value) if value is not None else default,
        safe_float=lambda value, default: float(value) if value is not None else default,
        cooldown_key=lambda model: f"{model.get('source')}::{model.get('upstream_id')}",
    )

    assert [model["id"] for model in ordered] == [
        "ficelle/openrouter/high-clean",
        "ficelle/openrouter/high-failures",
        "ficelle/openrouter/context-long",
        "ficelle/openrouter/context-short",
        "ficelle/openrouter/same-score-a",
        "ficelle/openrouter/same-score-z",
    ]


def test_manual_order_candidates_deduplicates_profile_models_and_sorts_auto_tail():
    eligible = [
        model_row("ficelle/openrouter/tail-b"),
        model_row("ficelle/openrouter/pinned"),
        model_row("ficelle/openrouter/tail-a"),
    ]
    profile = {
        "models": [
            "ficelle/openrouter/pinned",
            "ficelle/openrouter/missing",
            "ficelle/openrouter/pinned",
        ],
        "auto_tail": True,
    }

    ordered = manual_order_candidates_policy(
        "ficelle/auto-tools",
        eligible,
        {},
        profile,
        sort_available_for_virtual_model=lambda _requested, models, _state: sorted(
            models,
            key=lambda model: str(model.get("id") or ""),
        ),
    )

    assert [model["id"] for model in ordered] == [
        "ficelle/openrouter/pinned",
        "ficelle/openrouter/tail-a",
        "ficelle/openrouter/tail-b",
    ]


def test_candidates_for_profile_filters_requirements_and_uses_canonical_manual_order():
    available = [
        model_row("ficelle/openrouter/eligible"),
        model_row("ficelle/openrouter/ineligible", supports_tools=False),
        model_row("ficelle/openrouter/tail"),
    ]
    profile = {
        "mode": "manual_order",
        "models": ["ficelle/openrouter/tail"],
        "requirements": {"tools": True},
    }

    def manual_order(requested, eligible, _state, candidate_profile):
        assert requested == "ficelle/auto-tools"
        by_id = {model["id"]: model for model in eligible}
        return [by_id[model_id] for model_id in candidate_profile.get("models", []) if model_id in by_id]

    candidates = candidates_for_profile_policy(
        "ficelle/auto",
        available,
        {},
        profile,
        canonical_virtual_model_id=lambda profile_id: "ficelle/auto-tools" if profile_id == "ficelle/auto" else profile_id,
        model_matches_profile_requirements=lambda model, candidate_profile: (
            not candidate_profile.get("requirements", {}).get("tools", True) or bool(model.get("supports_tools"))
        ),
        manual_order_candidates=manual_order,
        sort_available_for_virtual_model=lambda _requested, eligible, _state: list(reversed(eligible)),
    )

    assert [model["id"] for model in candidates] == ["ficelle/openrouter/tail"]


def test_candidates_for_profile_uses_auto_sort_for_non_manual_profiles():
    available = [
        model_row("ficelle/openrouter/b"),
        model_row("ficelle/openrouter/a"),
    ]

    candidates = candidates_for_profile_policy(
        "ficelle/auto-tools",
        available,
        {},
        {"mode": "auto"},
        canonical_virtual_model_id=lambda profile_id: profile_id,
        model_matches_profile_requirements=lambda _model, _profile: True,
        manual_order_candidates=lambda _requested, _eligible, _state, _profile: [],
        sort_available_for_virtual_model=lambda _requested, eligible, _state: sorted(
            eligible,
            key=lambda model: str(model.get("id") or ""),
        ),
    )

    assert [model["id"] for model in candidates] == ["ficelle/openrouter/a", "ficelle/openrouter/b"]


# --- Lot 1 (synthetic-health remediation): stale grace and scoped exclusions ----------------


def test_stale_rows_block_past_grace_and_route_within_it():
    import dataclasses

    def stale_block(model, _config, _state):
        return "stale_catalog" if model.get("id") == "ficelle/openrouter/expired" else None

    policy = dataclasses.replace(simple_policy(), stale_model_block_reason=stale_block)
    catalog = {
        "models": [
            model_row("ficelle/openrouter/fresh"),
            model_row("ficelle/openrouter/expired", catalog_stale=True, stale_since="2026-08-08T09:00:00+00:00"),
            model_row("ficelle/openrouter/graced", catalog_stale=True, stale_since="2026-08-08T10:00:00+00:00"),
        ]
    }

    result = select_models_result_from_typed_rows(
        "ficelle/auto-tools", typed_catalog_rows(catalog), {}, {}, policy
    )

    selected = {candidate.id for candidate in result.candidates}
    assert "ficelle/openrouter/expired" not in selected
    assert {"ficelle/openrouter/fresh", "ficelle/openrouter/graced"} <= selected
    assert result.excluded_reasons["ficelle/openrouter/expired"] == "stale_catalog"


def test_cooldown_exclusions_carry_their_scope():
    import dataclasses

    def on_cooldown(model, _state):
        reasons = {
            "ficelle/openrouter/provider-cooled": (True, "provider:rate_limited"),
            "ficelle/openrouter/quota-cooled": (True, "quota:quota_exhausted"),
            "ficelle/openrouter/model-cooled": (True, "timeout"),
        }
        return reasons.get(str(model.get("id")), (False, None))

    policy = dataclasses.replace(simple_policy(), model_on_cooldown=on_cooldown)
    catalog = {
        "models": [
            model_row("ficelle/openrouter/provider-cooled"),
            model_row("ficelle/openrouter/quota-cooled"),
            model_row("ficelle/openrouter/model-cooled"),
            model_row("ficelle/openrouter/free"),
        ]
    }

    result = select_models_result_from_typed_rows(
        "ficelle/auto-tools", typed_catalog_rows(catalog), {}, {}, policy
    )

    assert result.excluded_reasons["ficelle/openrouter/provider-cooled"] == "provider_cooldown"
    assert result.excluded_reasons["ficelle/openrouter/quota-cooled"] == "quota_cooldown"
    assert result.excluded_reasons["ficelle/openrouter/model-cooled"] == "cooldown"
    assert {candidate.id for candidate in result.candidates} == {"ficelle/openrouter/free"}
