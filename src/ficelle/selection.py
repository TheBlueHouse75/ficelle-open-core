from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ficelle.domain_models import CatalogModel, SelectionPurpose, SelectionRequest, SelectionResult
from ficelle.use_cases.cooldowns import cooldown_block_scope


@dataclass(frozen=True)
class SelectionPolicy:
    model_on_cooldown: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, Any]]
    model_is_quarantined: Callable[[dict[str, Any], dict[str, Any]], bool]
    normalized_virtual_profiles: Callable[[dict[str, Any]], dict[str, dict[str, Any]]]
    canonical_virtual_model_id: Callable[[str], str]
    model_matches_profile_requirements: Callable[[dict[str, Any], dict[str, Any]], bool]
    manual_order_candidates: Callable[[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]], list[dict[str, Any]]]
    sort_available_for_virtual_model: Callable[[str, list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]
    route_competence_gate_result: Callable[[str, list[dict[str, Any]], dict[str, Any]], tuple[list[dict[str, Any]], bool]]
    virtual_models: set[str]
    # Returns a block reason for a catalog row preserved past its stale grace (L1-R1), or
    # None when the row is fresh or still within grace. Optional so pre-existing callers
    # and tests keep their behavior unchanged.
    stale_model_block_reason: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str | None] | None = None


def typed_catalog_rows(catalog: dict[str, Any]) -> list[tuple[CatalogModel, dict[str, Any]]]:
    raw_models = catalog.get("models", []) if isinstance(catalog, dict) else []
    return [
        (model, raw)
        for raw in raw_models
        if isinstance(raw, dict) and (model := CatalogModel.from_raw(raw)) is not None
    ]


def sort_available_for_virtual_model(
    requested_model: str,
    available: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    model_auto_score: Callable[[str, dict[str, Any], dict[str, Any]], float],
    safe_int: Callable[[Any, int], int],
    safe_float: Callable[[Any, float], float],
    cooldown_key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    ordered = list(available)
    ordered.sort(
        key=lambda model: (
            -model_auto_score(requested_model, model, state),
            safe_int(((state.get("stats") or {}).get(cooldown_key(model)) or {}).get("consecutive_failures"), 0),
            -safe_int(model.get("context_length"), 0),
            safe_float(model.get("burn_weight"), 1.0),
            str(model.get("id") or ""),
        )
    )
    return ordered


def manual_order_candidates(
    requested_model: str,
    eligible: list[dict[str, Any]],
    state: dict[str, Any],
    profile: dict[str, Any],
    *,
    sort_available_for_virtual_model: Callable[[str, list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_id = {str(model.get("id")): model for model in eligible}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model_id in profile.get("models") or []:
        model = by_id.get(str(model_id))
        if model is None or str(model.get("id")) in seen:
            continue
        selected.append(model)
        seen.add(str(model.get("id")))

    if profile.get("auto_tail", True):
        tail = [model for model in eligible if str(model.get("id")) not in seen]
        selected.extend(sort_available_for_virtual_model(requested_model, tail, state))
    return selected


def candidates_for_profile(
    profile_id: str,
    available: list[dict[str, Any]],
    state: dict[str, Any],
    profile: dict[str, Any],
    *,
    canonical_virtual_model_id: Callable[[str], str],
    model_matches_profile_requirements: Callable[[dict[str, Any], dict[str, Any]], bool],
    manual_order_candidates: Callable[[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]], list[dict[str, Any]]],
    sort_available_for_virtual_model: Callable[[str, list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    canonical_profile_id = canonical_virtual_model_id(profile_id)
    eligible = [model for model in available if model_matches_profile_requirements(model, profile)]
    if profile.get("mode") == "manual_order":
        return manual_order_candidates(canonical_profile_id, eligible, state, profile)
    return sort_available_for_virtual_model(canonical_profile_id, eligible, state)


def select_models_result_from_typed_rows(
    requested_model: str,
    typed_rows: list[tuple[CatalogModel, dict[str, Any]]],
    config: dict[str, Any],
    state: dict[str, Any],
    policy: SelectionPolicy,
    *,
    purpose: SelectionPurpose = "route",
) -> SelectionResult:
    typed_by_row_identity = {id(raw): model for model, raw in typed_rows}
    excluded_reasons = {
        model.id: "not_invokable"
        for model, raw in typed_rows
        if model.id and not raw.get("invokable")
    }
    invokable_rows = [raw for _model, raw in typed_rows if raw.get("invokable")]
    available: list[dict[str, Any]] = []
    for raw_model in invokable_rows:
        model_id = str(raw_model.get("id") or "")
        if policy.stale_model_block_reason is not None:
            stale_reason = policy.stale_model_block_reason(raw_model, config, state)
            if stale_reason:
                if model_id:
                    excluded_reasons[model_id] = stale_reason
                continue
        on_cd, cooldown_reason = policy.model_on_cooldown(raw_model, state)
        if on_cd:
            if model_id:
                # Keep the block's scope so refusals and telemetry can name it (L1-R4);
                # the prefix convention is decoded in one place, cooldown_block_scope.
                scope = cooldown_block_scope(cooldown_reason)
                excluded_reasons[model_id] = (
                    "provider_cooldown" if scope == "provider"
                    else "quota_cooldown" if scope == "quota"
                    else "cooldown"
                )
            continue
        if policy.model_is_quarantined(raw_model, state):
            if model_id:
                excluded_reasons[model_id] = "quarantined"
            continue
        available.append(raw_model)

    profiles = policy.normalized_virtual_profiles(config)
    canonical_requested_model = policy.canonical_virtual_model_id(requested_model)
    profile = profiles.get(requested_model) or profiles.get(canonical_requested_model)
    if profile:
        before_profile = {str(model.get("id") or "") for model in available}
        eligible = [model for model in available if policy.model_matches_profile_requirements(model, profile)]
        for model_id in before_profile - {str(model.get("id") or "") for model in eligible}:
            if model_id:
                excluded_reasons[model_id] = "profile_requirements"
        if profile.get("mode") == "manual_order":
            candidates = policy.manual_order_candidates(canonical_requested_model, eligible, state, profile)
        else:
            candidates = policy.sort_available_for_virtual_model(canonical_requested_model, eligible, state)
    elif canonical_requested_model in policy.virtual_models or requested_model in policy.virtual_models or requested_model in profiles:
        candidates = policy.sort_available_for_virtual_model(canonical_requested_model, available, state)
    else:
        candidates = next(
            ([model] for model in available if requested_model in {model.get("id"), model.get("upstream_id")}),
            [],
        )

    anti_empty_fallback = False
    if purpose == "route":
        before_competence = {str(model.get("id") or "") for model in candidates}
        candidates, anti_empty_fallback = policy.route_competence_gate_result(canonical_requested_model, candidates, state)
        for model_id in before_competence - {str(model.get("id") or "") for model in candidates}:
            if model_id:
                excluded_reasons[model_id] = "competence_gate"

    typed_candidates: list[CatalogModel] = []
    for raw in candidates:
        model = typed_by_row_identity.get(id(raw)) or CatalogModel.from_raw(raw)
        if model is not None:
            typed_candidates.append(model)
    return SelectionResult(
        request=SelectionRequest(
            requested_model=str(requested_model or ""),
            canonical_model=canonical_requested_model,
            purpose=purpose,
        ),
        candidates=typed_candidates,
        excluded_reasons=excluded_reasons,
        anti_empty_fallback=anti_empty_fallback,
    )
