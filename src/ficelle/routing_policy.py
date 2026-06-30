from __future__ import annotations

from typing import Any, Callable, Iterable


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def model_has_any(model: dict[str, Any], field: str, values: Iterable[str]) -> bool:
    current = {str(item) for item in model.get(field, []) if str(item)}
    return any(str(value) in current for value in values)


def model_has_all(model: dict[str, Any], field: str, values: Iterable[str]) -> bool:
    current = {str(item) for item in model.get(field, []) if str(item)}
    return all(str(value) in current for value in values)


def profile_excluded_model_ids(profile: dict[str, Any]) -> set[str]:
    return {str(item) for item in profile.get("excluded_models") or []}


def model_excluded_from_profile(model: dict[str, Any], profile: dict[str, Any]) -> bool:
    model_id = str(model.get("id") or "")
    return bool(model_id) and model_id in profile_excluded_model_ids(profile)


def model_matches_profile_requirements(
    model: dict[str, Any],
    profile: dict[str, Any],
    *,
    default_min_context: int,
    free_access_eligible: Callable[[dict[str, Any]], bool],
) -> bool:
    if model_excluded_from_profile(model, profile):
        return False
    requirements = profile.get("requirements") if isinstance(profile.get("requirements"), dict) else {}
    min_context = safe_int(requirements.get("min_context"), default_min_context)
    if requirements.get("free", True) and not free_access_eligible(model):
        return False
    if requirements.get("tools", True) and not bool(model.get("supports_tools")):
        return False
    structured = requirements.get("structured")
    if isinstance(structured, bool) and bool(model.get("supports_structured_outputs")) is not structured:
        return False
    if not model_has_all(model, "input_modalities", safe_string_list(requirements.get("input_modalities"))):
        return False
    input_any = safe_string_list(requirements.get("input_modalities_any"))
    if input_any and not model_has_any(model, "input_modalities", input_any):
        return False
    if not model_has_all(model, "output_modalities", safe_string_list(requirements.get("output_modalities"))):
        return False
    output_any = safe_string_list(requirements.get("output_modalities_any"))
    if output_any and not model_has_any(model, "output_modalities", output_any):
        return False
    if not model_has_all(model, "supported_parameters", safe_string_list(requirements.get("supported_parameters"))):
        return False
    params_any = safe_string_list(requirements.get("supported_parameters_any"))
    if params_any and not model_has_any(model, "supported_parameters", params_any):
        return False
    return safe_int(model.get("context_length"), 0) >= min_context


def competence_gate_result(
    profile_id: str,
    candidates: list[dict[str, Any]],
    *,
    is_multimodal_profile: Callable[[str], bool],
    is_specialized_profile: Callable[[str], bool],
    multimodal_route_eligible: Callable[[dict[str, Any]], bool],
    route_eligible_for_profile: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], bool]:
    """Return routed candidates plus whether anti-empty competence fallback was used."""
    if is_multimodal_profile(profile_id):
        is_eligible = multimodal_route_eligible
    elif is_specialized_profile(profile_id):
        is_eligible = route_eligible_for_profile
    else:
        return candidates, False

    eligible = [model for model in candidates if is_eligible(model)]
    return (eligible, False) if eligible else (candidates, bool(candidates))


def competence_label(
    profile_id: str,
    model: dict[str, Any],
    *,
    is_multimodal_profile: Callable[[str], bool],
    is_specialized_profile: Callable[[str], bool],
    multimodal_route_eligible: Callable[[dict[str, Any]], bool],
    multimodal_claimed_profiles: Callable[[dict[str, Any]], list[str]],
    profile_verified: Callable[[str, dict[str, Any]], bool],
    profile_failed: Callable[[str, dict[str, Any]], bool],
    capability_is_default_claimed: Callable[[str, dict[str, Any]], bool],
    capability_reference_routable: Callable[[str, dict[str, Any]], bool],
) -> str:
    if is_multimodal_profile(profile_id):
        if not multimodal_route_eligible(model):
            return "failed"
        if any(profile_verified(profile, model) for profile in multimodal_claimed_profiles(model)):
            return "verified"
        return "claimed_catalog"
    if not is_specialized_profile(profile_id):
        return "n/a"
    if profile_failed(profile_id, model):
        return "failed"
    if profile_verified(profile_id, model):
        return "verified"
    if not capability_is_default_claimed(profile_id, model):
        return "claimed_catalog"
    return "claimed_reference" if capability_reference_routable(profile_id, model) else "claimed_default"
