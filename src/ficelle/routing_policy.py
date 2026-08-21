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


def catalog_denies_input_modality(
    model: dict[str, Any],
    requirements: dict[str, Any],
    *,
    modalities_may_come_from_defaults: bool = False,
) -> bool:
    """True when the model declares its input modalities and the one a profile needs is missing.

    Why probing it anyway is pointless, structurally rather than by measurement:
    `model_matches_profile_requirements` filters this same field HARD when routing, and no verdict
    overrides it — so a `verified` won here could never be routed. An empty list, or a value that
    may have been synthesized from provider defaults, means "not declared", so an unknown model is
    never denied. Read from the live catalog row on every cycle, never persisted, so a model that
    gains a modality is due again on the next one.

    See `docs/components/router.md` § Automatic capability discovery for why discovery honours this
    for modalities only.
    """
    required_all = safe_string_list(requirements.get("input_modalities"))
    # Only `auto-multimodal` requires "any of", and it has no probe body today, so that branch is
    # unreachable from discovery. Kept so the two filters cannot drift apart.
    required_any = safe_string_list(requirements.get("input_modalities_any"))
    # Asked first: most profiles need no modality at all, and then the model never has to be read.
    if not (required_all or required_any):
        return False
    if modalities_may_come_from_defaults or not safe_string_list(model.get("input_modalities")):
        return False
    if required_all and not model_has_all(model, "input_modalities", required_all):
        return True
    return bool(required_any) and not model_has_any(model, "input_modalities", required_any)


def catalog_denies_structured_output(
    model: dict[str, Any],
    requirements: dict[str, Any],
    *,
    structured_support_may_come_from_defaults: bool = False,
) -> bool:
    """True when the model declares no structured-output support and the profile requires it.

    Same argument as `catalog_denies_input_modality`, same hard counterpart:
    `model_matches_profile_requirements` rejects on this very field when `structured` is required,
    and no verdict overrides it — so a `verified` won here could never be routed. Only an explicit
    `False` denies: a missing field means "not declared", and whether the row itself answered —
    the flag can be computed off defaults-inherited params (see the caller) — arrives through
    `structured_support_may_come_from_defaults`. Only the `structured: True` direction is read —
    the only one a probeable profile can carry, since built-in requirements overwrite user config
    for the three profiles that set it; a user `structured: false` on another built-in profile would
    need the opposite provenance guard (a defaults-inflated `True`) and buys nothing today.
    """
    if requirements.get("structured") is not True:
        return False
    if structured_support_may_come_from_defaults:
        return False
    return model.get("supports_structured_outputs") is False


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


def profile_capability_metadata(requirements: dict[str, Any]) -> dict[str, Any]:
    """What a virtual model guarantees about whichever candidate ends up answering.

    Read off the profile's requirements, not off today's candidates, because the
    requirements are a hard contract: `select_models_result_from_typed_rows` applies
    `model_matches_profile_requirements` BEFORE the manual_order/auto branch, both the
    manual list and the auto tail are built from that filtered set, and the competence
    gate's anti-empty fallback re-serves an already-filtered list. A model failing these
    can therefore never be routed here — so these are promises, not observations that
    change with cooldowns.

    `structured` is reported only when the profile requires it. `None` means "not
    required", not "not supported", and publishing `false` there would talk callers out
    of a capability the routed model most likely has. Same reason `context_length` is the
    required minimum: it is the floor the slot guarantees, while the model that answers
    usually offers more.
    """
    input_modalities = sorted(
        {
            *safe_string_list(requirements.get("input_modalities")),
            *safe_string_list(requirements.get("input_modalities_any")),
        }
    )
    supported_parameters = sorted(
        {
            *safe_string_list(requirements.get("supported_parameters")),
            *safe_string_list(requirements.get("supported_parameters_any")),
        }
    )
    # Same default as the routing filter above, so the two cannot drift apart.
    tools = bool(requirements.get("tools", True))
    structured = requirements.get("structured") is True

    capabilities = ["completion"]
    if tools:
        capabilities.append("tools")
    if structured:
        capabilities.append("structured")
    # "vision" rather than "image": the name every client that reads this field uses.
    if "image" in input_modalities:
        capabilities.append("vision")
    capabilities.extend(modality for modality in ("audio", "video") if modality in input_modalities)

    metadata: dict[str, Any] = {
        "context_length": safe_int(requirements.get("min_context"), 0),
        "supports_tools": tools,
        "supports_streaming": True,
        "capabilities": capabilities,
    }
    if structured:
        metadata["supports_structured_outputs"] = True
    if input_modalities:
        metadata["input_modalities"] = input_modalities
    if supported_parameters:
        metadata["supported_parameters"] = supported_parameters
    return metadata


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
