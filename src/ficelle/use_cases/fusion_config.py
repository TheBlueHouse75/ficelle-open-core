from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FusionConfigPorts:
    default_config: dict[str, Any]
    budget_policies: set[str]
    diversity_policies: set[str]
    raw_output_policies: set[str]
    judge_formats: set[str]
    fusion_model_id: str
    judge_max_attempts: int
    synthesizer_max_attempts: int
    is_virtual_profile_id: Callable[[str], bool]
    is_safe_virtual_profile_id: Callable[[str], bool]
    safe_detail: Callable[[Any], str]
    safe_int: Callable[[Any, int], int]


def fusion_bool(raw: dict[str, Any], key: str, default: bool, *, strict: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if strict:
        raise ValueError(f"fusion.{key} must be boolean")
    return default


def fusion_int(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    strict: bool,
) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        if strict:
            raise ValueError(f"fusion.{key} must be an integer")
        return default
    if strict and not isinstance(value, int):
        raise ValueError(f"fusion.{key} must be an integer")
    try:
        number = int(value)
    except Exception:
        if strict:
            raise ValueError(f"fusion.{key} must be an integer")
        return default
    if number < minimum or number > maximum:
        if strict:
            raise ValueError(f"fusion.{key} must be between {minimum} and {maximum}")
        return default
    return number


def fusion_choice(raw: dict[str, Any], key: str, default: str, allowed: set[str], *, strict: bool) -> str:
    raw_value = raw.get(key, default)
    if strict and (not isinstance(raw_value, str) or not raw_value.strip()):
        raise ValueError(f"fusion.{key} must be one of: {', '.join(sorted(allowed))}")
    value = str(raw_value or default).strip()
    if value in allowed:
        return value
    if strict:
        raise ValueError(f"fusion.{key} must be one of: {', '.join(sorted(allowed))}")
    return default


def fusion_profile_id(
    raw: dict[str, Any],
    key: str,
    default: str | None,
    *,
    ports: FusionConfigPorts,
    strict: bool,
    allowed_profile_ids: set[str] | None = None,
) -> str | None:
    value = raw.get(key, default)
    if value in (None, "") and default is None:
        return None
    profile_id = str(value or "").strip()
    if not profile_id:
        if strict:
            raise ValueError(f"fusion.{key} must not be blank")
        return default
    if profile_id == ports.fusion_model_id:
        if strict:
            raise ValueError(f"fusion.{key} cannot be {ports.fusion_model_id}")
        return default
    if not ports.is_virtual_profile_id(profile_id) or not ports.is_safe_virtual_profile_id(profile_id):
        detail = ports.safe_detail(profile_id) or "[redacted]"
        if strict:
            raise ValueError(f"fusion.{key} contains unsupported profile id: {detail}")
        return default
    if allowed_profile_ids is not None and profile_id not in allowed_profile_ids:
        detail = ports.safe_detail(profile_id) or "[redacted]"
        if strict:
            raise ValueError(f"fusion.{key} must reference a configured virtual profile: {detail}")
        return default
    return profile_id


def fusion_call_multiplier(config: dict[str, Any], ports: FusionConfigPorts) -> int:
    panel_size = ports.safe_int(config.get("panel_size"), ports.default_config["panel_size"])
    peer_ranking = config.get("peer_ranking") if isinstance(config.get("peer_ranking"), dict) else {}
    total = panel_size + ports.synthesizer_max_attempts + (1 if config.get("blind_draft") else 0)
    if not (peer_ranking.get("enabled") and peer_ranking.get("replaces_judge")):
        total += ports.judge_max_attempts
    if peer_ranking.get("enabled"):
        total += panel_size
    return total


def normalize_fusion_peer_ranking(raw: Any, *, ports: FusionConfigPorts, strict: bool) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    defaults = ports.default_config["peer_ranking"]
    return {
        "enabled": fusion_bool(source, "enabled", defaults["enabled"], strict=strict),
        "min_rankers": fusion_int(source, "min_rankers", defaults["min_rankers"], minimum=1, maximum=8, strict=strict),
        "replaces_judge": fusion_bool(source, "replaces_judge", defaults["replaces_judge"], strict=strict),
        "feed_synthesizer": fusion_bool(source, "feed_synthesizer", defaults["feed_synthesizer"], strict=strict),
        "emit_routing_signal": fusion_bool(source, "emit_routing_signal", defaults["emit_routing_signal"], strict=strict),
    }


def normalize_fusion_config(
    raw: Any,
    *,
    ports: FusionConfigPorts,
    strict: bool = False,
    allowed_profile_ids: set[str] | None = None,
) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    defaults = ports.default_config
    max_panel_size = fusion_int(source, "max_panel_size", defaults["max_panel_size"], minimum=1, maximum=8, strict=strict)
    panel_size = fusion_int(source, "panel_size", defaults["panel_size"], minimum=1, maximum=max_panel_size, strict=strict)
    min_success = fusion_int(
        source,
        "min_successful_panel_outputs",
        defaults["min_successful_panel_outputs"],
        minimum=1,
        maximum=panel_size,
        strict=strict,
    )
    normalized = {
        "enabled": fusion_bool(source, "enabled", defaults["enabled"], strict=strict),
        "visible_in_models": fusion_bool(source, "visible_in_models", defaults["visible_in_models"], strict=strict),
        "panel_profile": fusion_profile_id(source, "panel_profile", defaults["panel_profile"], ports=ports, strict=strict, allowed_profile_ids=allowed_profile_ids),
        "judge_profile": fusion_profile_id(source, "judge_profile", defaults["judge_profile"], ports=ports, strict=strict, allowed_profile_ids=allowed_profile_ids),
        "synthesizer_profile": fusion_profile_id(source, "synthesizer_profile", defaults["synthesizer_profile"], ports=ports, strict=strict, allowed_profile_ids=allowed_profile_ids),
        "draft_profile": fusion_profile_id(source, "draft_profile", defaults["draft_profile"], ports=ports, strict=strict, allowed_profile_ids=allowed_profile_ids),
        "panel_size": panel_size,
        "max_panel_size": max_panel_size,
        "min_successful_panel_outputs": min_success,
        "budget_policy": fusion_choice(source, "budget_policy", defaults["budget_policy"], ports.budget_policies, strict=strict),
        "max_total_calls": fusion_int(source, "max_total_calls", defaults["max_total_calls"], minimum=1, maximum=16, strict=strict),
        "per_call_timeout_seconds": fusion_int(source, "per_call_timeout_seconds", defaults["per_call_timeout_seconds"], minimum=1, maximum=300, strict=strict),
        "fusion_timeout_seconds": fusion_int(source, "fusion_timeout_seconds", defaults["fusion_timeout_seconds"], minimum=1, maximum=900, strict=strict),
        "diversity_policy": fusion_choice(source, "diversity_policy", defaults["diversity_policy"], ports.diversity_policies, strict=strict),
        "allow_single_panel_degrade": fusion_bool(source, "allow_single_panel_degrade", defaults["allow_single_panel_degrade"], strict=strict),
        "blind_draft": fusion_bool(source, "blind_draft", defaults["blind_draft"], strict=strict),
        "raw_output_policy": fusion_choice(source, "raw_output_policy", defaults["raw_output_policy"], ports.raw_output_policies, strict=strict),
        "anonymize_panel": fusion_bool(source, "anonymize_panel", defaults["anonymize_panel"], strict=strict),
        "judge_format": fusion_choice(source, "judge_format", defaults["judge_format"], ports.judge_formats, strict=strict),
        "peer_ranking": normalize_fusion_peer_ranking(source.get("peer_ranking"), ports=ports, strict=strict),
    }
    if normalized["fusion_timeout_seconds"] < normalized["per_call_timeout_seconds"]:
        if strict:
            raise ValueError("fusion.fusion_timeout_seconds must be >= fusion.per_call_timeout_seconds")
        normalized["fusion_timeout_seconds"] = max(defaults["fusion_timeout_seconds"], normalized["per_call_timeout_seconds"])
    if normalized["max_total_calls"] < fusion_call_multiplier(normalized, ports):
        if strict:
            raise ValueError("fusion.max_total_calls is too low for panel, judge, synthesizer, and draft settings")
        normalized["max_total_calls"] = max(defaults["max_total_calls"], fusion_call_multiplier(normalized, ports))
    if not normalized["enabled"]:
        normalized["visible_in_models"] = False
    return normalized


def fusion_visible_in_model_list(
    config: dict[str, Any],
    *,
    ports: FusionConfigPorts,
    configured_virtual_model_ids: Callable[[dict[str, Any]], list[str]],
) -> bool:
    fusion = normalize_fusion_config(
        config.get("fusion"),
        ports=ports,
        strict=False,
        allowed_profile_ids=set(configured_virtual_model_ids(config)),
    )
    return bool(fusion.get("enabled") and fusion.get("visible_in_models"))
