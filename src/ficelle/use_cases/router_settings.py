from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


NormalizeCompressionSettings = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class RouterSettingsPolicy:
    defaults: dict[str, Any]
    cooldown_reasons: Sequence[str]
    default_quota_probe_backoff_seconds: Sequence[int]
    cooldown_min_seconds: int
    cooldown_max_seconds: int
    backoff_max_steps: int
    verified_capability_ttl_min_seconds: int
    verified_capability_ttl_max_seconds: int
    auto_benchmark_interval_min_seconds: int
    auto_benchmark_interval_max_seconds: int
    auto_benchmark_models_min_per_cycle: int
    auto_benchmark_models_max_per_cycle: int
    benchmark_long_context_min_tokens: int
    benchmark_long_context_max_tokens: int
    benchmark_compression_min_chars: int
    benchmark_compression_max_chars: int
    normalize_compression_settings: NormalizeCompressionSettings


@dataclass(frozen=True)
class RouterSettingsSavePorts:
    update_config: Callable[[Callable[[dict[str, Any]], None]], dict[str, Any]]
    apply_verified_capability_ttl: Callable[[dict[str, Any]], int]
    apply_route_on_capability_reference: Callable[[dict[str, Any]], bool]


def settings_int(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    strict: bool,
    label: str | None = None,
) -> int:
    field = f"settings.{label or key}"
    value = raw.get(key, default)
    if isinstance(value, bool):
        if strict:
            raise ValueError(f"{field} must be an integer")
        return default
    if strict and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    try:
        number = int(value)
    except Exception:
        if strict:
            raise ValueError(f"{field} must be an integer")
        return default
    if number < minimum or number > maximum:
        if strict:
            raise ValueError(f"{field} must be between {minimum} and {maximum}")
        return default
    return number


def settings_bool(raw: dict[str, Any], key: str, default: bool, *, strict: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if strict:
        raise ValueError(f"settings.{key} must be a boolean")
    return default


def normalize_settings_backoff(raw: Any, *, policy: RouterSettingsPolicy, strict: bool) -> list[int]:
    defaults = list(policy.default_quota_probe_backoff_seconds)
    if raw is None:
        return defaults
    if not isinstance(raw, list):
        if strict:
            raise ValueError("settings.quota_probe_backoff_seconds must be a list of seconds")
        return defaults
    values: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            if strict:
                raise ValueError("settings.quota_probe_backoff_seconds entries must be integers")
            continue
        try:
            number = int(item)
        except Exception:
            if strict:
                raise ValueError("settings.quota_probe_backoff_seconds entries must be integers")
            continue
        if number < 1 or number > policy.cooldown_max_seconds:
            if strict:
                raise ValueError(
                    f"settings.quota_probe_backoff_seconds entries must be between 1 and {policy.cooldown_max_seconds}"
                )
            continue
        values.append(number)
    if not values:
        if strict:
            raise ValueError("settings.quota_probe_backoff_seconds must have at least one positive value")
        return defaults
    if len(values) > policy.backoff_max_steps:
        if strict:
            raise ValueError(
                f"settings.quota_probe_backoff_seconds cannot have more than {policy.backoff_max_steps} steps"
            )
        values = values[: policy.backoff_max_steps]
    return values


def normalize_router_settings(raw: Any, *, policy: RouterSettingsPolicy, strict: bool = False) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    defaults = policy.defaults
    raw_cooldowns = source.get("cooldown_seconds") if isinstance(source.get("cooldown_seconds"), dict) else {}
    cooldown_seconds = {
        reason: settings_int(
            raw_cooldowns,
            reason,
            defaults["cooldown_seconds"][reason],
            minimum=policy.cooldown_min_seconds,
            maximum=policy.cooldown_max_seconds,
            strict=strict,
            label=f"cooldown_seconds.{reason}",
        )
        for reason in policy.cooldown_reasons
    }
    return {
        "max_attempts_per_request": settings_int(
            source, "max_attempts_per_request", defaults["max_attempts_per_request"], minimum=0, maximum=12, strict=strict
        ),
        "catalog_timeout_seconds": settings_int(
            source, "catalog_timeout_seconds", defaults["catalog_timeout_seconds"], minimum=5, maximum=120, strict=strict
        ),
        "quota_probe_timeout_seconds": settings_int(
            source, "quota_probe_timeout_seconds", defaults["quota_probe_timeout_seconds"], minimum=1, maximum=60, strict=strict
        ),
        "verified_capability_ttl_seconds": settings_int(
            source,
            "verified_capability_ttl_seconds",
            defaults["verified_capability_ttl_seconds"],
            minimum=policy.verified_capability_ttl_min_seconds,
            maximum=policy.verified_capability_ttl_max_seconds,
            strict=strict,
        ),
        "route_on_capability_reference": settings_bool(
            source, "route_on_capability_reference", defaults["route_on_capability_reference"], strict=strict
        ),
        "auto_benchmark_enabled": settings_bool(
            source, "auto_benchmark_enabled", defaults["auto_benchmark_enabled"], strict=strict
        ),
        "auto_benchmark_interval_seconds": settings_int(
            source,
            "auto_benchmark_interval_seconds",
            defaults["auto_benchmark_interval_seconds"],
            minimum=policy.auto_benchmark_interval_min_seconds,
            maximum=policy.auto_benchmark_interval_max_seconds,
            strict=strict,
        ),
        "auto_benchmark_models_per_cycle": settings_int(
            source,
            "auto_benchmark_models_per_cycle",
            defaults["auto_benchmark_models_per_cycle"],
            minimum=policy.auto_benchmark_models_min_per_cycle,
            maximum=policy.auto_benchmark_models_max_per_cycle,
            strict=strict,
        ),
        "benchmark_long_context_tokens": settings_int(
            source,
            "benchmark_long_context_tokens",
            defaults["benchmark_long_context_tokens"],
            minimum=policy.benchmark_long_context_min_tokens,
            maximum=policy.benchmark_long_context_max_tokens,
            strict=strict,
        ),
        "benchmark_compression_chars": settings_int(
            source,
            "benchmark_compression_chars",
            defaults["benchmark_compression_chars"],
            minimum=policy.benchmark_compression_min_chars,
            maximum=policy.benchmark_compression_max_chars,
            strict=strict,
        ),
        "cooldown_seconds": cooldown_seconds,
        "quota_probe_backoff_seconds": normalize_settings_backoff(
            source.get("quota_probe_backoff_seconds"),
            policy=policy,
            strict=strict,
        ),
        "compression": policy.normalize_compression_settings(source.get("compression"), strict=strict),
    }


def validate_settings_payload(payload: dict[str, Any], *, policy: RouterSettingsPolicy) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("settings must be an object")
    raw = payload.get("settings", payload.get("config", payload))
    if not isinstance(raw, dict):
        raise ValueError("settings must be an object")
    return normalize_router_settings(raw, policy=policy, strict=True)


def router_settings_payload(config: dict[str, Any], *, policy: RouterSettingsPolicy) -> dict[str, Any]:
    return {"config": normalize_router_settings(config, policy=policy, strict=False)}


def verified_capability_ttl_seconds(config: dict[str, Any], *, policy: RouterSettingsPolicy) -> int:
    return settings_int(
        config if isinstance(config, dict) else {},
        "verified_capability_ttl_seconds",
        policy.defaults["verified_capability_ttl_seconds"],
        minimum=policy.verified_capability_ttl_min_seconds,
        maximum=policy.verified_capability_ttl_max_seconds,
        strict=False,
    )


def route_on_capability_reference(config: dict[str, Any], *, policy: RouterSettingsPolicy) -> bool:
    return settings_bool(
        config if isinstance(config, dict) else {},
        "route_on_capability_reference",
        policy.defaults["route_on_capability_reference"],
        strict=False,
    )


def auto_benchmark_enabled(config: dict[str, Any], *, policy: RouterSettingsPolicy) -> bool:
    return settings_bool(
        config if isinstance(config, dict) else {},
        "auto_benchmark_enabled",
        policy.defaults["auto_benchmark_enabled"],
        strict=False,
    )


def auto_benchmark_interval_seconds(config: dict[str, Any], *, policy: RouterSettingsPolicy) -> int:
    return settings_int(
        config if isinstance(config, dict) else {},
        "auto_benchmark_interval_seconds",
        policy.defaults["auto_benchmark_interval_seconds"],
        minimum=policy.auto_benchmark_interval_min_seconds,
        maximum=policy.auto_benchmark_interval_max_seconds,
        strict=False,
    )


def auto_benchmark_models_per_cycle(config: dict[str, Any], *, policy: RouterSettingsPolicy) -> int:
    return settings_int(
        config if isinstance(config, dict) else {},
        "auto_benchmark_models_per_cycle",
        policy.defaults["auto_benchmark_models_per_cycle"],
        minimum=policy.auto_benchmark_models_min_per_cycle,
        maximum=policy.auto_benchmark_models_max_per_cycle,
        strict=False,
    )


def save_router_settings(
    config: dict[str, Any],
    settings: dict[str, Any],
    *,
    policy: RouterSettingsPolicy,
    ports: RouterSettingsSavePorts,
) -> dict[str, Any]:
    normalized = normalize_router_settings(settings, policy=policy, strict=True)

    def apply_settings(disk_config: dict[str, Any]) -> None:
        for key, value in normalized.items():
            disk_config[key] = value

    ports.update_config(apply_settings)

    config["allow_paid_fallback"] = False
    for key, value in normalized.items():
        config[key] = value
    ports.apply_verified_capability_ttl(config)
    ports.apply_route_on_capability_reference(config)
    return normalized
