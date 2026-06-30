from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CatalogFingerprintPorts:
    safe_int: Callable[[Any, int], int]
    redact_sensitive_json: Callable[[Any], Any]
    credential_activation_fingerprint: Callable[[str, dict[str, Any]], dict[str, Any]]


def catalog_config_fingerprint(
    config: dict[str, Any],
    *,
    ports: CatalogFingerprintPorts,
    include_credentials: bool = True,
) -> str:
    providers: dict[str, Any] = {}
    raw_providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    for source, raw_provider in raw_providers.items():
        if not isinstance(raw_provider, dict):
            continue
        provider_fingerprint = provider_config_fingerprint(
            str(source),
            raw_provider,
            ports=ports,
            include_credentials=include_credentials,
        )
        providers[str(source)] = provider_fingerprint
    payload = {
        "min_context_length": ports.safe_int(config.get("min_context_length"), 0),
        "allow_paid_fallback": bool(config.get("allow_paid_fallback")),
        "providers": providers,
    }
    serialized = json.dumps(
        ports.redact_sensitive_json(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def catalog_config_structural_fingerprint(config: dict[str, Any], *, ports: CatalogFingerprintPorts) -> str:
    return catalog_config_fingerprint(config, ports=ports, include_credentials=False)


def provider_config_fingerprint(
    source: str,
    provider_cfg: dict[str, Any],
    *,
    ports: CatalogFingerprintPorts,
    include_credentials: bool,
) -> dict[str, Any]:
    fingerprint = {
        "enabled": provider_cfg.get("enabled", True),
        "catalog_url": provider_cfg.get("catalog_url"),
        "base_url": provider_cfg.get("base_url"),
        "source_type": provider_cfg.get("source_type"),
        "provider_class": provider_cfg.get("provider_class"),
        "activation_policy": provider_cfg.get("activation_policy"),
        "quota_reset_policy": provider_cfg.get("quota_reset_policy"),
        "free_mode": provider_cfg.get("free_mode"),
        "free_scope": provider_cfg.get("free_scope"),
        "free_status": provider_cfg.get("free_status"),
        "free_access_proof": provider_cfg.get("free_access_proof"),
        "free_note": provider_cfg.get("free_note"),
        "catalog_model_defaults": provider_cfg.get("catalog_model_defaults"),
        "model_id_exclude_patterns": provider_cfg.get("model_id_exclude_patterns"),
        "model_id_allowlist": provider_cfg.get("model_id_allowlist"),
    }
    if include_credentials and str(provider_cfg.get("activation_policy") or "") == "configured_credentials":
        fingerprint["credential_activation"] = ports.credential_activation_fingerprint(source, provider_cfg)
    return fingerprint
