from __future__ import annotations

from typing import Any

from ficelle.use_cases.catalog_fingerprint import (
    CatalogFingerprintPorts,
    catalog_config_fingerprint,
    catalog_config_structural_fingerprint,
    provider_config_fingerprint,
)


def ports(calls: list[tuple[str, dict[str, Any]]]) -> CatalogFingerprintPorts:
    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                redacted[str(key)] = "[redacted]" if "key" in str(key).lower() else redact(item)
            return redacted
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    def credential_activation_fingerprint(source: str, provider_cfg: dict[str, Any]) -> dict[str, Any]:
        calls.append((source, provider_cfg))
        return {"env_configured": bool(provider_cfg.get("api_key_env"))}

    return CatalogFingerprintPorts(
        safe_int=lambda value, default: int(value) if value is not None else default,
        redact_sensitive_json=redact,
        credential_activation_fingerprint=credential_activation_fingerprint,
    )


def test_catalog_config_fingerprint_tracks_structural_provider_fields() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    base = {
        "min_context_length": "128000",
        "allow_paid_fallback": True,
        "providers": {
            "openrouter": {
                "enabled": True,
                "catalog_url": "https://openrouter.example/models",
                "base_url": "https://openrouter.example/v1",
                "activation_policy": "always",
                "model_id_allowlist": ["free-a"],
            }
        },
    }
    changed = {
        **base,
        "providers": {
            "openrouter": {
                **base["providers"]["openrouter"],
                "model_id_allowlist": ["free-b"],
            }
        },
    }

    assert catalog_config_fingerprint(base, ports=ports(calls)) != catalog_config_fingerprint(
        changed,
        ports=ports(calls),
    )

    required_allowlist = {
        **base,
        "providers": {
            "openrouter": {
                **base["providers"]["openrouter"],
                "require_model_id_allowlist": True,
            }
        },
    }
    assert catalog_config_structural_fingerprint(base, ports=ports(calls)) != (
        catalog_config_structural_fingerprint(required_allowlist, ports=ports(calls))
    )

    flagged_catalog = {
        **base,
        "providers": {
            "openrouter": {
                **base["providers"]["openrouter"],
                "catalog_free_flag_field": "isFree",
            }
        },
    }
    assert catalog_config_structural_fingerprint(base, ports=ports(calls)) != (
        catalog_config_structural_fingerprint(flagged_catalog, ports=ports(calls))
    )


def test_structural_fingerprint_excludes_credential_activation_state() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    config = {
        "providers": {
            "nvidia": {
                "activation_policy": "configured_credentials",
                "api_key_env": "NVIDIA_API_KEY",
            }
        }
    }
    full = catalog_config_fingerprint(config, ports=ports(calls))
    structural = catalog_config_structural_fingerprint(config, ports=ports(calls))

    assert full != structural
    assert calls == [("nvidia", config["providers"]["nvidia"])]


def test_provider_config_fingerprint_includes_credentials_only_when_policy_requires_it() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    configured_provider = {"activation_policy": "configured_credentials", "api_key_env": "NVIDIA_API_KEY"}
    always_provider = {"activation_policy": "always", "api_key_env": "OPENROUTER_API_KEY"}

    configured = provider_config_fingerprint(
        "nvidia",
        configured_provider,
        ports=ports(calls),
        include_credentials=True,
    )
    always = provider_config_fingerprint(
        "openrouter",
        always_provider,
        ports=ports(calls),
        include_credentials=True,
    )
    structural = provider_config_fingerprint(
        "nvidia",
        configured_provider,
        ports=ports(calls),
        include_credentials=False,
    )

    assert configured["credential_activation"] == {"env_configured": True}
    assert "credential_activation" not in always
    assert "credential_activation" not in structural
    assert calls == [("nvidia", configured_provider)]
