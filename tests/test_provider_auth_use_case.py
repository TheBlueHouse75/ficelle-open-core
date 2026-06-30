from __future__ import annotations

from typing import Any

from ficelle.providers.base import ProviderAccess
from ficelle.use_cases.provider_auth import (
    ProviderAuthPorts,
    auth_row,
    auth_status,
    provider_auth_row,
)


def ports_for(
    access_by_source: dict[str, ProviderAccess],
    calls: list[tuple[str, dict[str, Any], bool]],
) -> ProviderAuthPorts:
    def provider_access(source: str, provider_cfg: dict[str, Any], require_base_url: bool) -> ProviderAccess:
        calls.append((source, provider_cfg, require_base_url))
        return access_by_source[source]

    return ProviderAuthPorts(
        provider_access=provider_access,
        credential_source_label=lambda reason: {
            "env:NVIDIA_API_KEY": "env",
            "dotenv:NVIDIA_API_KEY": ".env",
        }.get(str(reason)),
    )


def test_auth_row_redacts_key_and_maps_key_source() -> None:
    row = auth_row(
        True,
        "secret-key",
        "env:NVIDIA_API_KEY",
        "https://nvidia.example/v1",
        credential_source_label=lambda reason: "env" if reason == "env:NVIDIA_API_KEY" else None,
    )

    assert row == {
        "invokable": True,
        "reason": "configured",
        "key_source": "env",
        "base_url": "https://nvidia.example/v1",
    }
    assert "secret-key" not in repr(row)


def test_provider_auth_row_requires_key_and_base_url_unless_adapter_marks_invokable() -> None:
    calls: list[tuple[str, dict[str, Any], bool]] = []
    config = {
        "providers": {
            "nvidia": {"base_url": "https://nvidia.example/v1"},
            "openrouter": {},
            "localai": {"base_url": "http://127.0.0.1:8080/v1"},
        }
    }
    ports = ports_for(
        {
            "nvidia": ProviderAccess("key", "https://nvidia.example/v1", "env:NVIDIA_API_KEY"),
            "openrouter": ProviderAccess(
                "key-only",
                None,
                "env:OPENROUTER_API_KEY",
                auth_status_invokable=True,
            ),
            "localai": ProviderAccess(
                None,
                "http://127.0.0.1:8080/v1",
                "keyless_local",
                auth_status_invokable=True,
            ),
        },
        calls,
    )

    assert provider_auth_row("nvidia", config, ports=ports) == {
        "invokable": True,
        "reason": "configured",
        "key_source": "env",
        "base_url": "https://nvidia.example/v1",
    }
    assert provider_auth_row("openrouter", config, ports=ports)["invokable"] is True
    local = provider_auth_row("localai", config, ports=ports)
    assert local["invokable"] is True
    assert local["reason"] == "keyless_local"
    assert local["key_source"] is None
    assert calls[0] == ("nvidia", {"base_url": "https://nvidia.example/v1"}, False)


def test_auth_status_preserves_rows_for_each_provider_entry() -> None:
    calls: list[tuple[str, dict[str, Any], bool]] = []
    config = {"providers": {"nvidia": {"base_url": "https://nvidia.example/v1"}, "missing": "bad"}}
    ports = ports_for(
        {
            "nvidia": ProviderAccess(None, None, "missing NVIDIA_API_KEY"),
            "missing": ProviderAccess(None, None, "missing provider"),
        },
        calls,
    )

    status = auth_status(config, ports=ports)

    assert status["nvidia"] == {
        "invokable": False,
        "reason": "missing NVIDIA_API_KEY",
        "key_source": None,
        "base_url": "https://nvidia.example/v1",
    }
    assert status["missing"]["reason"] == "missing provider"
