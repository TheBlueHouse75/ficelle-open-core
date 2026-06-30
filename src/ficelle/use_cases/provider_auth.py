from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ficelle.providers.base import ProviderAccess


@dataclass(frozen=True)
class ProviderAuthPorts:
    provider_access: Callable[[str, dict[str, Any], bool], ProviderAccess]
    credential_source_label: Callable[[str | None], str | None]


def provider_auth_row(source: str, config: dict[str, Any], *, ports: ProviderAuthPorts) -> dict[str, Any]:
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    provider_cfg = providers.get(source) if isinstance(providers.get(source), dict) else {}
    access = ports.provider_access(source, provider_cfg, False)
    configured = bool(access.key and access.base_url)
    invokable = configured or access.auth_status_invokable
    key_for_row = access.key if invokable else None
    base_url_for_row = access.base_url or provider_cfg.get("base_url")
    return auth_row(
        invokable,
        key_for_row,
        access.reason,
        base_url_for_row,
        credential_source_label=ports.credential_source_label,
    )


def auth_status(config: dict[str, Any], *, ports: ProviderAuthPorts) -> dict[str, dict[str, Any]]:
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    return {str(source): provider_auth_row(str(source), config, ports=ports) for source in providers}


def auth_row(
    invokable: bool,
    key: Any,
    reason: str,
    base_url: Any,
    *,
    credential_source_label: Callable[[str | None], str | None],
) -> dict[str, Any]:
    return {
        "invokable": invokable,
        "reason": "configured" if key else reason,
        "key_source": credential_source_label(reason) if key else None,
        "base_url": base_url,
    }
