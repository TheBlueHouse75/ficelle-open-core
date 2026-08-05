from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


def invokable_provider_sources(auth: Mapping[str, Any]) -> list[str]:
    """Providers this install can actually call.

    ``invokable`` is the same gate ``invoke_model`` applies before sending anything —
    a resolved key with a base URL, or the keyless-local exemption — so an empty list
    means every completion will fail with ``credentials unavailable``, whatever the
    catalog says. The reference providers' catalogs are readable without a key, so
    ``ficelle models`` cannot be used to answer this question.
    """
    return [str(source) for source, row in auth.items() if isinstance(row, dict) and row.get("invokable")]


def unconfigured_provider_sources(auth: Mapping[str, Any]) -> list[str]:
    """The providers a user would have to give a key to, in configured order."""
    invokable = set(invokable_provider_sources(auth))
    return [str(source) for source in auth if str(source) not in invokable]


def provider_key_setup_commands(sources: Sequence[str], key_urls: Mapping[str, str]) -> list[str]:
    """One ready-to-paste ``ficelle set-key`` line per provider, unindented.

    The key-creation URLs come from the caller's registry (``PROVIDER_KEY_URLS``)
    rather than from anything written here, so the closed pack's providers stay
    unnamed in the open core and no second copy of a URL can drift. A provider the
    registry does not know still gets its command — the command is the actionable
    half, the link is the convenience.
    """
    commands = [f"ficelle set-key {source}" for source in sources]
    width = max((len(command) for command in commands), default=0)
    lines: list[str] = []
    for source, command in zip(sources, commands):
        url = str(key_urls.get(source) or "").strip()
        lines.append(f"{command.ljust(width)}  # create a key at {url}" if url else command)
    return lines
