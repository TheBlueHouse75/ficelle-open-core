from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ficelle.providers.base import ProviderAccess


ProviderAccessResult = Callable[[str, dict[str, Any]], ProviderAccess]


class ProviderSecretStore(Protocol):
    label: str

    def set(self, service: str, secret: str) -> bool:
        ...

    def delete(self, service: str) -> bool:
        ...


def is_usable_openrouter_key(value: Any) -> bool:
    key = str(value or "").strip()
    return key.startswith("sk-or-") and len(key) >= 20


# Per-provider credential resolution config as data (R8): a provider's extra env-var
# names, keychain service names, and key-format validator are configuration, not new
# per-provider code. Each list preserves the provider's historically accepted names.
# Providers absent from PROVIDER_SERVICE_ALIASES fall back to the generic
# [source, source.title(), "<source>-api-key"] pattern.
PROVIDER_ENV_ALIASES: dict[str, list[str]] = {
    "nvidia": ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NIM_API_KEY"],
}
PROVIDER_SERVICE_ALIASES: dict[str, list[str]] = {
    "openrouter": ["openrouter", "OpenRouter", "openrouter-api-key", "OPENROUTER"],
    "nous": ["nous", "Nous", "nous-api-key", "NOUS"],
    "mistral": ["mistral", "Mistral", "mistral-api-key", "MISTRAL"],
    "nvidia": [
        "nvidia", "Nvidia", "nvidia-api-key", "NVIDIA_NIM_API_KEY", "NIM_API_KEY",
        "NVIDIA", "NVIDIA_NIM", "NIM", "nvidia-nim-api-key", "nim-api-key",
    ],
}
PROVIDER_KEY_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "openrouter": is_usable_openrouter_key,
}


def generic_provider_credential_aliases(source: str, provider_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    env_name = str(provider_cfg.get("api_key_env") or f"{source.upper()}_API_KEY")
    env_names = [env_name]
    for alias in PROVIDER_ENV_ALIASES.get(source, []):
        if alias not in env_names:
            env_names.append(alias)
    service_aliases = PROVIDER_SERVICE_ALIASES.get(source) or [source, source.title(), f"{source}-api-key"]
    services = [env_name]
    for alias in service_aliases:
        if alias not in services:
            services.append(alias)
    return env_names, services


def generic_provider_credential_activation_fingerprint(
    source: str,
    provider_cfg: dict[str, Any],
    *,
    env_get: Callable[[str], str | None],
    parse_env_file: Callable[[Path], dict[str, str]],
    credential_env_files: tuple[Path, ...],
    keychain_paths: tuple[Path, ...],
) -> dict[str, Any]:
    env_names, _services = generic_provider_credential_aliases(source, provider_cfg)
    env_configured = any(bool(env_get(env_name)) for env_name in env_names)
    env_file_configured = False
    env_file_state: list[str] = []
    for index, env_file in enumerate(credential_env_files):
        env_values = parse_env_file(env_file)
        env_file_configured = env_file_configured or any(
            bool(env_values.get(env_name)) for env_name in env_names
        )
        try:
            stat = env_file.stat()
        except OSError:
            continue
        env_file_state.append(
            f"{index}:{env_file.name}:{stat.st_mtime_ns}:{stat.st_size}"
        )
    keychain_state: list[str] = []
    for keychain_path in keychain_paths:
        try:
            stat = keychain_path.stat()
        except OSError:
            continue
        keychain_state.append(f"{keychain_path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return {
        "env_configured": env_configured,
        "env_file_configured": env_file_configured,
        "env_file_state": env_file_state,
        "keychain_state": keychain_state,
    }


def provider_primary_service(source: str, config: dict[str, Any]) -> str:
    provider_cfg = (config.get("providers") or {}).get(source) or {}
    env_names, _services = generic_provider_credential_aliases(source, provider_cfg)
    return env_names[0]


def store_provider_key(
    source: str,
    config: dict[str, Any],
    secret: str,
    *,
    store: ProviderSecretStore,
    credential_env_file: Path,
    env_file_set_key: Callable[[Path, str, str], None],
) -> str:
    service = provider_primary_service(source, config)
    if store.set(service, secret):
        return f"{store.label}:{service}"
    env_file_set_key(credential_env_file, service, secret)
    return f"{credential_env_file}:{service} (plaintext)"


def remove_provider_key(
    source: str,
    config: dict[str, Any],
    *,
    store: ProviderSecretStore,
    credential_env_file: Path,
    env_file_delete_key: Callable[[Path, str], bool],
) -> list[str]:
    service = provider_primary_service(source, config)
    cleared: list[str] = []
    if store.delete(service):
        cleared.append(f"{store.label}:{service}")
    if env_file_delete_key(credential_env_file, service):
        cleared.append(f"{credential_env_file}:{service}")
    return cleared


def resolve_provider_credentials(
    source: str,
    config: dict[str, Any],
    provider_access_result: ProviderAccessResult,
) -> tuple[str | None, str | None, str]:
    providers = config.get("providers")
    provider_cfg = providers.get(source) if isinstance(providers, dict) else None
    if not isinstance(provider_cfg, dict):
        return None, None, f"unknown provider {source}"
    access = provider_access_result(source, provider_cfg)
    return access.key, access.base_url, access.reason
