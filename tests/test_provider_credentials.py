from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ficelle.provider_credentials import (
    PROVIDER_KEY_VALIDATORS,
    generic_provider_credential_aliases,
    generic_provider_credential_activation_fingerprint,
    is_usable_openrouter_key,
    provider_primary_service,
    remove_provider_key,
    resolve_provider_credentials,
    store_provider_key,
)
from ficelle.providers.base import ProviderAccess


def test_resolve_provider_credentials_uses_provider_access_port():
    calls: list[tuple[str, dict[str, Any]]] = []

    def provider_access_result(source: str, provider_cfg: dict[str, Any]) -> ProviderAccess:
        calls.append((source, provider_cfg))
        return ProviderAccess(
            key="provider-key",
            base_url="https://provider.example/v1",
            reason="env:PROVIDER_API_KEY",
            auth_status_invokable=True,
        )

    key, base_url, reason = resolve_provider_credentials(
        "example",
        {"providers": {"example": {"api_key_env": "PROVIDER_API_KEY"}}},
        provider_access_result,
    )

    assert (key, base_url, reason) == (
        "provider-key",
        "https://provider.example/v1",
        "env:PROVIDER_API_KEY",
    )
    assert calls == [("example", {"api_key_env": "PROVIDER_API_KEY"})]


def test_resolve_provider_credentials_reports_unknown_provider():
    key, base_url, reason = resolve_provider_credentials(
        "missing",
        {"providers": {"example": {}}},
        lambda _source, _provider_cfg: ProviderAccess(
            key="unused",
            base_url="https://unused.example/v1",
            reason="unused",
            auth_status_invokable=True,
        ),
    )

    assert key is None
    assert base_url is None
    assert reason == "unknown provider missing"


PROVIDER_CREDENTIAL_ALIAS_EXPECTATIONS = {
    "nvidia": (
        ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NIM_API_KEY"],
        [
            "NVIDIA_API_KEY",
            "nvidia",
            "Nvidia",
            "nvidia-api-key",
            "NVIDIA_NIM_API_KEY",
            "NIM_API_KEY",
            "NVIDIA",
            "NVIDIA_NIM",
            "NIM",
            "nvidia-nim-api-key",
            "nim-api-key",
        ],
    ),
    "groq": (["GROQ_API_KEY"], ["GROQ_API_KEY", "groq", "Groq", "groq-api-key"]),
    "gemini": (["GEMINI_API_KEY"], ["GEMINI_API_KEY", "gemini", "Gemini", "gemini-api-key"]),
    "openrouter": (
        ["OPENROUTER_API_KEY"],
        ["OPENROUTER_API_KEY", "openrouter", "OpenRouter", "openrouter-api-key", "OPENROUTER"],
    ),
    "mistral": (
        ["MISTRAL_API_KEY"],
        ["MISTRAL_API_KEY", "mistral", "Mistral", "mistral-api-key", "MISTRAL"],
    ),
    "nous": (["NOUS_API_KEY"], ["NOUS_API_KEY", "nous", "Nous", "nous-api-key", "NOUS"]),
}


@pytest.mark.parametrize("provider", sorted(PROVIDER_CREDENTIAL_ALIAS_EXPECTATIONS))
def test_provider_credential_aliases_are_preserved_in_provider_credentials_module(provider: str) -> None:
    provider_cfg = {"api_key_env": f"{provider.upper()}_API_KEY"}

    env_names, services = generic_provider_credential_aliases(provider, provider_cfg)

    assert (env_names, services) == PROVIDER_CREDENTIAL_ALIAS_EXPECTATIONS[provider]


def test_openrouter_validator_is_registered_in_provider_credentials_module() -> None:
    valid = "sk-or-v1-" + "a" * 40

    assert is_usable_openrouter_key(valid) is True
    assert is_usable_openrouter_key("not-a-valid-key") is False
    assert PROVIDER_KEY_VALIDATORS["openrouter"](valid) is True
    assert PROVIDER_KEY_VALIDATORS.get("groq") is None


def test_provider_credential_activation_fingerprint_reports_configured_sources(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    legacy_env_file = tmp_path / "legacy.env"
    keychain = tmp_path / "login.keychain-db"
    missing_keychain = tmp_path / "missing.keychain-db"
    env_file.write_text("CANONICAL_ONLY=1\n")
    legacy_env_file.write_text("OPENROUTER_API_KEY=env-file-key\n")
    keychain.write_text("keychain-state")

    fingerprint = generic_provider_credential_activation_fingerprint(
        "openrouter",
        {"api_key_env": "OPENROUTER_API_KEY"},
        env_get=lambda env_name: "env-key" if env_name == "OPENROUTER_API_KEY" else None,
        parse_env_file=lambda path: (
            {"OPENROUTER_API_KEY": "env-file-key"}
            if path == legacy_env_file
            else {}
        ),
        credential_env_files=(env_file, legacy_env_file),
        keychain_paths=(missing_keychain, keychain),
    )

    assert fingerprint["env_configured"] is True
    assert fingerprint["env_file_configured"] is True
    assert fingerprint["env_file_state"] == [
        f"0:{env_file.name}:{env_file.stat().st_mtime_ns}:{env_file.stat().st_size}",
        f"1:{legacy_env_file.name}:{legacy_env_file.stat().st_mtime_ns}:{legacy_env_file.stat().st_size}",
    ]
    expected_keychain_state = f"{keychain.name}:{keychain.stat().st_mtime_ns}:{keychain.stat().st_size}"
    assert fingerprint["keychain_state"] == [expected_keychain_state]


class WritableStore:
    label = "keychain"

    def __init__(self, *, can_write: bool = True, can_delete: bool = True) -> None:
        self.can_write = can_write
        self.can_delete = can_delete
        self.writes: dict[str, str] = {}
        self.deletes: list[str] = []

    def set(self, service: str, secret: str) -> bool:
        if not self.can_write:
            return False
        self.writes[service] = secret
        return True

    def delete(self, service: str) -> bool:
        self.deletes.append(service)
        return self.can_delete


def test_store_provider_key_prefers_store_and_uses_primary_service(tmp_path: Path) -> None:
    store = WritableStore()
    calls: list[tuple[Path, str, str]] = []
    config = {"providers": {"openrouter": {"api_key_env": "OPENROUTER_API_KEY"}}}

    target = store_provider_key(
        "openrouter",
        config,
        "secret",
        store=store,
        credential_env_file=tmp_path / ".env",
        env_file_set_key=lambda path, key, value: calls.append((path, key, value)),
    )

    assert provider_primary_service("openrouter", config) == "OPENROUTER_API_KEY"
    assert target == "keychain:OPENROUTER_API_KEY"
    assert store.writes == {"OPENROUTER_API_KEY": "secret"}
    assert calls == []


def test_store_provider_key_falls_back_to_env_file_when_store_fails(tmp_path: Path) -> None:
    store = WritableStore(can_write=False)
    calls: list[tuple[Path, str, str]] = []
    env_file = tmp_path / ".env"

    target = store_provider_key(
        "groq",
        {"providers": {"groq": {"api_key_env": "GROQ_API_KEY"}}},
        "secret",
        store=store,
        credential_env_file=env_file,
        env_file_set_key=lambda path, key, value: calls.append((path, key, value)),
    )

    assert target == f"{env_file}:GROQ_API_KEY (plaintext)"
    assert calls == [(env_file, "GROQ_API_KEY", "secret")]


def test_remove_provider_key_clears_store_and_env_file(tmp_path: Path) -> None:
    store = WritableStore()
    env_file = tmp_path / ".env"
    deleted_env: list[tuple[Path, str]] = []

    cleared = remove_provider_key(
        "openrouter",
        {"providers": {"openrouter": {"api_key_env": "OPENROUTER_API_KEY"}}},
        store=store,
        credential_env_file=env_file,
        env_file_delete_key=lambda path, key: deleted_env.append((path, key)) or True,
    )

    assert store.deletes == ["OPENROUTER_API_KEY"]
    assert cleared == ["keychain:OPENROUTER_API_KEY", f"{env_file}:OPENROUTER_API_KEY"]
    assert deleted_env == [(env_file, "OPENROUTER_API_KEY")]
