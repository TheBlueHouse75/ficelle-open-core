from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ficelle.provider_credentials import (
    PROVIDER_KEY_VALIDATORS,
    CredentialLocation,
    CredentialLocationPorts,
    generic_provider_credential_aliases,
    generic_provider_credential_activation_fingerprint,
    is_usable_openrouter_key,
    legacy_credential_sources,
    provider_credential_locations,
    provider_primary_service,
    purge_legacy_credentials,
    remove_provider_key,
    resolve_provider_access,
    store_provider_key,
)
from ficelle.providers.base import ProviderAccess


def test_resolve_provider_access_returns_the_port_record_untouched():
    """The whole record crosses the boundary, verdict included — not a triplet to rebuild it from."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def provider_access_result(source: str, provider_cfg: dict[str, Any]) -> ProviderAccess:
        calls.append((source, provider_cfg))
        return ProviderAccess(
            key="provider-key",
            base_url="https://provider.example/v1",
            reason="env:PROVIDER_API_KEY",
            auth_status_invokable=True,
        )

    access = resolve_provider_access(
        "example",
        {"providers": {"example": {"api_key_env": "PROVIDER_API_KEY"}}},
        provider_access_result,
    )

    assert (access.key, access.base_url, access.reason) == (
        "provider-key",
        "https://provider.example/v1",
        "env:PROVIDER_API_KEY",
    )
    assert access.auth_status_invokable is True
    assert access.can_invoke is True
    assert calls == [("example", {"api_key_env": "PROVIDER_API_KEY"})]


def test_resolve_provider_access_reports_unknown_provider():
    access = resolve_provider_access(
        "missing",
        {"providers": {"example": {}}},
        lambda _source, _provider_cfg: ProviderAccess(
            key="unused",
            base_url="https://unused.example/v1",
            reason="unused",
            auth_status_invokable=True,
        ),
    )

    assert access.key is None
    assert access.base_url is None
    assert access.reason == "unknown provider missing"
    assert access.can_invoke is False


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
        self.stored: dict[str, str] = {}
        self.legacy: dict[str, str] = {}

    def get(self, service: str) -> str | None:
        return self.stored.get(service)

    def get_legacy(self, service: str) -> str | None:
        return self.legacy.get(service)

    # A backend whose legacy tier is a second place, not just a second lookup — the macOS
    # keychain shape. Its labels name that place, which is why they are not the read labels.
    def probe_legacy(self, services) -> list[str]:
        return [f"keychain:legacy:{service}" for service in services if service in self.legacy]

    def delete_legacy(self, services) -> list[str]:
        return [
            f"keychain:legacy:{service}"
            for service in services
            if self.legacy.pop(service, None) is not None
        ]

    def set(self, service: str, secret: str) -> bool:
        if not self.can_write:
            return False
        self.writes[service] = secret
        return True

    def delete(self, service: str) -> bool:
        # Honest about absence, like every real backend: True only when an item was removed.
        self.deletes.append(service)
        return self.can_delete and self.stored.pop(service, None) is not None


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


class _Host:
    """In-memory stand-in for the host bindings the router supplies to the registry."""

    def __init__(self, tmp_path: Path, store: WritableStore | None = None) -> None:
        self.env: dict[str, str] = {}
        self.env_file = tmp_path / ".env"
        self.legacy_env_file = tmp_path / "legacy.env"
        self.files: dict[Path, dict[str, str]] = {self.env_file: {}, self.legacy_env_file: {}}
        # Taken at construction, not assigned afterwards: `legacy_store` aliases the store's
        # own dict, and a later `host.store = ...` would leave it pointing at the discarded one.
        self.store = store if store is not None else WritableStore()
        self.legacy_store: dict[str, str] = self.store.legacy
        self.deleted_env: list[tuple[Path, str]] = []

    def _delete_env_key(self, path: Path, key: str) -> bool:
        self.deleted_env.append((path, key))
        return self.files[path].pop(key, None) is not None

    def ports(self) -> CredentialLocationPorts:
        return CredentialLocationPorts(
            env_get=self.env.get,
            parse_env_file=lambda path: dict(self.files[path]),
            env_file_delete_key=self._delete_env_key,
            credential_env_file=self.env_file,
            legacy_credential_env_files=(self.legacy_env_file,),
            store=self.store,
        )

    def locations(self, provider: str) -> tuple[CredentialLocation, ...]:
        env_names, services = generic_provider_credential_aliases(
            provider, {"api_key_env": f"{provider.upper()}_API_KEY"}
        )
        return provider_credential_locations(env_names, services, ports=self.ports())


def test_credential_locations_follow_the_locked_resolution_order(tmp_path: Path) -> None:
    """Which copy of a key wins is decided by this order alone; it must not drift."""
    locations = _Host(tmp_path).locations("openrouter")

    assert [(location.kind, location.legacy) for location in locations] == [
        ("process_env", False),
        ("env_file", False),
        ("store", False),
        ("env_file", True),
        ("legacy_store", True),
    ]


def test_every_location_reports_and_clears_exactly_what_it_reads(tmp_path: Path) -> None:
    """The invariant the registry exists for: no place is readable but unreportable or
    unclearable, and no operation on a location sees a different alias list than the
    others. Both 2026-08-06 bugs were a read path reaching further than its write path."""
    host = _Host(tmp_path)
    # A secondary alias throughout: clearing only the primary one is exactly how a key
    # stayed resolvable with nothing left to remove.
    host.env["NIM_API_KEY"] = "from-process-env"
    host.files[host.env_file]["NVIDIA_NIM_API_KEY"] = "from-canonical-env-file"
    host.store.stored["nvidia-api-key"] = "from-canonical-store"
    host.files[host.legacy_env_file]["NIM_API_KEY"] = "from-legacy-env-file"
    host.legacy_store["nim-api-key"] = "from-legacy-store"

    for location in host.locations("nvidia"):
        read = list(location.read())
        probed = location.probe()
        assert read, f"{location.kind} holds a key but reads nothing"
        assert probed, f"{location.kind} holds a key but reports nothing"
        assert not any(secret in label for secret, _ in read for label in probed)

        if location.delete is None:
            assert list(location.read()), "an unremovable location must stay readable"
            continue
        assert location.delete() == probed, f"{location.kind} clears something other than it reports"
        assert list(location.read()) == []
        assert location.probe() == []


class _PortableLegacyStore(WritableStore):
    """A backend whose legacy tier is a second lookup, not a second keychain file.

    Nothing macOS-shaped: what `SecretToolStore` or `WindowsCredentialStore` would look like
    the day either learns to read a migration secret. It exists to prove that day needs no
    host-side rewiring — the registry asks the store for all three legacy operations, so the
    purge follows the backend instead of one platform's shape. Only the label and the two
    listing/clearing shapes differ from the base double; everything else is a store like
    any other, which is the point.
    """

    label = "secret-service"

    def probe_legacy(self, services) -> list[str]:
        return [f"{self.label}:{service}" for service in services if service in self.legacy]

    def delete_legacy(self, services) -> list[str]:
        return [
            f"{self.label}:{service}"
            for service in services
            if self.legacy.pop(service, None) is not None
        ]


def test_a_non_keychain_legacy_tier_is_reported_and_purged_without_host_rewiring(tmp_path: Path) -> None:
    """The debt this closes: readable, unreported, un-purgeable on any backend but macOS.

    Listing and clearing the legacy tier used to be host callables addressing macOS keychain
    files, while the read went through the store — so a key a second backend could read was
    resolvable, absent from the remaining-sources line, and untouched by `--purge-legacy`.
    Nothing here is macOS, and it is reported and cleared like any other place.
    """
    store = _PortableLegacyStore()
    store.legacy["nvidia-api-key"] = "portable-legacy-secret-value"
    locations = _Host(tmp_path, store=store).locations("nvidia")

    reported = legacy_credential_sources(locations)
    cleared = purge_legacy_credentials(locations)

    # Pinned to the literal rather than checked for absence of the secret: equality to a
    # label that plainly holds none is the stronger statement of the same thing.
    assert reported == ["secret-service:nvidia-api-key"]
    assert cleared == reported
    assert legacy_credential_sources(locations) == []
    assert store.legacy == {}


def test_process_environment_is_the_only_location_ficelle_cannot_clear(tmp_path: Path) -> None:
    locations = _Host(tmp_path).locations("openrouter")

    assert [location.kind for location in locations if not location.removable] == ["process_env"]


def test_remove_provider_key_clears_store_and_env_file(tmp_path: Path) -> None:
    host = _Host(tmp_path)
    host.store.stored["OPENROUTER_API_KEY"] = "sk-or-canonical"
    host.files[host.env_file]["OPENROUTER_API_KEY"] = "sk-or-canonical"

    cleared = remove_provider_key(host.locations("openrouter"))

    # Store before `.env`, which is not resolution order — it is the order the CLI line and
    # the admin toast have always listed, so it stays pinned rather than left to the sort.
    assert cleared == [
        "keychain:OPENROUTER_API_KEY",
        f"{host.env_file}:OPENROUTER_API_KEY",
    ]
    assert host.store.deletes[0] == "OPENROUTER_API_KEY"
    assert (host.env_file, "OPENROUTER_API_KEY") in host.deleted_env


def test_remove_provider_key_clears_secondary_aliases(tmp_path: Path) -> None:
    """Resolution accepts any alias, so removal has to clear all of them.

    NVIDIA is the case that shows it: a key written as ``NVIDIA_NIM_API_KEY`` — the name its
    own docs use — was resolved but never removed, leaving the provider configured with
    nothing left to delete.
    """
    host = _Host(tmp_path)
    host.files[host.env_file]["NVIDIA_NIM_API_KEY"] = "nv-secondary"

    cleared = remove_provider_key(host.locations("nvidia"))

    assert "NVIDIA_NIM_API_KEY" in host.store.deletes
    assert f"{host.env_file}:NVIDIA_NIM_API_KEY" in cleared
    assert (host.env_file, "NIM_API_KEY") in host.deleted_env


def test_remove_provider_key_never_touches_the_legacy_fallbacks(tmp_path: Path) -> None:
    """A legacy store may still be owned by another install, so only the opt-in purge
    clears it — but the purge must then reach everything resolution can read there."""
    host = _Host(tmp_path)
    host.files[host.legacy_env_file]["OPENROUTER_API_KEY"] = "sk-or-legacy-file"
    host.legacy_store["openrouter-api-key"] = "sk-or-legacy-store"

    assert remove_provider_key(host.locations("openrouter")) == []
    assert host.files[host.legacy_env_file] == {"OPENROUTER_API_KEY": "sk-or-legacy-file"}
    assert legacy_credential_sources(host.locations("openrouter")) == [
        f"{host.legacy_env_file}:OPENROUTER_API_KEY",
        "keychain:legacy:openrouter-api-key",
    ]

    purged = purge_legacy_credentials(host.locations("openrouter"))

    assert purged == [
        f"{host.legacy_env_file}:OPENROUTER_API_KEY",
        "keychain:legacy:openrouter-api-key",
    ]
    assert legacy_credential_sources(host.locations("openrouter")) == []
