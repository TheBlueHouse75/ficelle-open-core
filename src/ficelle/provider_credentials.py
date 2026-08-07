from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ficelle.providers.base import ProviderAccess


ProviderAccessResult = Callable[[str, dict[str, Any]], ProviderAccess]


class ProviderCredentialsUnavailable(RuntimeError):
    """No usable credential for a provider: the request was never sent upstream.

    A named class rather than a bare ``RuntimeError`` because an attempt record keeps
    the exception's *class name* (``error_type``) and drops its message, so this is the
    only signal a downstream reader — the failover demo — can key on without
    string-matching prose that a later edit would silently invalidate. It stays a
    ``RuntimeError`` subclass so every existing handler keeps catching it.
    """


class ProviderSecretStore(Protocol):
    """One backend, owning every operation on both of its tiers.

    The legacy tier is three operations, not one: a backend that can *read* a migration
    secret must also be able to say it holds one and to clear it. They live together here
    because splitting them is what makes a key readable, unreported and un-removable —
    resolution asks the backend, so a host-side purge wired to one platform's shape cannot
    follow a second backend that learns to read.
    """

    label: str

    def get(self, service: str) -> str | None:
        ...

    def get_legacy(self, service: str) -> str | None:
        """The read-only migration tier, for a backend that has one; else ``None``."""
        ...

    def probe_legacy(self, services: Sequence[str]) -> list[str]:
        """Redacted labels for the legacy entries holding one of ``services``; ``[]`` for a
        backend with no legacy tier."""
        ...

    def delete_legacy(self, services: Sequence[str]) -> list[str]:
        """Clear those entries; return the redacted labels actually cleared, ``[]`` for a
        backend with no legacy tier."""
        ...

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


@dataclass(frozen=True)
class CredentialLocation:
    """One place a provider key can live, carrying the operations that must agree on it.

    Declaring a location once, with all of them attached, is what makes the class of bug
    fixed on 2026-08-06 structurally impossible: a place resolution can *read* can no
    longer be a place nothing reports or clears, and every operation walks the same alias
    list instead of re-deriving its own.

    - ``read`` yields ``(secret, label)`` for each alias that holds a value, in precedence
      order. It is a generator, so a location further down the chain costs nothing when an
      earlier one already won.
    - ``probe`` returns the redacted labels that hold a value, and by default derives them
      from ``read`` so the two cannot drift. A location overrides it when it can answer
      presence more cheaply or more safely — the legacy keychains do, because listing them
      with ``security find-generic-password`` and no ``-w`` never pulls a secret at all.
    - ``delete`` returns the redacted labels it actually cleared, or is ``None`` for a
      place Ficelle may read but must not write. Every removal path skips those.

    No label ever contains key material, whichever operation produced it.
    """

    kind: str
    legacy: bool
    read: Callable[[], Iterator[tuple[str, str]]]
    probe: Callable[[], list[str]]
    delete: Callable[[], list[str]] | None

    @property
    def removable(self) -> bool:
        return self.delete is not None


@dataclass(frozen=True)
class CredentialLocationPorts:
    """Host-side bindings for the credential locations: real files, stores and keychains."""

    env_get: Callable[[str], str | None]
    parse_env_file: Callable[[Path], dict[str, str]]
    env_file_delete_key: Callable[[Path, str], bool]
    credential_env_file: Path
    legacy_credential_env_files: tuple[Path, ...]
    # Both tiers of the OS store, canonical and legacy, come from this one object. The
    # legacy tier used to arrive split: read through ``store.get_legacy``, but listed and
    # cleared through host callables wired to the macOS keychain shape. That split is what
    # made a second backend learning to read a legacy secret produce a key that resolves,
    # goes unreported and cannot be purged — the purge could not follow the store because
    # it was not asking the store.
    store: ProviderSecretStore


def _alias_reader(
    aliases: Sequence[str],
    label: Callable[[str], str],
    open_lookup: Callable[[], Callable[[str], str | None]],
) -> Callable[[], Iterator[tuple[str, str]]]:
    """A ``read`` over interchangeable alias names, opened once per traversal.

    ``open_lookup`` is called when the walk reaches this location, not when the registry is
    built, and its result serves every alias — which is what keeps a ``.env`` location to
    one file parse per walk while staying fresh after a delete.
    """

    def read() -> Iterator[tuple[str, str]]:
        lookup = open_lookup()
        for alias in aliases:
            value = lookup(alias)
            if value:
                yield value, label(alias)

    return read


def _alias_location(
    kind: str,
    aliases: Sequence[str],
    label: Callable[[str], str],
    open_lookup: Callable[[], Callable[[str], str | None]],
    *,
    legacy: bool = False,
    delete_alias: Callable[[str], bool] | None = None,
) -> CredentialLocation:
    """A location addressed by a list of aliases resolution accepts interchangeably.

    Read, probe and delete are built from the one ``aliases`` list, so an alias reachable
    by resolution is reachable by removal by construction — the second 2026-08-06 bug was
    exactly a removal that stopped at ``aliases[0]``.
    """
    read = _alias_reader(aliases, label, open_lookup)
    return CredentialLocation(
        kind=kind,
        legacy=legacy,
        read=read,
        probe=lambda: [source for _value, source in read()],
        delete=(
            None
            if delete_alias is None
            else lambda: [label(alias) for alias in aliases if delete_alias(alias)]
        ),
    )


def _env_file_location(
    path: Path,
    env_names: Sequence[str],
    ports: CredentialLocationPorts,
    *,
    legacy: bool,
) -> CredentialLocation:
    return _alias_location(
        "env_file",
        env_names,
        lambda env_name: f"{path}:{env_name}",
        lambda: ports.parse_env_file(path).get,
        legacy=legacy,
        delete_alias=lambda env_name: ports.env_file_delete_key(path, env_name),
    )


def provider_credential_locations(
    env_names: Sequence[str],
    services: Sequence[str],
    *,
    ports: CredentialLocationPorts,
) -> tuple[CredentialLocation, ...]:
    """Every place a provider key can live, in resolution precedence order.

    The order is load-bearing and must not change: process environment, canonical ``.env``,
    canonical OS store, legacy ``.env`` files, legacy OS store. It decides which copy of a
    key wins when several exist.
    """
    store = ports.store
    return (
        _alias_location(
            "process_env",
            env_names,
            lambda env_name: f"env:{env_name}",
            lambda: ports.env_get,
            # No ``delete_alias``: a process environment variable belongs to whoever
            # exported it, and Ficelle cannot unset it for that session.
        ),
        _env_file_location(ports.credential_env_file, env_names, ports, legacy=False),
        _alias_location(
            "store",
            services,
            lambda service: f"{store.label}:{service}",
            lambda: store.get,
            # Bound late, like every other operation here: building the registry must not
            # require a capability the traversal about to run will never use.
            delete_alias=lambda service: store.delete(service),
        ),
        *(
            _env_file_location(path, env_names, ports, legacy=True)
            for path in ports.legacy_credential_env_files
        ),
        CredentialLocation(
            kind="legacy_store",
            legacy=True,
            read=_alias_reader(
                services,
                lambda service: f"{store.label}:{service}",
                lambda: store.get_legacy,
            ),
            # Listing and clearing are the backend's too, and take the whole alias list at
            # once because a backend may address the tier as more than one place — the
            # macOS keychain is one entry per service *per keychain file*, and lists them
            # without ever reading a value. So these labels can name that file, which a
            # ``read`` label cannot: they are the removal vocabulary, not resolution's
            # ``key_source``.
            probe=lambda: store.probe_legacy(services),
            delete=lambda: store.delete_legacy(services),
        ),
    )


def _delete_credential_locations(locations: Sequence[CredentialLocation]) -> list[str]:
    """Clear every removable location, reporting the OS store before the rest.

    That is not the order resolution reads them in, but it is the order the CLI line and
    the admin toast have always listed; ``sorted`` is stable, so everything else keeps its
    registry position.
    """
    ordered = sorted(locations, key=lambda location: location.kind != "store")
    return [label for location in ordered if location.delete for label in location.delete()]


def legacy_credential_sources(locations: Sequence[CredentialLocation]) -> list[str]:
    """Redacted labels for the legacy locations that still hold a key."""
    return [label for location in locations if location.legacy for label in location.probe()]


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


def remove_provider_key(locations: Sequence[CredentialLocation]) -> list[str]:
    """Clear a provider's key from every canonical location, and only those.

    Walking the shared registry is what keeps removal aligned with resolution: each
    location clears the same alias list it reads, so a key stored under a secondary name
    can no longer be resolved and un-removable. The read-only legacy fallbacks are
    deliberately left alone — another install may still own them — and are cleared only by
    the explicit opt-in ``purge_legacy_credentials``.
    """
    return _delete_credential_locations([location for location in locations if not location.legacy])


def purge_legacy_credentials(locations: Sequence[CredentialLocation]) -> list[str]:
    """Delete a provider's key from the read-only legacy fallbacks.

    Resolution reads legacy keychains and ``.env`` files as a migration convenience, so a
    key that lives *only* there is used by the router but is out of reach of the ordinary
    removal. This is the explicit opt-in escape hatch, reached only from a confirmed
    removal, because it writes to files another install may still own.
    """
    return _delete_credential_locations([location for location in locations if location.legacy])


def resolve_provider_access(
    source: str,
    config: dict[str, Any],
    provider_access_result: ProviderAccessResult,
) -> ProviderAccess:
    """The whole access record for a configured provider, carried rather than unpacked.

    Callers used to get a ``(key, base_url, reason)`` triplet here and rebuild the
    invocability verdict from it — the shape both 2026-08-06 divergences took. The record
    answers that itself (``ProviderAccess.can_invoke``), so it is what crosses this
    boundary; a caller that only wants the triplet can still read the three fields.
    """
    providers = config.get("providers")
    provider_cfg = providers.get(source) if isinstance(providers, dict) else None
    if not isinstance(provider_cfg, dict):
        return ProviderAccess(None, None, f"unknown provider {source}")
    return provider_access_result(source, provider_cfg)
