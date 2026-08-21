from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CANONICAL_CREDENTIAL_FILENAMES = frozenset({".env", "ficelle-secrets.keychain-db"})
LEGACY_CREDENTIAL_FILENAMES = frozenset({"hermes-secrets.keychain-db"})
FICELLE_CREDENTIAL_FILENAMES = (
    CANONICAL_CREDENTIAL_FILENAMES | LEGACY_CREDENTIAL_FILENAMES
)


@dataclass(frozen=True)
class RuntimePaths:
    hermes_home: Path
    ficelle_home: Path
    router_dir: Path
    runtime_read_dir: Path
    catalog_path: Path
    state_path: Path
    config_path: Path
    route_log_path: Path
    request_log_store_path: Path
    admin_audit_log_path: Path
    state_lock_path: Path
    state_backup_dir: Path
    capability_discrepancy_log_path: Path
    catalog_refresh_attempts_path: Path
    compression_store_path: Path
    hermes_agent_dir: Path
    hermes_config_path: Path
    credential_env_file: Path
    legacy_credential_env_file: Path
    ficelle_secrets_keychain: Path
    legacy_ficelle_secrets_keychain: Path
    legacy_hermes_secrets_keychain: Path
    capability_oracle_cache_path: Path
    coding_certification_cache_path: Path
    coding_certification_status_path: Path
    admin_assets_dir: Path | None = None

    def read_path(self, canonical_path: Path) -> Path:
        """Resolve one canonical runtime path through the read-only compatibility root."""
        try:
            relative_path = canonical_path.relative_to(self.router_dir)
        except ValueError:
            return canonical_path
        if ".." in relative_path.parts:
            return canonical_path
        if canonical_path.exists() or self.runtime_read_dir == self.router_dir:
            return canonical_path
        fallback_path = self.runtime_read_dir / relative_path
        return fallback_path if fallback_path.exists() else canonical_path

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        package_dir: Path | None = None,
    ) -> "RuntimePaths":
        source = os.environ if environ is None else environ
        hermes_home = Path(source.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
        legacy_router_dir = hermes_home / "ficelle"
        explicit_ficelle_home = source.get("FICELLE_HOME")
        ficelle_home = Path(explicit_ficelle_home).expanduser() if explicit_ficelle_home else Path.home() / ".ficelle"
        explicit_runtime_dir = source.get("FICELLE_RUNTIME_DIR") or None
        if explicit_runtime_dir:
            runtime_read_dir = Path(explicit_runtime_dir).expanduser()
        elif explicit_ficelle_home is None and legacy_router_dir.is_dir():
            runtime_read_dir = legacy_router_dir
        else:
            runtime_read_dir = ficelle_home
        router_dir = ficelle_home
        admin_assets_dir = package_dir / "assets" / "admin" if package_dir is not None else None
        return cls(
            hermes_home=hermes_home,
            ficelle_home=ficelle_home,
            router_dir=router_dir,
            runtime_read_dir=runtime_read_dir,
            catalog_path=router_dir / "catalog.json",
            state_path=router_dir / "state.json",
            config_path=router_dir / "config.json",
            route_log_path=router_dir / "logs" / "routes.jsonl",
            request_log_store_path=router_dir / "requests.sqlite",
            admin_audit_log_path=router_dir / "logs" / "admin-actions.jsonl",
            state_lock_path=router_dir / "state.lock",
            state_backup_dir=router_dir / "state-backups",
            capability_discrepancy_log_path=router_dir / "logs" / "capability-discrepancies.jsonl",
            catalog_refresh_attempts_path=router_dir / "catalog-refresh-attempts.json",
            compression_store_path=router_dir / "compression.sqlite",
            hermes_agent_dir=hermes_home / "hermes-agent",
            hermes_config_path=hermes_home / "config.yaml",
            credential_env_file=ficelle_home / ".env",
            legacy_credential_env_file=hermes_home / ".env",
            ficelle_secrets_keychain=ficelle_home / "ficelle-secrets.keychain-db",
            legacy_ficelle_secrets_keychain=ficelle_home / "hermes-secrets.keychain-db",
            legacy_hermes_secrets_keychain=hermes_home / "hermes-secrets.keychain-db",
            capability_oracle_cache_path=router_dir / "capability_oracle.json",
            coding_certification_cache_path=router_dir / "coding-certifications.json",
            coding_certification_status_path=router_dir / "coding-certification-status.json",
            admin_assets_dir=admin_assets_dir,
        )
