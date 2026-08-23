from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


FICELLE_CREDENTIAL_FILENAMES = frozenset({".env", "ficelle-secrets.keychain-db"})


@dataclass(frozen=True)
class RuntimePaths:
    ficelle_home: Path
    router_dir: Path
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
    credential_env_file: Path
    ficelle_secrets_keychain: Path
    capability_oracle_cache_path: Path
    admin_assets_dir: Path | None = None

    def read_path(self, canonical_path: Path) -> Path:
        """Return a canonical Ficelle-owned runtime path."""
        return canonical_path

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        package_dir: Path | None = None,
    ) -> "RuntimePaths":
        source = os.environ if environ is None else environ
        explicit_ficelle_home = source.get("FICELLE_HOME")
        ficelle_home = Path(explicit_ficelle_home).expanduser() if explicit_ficelle_home else Path.home() / ".ficelle"
        router_dir = ficelle_home
        admin_assets_dir = package_dir / "assets" / "admin" if package_dir is not None else None
        return cls(
            ficelle_home=ficelle_home,
            router_dir=router_dir,
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
            credential_env_file=ficelle_home / ".env",
            ficelle_secrets_keychain=ficelle_home / "ficelle-secrets.keychain-db",
            capability_oracle_cache_path=router_dir / "capability_oracle.json",
            admin_assets_dir=admin_assets_dir,
        )
