from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RuntimePaths:
    hermes_home: Path
    router_dir: Path
    catalog_path: Path
    state_path: Path
    config_path: Path
    route_log_path: Path
    admin_audit_log_path: Path
    state_lock_path: Path
    state_backup_dir: Path
    capability_discrepancy_log_path: Path
    compression_store_path: Path
    hermes_agent_dir: Path
    hermes_config_path: Path
    ficelle_home: Path
    credential_env_file: Path
    hermes_secrets_keychain: Path
    capability_oracle_cache_path: Path
    admin_assets_dir: Path | None = None

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        package_dir: Path | None = None,
    ) -> "RuntimePaths":
        source = os.environ if environ is None else environ
        hermes_home = Path(source.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
        router_dir = hermes_home / "ficelle"
        ficelle_home = Path(source.get("FICELLE_HOME") or hermes_home).expanduser()
        admin_assets_dir = package_dir / "assets" / "admin" if package_dir is not None else None
        return cls(
            hermes_home=hermes_home,
            router_dir=router_dir,
            catalog_path=router_dir / "catalog.json",
            state_path=router_dir / "state.json",
            config_path=router_dir / "config.json",
            route_log_path=router_dir / "logs" / "routes.jsonl",
            admin_audit_log_path=router_dir / "logs" / "admin-actions.jsonl",
            state_lock_path=router_dir / "state.lock",
            state_backup_dir=router_dir / "state-backups",
            capability_discrepancy_log_path=router_dir / "logs" / "capability-discrepancies.jsonl",
            compression_store_path=router_dir / "compression.sqlite",
            hermes_agent_dir=hermes_home / "hermes-agent",
            hermes_config_path=hermes_home / "config.yaml",
            ficelle_home=ficelle_home,
            credential_env_file=ficelle_home / ".env",
            hermes_secrets_keychain=ficelle_home / "hermes-secrets.keychain-db",
            capability_oracle_cache_path=router_dir / "capability_oracle.json",
            admin_assets_dir=admin_assets_dir,
        )
