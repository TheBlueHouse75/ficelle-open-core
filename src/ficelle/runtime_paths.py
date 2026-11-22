from __future__ import annotations

import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


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
    spend_meter_path: Path
    spend_meter_lock_path: Path
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
        # Absolute here, at the single resolution point: service definitions and the active-home
        # pointer record this path, and a relative one is resolved by launchd/systemd, not the shell.
        ficelle_home = (
            Path(os.path.abspath(Path(explicit_ficelle_home).expanduser()))
            if explicit_ficelle_home
            else Path.home() / ".ficelle"
        )
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
            spend_meter_path=router_dir / "provider-spend.json",
            spend_meter_lock_path=router_dir / "provider-spend.lock",
            capability_discrepancy_log_path=router_dir / "logs" / "capability-discrepancies.jsonl",
            catalog_refresh_attempts_path=router_dir / "catalog-refresh-attempts.json",
            compression_store_path=router_dir / "compression.sqlite",
            credential_env_file=ficelle_home / ".env",
            ficelle_secrets_keychain=ficelle_home / "ficelle-secrets.keychain-db",
            capability_oracle_cache_path=router_dir / "capability_oracle.json",
            admin_assets_dir=admin_assets_dir,
        )


def validated_runtime_home(home: Path, *, create: bool = True) -> Path:
    """Return an absolute home safe to use as a Python child working directory."""
    home = Path(os.path.abspath(home.expanduser()))
    try:
        info = os.lstat(home)
    except FileNotFoundError:
        if not create:
            raise
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = os.lstat(home)
    if not stat.S_ISDIR(info.st_mode):
        raise OSError("Ficelle home must be a directory, not a symbolic link or file.")
    if hasattr(os, "getuid"):
        if info.st_uid != os.getuid():
            raise OSError("Ficelle home must be owned by the current user.")
        if info.st_mode & 0o022:
            raise OSError("Ficelle home must not be group- or other-writable.")
    return home


def absolute_python(python: str | Path) -> str:
    # Preserve a venv's interpreter symlink: resolving it selects the base runtime.
    value = str(Path(python).expanduser())
    return os.path.abspath(shutil.which(value) or value)


@contextmanager
def python_child_context(
    home: Path | None = None, *, environ: Mapping[str, str] | None = None,
) -> Iterator[tuple[str, dict[str, str]]]:
    """Keep Python imports out of the caller's directory, even before setup."""
    env = dict(os.environ if environ is None else environ)
    env["PYTHONSAFEPATH"] = "1"
    if home is not None:
        home = Path(os.path.abspath(home.expanduser()))
        env["FICELLE_HOME"] = str(home)
        try:
            cwd = validated_runtime_home(home, create=False)
        except FileNotFoundError:
            pass
        else:
            if os.access(cwd, os.R_OK | os.X_OK):
                yield str(cwd), env
                return
    with tempfile.TemporaryDirectory(prefix="ficelle-python-") as cwd:
        Path(cwd).chmod(0o700)
        yield str(Path(cwd).absolute()), env
