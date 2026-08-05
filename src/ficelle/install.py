"""User-proof installer for Ficelle local deployments."""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from ficelle.runtime_paths import FICELLE_CREDENTIAL_FILENAMES, RuntimePaths
from ficelle.service import active_home_pointer_path, persist_active_service_context
from ficelle.use_cases.provider_auth import (
    invokable_provider_sources,
    provider_key_setup_commands,
    unconfigured_provider_sources,
)

MANAGED_CONFIG_BEGIN = "# BEGIN FICELLE MANAGED CONFIG"
MANAGED_CONFIG_END = "# END FICELLE MANAGED CONFIG"
PROVIDER_PLUGIN_NAME = "ficelle"
COMPRESSION_PLUGIN_NAME = "ficelle-compression"
COMPRESSION_TOOLSET_NAME = "ficelle"
LEGACY_MIGRATION_MARKER = ".ficelle-legacy-migrated"
INSTALL_TARGETS = ("auto", "generic", "hermes", "openclaw")
InstallTarget = Literal["auto", "generic", "hermes", "openclaw"]
ResolvedInstallTarget = Literal["generic", "hermes", "openclaw"]
ManagedServiceStatus = Literal["active", "inactive", "unknown"]


@dataclass(frozen=True)
class InstallOptions:
    package: str
    editable: bool
    python: str
    ficelle_home: Path
    hermes_home: Path
    dry_run: bool
    skip_package: bool
    skip_plugin: bool
    skip_service: bool
    skip_smoke: bool
    preflight_only: bool
    configure_hermes: bool
    backup_existing: bool
    target: InstallTarget = "auto"
    rollback: bool = False
    ficelle_home_explicit: bool = True


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str
    action: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def default_hermes_home() -> Path:
    return Path.home() / ".hermes"


def default_ficelle_home() -> Path:
    return RuntimePaths.from_env().ficelle_home


def candidate_hermes_pythons(
    hermes_home: Path,
    selected_python: str | Path,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.getenv("HERMES_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            hermes_home / "hermes-agent" / "venv" / "bin" / "python",
            hermes_home / "venv" / "bin" / "python",
            Path.home() / ".local" / "share" / "hermes" / "venv" / "bin" / "python",
            Path(sys.executable),
        )
    )
    selected_value = str(selected_python)
    candidates.append(
        Path(shutil.which(selected_value) or selected_value).expanduser()
    )
    return tuple(dict.fromkeys(candidates))


def run_python_probe(python: str | Path, code: str) -> CommandResult:
    command = [str(python), "-c", code]
    try:
        completed = subprocess.run(command, text=True, capture_output=True)
    except OSError as exc:
        return CommandResult(command, 1, stderr=f"{type(exc).__name__}: {exc}")
    return CommandResult(
        command,
        completed.returncode,
        completed.stdout or "",
        completed.stderr or "",
    )


def probe_hermes_runtime(python: Path) -> CommandResult:
    code = (
        "import importlib.util; "
        "raise SystemExit(0 if importlib.util.find_spec('hermes_cli') is not None else 1)"
    )
    return run_python_probe(python, code)


def hermes_installation_signal(options: InstallOptions) -> Path | None:
    for candidate in candidate_hermes_pythons(
        options.hermes_home,
        options.python,
    ):
        if (
            candidate.is_file()
            and os.access(candidate, os.X_OK)
            and probe_hermes_runtime(candidate).returncode == 0
        ):
            return candidate
    hermes_cli = shutil.which("hermes")
    if hermes_cli:
        return Path(hermes_cli)
    config_path = options.hermes_home / "config.yaml"
    if config_path.is_file():
        return config_path
    agent_path = options.hermes_home / "hermes-agent"
    if agent_path.is_dir():
        return agent_path
    return None


def resolve_install_target(options: InstallOptions) -> ResolvedInstallTarget:
    if options.target != "auto":
        return options.target
    return "hermes" if hermes_installation_signal(options) is not None else "generic"


def recover_interrupted_legacy_migration(
    destination: Path,
    *,
    dry_run: bool,
) -> bool:
    """Restore credentials and remove staging left by an interrupted migration."""
    parent = destination.parent
    artifact_prefix = f".{destination.name.lstrip('.')}"
    previous_paths = sorted(
        parent.glob(f"{artifact_prefix}.pre-migration-*"),
        key=lambda path: path.name,
    )
    staging_paths = tuple(
        parent.glob(f"{artifact_prefix}.migration-*")
    )
    promotion_marker = destination / LEGACY_MIGRATION_MARKER
    recovered_promotion = destination.exists() and bool(previous_paths)
    promotion_completed = destination.exists() and (
        recovered_promotion or promotion_marker.is_file()
    )
    if not destination.exists() and previous_paths:
        latest_previous = previous_paths.pop()
        if dry_run:
            print(
                "DRY RUN: recover interrupted Ficelle migration "
                f"{latest_previous} -> {destination}"
            )
        else:
            latest_previous.rename(destination)
            print(f"Recovered interrupted Ficelle migration: {destination}")
    for path in (*previous_paths, *staging_paths):
        if dry_run:
            print(f"DRY RUN: remove interrupted migration artifact {path}")
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    if recovered_promotion:
        message = f"Recovered completed Ficelle migration at {destination}"
        print(f"DRY RUN: {message}" if dry_run else message)
    return promotion_completed


def migrate_legacy_ficelle_home(
    options: InstallOptions,
    *,
    before_copy: Callable[[], None] | None = None,
) -> bool:
    if options.ficelle_home_explicit:
        return False
    legacy_home = options.hermes_home / "ficelle"
    destination = options.ficelle_home
    if not legacy_home.is_dir():
        return False
    if recover_interrupted_legacy_migration(
        destination,
        dry_run=options.dry_run,
    ):
        return True
    existing_credentials: frozenset[str] = frozenset()
    if destination.exists():
        if not destination.is_dir():
            return False
        entries = tuple(destination.iterdir())
        if any(
            entry.name not in FICELLE_CREDENTIAL_FILENAMES or not entry.is_file()
            for entry in entries
        ):
            return False
        existing_credentials = frozenset(entry.name for entry in entries)
    if before_copy is not None:
        before_copy()
    if options.dry_run:
        print(f"DRY RUN: migrate Ficelle state {legacy_home} -> {destination} (source preserved)")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = timestamp_suffix()
    artifact_prefix = f".{destination.name.lstrip('.')}"
    staging = destination.with_name(f"{artifact_prefix}.migration-{suffix}")
    previous = destination.with_name(
        f"{artifact_prefix}.pre-migration-{suffix}"
    )
    moved_previous = False
    try:
        shutil.copytree(legacy_home, staging)
        for name in existing_credentials:
            shutil.copy2(destination / name, staging / name)
        (staging / LEGACY_MIGRATION_MARKER).write_text(
            "migrated\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rename(previous)
            moved_previous = True
        staging.rename(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if moved_previous and previous.exists() and not destination.exists():
            previous.rename(destination)
        raise
    if moved_previous:
        shutil.rmtree(previous, ignore_errors=True)
    print(f"Migrated Ficelle state to {destination}; preserved legacy source at {legacy_home}.")
    return True


def quiesce_legacy_service(
    options: InstallOptions,
    target: ResolvedInstallTarget,
    legacy_home: Path,
) -> None:
    """Stop the legacy service and stale server processes before copying runtime state."""
    env = command_env(options, target, legacy_home)
    package_root = str(Path(__file__).resolve().parents[1])
    inherited_python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((package_root, inherited_python_path))
        if inherited_python_path
        else package_root
    )
    result = run_command(
        [sys.executable, "-m", "ficelle.cli", "stop"],
        dry_run=options.dry_run,
        env=env,
    )
    status = run_command(
        [sys.executable, "-m", "ficelle.cli", "status"],
        dry_run=options.dry_run,
        env=env,
    )
    if options.dry_run:
        return
    manager_status = managed_service_status(status)
    listener_status = legacy_service_listener_status(legacy_home)
    if manager_status != "inactive" or listener_status != "absent":
        raise SystemExit(
            "Legacy Ficelle service inactivity could not be confirmed after stop "
            f"(manager={manager_status}, listener={listener_status}); "
            "migration aborted to protect runtime state."
        )
    if result.returncode != 0:
        print("No managed legacy service was loaded; no active listener remains.")


def managed_service_status(result: CommandResult) -> ManagedServiceStatus:
    """Normalize platform-specific service status output conservatively."""
    if result.returncode == 0:
        return "active"
    output = f"{result.stdout}\n{result.stderr}".lower()
    inactive_markers = (
        "not loaded",
        "could not find service",
        "could not be found",
        "not-found",
        "active: inactive",
        "inactive (dead)",
    )
    if any(marker in output for marker in inactive_markers):
        return "inactive"
    return "unknown"


def legacy_service_listener_status(
    legacy_home: Path,
) -> Literal["active", "absent", "unknown"]:
    """Confirm whether the legacy runtime's configured TCP listener still accepts."""
    config_path = legacy_home / "config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw_config = {}
    except (OSError, ValueError):
        return "unknown"
    config = raw_config if isinstance(raw_config, dict) else {}
    host = str(config.get("host") or "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    try:
        port = int(config.get("port") or 8646)
    except (TypeError, ValueError):
        return "unknown"
    if not 1 <= port <= 65_535:
        return "unknown"
    try:
        connection = socket.create_connection((host, port), timeout=1)
    except ConnectionRefusedError:
        return "absent"
    except OSError:
        return "unknown"
    connection.close()
    return "active"


def package_install_command(options: InstallOptions) -> list[str]:
    command = [options.python, "-m", "pip", "install"]
    package_path = Path(options.package)
    if options.editable and package_path.exists() and package_path.is_dir():
        command.append("-e")
    command.append(options.package)
    return command


def uv_package_install_command(options: InstallOptions) -> list[str]:
    command = ["uv", "pip", "install", "--python", options.python]
    package_path = Path(options.package)
    if options.editable and package_path.exists() and package_path.is_dir():
        command.append("-e")
    command.append(options.package)
    return command


def pip_is_unavailable(result: CommandResult) -> bool:
    combined = f"{result.stdout}\n{result.stderr}"
    return "No module named pip" in combined or "No module named 'pip'" in combined


def command_env(
    options: InstallOptions,
    target: ResolvedInstallTarget,
    runtime_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["FICELLE_HOME"] = str(options.ficelle_home)
    if runtime_dir != options.ficelle_home:
        env["FICELLE_RUNTIME_DIR"] = str(runtime_dir)
    else:
        env.pop("FICELLE_RUNTIME_DIR", None)
    if target == "hermes":
        env["HERMES_HOME"] = str(options.hermes_home)
    else:
        env.pop("HERMES_HOME", None)
    return env


def run_command(command: Sequence[str], *, dry_run: bool, env: Mapping[str, str] | None = None) -> CommandResult:
    printable = " ".join(command)
    env_hint = ""
    if env:
        hints = [
            f"{name}={env[name]}"
            for name in ("FICELLE_HOME", "FICELLE_RUNTIME_DIR", "HERMES_HOME")
            if env.get(name)
        ]
        if hints:
            env_hint = " " + " ".join(hints)
    if dry_run:
        print(f"DRY RUN:{env_hint} {printable}")
        return CommandResult(list(command), 0)
    print(f"RUN:{env_hint} {printable}")
    completed = subprocess.run(list(command), text=True, capture_output=True, env=dict(env) if env else None)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return CommandResult(list(command), completed.returncode, completed.stdout or "", completed.stderr or "")


def _packaged_hermes_plugin_dir(name: str) -> Path:
    return Path(str(resources.files("ficelle").joinpath("assets", "hermes-plugin", name)))


def packaged_plugin_dir() -> Path:
    return _packaged_hermes_plugin_dir("ficelle")


def packaged_compression_plugin_dir() -> Path:
    return _packaged_hermes_plugin_dir("ficelle-compression")


def hermes_plugin_install_specs(hermes_home: Path) -> list[tuple[Path, Path]]:
    return [
        (packaged_plugin_dir(), hermes_home / "plugins" / "model-providers" / "ficelle"),
        (packaged_compression_plugin_dir(), hermes_home / "plugins" / "ficelle-compression"),
    ]


def timestamp_suffix() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.backup-{timestamp_suffix()}")


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def backup_existing_path(path: Path, *, dry_run: bool) -> Path | None:
    if not path.exists():
        return None
    destination = backup_path(path)
    if dry_run:
        print(f"DRY RUN: backup {path} -> {destination}")
        return destination
    _copy_path(path, destination)
    print(f"backup created: {destination}")
    return destination


def trees_equal(source: Path, destination: Path) -> bool:
    if not source.is_dir() or not destination.is_dir():
        return False
    comparison = filecmp.dircmp(source, destination)
    if (
        comparison.left_only
        or comparison.right_only
        or comparison.funny_files
        or comparison.common_funny
    ):
        return False
    if any(
        not filecmp.cmp(source / name, destination / name, shallow=False)
        for name in comparison.common_files
    ):
        return False
    return all(trees_equal(source / name, destination / name) for name in comparison.common_dirs)


def copy_plugin_tree(
    source: Path,
    destination: Path,
    *,
    dry_run: bool,
    backup_existing: bool = True,
) -> bool:
    if not source.exists():
        raise FileNotFoundError(f"Ficelle Hermes plugin template not found: {source}")
    if trees_equal(source, destination):
        print(f"Hermes plugin already up to date: {destination}")
        return False
    if backup_existing:
        backup_existing_path(destination, dry_run=dry_run)
    if dry_run:
        print(f"DRY RUN: copy {source} -> {destination}")
        return True
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"installed Hermes plugin: {destination}")
    return True


def install_plugins(options: InstallOptions) -> None:
    install_specs = hermes_plugin_install_specs(options.hermes_home)
    for source, destination in install_specs:
        copy_plugin_tree(
            source,
            destination,
            dry_run=options.dry_run,
            backup_existing=options.backup_existing,
        )


def latest_backup(path: Path) -> Path | None:
    backups = sorted(path.parent.glob(f"{path.name}.backup-*"))
    return backups[-1] if backups else None


def _preserve_before_rollback(path: Path, *, dry_run: bool) -> Path | None:
    if not path.exists():
        return None
    destination = path.with_name(f"{path.name}.before-rollback-{timestamp_suffix()}")
    if dry_run:
        print(f"DRY RUN: preserve current {path} -> {destination}")
        return destination
    _copy_path(path, destination)
    print(f"Preserved current path before rollback: {destination}")
    return destination


def restore_latest_backup(path: Path, *, dry_run: bool) -> bool:
    backup = latest_backup(path)
    if backup is None:
        print(f"No backup available for {path}; left current path untouched.")
        return False
    _preserve_before_rollback(path, dry_run=dry_run)
    if dry_run:
        print(f"DRY RUN: restore {backup} -> {path}")
        return True
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    _copy_path(backup, path)
    print(f"Restored backup: {backup} -> {path}")
    return True


def rollback_last_hermes_install(options: InstallOptions) -> bool:
    changed_count = 0
    for _source, destination in hermes_plugin_install_specs(options.hermes_home):
        if restore_latest_backup(destination, dry_run=options.dry_run):
            changed_count += 1

    config_path = options.hermes_home / "config.yaml"
    if restore_latest_backup(config_path, dry_run=options.dry_run):
        changed_count += 1

    snippet_path = options.hermes_home / "ficelle" / "hermes-config.snippet.yaml"
    if snippet_path.exists():
        print(f"No backup available for {snippet_path}; left current path untouched.")

    if not changed_count:
        print("No Hermes integration backup was restored.")
        return False
    print(f"Rolled back {changed_count} Hermes integration path(s).")
    return True


def dedicated_keychain_path(options: InstallOptions) -> Path:
    """Ficelle-owned non-interactive keychain used by server-side credential writes."""
    return options.ficelle_home / "ficelle-secrets.keychain-db"


def ensure_dedicated_keychain(options: InstallOptions) -> None:
    """Create the dedicated, empty-password macOS keychain used by the Ficelle service.

    Idempotent (a no-op when the file already exists) and macOS-only: Windows Credential
    Manager and Linux libsecret write without a keychain file to bootstrap. The keychain
    carries an empty password and no auto-lock so the service can unlock it
    non-interactively, and it is deliberately NOT added to the search list
    (``security list-keychains -s``) — the router reads it by explicit path, and listing it
    would risk a blocking GUI prompt on unscoped lookups (incident 2026-06-16)."""
    if sys.platform != "darwin":
        return
    keychain = dedicated_keychain_path(options)
    target = str(keychain)
    if keychain.exists():
        print(f"dedicated secrets keychain already present: {target}")
        return
    if not options.dry_run:
        keychain.parent.mkdir(parents=True, exist_ok=True)
    created = run_command(["security", "create-keychain", "-p", "", target], dry_run=options.dry_run)
    if created.returncode != 0:
        print(f"WARN: could not create dedicated secrets keychain {target}; server-side writes will fall back to .env", file=sys.stderr)
        return
    run_command(["security", "set-keychain-settings", target], dry_run=options.dry_run)
    if options.dry_run:
        return
    try:
        keychain.chmod(0o600)
    except OSError as error:
        print(f"WARN: could not set 0600 on {target}: {error}", file=sys.stderr)
    print(f"created dedicated secrets keychain: {target}")


def ensure_success(result: CommandResult) -> None:
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def install_package(
    options: InstallOptions,
    target: ResolvedInstallTarget,
    runtime_dir: Path,
) -> None:
    env = command_env(options, target, runtime_dir)
    result = run_command(package_install_command(options), dry_run=options.dry_run, env=env)
    if result.returncode == 0:
        return
    if pip_is_unavailable(result) and shutil.which("uv"):
        print("python -m pip is unavailable; retrying package install with uv pip.")
        result = run_command(uv_package_install_command(options), dry_run=options.dry_run, env=env)
    ensure_success(result)


def package_is_local_reference(package: str) -> bool:
    if "://" in package or package.startswith(
        ("git+", "hg+", "svn+", "bzr+")
    ):
        return False
    path = Path(package).expanduser()
    return (
        package in {".", ".."}
        or package.startswith("./")
        or package.startswith("../")
        or package.startswith("~/")
        or "/" in package
        or path.suffix in {".whl", ".zip"}
        or package.endswith(".tar.gz")
        or path.exists()
    )


def probe_target_python(python: str) -> CommandResult:
    code = (
        "import sys; "
        "print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); "
        "raise SystemExit(0 if sys.version_info >= (3, 11) else 42)"
    )
    return run_python_probe(python, code)


def collect_preflight_checks(
    options: InstallOptions,
    target: ResolvedInstallTarget,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    checks.append(
        PreflightCheck(
            "target",
            "ok",
            f"selected target: {target}"
            + (" (auto-detected)" if options.target == "auto" else " (explicit)"),
        )
    )

    if options.skip_package:
        checks.append(PreflightCheck("package", "ok", "package install skipped"))
    elif package_is_local_reference(options.package):
        package_path = Path(options.package).expanduser()
        if package_path.exists():
            checks.append(PreflightCheck("package", "ok", f"local package found: {package_path}"))
        else:
            checks.append(PreflightCheck("package", "fail", f"local package not found: {package_path}", "Check --package or build the wheel first."))
    else:
        checks.append(PreflightCheck("package", "ok", f"package spec will be resolved by pip: {options.package}"))

    python_result = probe_target_python(options.python)
    if python_result.returncode == 0:
        checks.append(PreflightCheck("python", "ok", f"Python {python_result.stdout.strip()} at {options.python}"))
    elif python_result.returncode == 42:
        checks.append(PreflightCheck("python", "fail", f"Python is too old: {python_result.stdout.strip() or options.python}", "Use Python 3.11+."))
    else:
        detail = (python_result.stderr or python_result.stdout or "target Python did not run").strip()
        checks.append(PreflightCheck("python", "fail", detail, "Check --python points to a working interpreter."))

    if target != "hermes":
        checks.append(
            PreflightCheck(
                "plugin",
                "ok",
                f"Hermes plugin copy not applicable to target {target}",
            )
        )
    elif options.skip_plugin:
        checks.append(PreflightCheck("plugin", "ok", "plugin copy skipped"))
    else:
        plugin_dirs = [source for source, _destination in hermes_plugin_install_specs(options.hermes_home)]
        required = [path / name for path in plugin_dirs for name in ("__init__.py", "plugin.yaml")]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            checks.append(PreflightCheck("plugin", "fail", f"missing packaged plugin assets: {', '.join(missing)}", "Rebuild/reinstall the package."))
        else:
            checks.append(PreflightCheck("plugin", "ok", f"packaged plugin assets found: {', '.join(str(path) for path in plugin_dirs)}"))

    if options.skip_service:
        checks.append(PreflightCheck("service", "ok", "managed service install skipped"))
    elif sys.platform == "darwin":
        checks.append(PreflightCheck("service", "ok", "macOS LaunchAgent backend available"))
    elif sys.platform.startswith("linux"):
        if shutil.which("systemctl"):
            checks.append(PreflightCheck("service", "ok", "Linux systemd user backend available"))
        else:
            checks.append(PreflightCheck("service", "fail", "systemctl not found", "Install systemd tools or use --skip-service for developer mode."))
    else:
        checks.append(PreflightCheck("service", "fail", f"unsupported managed service platform: {sys.platform}", "Use --skip-service for developer mode or wait for a Windows backend."))

    if sys.platform == "darwin":
        keychain = dedicated_keychain_path(options)
        if keychain.exists():
            checks.append(PreflightCheck("keychain", "ok", f"dedicated secrets keychain present: {keychain}"))
        else:
            checks.append(PreflightCheck("keychain", "ok", f"dedicated secrets keychain will be created (encrypted server-side write target): {keychain}"))

    if target == "hermes":
        hermes_parent = options.hermes_home.parent
        if options.dry_run:
            checks.append(PreflightCheck("hermes-home", "ok", f"dry-run target: {options.hermes_home}"))
        elif options.hermes_home.exists() or hermes_parent.exists():
            checks.append(PreflightCheck("hermes-home", "ok", f"Hermes home target: {options.hermes_home}"))
        else:
            checks.append(PreflightCheck("hermes-home", "fail", f"parent does not exist: {hermes_parent}", "Create the parent directory or pass --hermes-home."))

    if options.configure_hermes and target != "hermes":
        checks.append(
            PreflightCheck(
                "hermes-config",
                "fail",
                f"--configure-hermes requires --target hermes (selected: {target})",
                "Select --target hermes or omit --configure-hermes.",
            )
        )
    elif options.configure_hermes:
        config_path = options.hermes_home / "config.yaml"
        if not config_path.exists():
            checks.append(PreflightCheck("hermes-config", "ok", "config.yaml is absent; setup can create a Ficelle-only config with backup-safe semantics"))
        else:
            text = config_path.read_text(encoding="utf-8", errors="replace")
            if MANAGED_CONFIG_BEGIN in text and MANAGED_CONFIG_END in text:
                checks.append(PreflightCheck("hermes-config", "ok", "existing Ficelle managed config block can be updated with backup"))
            else:
                checks.append(PreflightCheck("hermes-config", "warn", "existing config.yaml is unmanaged; setup will not rewrite it", "A ready snippet will be written under ~/.hermes/ficelle/."))
    elif target == "hermes":
        checks.append(PreflightCheck("hermes-config", "ok", "Hermes config edit disabled; use --configure-hermes to write a safe snippet/apply when possible"))

    return checks


def print_preflight_report(checks: Sequence[PreflightCheck]) -> None:
    print("Ficelle setup preflight:")
    for check in checks:
        icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}.get(check.status, check.status.upper())
        print(f"- {icon} {check.name}: {check.detail}")
        if check.action:
            print(f"  action: {check.action}")


def run_preflight(
    options: InstallOptions,
    target: ResolvedInstallTarget,
) -> None:
    checks = collect_preflight_checks(options, target)
    print_preflight_report(checks)
    if any(check.failed for check in checks):
        raise SystemExit(2)


def hermes_config_body() -> str:
    return "\n".join([
        "# Ficelle provider for Hermes. Strict-zero defaults only; no provider secret is included.",
        "providers:",
        "  ficelle:",
        "    name: \"Ficelle FREE\"",
        "    api: \"http://127.0.0.1:8646/v1\"",
        "    transport: openai_chat",
        "    models:",
        "      - \"ficelle/auto-orchestrator\"",
        "      - \"ficelle/auto-tools\"",
        "      - \"ficelle/auto-json\"",
        "      - \"ficelle/auto-compression\"",
        "      - \"ficelle/auto-long\"",
        "      - \"ficelle/auto-fast\"",
        "      - \"ficelle/auto-reasoning\"",
        "      - \"ficelle/auto-multimodal\"",
        "      - \"ficelle/auto-vision\"",
        "      - \"ficelle/auto-video\"",
        "      - \"ficelle/auto-audio\"",
        "",
        "# Low-risk slots first. Do not promote Ficelle to the main model before dogfood/canaries stay green.",
        "auxiliary:",
        "  title_generation:",
        "    provider: \"ficelle\"",
        "    model: \"ficelle/auto-fast\"",
        "  compression:",
        "    provider: \"ficelle\"",
        "    model: \"ficelle/auto-compression\"",
        "  web_extract:",
        "    provider: \"ficelle\"",
        "    model: \"ficelle/auto-json\"",
        "",
        "fallback_providers:",
        "  - provider: \"ficelle\"",
        "    model: \"ficelle/auto-tools\"",
        "",
    ])


def hermes_plugins_enabled_block() -> str:
    return "\n".join([
        "plugins:",
        "  enabled:",
        f"    - \"{PROVIDER_PLUGIN_NAME}\"",
        f"    - \"{COMPRESSION_PLUGIN_NAME}\"",
        "",
    ])


def hermes_toolsets_block() -> str:
    return "\n".join([
        "toolsets:",
        f"  - \"{COMPRESSION_TOOLSET_NAME}\"",
        "",
    ])


def managed_hermes_config() -> str:
    return f"{MANAGED_CONFIG_BEGIN}\n{hermes_config_body()}{MANAGED_CONFIG_END}\n"


def ensure_hermes_plugin_enabled(text: str, plugin_name: str = COMPRESSION_PLUGIN_NAME) -> str:
    lines = text.splitlines()
    plugins_index = next((index for index, line in enumerate(lines) if line.strip() == "plugins:" and not line.startswith((" ", "\t"))), None)
    if plugins_index is None:
        return f"{hermes_plugins_enabled_block()}{text}"

    block_end = len(lines)
    for index in range(plugins_index + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
            block_end = index
            break

    child_indent = "  "
    for index in range(plugins_index + 1, block_end):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        child_indent = line[:len(line) - len(line.lstrip())]
        break

    enabled_index = next(
        (
            index
            for index in range(plugins_index + 1, block_end)
            if lines[index].startswith(f"{child_indent}enabled:")
        ),
        None,
    )
    if enabled_index is None:
        lines[plugins_index + 1:plugins_index + 1] = [f"{child_indent}enabled:", f"{child_indent}  - \"{plugin_name}\""]
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    enabled_indent = lines[enabled_index][:len(lines[enabled_index]) - len(lines[enabled_index].lstrip())]
    inline_value = lines[enabled_index].split(":", 1)[1].strip()
    if inline_value:
        if inline_value.startswith("[") and inline_value.endswith("]"):
            values = [item.strip().strip("\"'") for item in inline_value[1:-1].split(",") if item.strip()]
        else:
            values = [inline_value.strip().strip("\"'")]
        if plugin_name in values:
            return text
        replacement = [f"{enabled_indent}enabled:"]
        replacement.extend(f"{enabled_indent}- {value}" for value in values)
        replacement.append(f"{enabled_indent}- \"{plugin_name}\"")
        lines[enabled_index:enabled_index + 1] = replacement
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    item_indent = None
    insert_at = enabled_index + 1
    while insert_at < block_end and lines[insert_at].lstrip().startswith("- "):
        value = lines[insert_at].lstrip()[2:].strip().strip("\"'")
        if value == plugin_name:
            return text
        if item_indent is None:
            item_indent = lines[insert_at][:len(lines[insert_at]) - len(lines[insert_at].lstrip())]
        insert_at += 1
    if item_indent is None:
        item_indent = f"{enabled_indent}  "
    lines.insert(insert_at, f"{item_indent}- \"{plugin_name}\"")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def ensure_top_level_list_item(text: str, key: str, item: str) -> str:
    quoted_item = f"\"{item}\""
    lines = text.splitlines()
    key_index = next((index for index, line in enumerate(lines) if line.startswith(f"{key}:")), None)
    if key_index is None:
        return f"{key}:\n  - {quoted_item}\n{text}"

    block_end = len(lines)
    for index in range(key_index + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
            block_end = index
            break

    inline_value = lines[key_index].split(":", 1)[1].strip()
    if inline_value:
        if inline_value.startswith("[") and inline_value.endswith("]"):
            values = [entry.strip().strip("\"'") for entry in inline_value[1:-1].split(",") if entry.strip()]
        else:
            values = [inline_value.strip().strip("\"'")]
        if item in values:
            return text
        replacement = [f"{key}:"]
        replacement.extend(f"  - {value}" for value in values)
        replacement.append(f"  - {quoted_item}")
        lines[key_index:key_index + 1] = replacement
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    item_indent = None
    insert_at = key_index + 1
    while insert_at < block_end and lines[insert_at].lstrip().startswith("- "):
        value = lines[insert_at].lstrip()[2:].strip().strip("\"'")
        if value == item:
            return text
        if item_indent is None:
            item_indent = lines[insert_at][:len(lines[insert_at]) - len(lines[insert_at].lstrip())]
        insert_at += 1
    if item_indent is None:
        item_indent = "  "
    lines.insert(insert_at, f"{item_indent}- {quoted_item}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def ensure_hermes_toolset_enabled(text: str, toolset_name: str = COMPRESSION_TOOLSET_NAME) -> str:
    return ensure_top_level_list_item(text, "toolsets", toolset_name)


def ensure_hermes_compression_plugin_enabled(text: str) -> str:
    with_provider = ensure_hermes_plugin_enabled(text, PROVIDER_PLUGIN_NAME)
    with_compression = ensure_hermes_plugin_enabled(
        with_provider,
        COMPRESSION_PLUGIN_NAME,
    )
    return ensure_hermes_toolset_enabled(with_compression)


def replace_managed_block(text: str, replacement: str) -> str:
    start = text.index(MANAGED_CONFIG_BEGIN)
    end = text.index(MANAGED_CONFIG_END, start) + len(MANAGED_CONFIG_END)
    return f"{text[:start]}{replacement.rstrip()}\n{text[end:].lstrip()}"


def write_text_file(path: Path, content: str, *, dry_run: bool) -> None:
    if path.exists() and path.read_text(encoding="utf-8", errors="replace") == content:
        print(f"Already up to date: {path}")
        return
    if dry_run:
        print(f"DRY RUN: write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote: {path}")


def configure_hermes(options: InstallOptions) -> None:
    snippet_path = options.hermes_home / "ficelle" / "hermes-config.snippet.yaml"
    config_path = options.hermes_home / "config.yaml"
    managed = managed_hermes_config()
    write_text_file(snippet_path, f"{hermes_plugins_enabled_block()}{hermes_toolsets_block()}{hermes_config_body()}", dry_run=options.dry_run)

    if not config_path.exists():
        write_text_file(config_path, ensure_hermes_compression_plugin_enabled(managed), dry_run=options.dry_run)
        print(f"Hermes config created with Ficelle low-risk slots: {config_path}")
        return

    current = config_path.read_text(encoding="utf-8", errors="replace")
    if MANAGED_CONFIG_BEGIN not in current or MANAGED_CONFIG_END not in current:
        print(f"Existing unmanaged Hermes config left untouched: {config_path}")
        print(f"Next: review and merge the ready snippet: {snippet_path}")
        return

    updated = ensure_hermes_compression_plugin_enabled(
        replace_managed_block(current, managed)
    )
    if updated == current:
        print(f"Hermes config already up to date: {config_path}")
        return
    if options.backup_existing:
        backup_existing_path(config_path, dry_run=options.dry_run)
    write_text_file(config_path, updated, dry_run=options.dry_run)
    print(f"Hermes config Ficelle block updated: {config_path}")


def doctor_auth_report(stdout: str) -> dict[str, Any] | None:
    """The `auth` block of a `ficelle doctor --json` run, or ``None`` when unreadable.

    Read from the smoke check that already ran rather than from an import of the
    router: setup drives the *installed* interpreter (`--python` need not be the one
    running this file), so its answer is the only one that describes the install the
    user is about to use. ``None`` means "not known" — a dry run, `--skip-smoke`, or
    output this cannot parse — and callers must then say nothing rather than guess.
    """
    text = str(stdout or "")
    try:
        # Setup narrates its own steps around the smoke checks, so the report is found
        # from its first brace rather than assumed to start at the first byte.
        payload = json.loads(text[text.index("{"):])
    except ValueError:  # no brace at all, or a body that is not JSON
        return None
    auth = payload.get("auth") if isinstance(payload, dict) else None
    return auth if isinstance(auth, dict) else None


def first_run_provider_key_notice(auth: dict[str, Any] | None) -> list[str]:
    """The closing block for an install that cannot serve a request yet.

    Empty when a provider is configured — a working install must not be nagged — and
    empty when credentials could not be read at all, since a false alarm here would
    send a configured user hunting a problem they do not have.
    """
    if auth is None or invokable_provider_sources(auth):
        return []
    # Imported here, not at module scope: the registry lives in router.py beside the
    # provider definitions, and router pulls in the HTTP stack that a source-checkout
    # bootstrap has not necessarily installed at the time this module is imported.
    try:
        from ficelle.router import PROVIDER_KEY_URLS
    except Exception:  # noqa: BLE001 - the advice is worth printing without its links
        PROVIDER_KEY_URLS = {}
    return [
        "",
        "No provider API key is configured, so Ficelle cannot serve a completion yet.",
        "Ficelle routes through your own provider accounts and ships no keys of its own.",
        "Create a key, then store it (input is hidden and never reaches shell history):",
        "",
        *(
            f"  {line}"
            for line in provider_key_setup_commands(unconfigured_provider_sources(auth), PROVIDER_KEY_URLS)
        ),
        "",
        "`ficelle models` lists the providers' public catalogs, which they serve without",
        "credentials — a populated model list does NOT mean a request can be served.",
        "`ficelle doctor --text` reports which providers are actually configured.",
    ]


def run_install(options: InstallOptions) -> int:
    if options.rollback:
        if options.target != "hermes":
            raise SystemExit("--rollback requires explicit --target hermes")
        return 0 if rollback_last_hermes_install(options) else 1

    target = resolve_install_target(options)
    run_preflight(options, target)
    if options.preflight_only:
        print("Preflight complete. No install actions were run.")
        return 0

    # Running setup is the explicit mutation boundary for one-command upgrades.
    legacy_home = options.hermes_home / "ficelle"
    migrated_legacy_home = migrate_legacy_ficelle_home(
        options,
        before_copy=lambda: quiesce_legacy_service(
            options,
            target,
            legacy_home,
        ),
    )
    runtime_dir = (
        legacy_home
        if not options.ficelle_home_explicit
        and legacy_home.is_dir()
        and not migrated_legacy_home
        else options.ficelle_home
    )

    if not options.skip_package:
        install_package(options, target, runtime_dir)

    if target == "hermes" and not options.skip_plugin:
        install_plugins(options)

    ensure_dedicated_keychain(options)

    env = command_env(options, target, runtime_dir)
    if not options.skip_service:
        ensure_success(run_command([options.python, "-m", "ficelle.cli", "install"], dry_run=options.dry_run, env=env))

    if target == "hermes" and options.configure_hermes:
        configure_hermes(options)

    provider_auth: dict[str, Any] | None = None
    if not options.skip_smoke:
        doctor_result = run_command([options.python, "-m", "ficelle.cli", "doctor", "--json"], dry_run=options.dry_run, env=env)
        ensure_success(doctor_result)
        provider_auth = doctor_auth_report(doctor_result.stdout)
        ensure_success(run_command([options.python, "-m", "ficelle.cli", "health"], dry_run=options.dry_run, env=env))
        ensure_success(run_command([options.python, "-m", "ficelle.cli", "models"], dry_run=options.dry_run, env=env))

    if options.skip_service and not options.dry_run:
        options.ficelle_home.mkdir(parents=True, exist_ok=True)
        if not persist_active_service_context(
            options.ficelle_home,
            active_home_pointer_path(),
            runtime_dir=runtime_dir,
            hermes_home=options.hermes_home if target == "hermes" else None,
        ):
            raise SystemExit("Ficelle setup could not persist the selected --ficelle-home.")

    key_notice = first_run_provider_key_notice(provider_auth)

    print("Ficelle setup complete.")
    if target == "hermes":
        if options.configure_hermes:
            print("Hermes config step complete or snippet written. Restart Hermes gateway after merging config changes.")
        else:
            print("Next: run `ficelle export --target hermes` or rerun setup with `--configure-hermes` for a safe config snippet/apply step.")
        print("Rollback: `ficelle-setup --target hermes --rollback` restores the latest available integration backups.")
        print("Do not make Ficelle the main Hermes model until dogfood/canaries stay green.")
    else:
        print("Local OpenAI-compatible base URL: http://127.0.0.1:8646/v1")
        # `ficelle models` answers from the public catalogs, which the reference providers
        # serve without credentials — so pointing an unconfigured install at it as its
        # verification step is what teaches the user that setup worked when it did not.
        if not key_notice:
            print("Verify: `ficelle health` and `ficelle models`.")
        print(
            "OpenAI client: `OpenAI(base_url=\"http://127.0.0.1:8646/v1\", "
            "api_key=\"ficelle-local\")`."
        )
        if target == "openclaw":
            print("OpenClaw target is experimental; merge `/admin/export/openclaw` into OpenClaw manually.")
    for line in key_notice:
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install Ficelle Core and an optional target integration.")
    parser.add_argument("--package", default=".", help="Package spec/path to install with pip. Default: current directory.")
    parser.add_argument("--no-editable", action="store_true", help="Do not use pip editable mode for local directory installs.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for pip and ficelle CLI commands.")
    parser.add_argument("--target", choices=INSTALL_TARGETS, default="auto", help="Integration target. Auto detects Hermes, otherwise installs standalone Core.")
    parser.add_argument("--ficelle-home", default=None, help="Ficelle state root. Default: FICELLE_HOME or ~/.ficelle")
    parser.add_argument("--hermes-home", default=str(default_hermes_home()), help="Hermes home directory. Default: ~/.hermes")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files or running commands.")
    parser.add_argument("--preflight-only", action="store_true", help="Run install preflight checks and stop before mutating anything.")
    parser.add_argument("--configure-hermes", action="store_true", help="Write a ready Hermes config snippet, and safely create/update config.yaml only when it is absent or Ficelle-managed.")
    parser.add_argument("--rollback", action="store_true", help="With --target hermes, restore the latest available plugin/config backups.")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup existing plugin/config files before replacement.")
    parser.add_argument("--skip-package", action="store_true", help="Skip pip install step.")
    parser.add_argument("--skip-plugin", action="store_true", help="Skip Hermes provider plugin copy.")
    parser.add_argument("--skip-service", action="store_true", help="Skip managed service install/start step.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip doctor/health/models smoke checks.")
    return parser


def options_from_args(args: argparse.Namespace) -> InstallOptions:
    configured_ficelle_home = args.ficelle_home or os.getenv("FICELLE_HOME")
    return InstallOptions(
        package=args.package,
        editable=not args.no_editable,
        python=args.python,
        target=args.target,
        ficelle_home=Path(configured_ficelle_home or default_ficelle_home()).expanduser().resolve(),
        hermes_home=Path(args.hermes_home).expanduser().resolve(),
        dry_run=args.dry_run,
        skip_package=args.skip_package,
        skip_plugin=args.skip_plugin,
        skip_service=args.skip_service,
        skip_smoke=args.skip_smoke,
        preflight_only=args.preflight_only,
        configure_hermes=args.configure_hermes,
        backup_existing=not args.no_backup,
        rollback=args.rollback,
        ficelle_home_explicit=configured_ficelle_home is not None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_install(options_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
