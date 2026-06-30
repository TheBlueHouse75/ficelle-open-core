"""User-proof installer for Ficelle local deployments."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Mapping, Sequence

MANAGED_CONFIG_BEGIN = "# BEGIN FICELLE MANAGED CONFIG"
MANAGED_CONFIG_END = "# END FICELLE MANAGED CONFIG"
COMPRESSION_PLUGIN_NAME = "ficelle-compression"
COMPRESSION_TOOLSET_NAME = "ficelle"


@dataclass(frozen=True)
class InstallOptions:
    package: str
    editable: bool
    python: str
    hermes_home: Path
    dry_run: bool
    skip_package: bool
    skip_plugin: bool
    skip_service: bool
    skip_smoke: bool
    preflight_only: bool
    configure_hermes: bool
    backup_existing: bool


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


def command_env(options: InstallOptions) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(options.hermes_home)
    return env


def run_command(command: Sequence[str], *, dry_run: bool, env: Mapping[str, str] | None = None) -> CommandResult:
    printable = " ".join(command)
    env_hint = ""
    if env and env.get("HERMES_HOME"):
        env_hint = f" HERMES_HOME={env['HERMES_HOME']}"
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
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.backup-{timestamp_suffix()}")


def backup_existing_path(path: Path, *, dry_run: bool) -> Path | None:
    if not path.exists():
        return None
    destination = backup_path(path)
    if dry_run:
        print(f"DRY RUN: backup {path} -> {destination}")
        return destination
    if path.is_dir():
        shutil.copytree(path, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(f"backup created: {destination}")
    return destination


def copy_plugin_tree(source: Path, destination: Path, *, dry_run: bool, backup_existing: bool = True) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Ficelle Hermes plugin template not found: {source}")
    if backup_existing:
        backup_existing_path(destination, dry_run=dry_run)
    if dry_run:
        print(f"DRY RUN: copy {source} -> {destination}")
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"installed Hermes plugin: {destination}")


def install_plugins(options: InstallOptions) -> None:
    install_specs = hermes_plugin_install_specs(options.hermes_home)
    for source, destination in install_specs:
        copy_plugin_tree(
            source,
            destination,
            dry_run=options.dry_run,
            backup_existing=options.backup_existing,
        )


def dedicated_keychain_path(options: InstallOptions) -> Path:
    """Path of the dedicated, non-interactive secrets keychain the router's server-side
    write path targets. Mirrors ``router.HERMES_SECRETS_KEYCHAIN``: ``FICELLE_HOME``
    defaults to the Hermes home, and the launchd daemon only carries ``HERMES_HOME`` in
    its plist env, so install and daemon resolve to the same file."""
    ficelle_home = Path(os.getenv("FICELLE_HOME") or options.hermes_home).expanduser()
    return ficelle_home / "hermes-secrets.keychain-db"


def ensure_dedicated_keychain(options: InstallOptions) -> None:
    """Create the dedicated, empty-password macOS keychain that the R9 server-side
    credential write targets, so the launchd daemon and admin web form can store a pasted
    key in an encrypted store instead of silently degrading to plaintext
    ``FICELLE_HOME/.env``.

    Idempotent (a no-op when the file already exists) and macOS-only: Windows Credential
    Manager and Linux libsecret write without a keychain file to bootstrap. The keychain
    carries an empty password and no auto-lock so ``_unlock_hermes_keychain`` can unlock it
    non-interactively under launchd, and it is deliberately NOT added to the search list
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


def install_package(options: InstallOptions) -> None:
    result = run_command(package_install_command(options), dry_run=options.dry_run, env=command_env(options))
    if result.returncode == 0:
        return
    if pip_is_unavailable(result) and shutil.which("uv"):
        print("python -m pip is unavailable; retrying package install with uv pip.")
        result = run_command(uv_package_install_command(options), dry_run=options.dry_run, env=command_env(options))
    ensure_success(result)


def package_is_local_reference(package: str) -> bool:
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
    completed = subprocess.run([python, "-c", code], text=True, capture_output=True)
    return CommandResult([python, "-c", code], completed.returncode, completed.stdout or "", completed.stderr or "")


def collect_preflight_checks(options: InstallOptions) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []

    if package_is_local_reference(options.package):
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

    if options.skip_plugin:
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

    hermes_parent = options.hermes_home.parent
    if options.dry_run:
        checks.append(PreflightCheck("hermes-home", "ok", f"dry-run target: {options.hermes_home}"))
    elif options.hermes_home.exists() or hermes_parent.exists():
        checks.append(PreflightCheck("hermes-home", "ok", f"Hermes home target: {options.hermes_home}"))
    else:
        checks.append(PreflightCheck("hermes-home", "fail", f"parent does not exist: {hermes_parent}", "Create the parent directory or pass --hermes-home."))

    if options.configure_hermes:
        config_path = options.hermes_home / "config.yaml"
        if not config_path.exists():
            checks.append(PreflightCheck("hermes-config", "ok", "config.yaml is absent; setup can create a Ficelle-only config with backup-safe semantics"))
        else:
            text = config_path.read_text(encoding="utf-8", errors="replace")
            if MANAGED_CONFIG_BEGIN in text and MANAGED_CONFIG_END in text:
                checks.append(PreflightCheck("hermes-config", "ok", "existing Ficelle managed config block can be updated with backup"))
            else:
                checks.append(PreflightCheck("hermes-config", "warn", "existing config.yaml is unmanaged; setup will not rewrite it", "A ready snippet will be written under ~/.hermes/ficelle/."))
    else:
        checks.append(PreflightCheck("hermes-config", "ok", "Hermes config edit disabled; use --configure-hermes to write a safe snippet/apply when possible"))

    return checks


def print_preflight_report(checks: Sequence[PreflightCheck]) -> None:
    print("Ficelle setup preflight:")
    for check in checks:
        icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}.get(check.status, check.status.upper())
        print(f"- {icon} {check.name}: {check.detail}")
        if check.action:
            print(f"  action: {check.action}")


def run_preflight(options: InstallOptions) -> None:
    checks = collect_preflight_checks(options)
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
    return ensure_hermes_toolset_enabled(ensure_hermes_plugin_enabled(text))


def replace_managed_block(text: str, replacement: str) -> str:
    start = text.index(MANAGED_CONFIG_BEGIN)
    end = text.index(MANAGED_CONFIG_END, start) + len(MANAGED_CONFIG_END)
    return f"{text[:start]}{replacement.rstrip()}\n{text[end:].lstrip()}"


def write_text_file(path: Path, content: str, *, dry_run: bool) -> None:
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

    if options.backup_existing:
        backup_existing_path(config_path, dry_run=options.dry_run)
    write_text_file(config_path, ensure_hermes_compression_plugin_enabled(replace_managed_block(current, managed)), dry_run=options.dry_run)
    print(f"Hermes config Ficelle block updated: {config_path}")


def run_install(options: InstallOptions) -> int:
    run_preflight(options)
    if options.preflight_only:
        print("Preflight complete. No install actions were run.")
        return 0

    if not options.skip_package:
        install_package(options)

    if not options.skip_plugin:
        install_plugins(options)

    ensure_dedicated_keychain(options)

    if not options.skip_service:
        ensure_success(run_command([options.python, "-m", "ficelle.cli", "install"], dry_run=options.dry_run, env=command_env(options)))

    if options.configure_hermes:
        configure_hermes(options)

    if not options.skip_smoke:
        ensure_success(run_command([options.python, "-m", "ficelle.cli", "doctor", "--json"], dry_run=options.dry_run, env=command_env(options)))
        ensure_success(run_command([options.python, "-m", "ficelle.cli", "health"], dry_run=options.dry_run, env=command_env(options)))
        ensure_success(run_command([options.python, "-m", "ficelle.cli", "models"], dry_run=options.dry_run, env=command_env(options)))

    print("Ficelle setup complete.")
    if options.configure_hermes:
        print("Hermes config step complete or snippet written. Restart Hermes gateway after merging config changes.")
    else:
        print("Next: run `ficelle export` or rerun setup with `--configure-hermes` for a safe config snippet/apply step.")
    print("Do not make Ficelle the main Hermes model until dogfood/canaries stay green.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install Ficelle package, Hermes plugin, local service backend, and smoke checks.")
    parser.add_argument("--package", default=".", help="Package spec/path to install with pip. Default: current directory.")
    parser.add_argument("--no-editable", action="store_true", help="Do not use pip editable mode for local directory installs.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for pip and ficelle CLI commands.")
    parser.add_argument("--hermes-home", default=str(default_hermes_home()), help="Hermes home directory. Default: ~/.hermes")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files or running commands.")
    parser.add_argument("--preflight-only", action="store_true", help="Run install preflight checks and stop before mutating anything.")
    parser.add_argument("--configure-hermes", action="store_true", help="Write a ready Hermes config snippet, and safely create/update config.yaml only when it is absent or Ficelle-managed.")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup existing plugin/config files before replacement.")
    parser.add_argument("--skip-package", action="store_true", help="Skip pip install step.")
    parser.add_argument("--skip-plugin", action="store_true", help="Skip Hermes provider plugin copy.")
    parser.add_argument("--skip-service", action="store_true", help="Skip managed service install/start step.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip doctor/health/models smoke checks.")
    return parser


def options_from_args(args: argparse.Namespace) -> InstallOptions:
    return InstallOptions(
        package=args.package,
        editable=not args.no_editable,
        python=args.python,
        hermes_home=Path(args.hermes_home).expanduser(),
        dry_run=args.dry_run,
        skip_package=args.skip_package,
        skip_plugin=args.skip_plugin,
        skip_service=args.skip_service,
        skip_smoke=args.skip_smoke,
        preflight_only=args.preflight_only,
        configure_hermes=args.configure_hermes,
        backup_existing=not args.no_backup,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_install(options_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
