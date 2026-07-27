"""Ficelle command line interface."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ficelle.runtime_paths import RuntimePaths
from ficelle.service import (
    LaunchAgentServiceBackend,
    ServiceBackend,
    ServicePaths,
    active_home_pointer_path,
    read_active_service_context,
    select_service_backend,
)

LABEL = "com.ficelle.router"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
SYSTEMD_UNIT = Path.home() / ".config" / "systemd" / "user" / f"{LABEL}.service"
ACTIVE_HOME_POINTER = active_home_pointer_path()


def cli_runtime_context() -> tuple[RuntimePaths, Path | None]:
    """Resolve env first, then persisted service context, then legacy/default discovery."""
    source = os.environ.copy()
    if not source.get("FICELLE_HOME") and not source.get("FICELLE_RUNTIME_DIR"):
        context = read_active_service_context(ACTIVE_HOME_POINTER)
        if context is not None:
            ficelle_home, runtime_dir, hermes_home = context
            source["FICELLE_HOME"] = str(ficelle_home)
            source["FICELLE_RUNTIME_DIR"] = str(runtime_dir)
            if not source.get("HERMES_HOME"):
                if hermes_home is not None:
                    source["HERMES_HOME"] = str(hermes_home)
                else:
                    source.pop("HERMES_HOME", None)
    paths = RuntimePaths.from_env(environ=source)
    hermes_home = (
        Path(source["HERMES_HOME"]).expanduser()
        if source.get("HERMES_HOME")
        else None
    )
    return paths, hermes_home


RUNTIME_PATHS, ACTIVE_HERMES_HOME = cli_runtime_context()
FICELLE_HOME = RUNTIME_PATHS.ficelle_home
RUNTIME_DIR = RUNTIME_PATHS.runtime_read_dir
INSTALL_PYTHON = Path(sys.executable)


@contextmanager
def active_runtime_environment() -> Iterator[None]:
    """Temporarily expose the selected roots to runtime modules, including lazy imports."""
    previous = {
        name: os.environ.get(name)
        for name in ("FICELLE_HOME", "FICELLE_RUNTIME_DIR", "HERMES_HOME")
    }
    try:
        os.environ["FICELLE_HOME"] = str(FICELLE_HOME)
        if RUNTIME_DIR != FICELLE_HOME:
            os.environ["FICELLE_RUNTIME_DIR"] = str(RUNTIME_DIR)
        else:
            os.environ.pop("FICELLE_RUNTIME_DIR", None)
        if ACTIVE_HERMES_HOME is not None:
            os.environ["HERMES_HOME"] = str(ACTIVE_HERMES_HOME)
        else:
            os.environ.pop("HERMES_HOME", None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


with active_runtime_environment():
    from ficelle import license_ops, router  # noqa: E402


def _run(cmd: list[str], *, check: bool = False, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=check)


def _uid() -> str:
    return str(os.getuid())


def service_paths() -> ServicePaths:
    return ServicePaths(
        ficelle_home=FICELLE_HOME,
        runtime_dir=RUNTIME_DIR,
        label=LABEL,
        plist=PLIST,
        systemd_unit=SYSTEMD_UNIT,
        install_python=INSTALL_PYTHON,
        hermes_home=ACTIVE_HERMES_HOME,
        active_home_pointer=ACTIVE_HOME_POINTER,
    )


def ficelle_server_pids() -> list[int]:
    """Return live Ficelle server PIDs using a narrow command signature."""
    result = _run(["ps", "-axo", "pid=,command="])
    if result.returncode != 0:
        return []
    current_pid = os.getpid()
    pids: list[int] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if "-m ficelle.router --serve" in command:
            pids.append(pid)
    return pids


def terminate_stale_servers(*, grace_seconds: float = 2.0) -> list[int]:
    """Terminate orphan Ficelle server processes left behind after launchctl bootout."""
    pids = ficelle_server_pids()
    if not pids:
        return []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        remaining = [pid for pid in ficelle_server_pids() if pid in pids]
        if not remaining:
            return pids
        time.sleep(0.1)
    for pid in [pid for pid in ficelle_server_pids() if pid in pids]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def current_service_backend(*, platform_name: str | None = None) -> ServiceBackend:
    return select_service_backend(
        platform_name=platform_name or sys.platform,
        paths=service_paths(),
        run_command=_run,
        uid_provider=_uid,
        wait_for_ready=wait_for_admin_status,
        terminate_stale_servers=terminate_stale_servers,
        report_stale_servers=report_terminated_stale_servers,
    )


def launchagent_backend() -> LaunchAgentServiceBackend:
    backend = select_service_backend(
        platform_name="darwin",
        paths=service_paths(),
        run_command=_run,
        uid_provider=_uid,
        wait_for_ready=wait_for_admin_status,
        terminate_stale_servers=terminate_stale_servers,
        report_stale_servers=report_terminated_stale_servers,
    )
    if not isinstance(backend, LaunchAgentServiceBackend):
        raise RuntimeError("expected LaunchAgentServiceBackend")
    return backend


def plist_payload() -> dict[str, object]:
    return launchagent_backend().plist_payload()


def bootout_launchagent() -> subprocess.CompletedProcess[str]:
    return launchagent_backend().bootout()


def report_terminated_stale_servers(stale_pids: list[int]) -> None:
    if stale_pids:
        print(f"terminated stale Ficelle server pid(s): {', '.join(str(pid) for pid in stale_pids)}")


def router_url(path: str) -> str:
    config = router.load_config()
    host = config.get("host") or "127.0.0.1"
    port = int(config.get("port") or 8646)
    return f"http://{host}:{port}{path}"


def read_admin_status(*, timeout_seconds: float = 2.0) -> dict[str, Any]:
    """Read the live machine-readable admin status without exposing credentials."""
    url = router_url("/admin/status.json")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
    except urllib.error.HTTPError as exc:
        return {"ready": False, "url": url, "http_status": exc.code, "error": "http_error"}
    except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
        return {"ready": False, "url": url, "error": exc.__class__.__name__, "detail": str(exc)[:200]}
    if response.status != 200 or not isinstance(payload, dict):
        return {"ready": False, "url": url, "http_status": response.status, "error": "invalid_status_payload"}
    raw_catalog = payload.get("catalog")
    catalog = raw_catalog if isinstance(raw_catalog, dict) else {}
    return {
        "ready": payload.get("status") in {"ok", "degraded", "fail"},
        "url": url,
        "status": payload.get("status") or "unknown",
        "generated_at": payload.get("generated_at"),
        "model_count": catalog.get("model_count"),
        "available_count": catalog.get("available_count"),
        "model_cooldowns": len(payload.get("model_cooldowns") or []),
        "provider_cooldowns": len(payload.get("provider_cooldowns") or []),
    }


def wait_for_admin_status(*, timeout_seconds: float = 20.0, interval_seconds: float = 0.5) -> bool:
    """Wait until the freshly started router answers the machine-readable admin status."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if read_admin_status(timeout_seconds=min(2.0, max(0.2, interval_seconds))).get("ready") is True:
            return True
        time.sleep(interval_seconds)
    return False


def install_launchagent() -> int:
    return launchagent_backend().install()


def restart_launchagent() -> int:
    return launchagent_backend().restart()


def stop_launchagent() -> int:
    return launchagent_backend().stop()


def uninstall_launchagent() -> int:
    return launchagent_backend().uninstall()


def service_status() -> int:
    return launchagent_backend().status()


def install_service() -> int:
    return current_service_backend().install()


def restart_service() -> int:
    return current_service_backend().restart()


def stop_service() -> int:
    return current_service_backend().stop()


def uninstall_service() -> int:
    return current_service_backend().uninstall()


def managed_service_status() -> int:
    return current_service_backend().status()


def http_json(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 240) -> int:
    data = json.dumps(payload or {}).encode("utf-8") if method != "GET" else None
    req = urllib.request.Request(router_url(path), data=data, method=method, headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(body)
        return 1
    print(body)
    return 0


def doctor_status() -> dict[str, Any]:
    config = router.load_config()
    service = read_admin_status()
    config_read_path = router.runtime_read_path(router.CONFIG_PATH)
    catalog_read_path = router.runtime_read_path(router.CATALOG_PATH)
    state_read_path = router.runtime_read_path(router.STATE_PATH)
    return {
        "status": "ok" if service.get("ready") else "degraded",
        "config": str(router.CONFIG_PATH),
        "catalog_exists": catalog_read_path.exists(),
        "state_exists": state_read_path.exists(),
        "runtime": {
            "write_dir": str(router.ROUTER_DIR),
            "read_dir": str(router.RUNTIME_PATHS.runtime_read_dir),
            "config_read_path": str(config_read_path),
            "catalog_read_path": str(catalog_read_path),
            "state_read_path": str(state_read_path),
        },
        "service": service,
        "auth": router.auth_status(config),
    }


def doctor(*, json_output: bool = True) -> int:
    status = doctor_status()
    if json_output:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"status: {status['status']}")
        print(f"config: {status['config']}")
        print(f"catalog_exists: {status['catalog_exists']}")
        print(f"state_exists: {status['state_exists']}")
    return 0


def set_key(provider: str) -> int:
    config = router.load_config()
    if provider not in (config.get("providers") or {}):
        print(f"Unknown provider: {provider}", file=sys.stderr)
        return 2
    secret = getpass.getpass(f"Paste the {provider} API key (input hidden): ").strip()
    if not secret:
        print("No key entered; aborted.", file=sys.stderr)
        return 1
    validator = router.PROVIDER_KEY_VALIDATORS.get(provider)
    if validator and not validator(secret):
        print(f"That does not look like a valid {provider} API key; aborted.", file=sys.stderr)
        return 1
    target = router.store_provider_key(provider, config, secret)
    print(f"Stored {provider} key in {target}.")
    router.main_args(["--refresh"])
    print(f"Refreshed. Run `ficelle auth-status` to confirm {provider} is configured.")
    return 0


def remove_key(provider: str) -> int:
    config = router.load_config()
    if provider not in (config.get("providers") or {}):
        print(f"Unknown provider: {provider}", file=sys.stderr)
        return 2
    cleared = router.remove_provider_key(provider, config)
    remaining_legacy_sources = router.legacy_provider_credential_sources(
        provider,
        config,
    )
    if cleared:
        print(f"Removed {provider} key from: {', '.join(cleared)}.")
    else:
        print(f"No stored {provider} key found in the OS store or .env.")
    if remaining_legacy_sources:
        print(
            "Legacy credential fallback sources still apply; remove them manually from: "
            f"{', '.join(remaining_legacy_sources)}. "
            f"{provider} may remain configured until cleanup is complete."
        )
    router.main_args(["--refresh"])
    return 0


def _print_license_status(entitlement: Any, *, json_output: bool) -> int:
    if json_output:
        print(json.dumps(license_ops.status_dict(entitlement)))
        return 0
    if entitlement is None:
        print("License: not activated. Run: ficelle license activate <license-key>")
        return 1
    serving = "yes" if entitlement.is_entitled() else "no (expired / past grace — refresh billing)"
    print(f"License: {entitlement.status} (plan {entitlement.plan or 'n/a'}) — Pro serving: {serving}")
    return 0


def cmd_license(args: argparse.Namespace) -> int:
    # All license operations are shared with the admin daemon (/admin/license) and live in
    # ficelle.license_ops. ensure_installed() is the core-only gate: the closed pack ships the
    # license client, so the free open core reports cleanly and exits non-zero.
    json_output = bool(getattr(args, "json_output", False))
    if json_output and args.action != "status":
        print("--json is supported only for `ficelle license status`.", file=sys.stderr)
        return 2
    try:
        license_ops.ensure_installed()
    except license_ops.LicenseNotInstalled:
        if json_output:
            print(json.dumps(license_ops.status_dict(None)))
            return 1
        print("Ficelle Pro is not installed (this is the free open core); no license to manage.")
        return 1
    if args.action == "status":
        return _print_license_status(
            license_ops.cached_entitlement(), json_output=json_output
        )
    if args.action == "activate":
        # Env fallback lets automation (the bootstrap handoff) pass the key without putting it on
        # the command line, where it would be echoed to logs (bootstrap's run_command prints argv).
        key = args.key or os.getenv("FICELLE_LICENSE_KEY", "")
        if not key:
            print("usage: ficelle license activate <license-key>")
            return 2
        try:
            entitlement = license_ops.activate(key)
        except license_ops.LicenseOperationError as exc:
            print(f"license activation failed: {exc}")
            return 1
        print(f"License activated: {entitlement.status} (plan {entitlement.plan or 'n/a'}).")
        return 0
    if args.action == "refresh":
        try:
            entitlement = license_ops.refresh()
        except license_ops.LicenseOperationError as exc:
            print(f"license refresh failed: {exc}")
            return 1
        print(f"License refreshed: {entitlement.status}.")
        return 0
    if args.action == "deactivate":
        if license_ops.deactivate():
            print("License deactivated locally.")
        else:
            print("warning: service deactivation failed; cleared the local license anyway.")
        return 0
    return 2


def _run_pro_install_locked(pro_install: Any, license_key: str) -> int:
    """Run the installer and persist its terminal state while the caller holds the lock."""
    if not pro_install.best_effort_write_install_status("installing"):
        print("Pro install failed: could not persist installer status.")
        return 1
    try:
        pro_install.install_pro(license_key)
        pro_install.best_effort_write_install_status("restarting")
        print("Ficelle Pro pack installed; restarting the service to load it…")
        restart_status = current_service_backend().restart()
        if restart_status == 0:
            pro_install.best_effort_write_install_status("installed")
        else:
            pro_install.best_effort_write_install_status(
                "failed",
                "Ficelle Pro was installed, but the service restart failed.",
            )
        return restart_status
    except pro_install.ProInstallError as exc:
        pro_install.best_effort_write_install_status("failed", str(exc))
        print(f"Pro install failed: {exc}")
        return 1
    except Exception as exc:
        pro_install.best_effort_write_install_status(
            "failed",
            "The Ficelle Pro installer stopped unexpectedly.",
        )
        print(f"Pro install failed unexpectedly ({type(exc).__name__}).")
        return 1


def cmd_install_pro() -> int:
    # Detached helper for the in-app Pro unlock (spawned by POST /admin/license/install): download
    # and install the Pro pack, then restart the service so it loads. Running detached — not as the
    # daemon — is what makes the restart safe (the daemon can't boot itself out and relaunch). The
    # key arrives via FICELLE_LICENSE_KEY, never argv, which the daemon/logs would echo.
    with active_runtime_environment():
        from ficelle import pro_install

        license_key = os.environ.pop("FICELLE_LICENSE_KEY", "").strip()
        queued_install = os.environ.pop("FICELLE_INSTALL_QUEUED", "") == "1"
        if not license_key:
            pro_install.best_effort_write_install_status("failed", "No license key was provided.")
            print("install-pro: no license key (set FICELLE_LICENSE_KEY).")
            return 2
        try:
            with pro_install.exclusive_install_lock(wait=queued_install):
                return _run_pro_install_locked(pro_install, license_key)
        except pro_install.ProInstallInProgress:
            print("A Ficelle Pro installation is already running.")
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ficelle", description="Ficelle local OpenAI-compatible model router")
    sub = parser.add_subparsers(dest="command")
    for name in ["serve", "refresh", "auth-status", "canary"]:
        sub.add_parser(name)
    sub.add_parser("install")
    sub.add_parser("uninstall")
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("restart")
    sub.add_parser("status")
    doctor_parser = sub.add_parser("doctor")
    doctor_parser.add_argument("--json", action="store_true", dest="json_output", help="Print machine-readable JSON (default)")
    doctor_parser.add_argument("--text", action="store_true", dest="text_output", help="Print a compact human-readable summary")
    sub.add_parser("models")
    sub.add_parser("health")
    export_parser = sub.add_parser("export", help="Export a client configuration")
    export_parser.add_argument(
        "--target",
        choices=["generic", "hermes"],
        default="generic",
        help="Client target (default: generic)",
    )
    set_key_parser = sub.add_parser("set-key", help="Store a provider API key (prompted, hidden input)")
    set_key_parser.add_argument("provider")
    remove_key_parser = sub.add_parser("remove-key", help="Remove a stored provider API key")
    remove_key_parser.add_argument("provider")
    license_parser = sub.add_parser("license", help="Manage the Ficelle Pro license (Pro only)")
    license_parser.add_argument("action", choices=["activate", "status", "refresh", "deactivate"])
    license_parser.add_argument("key", nargs="?", help="License key (for activate)")
    license_parser.add_argument("--json", action="store_true", dest="json_output", help="Machine-readable status (JSON)")
    sub.add_parser("install-pro", help="Install/upgrade to Ficelle Pro from a license key (detached helper; key via FICELLE_LICENSE_KEY)")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return router.main_args(["--serve"])
    if args.command == "refresh":
        return router.main_args(["--refresh"])
    if args.command == "auth-status":
        return router.main_args(["--auth-status"])
    if args.command == "canary":
        return router.main_args(["--canary"])
    if args.command == "install":
        return install_service()
    if args.command == "uninstall":
        return uninstall_service()
    if args.command == "start":
        return install_service()
    if args.command == "stop":
        return stop_service()
    if args.command == "restart":
        return restart_service()
    if args.command == "status":
        return managed_service_status()
    if args.command == "doctor":
        return doctor(json_output=not getattr(args, "text_output", False))
    if args.command == "models":
        return http_json("/v1/models")
    if args.command == "health":
        return http_json("/health")
    if args.command == "export":
        return http_json(f"/admin/export/{args.target}")
    if args.command == "set-key":
        return set_key(args.provider)
    if args.command == "remove-key":
        return remove_key(args.provider)
    if args.command == "license":
        return cmd_license(args)
    if args.command == "install-pro":
        return cmd_install_pro()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
