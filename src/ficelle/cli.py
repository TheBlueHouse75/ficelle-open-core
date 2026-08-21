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

from ficelle.build_identity import package_build_identity
from ficelle.runtime_paths import RuntimePaths
from ficelle.url_security import connectable_http_url
from ficelle.service import (
    SERVER_COMMAND_SIGNATURES,
    LaunchAgentServiceBackend,
    ServiceBackend,
    ServicePaths,
    active_home_pointer_path,
    read_active_service_context,
    select_service_backend,
)
from ficelle.use_cases.provider_auth import (
    invokable_provider_sources,
    provider_key_setup_commands,
    unconfigured_provider_sources,
    unreadable_provider_reasons,
    unusable_key_provider_lines,
    unusable_key_provider_reasons,
)
from ficelle.use_cases.synthetic_health import build_identity_match, describe_build, unverified_build_verdict

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
    from ficelle import coding_certification, license_ops, router, update as update_service  # noqa: E402


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
    if sys.platform == "win32":
        # No `ps` on Windows; WQL pre-filters, the precise match below stays shared.
        result = _run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"CommandLine LIKE '%ficelle%'\" | "
                'ForEach-Object { "$($_.ProcessId) $($_.CommandLine)" }',
            ]
        )
    else:
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
        if any(signature in command for signature in SERVER_COMMAND_SIGNATURES):
            pids.append(pid)
    return pids


# Windows has no SIGKILL; os.kill there hard-terminates via TerminateProcess anyway.
_FORCE_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _pid_alive(pid: int) -> bool:
    """Per-pid liveness probe, so the grace loop does not re-enumerate all processes."""
    if sys.platform == "win32":
        # os.kill(pid, 0) is not a probe on Windows (0 is CTRL_C_EVENT); ask the kernel.
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_stale_servers(*, grace_seconds: float = 2.0) -> list[int]:
    """Terminate orphan Ficelle server processes left behind by the service manager."""
    pids = ficelle_server_pids()
    if not pids:
        return []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            # A dead pid raises ProcessLookupError on POSIX but a plain OSError on Windows.
            pass
    deadline = time.time() + grace_seconds
    remaining = pids
    while remaining and time.time() < deadline:
        time.sleep(0.1)
        remaining = [pid for pid in remaining if _pid_alive(pid)]
    if remaining:
        # One final enumeration before SIGKILL: a pid the OS recycled to an unrelated
        # process since the last liveness sample must not be force-killed.
        still_ficelle = set(ficelle_server_pids())
        for pid in remaining:
            if pid not in still_ficelle:
                continue
            try:
                os.kill(pid, _FORCE_KILL_SIGNAL)
            except OSError:
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
    port = int(config.get("port") or 8646)
    return connectable_http_url(config.get("host"), port, path)


def router_request_headers(path: str) -> dict[str, str]:
    """Authenticate owner CLI calls when the router listens beyond loopback."""
    config = router.load_config()
    if router.bind_host_is_loopback(config.get("host")):
        return {}
    token = router.admin_token() if path == "/admin" or path.startswith("/admin/") else router.api_token()
    return {"Authorization": f"Bearer {token}"}


def read_admin_status(*, timeout_seconds: float = 2.0) -> dict[str, Any]:
    """Read the live machine-readable admin status without exposing credentials."""
    url = router_url("/admin/status.json")
    headers = {"Accept": "application/json", **router_request_headers("/admin/status.json")}
    req = urllib.request.Request(url, headers=headers)
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
    raw_build = payload.get("build")
    return {
        "ready": payload.get("status") in {"ok", "degraded", "fail"},
        "url": url,
        "status": payload.get("status") or "unknown",
        "generated_at": payload.get("generated_at"),
        "model_count": catalog.get("model_count"),
        "available_count": catalog.get("available_count"),
        "model_cooldowns": len(payload.get("model_cooldowns") or []),
        "provider_cooldowns": len(payload.get("provider_cooldowns") or []),
        "build": raw_build if isinstance(raw_build, dict) else None,
    }


def wait_for_admin_status(
    *,
    timeout_seconds: float = 90.0,
    interval_seconds: float = 0.5,
    probe_timeout_seconds: float = 4.0,
) -> bool:
    """Wait until the freshly started router answers the machine-readable admin status.

    The per-probe budget is the one with a measured cause. `serve()` binds before warming,
    and the status reads the *cached* catalog, so a start answers in ~0.5s whether the
    catalog is warm or absent (measured 15/08/2026, both ways) — but the first response
    costs ~1s to build, then ~0.3s. Deriving the probe timeout from `interval_seconds`
    capped it at half a second, so that first answer could never be read.

    The total budget is margin, not a measurement. The 14/08/2026 incident — three starts
    in a row reporting "did not become ready", each naming its own new pid as a busy port,
    with `Ficelle listening on …` followed by `signal 15` in the log — has never been
    reproduced: no measured start comes close to needing 20s, let alone 90s. What fits the
    traces is a *second actor* sending the SIGTERM, and this repo has one: every lifecycle
    command runs `terminate_stale_servers()`, which signals any `-m ficelle.router --serve`
    on the machine, so a sibling session's `start` kills this session's service. Until that
    is understood, the budget stays wide: it costs nothing on a healthy start (the loop
    returns on the first ready answer) and only delays the verdict on a real failure.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if read_admin_status(timeout_seconds=probe_timeout_seconds).get("ready") is True:
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
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **router_request_headers(path),
    }
    req = urllib.request.Request(router_url(path), data=data, method=method, headers=headers)
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
    cli_build = package_build_identity()
    if service.get("ready"):
        # Fail closed like the synthetic-health preflight: a reachable service that does
        # not prove the CLI's runtime content hash is the stale-venv incident, not "ok".
        build_verdict = build_identity_match(cli_build, service.get("build"))
    else:
        # No service answered; "mismatch" would be a guess and "match" a lie.
        build_verdict = unverified_build_verdict("service unreachable — build identity not verified")
    return {
        "status": "ok" if build_verdict["match"] is True else "degraded",
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
        # The service's raw identity already ships under service.build; repeating it here
        # would be a second copy to keep in sync.
        "build": {"cli": cli_build, **build_verdict},
        "service": service,
        "auth": router.auth_status(config),
    }


def doctor_provider_key_lines(auth: dict[str, Any]) -> list[str]:
    """The credential half of the text report.

    `status` above tracks the *service*, so a running router with no provider key
    reads `status: ok` while every completion fails with `credentials unavailable`.
    A diagnostic that stays silent about the commonest reason nothing works is the
    bug, so the text report always states which providers can actually be called.

    A provider holding a key that still cannot be used is reported on its own line, with
    the reason and no `set-key` command: the key is not the missing piece, and offering to
    store it again is advice that cannot work. It is stated even when another provider
    does serve — a resolvable key that reaches nothing is exactly what a doctor is for.
    """
    configured = invokable_provider_sources(auth)
    unusable = unusable_key_provider_reasons(auth)
    unreadable = unreadable_provider_reasons(auth)
    unusable_lines = [f"  {line}" for line in unusable_key_provider_lines(unusable)]
    unreadable_lines = [f"  {line}" for line in unusable_key_provider_lines(unreadable)]
    if configured:
        return [
            f"provider_keys: {', '.join(configured)}",
            *unreadable_lines,
            *unusable_lines,
        ]
    missing = unconfigured_provider_sources(auth)
    # "none configured" would itself be the lie this line exists to avoid when a key is
    # sitting in a store and only the provider entry is incomplete or unreadable.
    headline = "none usable" if unusable or unreadable else "none configured"
    return [
        f"provider_keys: {headline} — no completion can be served.",
        *(f"  {line}" for line in provider_key_setup_commands(missing, router.PROVIDER_KEY_URLS)),
        *unreadable_lines,
        *unusable_lines,
    ]


def doctor(*, json_output: bool = True) -> int:
    status = doctor_status()
    if json_output:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"status: {status['status']}")
        print(f"config: {status['config']}")
        print(f"catalog_exists: {status['catalog_exists']}")
        print(f"state_exists: {status['state_exists']}")
        build = status.get("build") or {}
        if build.get("match") is True:
            print(f"build: service runs the CLI build — {describe_build(build.get('cli') or {})}")
        else:
            print(f"build: {build.get('diagnostic') or 'not verified'}")
        for line in doctor_provider_key_lines(status.get("auth") or {}):
            print(line)
    # Exit code follows the reported status so a script or a CI step can gate on `ficelle doctor`
    # instead of parsing its output. `status` here describes the *service* — reachable and running
    # the same build as this CLI — so a keyless install still reads ok; the provider-key lines
    # above are what surface that.
    return 0 if status.get("status") == "ok" else 1


def set_key(provider: str, *, from_stdin: bool = False, allow_plaintext: bool = False) -> int:
    config = router.load_config()
    if provider not in (config.get("providers") or {}):
        print(f"Unknown provider: {provider}", file=sys.stderr)
        return 2
    if from_stdin:
        # One line piped by a wrapper (ficelle-setup's inline key capture, scripts): the
        # key stays off argv and out of the process environment.
        secret = sys.stdin.readline().strip()
    else:
        secret = getpass.getpass(f"Paste the {provider} API key (input hidden): ").strip()
    if not secret:
        print("No key entered; aborted.", file=sys.stderr)
        return 1
    validator = router.PROVIDER_KEY_VALIDATORS.get(provider)
    if validator and not validator(secret):
        print(f"That does not look like a valid {provider} API key; aborted.", file=sys.stderr)
        return 1
    try:
        target = router.store_provider_key(provider, config, secret, allow_plaintext=allow_plaintext)
    except router.SecretStoreWriteRefused as exc:
        # Refusing beats the old silent downgrade: the key would have landed in a plaintext
        # `.env` while the user believed it went to the OS vault. Name the store, the OS
        # reason when the backend has one, and the two ways out.
        print(
            f"Did not store the {provider} key: {exc}.\n"
            "Fix the store and retry, or pass --allow-plaintext to write it to "
            f"{router.CREDENTIAL_ENV_FILE} in plaintext instead.",
            file=sys.stderr,
        )
        return 1
    print(f"Stored {provider} key in {target}.")
    router.main_args(["--refresh"])
    print(f"Refreshed. Run `ficelle auth-status` to confirm {provider} is configured.")
    return 0


# The `auth.key_source` families a removal cannot reach, mapped to the next move. Mirrors
# ``KEY_SOURCE_REMEDIES`` in ``assets/admin/lib.js``: ``env`` and ``external`` are the two
# nothing inside Ficelle can clear and the two ``remaining_legacy_sources`` never reported, so
# they carry their own instruction. Any other family is named as-is — the legacy line already
# says what to do about a legacy store, and a family added later is still stated.
RESIDUAL_KEY_SOURCE_HINTS = {
    "env": "a process environment variable; unset it where the service starts, then restart",
    "external": "an external credential resolver; clear the key in that integration",
}


def remove_key(provider: str, *, purge_legacy: bool = False) -> int:
    config = router.load_config()
    if provider not in (config.get("providers") or {}):
        print(f"Unknown provider: {provider}", file=sys.stderr)
        return 2
    removal = router.remove_provider_key(provider, config)
    cleared = removal.cleared
    if purge_legacy:
        cleared = [*cleared, *router.purge_legacy_provider_credentials(provider, config)]
    remaining_legacy_sources = router.legacy_provider_credential_sources(
        provider,
        config,
    )
    if cleared:
        print(f"Removed {provider} key from: {', '.join(cleared)}.")
    if not removal.verified:
        print(
            f"Could not verify the {provider} key removal: {'; '.join(removal.unverified)}. "
            f"The key may still be stored — a store that refuses a delete reads as empty "
            f"too, so nothing below can prove otherwise. Retry from a session that can "
            f"reach the store (on Windows, an interactive logon).",
            file=sys.stderr,
        )
    elif not cleared:
        # Withheld when the removal was not confirmed: "no key was stored" is a claim about
        # a store that just declined to answer, and it is the first half of the false
        # all-clear the refusal above replaces.
        print(f"No stored {provider} key found in the OS store or .env.")
    # Resolution re-run after the removal: `key_source` names the store a key would still
    # be read from, or None when none would. It is the verdict for every family, including
    # the two `remaining_legacy_sources` cannot enumerate — a process environment variable
    # and an external resolver (R4) — which used to leave a provider configured behind a
    # clean "removed" line. Computed after the lines above, not before: on a cold CLI process
    # this resolution shells out to `security` once per alias and would hold them back.
    key_source = router.provider_auth_row(provider, config).get("key_source")
    if key_source:
        hint = RESIDUAL_KEY_SOURCE_HINTS.get(key_source, key_source)
        print(f"A {provider} key still resolves from {hint}, so {provider} stays configured.")
    elif removal.verified and not remaining_legacy_sources:
        # Only claimed when nothing is left to qualify it: a locked legacy keychain reads as
        # empty while it stays locked, and an unconfirmed removal reads as empty for the very
        # reason it failed, so "no key resolves" would be a promise for today only.
        print(f"No {provider} key resolves any more.")
    if remaining_legacy_sources:
        sources = ", ".join(remaining_legacy_sources)
        # After an explicit purge these are the ones it could not reach — a locked keychain,
        # or a store another install owns — so the only remaining move is a manual one.
        next_step = (
            "remove them manually"
            if purge_legacy
            else "re-run with --purge-legacy to clear them"
        )
        print(f"Legacy credential fallback sources still apply: {sources}. Please {next_step}.")
    router.main_args(["--refresh"])
    # Non-zero on an unconfirmed removal so a script revoking a key can gate on it. A
    # revocation that may not have happened is a failure, whatever else the run cleared.
    return 0 if removal.verified else 1


def _print_license_status(entitlement: Any, *, json_output: bool) -> int:
    if json_output:
        print(json.dumps(license_ops.status_dict(entitlement)))
        return 0
    if entitlement is None:
        print("License: not activated. Set FICELLE_LICENSE_KEY or run `ficelle license activate` interactively.")
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
        # Automation uses the environment; interactive terminals use a hidden prompt. License
        # keys are never accepted on argv, where shell history and process listings expose them.
        key = os.getenv("FICELLE_LICENSE_KEY", "").strip()
        if not key and sys.stdin.isatty():
            key = getpass.getpass("License key: ").strip()
        if not key:
            print("Set FICELLE_LICENSE_KEY or run `ficelle license activate` interactively.")
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


def cmd_demo(args: argparse.Namespace) -> int:
    """Show the failover on a real request, with the leading candidate forced to fail.

    Runs in-process against the local configuration rather than through the served
    endpoint: the fault injection lives in the demo, so the running router keeps no
    way of being asked to fail a request.
    """
    from ficelle.use_cases.failover_demo import failover_demo_payload, render_failover_demo

    try:
        result = router.run_failover_demo(
            router.load_config(),
            profile=getattr(args, "profile", None),
            prompt=getattr(args, "prompt", None),
            knock_out=int(getattr(args, "knock_out", 1) or 1),
        )
    except (ValueError, RuntimeError) as exc:
        # These carry the actionable message (`ficelle doctor`, add a key, pick a
        # profile). Printing the class name on top of it would only bury it.
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - a demo must not hand a user a traceback
        print(f"The failover demo could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json_output", False):
        print(json.dumps(failover_demo_payload(result), ensure_ascii=False, indent=2))
    else:
        print(render_failover_demo(result, key_urls=router.PROVIDER_KEY_URLS))
    # A pool that answered nothing is a real finding, not a crash: report it as a
    # non-zero exit so a scripted check can tell the two apart.
    return 0 if result.answered_by is not None else 1


def cmd_update(args: argparse.Namespace) -> int:
    try:
        if bool(getattr(args, "recover", False)):
            update_service.recover_interrupted_update()
            return 0
        if bool(getattr(args, "apply", False)):
            return update_service.apply_update()
        if bool(getattr(args, "install", False)):
            status = update_service.spawn_update()
            if getattr(args, "json_output", False):
                print(json.dumps(update_service.public_update_status(), ensure_ascii=False))
            else:
                print(
                    "Ficelle update queued for "
                    f"{status.get('latest_version') or 'the latest release'}."
                )
            return 0
        status = update_service.check_for_updates(force=True)
    except update_service.UpdateInProgress as exc:
        if getattr(args, "json_output", False):
            print(json.dumps(update_service.public_update_status(), ensure_ascii=False))
        else:
            print(str(exc), file=sys.stderr)
        return 1
    except (update_service.UpdateError, update_service.UpdateUnavailable) as exc:
        if getattr(args, "json_output", False):
            print(json.dumps(update_service.public_update_status(), ensure_ascii=False))
        else:
            print(str(exc), file=sys.stderr)
        return 1
    except OSError:
        if getattr(args, "json_output", False):
            print(json.dumps(update_service.public_update_status(), ensure_ascii=False))
        else:
            print("Ficelle could not access the local update state.", file=sys.stderr)
        return 1
    if getattr(args, "json_output", False):
        print(json.dumps(update_service.public_update_status(), ensure_ascii=False))
        return 0 if status.get("status") != "error" else 1
    if status.get("update_available"):
        print(
            f"Ficelle {status.get('latest_version')} is available "
            f"(current: {status.get('current_version')})."
        )
        if status.get("release_url"):
            print(f"Release notes: {status['release_url']}")
        if status.get("pro_update_required"):
            print(str(status.get("message") or "A compatible Pro update is required."))
        else:
            print("Run `ficelle update --install` or use the Admin Control Center to install it.")
    elif status.get("status") == "error":
        print(str(status.get("message") or "Ficelle update check failed."), file=sys.stderr)
        return 1
    else:
        print(f"Ficelle is up to date ({status.get('current_version')}).")
    return 0


def cmd_access_token(scope: str) -> int:
    """Print an explicitly requested owner token for an exposed listener."""
    token = router.admin_token() if scope == "admin" else router.api_token()
    print(token)
    return 0


def cmd_coding_certification(args: argparse.Namespace) -> int:
    action = str(getattr(args, "coding_certification_action", "status"))
    if action == "refresh":
        status = coding_certification.refresh_cache(
            RUNTIME_PATHS.coding_certification_cache_path,
            RUNTIME_PATHS.coding_certification_status_path,
        )
    else:
        status = coding_certification.public_status(
            RUNTIME_PATHS.coding_certification_cache_path,
            RUNTIME_PATHS.coding_certification_status_path,
        )
    if bool(getattr(args, "json_output", False)):
        print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Coding certification: {status.get('status', 'unavailable')}")
        print(f"Manifest: {status.get('manifest_id') or 'none'}")
        print(f"Expires: {status.get('expires_at') or 'unknown'}")
        print(f"Certified models: {status.get('certification_count', 0)}")
        if status.get("message"):
            print(str(status["message"]), file=sys.stderr)
    return 0 if status.get("status") in {"valid", "valid_cache"} else 1


def _print_synthetic_health_result(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    print(f"Synthetic health: {payload.get('status', 'unknown')}")
    if payload.get("run_id"):
        print(f"Run: {payload['run_id']}")
    cases = payload.get("cases")
    if isinstance(cases, dict):
        print(f"Cases: {cases.get('completed', 0)}/{cases.get('planned', 0)}")
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict) and artifacts.get("summary_markdown"):
        print(f"Report: {artifacts['summary_markdown']}")


def cmd_synthetic_health(args: argparse.Namespace) -> int:
    from ficelle import synthetic_health

    action = getattr(args, "synthetic_action", None)
    json_output = bool(getattr(args, "json_output", False))
    try:
        if action == "run":
            payload, exit_code = synthetic_health.run_synthetic_health(
                FICELLE_HOME,
                depth=args.depth,
                max_duration_seconds=args.max_duration_seconds,
            )
        elif action == "status":
            payload = synthetic_health.synthetic_health_status(FICELLE_HOME)
            exit_code = 0
        elif action == "report":
            payload, exit_code = synthetic_health.load_run_report(FICELLE_HOME, args.run_id)
        elif action == "resume":
            payload, exit_code = synthetic_health.resume_synthetic_health(
                FICELLE_HOME,
                args.run_id,
                max_duration_seconds=args.max_duration_seconds,
            )
        elif action == "rerun":
            payload, exit_code = synthetic_health.rerun_failed_synthetic_health(
                FICELLE_HOME,
                args.run_id,
                max_duration_seconds=args.max_duration_seconds,
            )
        elif action == "schedule":
            schedule_action = getattr(args, "schedule_action", None)
            if schedule_action == "install":
                hour_text, separator, minute_text = args.time.partition(":")
                if not separator:
                    raise synthetic_health.SyntheticHealthError("schedule time must use HH:MM")
                weekday = {"sun": 1, "mon": 2, "tue": 3, "wed": 4, "thu": 5, "fri": 6, "sat": 7}[args.weekday]
                payload = synthetic_health.install_weekly_schedule(
                    FICELLE_HOME,
                    weekday=weekday,
                    hour=int(hour_text),
                    minute=int(minute_text),
                    max_duration_seconds=args.max_duration_seconds,
                )
            elif schedule_action == "remove":
                payload = synthetic_health.remove_weekly_schedule()
            else:
                payload = synthetic_health.schedule_status()
            exit_code = 0
        else:
            raise synthetic_health.SyntheticHealthError("missing synthetic-health action")
    except (synthetic_health.SyntheticHealthError, ValueError) as exc:
        payload = {"status": "error", "error": str(exc)}
        exit_code = 2
    _print_synthetic_health_result(payload, json_output=json_output)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ficelle", description="Ficelle local OpenAI-compatible model router")
    sub = parser.add_subparsers(dest="command")
    for name in ["serve", "refresh", "auth-status"]:
        sub.add_parser(name)
    canary_parser = sub.add_parser("canary")
    canary_parser.add_argument("--profile", action="append", default=[])
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
    demo_parser = sub.add_parser(
        "demo",
        help="Show the failover: knock out the model that would have answered, on a real request",
    )
    demo_parser.add_argument("--profile", help=f"Routing profile to demonstrate (default: {router.DEFAULT_CHAT_COMPLETION_MODEL})")
    demo_parser.add_argument("--prompt", help="Prompt to send (default: a one-sentence question)")
    demo_parser.add_argument(
        "--knock-out",
        type=int,
        default=1,
        dest="knock_out",
        help="How many leading candidates to fail (default: 1; one candidate is always left alive)",
    )
    demo_parser.add_argument("--json", action="store_true", dest="json_output", help="Print machine-readable JSON")
    set_key_parser = sub.add_parser("set-key", help="Store a provider API key (prompted, hidden input)")
    set_key_parser.add_argument("provider")
    set_key_parser.add_argument(
        "--stdin",
        action="store_true",
        dest="from_stdin",
        help="Read the key from standard input instead of prompting (for wrappers and scripts; keeps the key off argv).",
    )
    set_key_parser.add_argument(
        "--allow-plaintext",
        action="store_true",
        dest="allow_plaintext",
        help="Write the key to FICELLE_HOME/.env in plaintext when this host's secret store refuses it (default: refuse).",
    )
    remove_key_parser = sub.add_parser("remove-key", help="Remove a stored provider API key")
    remove_key_parser.add_argument("provider")
    remove_key_parser.add_argument(
        "--purge-legacy",
        action="store_true",
        dest="purge_legacy",
        help="Also delete the key from read-only legacy stores (previous Hermes/Ficelle keychains and .env files), which the router still resolves. Other tools reading them lose access.",
    )
    license_parser = sub.add_parser("license", help="Manage the Ficelle Pro license (Pro only)")
    license_parser.add_argument("action", choices=["activate", "status", "refresh", "deactivate"])
    license_parser.add_argument("--json", action="store_true", dest="json_output", help="Machine-readable status (JSON)")
    access_token_parser = sub.add_parser(
        "access-token",
        help="Print an owner-only token for a non-loopback listener",
    )
    access_token_parser.add_argument("scope", choices=["admin", "api"])
    sub.add_parser("install-pro", help="Install/upgrade to Ficelle Pro from a license key (detached helper; key via FICELLE_LICENSE_KEY)")
    update_parser = sub.add_parser("update", help="Check for or install a verified Ficelle update")
    update_group = update_parser.add_mutually_exclusive_group()
    update_group.add_argument("--check", action="store_true", help="Check the release service (default).")
    update_group.add_argument("--install", action="store_true", help="Queue the verified update and restart the service.")
    update_group.add_argument("--recover", action="store_true", help=argparse.SUPPRESS)
    update_parser.add_argument("--json", action="store_true", dest="json_output", help="Print machine-readable status.")
    update_parser.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)
    coding_parser = sub.add_parser(
        "coding-certification",
        help="Inspect or refresh the signed auto-coding certification manifest",
    )
    coding_sub = coding_parser.add_subparsers(dest="coding_certification_action", required=True)
    for action in ("status", "refresh"):
        action_parser = coding_sub.add_parser(action)
        action_parser.add_argument("--json", action="store_true", dest="json_output")
    synthetic_parser = sub.add_parser("synthetic-health", help="Run and inspect deep synthetic user health checks")
    synthetic_sub = synthetic_parser.add_subparsers(dest="synthetic_action", required=True)
    synthetic_run = synthetic_sub.add_parser("run", help="Run the synthetic user corpus")
    synthetic_run.add_argument("--depth", choices=["deep"], default="deep")
    synthetic_run.add_argument("--max-duration-seconds", type=float)
    synthetic_run.add_argument("--json", action="store_true", dest="json_output")
    synthetic_status = synthetic_sub.add_parser("status", help="Show active and latest run status")
    synthetic_status.add_argument("--json", action="store_true", dest="json_output")
    synthetic_report = synthetic_sub.add_parser("report", help="Show a completed run report")
    report_selection = synthetic_report.add_mutually_exclusive_group()
    report_selection.add_argument("--latest", action="store_true", help="Show the latest run (default)")
    report_selection.add_argument("--run-id")
    synthetic_report.add_argument("--json", action="store_true", dest="json_output")
    synthetic_resume = synthetic_sub.add_parser("resume", help="Resume an identity-compatible interrupted run")
    synthetic_resume.add_argument("--run-id", required=True)
    synthetic_resume.add_argument("--max-duration-seconds", type=float)
    synthetic_resume.add_argument("--json", action="store_true", dest="json_output")
    synthetic_rerun = synthetic_sub.add_parser("rerun", help="Create a child run for failed cases")
    synthetic_rerun.add_argument("--run-id", required=True)
    synthetic_rerun.add_argument("--failed", action="store_true", required=True)
    synthetic_rerun.add_argument("--max-duration-seconds", type=float)
    synthetic_rerun.add_argument("--json", action="store_true", dest="json_output")
    synthetic_schedule = synthetic_sub.add_parser("schedule", help="Manage the disabled-by-default macOS weekly launcher")
    schedule_sub = synthetic_schedule.add_subparsers(dest="schedule_action", required=True)
    schedule_status_parser = schedule_sub.add_parser("status")
    schedule_status_parser.add_argument("--json", action="store_true", dest="json_output")
    schedule_install = schedule_sub.add_parser("install")
    schedule_install.add_argument("--weekday", choices=["mon", "tue", "wed", "thu", "fri", "sat", "sun"], required=True)
    schedule_install.add_argument("--time", required=True, help="Local time in HH:MM format")
    schedule_install.add_argument("--max-duration-seconds", type=float)
    schedule_install.add_argument("--json", action="store_true", dest="json_output")
    schedule_remove = schedule_sub.add_parser("remove")
    schedule_remove.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return router.main_args(["--serve"])
    if args.command == "refresh":
        return router.main_args(["--refresh"])
    if args.command == "auth-status":
        return router.main_args(["--auth-status"])
    if args.command == "canary":
        canary_args = ["--canary"]
        for profile_id in args.profile:
            canary_args.extend(["--canary-profile", profile_id])
        return router.main_args(canary_args)
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
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "set-key":
        return set_key(
            args.provider,
            from_stdin=bool(getattr(args, "from_stdin", False)),
            allow_plaintext=bool(getattr(args, "allow_plaintext", False)),
        )
    if args.command == "remove-key":
        return remove_key(args.provider, purge_legacy=bool(getattr(args, "purge_legacy", False)))
    if args.command == "license":
        return cmd_license(args)
    if args.command == "access-token":
        return cmd_access_token(args.scope)
    if args.command == "install-pro":
        return cmd_install_pro()
    if args.command == "update":
        return cmd_update(args)
    if args.command == "coding-certification":
        return cmd_coding_certification(args)
    if args.command == "synthetic-health":
        return cmd_synthetic_health(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
