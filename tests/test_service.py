from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ficelle.service import (
    ServicePaths,
    SystemdUserServiceBackend,
    UnsupportedServiceBackend,
    persist_active_service_context,
    read_active_service_context,
    select_service_backend,
)


def make_paths(
    tmp_path: Path,
    *,
    runtime_dir: Path | None = None,
    hermes_home: Path | None = None,
    persist_home: bool = False,
) -> ServicePaths:
    ficelle_home = tmp_path / ".ficelle"
    return ServicePaths(
        ficelle_home=ficelle_home,
        runtime_dir=runtime_dir or ficelle_home,
        label="com.ficelle.router",
        plist=tmp_path / "Library" / "LaunchAgents" / "com.ficelle.router.plist",
        systemd_unit=tmp_path / ".config" / "systemd" / "user" / "com.ficelle.router.service",
        install_python=Path(sys.executable),
        hermes_home=hermes_home,
        active_home_pointer=tmp_path / ".config" / "ficelle" / "active-home"
        if persist_home
        else None,
    )


def test_active_service_context_roundtrip_preserves_optional_hermes_home(tmp_path):
    ficelle_home = tmp_path / "custom-ficelle"
    runtime_dir = tmp_path / "custom-runtime"
    hermes_home = tmp_path / "custom-hermes"
    ficelle_home.mkdir()
    runtime_dir.mkdir()
    pointer = tmp_path / ".config" / "ficelle" / "active-home"

    assert persist_active_service_context(
        ficelle_home,
        pointer,
        runtime_dir=runtime_dir,
        hermes_home=hermes_home,
    )
    assert read_active_service_context(pointer) == (
        ficelle_home,
        runtime_dir,
        hermes_home,
    )


def test_active_service_context_defaults_legacy_pointer_runtime_to_ficelle_home(
    tmp_path,
):
    ficelle_home = tmp_path / "custom-ficelle"
    ficelle_home.mkdir()
    pointer = tmp_path / "active-home"
    pointer.write_text(str(ficelle_home), encoding="utf-8")

    assert read_active_service_context(pointer) == (
        ficelle_home,
        ficelle_home,
        None,
    )


def test_select_service_backend_uses_launchagent_on_darwin(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    backend = select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=lambda cmd: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "501",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.name == "launchagent"
    assert backend.install() == 0
    assert ["launchctl", "bootstrap", "gui/501", str(paths.plist)] in calls
    payload = backend.plist_payload()
    assert payload["WorkingDirectory"] == str(paths.runtime_dir)
    assert payload["EnvironmentVariables"] == {"FICELLE_HOME": str(paths.ficelle_home)}
    assert payload["StandardOutPath"] == str(paths.log_dir / "ficelle.log")
    assert "HERMES_HOME" not in payload["EnvironmentVariables"]


def test_select_service_backend_uses_systemd_user_on_linux(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    backend = select_service_backend(
        platform_name="linux",
        paths=paths,
        run_command=lambda cmd: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "1000",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert isinstance(backend, SystemdUserServiceBackend)
    assert backend.name == "systemd-user"
    assert backend.install() == 0
    assert paths.systemd_unit.exists()
    unit = paths.systemd_unit.read_text(encoding="utf-8")
    assert "ExecStart=" + sys.executable + " -m ficelle.router --serve" in unit
    assert f"WorkingDirectory={paths.runtime_dir}" in unit
    assert f'Environment="FICELLE_HOME={paths.ficelle_home}"' in unit
    assert "HERMES_HOME" not in unit
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "--now", paths.systemd_unit.name] in calls


def test_launchagent_persists_custom_hermes_home_for_hermes_target(tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    paths = make_paths(tmp_path, hermes_home=hermes_home)
    backend = select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "501",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.plist_payload()["EnvironmentVariables"] == {
        "FICELLE_HOME": str(paths.ficelle_home),
        "HERMES_HOME": str(hermes_home),
    }


def test_systemd_persists_custom_hermes_home_for_hermes_target(tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    paths = make_paths(tmp_path, hermes_home=hermes_home)
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    unit = backend.unit_payload()

    assert f'Environment="FICELLE_HOME={paths.ficelle_home}"' in unit
    assert f'Environment="HERMES_HOME={hermes_home}"' in unit


def test_systemd_quotes_complete_environment_assignments(tmp_path):
    ficelle_home = tmp_path / 'Ficelle "home" \\ credentials'
    runtime_dir = tmp_path / 'Runtime "dir" \\ state'
    hermes_home = tmp_path / 'Hermes "home" \\ config'
    paths = ServicePaths(
        ficelle_home=ficelle_home,
        runtime_dir=runtime_dir,
        label="com.ficelle.router",
        plist=tmp_path / "router.plist",
        systemd_unit=tmp_path / "router.service",
        install_python=Path(sys.executable),
        hermes_home=hermes_home,
    )
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr=""
        ),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    unit = backend.unit_payload()

    def escaped(path: Path) -> str:
        return str(path).replace("\\", "\\\\").replace('"', '\\"')

    assert f'Environment="FICELLE_HOME={escaped(ficelle_home)}"' in unit
    assert f'Environment="FICELLE_RUNTIME_DIR={escaped(runtime_dir)}"' in unit
    assert f'Environment="HERMES_HOME={escaped(hermes_home)}"' in unit


def test_service_reads_legacy_runtime_but_keeps_artifacts_canonical(tmp_path):
    runtime_dir = tmp_path / ".hermes" / "ficelle"
    runtime_dir.mkdir(parents=True)
    state_path = runtime_dir / "state.json"
    state_path.write_text('{"legacy": true}\n', encoding="utf-8")
    paths = make_paths(tmp_path, runtime_dir=runtime_dir, persist_home=True)
    backend = select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=lambda cmd: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr=""
        ),
        uid_provider=lambda: "501",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    payload = backend.plist_payload()

    assert payload["WorkingDirectory"] == str(paths.ficelle_home)
    assert payload["StandardOutPath"] == str(
        paths.ficelle_home / "logs" / "ficelle.log"
    )
    assert payload["EnvironmentVariables"] == {
        "FICELLE_HOME": str(paths.ficelle_home),
        "FICELLE_RUNTIME_DIR": str(runtime_dir),
    }
    assert backend.install() == 0
    assert paths.ficelle_home.is_dir()
    assert (paths.ficelle_home / "logs").is_dir()
    assert not (runtime_dir / "logs").exists()
    assert state_path.read_text(encoding="utf-8") == '{"legacy": true}\n'
    assert read_active_service_context(paths.active_home_pointer) == (
        paths.ficelle_home,
        runtime_dir,
        None,
    )


def test_service_paths_reads_optional_hermes_target_context(monkeypatch, tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    paths = ServicePaths(
        ficelle_home=tmp_path / ".ficelle",
        runtime_dir=tmp_path / ".ficelle",
        label="com.ficelle.router",
        plist=tmp_path / "router.plist",
        systemd_unit=tmp_path / "router.service",
        install_python=Path(sys.executable),
    )

    assert paths.hermes_home == hermes_home


def test_systemd_user_restart_reuses_existing_unit(tmp_path):
    calls = []
    paths = make_paths(tmp_path, persist_home=True)
    paths.systemd_unit.parent.mkdir(parents=True)
    paths.systemd_unit.write_text("existing", encoding="utf-8")
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.restart() == 0
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "restart", paths.systemd_unit.name],
    ]
    assert f'Environment="FICELLE_HOME={paths.ficelle_home}"' in paths.systemd_unit.read_text(
        encoding="utf-8"
    )
    assert read_active_service_context(paths.active_home_pointer) == (
        paths.ficelle_home,
        paths.runtime_dir,
        None,
    )


def test_systemd_user_uninstall_removes_unit_and_reloads(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    paths.systemd_unit.parent.mkdir(parents=True)
    paths.systemd_unit.write_text("unit", encoding="utf-8")
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.uninstall() == 0
    assert not paths.systemd_unit.exists()
    assert calls == [
        ["systemctl", "--user", "disable", "--now", paths.systemd_unit.name],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_select_service_backend_rejects_windows_platform(tmp_path, capsys):
    backend = select_service_backend(
        platform_name="win32",
        paths=make_paths(tmp_path),
        run_command=lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "1000",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert isinstance(backend, UnsupportedServiceBackend)
    assert backend.install() == 1
    assert "macOS LaunchAgent and Linux systemd --user" in capsys.readouterr().err


def _launchagent(paths, run_command):
    return select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=run_command,
        uid_provider=lambda: "501",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )


def test_restart_reuses_the_interpreter_the_plist_already_records(tmp_path, capsys):
    # AGENTS.md warns the agent must run under the Python setup used, and the CLI honoured that
    # at setup then re-decided it on every restart from sys.executable. Running `ficelle restart`
    # from a second install or an ephemeral uvx environment silently repointed the service.
    import plistlib

    installed_python = tmp_path / "venv" / "bin" / "python"
    installed_python.parent.mkdir(parents=True)
    installed_python.write_text("#!/bin/sh\n", encoding="utf-8")

    paths = make_paths(tmp_path)
    paths.plist.parent.mkdir(parents=True, exist_ok=True)
    with paths.plist.open("wb") as handle:
        plistlib.dump({"ProgramArguments": [str(installed_python), "-m", "ficelle.router", "--serve"]}, handle)

    backend = _launchagent(paths, lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    assert backend.paths.install_python != installed_python, "precondition: a different interpreter is running"

    assert backend.restart() == 0

    with paths.plist.open("rb") as handle:
        rewritten = plistlib.load(handle)
    assert rewritten["ProgramArguments"][0] == str(installed_python), "the recorded interpreter must survive a restart"
    assert "reusing the interpreter" in capsys.readouterr().out


def test_restart_repoints_when_the_recorded_interpreter_cannot_import_ficelle(tmp_path, capsys):
    import plistlib

    broken_python = tmp_path / "broken" / "python"
    broken_python.parent.mkdir(parents=True)
    broken_python.write_text("#!/bin/sh\n", encoding="utf-8")

    paths = make_paths(tmp_path)
    paths.plist.parent.mkdir(parents=True, exist_ok=True)
    with paths.plist.open("wb") as handle:
        plistlib.dump({"ProgramArguments": [str(broken_python), "-m", "ficelle.router", "--serve"]}, handle)

    def run_command(cmd):
        # The import probe fails for the recorded interpreter; launchctl calls succeed.
        if cmd[:2] == [str(broken_python), "-c"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ModuleNotFoundError")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    backend = _launchagent(paths, run_command)

    assert backend.restart() == 0

    with paths.plist.open("rb") as handle:
        rewritten = plistlib.load(handle)
    assert rewritten["ProgramArguments"][0] == str(paths.install_python), "a dead interpreter must be replaced"
    assert "unusable" in capsys.readouterr().out
