from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ficelle.service import ServicePaths, SystemdUserServiceBackend, UnsupportedServiceBackend, select_service_backend


def make_paths(tmp_path: Path) -> ServicePaths:
    return ServicePaths(
        hermes_home=tmp_path / ".hermes",
        ficelle_dir=tmp_path / ".hermes" / "ficelle",
        label="com.ficelle.router",
        plist=tmp_path / "Library" / "LaunchAgents" / "com.ficelle.router.plist",
        systemd_unit=tmp_path / ".config" / "systemd" / "user" / "com.ficelle.router.service",
        install_python=Path(sys.executable),
        log_dir=tmp_path / ".hermes" / "logs",
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
    assert "ExecStart=" + sys.executable + " -m ficelle.router --serve" in paths.systemd_unit.read_text()
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "--now", paths.systemd_unit.name] in calls


def test_systemd_user_restart_reuses_existing_unit(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
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
    assert calls == [["systemctl", "--user", "restart", paths.systemd_unit.name]]


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
