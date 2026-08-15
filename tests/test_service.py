from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ficelle.service import (
    ServicePaths,
    SystemdUserServiceBackend,
    UnsupportedServiceBackend,
    WindowsScheduledTaskBackend,
    persist_active_service_context,
    read_active_service_context,
    select_service_backend,
    windows_headless_python,
    windows_task_account,
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


def _launchagent_backend(
    paths,
    calls,
    *,
    ready: bool,
    port_holder_pid: int | None = None,
    launchd_pid: int | None = None,
):
    """A LaunchAgent backend whose `launchctl`/`lsof` answers are scripted."""

    def run(cmd):
        calls.append(cmd)
        if cmd[0] == "lsof":
            if port_holder_pid is None:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
            stdout = (
                "COMMAND     PID             USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME\n"
                f"python3.1 {port_holder_pid} cyril    3u  IPv4 0x7b5666      0t0  TCP 127.0.0.1:8646 (LISTEN)\n"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[:2] == ["launchctl", "print"]:
            if launchd_pid is None:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not loaded")
            stdout = f"com.ficelle.router = {{\n\tstate = running\n\tpid = {launchd_pid}\n\truns = 1\n}}\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=run,
        uid_provider=lambda: "501",
        wait_for_ready=lambda: ready,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )


def _launchctl_bootouts(calls) -> list[list[str]]:
    return [cmd for cmd in calls if cmd[:2] == ["launchctl", "bootout"]]


def test_launchagent_keeps_a_slow_service_that_holds_the_port_itself(tmp_path, capsys):
    """A service that bound the port and is only slow to answer must survive the wait.

    Regression (14/08/2026): three `ficelle start` runs in a row killed the service they
    had just started. `wait_for_ready()` expired during the catalog warm, the fallback
    diagnosis ran `lsof`, found the pid `launchctl kickstart` had just produced, printed it
    as a busy port — a different pid each run — then booted the agent out. The log showed
    the truth: `Ficelle listening on …` followed by `signal 15 received`.
    """
    calls = []
    paths = make_paths(tmp_path)
    backend = _launchagent_backend(
        paths, calls, ready=False, port_holder_pid=41107, launchd_pid=41107
    )

    assert backend.install() == 1
    err = capsys.readouterr().err
    assert "did not become ready" in err
    assert "holds port 8646 itself" in err
    assert "Left it running" in err
    assert "looks busy" not in err
    # Only install()'s own pre-bootstrap bootout: the started service is not booted out.
    assert len(_launchctl_bootouts(calls)) == 1


def test_launchagent_boots_out_when_a_foreign_process_holds_the_port(tmp_path, capsys):
    """The anti-relaunch-loop guard still fires for the case it was written for."""
    calls = []
    paths = make_paths(tmp_path)
    backend = _launchagent_backend(
        paths, calls, ready=False, port_holder_pid=999, launchd_pid=41107
    )

    assert backend.install() == 1
    err = capsys.readouterr().err
    assert "port 8646 is held by" in err
    assert "Booted the agent out" in err
    assert len(_launchctl_bootouts(calls)) == 2


def test_launchagent_probes_the_canonical_port_not_the_legacy_one(tmp_path):
    """The port the service binds comes from the canonical config, legacy root second.

    A legacy install keeps `runtime_dir` as a read-only compatibility root. Reading the port
    from there probes a port nobody holds, and the caller reads "no holder" as "the service
    never bound" — then boots out a service that is listening on the canonical port.
    """
    runtime_dir = tmp_path / ".hermes" / "ficelle"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "config.json").write_text('{"port": 8646}\n', encoding="utf-8")
    paths = make_paths(tmp_path, runtime_dir=runtime_dir)
    paths.ficelle_home.mkdir(parents=True)
    (paths.ficelle_home / "config.json").write_text('{"port": 8700}\n', encoding="utf-8")
    calls = []
    backend = _launchagent_backend(paths, calls, ready=True)

    assert backend.configured_port() == 8700


def test_launchagent_boots_out_when_nothing_holds_the_port(tmp_path, capsys):
    """Nothing listening means the service never bound: that is the relaunch loop."""
    calls = []
    paths = make_paths(tmp_path)
    backend = _launchagent_backend(paths, calls, ready=False, launchd_pid=41107)

    assert backend.install() == 1
    err = capsys.readouterr().err
    assert "is held by" not in err
    assert "Booted the agent out" in err
    assert len(_launchctl_bootouts(calls)) == 2


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


def test_systemd_persists_the_context_of_a_service_it_leaves_running(tmp_path, capsys):
    """systemd escaped the 14/08/2026 kill — it never stopped the unit on a readiness
    timeout — but it also never described the service it left running. `ficelle health`, the
    follow-up the other two backends now recommend, reads that context."""
    paths = make_paths(tmp_path, persist_home=True)
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: False,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.install() == 1
    assert "left running" in capsys.readouterr().err
    assert read_active_service_context(paths.active_home_pointer) == (
        paths.ficelle_home,
        paths.ficelle_home,
        None,
    )


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


def _windows_backend(paths, calls=None, *, account="EXAMPLE\\cyril", ready=True):
    recorded = calls if calls is not None else []

    def run(cmd):
        recorded.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return WindowsScheduledTaskBackend(
        paths=paths,
        run_command=run,
        wait_for_ready=lambda: ready,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
        account_provider=lambda: account,
    )


def test_select_service_backend_uses_scheduled_task_on_windows(tmp_path):
    backend = select_service_backend(
        platform_name="win32",
        paths=make_paths(tmp_path),
        run_command=lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "1000",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert isinstance(backend, WindowsScheduledTaskBackend)
    assert backend.name == "scheduled-task"


def test_scheduled_task_install_registers_and_runs_the_task(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    backend = _windows_backend(paths, calls)

    assert backend.install() == 0
    assert backend.task_xml.exists()
    assert ["schtasks", "/Create", "/TN", paths.label, "/XML", str(backend.task_xml), "/F"] in calls
    assert ["schtasks", "/Run", "/TN", paths.label] in calls
    payload = backend.task_xml.read_text(encoding="utf-16")
    assert '<?xml version="1.0" encoding="UTF-16"?>' in payload
    assert "<UserId>EXAMPLE\\cyril</UserId>" in payload
    assert "-m ficelle.windows_entry" in payload
    assert f"FICELLE_HOME={paths.ficelle_home}" in payload
    assert f"<WorkingDirectory>{paths.ficelle_home}</WorkingDirectory>" in payload
    assert "<LogonTrigger>" in payload
    assert "<RunLevel>LeastPrivilege</RunLevel>" in payload
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in payload
    assert "FICELLE_RUNTIME_DIR=" not in payload
    assert "HERMES_HOME=" not in payload


def test_scheduled_task_passes_runtime_and_hermes_context_as_assignments(tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    runtime_dir = tmp_path / ".hermes" / "ficelle"
    runtime_dir.mkdir(parents=True)
    paths = make_paths(tmp_path, runtime_dir=runtime_dir, hermes_home=hermes_home)
    backend = _windows_backend(paths)

    assert backend.entry_arguments() == [
        "-m",
        "ficelle.windows_entry",
        f"FICELLE_HOME={paths.ficelle_home}",
        f"FICELLE_RUNTIME_DIR={runtime_dir}",
        f"HERMES_HOME={hermes_home}",
    ]


def test_scheduled_task_stop_and_uninstall_use_schtasks(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    backend = _windows_backend(paths, calls)
    assert backend.install() == 0
    assert backend.task_xml.exists()

    assert backend.stop() == 0
    assert ["schtasks", "/End", "/TN", paths.label] in calls
    assert backend.uninstall() == 0
    assert ["schtasks", "/Delete", "/TN", paths.label, "/F"] in calls
    assert not backend.task_xml.exists()


def test_scheduled_task_leaves_a_slow_service_registered(tmp_path, capsys):
    """Same 14/08/2026 lesson as the LaunchAgent: a slow start is not a failed start.

    Ending the task killed a service that may well have bound the port, and Task Scheduler
    already bounds its own retries (`RestartOnFailure`, 10 attempts), so the local guard
    bought little and cost the running service.
    """
    calls = []
    paths = make_paths(tmp_path)
    backend = _windows_backend(paths, calls, ready=False)

    assert backend.install() == 1
    err = capsys.readouterr().err
    assert "did not become ready" in err
    assert "Left the task registered" in err
    # Only install()'s own pre-run /End: the started task is not ended.
    assert len([cmd for cmd in calls if cmd[:2] == ["schtasks", "/End"]]) == 1


def test_scheduled_task_install_persists_active_context(tmp_path):
    paths = make_paths(tmp_path, persist_home=True)
    backend = _windows_backend(paths)

    assert backend.install() == 0
    assert read_active_service_context(paths.active_home_pointer) == (
        paths.ficelle_home,
        paths.runtime_dir,
        None,
    )


def test_windows_task_account_prefers_username_over_getpass_chain(monkeypatch):
    # MSYS2/Git Bash export LOGNAME/USER, which getpass.getuser() would prefer even
    # though schtasks cannot map those names to a Windows account.
    monkeypatch.setenv("USERDOMAIN", "EXAMPLE")
    monkeypatch.setenv("USERNAME", "cyril")
    monkeypatch.setenv("LOGNAME", "msys-name")
    monkeypatch.setenv("USER", "msys-name")

    assert windows_task_account() == "EXAMPLE\\cyril"

    monkeypatch.delenv("USERDOMAIN")

    assert windows_task_account() == "cyril"


def test_windows_headless_python_prefers_pythonw_sibling(tmp_path):
    python = tmp_path / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert windows_headless_python(python) == python

    pythonw = python.with_name("pythonw.exe")
    pythonw.write_text("", encoding="utf-8")

    assert windows_headless_python(python) == pythonw


def test_select_service_backend_rejects_unknown_platform(tmp_path, capsys):
    backend = select_service_backend(
        platform_name="sunos5",
        paths=make_paths(tmp_path),
        run_command=lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "1000",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert isinstance(backend, UnsupportedServiceBackend)
    assert backend.install() == 1
    assert "Windows per-user Scheduled Tasks" in capsys.readouterr().err


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
