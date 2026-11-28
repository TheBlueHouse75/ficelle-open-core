from __future__ import annotations

import os
import shlex
import subprocess
import sys

import pytest
from dataclasses import replace
from pathlib import Path

from ficelle.service import (
    ServicePaths,
    SystemdUserServiceBackend,
    UnsupportedServiceBackend,
    WindowsScheduledTaskBackend,
    persist_active_service_context,
    read_active_service_context,
    select_service_backend,
    windows_command_names_home,
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
        label="com.ficelle.router",
        plist=tmp_path / "Library" / "LaunchAgents" / "com.ficelle.router.plist",
        systemd_unit=tmp_path / ".config" / "systemd" / "user" / "com.ficelle.router.service",
        install_python=Path(sys.executable),
        active_home_pointer=tmp_path / ".config" / "ficelle" / "active-home"
        if persist_home
        else None,
    )


def test_active_service_context_roundtrip_persists_only_ficelle_home(tmp_path):
    ficelle_home = tmp_path / "custom-ficelle"
    runtime_dir = tmp_path / "custom-runtime"
    hermes_home = tmp_path / "custom-hermes"
    ficelle_home.mkdir()
    runtime_dir.mkdir()
    pointer = tmp_path / ".config" / "ficelle" / "active-home"

    assert persist_active_service_context(ficelle_home, pointer)
    assert read_active_service_context(pointer) == ficelle_home


def test_active_service_context_reads_plain_home_pointer(
    tmp_path,
):
    ficelle_home = tmp_path / "custom-ficelle"
    ficelle_home.mkdir()
    pointer = tmp_path / "active-home"
    pointer.write_text(str(ficelle_home), encoding="utf-8")

    assert read_active_service_context(pointer) == ficelle_home


def test_a_non_utf8_active_home_pointer_reads_as_unreadable(tmp_path):
    pointer = tmp_path / "active-home"
    pointer.write_bytes(b"\xff\xfe/not-utf8")

    assert read_active_service_context(pointer) is None


def test_select_service_backend_uses_launchagent_on_darwin(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    backend = select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "501",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.name == "launchagent"
    assert backend.install() == 0
    assert ["launchctl", "bootstrap", "gui/501", str(paths.plist)] in calls
    payload = backend.plist_payload()
    assert payload["WorkingDirectory"] == str(paths.ficelle_home)
    assert payload["ProgramArguments"] == [str(paths.install_python), "-P", "-m", "ficelle.router", "--serve"]
    assert payload["EnvironmentVariables"] == {
        "FICELLE_HOME": str(paths.ficelle_home), "PYTHONSAFEPATH": "1",
    }
    assert payload["StandardOutPath"] == str(paths.log_dir / "ficelle.log")
    assert "HERMES_HOME" not in payload["EnvironmentVariables"]


def test_select_service_backend_uses_systemd_user_on_linux(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    backend = select_service_backend(
        platform_name="linux",
        paths=paths,
        run_command=lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
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
    assert "ExecStart=" + sys.executable + " -P -m ficelle.router --serve" in unit
    assert f"WorkingDirectory={paths.ficelle_home}" in unit
    assert f'Environment="FICELLE_HOME={paths.ficelle_home}"' in unit
    assert 'Environment="PYTHONSAFEPATH=1"' in unit
    assert "HERMES_HOME" not in unit
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "--now", paths.systemd_unit.name] in calls


def test_launchagent_ignores_client_home_environment(tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    paths = make_paths(tmp_path, hermes_home=hermes_home)
    backend = select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        uid_provider=lambda: "501",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.plist_payload()["EnvironmentVariables"] == {
        "FICELLE_HOME": str(paths.ficelle_home),
        "PYTHONSAFEPATH": "1",
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

    def run(cmd, **kwargs):
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

    backend = select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=run,
        uid_provider=lambda: "501",
        wait_for_ready=lambda: ready,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )
    backend.pid_alive = lambda _pid: False  # the scripted pid is not a real process
    return backend


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


def test_launchagent_probes_the_canonical_configured_port(tmp_path):
    paths = make_paths(tmp_path)
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


def test_systemd_ignores_client_home_environment(tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    paths = make_paths(tmp_path, hermes_home=hermes_home)
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    unit = backend.unit_payload()

    assert f'Environment="FICELLE_HOME={paths.ficelle_home}"' in unit
    assert "HERMES_HOME" not in unit


def test_systemd_persists_the_context_of_a_service_it_leaves_running(tmp_path, capsys):
    """systemd escaped the 14/08/2026 kill — it never stopped the unit on a readiness
    timeout — but it also never described the service it left running. `ficelle health`, the
    follow-up the other two backends now recommend, reads that context."""
    paths = make_paths(tmp_path, persist_home=True)
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: False,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.install() == 1
    assert "left running" in capsys.readouterr().err
    assert read_active_service_context(paths.active_home_pointer) == paths.ficelle_home


def test_systemd_quotes_complete_environment_assignments(tmp_path):
    ficelle_home = tmp_path / 'Ficelle "home" \\ credentials'
    runtime_dir = tmp_path / 'Runtime "dir" \\ state'
    hermes_home = tmp_path / 'Hermes "home" \\ config'
    paths = ServicePaths(
        ficelle_home=ficelle_home,
        label="com.ficelle.router",
        plist=tmp_path / "router.plist",
        systemd_unit=tmp_path / "router.service",
        install_python=Path(sys.executable),
    )
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(
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
    assert "FICELLE_RUNTIME_DIR" not in unit
    assert "HERMES_HOME" not in unit


def test_service_keeps_all_artifacts_under_ficelle_home(tmp_path):
    external_runtime = tmp_path / ".host" / "ficelle"
    external_runtime.mkdir(parents=True)
    state_path = external_runtime / "state.json"
    state_path.write_text('{"external": true}\n', encoding="utf-8")
    paths = make_paths(tmp_path, persist_home=True)
    backend = select_service_backend(
        platform_name="darwin",
        paths=paths,
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(
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
        "FICELLE_HOME": str(paths.ficelle_home), "PYTHONSAFEPATH": "1",
    }
    assert backend.install() == 0
    assert paths.ficelle_home.is_dir()
    assert (paths.ficelle_home / "logs").is_dir()
    assert not (external_runtime / "logs").exists()
    assert state_path.read_text(encoding="utf-8") == '{"external": true}\n'
    assert read_active_service_context(paths.active_home_pointer) == paths.ficelle_home


def test_service_paths_ignore_optional_client_context(monkeypatch, tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    paths = ServicePaths(
        ficelle_home=tmp_path / ".ficelle",
        label="com.ficelle.router",
        plist=tmp_path / "router.plist",
        systemd_unit=tmp_path / "router.service",
        install_python=Path(sys.executable),
    )

    assert not hasattr(paths, "hermes_home")


def test_systemd_user_restart_reuses_existing_unit(tmp_path):
    calls = []
    paths = make_paths(tmp_path, persist_home=True)
    paths.systemd_unit.parent.mkdir(parents=True)
    paths.systemd_unit.write_text("existing", encoding="utf-8")
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.restart() == 0
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "stop", paths.systemd_unit.name],
        ["systemctl", "--user", "restart", paths.systemd_unit.name],
    ]
    assert f'Environment="FICELLE_HOME={paths.ficelle_home}"' in paths.systemd_unit.read_text(
        encoding="utf-8"
    )
    assert read_active_service_context(paths.active_home_pointer) == paths.ficelle_home


@pytest.mark.parametrize("safe_path", [False, True])
def test_systemd_user_restart_reuses_the_interpreter_recorded_in_the_unit(tmp_path, capsys, safe_path):
    installed_python = tmp_path / "venv-A" / "bin" / "python"
    installed_python.parent.mkdir(parents=True)
    installed_python.write_text("", encoding="utf-8")
    current_python = tmp_path / "venv-B" / "bin" / "python"
    current_python.parent.mkdir(parents=True)
    current_python.write_text("", encoding="utf-8")
    paths = make_paths(tmp_path)

    installed = SystemdUserServiceBackend(
        paths=replace(paths, install_python=installed_python),
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )
    assert installed.install() == 0
    if not safe_path:
        unit = paths.systemd_unit.read_text(encoding="utf-8")
        paths.systemd_unit.write_text(unit.replace(" -P -m ", " -m "), encoding="utf-8")

    restarted = SystemdUserServiceBackend(
        paths=replace(paths, install_python=current_python),
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )
    assert restarted.restart() == 0

    unit = paths.systemd_unit.read_text(encoding="utf-8")
    assert f"ExecStart={installed_python} -P -m ficelle.router --serve" in unit
    assert "reusing the interpreter" in capsys.readouterr().out


def test_systemd_user_restart_repoints_when_recorded_interpreter_is_unusable(tmp_path):
    broken_python = tmp_path / "venv-A" / "bin" / "python"
    current_python = tmp_path / "venv-B" / "bin" / "python"
    broken_python.parent.mkdir(parents=True)
    broken_python.write_text("", encoding="utf-8")
    current_python.parent.mkdir(parents=True)
    current_python.write_text("", encoding="utf-8")
    paths = make_paths(tmp_path)
    paths.systemd_unit.parent.mkdir(parents=True)
    paths.systemd_unit.write_text(
        f"ExecStart={broken_python} -m ficelle.router --serve\n", encoding="utf-8"
    )

    def run_command(command, **kwargs):
        if command[:3] == [str(broken_python), "-P", "-c"]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="broken")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    restarted = SystemdUserServiceBackend(
        paths=replace(paths, install_python=current_python),
        run_command=run_command,
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )
    assert restarted.restart() == 0
    assert f"ExecStart={current_python} -P -m ficelle.router --serve" in paths.systemd_unit.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("existing_unit, start_verb", [(True, "restart"), (False, "enable")])
def test_systemd_user_stops_the_unit_before_sweeping_orphans(tmp_path, existing_unit, start_verb):
    # The unit's own router matches the stale-server signature: sweeping before `systemctl`
    # stops it would SIGKILL it after 2 s and cut the router's 20 s shutdown drain short.
    events: list = []
    paths = make_paths(tmp_path)
    if existing_unit:
        paths.systemd_unit.parent.mkdir(parents=True)
        paths.systemd_unit.write_text("unit", encoding="utf-8")
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: events.append(tuple(cmd)) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: events.append("sweep") or [],
        report_stale_servers=lambda _pids: None,
    )

    assert backend.restart() == 0

    stop = ("systemctl", "--user", "stop", paths.systemd_unit.name)
    start = next(event for event in events if event[2:3] == (start_verb,))
    assert events.index(stop) < events.index("sweep") < events.index(start)


def test_systemd_unit_quotes_interpreter_paths_systemd_would_split_or_expand(tmp_path):
    # `$` stays single: systemd substitutes variables in the arguments, never in the executable.
    spaced = tmp_path / "my 100% $venv" / "bin" / "python"
    spaced.parent.mkdir(parents=True)
    spaced.write_text("", encoding="utf-8")
    paths = replace(make_paths(tmp_path), install_python=spaced)
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    backend.write_unit()

    unit = paths.systemd_unit.read_text(encoding="utf-8")
    assert f'ExecStart="{tmp_path}/my 100%% $venv/bin/python" -P -m ficelle.router --serve' in unit
    assert backend.recorded_interpreter() == spaced


def test_systemd_user_uninstall_removes_unit_and_reloads(tmp_path):
    calls = []
    paths = make_paths(tmp_path)
    paths.systemd_unit.parent.mkdir(parents=True)
    paths.systemd_unit.write_text("unit", encoding="utf-8")
    backend = SystemdUserServiceBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
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

    def run(cmd, **kwargs):
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
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
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
    assert "-P -m ficelle.windows_entry" in payload
    assert f"FICELLE_HOME={paths.ficelle_home}" in payload
    assert "PYTHONSAFEPATH=" not in payload
    assert f"<WorkingDirectory>{paths.ficelle_home}</WorkingDirectory>" in payload
    assert "<LogonTrigger>" in payload
    assert "<RunLevel>LeastPrivilege</RunLevel>" in payload
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in payload
    assert "FICELLE_RUNTIME_DIR=" not in payload
    assert "HERMES_HOME=" not in payload


def test_scheduled_task_passes_only_ficelle_context_as_assignment(tmp_path):
    hermes_home = tmp_path / "custom-hermes"
    runtime_dir = tmp_path / ".hermes" / "ficelle"
    runtime_dir.mkdir(parents=True)
    paths = make_paths(tmp_path, runtime_dir=runtime_dir, hermes_home=hermes_home)
    backend = _windows_backend(paths)

    assert backend.entry_arguments() == [
        "-P",
        "-m",
        "ficelle.windows_entry",
        f"FICELLE_HOME={paths.ficelle_home}",
    ]


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_generated_service_command_reaches_router_serve(tmp_path, monkeypatch, platform):
    from ficelle import router, windows_entry

    paths = make_paths(tmp_path)
    backend = select_service_backend(
        platform_name=platform,
        paths=paths,
        run_command=lambda *_args, **_kwargs: pytest.fail("command generation must not start a service"),
        uid_provider=lambda: "1000",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )
    monkeypatch.setenv("FICELLE_HOME", str(tmp_path / "other-home"))
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    # The router module may already be imported by another test.
    monkeypatch.setattr(router, "CONFIG_PATH", paths.ficelle_home / "config.json")
    served = []
    monkeypatch.setattr(router, "serve", lambda _config: served.append(os.environ["FICELLE_HOME"]))

    if platform == "win32":
        safe_path, module_flag, module, *argv = backend.entry_arguments()
        assert (safe_path, module_flag, module) == ("-P", "-m", "ficelle.windows_entry")
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            assert windows_entry.main(argv) == 0
        finally:
            if sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not original_stderr:
                sys.stderr.close()
            sys.stdout, sys.stderr = original_stdout, original_stderr
    else:
        if platform == "darwin":
            payload = backend.plist_payload()
            command = payload["ProgramArguments"]
            environment = payload["EnvironmentVariables"].items()
        else:
            entries = backend.unit_payload().splitlines()
            command = shlex.split(next(
                line.removeprefix("ExecStart=") for line in entries if line.startswith("ExecStart=")
            ))
            environment = (
                shlex.split(line.removeprefix("Environment="))[0].split("=", 1)
                for line in entries if line.startswith("Environment=")
            )
        for name, value in environment:
            monkeypatch.setenv(name, value)
        assert os.environ["PYTHONSAFEPATH"] == "1"
        _, safe_path, module_flag, module, *argv = command
        assert (safe_path, module_flag, module) == ("-P", "-m", "ficelle.router")
        assert router.main_args(argv) == 0

    assert served == [str(paths.ficelle_home)]


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


@pytest.mark.parametrize("safe_path", [False, True])
def test_scheduled_task_restart_reuses_the_interpreter_recorded_in_xml(tmp_path, capsys, safe_path):
    installed_python = tmp_path / "venv-A" / "python.exe"
    installed_python.parent.mkdir(parents=True)
    installed_python.write_text("", encoding="utf-8")
    current_python = tmp_path / "venv-B" / "python.exe"
    current_python.parent.mkdir(parents=True)
    current_python.write_text("", encoding="utf-8")
    paths = make_paths(tmp_path)

    installed = _windows_backend(replace(paths, install_python=installed_python))
    assert installed.install() == 0
    if not safe_path:
        payload = installed.task_xml.read_text(encoding="utf-16")
        installed.task_xml.write_text(payload.replace("-P -m ", "-m "), encoding="utf-16")

    restarted = _windows_backend(replace(paths, install_python=current_python))
    assert restarted.restart() == 0

    payload = paths.ficelle_home.joinpath(f"{paths.label}.task.xml").read_text(encoding="utf-16")
    assert f"<Command>{installed_python}</Command>" in payload
    assert "reusing the interpreter" in capsys.readouterr().out


def test_scheduled_task_restart_repoints_when_recorded_interpreter_is_unusable(tmp_path):
    broken_python = tmp_path / "venv-A" / "python.exe"
    current_python = tmp_path / "venv-B" / "python.exe"
    broken_python.parent.mkdir(parents=True)
    broken_python.write_text("", encoding="utf-8")
    current_python.parent.mkdir(parents=True)
    current_python.write_text("", encoding="utf-8")
    paths = make_paths(tmp_path)
    paths.ficelle_home.mkdir(parents=True)
    paths.ficelle_home.joinpath(f"{paths.label}.task.xml").write_text(
        WindowsScheduledTaskBackend(
            paths=replace(paths, install_python=broken_python),
            run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
            wait_for_ready=lambda: True,
            terminate_stale_servers=lambda: [],
            report_stale_servers=lambda _pids: None,
            account_provider=lambda: "EXAMPLE\\cyril",
        ).task_payload(),
        encoding="utf-16",
    )

    def run_command(command, **kwargs):
        if command[:3] == [str(broken_python), "-P", "-c"]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="broken")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    restarted = WindowsScheduledTaskBackend(
        paths=replace(paths, install_python=current_python),
        run_command=run_command,
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
        account_provider=lambda: "EXAMPLE\\cyril",
    )
    assert restarted.restart() == 0
    payload = paths.ficelle_home.joinpath(f"{paths.label}.task.xml").read_text(encoding="utf-16")
    assert f"<Command>{current_python}</Command>" in payload


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
    assert read_active_service_context(paths.active_home_pointer) == paths.ficelle_home


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="requires POSIX file modes")
@pytest.mark.parametrize(("platform", "writer"), [
    ("darwin", "install"),
    ("linux", "write_unit"),
    ("win32", "write_task_definition"),
])
def test_service_writers_refuse_a_symlink_and_create_a_private_home(tmp_path, platform, writer):
    paths = make_paths(tmp_path)
    target = tmp_path / "real-home"
    target.mkdir()
    paths.ficelle_home.symlink_to(target, target_is_directory=True)
    calls = []
    backend = select_service_backend(
        platform_name=platform,
        paths=paths,
        run_command=lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""),
        uid_provider=lambda: "501",
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
    )

    with pytest.raises(OSError, match="symbolic link"):
        getattr(backend, writer)()
    assert list(target.iterdir()) == []
    assert not paths.plist.exists()
    assert not paths.systemd_unit.exists()
    assert calls == []

    paths.ficelle_home.unlink()
    getattr(backend, writer)()
    assert paths.ficelle_home.stat().st_mode & 0o777 == 0o700


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
        run_command=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
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


@pytest.mark.parametrize("flags", [[], ["-P"]])
def test_restart_reuses_the_interpreter_the_plist_already_records(tmp_path, capsys, flags):
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
        plistlib.dump({"ProgramArguments": [str(installed_python), *flags, "-m", "ficelle.router", "--serve"]}, handle)

    backend = _launchagent(paths, lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
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

    def run_command(cmd, **kwargs):
        # The import probe fails for the recorded interpreter; launchctl calls succeed.
        if cmd[:3] == [str(broken_python), "-P", "-c"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ModuleNotFoundError")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    backend = _launchagent(paths, run_command)

    assert backend.restart() == 0

    with paths.plist.open("rb") as handle:
        rewritten = plistlib.load(handle)
    assert rewritten["ProgramArguments"][0] == str(paths.install_python), "a dead interpreter must be replaced"
    assert "unusable" in capsys.readouterr().out


@pytest.mark.parametrize("shadow", ["ficelle.py", "ficelle/__init__.py"])
def test_recorded_interpreter_probe_cannot_import_caller_package(tmp_path, monkeypatch, shadow):
    from ficelle.service import _usable_recorded_interpreter

    caller = tmp_path / "caller"
    planted = caller / shadow
    planted.parent.mkdir(parents=True)
    planted.write_text("raise RuntimeError('caller shadow')\n")
    interpreter = tmp_path / "venv" / "python"
    interpreter.parent.mkdir()
    interpreter.touch()
    monkeypatch.chdir(caller)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    calls = []
    def run(command, **kwargs):
        assert Path(kwargs["cwd"]).stat().st_mode & 0o777 == 0o700
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    assert _usable_recorded_interpreter(interpreter, run) == interpreter
    (command, kwargs), = calls
    assert command == [str(interpreter), "-P", "-c", "import ficelle"]
    assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
    assert Path(kwargs["cwd"]).is_absolute() and kwargs["cwd"] != str(caller)


def _registered_task(home: Path) -> str:
    """The XML `schtasks /Query /XML` prints for the task the backend of `home` registers."""
    return _windows_backend(replace(make_paths(home.parent), ficelle_home=home)).task_payload()


@pytest.mark.parametrize(
    ("home", "registered", "expected"),
    [
        (".ficelle", ".ficelle", True),
        # list2cmdline quotes an argument holding a space.
        ("My Home/.ficelle", "My Home/.ficelle", True),
        # A test router registered under the same per-user task name with another home: an
        # update must neither stop it nor re-register the task as this home's service.
        (".ficelle", "canary/.ficelle", False),
        # `.ficelle-canary` starts with `.ficelle`; only the whole argument names the home.
        (".ficelle", ".ficelle-canary", False),
        (".ficelle", "not-xml", False),
        (".ficelle", "not-registered", False),
        (".ficelle", "no-schtasks", False),
    ],
)
def test_scheduled_task_installed_requires_the_task_of_this_home(tmp_path, home, registered, expected):
    # Updates require an installed service. On Windows neither the plist nor the systemd unit
    # ever exists, and the task XML is written before registration, so only Task Scheduler can
    # tell, and its task name is per user, not per FICELLE_HOME.
    paths = replace(make_paths(tmp_path), ficelle_home=tmp_path / home)

    def query(cmd):
        if registered == "no-schtasks":
            raise FileNotFoundError("schtasks")
        if registered == "not-registered":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: not found")
        if registered == "not-xml":
            return subprocess.CompletedProcess(cmd, 0, stdout="TaskName: com.ficelle.router", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_registered_task(tmp_path / registered), stderr="")

    calls = []
    backend = WindowsScheduledTaskBackend(
        paths=paths,
        run_command=lambda cmd, **kwargs: calls.append(cmd) or query(cmd),
        wait_for_ready=lambda: True,
        terminate_stale_servers=lambda: [],
        report_stale_servers=lambda _pids: None,
        account_provider=lambda: "EXAMPLE\\cyril",
    )

    assert backend.installed() is expected
    assert calls == [["schtasks", "/Query", "/TN", paths.label, "/XML"]]


@pytest.mark.parametrize(
    ("command", "home", "expected"),
    [
        (r"-m ficelle.windows_entry FICELLE_HOME=C:\Users\me\.ficelle", r"C:\Users\me\.ficelle", True),
        # Another instance whose home only starts with this one's, then a space inside the quotes.
        (r'-m ficelle.windows_entry "FICELLE_HOME=C:\Users\me\.ficelle canary"', r"C:\Users\me\.ficelle", False),
        (
            r'"C:\Program Files\Python\pythonw.exe" -m ficelle.windows_entry "FICELLE_HOME=C:\Users\My Home\.ficelle"',
            r"C:\Users\My Home\.ficelle",
            True,
        ),
        # Windows paths are case-insensitive: the same home spelled differently is still this one.
        (r"-m ficelle.windows_entry FICELLE_HOME=c:\users\ME\.Ficelle", r"C:\Users\me\.ficelle", True),
        ("-m ficelle.windows_entry FICELLE_HOME=C:\\Users\\me\\.ficelle\\", r"C:\Users\me\.ficelle", True),
        # list2cmdline doubles a trailing backslash before the closing quote.
        (r'-m ficelle.windows_entry "FICELLE_HOME=C:\Users\My Home\.ficelle\\"', r"C:\Users\My Home\.ficelle", True),
        (r"-m ficelle.windows_entry FICELLE_HOME=C:\Users\me\.ficelle-canary", r"C:\Users\me\.ficelle", False),
        (r"-m ficelle.windows_entry", r"C:\Users\me\.ficelle", False),
    ],
)
def test_windows_command_names_home_compares_the_parsed_home_argument(command, home, expected):
    assert windows_command_names_home(command, Path(home)) is expected


def test_launchagent_waits_for_its_draining_router_before_the_stale_sweep(tmp_path, monkeypatch):
    # `launchctl bootout` returns once SIGTERM is sent. Sweeping right after matched our own
    # draining router and SIGKILLed it 2 s later, so its 15 s drain never ended requests cleanly;
    # launchd's unset ExitTimeOut (about 5 s) cut it short too.
    from ficelle import router, service

    events: list[str] = []
    polls = iter([True, True, False])
    monkeypatch.setattr(service.time, "sleep", lambda _s: None)
    calls: list[list[str]] = []
    backend = _launchagent_backend(make_paths(tmp_path), calls, ready=True, launchd_pid=4242)
    backend.pid_alive = lambda pid: events.append(f"poll {pid}") or next(polls)
    backend.terminate_stale_servers = lambda: events.append("sweep") or []

    assert backend.stop() == 0

    assert events == ["poll 4242", "poll 4242", "poll 4242", "sweep"]
    assert backend.plist_payload()["ExitTimeOut"] == service.LAUNCHD_EXIT_TIMEOUT_SECONDS
    assert service.LAUNCHD_EXIT_TIMEOUT_SECONDS > router.SHUTDOWN_DRAIN_SECONDS
