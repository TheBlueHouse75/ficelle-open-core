"""Local service backends for Ficelle."""
from __future__ import annotations

import getpass
import json
import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Protocol
from xml.sax.saxutils import escape as xml_escape


LOG_FILE_NAME = "ficelle.log"
ERROR_LOG_FILE_NAME = "ficelle.error.log"

# Substrings that identify a Ficelle server process command line. Defined here, next to
# the backends that construct those command lines, and consumed by the CLI's
# stale-server cleanup so the two halves of the contract cannot drift apart.
SERVER_COMMAND_SIGNATURES = ("-m ficelle.router --serve", "-m ficelle.windows_entry")


def service_log_dir(ficelle_home: Path) -> Path:
    """The service log directory for a Ficelle home, shared with ficelle.windows_entry."""
    return ficelle_home / "logs"


RunCommand = Callable[[list[str]], subprocess.CompletedProcess[str]]
ReadyCheck = Callable[[], bool]
StaleServerTerminator = Callable[[], list[int]]
StaleServerReporter = Callable[[list[int]], None]
UidProvider = Callable[[], str]


def active_home_pointer_path() -> Path:
    """Return the user-scoped metadata file for the active service context."""
    return Path.home() / ".config" / "ficelle" / "active-home"


def read_active_service_context(
    pointer: Path | None = None,
) -> tuple[Path, Path, Path | None] | None:
    """Read persisted credential, runtime, and optional Hermes roots."""
    path = pointer or active_home_pointer_path()
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"ficelle_home": raw}
    if not isinstance(payload, dict):
        return None
    ficelle_home = Path(str(payload.get("ficelle_home") or "")).expanduser()
    if not ficelle_home.is_absolute() or not ficelle_home.is_dir():
        return None
    raw_runtime_dir = payload.get("runtime_dir")
    runtime_dir = (
        Path(str(raw_runtime_dir)).expanduser()
        if isinstance(raw_runtime_dir, str) and raw_runtime_dir
        else ficelle_home
    )
    if not runtime_dir.is_absolute() or not runtime_dir.is_dir():
        return None
    raw_hermes_home = payload.get("hermes_home")
    hermes_home = (
        Path(str(raw_hermes_home)).expanduser()
        if isinstance(raw_hermes_home, str) and raw_hermes_home
        else None
    )
    if hermes_home is not None and not hermes_home.is_absolute():
        return None
    return ficelle_home, runtime_dir, hermes_home


def persist_active_service_context(
    ficelle_home: Path,
    pointer: Path | None = None,
    *,
    runtime_dir: Path | None = None,
    hermes_home: Path | None = None,
) -> bool:
    """Atomically persist the active service roots."""
    path = pointer or active_home_pointer_path()
    home = ficelle_home.expanduser()
    if not home.is_absolute() or not home.is_dir():
        return False
    active_runtime_dir = runtime_dir.expanduser() if runtime_dir is not None else home
    if not active_runtime_dir.is_absolute() or not active_runtime_dir.is_dir():
        return False
    expanded_hermes_home = hermes_home.expanduser() if hermes_home is not None else None
    if expanded_hermes_home is not None and not expanded_hermes_home.is_absolute():
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(
                {
                    "ficelle_home": str(home),
                    "runtime_dir": str(active_runtime_dir),
                    "hermes_home": str(expanded_hermes_home)
                    if expanded_hermes_home is not None
                    else None,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False
    return True


def hermes_home_from_environment() -> Path | None:
    """Return the optional Hermes integration root passed by target-aware setup."""
    value = os.getenv("HERMES_HOME")
    return Path(value).expanduser() if value else None


class ServiceBackend(Protocol):
    """Platform service lifecycle backend."""

    name: str

    def install(self) -> int:
        """Install and start the managed service."""
        ...

    def restart(self) -> int:
        """Restart the managed service."""
        ...

    def stop(self) -> int:
        """Stop the managed service."""
        ...

    def uninstall(self) -> int:
        """Uninstall the managed service."""
        ...

    def status(self) -> int:
        """Print managed service status."""
        ...


@dataclass(frozen=True)
class ServicePaths:
    ficelle_home: Path
    runtime_dir: Path
    label: str
    plist: Path
    systemd_unit: Path
    install_python: Path
    hermes_home: Path | None = field(default_factory=hermes_home_from_environment)
    active_home_pointer: Path | None = None

    @property
    def log_dir(self) -> Path:
        return service_log_dir(self.ficelle_home)


def service_environment(paths: ServicePaths) -> dict[str, str]:
    environment = {"FICELLE_HOME": str(paths.ficelle_home)}
    if paths.runtime_dir != paths.ficelle_home:
        environment["FICELLE_RUNTIME_DIR"] = str(paths.runtime_dir)
    if paths.hermes_home is not None:
        environment["HERMES_HOME"] = str(paths.hermes_home)
    return environment


def _systemd_environment_line(name: str, value: str) -> str:
    """Quote a complete systemd environment assignment."""
    assignment = json.dumps(f"{name}={value}", ensure_ascii=False)
    return f"Environment={assignment}"


def write_text_file(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def persist_service_context(paths: ServicePaths) -> bool:
    if paths.active_home_pointer is None:
        return True
    if persist_active_service_context(
        paths.ficelle_home,
        paths.active_home_pointer,
        runtime_dir=paths.runtime_dir,
        hermes_home=paths.hermes_home,
    ):
        return True
    sys.stderr.write(
        f"Ficelle service started, but its active context could not be persisted at "
        f"{paths.active_home_pointer}.\n"
    )
    return False


class LaunchAgentServiceBackend:
    """macOS LaunchAgent backend."""

    name = "launchagent"

    def __init__(
        self,
        *,
        paths: ServicePaths,
        run_command: RunCommand,
        uid_provider: UidProvider,
        wait_for_ready: ReadyCheck,
        terminate_stale_servers: StaleServerTerminator,
        report_stale_servers: StaleServerReporter,
    ) -> None:
        self.paths = paths
        self.run_command = run_command
        self.uid_provider = uid_provider
        self.wait_for_ready = wait_for_ready
        self.terminate_stale_servers = terminate_stale_servers
        self.report_stale_servers = report_stale_servers

    def plist_payload(self) -> dict[str, object]:
        return {
            "Label": self.paths.label,
            "ProgramArguments": [str(self.paths.install_python), "-m", "ficelle.router", "--serve"],
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(self.paths.log_dir / LOG_FILE_NAME),
            "StandardErrorPath": str(self.paths.log_dir / ERROR_LOG_FILE_NAME),
            "WorkingDirectory": str(self.paths.ficelle_home),
            "EnvironmentVariables": service_environment(self.paths),
        }

    def bootout(self) -> subprocess.CompletedProcess[str]:
        return self.run_command(["launchctl", "bootout", f"gui/{self.uid_provider()}/{self.paths.label}"])

    def install(self) -> int:
        self.paths.ficelle_home.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.plist.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.plist.open("wb") as fh:
            plistlib.dump(self.plist_payload(), fh, sort_keys=False)
        self.bootout()
        self.report_stale_servers(self.terminate_stale_servers())
        load = self.run_command(["launchctl", "bootstrap", f"gui/{self.uid_provider()}", str(self.paths.plist)])
        if load.returncode != 0:
            sys.stderr.write(load.stderr or load.stdout)
            return load.returncode
        self.run_command(["launchctl", "enable", f"gui/{self.uid_provider()}/{self.paths.label}"])
        self.run_command(["launchctl", "kickstart", "-k", f"gui/{self.uid_provider()}/{self.paths.label}"])
        if not self.wait_for_ready():
            # Left as-is, a port already held by something else looks like a mystery: launchd
            # relaunches forever (KeepAlive on non-zero exit), appending a traceback each time,
            # while the CLI says only "did not become ready". Name the holder and stop the loop.
            sys.stderr.write("Ficelle LaunchAgent started but /admin/status.json did not become ready.\n")
            holder = self.port_holder_description()
            if holder:
                sys.stderr.write(f"The configured port looks busy: {holder}\n")
            self.bootout()
            sys.stderr.write("Booted the agent out so it does not relaunch in a loop.\n")
            return 1
        if not persist_service_context(self.paths):
            return 1
        print(f"installed {self.paths.plist}")
        return 0

    def port_holder_description(self) -> str | None:
        """Who is listening on the configured port, when anyone is."""
        try:
            config = json.loads((self.paths.runtime_dir / "config.json").read_text(encoding="utf-8"))
            port = int(config.get("port") or 8646)
        except (OSError, ValueError, TypeError):
            port = 8646
        probe = self.run_command(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"])
        if probe.returncode != 0 or not (probe.stdout or "").strip():
            return None
        lines = [line for line in probe.stdout.splitlines() if line.strip()]
        detail = lines[1] if len(lines) > 1 else lines[0]
        return f"port {port} is held by: {detail.strip()}"

    def recorded_interpreter(self) -> Path | None:
        """The interpreter the installed plist already points at, when it still works.

        AGENTS.md warns that the LaunchAgent must run under the Python that setup used. The CLI
        honoured that at setup and then re-decided it on every restart, because `restart()`
        rewrites the plist from `sys.executable`: running `ficelle restart` from a second
        install, a `uvx` environment, or system Python silently repointed the service, and an
        interpreter that cannot import `ficelle` leaves launchd in a crash loop.
        """
        try:
            with self.paths.plist.open("rb") as handle:
                payload = plistlib.load(handle)
        except (OSError, ValueError):
            return None
        arguments = payload.get("ProgramArguments")
        if not isinstance(arguments, list) or not arguments:
            return None
        candidate = Path(str(arguments[0]))
        if not candidate.exists():
            return None
        probe = self.run_command([str(candidate), "-c", "import ficelle"])
        if probe.returncode != 0:
            return None
        return candidate

    def restart(self) -> int:
        recorded = self.recorded_interpreter()
        if recorded is not None and recorded != self.paths.install_python:
            print(f"reusing the interpreter the service was installed with: {recorded}")
            self.paths = replace(self.paths, install_python=recorded)
        elif recorded is None and self.paths.plist.exists():
            print(f"installed interpreter unusable; repointing the service at {self.paths.install_python}")
        return self.install()

    def stop(self) -> int:
        result = self.bootout()
        self.report_stale_servers(self.terminate_stale_servers())
        return result.returncode

    def uninstall(self) -> int:
        self.stop()
        if self.paths.plist.exists():
            self.paths.plist.unlink()
        print(f"removed {self.paths.plist}")
        return 0

    def status(self) -> int:
        result = self.run_command(["launchctl", "print", f"gui/{self.uid_provider()}/{self.paths.label}"])
        if result.returncode != 0:
            print(result.stderr.strip() or result.stdout.strip() or "not loaded")
            return result.returncode
        lines = [line for line in result.stdout.splitlines() if "state =" in line or "pid =" in line or "runs =" in line]
        print("\n".join(lines) if lines else result.stdout[:1200])
        return 0


class SystemdUserServiceBackend:
    """Linux systemd --user backend."""

    name = "systemd-user"

    def __init__(
        self,
        *,
        paths: ServicePaths,
        run_command: RunCommand,
        wait_for_ready: ReadyCheck,
        terminate_stale_servers: StaleServerTerminator,
        report_stale_servers: StaleServerReporter,
    ) -> None:
        self.paths = paths
        self.run_command = run_command
        self.wait_for_ready = wait_for_ready
        self.terminate_stale_servers = terminate_stale_servers
        self.report_stale_servers = report_stale_servers

    @property
    def unit_name(self) -> str:
        return self.paths.systemd_unit.name

    def unit_payload(self) -> str:
        environment = [
            _systemd_environment_line(name, value)
            for name, value in service_environment(self.paths).items()
        ]
        return "\n".join(
            [
                "[Unit]",
                "Description=Ficelle local OpenAI-compatible model router",
                "After=network-online.target",
                "Wants=network-online.target",
                "",
                "[Service]",
                "Type=simple",
                f"WorkingDirectory={self.paths.ficelle_home}",
                *environment,
                f"ExecStart={self.paths.install_python} -m ficelle.router --serve",
                "Restart=on-failure",
                "RestartSec=2",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )

    def daemon_reload(self) -> subprocess.CompletedProcess[str]:
        return self.run_command(["systemctl", "--user", "daemon-reload"])

    def write_unit(self) -> subprocess.CompletedProcess[str]:
        self.paths.ficelle_home.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        write_text_file(self.paths.systemd_unit, self.unit_payload())
        return self.daemon_reload()

    def install(self) -> int:
        reload_result = self.write_unit()
        if reload_result.returncode != 0:
            sys.stderr.write(reload_result.stderr or reload_result.stdout)
            return reload_result.returncode
        self.report_stale_servers(self.terminate_stale_servers())
        start = self.run_command(["systemctl", "--user", "enable", "--now", self.unit_name])
        if start.returncode != 0:
            sys.stderr.write(start.stderr or start.stdout)
            return start.returncode
        if not self.wait_for_ready():
            sys.stderr.write("Ficelle systemd user service started but /admin/status.json did not become ready.\n")
            return 1
        if not persist_service_context(self.paths):
            return 1
        print(f"installed {self.paths.systemd_unit}")
        return 0

    def restart(self) -> int:
        if not self.paths.systemd_unit.exists():
            return self.install()
        reload_result = self.write_unit()
        if reload_result.returncode != 0:
            sys.stderr.write(reload_result.stderr or reload_result.stdout)
            return reload_result.returncode
        self.report_stale_servers(self.terminate_stale_servers())
        result = self.run_command(["systemctl", "--user", "restart", self.unit_name])
        if result.returncode != 0:
            sys.stderr.write(result.stderr or result.stdout)
            return result.returncode
        if not self.wait_for_ready():
            sys.stderr.write("Ficelle systemd user service restarted but /admin/status.json did not become ready.\n")
            return 1
        if not persist_service_context(self.paths):
            return 1
        return 0

    def stop(self) -> int:
        result = self.run_command(["systemctl", "--user", "stop", self.unit_name])
        self.report_stale_servers(self.terminate_stale_servers())
        return result.returncode

    def uninstall(self) -> int:
        self.run_command(["systemctl", "--user", "disable", "--now", self.unit_name])
        self.report_stale_servers(self.terminate_stale_servers())
        if self.paths.systemd_unit.exists():
            self.paths.systemd_unit.unlink()
        reload_result = self.daemon_reload()
        if reload_result.returncode != 0:
            sys.stderr.write(reload_result.stderr or reload_result.stdout)
            return reload_result.returncode
        print(f"removed {self.paths.systemd_unit}")
        return 0

    def status(self) -> int:
        result = self.run_command(["systemctl", "--user", "status", "--no-pager", self.unit_name])
        print(result.stdout.strip() or result.stderr.strip() or "not loaded")
        return result.returncode


def windows_headless_python(install_python: Path) -> Path:
    """Prefer the console-less ``pythonw.exe`` beside the install interpreter.

    A logon-triggered task with an InteractiveToken principal runs in the user's desktop
    session, so ``python.exe`` would keep a console window open for the lifetime of the
    service. ``pythonw.exe`` ships beside ``python.exe`` in CPython installs and
    virtualenvs; fall back to the recorded interpreter when it is absent.
    """
    headless = install_python.with_name("pythonw.exe")
    return headless if headless.exists() else install_python


def windows_task_account() -> str:
    """The ``DOMAIN\\user`` principal for the per-user task.

    ``USERNAME`` must win: ``getpass.getuser()`` prefers ``LOGNAME``/``USER``, which
    MSYS2/Git Bash shells set to names ``schtasks`` cannot map to a Windows account.
    """
    domain = os.environ.get("USERDOMAIN")
    user = os.environ.get("USERNAME") or getpass.getuser()
    return f"{domain}\\{user}" if domain else user


class WindowsScheduledTaskBackend:
    """Windows per-user Scheduled Task backend (logon trigger, no elevation)."""

    name = "scheduled-task"

    def __init__(
        self,
        *,
        paths: ServicePaths,
        run_command: RunCommand,
        wait_for_ready: ReadyCheck,
        terminate_stale_servers: StaleServerTerminator,
        report_stale_servers: StaleServerReporter,
        account_provider: UidProvider = windows_task_account,
    ) -> None:
        self.paths = paths
        self.run_command = run_command
        self.wait_for_ready = wait_for_ready
        self.terminate_stale_servers = terminate_stale_servers
        self.report_stale_servers = report_stale_servers
        self.account_provider = account_provider

    @property
    def task_name(self) -> str:
        return self.paths.label

    @property
    def task_xml(self) -> Path:
        return self.paths.ficelle_home / f"{self.paths.label}.task.xml"

    def entry_arguments(self) -> list[str]:
        # Task Scheduler cannot set per-task environment variables, so the service context
        # travels as NAME=VALUE assignments (systemd's Environment= shape) that
        # ficelle.windows_entry republishes; service_environment() is the one definition.
        return ["-m", "ficelle.windows_entry"] + [
            f"{name}={value}" for name, value in service_environment(self.paths).items()
        ]

    def task_payload(self) -> str:
        account = xml_escape(self.account_provider())
        command = xml_escape(str(windows_headless_python(self.paths.install_python)))
        arguments = xml_escape(subprocess.list2cmdline(self.entry_arguments()))
        working_directory = xml_escape(str(self.paths.ficelle_home))
        # ExecutionTimeLimit PT0S disables the schema default, which kills the process
        # after 72 hours.
        return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Ficelle local OpenAI-compatible model router</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{account}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{account}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>10</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{working_directory}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""

    def write_task_definition(self) -> None:
        self.paths.ficelle_home.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        # schtasks expects the exported-task encoding: UTF-16 with a BOM.
        write_text_file(self.task_xml, self.task_payload(), encoding="utf-16")

    def install(self) -> int:
        self.write_task_definition()
        self.run_command(["schtasks", "/End", "/TN", self.task_name])
        self.report_stale_servers(self.terminate_stale_servers())
        register = self.run_command(
            ["schtasks", "/Create", "/TN", self.task_name, "/XML", str(self.task_xml), "/F"]
        )
        if register.returncode != 0:
            sys.stderr.write(register.stderr or register.stdout)
            return register.returncode
        start = self.run_command(["schtasks", "/Run", "/TN", self.task_name])
        if start.returncode != 0:
            sys.stderr.write(start.stderr or start.stdout)
            return start.returncode
        if not self.wait_for_ready():
            sys.stderr.write("Ficelle scheduled task started but /admin/status.json did not become ready.\n")
            self.run_command(["schtasks", "/End", "/TN", self.task_name])
            sys.stderr.write(
                "Ended the task so it does not restart-loop; check the logs under FICELLE_HOME and the configured port.\n"
            )
            return 1
        if not persist_service_context(self.paths):
            return 1
        print(f"installed {self.task_xml}")
        return 0

    def restart(self) -> int:
        # Unlike the LaunchAgent's recorded_interpreter(), this re-decides the interpreter
        # from the current process on every restart — the systemd backend shares that
        # trade-off. Revisit if the second-install repointing incident recurs on Windows.
        return self.install()

    def stop(self) -> int:
        result = self.run_command(["schtasks", "/End", "/TN", self.task_name])
        self.report_stale_servers(self.terminate_stale_servers())
        return result.returncode

    def uninstall(self) -> int:
        self.stop()
        self.run_command(["schtasks", "/Delete", "/TN", self.task_name, "/F"])
        if self.task_xml.exists():
            self.task_xml.unlink()
        print(f"removed {self.task_xml}")
        return 0

    def status(self) -> int:
        # No /V and no label filtering: schtasks localizes its labels, so the compact
        # default listing printed verbatim is the portable answer.
        result = self.run_command(["schtasks", "/Query", "/TN", self.task_name, "/FO", "LIST"])
        print(result.stdout.strip() or result.stderr.strip() or "not registered")
        return result.returncode


class UnsupportedServiceBackend:
    """Explicit backend for platforms that do not have a user-proof service installer yet."""

    name = "unsupported"

    def __init__(self, platform_name: str) -> None:
        self.platform_name = platform_name

    def _fail(self) -> int:
        sys.stderr.write(
            "Ficelle managed service install is supported on macOS LaunchAgent, Linux systemd --user, "
            "and Windows per-user Scheduled Tasks. "
            f"Detected platform: {self.platform_name}. "
            "Run `python -m ficelle.router --serve` for developer foreground mode, or add a platform service backend first.\n"
        )
        return 1

    def install(self) -> int:
        return self._fail()

    def restart(self) -> int:
        return self._fail()

    def stop(self) -> int:
        return self._fail()

    def uninstall(self) -> int:
        return self._fail()

    def status(self) -> int:
        return self._fail()


def select_service_backend(
    *,
    platform_name: str,
    paths: ServicePaths,
    run_command: RunCommand,
    uid_provider: UidProvider,
    wait_for_ready: ReadyCheck,
    terminate_stale_servers: StaleServerTerminator,
    report_stale_servers: StaleServerReporter,
) -> ServiceBackend:
    """Return the managed service backend for the current platform."""
    if platform_name == "darwin":
        return LaunchAgentServiceBackend(
            paths=paths,
            run_command=run_command,
            uid_provider=uid_provider,
            wait_for_ready=wait_for_ready,
            terminate_stale_servers=terminate_stale_servers,
            report_stale_servers=report_stale_servers,
        )
    if platform_name.startswith("linux"):
        return SystemdUserServiceBackend(
            paths=paths,
            run_command=run_command,
            wait_for_ready=wait_for_ready,
            terminate_stale_servers=terminate_stale_servers,
            report_stale_servers=report_stale_servers,
        )
    if platform_name == "win32":
        return WindowsScheduledTaskBackend(
            paths=paths,
            run_command=run_command,
            wait_for_ready=wait_for_ready,
            terminate_stale_servers=terminate_stale_servers,
            report_stale_servers=report_stale_servers,
        )
    return UnsupportedServiceBackend(platform_name)
