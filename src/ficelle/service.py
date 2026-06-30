"""Local service backends for Ficelle."""
from __future__ import annotations

import plistlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


RunCommand = Callable[[list[str]], subprocess.CompletedProcess[str]]
ReadyCheck = Callable[[], bool]
StaleServerTerminator = Callable[[], list[int]]
StaleServerReporter = Callable[[list[int]], None]
UidProvider = Callable[[], str]


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
    hermes_home: Path
    ficelle_dir: Path
    label: str
    plist: Path
    systemd_unit: Path
    install_python: Path
    log_dir: Path


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
            "StandardOutPath": str(self.paths.log_dir / "ficelle.log"),
            "StandardErrorPath": str(self.paths.log_dir / "ficelle.error.log"),
            "WorkingDirectory": str(self.paths.ficelle_dir),
            "EnvironmentVariables": {
                "HERMES_HOME": str(self.paths.hermes_home),
            },
        }

    def bootout(self) -> subprocess.CompletedProcess[str]:
        return self.run_command(["launchctl", "bootout", f"gui/{self.uid_provider()}/{self.paths.label}"])

    def install(self) -> int:
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
            sys.stderr.write("Ficelle LaunchAgent started but /admin/status.json did not become ready.\n")
            return 1
        print(f"installed {self.paths.plist}")
        return 0

    def restart(self) -> int:
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
        return "\n".join(
            [
                "[Unit]",
                "Description=Ficelle local OpenAI-compatible model router",
                "After=network-online.target",
                "Wants=network-online.target",
                "",
                "[Service]",
                "Type=simple",
                f"WorkingDirectory={self.paths.ficelle_dir}",
                f"Environment=HERMES_HOME={self.paths.hermes_home}",
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

    def install(self) -> int:
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.ficelle_dir.mkdir(parents=True, exist_ok=True)
        write_text_file(self.paths.systemd_unit, self.unit_payload())
        reload_result = self.daemon_reload()
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
        print(f"installed {self.paths.systemd_unit}")
        return 0

    def restart(self) -> int:
        if not self.paths.systemd_unit.exists():
            return self.install()
        self.report_stale_servers(self.terminate_stale_servers())
        result = self.run_command(["systemctl", "--user", "restart", self.unit_name])
        if result.returncode != 0:
            sys.stderr.write(result.stderr or result.stdout)
            return result.returncode
        if not self.wait_for_ready():
            sys.stderr.write("Ficelle systemd user service restarted but /admin/status.json did not become ready.\n")
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


class UnsupportedServiceBackend:
    """Explicit backend for platforms that do not have a user-proof service installer yet."""

    name = "unsupported"

    def __init__(self, platform_name: str) -> None:
        self.platform_name = platform_name

    def _fail(self) -> int:
        sys.stderr.write(
            "Ficelle managed service install is supported on macOS LaunchAgent and Linux systemd --user. "
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
    return UnsupportedServiceBackend(platform_name)
