"""Windows Scheduled Task entry point for the Ficelle router service.

Task Scheduler has no equivalent of LaunchAgent ``EnvironmentVariables`` or systemd
``Environment=`` lines, and the console-less ``pythonw.exe`` interpreter starts with
``sys.stdout``/``sys.stderr`` set to ``None``, so a bare ``print()`` would crash the
server. This entry point closes both gaps before the router loop starts: it receives
the service environment as ``NAME=VALUE`` assignments (built by
``WindowsScheduledTaskBackend.entry_arguments()`` from ``service_environment()``, the
one definition of that contract), republishes them as environment variables, then
points stdio at the same log files the LaunchAgent backend writes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ficelle.service import ERROR_LOG_FILE_NAME, LOG_FILE_NAME, service_log_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ficelle-windows-entry")
    parser.add_argument(
        "assignments",
        nargs="+",
        metavar="NAME=VALUE",
        help="Service environment assignments, e.g. FICELLE_HOME=C:\\Users\\me\\.ficelle",
    )
    return parser.parse_args(argv)


def service_environment_from_assignments(assignments: list[str]) -> dict[str, str]:
    environment = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not name or not separator or not value:
            raise SystemExit(f"invalid service environment assignment: {assignment!r}")
        environment[name] = value
    if "FICELLE_HOME" not in environment:
        raise SystemExit("a FICELLE_HOME=... assignment is required")
    return environment


def redirect_stdio_to_logs(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(log_dir / LOG_FILE_NAME, "a", encoding="utf-8", buffering=1)
    sys.stderr = open(log_dir / ERROR_LOG_FILE_NAME, "a", encoding="utf-8", buffering=1)


def main(argv: list[str] | None = None) -> int:
    environment = service_environment_from_assignments(parse_args(argv).assignments)
    os.environ.update(environment)
    redirect_stdio_to_logs(service_log_dir(Path(environment["FICELLE_HOME"])))
    # Imported only after the environment is set: router path resolution reads it at import.
    from ficelle import router

    return router.main_args(["--serve"])


if __name__ == "__main__":
    raise SystemExit(main())
