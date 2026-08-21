#!/usr/bin/env python3
"""Run an official coding benchmark harness at an exact git commit.

Ficelle deliberately delegates task execution to upstream. This wrapper supplies reproducibility:
an immutable commit, a clean checkout, a recorded command/settings fingerprint, and a bounded
machine-readable run record. Network, containers, provider credentials and benchmark licences
remain operator responsibilities. The harness is refused on any process that can access Ficelle's
manifest-signing material: benchmark execution and signing must happen in separate trust domains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


SIGNING_KEY_ENV = "FICELLE_CODING_CERT_PRIVATE_KEY_B64"
SIGNING_KEYCHAIN_SERVICE = "ai.ficelle.coding-certification"
SIGNING_KEYCHAIN_ACCOUNT = "manifest-signing-key-v1"


def signing_material_is_accessible() -> bool:
    """Fail closed before giving an untrusted harness this process' OS authority."""
    if os.getenv(SIGNING_KEY_ENV, "").strip():
        return True
    if sys.platform != "darwin":
        return False
    probe = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-s",
            SIGNING_KEYCHAIN_SERVICE,
            "-a",
            SIGNING_KEYCHAIN_ACCOUNT,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--benchmark", required=True)
    root.add_argument("--repository", required=True)
    root.add_argument("--commit", required=True)
    root.add_argument("--settings", type=Path, required=True)
    root.add_argument("--result", type=Path, required=True, help="Official harness JSON output path")
    root.add_argument("--record", type=Path, required=True, help="Ficelle run metadata output")
    root.add_argument(
        "--pass-env",
        action="append",
        default=[],
        help="Environment variable to pass explicitly to the untrusted harness (repeatable)",
    )
    root.add_argument("command", nargs=argparse.REMAINDER, help="Official harness command after --")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repository = urlparse(args.repository)
    commit = args.commit.lower()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if repository.scheme != "https" or repository.hostname != "github.com":
        sys.stderr.write("coding-benchmark-runner: repository must be an HTTPS GitHub URL\n")
        return 2
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        sys.stderr.write("coding-benchmark-runner: commit must be a full 40-character hexadecimal git commit\n")
        return 2
    if not command:
        sys.stderr.write("coding-benchmark-runner: official harness command is required after --\n")
        return 2
    settings = args.settings.resolve()
    result = args.result.resolve()
    record = args.record.resolve()
    if not settings.is_file():
        sys.stderr.write("coding-benchmark-runner: settings file does not exist\n")
        return 2
    if result.exists():
        sys.stderr.write("coding-benchmark-runner: result path must not already exist\n")
        return 2
    for name in args.pass_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            sys.stderr.write("coding-benchmark-runner: invalid --pass-env name\n")
            return 2
        if name == SIGNING_KEY_ENV:
            sys.stderr.write("coding-benchmark-runner: refusing to pass the certification signing key\n")
            return 2
        if name not in os.environ:
            sys.stderr.write(f"coding-benchmark-runner: requested environment variable is unset: {name}\n")
            return 2
    if signing_material_is_accessible():
        sys.stderr.write(
            "coding-benchmark-runner: signing material is accessible; run the untrusted harness "
            "under a separate account/container/host with no Ficelle signing key\n"
        )
        return 2
    result.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    with tempfile.TemporaryDirectory(prefix="ficelle-coding-benchmark-") as temporary:
        checkout = Path(temporary) / "harness"
        clone = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", args.repository, str(checkout)],
            check=False,
        )
        if clone.returncode != 0:
            return clone.returncode
        fetch = subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", "origin", commit], check=False)
        if fetch.returncode != 0:
            return fetch.returncode
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
        resolved = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not resolved.startswith(commit):
            sys.stderr.write("coding-benchmark-runner: checkout does not match requested commit\n")
            return 2
        environment = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "TMPDIR")
            if key in os.environ
        }
        environment.update({name: os.environ[name] for name in args.pass_env})
        environment.update({
            "FICELLE_BENCHMARK_SETTINGS": str(settings),
            "FICELLE_BENCHMARK_RESULT": str(result),
        })
        completed = subprocess.run(command, cwd=checkout, env=environment, check=False)
    finished_at = datetime.now(UTC)
    metadata = {
        "benchmark": args.benchmark,
        "harness_repository": args.repository,
        "harness_commit": resolved,
        "settings_fingerprint": "sha256:" + hashlib.sha256(settings.read_bytes()).hexdigest(),
        "command_executable": Path(command[0]).name,
        "command_fingerprint": "sha256:" + hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "exit_code": completed.returncode,
        "official_result_exists": result.is_file(),
        "official_result_fingerprint": (
            "sha256:" + hashlib.sha256(result.read_bytes()).hexdigest() if result.is_file() else None
        ),
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return completed.returncode if completed.returncode else (0 if result.is_file() else 1)


if __name__ == "__main__":
    raise SystemExit(main())
