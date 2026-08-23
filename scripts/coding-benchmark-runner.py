#!/usr/bin/env python3
"""Run an official coding benchmark harness at an exact git commit.

Ficelle deliberately delegates task execution to upstream. This wrapper supplies reproducibility:
an immutable commit, a clean checkout, a recorded command/settings fingerprint, and a bounded
machine-readable run record. Network, containers, provider credentials and benchmark licences
remain operator responsibilities.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ficelle.coding_benchmark_policy import (  # noqa: E402
    BENCHMARK_POLICIES,
    CodingBenchmarkPolicyError,
    canonical_repository,
    policy_fingerprint,
    validate_model_identity,
)

def strict_json_object(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("settings must contain an object")
    return payload


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
    policy = BENCHMARK_POLICIES.get(args.benchmark)
    if policy is None:
        sys.stderr.write("coding-benchmark-runner: benchmark is not in the pinned coding policy\n")
        return 2
    try:
        requested_repository = canonical_repository(args.repository)
    except CodingBenchmarkPolicyError as exc:
        sys.stderr.write(f"coding-benchmark-runner: {exc}\n")
        return 2
    if requested_repository != canonical_repository(policy.harness_repository) or commit != policy.harness_commit:
        sys.stderr.write("coding-benchmark-runner: harness repository or commit is outside pinned policy\n")
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
    try:
        settings_payload = strict_json_object(settings)
        provider, upstream_model_id = validate_model_identity(
            str(settings_payload.get("provider") or ""),
            str(settings_payload.get("upstream_model_id") or ""),
        )
        run_mode = str(settings_payload.get("run_mode") or "")
        if run_mode not in {"calibration", "certification"}:
            raise ValueError("settings run_mode must be calibration or certification")
        expected_tasks = (
            policy.calibration_task_count
            if run_mode == "calibration"
            else policy.certification_task_count
        )
        expected_attempts = (
            policy.calibration_attempts_per_task
            if run_mode == "calibration"
            else policy.certification_attempts_per_task
        )
        if expected_tasks is None or expected_attempts is None:
            raise ValueError("certification sample is not frozen for this benchmark")
        if settings_payload.get("task_count") != expected_tasks:
            raise ValueError("settings task_count is outside pinned policy")
        if settings_payload.get("attempts_per_task") != expected_attempts:
            raise ValueError("settings attempts_per_task is outside pinned policy")
        if settings_payload.get("wall_clock_timeout_seconds") != policy.calibration_wall_clock_seconds:
            raise ValueError("settings wall_clock_timeout_seconds is outside pinned policy")
    except (CodingBenchmarkPolicyError, json.JSONDecodeError, OSError, ValueError) as exc:
        sys.stderr.write(f"coding-benchmark-runner: invalid settings: {exc}\n")
        return 2
    if result.exists():
        sys.stderr.write("coding-benchmark-runner: result path must not already exist\n")
        return 2
    if record.exists():
        sys.stderr.write("coding-benchmark-runner: run record path must not already exist\n")
        return 2
    if result == record:
        sys.stderr.write("coding-benchmark-runner: result and run record paths must differ\n")
        return 2
    for name in args.pass_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            sys.stderr.write("coding-benchmark-runner: invalid --pass-env name\n")
            return 2
        if name not in os.environ:
            sys.stderr.write(f"coding-benchmark-runner: requested environment variable is unset: {name}\n")
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
        if resolved != commit:
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
        "harness_repository": requested_repository,
        "harness_commit": resolved,
        "run_mode": run_mode,
        "provider": provider,
        "upstream_model_id": upstream_model_id,
        "policy_fingerprint": policy_fingerprint(),
        "source_revisions": [
            {"name": source.name, "repository": source.repository, "commit": source.commit}
            for source in policy.sources
        ],
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
