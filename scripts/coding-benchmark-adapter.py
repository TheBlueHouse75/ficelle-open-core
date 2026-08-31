#!/usr/bin/env python3
"""Thin adapters from pinned upstream coding harnesses to Ficelle result JSON.

Adapters never decide whether a model is certifiable. They execute the exact harness and sources
from :mod:`ficelle.coding_benchmark_policy`, then emit evidence for the common normalizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ficelle.coding_benchmark_policy import (  # noqa: E402
    AIDER_CALIBRATION_TASKS,
    BENCHMARK_POLICIES,
    CodingBenchmarkPolicyError,
    aider_model_settings,
    canonical_repository,
    policy_fingerprint,
    validate_evidence_metadata,
    validate_model_identity,
)


AIDER_DOCKERFILE_SHA256 = "352813bb478c35b88981d03a23679b33f5d1a78871c53173a68b52422fe68e80"
AIDER_BENCHMARK_SHA256 = "ba350b7b3ebcc9da6f588c0724dade552791b3c5edd9f16641eadb5931de9c30"
AIDER_JAVA_CALIBRATION_TASK = "java/exercises/practice/bank-account"
AIDER_GRADLE_RESOLVE_SCRIPT = """\
allprojects {
    tasks.register("ficelleResolveTestRuntime") {
        doLast {
            configurations.getByName("testRuntimeClasspath").files.each { file -> file.length() }
        }
    }
}
"""
AIDER_CALIBRATION_NETWORK_ENV = """\
ENV NPM_CONFIG_FETCH_RETRIES=5 \\
    NPM_CONFIG_FETCH_RETRY_MINTIMEOUT=1000 \\
    NPM_CONFIG_FETCH_RETRY_MAXTIMEOUT=10000 \\
    NPM_CONFIG_FETCH_TIMEOUT=60000
"""
RELAY_TOKEN_ENV = "FICELLE_BENCHMARK_RELAY_TOKEN"
_PROVIDER_ERROR_PATTERN = re.compile(
    r"(?:http|status|error)[^\n]{0,30}(?:408|429|500|502|503|504)|"
    r"rate.?limit|cooldown|provider|upstream|connection|remote.?disconnect|timed?\s*out",
    re.IGNORECASE,
)
_MODEL_RESPONSE_FAILURE_PATTERN = re.compile(
    r"truncated_before_content|empty_assistant_message|"
    r"completion token budget ran out before .* emitted .* content",
    re.IGNORECASE,
)
_PROVIDER_RETRY_PATTERN = re.compile(
    r"litellm\.(?:APIConnectionError|BadGatewayError|RateLimitError|ServiceUnavailableError|Timeout)|"
    r"API provider's servers are down or overloaded|provider_cooldown|RemoteDisconnected|"
    r"rate limit exceeded|upstream_failure|RequestDeadlineExceeded|request_deadline_exceeded",
    re.IGNORECASE,
)


class BenchmarkInterrupted(RuntimeError):
    """The operator stopped the benchmark and cleanup must still run."""


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {token}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("settings must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _required_text(settings: dict[str, Any], field: str) -> str:
    value = settings.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"settings {field} must be a non-empty string")
    return value.strip()


def _validate_common_settings(settings: dict[str, Any], benchmark: str) -> tuple[str, str]:
    if settings.get("benchmark") != benchmark:
        raise ValueError("settings benchmark does not match adapter")
    provider, upstream_model_id = validate_model_identity(
        _required_text(settings, "provider"), _required_text(settings, "upstream_model_id")
    )
    expected_ficelle_id = f"ficelle/{provider}/{upstream_model_id}"
    if settings.get("ficelle_model_id") != expected_ficelle_id:
        raise ValueError("ficelle_model_id must bind the exact provider and upstream model")
    base_url = urlparse(_required_text(settings, "base_url"))
    if (
        base_url.scheme != "http"
        or base_url.hostname != "host.docker.internal"
        or base_url.path.rstrip("/") != "/v1"
        or base_url.query
        or base_url.fragment
    ):
        raise ValueError("base_url must be a Docker-to-host Ficelle /v1 endpoint")
    host_header = settings.get("ficelle_host_header", "127.0.0.1:8646")
    if not isinstance(host_header, str) or not re.fullmatch(r"127\.0\.0\.1:(?:[1-9][0-9]{0,4})", host_header):
        raise ValueError("ficelle_host_header must be an explicit IPv4 loopback authority")
    if int(host_header.rsplit(":", 1)[1]) > 65535:
        raise ValueError("ficelle_host_header port is out of range")
    return provider, upstream_model_id


def _clone_revision(destination: Path, repository: str, commit: str) -> None:
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth=1", "origin", commit],
        check=True,
    )
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit], check=True)
    resolved = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != commit:
        raise ValueError("secondary source checkout does not match pinned commit")


def _aider_calibration_recipe(official_dockerfile: bytes) -> tuple[str, str]:
    if hashlib.sha256(official_dockerfile).hexdigest() != AIDER_DOCKERFILE_SHA256:
        raise ValueError("pinned Aider Dockerfile fingerprint does not match policy")
    recipe = official_dockerfile.decode("utf-8").replace(
        "FROM buildpack-deps:jammy\n",
        "FROM buildpack-deps:jammy\n" + AIDER_CALIBRATION_NETWORK_ENV + "\n",
        1,
    )
    unpinned_jest = "    npm install \\\n    jest \\\n"
    if recipe.count(unpinned_jest) != 1:
        raise ValueError("pinned Aider Dockerfile no longer has the expected Jest dependency")
    recipe = recipe.replace(unpinned_jest, "    npm install \\\n    jest@29.7.0 \\\n", 1)
    copy_source = "COPY . /aider\n"
    if recipe.count(copy_source) != 1:
        raise ValueError("pinned Aider Dockerfile no longer has the expected source copy")
    java_prewarm = (
        copy_source
        + "RUN cd /aider/benchmark/ficelle-build-cache/java-bank-account \\\n"
        + "    && ./gradlew --no-daemon testClasses \\\n"
        + "    && ./gradlew --no-daemon --init-script ficelle-resolve.gradle "
        + "ficelleResolveTestRuntime \\\n"
        + "    && rm -rf build .gradle\n"
    )
    recipe = recipe.replace(copy_source, java_prewarm, 1)
    return recipe, "sha256:" + hashlib.sha256(recipe.encode("utf-8")).hexdigest()


def _stage_aider_build_cache(checkout: Path, source_checkout: Path) -> None:
    source = source_checkout / AIDER_JAVA_CALIBRATION_TASK
    if not source.is_dir():
        raise ValueError("pinned Aider Java task is missing for dependency prewarm")
    destination = checkout / "benchmark" / "ficelle-build-cache" / "java-bank-account"
    shutil.copytree(source, destination)
    (destination / "ficelle-resolve.gradle").write_text(
        AIDER_GRADLE_RESOLVE_SCRIPT,
        encoding="utf-8",
    )


def _aider_calibration_harness(official_harness: bytes) -> tuple[str, str]:
    if hashlib.sha256(official_harness).hexdigest() != AIDER_BENCHMARK_SHA256:
        raise ValueError("pinned Aider benchmark fingerprint does not match policy")
    harness = official_harness.decode("utf-8")
    shuffle = "    random.shuffle(test_dnames)\n"
    if harness.count(shuffle) != 1:
        raise ValueError("pinned Aider benchmark no longer has the expected task shuffle")
    harness = harness.replace(shuffle, "    test_dnames.sort()\n", 1)
    completion = "    summarize_results(dirname)\n\n    return 0\n"
    if harness.count(completion) != 1:
        raise ValueError("pinned Aider benchmark no longer has the expected completion block")
    harness = harness.replace(
        completion,
        "    summarize_results(dirname)\n"
        "    sys.stdout.flush()\n"
        "    sys.stderr.flush()\n"
        "    os._exit(0)\n\n"
        "    return 0\n",
        1,
    )
    return harness, "sha256:" + hashlib.sha256(harness.encode("utf-8")).hexdigest()


def _aider_model_settings(completion_token_budget: int) -> tuple[str, str]:
    """Give every candidate the same output room, including reasoning tokens."""
    policy = BENCHMARK_POLICIES["aider-polyglot"]
    if completion_token_budget != policy.completion_token_budget:
        raise ValueError("completion token budget does not match pinned policy")
    return aider_model_settings(policy)


def _aider_result_rows(
    run_dir: Path,
    task_ids: tuple[str, ...],
    *,
    run_diagnostic: str = "",
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    completed = 0
    model_failure_pending = bool(_MODEL_RESPONSE_FAILURE_PATTERN.search(run_diagnostic))
    provider_error_pending = bool(_PROVIDER_RETRY_PATTERN.search(run_diagnostic))
    for task_id in task_ids:
        result_path = run_dir / task_id / ".aider.results.json"
        row: dict[str, Any] = {"task_id": task_id, "passed": False, "status": "harness_error"}
        if result_path.is_file():
            payload = _read_json(result_path)
            duration = payload.get("duration")
            if (
                isinstance(duration, (int, float))
                and not isinstance(duration, bool)
                and math.isfinite(duration)
                and duration >= 0
            ):
                row["duration_seconds"] = float(duration)
            for metric in ("prompt_tokens", "completion_tokens"):
                value = payload.get(metric)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    row[metric] = value
            outcomes = payload.get("tests_outcomes")
            if isinstance(outcomes, list) and outcomes:
                row["passed"] = bool(outcomes[0])
                row["status"] = "passed" if row["passed"] else "failed"
                completed += 1
            elif payload.get("exception"):
                diagnostic = json.dumps(payload.get("exception"), ensure_ascii=False, default=str)
                if _MODEL_RESPONSE_FAILURE_PATTERN.search(diagnostic):
                    row["status"] = "failed"
                    completed += 1
                    model_failure_pending = False
                elif _PROVIDER_ERROR_PATTERN.search(diagnostic):
                    row["status"] = "provider_error"
                    provider_error_pending = False
        elif model_failure_pending:
            # Spending the common output budget without emitting editable content is a model
            # verdict. The router reports it as an upstream-style HTTP error only because no
            # completion body can be returned.
            row["status"] = "failed"
            completed += 1
            model_failure_pending = False
        elif provider_error_pending:
            # Tasks run sequentially. If an abnormal exit ends a provider retry loop, the first
            # missing result is the in-flight provider incident; later tasks were never run.
            row["status"] = "provider_error"
            provider_error_pending = False
        rows.append(row)
    return rows, completed


def _aider_efficiency_summary(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    measured = [
        row
        for row in rows
        if row.get("status") in {"passed", "failed"}
        and all(field in row for field in ("duration_seconds", "prompt_tokens", "completion_tokens"))
    ]
    durations = [row["duration_seconds"] for row in measured]
    prompt_tokens = sum(row["prompt_tokens"] for row in measured)
    completion_tokens = sum(row["completion_tokens"] for row in measured)
    duration_seconds = sum(durations)
    return {
        "measured_task_count": len(durations),
        "duration_seconds": duration_seconds,
        "mean_duration_seconds": duration_seconds / len(durations) if durations else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def run_aider(settings: dict[str, Any], result_path: Path) -> None:
    policy = BENCHMARK_POLICIES["aider-polyglot"]
    _validate_common_settings(settings, policy.name)
    run_mode = str(settings.get("run_mode") or "")
    if run_mode not in {"calibration", "certification"}:
        raise ValueError("Aider run_mode must be calibration or certification")
    if settings.get("task_count") != len(AIDER_CALIBRATION_TASKS):
        raise ValueError("Aider calibration task_count does not match pinned task set")
    if settings.get("attempts_per_task") != 1:
        raise ValueError("Aider calibration requires one attempt per task")
    if settings.get("tasks") != list(AIDER_CALIBRATION_TASKS):
        raise ValueError("Aider calibration tasks do not match pinned task set")
    if settings.get("edit_format") != "whole" or settings.get("threads") != 1:
        raise ValueError("Aider calibration requires whole edit format and one thread")
    if settings.get("temperature") != 0:
        raise ValueError("Aider calibration temperature must be zero")
    if settings.get("wall_clock_timeout_seconds") != policy.calibration_wall_clock_seconds:
        raise ValueError("Aider calibration wall-clock timeout does not match pinned policy")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must contain the Ficelle owner token")
    if shutil.which("docker") is None:
        raise ValueError("Docker is required for the Aider benchmark")

    checkout = Path.cwd().resolve()
    resolved_harness = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if resolved_harness != policy.harness_commit:
        raise ValueError("Aider checkout does not match pinned harness commit")

    benchmark_dir = checkout / "tmp.benchmarks"
    benchmark_dir.mkdir(exist_ok=True)
    source = policy.sources[0]
    source_checkout = benchmark_dir / source.name
    _clone_revision(source_checkout, source.repository, source.commit)
    subset = benchmark_dir / "ficelle-calibration"
    for task_id in AIDER_CALIBRATION_TASKS:
        task_source = source_checkout / task_id
        if not task_source.is_dir():
            raise ValueError(f"pinned Aider task is missing: {task_id}")
        task_destination = subset / task_id
        task_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task_source, task_destination)
    _stage_aider_build_cache(checkout, source_checkout)

    recipe, recipe_fingerprint = _aider_calibration_recipe(
        (checkout / "benchmark" / "Dockerfile").read_bytes()
    )
    recipe_path = benchmark_dir / "Dockerfile.ficelle-calibration"
    recipe_path.write_text(recipe, encoding="utf-8")
    harness, harness_recipe_fingerprint = _aider_calibration_harness(
        (checkout / "benchmark" / "benchmark.py").read_bytes()
    )
    harness_path = benchmark_dir / "benchmark.ficelle-calibration.py"
    harness_path.write_text(harness, encoding="utf-8")
    model_settings, model_settings_fingerprint = _aider_model_settings(
        policy.completion_token_budget
    )
    model_settings_path = benchmark_dir / "model-settings.ficelle-calibration.yml"
    model_settings_path.write_text(model_settings, encoding="utf-8")
    image = f"ficelle-aider-benchmark:{policy.harness_commit[:12]}"
    subprocess.run(
        ["docker", "build", "--file", str(recipe_path), "--tag", image, "."],
        check=True,
    )
    image_id = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not image_id.startswith("sha256:"):
        raise ValueError("Docker did not return a content-addressed image id")
    before = {path.resolve() for path in benchmark_dir.iterdir() if path.is_dir()}
    network_name = f"ficelle-aider-{os.getpid()}-{secrets.token_hex(4)}"
    proxy_name = network_name + "-proxy"
    relay_token = secrets.token_urlsafe(32)
    network_created = False
    proxy_started = False
    subprocess.run(["docker", "network", "create", "--internal", network_name], check=True)
    network_created = True
    cid_path = benchmark_dir / "aider-container.cid"
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                proxy_name,
                "--network",
                network_name,
                "--add-host=host.docker.internal:host-gateway",
                "--volume",
                f"{REPO_ROOT / 'scripts' / 'coding-benchmark-loopback-proxy.py'}:/ficelle/loopback-proxy.py:ro",
                "--env",
                "OPENAI_API_KEY",
                "--env",
                RELAY_TOKEN_ENV,
                image,
                "python3",
                "/ficelle/loopback-proxy.py",
                "--target",
                str(settings["base_url"]),
                "--host-header",
                str(settings.get("ficelle_host_header", "127.0.0.1:8646")),
                "--listen-host",
                "0.0.0.0",
                "--serve-only",
            ],
            check=True,
            env={**os.environ, "OPENAI_API_KEY": api_key, RELAY_TOKEN_ENV: relay_token},
            stdout=subprocess.DEVNULL,
        )
        proxy_started = True
        subprocess.run(["docker", "network", "connect", "bridge", proxy_name], check=True)
    except Exception:
        if proxy_started:
            subprocess.run(
                ["docker", "stop", "--timeout", "5", proxy_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if network_created:
            subprocess.run(
                ["docker", "network", "rm", network_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        raise
    container_command = [
        "docker",
        "run",
        "--rm",
        "--cidfile",
        str(benchmark_dir / "aider-container.cid"),
        "--memory=12g",
        "--memory-swap=12g",
        "--network",
        network_name,
        "--volume",
        f"{checkout}:/aider",
        "--volume",
        f"{harness_path}:/aider/benchmark/benchmark.py:ro",
        "--volume",
        f"{benchmark_dir}:/benchmarks",
        "--env",
        f"OPENAI_API_KEY={relay_token}",
        "--env",
        f"OPENAI_API_BASE=http://{proxy_name}:8765/v1",
        "--env",
        "AIDER_DOCKER=1",
        "--env",
        "AIDER_BENCHMARK_DIR=/benchmarks",
        image,
        "python3",
        "benchmark/benchmark.py",
        "ficelle-calibration",
        "--model",
        f"openai/{settings['ficelle_model_id']}",
        "--edit-format",
        "whole",
        "--threads",
        "1",
        "--tries",
        "1",
        "--num-tests",
        str(len(AIDER_CALIBRATION_TASKS)),
        "--exercises-dir",
        "ficelle-calibration",
        "--read-model-settings",
        "/benchmarks/model-settings.ficelle-calibration.yml",
    ]
    timed_out = False
    harness_exit_code = 0
    harness_log_path = benchmark_dir / "aider-harness.log"
    try:
        with harness_log_path.open("wb") as harness_log:
            try:
                harness_exit_code = subprocess.run(
                    container_command,
                    check=False,
                    timeout=policy.calibration_wall_clock_seconds,
                    stdout=harness_log,
                    stderr=subprocess.STDOUT,
                ).returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                harness_exit_code = 124
                container_id = (
                    cid_path.read_text(encoding="utf-8").strip() if cid_path.is_file() else ""
                )
                if container_id and all(
                    character in "0123456789abcdef" for character in container_id.lower()
                ):
                    subprocess.run(
                        ["docker", "stop", "--timeout", "5", container_id],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
    finally:
        container_id = cid_path.read_text(encoding="utf-8").strip() if cid_path.is_file() else ""
        if container_id and all(character in "0123456789abcdef" for character in container_id.lower()):
            subprocess.run(
                ["docker", "stop", "--timeout", "5", container_id],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if proxy_started:
            subprocess.run(
                ["docker", "stop", "--timeout", "5", proxy_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if network_created:
            subprocess.run(
                ["docker", "network", "rm", network_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    created = [
        path
        for path in benchmark_dir.iterdir()
        if path.is_dir() and path.resolve() not in before
    ]
    if len(created) != 1:
        raise ValueError("Aider harness did not create exactly one result directory")
    run_diagnostic = ""
    if harness_exit_code != 0 and harness_log_path.is_file():
        with harness_log_path.open("rb") as harness_log:
            harness_log.seek(max(0, harness_log_path.stat().st_size - 65_536))
            run_diagnostic = harness_log.read().decode("utf-8", errors="replace")
    rows, completed = _aider_result_rows(
        created[0],
        AIDER_CALIBRATION_TASKS,
        run_diagnostic=run_diagnostic,
    )
    model_verdict_count = sum(1 for row in rows if row["status"] in {"passed", "failed"})
    provider_error_count = sum(1 for row in rows if row["status"] == "provider_error")
    harness_error_count = sum(1 for row in rows if row["status"] == "harness_error")
    payload: dict[str, Any] = {
        "benchmark": policy.name,
        "run_mode": run_mode,
        "suite_version": policy.suite_version,
        "harness_repository": canonical_repository(policy.harness_repository),
        "harness_commit": policy.harness_commit,
        "reference_agent": policy.reference_agent,
        "reference_agent_commit": policy.reference_agent_commit,
        "source_revisions": [
            {"name": item.name, "repository": item.repository, "commit": item.commit}
            for item in policy.sources
        ],
        "policy_fingerprint": policy_fingerprint(),
        "task_count": len(rows),
        "completed_count": completed,
        "model_verdict_count": model_verdict_count,
        "provider_error_count": provider_error_count,
        "harness_error_count": harness_error_count,
        "attempts_per_task": 1,
        "container_image_id": image_id,
        "container_recipe_fingerprint": recipe_fingerprint,
        "harness_recipe_fingerprint": harness_recipe_fingerprint,
        "model_settings_fingerprint": model_settings_fingerprint,
        "completion_token_budget": policy.completion_token_budget,
        "harness_exit_code": harness_exit_code,
        "timed_out": timed_out,
        "wall_clock_timeout_seconds": policy.calibration_wall_clock_seconds,
        # Availability and harness incidents are unmeasured, never coding failures. A publishable
        # qualification still requires a model verdict for every frozen task.
        "pass_at_1": (
            sum(1 for row in rows if row["passed"]) / model_verdict_count
            if model_verdict_count
            else 0.0
        ),
        "efficiency": _aider_efficiency_summary(rows),
        "results": rows,
    }
    validate_evidence_metadata(payload, run_mode=run_mode)
    _write_json(result_path, payload)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("adapter", choices=("aider-polyglot",))
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings_path = os.getenv("FICELLE_BENCHMARK_SETTINGS", "")
    result_path = os.getenv("FICELLE_BENCHMARK_RESULT", "")
    if not settings_path or not result_path:
        sys.stderr.write("coding-benchmark-adapter: runner environment is missing\n")
        return 2
    def interrupt(_signum: int, _frame: object) -> None:
        raise BenchmarkInterrupted("benchmark interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        settings = _read_json(Path(settings_path))
        if args.adapter == "aider-polyglot":
            run_aider(settings, Path(result_path))
        return 0
    except BenchmarkInterrupted:
        sys.stderr.write("coding-benchmark-adapter: interrupted after cleanup\n")
        return 130
    except (CodingBenchmarkPolicyError, json.JSONDecodeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        sys.stderr.write(f"coding-benchmark-adapter: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
