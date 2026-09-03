#!/usr/bin/env python3
"""Normalize coding benchmark results and build Ficelle's bundled qualification manifest.

This tool consumes machine-readable output from official benchmark harnesses. It does not run or
reimplement their tasks; ``coding-benchmark-runner.py`` pins and executes those upstream harnesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ficelle.coding_certification import (  # noqa: E402
    COMPATIBILITY_CANARY_VERSION,
    LEGACY_VERIFIED_BENCHMARKS,
    POLICY_VERSION,
    REQUIRED_BENCHMARKS,
    validate_manifest,
)
from ficelle.coding_benchmark_policy import (  # noqa: E402
    MIN_QUALIFYING_PASS_AT_1,
    MIN_QUALIFYING_RESOLVED_RATE,
    CodingBenchmarkPolicyError,
    canonical_repository,
    validate_evidence_metadata,
    validate_model_identity,
)


BENCHMARK_WEIGHTS = {
    "aider-polyglot": 1.0,
    "aider-practical": 1.0,
}


def read_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _pass_summary(payload: dict[str, Any]) -> tuple[int, float]:
    task_count = payload.get("task_count") or payload.get("total") or payload.get("total_tasks")
    score = payload.get("pass_at_1")
    if score is None:
        score = payload.get("pass_rate")
    if score is None:
        score = payload.get("resolved_rate")
    rows = payload.get("results") or payload.get("instances") or payload.get("tasks")
    if isinstance(rows, list) and rows:
        verdicts: list[bool] = []
        has_explicit_outcomes = False
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("result rows must be objects")
            status = str(row.get("status") or "").lower()
            if status in {"provider_error", "harness_error"}:
                has_explicit_outcomes = True
                continue
            first_passed = row.get("first_passed")
            if isinstance(first_passed, bool):
                has_explicit_outcomes = True
                verdicts.append(first_passed)
                continue
            if status in {"pass", "passed", "resolved", "success", "fail", "failed"}:
                has_explicit_outcomes = True
                verdicts.append(status in {"pass", "passed", "resolved", "success"})
                continue
            verdict = row.get("passed", row.get("resolved"))
            if isinstance(verdict, bool):
                verdicts.append(verdict)
        if verdicts:
            total = task_count if isinstance(task_count, int) and not isinstance(task_count, bool) else len(rows)
            rate = sum(verdicts) / len(verdicts)
            if has_explicit_outcomes and isinstance(score, (int, float)) and not isinstance(score, bool):
                declared = float(score) / 100.0 if float(score) > 1 else float(score)
                if not math.isclose(rate, declared, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("declared pass_at_1 does not match model verdict rows")
            return total, rate

    if (
        isinstance(task_count, int)
        and not isinstance(task_count, bool)
        and task_count > 0
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
    ):
        rate = float(score)
        return task_count, rate / 100.0 if rate > 1 else rate
    raise ValueError("official result must expose task_count/pass_at_1 or model verdict rows")


def _efficiency_summary(payload: dict[str, Any], task_count: int) -> dict[str, int | float] | None:
    raw = payload.get("efficiency")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("efficiency must be an object")
    measured = raw.get("measured_task_count")
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if isinstance(measured, bool) or not isinstance(measured, int) or not 0 <= measured <= task_count:
        raise ValueError("efficiency measured_task_count is invalid")
    tokens: dict[str, int] = {}
    for field in token_fields:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"efficiency {field} must be a non-negative integer")
        tokens[field] = value
    if tokens["total_tokens"] != tokens["prompt_tokens"] + tokens["completion_tokens"]:
        raise ValueError("efficiency total_tokens does not match its components")
    duration = raw.get("duration_seconds")
    mean = raw.get("mean_duration_seconds")
    for field, value in (("duration_seconds", duration), ("mean_duration_seconds", mean)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"efficiency {field} must be a finite non-negative number")
    expected_mean = float(duration) / measured if measured else 0.0
    if measured == 0 and (float(duration) != 0 or any(tokens.values())):
        raise ValueError("efficiency with zero measured tasks must contain only zero metrics")
    if not math.isclose(float(mean), expected_mean, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("efficiency mean_duration_seconds does not match duration/task count")
    return {
        "measured_task_count": measured,
        "duration_seconds": float(duration),
        "mean_duration_seconds": float(mean),
        **tokens,
    }


def normalize_result(args: argparse.Namespace) -> dict[str, Any]:
    if args.benchmark not in REQUIRED_BENCHMARKS | LEGACY_VERIFIED_BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {args.benchmark}")
    payload = read_json(args.input)
    if isinstance(payload, list):
        payload = {"results": payload}
    if not isinstance(payload, dict):
        raise ValueError("official benchmark result must be an object or result array")
    # Older adapter results did not expose the three outcome counters. Derive them only when every
    # row has an explicit terminal status; the source artifact and its fingerprint stay unchanged.
    result_rows = payload.get("results")
    if isinstance(result_rows, list) and all(isinstance(item, dict) for item in result_rows):
        statuses = [str(item.get("status") or "").lower() for item in result_rows]
        if all(status in {"passed", "failed", "provider_error", "harness_error"} for status in statuses):
            payload.setdefault("model_verdict_count", sum(status in {"passed", "failed"} for status in statuses))
            payload.setdefault("provider_error_count", statuses.count("provider_error"))
            payload.setdefault("harness_error_count", statuses.count("harness_error"))
    policy = validate_evidence_metadata(payload, run_mode=args.run_mode)
    if args.suite_version != policy.suite_version:
        raise ValueError("--suite-version does not match pinned policy")
    task_count, pass_at_1 = _pass_summary(payload)
    if not 0 <= pass_at_1 <= 1:
        raise ValueError("normalized pass_at_1 must be between 0 and 1")
    commit = args.harness_commit.lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("--harness-commit must be a full 40-character hexadecimal git commit")
    settings_bytes = args.settings.read_bytes()
    settings = read_json(args.settings)
    if not isinstance(settings, dict):
        raise ValueError("benchmark settings must be an object")
    settings_provider, settings_model = validate_model_identity(
        str(settings.get("provider") or ""),
        str(settings.get("upstream_model_id") or ""),
    )
    if settings_provider != args.provider.strip().lower() or settings_model != args.model.strip():
        raise ValueError("--provider and --model must match settings provider/upstream_model_id")
    settings_fingerprint = "sha256:" + hashlib.sha256(settings_bytes).hexdigest()
    official_result_fingerprint = "sha256:" + hashlib.sha256(args.input.read_bytes()).hexdigest()
    run_record_bytes = args.run_record.read_bytes()
    run_record = read_json(args.run_record)
    if not isinstance(run_record, dict):
        raise ValueError("run record must be an object")
    expected_record = {
        "benchmark": args.benchmark,
        "harness_repository": canonical_repository(args.harness_repository),
        "harness_commit": commit,
        "run_mode": args.run_mode,
        "provider": settings_provider,
        "upstream_model_id": settings_model,
        "policy_fingerprint": payload["policy_fingerprint"],
        "source_revisions": payload["source_revisions"],
        "settings_fingerprint": settings_fingerprint,
        "official_result_fingerprint": official_result_fingerprint,
        "exit_code": 0,
        "official_result_exists": True,
    }
    mismatched = [key for key, expected in expected_record.items() if run_record.get(key) != expected]
    if mismatched:
        raise ValueError(f"run record does not match result metadata: {', '.join(mismatched)}")
    command_fingerprint = str(run_record.get("command_fingerprint") or "")
    command_digest = command_fingerprint.removeprefix("sha256:")
    if len(command_digest) != 64 or any(character not in "0123456789abcdef" for character in command_digest):
        raise ValueError("run record command_fingerprint must be a SHA-256 digest")
    row: dict[str, Any] = {
        "provider": args.provider.lower(),
        "upstream_model_id": args.model,
        "name": args.benchmark,
        "suite_version": policy.suite_version,
        "harness_repository": canonical_repository(args.harness_repository),
        "harness_commit": commit,
        "task_count": task_count,
        "pass_at_1": round(pass_at_1, 8),
        "run_mode": args.run_mode,
        "attempts_per_task": payload["attempts_per_task"],
        "completed_count": payload.get("completed_count"),
        "model_verdict_count": payload.get("model_verdict_count"),
        "provider_error_count": payload.get("provider_error_count"),
        "harness_error_count": payload.get("harness_error_count"),
        "harness_exit_code": payload["harness_exit_code"],
        "timed_out": payload["timed_out"],
        "wall_clock_timeout_seconds": payload["wall_clock_timeout_seconds"],
        "reference_agent": policy.reference_agent,
        "reference_agent_commit": policy.reference_agent_commit,
        "source_revisions": payload["source_revisions"],
        "policy_fingerprint": payload["policy_fingerprint"],
        "completion_token_budget": payload["completion_token_budget"],
        "relay_timeout_seconds": payload.get(
            "relay_timeout_seconds",
            policy.relay_timeout_seconds,
        ),
        "model_settings_fingerprint": payload["model_settings_fingerprint"],
        "settings_fingerprint": settings_fingerprint,
        "run_record_fingerprint": "sha256:" + hashlib.sha256(run_record_bytes).hexdigest(),
        "official_result_fingerprint": official_result_fingerprint,
        "command_fingerprint": command_fingerprint,
        "evidence_kind": "central_run" if args.run_mode == "certification" else "calibration_run",
        "observed_at": datetime.now(UTC).isoformat(),
    }
    resolved_rate = payload.get("resolved_rate")
    if isinstance(resolved_rate, (int, float)) and not isinstance(resolved_rate, bool):
        normalized_resolved_rate = float(resolved_rate)
        if not math.isfinite(normalized_resolved_rate) or not 0 <= normalized_resolved_rate <= 1:
            raise ValueError("resolved_rate must be between 0 and 1")
        if isinstance(result_rows, list):
            final_verdicts = [
                bool(item["passed"])
                for item in result_rows
                if isinstance(item, dict)
                and str(item.get("status") or "").lower() in {"passed", "failed"}
                and isinstance(item.get("passed"), bool)
            ]
            if final_verdicts:
                measured_resolved_rate = sum(final_verdicts) / len(final_verdicts)
                if not math.isclose(
                    normalized_resolved_rate,
                    measured_resolved_rate,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("declared resolved_rate does not match final model verdict rows")
        row["resolved_rate"] = round(normalized_resolved_rate, 8)
    if isinstance(result_rows, list):
        row["results"] = result_rows
    relay_audit = payload.get("relay_audit")
    if isinstance(relay_audit, dict):
        row["relay_audit"] = relay_audit
    extended_token_diagnostic = payload.get("extended_token_diagnostic")
    if isinstance(extended_token_diagnostic, dict):
        row["extended_token_diagnostic"] = extended_token_diagnostic
    languages = payload.get("languages") or payload.get("language_breakdown")
    if isinstance(languages, dict):
        row["languages"] = languages
    efficiency = _efficiency_summary(payload, task_count)
    if efficiency is not None:
        row["efficiency"] = efficiency
    return row


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in args.results:
        row = read_json(path)
        if not isinstance(row, dict):
            raise ValueError(f"normalized result must be an object: {path}")
        run_mode = str(row.get("run_mode") or "")
        validate_evidence_metadata({**row, "benchmark": row.get("name")}, run_mode=run_mode)
        if int(row.get("model_verdict_count") or 0) != int(row.get("task_count") or 0):
            raise ValueError("qualification requires a model verdict for every task")
        identity = str(row.get("upstream_model_id") or "")
        if not identity or not str(row.get("provider") or ""):
            raise ValueError(f"normalized result is missing provider/model identity: {path}")
        grouped.setdefault(identity, []).append(row)

    certifications = []
    now = datetime.now(UTC)
    for model_id, rows in sorted(grouped.items()):
        names = {str(row.get("name") or "") for row in rows}
        if names != REQUIRED_BENCHMARKS or len(rows) != len(REQUIRED_BENCHMARKS):
            raise ValueError(f"{model_id} does not have the exact required benchmark set")
        provider = str(rows[0]["provider"]).lower()
        score = sum(
            float(row.get("resolved_rate", row["pass_at_1"]))
            * 100
            * BENCHMARK_WEIGHTS[row["name"]]
            for row in rows
        )
        floor = (
            MIN_QUALIFYING_RESOLVED_RATE
            if names == REQUIRED_BENCHMARKS
            else MIN_QUALIFYING_PASS_AT_1
        )
        if score < floor * 100:
            raise ValueError(
                f"{model_id} scored {score:.1f}, below the {floor * 100:.1f} qualification floor"
            )
        certifications.append(
            {
                "provider": provider,
                "upstream_model_id": model_id,
                "aliases": [],
                "tier": "verified",
                "quality_score": round(score, 4),
                "certified_at": now.isoformat(),
                "compatibility_canary_version": COMPATIBILITY_CANARY_VERSION,
                "benchmarks": sorted(rows, key=lambda row: row["name"]),
            }
        )

    priors: list[dict[str, Any]] = []
    for path in args.priors:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"prior file must contain an object: {path}")
        priors.append(value)
    provisionals: list[dict[str, Any]] = []
    for path in getattr(args, "provisionals", []):
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"provisional file must contain an object: {path}")
        provisionals.append(value)
    manifest = {
        "schema_version": 2,
        "manifest_id": args.manifest_id or now.strftime("%Y-%m-%dT%H%M%SZ"),
        "policy_version": POLICY_VERSION,
        "certifications": certifications,
        "provisionals": provisionals,
        "priors": priors,
    }
    return validate_manifest(manifest, require_complete_policy=True)


def prior(args: argparse.Namespace) -> dict[str, Any]:
    parsed = urlparse(args.source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("--source-url must use HTTPS")
    score = args.score
    task_count = None
    if args.input is not None:
        payload = read_json(args.input)
        if isinstance(payload, list):
            payload = {"results": payload}
        if not isinstance(payload, dict):
            raise ValueError("public result must be an object or result array")
        task_count, pass_at_1 = _pass_summary(payload)
        score = pass_at_1 * 100
    if score is None or not 0 <= score <= 100:
        raise ValueError("--score must be between 0 and 100")
    commit = args.harness_commit.lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("--harness-commit must be a full 40-character hexadecimal git commit")
    row = {
        "source_url": args.source_url,
        "benchmark": args.benchmark,
        "observed_at": args.observed_at or datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "upstream_model_id": args.model,
        "score": round(score, 8),
        "suite_version": args.suite_version,
        "harness_commit": commit,
        "evidence_kind": "prior",
    }
    if task_count is not None:
        row["task_count"] = task_count
        row["source_result_fingerprint"] = "sha256:" + hashlib.sha256(args.input.read_bytes()).hexdigest()
    return row


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    normalize = commands.add_parser("normalize", help="Normalize official harness JSON")
    normalize.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(REQUIRED_BENCHMARKS | LEGACY_VERIFIED_BENCHMARKS),
    )
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--provider", required=True)
    normalize.add_argument("--model", required=True)
    normalize.add_argument("--suite-version", required=True)
    normalize.add_argument("--harness-repository", required=True)
    normalize.add_argument("--harness-commit", required=True)
    normalize.add_argument("--settings", type=Path, required=True)
    normalize.add_argument("--run-record", type=Path, required=True)
    normalize.add_argument("--run-mode", choices=("calibration", "certification"), required=True)

    build = commands.add_parser("build", help="Build the bundled qualification manifest")
    build.add_argument("--results", type=Path, nargs="*", default=[])
    build.add_argument("--priors", type=Path, nargs="*", default=[])
    build.add_argument("--provisionals", type=Path, nargs="*", default=[])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest-id")

    add_prior = commands.add_parser("prior", help="Create a provenance-only public benchmark prior")
    add_prior.add_argument("--source-url", required=True)
    add_prior.add_argument("--benchmark", required=True)
    add_prior.add_argument("--observed-at")
    add_prior.add_argument("--provider", required=True)
    add_prior.add_argument("--model", required=True)
    prior_score = add_prior.add_mutually_exclusive_group(required=True)
    prior_score.add_argument("--score", type=float)
    prior_score.add_argument("--input", type=Path, help="Official public result JSON; score is derived")
    add_prior.add_argument("--suite-version", required=True)
    add_prior.add_argument("--harness-commit", required=True)
    add_prior.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "normalize":
            write_json(args.output, normalize_result(args))
        elif args.command == "build":
            write_json(args.output, build_manifest(args))
        elif args.command == "prior":
            write_json(args.output, prior(args))
        return 0
    except (CodingBenchmarkPolicyError, OSError, ValueError) as exc:
        sys.stderr.write(f"coding-certification: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
