#!/usr/bin/env python3
"""Build, sign, and verify Ficelle's central coding-certification manifest.

This tool consumes machine-readable output from official benchmark harnesses. It does not run or
reimplement their tasks; ``coding-benchmark-runner.py`` pins and executes those upstream harnesses.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from ficelle.coding_certification import (  # noqa: E402
    COMPATIBILITY_CANARY_VERSION,
    KEY_ID,
    POLICY_VERSION,
    REQUIRED_BENCHMARKS,
    canonical_json,
    validate_manifest,
    verify_envelope,
)


BENCHMARK_WEIGHTS = {
    "aider-polyglot": 0.35,
    "swe-rebench": 0.40,
    "terminal-bench-2.1": 0.25,
}
KEYCHAIN_SERVICE = "ai.ficelle.coding-certification"
KEYCHAIN_ACCOUNT = "manifest-signing-key-v1"
PRIVATE_KEY_ENV = "FICELLE_CODING_CERT_PRIVATE_KEY_B64"


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
    if (
        isinstance(task_count, int)
        and not isinstance(task_count, bool)
        and task_count > 0
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
    ):
        rate = float(score)
        if rate > 1:
            rate /= 100.0
        return task_count, rate

    rows = payload.get("results") or payload.get("instances") or payload.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("official result must expose task_count/pass_at_1 or a non-empty result list")
    passed = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("result rows must be objects")
        verdict = row.get("passed")
        if verdict is None:
            verdict = row.get("resolved")
        if verdict is None:
            verdict = str(row.get("status") or "").lower() in {"pass", "passed", "resolved", "success"}
        passed += int(bool(verdict))
    return len(rows), passed / len(rows)


def normalize_result(args: argparse.Namespace) -> dict[str, Any]:
    if args.benchmark not in REQUIRED_BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {args.benchmark}")
    payload = read_json(args.input)
    if isinstance(payload, list):
        payload = {"results": payload}
    if not isinstance(payload, dict):
        raise ValueError("official benchmark result must be an object or result array")
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
    settings_provider = str(settings.get("provider") or "").strip().lower()
    settings_model = str(settings.get("upstream_model_id") or "").strip()
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
        "harness_repository": args.harness_repository,
        "harness_commit": commit,
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
        "suite_version": args.suite_version,
        "harness_repository": args.harness_repository,
        "harness_commit": commit,
        "task_count": task_count,
        "pass_at_1": round(pass_at_1, 8),
        "settings_fingerprint": settings_fingerprint,
        "run_record_fingerprint": "sha256:" + hashlib.sha256(run_record_bytes).hexdigest(),
        "official_result_fingerprint": official_result_fingerprint,
        "command_fingerprint": command_fingerprint,
        "evidence_kind": "central_run",
        "observed_at": datetime.now(UTC).isoformat(),
    }
    languages = payload.get("languages") or payload.get("language_breakdown")
    if isinstance(languages, dict):
        row["languages"] = languages
    return row


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in args.results:
        row = read_json(path)
        if not isinstance(row, dict):
            raise ValueError(f"normalized result must be an object: {path}")
        identity = (str(row.get("provider") or "").lower(), str(row.get("upstream_model_id") or ""))
        if not all(identity):
            raise ValueError(f"normalized result is missing provider/model identity: {path}")
        grouped.setdefault(identity, []).append(row)

    certifications = []
    now = datetime.now(UTC)
    for (provider, model_id), rows in sorted(grouped.items()):
        names = {str(row.get("name") or "") for row in rows}
        if names != REQUIRED_BENCHMARKS or len(rows) != len(REQUIRED_BENCHMARKS):
            raise ValueError(f"{provider}/{model_id} does not have the exact required benchmark set")
        score = sum(float(row["pass_at_1"]) * 100 * BENCHMARK_WEIGHTS[row["name"]] for row in rows)
        certifications.append(
            {
                "provider": provider,
                "upstream_model_id": model_id,
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
    manifest = {
        "schema_version": 1,
        "manifest_id": args.manifest_id or now.strftime("%Y-%m-%dT%H%M%SZ"),
        "generated_at": now.isoformat(),
        "expires_at": (now + timedelta(days=args.expires_days)).isoformat(),
        "policy_version": POLICY_VERSION,
        "certifications": certifications,
        "priors": priors,
    }
    return validate_manifest(manifest, require_complete_policy=True)


def private_key_bytes() -> bytes:
    raw = os.getenv(PRIVATE_KEY_ENV, "").strip()
    if not raw and sys.platform == "darwin":
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            raw = result.stdout.strip()
    if not raw:
        raise ValueError(
            f"signing key unavailable; set {PRIVATE_KEY_ENV} or install {KEYCHAIN_SERVICE}/{KEYCHAIN_ACCOUNT}"
        )
    try:
        key = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise ValueError("signing key is not valid base64") from exc
    if len(key) != 32:
        raise ValueError("signing key must contain exactly 32 raw Ed25519 bytes")
    return key


def sign_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(payload, require_complete_policy=True)
    signature = Ed25519PrivateKey.from_private_bytes(private_key_bytes()).sign(canonical_json(payload))
    envelope = {
        "key_id": KEY_ID,
        "manifest": payload,
        "signature": base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii"),
    }
    verify_envelope(envelope)
    return envelope


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
    normalize.add_argument("--benchmark", required=True, choices=sorted(REQUIRED_BENCHMARKS))
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--provider", required=True)
    normalize.add_argument("--model", required=True)
    normalize.add_argument("--suite-version", required=True)
    normalize.add_argument("--harness-repository", required=True)
    normalize.add_argument("--harness-commit", required=True)
    normalize.add_argument("--settings", type=Path, required=True)
    normalize.add_argument("--run-record", type=Path, required=True)

    build = commands.add_parser("build", help="Build a policy-complete unsigned manifest")
    build.add_argument("--results", type=Path, nargs="*", default=[])
    build.add_argument("--priors", type=Path, nargs="*", default=[])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest-id")
    build.add_argument("--expires-days", type=int, default=30, choices=range(1, 91))

    sign = commands.add_parser("sign", help="Sign an unsigned manifest")
    sign.add_argument("--manifest", type=Path, required=True)
    sign.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify", help="Verify a signed envelope")
    verify.add_argument("--envelope", type=Path, required=True)

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
        elif args.command == "sign":
            payload = read_json(args.manifest)
            if not isinstance(payload, dict):
                raise ValueError("unsigned manifest must be an object")
            write_json(args.output, sign_manifest(payload))
        elif args.command == "verify":
            verify_envelope(read_json(args.envelope))
        elif args.command == "prior":
            write_json(args.output, prior(args))
        return 0
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"coding-certification: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
