"""Bundled coding qualifications for the ``ficelle/auto-coding`` routing lane.

Public benchmark results are deliberately represented as priors in the manifest but are never
indexed by the route helpers below. Only an exact bundled Ficelle qualification can make a model
eligible; provider availability and compatibility remain separate runtime gates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ficelle.coding_benchmark_policy import (
    MIN_QUALIFYING_PASS_AT_1,
    CodingBenchmarkPolicyError,
    validate_evidence_metadata,
    validate_model_identity,
)


CODING_PROFILE_ID = "ficelle/auto-coding"
MANIFEST_SCHEMA_VERSION = 1
BUILTIN_MANIFEST_PATH = Path(__file__).with_name("assets") / "auto-coding-manifest.json"
REQUIRED_BENCHMARKS = frozenset({"aider-polyglot"})
POLICY_VERSION = "coding-v2"
COMPATIBILITY_CANARY_VERSION = "coding-compatibility-v1"


class CodingCertificationError(ValueError):
    """A manifest could not be trusted or used."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CodingCertificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes | str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CodingCertificationError(f"non-finite JSON number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CodingCertificationError("certification manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CodingCertificationError("certification manifest must be an object")
    return value


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CodingCertificationError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CodingCertificationError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CodingCertificationError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _required_text(row: Mapping[str, Any], field: str, *, limit: int = 300) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise CodingCertificationError(f"{field} must be a non-empty string")
    return value.strip()


def _bounded_score(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodingCertificationError(f"{field} must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise CodingCertificationError(f"{field} must be between 0 and 100")
    return score


def _validate_benchmark(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise CodingCertificationError("benchmark rows must be objects")
    name = _required_text(row, "name", limit=80)
    _required_text(row, "suite_version", limit=120)
    harness_repository = _required_text(row, "harness_repository", limit=300)
    parsed_repository = urlparse(harness_repository)
    if parsed_repository.scheme != "https" or not parsed_repository.netloc:
        raise CodingCertificationError("harness_repository must use HTTPS")
    commit = _required_text(row, "harness_commit", limit=64)
    if not all(character in "0123456789abcdefABCDEF" for character in commit) or len(commit) != 40:
        raise CodingCertificationError("harness_commit must be a full 40-character git commit")
    task_count = row.get("task_count")
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count <= 0:
        raise CodingCertificationError("task_count must be a positive integer")
    pass_at_1 = row.get("pass_at_1")
    if isinstance(pass_at_1, bool) or not isinstance(pass_at_1, (int, float)):
        raise CodingCertificationError("pass_at_1 must be a number")
    if not math.isfinite(float(pass_at_1)) or not 0 <= float(pass_at_1) <= 1:
        raise CodingCertificationError("pass_at_1 must be between 0 and 1")
    fingerprint = _required_text(row, "settings_fingerprint", limit=128)
    for field, value in (
        ("settings_fingerprint", fingerprint),
        ("run_record_fingerprint", _required_text(row, "run_record_fingerprint", limit=128)),
        ("official_result_fingerprint", _required_text(row, "official_result_fingerprint", limit=128)),
        ("command_fingerprint", _required_text(row, "command_fingerprint", limit=128)),
    ):
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise CodingCertificationError(f"{field} must be a SHA-256 digest")
    run_mode = str(row.get("run_mode") or "")
    expected_evidence_kind = "central_run" if run_mode == "certification" else "calibration_run"
    if row.get("evidence_kind") != expected_evidence_kind:
        raise CodingCertificationError("benchmark evidence_kind does not match run_mode")
    try:
        validate_evidence_metadata({**row, "benchmark": name}, run_mode=run_mode)
    except CodingBenchmarkPolicyError as exc:
        raise CodingCertificationError(f"benchmark evidence is outside pinned policy: {exc}") from exc
    result = dict(row)
    result["name"] = name
    return result


def validate_manifest(
    manifest: Any,
    *,
    require_complete_policy: bool = False,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise CodingCertificationError("manifest must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CodingCertificationError("unsupported certification manifest schema")
    _required_text(manifest, "manifest_id", limit=120)
    if _required_text(manifest, "policy_version", limit=80) != POLICY_VERSION:
        raise CodingCertificationError("unsupported coding certification policy")

    raw_certifications = manifest.get("certifications")
    raw_priors = manifest.get("priors")
    if not isinstance(raw_certifications, list) or not isinstance(raw_priors, list):
        raise CodingCertificationError("certifications and priors must be arrays")
    identities: set[str] = set()
    certifications: list[dict[str, Any]] = []
    for raw in raw_certifications:
        if not isinstance(raw, dict):
            raise CodingCertificationError("certification rows must be objects")
        try:
            provider, upstream_model_id = validate_model_identity(
                _required_text(raw, "provider", limit=80),
                _required_text(raw, "upstream_model_id", limit=240),
            )
        except CodingBenchmarkPolicyError as exc:
            raise CodingCertificationError(f"certification identity is outside pinned policy: {exc}") from exc
        # Quality belongs to the model, not to the route used to measure it. ``provider`` is kept
        # as provenance; per-provider compatibility remains a separate local canary gate.
        identity = upstream_model_id
        if identity in identities:
            raise CodingCertificationError("duplicate model certification")
        identities.add(identity)
        declared_score = _bounded_score(raw, "quality_score")
        _parse_timestamp(raw.get("certified_at"), "certified_at")
        if _required_text(raw, "compatibility_canary_version", limit=80) != COMPATIBILITY_CANARY_VERSION:
            raise CodingCertificationError("unsupported compatibility canary version")
        raw_benchmarks = raw.get("benchmarks")
        if not isinstance(raw_benchmarks, list):
            raise CodingCertificationError("benchmarks must be an array")
        benchmarks = [_validate_benchmark(item) for item in raw_benchmarks]
        if not benchmarks:
            raise CodingCertificationError("certification must contain benchmark evidence")
        benchmark_names = {item["name"] for item in benchmarks}
        if len(benchmark_names) != len(benchmarks):
            raise CodingCertificationError("duplicate benchmark in certification")
        if require_complete_policy and benchmark_names != REQUIRED_BENCHMARKS:
            raise CodingCertificationError("certification does not contain the exact required benchmark set")
        measured_score = round(
            sum(float(item["pass_at_1"]) for item in benchmarks) * 100 / len(benchmarks),
            4,
        )
        if measured_score < MIN_QUALIFYING_PASS_AT_1 * 100:
            raise CodingCertificationError("certification score is below the qualification floor")
        if not math.isclose(declared_score, measured_score, abs_tol=0.0001):
            raise CodingCertificationError("quality_score does not match benchmark evidence")
        row = dict(raw)
        row.update({"provider": provider, "upstream_model_id": upstream_model_id, "benchmarks": benchmarks})
        certifications.append(row)
    for prior in raw_priors:
        if not isinstance(prior, dict):
            raise CodingCertificationError("prior rows must be objects")
        source_url = _required_text(prior, "source_url", limit=500)
        parsed_source = urlparse(source_url)
        if parsed_source.scheme != "https" or not parsed_source.netloc:
            raise CodingCertificationError("prior source_url must use HTTPS")
        _required_text(prior, "benchmark", limit=80)
        _parse_timestamp(_required_text(prior, "observed_at", limit=80), "observed_at")
        _required_text(prior, "provider", limit=80)
        _required_text(prior, "upstream_model_id", limit=240)
        _required_text(prior, "suite_version", limit=120)
        prior_commit = _required_text(prior, "harness_commit", limit=40)
        if len(prior_commit) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in prior_commit
        ):
            raise CodingCertificationError("prior harness_commit must be a full git commit")
        _bounded_score(prior, "score")
        if prior.get("evidence_kind") != "prior":
            raise CodingCertificationError("public benchmark evidence_kind must be prior")
    result = dict(manifest)
    result["certifications"] = certifications
    return result


def certification_identity(model: Mapping[str, Any]) -> str:
    return str(model.get("upstream_id") or "")


def certification_index(manifest: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        return {}
    rows = manifest.get("certifications")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("upstream_model_id") or ""): dict(row)
        for row in rows
        if isinstance(row, dict)
    }


def certification_for_model(model: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return certification_index(manifest).get(certification_identity(model))


def cached_manifest() -> dict[str, Any] | None:
    try:
        return validate_manifest(
            strict_json_loads(BUILTIN_MANIFEST_PATH.read_bytes()),
            require_complete_policy=True,
        )
    except (CodingCertificationError, OSError):
        return None


def public_status() -> dict[str, Any]:
    manifest = cached_manifest()
    return {
        "status": "bundled" if manifest is not None else "unavailable",
        "manifest_id": manifest.get("manifest_id") if manifest else None,
        "certification_count": len(manifest.get("certifications") or []) if manifest else 0,
        "prior_count": len(manifest.get("priors") or []) if manifest else 0,
        "message": "" if manifest else "bundled coding qualifications unavailable",
    }


def quality_score(model: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> float:
    row = certification_for_model(model, manifest)
    return float(row.get("quality_score") or 0.0) if row else 0.0
