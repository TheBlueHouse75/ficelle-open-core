"""Bundled coding qualifications for the ``ficelle/auto-coding`` routing lane.

Ficelle-verified benchmark results and explicitly reviewed provisional evidence are separate
tiers. Provider availability and compatibility remain deployment-specific runtime gates.
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
    MIN_QUALIFYING_RESOLVED_RATE,
    CodingBenchmarkPolicyError,
    validate_evidence_metadata,
    validate_model_identity,
)


CODING_PROFILE_ID = "ficelle/auto-coding"
MANIFEST_SCHEMA_VERSION = 2
BUILTIN_MANIFEST_PATH = Path(__file__).with_name("assets") / "auto-coding-manifest.json"
REQUIRED_BENCHMARKS = frozenset({"aider-practical"})
LEGACY_VERIFIED_BENCHMARKS = frozenset({"aider-polyglot"})
POLICY_VERSION = "coding-v3"
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


def _model_ids(row: Mapping[str, Any], provider: str) -> tuple[str, ...]:
    primary = _required_text(row, "upstream_model_id", limit=240)
    raw_aliases = row.get("aliases", [])
    if not isinstance(raw_aliases, list) or any(not isinstance(alias, str) for alias in raw_aliases):
        raise CodingCertificationError("aliases must be an array of model ids")
    aliases = [alias.strip() for alias in raw_aliases]
    if any(not alias or len(alias) > 240 for alias in aliases):
        raise CodingCertificationError("aliases must contain non-empty model ids")
    values = [primary, *aliases]
    if len(set(values)) != len(values):
        raise CodingCertificationError("qualification model ids must be unique")
    try:
        return tuple(validate_model_identity(provider, value)[1] for value in values)
    except CodingBenchmarkPolicyError as exc:
        raise CodingCertificationError(f"qualification identity is outside pinned policy: {exc}") from exc


def _benchmark_quality_rate(row: Mapping[str, Any]) -> float:
    value = row.get("resolved_rate", row.get("pass_at_1"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodingCertificationError("benchmark must expose pass_at_1 or resolved_rate")
    rate = float(value)
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise CodingCertificationError("benchmark quality rate must be between 0 and 1")
    return rate


def _validate_provisional_evidence(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise CodingCertificationError("provisional evidence rows must be objects")
    source_url = _required_text(row, "source_url", limit=500)
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise CodingCertificationError("provisional evidence source_url must use HTTPS")
    _required_text(row, "benchmark", limit=120)
    _parse_timestamp(row.get("observed_at"), "observed_at")
    provenance = _required_text(row, "provenance", limit=40)
    if provenance not in {"vendor", "independent"}:
        raise CodingCertificationError("provisional evidence provenance is unsupported")
    _bounded_score(row, "score")
    return dict(row)


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
    raw_provisionals = manifest.get("provisionals")
    raw_priors = manifest.get("priors")
    if (
        not isinstance(raw_certifications, list)
        or not isinstance(raw_provisionals, list)
        or not isinstance(raw_priors, list)
    ):
        raise CodingCertificationError("certifications, provisionals and priors must be arrays")
    identities: set[str] = set()
    certifications: list[dict[str, Any]] = []
    for raw in raw_certifications:
        if not isinstance(raw, dict):
            raise CodingCertificationError("certification rows must be objects")
        provider = _required_text(raw, "provider", limit=80).lower()
        model_ids = _model_ids(raw, provider)
        upstream_model_id = model_ids[0]
        # Quality belongs to the model, not to the route used to measure it. ``provider`` is kept
        # as provenance; per-provider compatibility remains a separate local canary gate.
        if any(identity in identities for identity in model_ids):
            raise CodingCertificationError("duplicate model certification")
        identities.update(model_ids)
        if raw.get("tier") != "verified":
            raise CodingCertificationError("certification tier must be verified")
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
        accepted_sets = {REQUIRED_BENCHMARKS, LEGACY_VERIFIED_BENCHMARKS}
        if require_complete_policy and benchmark_names not in accepted_sets:
            raise CodingCertificationError("certification does not contain an accepted complete benchmark set")
        measured_score = round(
            sum(_benchmark_quality_rate(item) for item in benchmarks) * 100 / len(benchmarks),
            4,
        )
        floor = (
            MIN_QUALIFYING_RESOLVED_RATE
            if benchmark_names == REQUIRED_BENCHMARKS
            else MIN_QUALIFYING_PASS_AT_1
        )
        if measured_score < floor * 100:
            raise CodingCertificationError("certification score is below the qualification floor")
        if not math.isclose(declared_score, measured_score, abs_tol=0.0001):
            raise CodingCertificationError("quality_score does not match benchmark evidence")
        row = dict(raw)
        row.update(
            {
                "provider": provider,
                "upstream_model_id": upstream_model_id,
                "aliases": list(model_ids[1:]),
                "benchmarks": benchmarks,
            }
        )
        certifications.append(row)
    provisionals: list[dict[str, Any]] = []
    for raw in raw_provisionals:
        if not isinstance(raw, dict):
            raise CodingCertificationError("provisional rows must be objects")
        provider = _required_text(raw, "provider", limit=80).lower()
        model_ids = _model_ids(raw, provider)
        if any(identity in identities for identity in model_ids):
            raise CodingCertificationError("duplicate model qualification")
        identities.update(model_ids)
        if raw.get("tier") != "provisional":
            raise CodingCertificationError("provisional tier must be provisional")
        _parse_timestamp(raw.get("admitted_at"), "admitted_at")
        evidence = raw.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise CodingCertificationError("provisional qualification requires evidence")
        row = dict(raw)
        row.update(
            {
                "provider": provider,
                "upstream_model_id": model_ids[0],
                "aliases": list(model_ids[1:]),
                "evidence": [_validate_provisional_evidence(item) for item in evidence],
            }
        )
        provisionals.append(row)
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
    result["provisionals"] = provisionals
    return result


def certification_identity(model: Mapping[str, Any]) -> str:
    return str(model.get("upstream_id") or "")


def certification_index(manifest: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        return {}
    certifications = manifest.get("certifications")
    provisionals = manifest.get("provisionals")
    rows = [
        *(certifications if isinstance(certifications, list) else []),
        *(provisionals if isinstance(provisionals, list) else []),
    ]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        for model_id in [row.get("upstream_model_id"), *(row.get("aliases") or [])]:
            if isinstance(model_id, str) and model_id:
                result[model_id] = normalized
    return result


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
        "provisional_count": len(manifest.get("provisionals") or []) if manifest else 0,
        "qualification_count": (
            len(manifest.get("certifications") or []) + len(manifest.get("provisionals") or [])
            if manifest
            else 0
        ),
        "prior_count": len(manifest.get("priors") or []) if manifest else 0,
        "message": "" if manifest else "bundled coding qualifications unavailable",
    }


def quality_score(model: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> float:
    row = certification_for_model(model, manifest)
    return float(row.get("quality_score") or 0.0) if row else 0.0
