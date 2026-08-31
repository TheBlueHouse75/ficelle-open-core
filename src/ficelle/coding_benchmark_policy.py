"""Pinned policy for Ficelle's lightweight coding-model qualification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from urllib.parse import urlparse


class CodingBenchmarkPolicyError(ValueError):
    """Raised when benchmark evidence is outside the pinned Ficelle policy."""


@dataclass(frozen=True)
class SourceRevision:
    name: str
    repository: str
    commit: str


@dataclass(frozen=True)
class BenchmarkPolicy:
    name: str
    suite_version: str
    harness_repository: str
    harness_commit: str
    reference_agent: str
    reference_agent_commit: str
    calibration_task_count: int
    calibration_attempts_per_task: int
    calibration_wall_clock_seconds: int
    completion_token_budget: int
    certification_task_count: int
    certification_attempts_per_task: int
    sources: tuple[SourceRevision, ...]


MIN_QUALIFYING_PASS_AT_1 = 0.6

AIDER_CALIBRATION_TASKS = (
    "cpp/exercises/practice/circular-buffer",
    "go/exercises/practice/crypto-square",
    "java/exercises/practice/bank-account",
    "python/exercises/practice/affine-cipher",
    "rust/exercises/practice/doubly-linked-list",
)

BENCHMARK_POLICIES: dict[str, BenchmarkPolicy] = {
    "aider-polyglot": BenchmarkPolicy(
        name="aider-polyglot",
        suite_version="polyglot-2026-08-21",
        harness_repository="https://github.com/Aider-AI/aider",
        harness_commit="5dc9490bb35f9729ef2c95d00a19ccd30c26339c",
        reference_agent="aider",
        reference_agent_commit="5dc9490bb35f9729ef2c95d00a19ccd30c26339c",
        calibration_task_count=5,
        calibration_attempts_per_task=1,
        calibration_wall_clock_seconds=1800,
        completion_token_budget=12_000,
        # Ficelle guides routing; it does not reproduce a research leaderboard. The same frozen
        # five-language sample is the complete qualification gate for every model.
        certification_task_count=5,
        certification_attempts_per_task=1,
        sources=(
            SourceRevision(
                name="polyglot-benchmark",
                repository="https://github.com/Aider-AI/polyglot-benchmark",
                commit="7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f",
            ),
        ),
    ),
}

_MUTABLE_MODEL_PATTERNS = (
    re.compile(r"(^|/)stealth/", re.IGNORECASE),
    re.compile(r"(^|[-/:])latest(?:$|[-/:])", re.IGNORECASE),
    re.compile(r"(^|[-/:])preview(?:$|[-/:])", re.IGNORECASE),
    re.compile(r"(^|/)(?:free|auto)$", re.IGNORECASE),
)

# Ox Alpha is published under this exact upstream id by several providers. The ``stealth``
# namespace is otherwise opaque and remains rejected: this narrow exception records the public
# model name without allowing a provider's arbitrary ``stealth/*`` alias into certification.
_PINNED_OPAQUE_MODEL_IDS = frozenset({"stealth/ox-alpha"})


def canonical_repository(value: str) -> str:
    repository = value.strip().rstrip("/")
    if repository.endswith(".git"):
        repository = repository[:-4]
    parsed = urlparse(repository)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise CodingBenchmarkPolicyError("repository must be a canonical HTTPS URL")
    return repository


def validate_commit(value: str, field: str = "commit") -> str:
    commit = value.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise CodingBenchmarkPolicyError(f"{field} must be a full 40-character git commit")
    return commit


def validate_model_identity(provider: str, upstream_model_id: str) -> tuple[str, str]:
    normalized_provider = provider.strip().lower()
    model_id = upstream_model_id.strip()
    if not normalized_provider or not model_id:
        raise CodingBenchmarkPolicyError("provider and upstream_model_id are required")
    if model_id not in _PINNED_OPAQUE_MODEL_IDS and any(
        pattern.search(model_id) for pattern in _MUTABLE_MODEL_PATTERNS
    ):
        raise CodingBenchmarkPolicyError("mutable or opaque model ids cannot enter coding certification")
    return normalized_provider, model_id


def policy_payload() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "aider_calibration_tasks": AIDER_CALIBRATION_TASKS,
        "benchmarks": {
            name: asdict(policy) for name, policy in sorted(BENCHMARK_POLICIES.items())
        },
    }


def policy_fingerprint() -> str:
    encoded = json.dumps(
        policy_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def aider_model_settings(policy: BenchmarkPolicy) -> tuple[str, str]:
    """Return the canonical Aider settings and their content fingerprint."""
    model_settings = (
        "- name: aider/extra_params\n"
        "  extra_params:\n"
        f"    max_tokens: {policy.completion_token_budget}\n"
    )
    fingerprint = "sha256:" + hashlib.sha256(model_settings.encode("utf-8")).hexdigest()
    return model_settings, fingerprint


def validate_evidence_metadata(payload: Mapping[str, Any], *, run_mode: str) -> BenchmarkPolicy:
    benchmark = str(payload.get("benchmark") or "")
    policy = BENCHMARK_POLICIES.get(benchmark)
    if policy is None:
        raise CodingBenchmarkPolicyError("benchmark is not in the pinned coding policy")
    if run_mode not in {"calibration", "certification"}:
        raise CodingBenchmarkPolicyError("run_mode must be calibration or certification")
    if payload.get("run_mode") != run_mode:
        raise CodingBenchmarkPolicyError("result run_mode does not match the requested mode")
    _, expected_model_settings_fingerprint = aider_model_settings(policy)
    expected = {
        "suite_version": policy.suite_version,
        "harness_repository": canonical_repository(policy.harness_repository),
        "harness_commit": policy.harness_commit,
        "reference_agent": policy.reference_agent,
        "reference_agent_commit": policy.reference_agent_commit,
        "completion_token_budget": policy.completion_token_budget,
        "model_settings_fingerprint": expected_model_settings_fingerprint,
    }
    actual = {
        "suite_version": payload.get("suite_version"),
        "harness_repository": canonical_repository(str(payload.get("harness_repository") or "")),
        "harness_commit": validate_commit(str(payload.get("harness_commit") or ""), "harness_commit"),
        "reference_agent": payload.get("reference_agent"),
        "reference_agent_commit": validate_commit(
            str(payload.get("reference_agent_commit") or ""), "reference_agent_commit"
        ),
        "completion_token_budget": payload.get("completion_token_budget"),
        "model_settings_fingerprint": payload.get("model_settings_fingerprint"),
    }
    mismatched = [key for key, expected_value in expected.items() if actual[key] != expected_value]
    if mismatched:
        raise CodingBenchmarkPolicyError(
            "result does not match pinned policy: " + ", ".join(mismatched)
        )
    fingerprint = str(payload.get("policy_fingerprint") or "")
    digest = fingerprint.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CodingBenchmarkPolicyError("policy_fingerprint must be a SHA-256 digest")
    if fingerprint != policy_fingerprint():
        raise CodingBenchmarkPolicyError("result does not match the current coding policy fingerprint")

    source_rows = payload.get("source_revisions")
    if not isinstance(source_rows, list):
        raise CodingBenchmarkPolicyError("source_revisions must be an array")
    normalized_sources = {
        (
            str(row.get("name") or ""),
            canonical_repository(str(row.get("repository") or "")),
            validate_commit(str(row.get("commit") or ""), "source commit"),
        )
        for row in source_rows
        if isinstance(row, Mapping)
    }
    expected_sources = {
        (source.name, canonical_repository(source.repository), source.commit)
        for source in policy.sources
    }
    if len(normalized_sources) != len(source_rows) or normalized_sources != expected_sources:
        raise CodingBenchmarkPolicyError("source revisions do not match pinned policy")

    task_count = payload.get("task_count")
    attempts = payload.get("attempts_per_task")
    expected_task_count = (
        policy.calibration_task_count
        if run_mode == "calibration"
        else policy.certification_task_count
    )
    expected_attempts = (
        policy.calibration_attempts_per_task
        if run_mode == "calibration"
        else policy.certification_attempts_per_task
    )
    if task_count != expected_task_count or attempts != expected_attempts:
        raise CodingBenchmarkPolicyError("task count or attempts do not match pinned policy")
    completed_count = payload.get("completed_count")
    if (
        isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or not 0 <= completed_count <= task_count
    ):
        raise CodingBenchmarkPolicyError("completed_count must be between zero and task_count")
    harness_exit_code = payload.get("harness_exit_code")
    if isinstance(harness_exit_code, bool) or not isinstance(harness_exit_code, int):
        raise CodingBenchmarkPolicyError("harness_exit_code must be an integer")
    if not isinstance(payload.get("timed_out"), bool):
        raise CodingBenchmarkPolicyError("timed_out must be a boolean")
    if (
        payload.get("wall_clock_timeout_seconds") != policy.calibration_wall_clock_seconds
    ):
        raise CodingBenchmarkPolicyError("wall-clock timeout does not match pinned policy")
    counts: dict[str, int] = {}
    for field in ("model_verdict_count", "provider_error_count", "harness_error_count"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CodingBenchmarkPolicyError(f"{field} must be a non-negative integer")
        counts[field] = value
    if sum(counts.values()) != task_count:
        raise CodingBenchmarkPolicyError("model/provider/harness counts must cover every task")
    if completed_count != counts["model_verdict_count"]:
        raise CodingBenchmarkPolicyError("completed_count must equal model_verdict_count")
    if run_mode == "certification":
        if counts["model_verdict_count"] != task_count:
            raise CodingBenchmarkPolicyError("qualification evidence must contain a model verdict for every task")
    return policy
