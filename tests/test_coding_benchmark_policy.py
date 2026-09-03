from __future__ import annotations

import pytest

from ficelle import coding_benchmark_policy as policy


def evidence(benchmark: str = "aider-polyglot") -> dict[str, object]:
    spec = policy.BENCHMARK_POLICIES[benchmark]
    return {
        "benchmark": benchmark,
        "run_mode": "calibration",
        "suite_version": spec.suite_version,
        "harness_repository": spec.harness_repository,
        "harness_commit": spec.harness_commit,
        "reference_agent": spec.reference_agent,
        "reference_agent_commit": spec.reference_agent_commit,
        "source_revisions": [
            {"name": item.name, "repository": item.repository, "commit": item.commit}
            for item in spec.sources
        ],
        "policy_fingerprint": policy.policy_fingerprint(),
        "completion_token_budget": spec.completion_token_budget,
        "relay_timeout_seconds": spec.relay_timeout_seconds,
        "model_settings_fingerprint": policy.aider_model_settings(spec)[1],
        "task_count": spec.calibration_task_count,
        "attempts_per_task": spec.calibration_attempts_per_task,
        "wall_clock_timeout_seconds": spec.calibration_wall_clock_seconds,
        "completed_count": spec.calibration_task_count,
        "model_verdict_count": spec.calibration_task_count,
        "provider_error_count": 0,
        "harness_error_count": 0,
        "harness_exit_code": 0,
        "timed_out": False,
    }


def test_policy_pins_harness_sources_and_calibration_shape():
    row = evidence()

    spec = policy.validate_evidence_metadata(row, run_mode="calibration")

    assert spec.harness_commit == "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
    assert spec.sources[0].commit == "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
    assert spec.completion_token_budget == 12_000
    assert spec.relay_timeout_seconds == 900
    assert policy.policy_fingerprint().startswith("sha256:")


def test_practical_policy_is_a_three_task_repair_gate():
    spec = policy.BENCHMARK_POLICIES["aider-practical"]

    assert policy.AIDER_PRACTICAL_TASKS == (
        "go/exercises/practice/crypto-square",
        "java/exercises/practice/bank-account",
        "python/exercises/practice/affine-cipher",
    )
    assert spec.certification_task_count == 3
    assert spec.certification_attempts_per_task == 2
    assert policy.MIN_QUALIFYING_RESOLVED_RATE == pytest.approx(2 / 3)
    assert policy.EXTENDED_DIAGNOSTIC_TOKEN_BUDGET == 24_000


def test_policy_accepts_extended_diagnostic_without_changing_the_standard_verdict():
    row = evidence("aider-practical")
    task_ids = policy.AIDER_PRACTICAL_TASKS
    row["results"] = [
        {"task_id": task_ids[0], "status": "passed", "passed": True},
        {"task_id": task_ids[1], "status": "passed", "passed": True},
        {
            "task_id": task_ids[2],
            "status": "failed",
            "passed": False,
            "diagnostic": "truncated_before_content",
        },
    ]
    _, settings_fingerprint = policy.aider_model_settings(
        policy.BENCHMARK_POLICIES["aider-practical"],
        completion_token_budget=24_000,
    )
    row["extended_token_diagnostic"] = {
        "trigger_reason": "truncated_before_content",
        "completion_token_budget": 24_000,
        "task_count": 1,
        "completed_count": 1,
        "model_verdict_count": 1,
        "provider_error_count": 0,
        "harness_error_count": 0,
        "attempts_per_task": 2,
        "model_settings_fingerprint": settings_fingerprint,
        "harness_exit_code": 0,
        "timed_out": False,
        "wall_clock_timeout_seconds": 2700,
        "classification": "capable_but_output_hungry",
        "results": [
            {"task_id": task_ids[2], "status": "passed", "passed": True}
        ],
    }

    validated = policy.validate_evidence_metadata(row, run_mode="calibration")

    assert validated.completion_token_budget == 12_000


def test_policy_rejects_extended_diagnostic_when_truncation_is_not_the_only_problem():
    row = evidence("aider-practical")
    task_ids = policy.AIDER_PRACTICAL_TASKS
    row["results"] = [
        {"task_id": task_ids[0], "status": "passed", "passed": True},
        {"task_id": task_ids[1], "status": "provider_error", "passed": False},
        {
            "task_id": task_ids[2],
            "status": "failed",
            "passed": False,
            "diagnostic": "truncated_before_content",
        },
    ]
    _, settings_fingerprint = policy.aider_model_settings(
        policy.BENCHMARK_POLICIES["aider-practical"],
        completion_token_budget=24_000,
    )
    row["extended_token_diagnostic"] = {
        "trigger_reason": "truncated_before_content",
        "completion_token_budget": 24_000,
        "task_count": 1,
        "completed_count": 1,
        "model_verdict_count": 1,
        "provider_error_count": 0,
        "harness_error_count": 0,
        "attempts_per_task": 2,
        "model_settings_fingerprint": settings_fingerprint,
        "harness_exit_code": 0,
        "timed_out": False,
        "wall_clock_timeout_seconds": 2700,
        "classification": "capable_but_output_hungry",
        "results": [
            {"task_id": task_ids[2], "status": "passed", "passed": True}
        ],
    }

    with pytest.raises(policy.CodingBenchmarkPolicyError, match="sole unresolved condition"):
        policy.validate_evidence_metadata(row, run_mode="calibration")


def test_legacy_polyglot_fingerprint_remains_readable():
    row = evidence("aider-polyglot")
    row["policy_fingerprint"] = next(
        iter(policy.LEGACY_POLICY_FINGERPRINTS_BY_SUITE["polyglot-2026-08-21"])
    )

    assert policy.validate_evidence_metadata(row, run_mode="calibration").name == "aider-polyglot"


@pytest.mark.parametrize(
    "model_id",
    [
        "stealth/unknown-model",
        "devstral-latest",
        "deepseek-v4:preview",
        "openrouter/free",
        "orcarouter/free",
        "openrouter/auto",
        "free",
    ],
)
def test_policy_rejects_mutable_or_opaque_model_ids(model_id):
    with pytest.raises(policy.CodingBenchmarkPolicyError, match="mutable or opaque"):
        policy.validate_model_identity("openrouter", model_id)


def test_policy_allows_the_exact_pinned_ox_alpha_identity_only():
    assert policy.validate_model_identity("nous", "stealth/ox-alpha") == (
        "nous",
        "stealth/ox-alpha",
    )

    with pytest.raises(policy.CodingBenchmarkPolicyError, match="mutable or opaque"):
        policy.validate_model_identity("nous", "stealth/ox-alpha-preview")


def test_policy_rejects_a_changed_secondary_source():
    row = evidence()
    row["source_revisions"][0]["commit"] = "a" * 40

    with pytest.raises(policy.CodingBenchmarkPolicyError, match="source revisions"):
        policy.validate_evidence_metadata(row, run_mode="calibration")


def test_policy_rejects_a_stale_policy_fingerprint():
    row = evidence()
    row["policy_fingerprint"] = "sha256:" + "a" * 64

    with pytest.raises(policy.CodingBenchmarkPolicyError, match="current coding policy fingerprint"):
        policy.validate_evidence_metadata(row, run_mode="calibration")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("completion_token_budget", 8_000, "completion_token_budget"),
        ("model_settings_fingerprint", "sha256:" + "b" * 64, "model_settings_fingerprint"),
    ],
)
def test_policy_rejects_changed_model_budget_settings(field, value, message):
    row = evidence()
    row[field] = value

    with pytest.raises(policy.CodingBenchmarkPolicyError, match=message):
        policy.validate_evidence_metadata(row, run_mode="calibration")


def test_policy_rejects_incomplete_certification_but_keeps_harness_incident_separate():
    row = evidence()
    spec = policy.BENCHMARK_POLICIES["aider-polyglot"]
    row.update(
        {
            "run_mode": "certification",
            "task_count": spec.certification_task_count,
            "attempts_per_task": spec.certification_attempts_per_task,
            "completed_count": spec.certification_task_count - 1,
            "model_verdict_count": spec.certification_task_count - 1,
            "provider_error_count": 1,
            "harness_exit_code": 0,
            "timed_out": False,
        }
    )

    with pytest.raises(policy.CodingBenchmarkPolicyError, match="model verdict for every task"):
        policy.validate_evidence_metadata(row, run_mode="certification")

    row["completed_count"] = spec.certification_task_count
    row["model_verdict_count"] = spec.certification_task_count
    row["provider_error_count"] = 0
    row["harness_exit_code"] = 1
    assert policy.validate_evidence_metadata(row, run_mode="certification").name == "aider-polyglot"


def test_provider_and_harness_errors_are_unmeasured_but_cover_the_sample():
    row = evidence()
    row.update(
        {
            "completed_count": 3,
            "model_verdict_count": 3,
            "provider_error_count": 1,
            "harness_error_count": 1,
        }
    )

    assert policy.validate_evidence_metadata(row, run_mode="calibration").name == "aider-polyglot"


def test_policy_validates_optional_relay_audit_counts():
    row = evidence()
    row["relay_audit"] = {
        "schema_version": 1,
        "chat_completion_count": 8,
        "task_count": 5,
        "provider_incident_count": 2,
        "model_failure_incident_count": 1,
        "unresolved_provider_error_count": 1,
        "unresolved_model_failure_count": 1,
    }

    assert policy.validate_evidence_metadata(row, run_mode="calibration").name == "aider-polyglot"

    row["relay_audit"]["unresolved_provider_error_count"] = 3
    with pytest.raises(policy.CodingBenchmarkPolicyError, match="provider incident count"):
        policy.validate_evidence_metadata(row, run_mode="calibration")
