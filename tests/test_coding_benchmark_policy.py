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
    assert policy.policy_fingerprint().startswith("sha256:")


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
