from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ficelle import coding_benchmark_policy, coding_certification


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coding-certification.py"
SPEC = importlib.util.spec_from_file_location("ficelle_coding_certification_tool", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def write_run_record(
    path: Path,
    *,
    official: Path,
    settings: Path,
    benchmark: str,
    repository: str,
    commit: str,
) -> None:
    payload = json.loads(official.read_text(encoding="utf-8"))
    settings_payload = json.loads(settings.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "harness_repository": repository,
                "harness_commit": commit,
                "run_mode": payload["run_mode"],
                "provider": settings_payload["provider"],
                "upstream_model_id": settings_payload["upstream_model_id"],
                "policy_fingerprint": payload["policy_fingerprint"],
                "source_revisions": payload["source_revisions"],
                "settings_fingerprint": "sha256:" + hashlib.sha256(settings.read_bytes()).hexdigest(),
                "official_result_fingerprint": "sha256:" + hashlib.sha256(official.read_bytes()).hexdigest(),
                "command_fingerprint": "sha256:" + "d" * 64,
                "exit_code": 0,
                "official_result_exists": True,
            }
        ),
        encoding="utf-8",
    )


def official_payload(benchmark: str, *, pass_at_1: float) -> dict[str, object]:
    policy = coding_benchmark_policy.BENCHMARK_POLICIES[benchmark]
    return {
        "benchmark": benchmark,
        "run_mode": "calibration",
        "suite_version": policy.suite_version,
        "harness_repository": policy.harness_repository,
        "harness_commit": policy.harness_commit,
        "reference_agent": policy.reference_agent,
        "reference_agent_commit": policy.reference_agent_commit,
        "source_revisions": [
            {
                "name": source.name,
                "repository": source.repository,
                "commit": source.commit,
            }
            for source in policy.sources
        ],
        "policy_fingerprint": coding_benchmark_policy.policy_fingerprint(),
        "completion_token_budget": policy.completion_token_budget,
        "model_settings_fingerprint": coding_benchmark_policy.aider_model_settings(policy)[1],
        "task_count": policy.calibration_task_count,
        "completed_count": policy.calibration_task_count,
        "model_verdict_count": policy.calibration_task_count,
        "provider_error_count": 0,
        "harness_error_count": 0,
        "attempts_per_task": policy.calibration_attempts_per_task,
        "wall_clock_timeout_seconds": policy.calibration_wall_clock_seconds,
        "harness_exit_code": 0,
        "timed_out": False,
        "pass_at_1": pass_at_1,
        "efficiency": {
            "measured_task_count": policy.calibration_task_count,
            "duration_seconds": 50.0,
            "mean_duration_seconds": 50.0 / policy.calibration_task_count,
            "prompt_tokens": 1200,
            "completion_tokens": 800,
            "total_tokens": 2000,
        },
    }


def test_normalizer_preserves_exact_identity_and_provenance(tmp_path):
    official = tmp_path / "official.json"
    settings = tmp_path / "settings.json"
    run_record = tmp_path / "run.json"
    policy = coding_benchmark_policy.BENCHMARK_POLICIES["aider-polyglot"]
    official.write_text(
        json.dumps(official_payload("aider-polyglot", pass_at_1=0.5)), encoding="utf-8"
    )
    settings.write_text(
        '{"provider":"openrouter","upstream_model_id":"exact/code-id","temperature":0}',
        encoding="utf-8",
    )
    write_run_record(
        run_record,
        official=official,
        settings=settings,
        benchmark="aider-polyglot",
        repository="https://github.com/Aider-AI/aider",
        commit=policy.harness_commit,
    )

    row = tool.normalize_result(
        argparse.Namespace(
            benchmark="aider-polyglot",
            input=official,
            provider="OpenRouter",
            model="exact/code-id",
            suite_version=policy.suite_version,
            harness_repository=policy.harness_repository,
            harness_commit=policy.harness_commit,
            settings=settings,
            run_record=run_record,
            run_mode="calibration",
        )
    )

    assert row["provider"] == "openrouter"
    assert row["upstream_model_id"] == "exact/code-id"
    assert row["task_count"] == policy.calibration_task_count
    assert row["pass_at_1"] == 0.5
    assert row["efficiency"]["mean_duration_seconds"] == 10.0
    assert row["efficiency"]["total_tokens"] == 2000
    assert row["completion_token_budget"] == policy.completion_token_budget
    assert row["model_settings_fingerprint"] == coding_benchmark_policy.aider_model_settings(
        policy
    )[1]
    assert row["settings_fingerprint"].startswith("sha256:")
    assert row["evidence_kind"] == "calibration_run"


def test_normalizer_accepts_a_zero_score_and_rejects_an_identity_mismatch(tmp_path):
    official = tmp_path / "official.json"
    settings = tmp_path / "settings.json"
    run_record = tmp_path / "run.json"
    policy = coding_benchmark_policy.BENCHMARK_POLICIES["aider-polyglot"]
    official.write_text(
        json.dumps(official_payload("aider-polyglot", pass_at_1=0)), encoding="utf-8"
    )
    settings.write_text(
        '{"provider":"openrouter","upstream_model_id":"exact/code-id"}',
        encoding="utf-8",
    )
    write_run_record(
        run_record,
        official=official,
        settings=settings,
        benchmark="aider-polyglot",
        repository=policy.harness_repository,
        commit=policy.harness_commit,
    )
    args = argparse.Namespace(
        benchmark="aider-polyglot",
        input=official,
        provider="openrouter",
        model="exact/code-id",
        suite_version=policy.suite_version,
        harness_repository=policy.harness_repository,
        harness_commit=policy.harness_commit,
        settings=settings,
        run_record=run_record,
        run_mode="calibration",
    )

    assert tool.normalize_result(args)["pass_at_1"] == 0

    args.provider = "nous"
    with pytest.raises(ValueError, match="must match settings"):
        tool.normalize_result(args)


def test_efficiency_rejects_nonzero_metrics_without_measured_tasks():
    with pytest.raises(ValueError, match="zero measured tasks"):
        tool._efficiency_summary(
            {
                "efficiency": {
                    "measured_task_count": 0,
                    "duration_seconds": 1.0,
                    "mean_duration_seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            },
            5,
        )


def test_pass_summary_excludes_provider_and_harness_errors():
    payload = {
        "task_count": 5,
        "pass_at_1": 2 / 3,
        "results": [
            {"status": "passed", "passed": True},
            {"status": "failed", "passed": False},
            {"status": "passed", "passed": True},
            {"status": "provider_error", "passed": False},
            {"status": "harness_error", "passed": False},
        ],
    }

    assert tool._pass_summary(payload) == (5, pytest.approx(2 / 3))

    payload["pass_at_1"] = 0.4
    with pytest.raises(ValueError, match="does not match model verdict"):
        tool._pass_summary(payload)


def test_builder_refuses_incomplete_or_weak_results_and_weights_complete_results(tmp_path):
    paths = []
    scores = {"aider-polyglot": 0.8}
    for name, score in scores.items():
        policy = coding_benchmark_policy.BENCHMARK_POLICIES[name]
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "provider": "openrouter",
                    "upstream_model_id": "exact/code-id",
                    "name": name,
                    "run_mode": "certification",
                    "suite_version": policy.suite_version,
                    "harness_repository": policy.harness_repository,
                    "harness_commit": policy.harness_commit,
                    "reference_agent": policy.reference_agent,
                    "reference_agent_commit": policy.reference_agent_commit,
                    "source_revisions": [
                        {
                            "name": source.name,
                            "repository": source.repository,
                            "commit": source.commit,
                        }
                        for source in policy.sources
                    ],
                    "policy_fingerprint": coding_benchmark_policy.policy_fingerprint(),
                    "completion_token_budget": policy.completion_token_budget,
                    "model_settings_fingerprint": coding_benchmark_policy.aider_model_settings(policy)[1],
                    "task_count": policy.certification_task_count,
                    "completed_count": policy.certification_task_count,
                    "model_verdict_count": policy.certification_task_count,
                    "provider_error_count": 0,
                    "harness_error_count": 0,
                    "attempts_per_task": policy.certification_attempts_per_task,
                    "harness_exit_code": 0,
                    "timed_out": False,
                    "wall_clock_timeout_seconds": policy.calibration_wall_clock_seconds,
                    "pass_at_1": score,
                    "settings_fingerprint": "sha256:" + "a" * 64,
                    "run_record_fingerprint": "sha256:" + "b" * 64,
                    "official_result_fingerprint": "sha256:" + "c" * 64,
                    "command_fingerprint": "sha256:" + "d" * 64,
                    "evidence_kind": "central_run",
                    "observed_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    args = argparse.Namespace(results=[], priors=[], manifest_id="test", expires_days=30)
    with pytest.raises(ValueError, match="exact required"):
        tool.build_manifest(argparse.Namespace(results=[paths[0], paths[0]], priors=[], manifest_id="test", expires_days=30))

    args.results = paths
    manifest = tool.build_manifest(args)
    row = manifest["certifications"][0]
    assert row["quality_score"] == pytest.approx(80.0)
    assert {item["name"] for item in row["benchmarks"]} == coding_certification.REQUIRED_BENCHMARKS

    weak = json.loads(paths[0].read_text(encoding="utf-8"))
    weak["pass_at_1"] = 0.6
    paths[0].write_text(json.dumps(weak), encoding="utf-8")
    with pytest.raises(ValueError, match="below the 80 qualification floor"):
        tool.build_manifest(args)


def test_public_result_import_is_structurally_a_prior(tmp_path):
    public_result = tmp_path / "public.json"
    public_result.write_text('[{"passed":true},{"passed":false}]', encoding="utf-8")
    row = tool.prior(
        argparse.Namespace(
            source_url="https://example.test/results.json",
            benchmark="aider-polyglot",
            observed_at=None,
            provider="openrouter",
            model="exact/code-id",
            score=None,
            input=public_result,
            suite_version="2026-08",
            harness_commit="abcdef12" * 5,
        )
    )

    assert row["evidence_kind"] == "prior"
    assert row["score"] == 50
    assert row["task_count"] == 2


def test_bundled_manifest_is_valid():
    path = Path(__file__).resolve().parents[1] / "src" / "ficelle" / "assets" / "auto-coding-manifest.json"
    manifest = coding_certification.validate_manifest(
        coding_certification.strict_json_loads(path.read_bytes()),
        require_complete_policy=True,
    )

    assert [row["upstream_model_id"] for row in manifest["certifications"]] == [
        "moonshotai/kimi-k3",
    ]
    assert manifest["priors"] == []
