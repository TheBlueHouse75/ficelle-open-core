from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from ficelle.coding_benchmark_policy import AIDER_CALIBRATION_TASKS


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coding-benchmark-adapter.py"
SPEC = importlib.util.spec_from_file_location("ficelle_coding_benchmark_adapter", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def settings() -> dict[str, object]:
    return {
        "benchmark": "aider-polyglot",
        "provider": "nous",
        "upstream_model_id": "poolside/laguna-s-2.1:free",
        "ficelle_model_id": "ficelle/nous/poolside/laguna-s-2.1:free",
        "base_url": "http://host.docker.internal:8646/v1",
        "ficelle_host_header": "127.0.0.1:8647",
    }


def test_common_settings_bind_exact_provider_model_and_local_endpoint():
    assert adapter._validate_common_settings(settings(), "aider-polyglot") == (
        "nous",
        "poolside/laguna-s-2.1:free",
    )

    changed = settings()
    changed["ficelle_model_id"] = "ficelle/openrouter/poolside/laguna-s-2.1:free"
    with pytest.raises(ValueError, match="exact provider"):
        adapter._validate_common_settings(changed, "aider-polyglot")

    changed = settings()
    changed["base_url"] = "https://provider.example/v1"
    with pytest.raises(ValueError, match="Docker-to-host"):
        adapter._validate_common_settings(changed, "aider-polyglot")

    changed = settings()
    changed["ficelle_host_header"] = "router.example:8647"
    with pytest.raises(ValueError, match="loopback authority"):
        adapter._validate_common_settings(changed, "aider-polyglot")


def test_aider_result_rows_preserve_missing_and_failed_tasks(tmp_path):
    run_dir = tmp_path / "run"
    first = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    first.parent.mkdir(parents=True)
    first.write_text(
        json.dumps(
            {
                "tests_outcomes": [True],
                "duration": 12.5,
                "prompt_tokens": 120,
                "completion_tokens": 80,
            }
        ),
        encoding="utf-8",
    )
    second = run_dir / AIDER_CALIBRATION_TASKS[1] / ".aider.results.json"
    second.parent.mkdir(parents=True)
    second.write_text(json.dumps({"tests_outcomes": [False]}), encoding="utf-8")

    rows, completed = adapter._aider_result_rows(run_dir, AIDER_CALIBRATION_TASKS)

    assert completed == 2
    assert len(rows) == 5
    assert rows[0]["status"] == "passed"
    assert rows[0]["duration_seconds"] == 12.5
    assert rows[0]["prompt_tokens"] == 120
    assert rows[0]["completion_tokens"] == 80
    assert rows[1]["status"] == "failed"
    assert rows[2]["status"] == "harness_error"


def test_aider_result_rows_separate_provider_and_harness_errors(tmp_path):
    run_dir = tmp_path / "run"
    provider = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    provider.parent.mkdir(parents=True)
    provider.write_text(json.dumps({"exception": "HTTP 502 upstream provider error"}), encoding="utf-8")
    harness = run_dir / AIDER_CALIBRATION_TASKS[1] / ".aider.results.json"
    harness.parent.mkdir(parents=True)
    harness.write_text(json.dumps({"exception": "test fixture could not be loaded"}), encoding="utf-8")

    rows, completed = adapter._aider_result_rows(run_dir, AIDER_CALIBRATION_TASKS)

    assert completed == 0
    assert rows[0]["status"] == "provider_error"
    assert rows[1]["status"] == "harness_error"


def test_aider_result_rows_attributes_provider_timeout_to_in_flight_task(tmp_path):
    run_dir = tmp_path / "run"
    completed = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps({"tests_outcomes": [True]}), encoding="utf-8")

    rows, completed_count = adapter._aider_result_rows(
        run_dir,
        AIDER_CALIBRATION_TASKS,
        run_diagnostic="litellm.BadGatewayError: The API provider's servers are down",
    )

    assert completed_count == 1
    assert rows[1]["status"] == "provider_error"
    assert rows[2]["status"] == "harness_error"


def test_aider_result_rows_does_not_infer_provider_incident_from_model_text(tmp_path):
    rows, completed_count = adapter._aider_result_rows(
        tmp_path / "run",
        AIDER_CALIBRATION_TASKS,
        run_diagnostic="The provider class validates an upstream value.",
    )

    assert completed_count == 0
    assert rows[0]["status"] == "harness_error"


def test_aider_result_rows_treats_exhausted_output_budget_as_model_failure(tmp_path):
    run_dir = tmp_path / "run"

    rows, completed_count = adapter._aider_result_rows(
        run_dir,
        AIDER_CALIBRATION_TASKS,
        run_diagnostic=(
            "HTTP 502: truncated_before_content; completion token budget ran out before "
            "the model emitted any content"
        ),
    )

    assert completed_count == 1
    assert rows[0]["status"] == "failed"
    assert rows[1]["status"] == "harness_error"


@pytest.mark.parametrize(
    ("exception", "run_diagnostic", "expected_status", "expected_completed"),
    [
        (
            "HTTP 502 upstream provider error",
            "litellm.BadGatewayError: The API provider's servers are down",
            "provider_error",
            0,
        ),
        (
            "HTTP 502 truncated_before_content",
            "HTTP 502 truncated_before_content",
            "failed",
            1,
        ),
    ],
)
def test_aider_result_rows_does_not_repeat_persisted_final_incident(
    tmp_path,
    exception,
    run_diagnostic,
    expected_status,
    expected_completed,
):
    run_dir = tmp_path / "run"
    result = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"exception": exception}), encoding="utf-8")

    rows, completed_count = adapter._aider_result_rows(
        run_dir,
        AIDER_CALIBRATION_TASKS,
        run_diagnostic=run_diagnostic,
    )

    assert completed_count == expected_completed
    assert rows[0]["status"] == expected_status
    assert rows[1]["status"] == "harness_error"


def test_aider_result_rows_preserves_distinct_model_and_provider_incidents(tmp_path):
    run_dir = tmp_path / "run"
    result = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"exception": "HTTP 502 truncated_before_content"}),
        encoding="utf-8",
    )

    rows, completed_count = adapter._aider_result_rows(
        run_dir,
        AIDER_CALIBRATION_TASKS,
        run_diagnostic=(
            "HTTP 502 truncated_before_content\n"
            "litellm.BadGatewayError: The API provider's servers are down"
        ),
    )

    assert completed_count == 1
    assert rows[0]["status"] == "failed"
    assert rows[1]["status"] == "provider_error"
    assert rows[2]["status"] == "harness_error"


def test_aider_result_rows_ignore_invalid_efficiency_metrics(tmp_path):
    run_dir = tmp_path / "run"
    result = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "tests_outcomes": [True],
                "duration": -1,
                "prompt_tokens": True,
                "completion_tokens": "80",
            }
        ),
        encoding="utf-8",
    )

    rows, _ = adapter._aider_result_rows(run_dir, AIDER_CALIBRATION_TASKS)

    assert "duration_seconds" not in rows[0]
    assert "prompt_tokens" not in rows[0]
    assert "completion_tokens" not in rows[0]


def test_aider_efficiency_summary_aggregates_completed_rows():
    rows = [
        {
            "status": "passed",
            "duration_seconds": 12.5,
            "prompt_tokens": 120,
            "completion_tokens": 80,
        },
        {
            "status": "failed",
            "duration_seconds": 7.5,
            "prompt_tokens": 30,
            "completion_tokens": 20,
        },
        {"prompt_tokens": 999, "completion_tokens": 999},
        {"status": "error"},
    ]

    assert adapter._aider_efficiency_summary(rows) == {
        "measured_task_count": 2,
        "duration_seconds": 20.0,
        "mean_duration_seconds": 10.0,
        "prompt_tokens": 150,
        "completion_tokens": 100,
        "total_tokens": 250,
    }


def test_aider_calibration_recipe_refuses_upstream_drift(monkeypatch):
    official = b"FROM buildpack-deps:jammy\nRUN true && \\\n    npm install \\\n    jest \\\n    babel-jest@29.6.4\nCOPY . /aider\n"
    monkeypatch.setattr(
        adapter,
        "AIDER_DOCKERFILE_SHA256",
        __import__("hashlib").sha256(official).hexdigest(),
    )

    recipe, fingerprint = adapter._aider_calibration_recipe(official)

    assert "NPM_CONFIG_FETCH_RETRY_MINTIMEOUT=1000" in recipe
    assert "jest@29.7.0" in recipe
    assert "ficelle-build-cache/java-bank-account" in recipe
    assert "./gradlew --no-daemon testClasses" in recipe
    assert "ficelleResolveTestRuntime" in recipe
    assert fingerprint.startswith("sha256:")
    with pytest.raises(ValueError, match="fingerprint"):
        adapter._aider_calibration_recipe(official + b"# drift\n")


def test_aider_build_cache_stages_the_pinned_java_task(tmp_path):
    checkout = tmp_path / "aider"
    source_checkout = tmp_path / "polyglot"
    wrapper = source_checkout / adapter.AIDER_JAVA_CALIBRATION_TASK / "gradlew"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")

    adapter._stage_aider_build_cache(checkout, source_checkout)

    staged = checkout / "benchmark" / "ficelle-build-cache" / "java-bank-account" / "gradlew"
    assert staged.read_text(encoding="utf-8") == "#!/bin/sh\n"
    resolver = staged.parent / "ficelle-resolve.gradle"
    assert 'getByName("testRuntimeClasspath").files' in resolver.read_text(encoding="utf-8")


def test_aider_calibration_harness_pins_task_order_and_process_exit(monkeypatch):
    official = (
        b"before\n    random.shuffle(test_dnames)\nafter\n"
        b"    summarize_results(dirname)\n\n    return 0\n\n"
        b'if __name__ == "__main__":\n    app()\n'
    )
    monkeypatch.setattr(
        adapter,
        "AIDER_BENCHMARK_SHA256",
        __import__("hashlib").sha256(official).hexdigest(),
    )

    harness, fingerprint = adapter._aider_calibration_harness(official)

    assert "random.shuffle" not in harness
    assert "test_dnames.sort()" in harness
    assert "    sys.stdout.flush()\n    sys.stderr.flush()\n    os._exit(0)\n" in harness
    assert fingerprint.startswith("sha256:")
    with pytest.raises(ValueError, match="fingerprint"):
        adapter._aider_calibration_harness(official + b"# drift\n")


def test_aider_model_settings_give_every_candidate_the_same_completion_budget():
    model_settings, fingerprint = adapter._aider_model_settings(12_000)

    assert model_settings == (
        "- name: aider/extra_params\n"
        "  extra_params:\n"
        "    max_tokens: 12000\n"
    )
    assert fingerprint == "sha256:" + hashlib.sha256(model_settings.encode("utf-8")).hexdigest()
