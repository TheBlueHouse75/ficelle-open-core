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


def test_aider_result_rows_distinguish_first_pass_from_repaired_success(tmp_path):
    run_dir = tmp_path / "run"
    result = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"tests_outcomes": [False, True]}), encoding="utf-8")

    rows, completed = adapter._aider_result_rows(run_dir, AIDER_CALIBRATION_TASKS)

    assert completed == 1
    assert rows[0]["first_passed"] is False
    assert rows[0]["passed"] is True
    assert rows[0]["attempt_count"] == 2
    assert rows[0]["status"] == "passed"


def test_aider_result_rows_do_not_score_a_zero_token_http_failure(tmp_path):
    run_dir = tmp_path / "run"
    result = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "tests_outcomes": [False, False],
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        ),
        encoding="utf-8",
    )

    rows, completed = adapter._aider_result_rows(run_dir, AIDER_CALIBRATION_TASKS)

    assert completed == 0
    assert rows[0]["status"] == "harness_error"
    assert rows[0]["diagnostic"] == "no_measured_generation"


def test_relay_audit_maps_retries_and_repairs_to_explicit_tasks(tmp_path):
    audit = tmp_path / "relay-audit.jsonl"
    task_one = AIDER_CALIBRATION_TASKS[0]
    task_two = AIDER_CALIBRATION_TASKS[1]
    events = [
        {
            "schema_version": 1,
            "event_index": 1,
            "request_fingerprint": "a" * 64,
            "task_id": task_one,
            "response_status": 429,
            "outcome": "provider_error",
        },
        {
            "schema_version": 1,
            "event_index": 2,
            "request_fingerprint": "a" * 64,
            "task_id": task_one,
            "response_status": 200,
            "outcome": "success",
        },
        {
            "schema_version": 1,
            "event_index": 3,
            "request_fingerprint": "b" * 64,
            "task_id": task_one,
            "response_status": 429,
            "outcome": "provider_error",
        },
        {
            "schema_version": 1,
            "event_index": 4,
            "request_fingerprint": "c" * 64,
            "task_id": task_two,
            "response_status": 502,
            "outcome": "model_failure",
        },
    ]
    audit.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    incidents, summary = adapter._relay_audit_by_task(audit, AIDER_CALIBRATION_TASKS)

    assert incidents == {
        AIDER_CALIBRATION_TASKS[0]: {"provider_error"},
        AIDER_CALIBRATION_TASKS[1]: {"model_failure"},
    }
    assert summary == {
        "schema_version": 1,
        "chat_completion_count": 4,
        "task_count": 2,
        "provider_incident_count": 2,
        "model_failure_incident_count": 1,
        "unresolved_provider_error_count": 1,
        "unresolved_model_failure_count": 1,
    }


def test_relay_audit_separates_standard_and_extended_budget_incidents(tmp_path):
    audit = tmp_path / "relay-audit.jsonl"
    task = AIDER_CALIBRATION_TASKS[0]
    events = [
        {
            "schema_version": 2,
            "event_index": 1,
            "request_fingerprint": "a" * 64,
            "task_id": task,
            "response_status": 502,
            "outcome": "model_failure",
            "failure_reason": "truncated_before_content",
            "completion_token_budget": 12_000,
        },
        {
            "schema_version": 2,
            "event_index": 2,
            "request_fingerprint": "b" * 64,
            "task_id": task,
            "response_status": 200,
            "outcome": "success",
            "failure_reason": None,
            "completion_token_budget": 24_000,
        },
    ]
    audit.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    standard, standard_summary = adapter._relay_audit_by_task(
        audit,
        (task,),
        completion_token_budget=12_000,
    )
    extended, extended_summary = adapter._relay_audit_by_task(
        audit,
        (task,),
        completion_token_budget=24_000,
    )

    assert standard == {task: {"truncated_before_content"}}
    assert standard_summary["model_failure_incident_count"] == 1
    assert extended == {}
    assert extended_summary["chat_completion_count"] == 1


def test_relay_audit_reclassifies_contaminated_repair_and_empty_model_response(tmp_path):
    run_dir = tmp_path / "run"
    provider_result = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    provider_result.parent.mkdir(parents=True)
    provider_result.write_text(
        json.dumps(
            {
                "tests_outcomes": [False, False],
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        ),
        encoding="utf-8",
    )
    model_result = run_dir / AIDER_CALIBRATION_TASKS[1] / ".aider.results.json"
    model_result.parent.mkdir(parents=True)
    model_result.write_text(
        json.dumps(
            {
                "tests_outcomes": [False, False],
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        ),
        encoding="utf-8",
    )

    rows, completed = adapter._aider_result_rows(
        run_dir,
        AIDER_CALIBRATION_TASKS,
        relay_incidents={
            AIDER_CALIBRATION_TASKS[0]: {"provider_error"},
            AIDER_CALIBRATION_TASKS[1]: {"model_failure"},
        },
    )

    assert completed == 1
    assert rows[0]["status"] == "provider_error"
    assert rows[0]["diagnostic"] == "unresolved_provider_request"
    assert rows[1]["status"] == "failed"
    assert rows[1]["diagnostic"] == "unresolved_model_response"


def test_exact_truncation_reason_is_preserved_for_extended_diagnostic_trigger(tmp_path):
    rows, completed = adapter._aider_result_rows(
        tmp_path / "run",
        (AIDER_CALIBRATION_TASKS[0],),
        relay_incidents={AIDER_CALIBRATION_TASKS[0]: {"truncated_before_content"}},
    )

    assert completed == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["diagnostic"] == "truncated_before_content"
    assert adapter._extended_diagnostic_task_ids(rows) == (AIDER_CALIBRATION_TASKS[0],)


def test_mixed_model_failure_reasons_do_not_trigger_extended_diagnostic(tmp_path):
    task = AIDER_CALIBRATION_TASKS[0]
    rows, completed = adapter._aider_result_rows(
        tmp_path / "run",
        (task,),
        relay_incidents={
            task: {"truncated_before_content", "empty_assistant_message"}
        },
    )

    assert completed == 1
    assert rows[0]["diagnostic"] == "unresolved_model_response"
    assert adapter._extended_diagnostic_task_ids(rows) == ()


def test_repair_truncation_remains_visible_after_measured_first_attempt(tmp_path):
    task = AIDER_CALIBRATION_TASKS[0]
    result = tmp_path / "run" / task / ".aider.results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "tests_outcomes": [False, False],
                "prompt_tokens": 100,
                "completion_tokens": 80,
            }
        ),
        encoding="utf-8",
    )

    rows, completed = adapter._aider_result_rows(
        tmp_path / "run",
        (task,),
        relay_incidents={task: {"truncated_before_content"}},
    )

    assert completed == 1
    assert rows[0]["diagnostic"] == "truncated_before_content"
    assert adapter._extended_diagnostic_task_ids(rows) == (task,)


def test_extended_diagnostic_does_not_run_for_other_or_mixed_failures():
    assert adapter._extended_diagnostic_task_ids(
        [{"task_id": "go/task", "status": "failed", "diagnostic": "test_failure"}]
    ) == ()
    assert adapter._extended_diagnostic_task_ids(
        [
            {
                "task_id": "go/task",
                "status": "failed",
                "diagnostic": "truncated_before_content",
            },
            {"task_id": "java/task", "status": "provider_error"},
        ]
    ) == ()


def test_extended_diagnostic_classification_keeps_quality_and_availability_separate():
    assert adapter._extended_diagnostic_classification([{"status": "passed"}]) == (
        "capable_but_output_hungry"
    )
    assert adapter._extended_diagnostic_classification([{"status": "failed"}]) == (
        "failed_at_extended_budget"
    )
    assert adapter._extended_diagnostic_classification([{"status": "provider_error"}]) == (
        "inconclusive"
    )


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


def test_aider_result_rows_attributes_ficelle_request_deadline_to_provider(tmp_path):
    run_dir = tmp_path / "run"
    completed = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps({"tests_outcomes": [True]}), encoding="utf-8")

    rows, completed_count = adapter._aider_result_rows(
        run_dir,
        AIDER_CALIBRATION_TASKS,
        run_diagnostic=(
            "litellm.InternalServerError: upstream request failed: "
            "RequestDeadlineExceeded request_deadline_exceeded"
        ),
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


def test_aider_result_rows_treats_empty_assistant_response_as_model_failure(tmp_path):
    run_dir = tmp_path / "run"
    result = run_dir / AIDER_CALIBRATION_TASKS[0] / ".aider.results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "exception": (
                    "litellm.BadRequestError: HTTP 422 upstream_failure "
                    "empty_assistant_message"
                )
            }
        ),
        encoding="utf-8",
    )

    rows, completed_count = adapter._aider_result_rows(run_dir, AIDER_CALIBRATION_TASKS)

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
        b"    LONG_TIMEOUT = 24 * 60 * 60\n"
        b"    main_model = models.Model(\n"
        b"        model_name,\n"
        b"        weak_model=weak_model_name,\n"
        b"        editor_model=editor_model,\n"
        b"        editor_edit_format=editor_edit_format,\n"
        b"        verbose=verbose,\n"
        b"    )\n"
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
    assert "LONG_TIMEOUT = 0" in harness
    assert '"X-Ficelle-Benchmark-Task": "/".join(testdir.parts[-4:])' in harness
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


def test_aider_model_settings_allow_only_the_pinned_extended_diagnostic_budget():
    model_settings, _ = adapter._aider_model_settings(
        24_000,
        "aider-practical",
        extended_diagnostic=True,
    )

    assert "max_tokens: 24000" in model_settings
    with pytest.raises(ValueError, match="does not match pinned policy"):
        adapter._aider_model_settings(
            20_000,
            "aider-practical",
            extended_diagnostic=True,
        )
