from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coding-benchmark-runner.py"
SPEC = importlib.util.spec_from_file_location("ficelle_coding_benchmark_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def base_args(tmp_path: Path) -> list[str]:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "provider": "openrouter",
                "upstream_model_id": "exact/code-id",
                "run_mode": "calibration",
                "task_count": 5,
                "attempts_per_task": 1,
                "wall_clock_timeout_seconds": 1800,
            }
        ),
        encoding="utf-8",
    )
    return [
        "--benchmark",
        "aider-polyglot",
        "--repository",
        "https://github.com/Aider-AI/aider.git",
        "--commit",
        "5dc9490bb35f9729ef2c95d00a19ccd30c26339c",
        "--settings",
        str(settings),
        "--result",
        str(tmp_path / "result.json"),
        "--record",
        str(tmp_path / "record.json"),
    ]


def test_runner_rejects_a_preexisting_result(tmp_path):
    args = base_args(tmp_path)
    (tmp_path / "result.json").write_text('{"stale":true}', encoding="utf-8")

    assert runner.main([*args, "--", "/usr/bin/true"]) == 2


def test_runner_rejects_a_preexisting_run_record(tmp_path):
    args = base_args(tmp_path)
    (tmp_path / "record.json").write_text('{"stale":true}', encoding="utf-8")

    assert runner.main([*args, "--", "/usr/bin/true"]) == 2


def test_runner_rejects_identical_result_and_record_paths(tmp_path):
    args = base_args(tmp_path)
    record_flag = args.index("--record")
    args[record_flag + 1] = args[args.index("--result") + 1]

    assert runner.main([*args, "--", "/usr/bin/true"]) == 2


def test_runner_rejects_duplicate_settings_keys(tmp_path):
    args = base_args(tmp_path)
    settings = tmp_path / "settings.json"
    settings.write_text(
        '{"provider":"openrouter","provider":"nous","upstream_model_id":"exact/code-id",'
        '"run_mode":"calibration","task_count":5,"attempts_per_task":1}',
        encoding="utf-8",
    )

    assert runner.main([*args, "--", "/usr/bin/true"]) == 2


def test_runner_passes_only_a_minimal_explicit_environment(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    checkout_commit = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
    command_environment = {}
    monkeypatch.setenv("PROVIDER_API_KEY", "provider-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")
    def fake_run(command, **kwargs):
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir()
        if len(command) >= 4 and command[0] == "git" and command[1] == "-C" and command[3] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=checkout_commit + "\n")
        if command[0] == "/usr/bin/true":
            command_environment.update(kwargs["env"])
            Path(command_environment["FICELLE_BENCHMARK_RESULT"]).write_text(
                '{"passed":true}', encoding="utf-8"
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main([*args, "--pass-env", "PROVIDER_API_KEY", "--", "/usr/bin/true"]) == 0
    assert command_environment["PROVIDER_API_KEY"] == "provider-secret"
    assert "UNRELATED_SECRET" not in command_environment
