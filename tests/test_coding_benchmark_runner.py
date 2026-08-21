from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coding-benchmark-runner.py"
SPEC = importlib.util.spec_from_file_location("ficelle_coding_benchmark_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def base_args(tmp_path: Path) -> list[str]:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    return [
        "--benchmark",
        "aider-polyglot",
        "--repository",
        "https://github.com/example/harness.git",
        "--commit",
        "abcdef12" * 5,
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


def test_runner_passes_only_a_minimal_explicit_environment(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    checkout_commit = "abcdef12" * 5
    command_environment = {}
    monkeypatch.setenv("PROVIDER_API_KEY", "provider-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")
    monkeypatch.setattr(runner, "signing_material_is_accessible", lambda: False)

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
    assert "FICELLE_CODING_CERT_PRIVATE_KEY_B64" not in command_environment


def test_runner_refuses_to_execute_when_signing_material_is_accessible(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "signing_material_is_accessible", lambda: True)
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    assert runner.main([*base_args(tmp_path), "--", "/usr/bin/true"]) == 2
    assert calls == []


def test_runner_refuses_to_pass_the_signing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FICELLE_CODING_CERT_PRIVATE_KEY_B64", "signing-secret")

    assert runner.main(
        [
            *base_args(tmp_path),
            "--pass-env",
            "FICELLE_CODING_CERT_PRIVATE_KEY_B64",
            "--",
            "/usr/bin/true",
        ]
    ) == 2
