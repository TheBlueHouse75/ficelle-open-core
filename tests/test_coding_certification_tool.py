from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ficelle import coding_certification


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
    path.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "harness_repository": repository,
                "harness_commit": commit,
                "settings_fingerprint": "sha256:" + hashlib.sha256(settings.read_bytes()).hexdigest(),
                "official_result_fingerprint": "sha256:" + hashlib.sha256(official.read_bytes()).hexdigest(),
                "command_fingerprint": "sha256:" + "d" * 64,
                "exit_code": 0,
                "official_result_exists": True,
            }
        ),
        encoding="utf-8",
    )


def test_normalizer_preserves_exact_identity_and_provenance(tmp_path):
    official = tmp_path / "official.json"
    settings = tmp_path / "settings.json"
    run_record = tmp_path / "run.json"
    official.write_text(json.dumps({"results": [{"passed": True}, {"passed": False}]}), encoding="utf-8")
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
        commit="abcdef12" * 5,
    )

    row = tool.normalize_result(
        argparse.Namespace(
            benchmark="aider-polyglot",
            input=official,
            provider="OpenRouter",
            model="exact/code-id",
            suite_version="2026-08",
            harness_repository="https://github.com/Aider-AI/aider",
            harness_commit="abcdef12" * 5,
            settings=settings,
            run_record=run_record,
        )
    )

    assert row["provider"] == "openrouter"
    assert row["upstream_model_id"] == "exact/code-id"
    assert row["task_count"] == 2
    assert row["pass_at_1"] == 0.5
    assert row["settings_fingerprint"].startswith("sha256:")
    assert row["evidence_kind"] == "central_run"


def test_normalizer_accepts_an_official_result_array_and_a_zero_score(tmp_path):
    official = tmp_path / "official.json"
    settings = tmp_path / "settings.json"
    run_record = tmp_path / "run.json"
    official.write_text('[{"resolved":false}]', encoding="utf-8")
    settings.write_text(
        '{"provider":"openrouter","upstream_model_id":"exact/code-id"}',
        encoding="utf-8",
    )
    write_run_record(
        run_record,
        official=official,
        settings=settings,
        benchmark="swe-rebench",
        repository="https://github.com/example/harness",
        commit="abcdef12" * 5,
    )
    args = argparse.Namespace(
        benchmark="swe-rebench",
        input=official,
        provider="openrouter",
        model="exact/code-id",
        suite_version="2026-08",
        harness_repository="https://github.com/example/harness",
        harness_commit="abcdef12" * 5,
        settings=settings,
        run_record=run_record,
    )

    assert tool.normalize_result(args)["pass_at_1"] == 0

    args.provider = "nous"
    with pytest.raises(ValueError, match="must match settings"):
        tool.normalize_result(args)


def test_builder_refuses_incomplete_suite_and_weights_complete_results(tmp_path):
    paths = []
    scores = {"aider-polyglot": 0.8, "swe-rebench": 0.6, "terminal-bench-2.1": 0.4}
    for name, score in scores.items():
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "provider": "openrouter",
                    "upstream_model_id": "exact/code-id",
                    "name": name,
                    "suite_version": "2026-08",
                    "harness_repository": "https://github.com/example/harness",
                    "harness_commit": "abcdef12" * 5,
                    "task_count": 10,
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
    args = argparse.Namespace(results=paths[:-1], priors=[], manifest_id="test", expires_days=30)
    with pytest.raises(ValueError, match="exact required"):
        tool.build_manifest(args)

    args.results = paths
    manifest = tool.build_manifest(args)
    row = manifest["certifications"][0]
    assert row["quality_score"] == pytest.approx(62.0)
    assert {item["name"] for item in row["benchmarks"]} == coding_certification.REQUIRED_BENCHMARKS


def test_signer_uses_raw_private_key_and_core_verifier(monkeypatch):
    private = Ed25519PrivateKey.generate()
    raw_private = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    monkeypatch.setattr(tool, "private_key_bytes", lambda: raw_private)
    monkeypatch.setitem(
        coding_certification.PUBLIC_KEYS_B64,
        tool.KEY_ID,
        base64.b64encode(public).decode("ascii"),
    )
    now = datetime.now(UTC)
    manifest = {
        "schema_version": 1,
        "manifest_id": "empty-test",
        "generated_at": now.isoformat(),
        "expires_at": (now + __import__("datetime").timedelta(days=1)).isoformat(),
        "policy_version": "coding-v1",
        "certifications": [],
        "priors": [],
    }

    envelope = tool.sign_manifest(manifest)

    assert coding_certification.verify_envelope(envelope)["manifest_id"] == "empty-test"


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


def test_checked_in_launch_envelope_is_valid_and_intentionally_empty():
    path = Path(__file__).resolve().parents[1] / "certifications" / "auto-coding-manifest.json"
    envelope = coding_certification.strict_json_loads(path.read_bytes())

    manifest = coding_certification.verify_envelope(
        envelope,
        now=datetime(2026, 8, 22, tzinfo=UTC),
    )

    assert manifest["certifications"] == []
    assert manifest["priors"] == []
