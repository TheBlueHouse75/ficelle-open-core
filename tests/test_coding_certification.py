from __future__ import annotations

import copy

import pytest

from ficelle import coding_certification


def test_bundled_manifest_qualifies_models_across_providers():
    manifest = coding_certification.cached_manifest()

    assert manifest is not None
    assert coding_certification.certification_for_model(
        {"source": "openrouter", "upstream_id": "moonshotai/kimi-k3"}, manifest
    )["quality_score"] == 100
    assert coding_certification.certification_for_model(
        {"source": "nous", "upstream_id": "moonshotai/kimi-k3"}, manifest
    )["quality_score"] == 100
    assert coding_certification.certification_for_model(
        {"source": "ollama", "upstream_id": "moonshotai/kimi-k3"}, manifest
    )["quality_score"] == 100
    assert coding_certification.certification_for_model(
        {"source": "openrouter", "upstream_id": "deepseek/deepseek-v4-pro-free"}, manifest
    )["quality_score"] == 60
    assert coding_certification.certification_for_model(
        {"source": "nvidia", "upstream_id": "deepseek-ai/deepseek-v4-pro-0813"}, manifest
    )["tier"] == "verified"
    assert coding_certification.certification_for_model(
        {"source": "ollama", "upstream_id": "deepseek-v4-pro:0813"}, manifest
    )["tier"] == "verified"
    assert coding_certification.certification_for_model(
        {"source": "gemini", "upstream_id": "models/gemini-3.6-flash"}, manifest
    )["quality_score"] == 100


def test_bundled_manifest_rejects_an_unqualified_model():
    manifest = coding_certification.cached_manifest()

    assert coding_certification.certification_for_model(
        {"source": "openrouter", "upstream_id": "unqualified/model"}, manifest
    ) is None


def test_manifest_rejects_weak_or_inconsistent_qualification_scores():
    manifest = coding_certification.cached_manifest()
    assert manifest is not None

    weak = copy.deepcopy(manifest)
    weak["certifications"][0]["benchmarks"][0]["pass_at_1"] = 0.4
    weak["certifications"][0]["quality_score"] = 40
    with pytest.raises(coding_certification.CodingCertificationError, match="qualification floor"):
        coding_certification.validate_manifest(weak, require_complete_policy=True)

    inconsistent = copy.deepcopy(manifest)
    inconsistent["certifications"][0]["quality_score"] = 80
    with pytest.raises(coding_certification.CodingCertificationError, match="does not match"):
        coding_certification.validate_manifest(inconsistent, require_complete_policy=True)


def test_bundled_status_reports_verified_and_provisional_models():
    status = coding_certification.public_status()

    assert status["status"] == "bundled"
    assert status["certification_count"] == 10
    assert status["provisional_count"] == 0
    assert status["qualification_count"] == 10
    assert status["manifest_id"] == "2026-09-17-coding-pool-v3"


def test_bundled_manifest_stays_bound_to_the_process_policy(monkeypatch, tmp_path):
    loaded = coding_certification.cached_manifest()
    replacement = tmp_path / "auto-coding-manifest.json"
    replacement.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(coding_certification, "BUILTIN_MANIFEST_PATH", replacement)

    assert loaded is not None
    assert coding_certification.cached_manifest() is loaded


def test_bundled_manifest_verifies_union_alpha():
    manifest = coding_certification.cached_manifest()

    assert manifest is not None
    row = coding_certification.certification_for_model(
        {"source": "openrouter", "upstream_id": "stealth/union-alpha"}, manifest
    )
    assert row is not None
    assert row["tier"] == "verified"
    assert row["upstream_model_id"] == "stealth/union-alpha"
    assert row["quality_score"] == 66.6667
    assert row["benchmarks"][0]["pass_at_1"] == pytest.approx(2 / 3)
    assert row["benchmarks"][0]["resolved_rate"] == pytest.approx(2 / 3)
    kilo_row = coding_certification.certification_for_model(
        {"source": "kilo", "upstream_id": "stealth/union-alpha"}, manifest
    )
    assert kilo_row is not None
    assert kilo_row["tier"] == "verified"
    assert coding_certification.certification_for_model(
        {"source": "orcarouter", "upstream_id": "stealth/union-alpha-free"}, manifest
    ) is None


def test_bundled_manifest_verifies_qwen_aliases():
    manifest = coding_certification.cached_manifest()

    assert manifest is not None
    for upstream_id in (
        "@cf/qwen/qwen3.8-27b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.8-27b:free",
        "qwen/qwen3.8-27b-free",
    ):
        row = coding_certification.certification_for_model(
            {"source": "provider", "upstream_id": upstream_id}, manifest
        )
        assert row is not None
        assert row["tier"] == "verified"
        assert row["benchmarks"][0]["pass_at_1"] == pytest.approx(2 / 3)
        assert row["benchmarks"][0]["resolved_rate"] == 1.0


def test_bundled_manifest_verifies_gemma_provider_aliases():
    manifest = coding_certification.cached_manifest()

    assert manifest is not None
    for upstream_id in (
        "@cf/google/gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it:free",
        "models/gemma-4-26b-a4b-it",
    ):
        row = coding_certification.certification_for_model(
            {"source": "provider", "upstream_id": upstream_id}, manifest
        )
        assert row is not None
        assert row["tier"] == "verified"


@pytest.mark.parametrize(
    ("upstream_id", "expected_primary"),
    [
        ("google/gemma-4-26b-a4b-it:free", "models/gemma-4-26b-a4b-it"),
        ("tencent/hy3:free", "tencent/hy3-free"),
    ],
)
def test_bundled_manifest_verifies_candidate_wave_aliases(upstream_id, expected_primary):
    manifest = coding_certification.cached_manifest()

    assert manifest is not None
    row = coding_certification.certification_for_model(
        {"source": "provider", "upstream_id": upstream_id}, manifest
    )
    assert row is not None
    assert row["upstream_model_id"] == expected_primary
    assert row["tier"] == "verified"


def test_strict_parser_rejects_duplicate_keys_and_non_finite_numbers():
    try:
        coding_certification.strict_json_loads('{"a":1,"a":2}')
    except coding_certification.CodingCertificationError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate key was accepted")

    try:
        coding_certification.strict_json_loads('{"score":NaN}')
    except coding_certification.CodingCertificationError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite number was accepted")
