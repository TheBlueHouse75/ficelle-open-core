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
        {"source": "gemini", "upstream_id": "models/gemini-3.6-flash"}, manifest
    )["quality_score"] == 60


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


def test_bundled_status_reports_three_models():
    status = coding_certification.public_status()

    assert status["status"] == "bundled"
    assert status["certification_count"] == 3
    assert status["manifest_id"] == "2026-08-23-coding-pool-3"


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
