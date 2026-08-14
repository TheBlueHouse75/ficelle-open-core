from __future__ import annotations

import json

import pytest

from ficelle.use_cases.synthetic_health import (
    SyntheticHealthCase,
    aggregate_case_status,
    build_identity_match,
    corpus_hash,
    request_shape,
    request_template_hash,
    unverified_build_verdict,
    validate_unique_case_ids,
)


def make_case(case_id: str, marker: str = "private-prompt-marker") -> SyntheticHealthCase:
    return SyntheticHealthCase(
        id=case_id,
        phase=3,
        lane="live",
        target_kind="direct-model",
        target_id="new-provider/model-a",
        validator="exact-text",
        timeout_class="fast",
        body={
            "model": "new-provider/model-a",
            "messages": [{"role": "user", "content": marker}],
            "stream": True,
            "top_k": 1,
        },
        expected={"marker": marker},
    )


def test_request_shape_never_persists_prompt_text() -> None:
    shape = request_shape(make_case("case-a").body or {})

    assert "private-prompt-marker" not in json.dumps(shape)
    assert shape["message_roles"] == ["user"]
    assert shape["character_count"] == len("private-prompt-marker")
    assert shape["optional_parameters"] == ["top_k"]


def test_request_shape_and_hash_ignore_internal_dependency_metadata() -> None:
    public_body = {"model": "ficelle/auto-tools", "messages": [], "temperature": 0}
    internal_body = {**public_body, "__dependency_case_id": "p3.tools.choice"}

    assert request_shape(internal_body) == request_shape(public_body)
    assert request_template_hash(internal_body) == request_template_hash(public_body)


def test_corpus_hash_is_stable_across_generated_markers_of_equal_shape() -> None:
    assert corpus_hash([make_case("case-a", "marker-111")]) == corpus_hash([make_case("case-a", "marker-222")])


def test_duplicate_case_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate synthetic-health case ids"):
        validate_unique_case_ids([make_case("same"), make_case("same")])


def test_aggregate_status_preserves_fallback_and_safety_precedence() -> None:
    axes = {name: {"status": "pass"} for name in ("transport", "protocol", "semantic", "routing", "policy", "telemetry", "safety")}
    assert aggregate_case_status(axes, fallback_observed=True) == "pass_with_fallback"

    axes["safety"] = {"status": "fail"}
    assert aggregate_case_status(axes, fallback_observed=True) == "safety_breach"


def make_build(content_hash: str | None, **overrides: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "ficelle_version": "0.1.8",
        "package_dir": "/repo/src/ficelle",
        "install_mode": "editable",
        "runtime_content_hash": content_hash,
        "complete": True,
    }
    identity.update(overrides)
    return identity


def test_build_identity_match_compares_content_not_location() -> None:
    verdict = build_identity_match(
        make_build("hash-a"),
        make_build("hash-a", package_dir="/venv/site-packages/ficelle", install_mode="distribution"),
    )

    assert verdict == {"match": True, "diagnostic": None}


def test_build_identity_mismatch_names_both_builds() -> None:
    # The incident shape (2026-08-10): same ficelle_version on both sides, so only the
    # runtime content hash discriminates; the diagnostic must point at both locations.
    harness = make_build("a" * 64)
    service = make_build("b" * 64, package_dir="/stale/venv/site-packages/ficelle")

    verdict = build_identity_match(harness, service)

    assert verdict["match"] is False
    assert verdict["diagnostic"].startswith("build mismatch")
    assert "/repo/src/ficelle" in verdict["diagnostic"]
    assert "/stale/venv/site-packages/ficelle" in verdict["diagnostic"]


def test_build_identity_match_fails_closed_without_a_service_identity() -> None:
    for service in (None, "not-a-dict", {}, make_build(None)):
        verdict = build_identity_match(make_build("hash-a"), service)

        assert verdict["match"] is False
        assert "build mismatch" in verdict["diagnostic"]


def test_build_identity_match_fails_closed_without_a_harness_hash() -> None:
    verdict = build_identity_match(make_build(None), make_build("hash-a"))

    assert verdict["match"] is False
    assert "harness" in verdict["diagnostic"]


def test_unverified_build_verdict_is_neither_match_nor_mismatch() -> None:
    verdict = unverified_build_verdict("service unreachable")

    assert verdict["match"] is None
    assert verdict["diagnostic"] == "service unreachable"
