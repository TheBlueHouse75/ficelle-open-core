from __future__ import annotations

import copy

from ficelle import router


def model(source: str, upstream_id: str, model_id: str) -> dict:
    return {
        "id": model_id,
        "source": source,
        "upstream_id": upstream_id,
        "invokable": True,
        "context_length": 128_000,
        "supports_tools": True,
        "supports_structured_outputs": True,
        "pricing": {"prompt": "0", "completion": "0"},
        "free_access": {
            "eligible": True,
            "mode": "catalog_free",
            "proof": "provider_pricing",
            "scope": "model",
            "status": "available",
        },
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "supported_parameters": ["tools", "tool_choice"],
    }


def test_auto_coding_is_exported_and_has_a_local_compatibility_probe():
    assert "ficelle/auto-coding" in router.VIRTUAL_MODELS
    assert "ficelle/auto-coding" in router.TARGET_EXPORT_VIRTUAL_MODELS
    body, test_type, expected = router.benchmark_body("ficelle/auto-coding")
    assert test_type == "coding_compatibility"
    assert expected == "ficelle_open_file:ready"
    assert body["tools"][0]["function"]["name"] == "ficelle_open_file"


def test_auto_coding_gate_is_exact_and_never_uses_anti_empty_fallback(monkeypatch):
    certified = model("openrouter", "acme/code", "ficelle/openrouter/acme/code")
    same_model_elsewhere = model("nous", "acme/code", "ficelle/nous/acme/code")
    manifest = {
        "certifications": [
            {
                "provider": "openrouter",
                "upstream_model_id": "acme/code",
                "quality_score": 80,
            }
        ]
    }
    monkeypatch.setattr(router.coding_certification, "cached_manifest", lambda _path: manifest)

    kept, fallback = router.route_competence_gate_result(
        "ficelle/auto-coding", [certified, same_model_elsewhere], {}
    )
    assert kept == [certified]
    assert fallback is False

    kept, fallback = router.route_competence_gate_result(
        "ficelle/auto-coding", [same_model_elsewhere], {}
    )
    assert kept == []
    assert fallback is False


def test_auto_coding_canary_selection_excludes_uncertified_models(monkeypatch):
    certified = model("openrouter", "acme/code", "ficelle/openrouter/acme/code")
    uncertified = model("openrouter", "acme/other", "ficelle/openrouter/acme/other")
    monkeypatch.setattr(
        router.coding_certification,
        "cached_manifest",
        lambda _path: {
            "certifications": [
                {"provider": "openrouter", "upstream_model_id": "acme/code", "quality_score": 80}
            ]
        },
    )

    result = router.select_models_result_from_state(
        "ficelle/auto-coding",
        {"models": [certified, uncertified]},
        copy.deepcopy(router.DEFAULT_CONFIG),
        {},
        purpose="benchmark",
    )

    assert [candidate.id for candidate in result.candidates] == [certified["id"]]

    failed_state = {
        "verified_capabilities": {
            router.cooldown_key(certified): {
                "ficelle/auto-coding": {
                    "status": "failed",
                    "test_type": "coding_compatibility",
                    "tested_at": router.now_iso(),
                }
            }
        }
    }
    retry = router.select_models_result_from_state(
        "ficelle/auto-coding",
        {"models": [certified, uncertified]},
        copy.deepcopy(router.DEFAULT_CONFIG),
        failed_state,
        purpose="benchmark",
    )

    assert [candidate.id for candidate in retry.candidates] == [certified["id"]]


def test_auto_coding_is_not_probed_by_generic_capability_discovery():
    assert "ficelle/auto-coding" not in router.probeable_capability_profiles(
        copy.deepcopy(router.DEFAULT_CONFIG)
    )

def test_auto_coding_order_uses_central_quality_before_local_transport(monkeypatch):
    lower = model("openrouter", "acme/lower", "ficelle/openrouter/acme/lower")
    higher = model("openrouter", "acme/higher", "ficelle/openrouter/acme/higher")
    manifest = {
        "certifications": [
            {"provider": "openrouter", "upstream_model_id": "acme/lower", "quality_score": 70},
            {"provider": "openrouter", "upstream_model_id": "acme/higher", "quality_score": 90},
        ]
    }
    monkeypatch.setattr(router.coding_certification, "cached_manifest", lambda _path: manifest)
    state = {
        "stats": {
            router.cooldown_key(lower): {"successes": 100, "failures": 0, "latency_ewma": 0.1},
            router.cooldown_key(higher): {"successes": 0, "failures": 4, "latency_ewma": 29},
        }
    }

    assert router.sort_available_for_virtual_model("ficelle/auto-coding", [lower, higher], state) == [higher, lower]


def test_failed_local_compatibility_canary_blocks_but_cannot_certify(monkeypatch):
    candidate = model("openrouter", "acme/code", "ficelle/openrouter/acme/code")
    certified_manifest = {
        "certifications": [
            {"provider": "openrouter", "upstream_model_id": "acme/code", "quality_score": 80}
        ]
    }
    monkeypatch.setattr(router.coding_certification, "cached_manifest", lambda _path: certified_manifest)
    state = {
        "verified_capabilities": {
            router.cooldown_key(candidate): {
                "ficelle/auto-coding": {
                    "status": "failed",
                    "test_type": "coding_compatibility",
                    "tested_at": router.now_iso(),
                }
            }
        }
    }
    assert router.route_competence_gate_result("ficelle/auto-coding", [candidate], state)[0] == []

    monkeypatch.setattr(router.coding_certification, "cached_manifest", lambda _path: {"certifications": []})
    success_state = {
        "verified_capabilities": {
            router.cooldown_key(candidate): {
                "ficelle/auto-coding": {
                    "status": "verified",
                    "test_type": "coding_compatibility",
                    "tested_at": router.now_iso(),
                }
            }
        }
    }
    assert router.route_competence_gate_result("ficelle/auto-coding", [candidate], success_state)[0] == []
