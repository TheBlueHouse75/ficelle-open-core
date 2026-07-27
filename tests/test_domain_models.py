from __future__ import annotations

import importlib

from ficelle.domain_models import CatalogModel, SelectionResult


def free_model(model_id: str, **overrides):
    model = {
        "id": model_id,
        "source": "openrouter",
        "upstream_id": model_id.removeprefix("ficelle/openrouter/"),
        "name": model_id,
        "context_length": 131_072,
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
        "supported_parameters": ["tools", "tool_choice", "response_format"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "invokable": True,
    }
    model.update(overrides)
    return model


def load_router(monkeypatch, tmp_path):
    ficelle_home = tmp_path / ".ficelle"
    ficelle_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FICELLE_HOME", str(ficelle_home))
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", str(ficelle_home))
    import ficelle.router as router

    return importlib.reload(router)


def test_catalog_model_conversion_handles_missing_malformed_and_unknown_fields():
    assert CatalogModel.from_raw(None) is None

    model = CatalogModel.from_raw(
        {
            "id": "ficelle/openrouter/example",
            "source": "openrouter",
            "context_length": "bad",
            "supports_tools": "yes",
            "invokable": True,
            "provider_token": "sk-should-not-leak",
            "apiKey": "camel-secret",
            "x-api-key": "header-secret",
            "metadata": {"api_key": "nested-secret", "safe": ["ok", {"refreshToken": "nested-refresh-secret"}]},
            "vendor_hint": {"tier": "free"},
            "supported_parameters": [" tools ", "", "response_format"],
        }
    )

    assert model is not None
    assert model.id == "ficelle/openrouter/example"
    assert model.upstream_id == "example"
    assert model.context_length == 0
    assert model.supports_tools is True
    assert model.invokable is True
    assert model.supported_parameters == ["tools", "response_format"]
    assert model.extra_fields["vendor_hint"] == {"tier": "free"}
    assert model.extra_fields["provider_token"] == "[redacted]"
    assert model.extra_fields["apiKey"] == "[redacted]"
    assert model.extra_fields["x-api-key"] == "[redacted]"
    assert model.extra_fields["metadata"] == {
        "api_key": "[redacted]",
        "safe": ["ok", {"refreshToken": "[redacted]"}],
    }
    assert "sk-should-not-leak" not in repr(model)


def test_catalog_model_derives_upstream_id_without_dropping_provider_namespace():
    model = CatalogModel.from_raw(
        {
            "id": "ficelle/openrouter/google/gemma-4-31b-it:free",
            "source": "openrouter",
            "invokable": True,
        }
    )

    assert model is not None
    assert model.upstream_id == "google/gemma-4-31b-it:free"


def test_catalog_model_keeps_legacy_row_round_trip():
    raw = free_model("ficelle/openrouter/good", vendor_hint="kept")

    model = CatalogModel.from_raw(raw)

    assert model is not None
    assert model.id == "ficelle/openrouter/good"
    assert model.as_legacy_dict() == raw


def test_select_models_result_exposes_typed_candidates_and_exclusion_reasons(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    good = free_model("ficelle/openrouter/good")
    no_tools = free_model("ficelle/openrouter/no-tools", supports_tools=False, supported_parameters=[])
    not_invokable = free_model("ficelle/openrouter/not-invokable", invokable=False)
    legacy_truthy_invokable = free_model("ficelle/openrouter/legacy-truthy", invokable=1)

    result = router.select_models_result(
        "ficelle/auto-tools",
        {"models": [good, no_tools, not_invokable, legacy_truthy_invokable]},
        router.load_config(),
    )

    assert isinstance(result, SelectionResult)
    assert result.request.canonical_model == "ficelle/auto-tools"
    assert [candidate.id for candidate in result.candidates] == [
        "ficelle/openrouter/good",
        "ficelle/openrouter/legacy-truthy",
    ]
    assert result.candidates[0].context_length == 131_072
    assert result.candidates[1].invokable is True
    assert result.excluded_reasons["ficelle/openrouter/no-tools"] == "profile_requirements"
    assert result.excluded_reasons["ficelle/openrouter/not-invokable"] == "not_invokable"
    assert result.as_legacy_models() == [good, legacy_truthy_invokable]


def test_select_models_result_accepts_catalog_without_models_key(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)

    result = router.select_models_result("ficelle/auto-tools", {}, router.load_config())

    assert result.candidates == []
    assert result.as_legacy_models() == []


def test_select_models_result_records_manual_profile_requirement_exclusions(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    good = free_model("ficelle/openrouter/manual-good")
    no_tools = free_model("ficelle/openrouter/manual-no-tools", supports_tools=False, supported_parameters=[])
    tail = free_model("ficelle/openrouter/manual-tail")
    config = router.load_config()
    config["virtual_profiles"]["ficelle/manual-test"] = {
        "mode": "manual_order",
        "models": [no_tools["id"], good["id"]],
        "excluded_models": [],
        "auto_tail": True,
        "requirements": router.DEFAULT_PROFILE_REQUIREMENTS,
    }

    result = router.select_models_result(
        "ficelle/manual-test",
        {"models": [good, no_tools, tail]},
        config,
    )

    assert [candidate.id for candidate in result.candidates] == [
        "ficelle/openrouter/manual-good",
        "ficelle/openrouter/manual-tail",
    ]
    assert result.excluded_reasons["ficelle/openrouter/manual-no-tools"] == "profile_requirements"


def test_select_models_result_marks_anti_empty_competence_fallback(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    model_a = free_model("ficelle/openrouter/a", upstream_id="a")
    model_b = free_model("ficelle/openrouter/b", upstream_id="b")
    router.atomic_write_json(
        router.STATE_PATH,
        {
            "verified_capabilities": {
                key: {"ficelle/auto-tools": {"status": "failed", "test_type": "tool_call", "capability": "tool_call"}}
                for key in ("openrouter::a", "openrouter::b")
            }
        },
    )

    result = router.select_models_result(
        "ficelle/auto-tools",
        {"models": [model_a, model_b]},
        router.load_config(),
    )

    assert {candidate.id for candidate in result.candidates} == {"ficelle/openrouter/a", "ficelle/openrouter/b"}
    assert result.anti_empty_fallback is True
    assert result.excluded_reasons == {}


def test_select_models_result_from_state_has_no_runtime_io_or_quota_probes(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    catalog = {"models": [free_model("ficelle/openrouter/good")]}
    config = router.load_config()

    def fail_state_read():
        raise AssertionError("state read")

    def fail_probe(*_args, **_kwargs):
        raise AssertionError("probe")

    monkeypatch.setattr(router, "fresh_runtime_state", fail_state_read)
    monkeypatch.setattr(router, "run_due_quota_probes", fail_probe)

    result = router.select_models_result_from_state(
        "ficelle/auto-tools",
        catalog,
        config,
        {},
    )

    assert [candidate.id for candidate in result.candidates] == ["ficelle/openrouter/good"]


def test_select_models_result_from_state_uses_supplied_cooldown_state(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    blocked = free_model("ficelle/openrouter/blocked", upstream_id="blocked")
    healthy = free_model("ficelle/openrouter/healthy", upstream_id="healthy")
    catalog = {"models": [blocked, healthy]}
    config = router.load_config()
    state = {
        "cooldowns": {
            "openrouter::blocked": {
                "reason": "server_error",
                "until": router.time.time() + 3600,
            }
        }
    }

    result = router.select_models_result_from_state(
        "ficelle/auto-tools",
        catalog,
        config,
        state,
    )

    assert [candidate.id for candidate in result.candidates] == ["ficelle/openrouter/healthy"]
    assert result.excluded_reasons["ficelle/openrouter/blocked"] == "cooldown"


def test_select_models_result_from_state_applies_reference_routing_config(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    model = free_model(
        "ficelle/openrouter/ref-tools",
        capabilities_from_defaults=["tools"],
        capabilities_from_reference=["tools"],
        reference_confidence="high",
    )
    catalog = {"models": [model]}
    config = router.load_config()
    config["route_on_capability_reference"] = True
    router.apply_route_on_capability_reference({"route_on_capability_reference": False})

    result = router.select_models_result_from_state(
        "ficelle/auto-tools",
        catalog,
        config,
        {},
    )

    assert [candidate.id for candidate in result.candidates] == ["ficelle/openrouter/ref-tools"]
    assert result.anti_empty_fallback is False
