from __future__ import annotations

from ficelle.routing_policy import (
    catalog_denies_structured_output,
    competence_gate_result,
    competence_label,
    model_matches_profile_requirements,
)


def free_access_eligible(model):
    return bool(model.get("free_access", {}).get("eligible"))


def model_row(**overrides):
    row = {
        "id": "ficelle/openrouter/example",
        "context_length": 131_072,
        "supports_tools": True,
        "supports_structured_outputs": True,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "supported_parameters": ["tools", "tool_choice", "response_format"],
        "free_access": {"eligible": True},
    }
    row.update(overrides)
    return row


def profile_requirements(**requirements):
    return {"requirements": requirements}


def matches(model, profile):
    return model_matches_profile_requirements(
        model,
        profile,
        default_min_context=128_000,
        free_access_eligible=free_access_eligible,
    )


def test_policy_requirements_accept_matching_model():
    profile = profile_requirements(
        free=True,
        tools=True,
        structured=True,
        input_modalities=["text"],
        input_modalities_any=["image", "audio"],
        output_modalities=["text"],
        supported_parameters=["tools"],
        supported_parameters_any=["response_format", "reasoning"],
        min_context=128_000,
    )

    assert matches(model_row(), profile) is True


def test_policy_requirements_reject_excluded_or_non_free_model():
    excluded = model_row(id="ficelle/openrouter/excluded")
    profile = {
        "excluded_models": ["ficelle/openrouter/excluded"],
        "requirements": {"free": True},
    }

    assert matches(excluded, profile) is False
    assert matches(model_row(free_access={"eligible": False}), profile_requirements(free=True)) is False


def test_policy_requirements_reject_missing_capabilities_and_small_context():
    assert matches(model_row(supports_tools=False), profile_requirements(tools=True)) is False
    assert matches(model_row(supports_structured_outputs=False), profile_requirements(structured=True)) is False
    assert matches(model_row(input_modalities=["text"]), profile_requirements(input_modalities=["image"])) is False
    assert matches(model_row(input_modalities=["text"]), profile_requirements(input_modalities_any=["audio"])) is False
    assert matches(model_row(output_modalities=[]), profile_requirements(output_modalities=["text"])) is False
    assert matches(model_row(supported_parameters=["tools"]), profile_requirements(supported_parameters=["response_format"])) is False
    assert matches(model_row(supported_parameters=["tools"]), profile_requirements(supported_parameters_any=["reasoning"])) is False
    assert matches(model_row(context_length=32_000), profile_requirements(min_context=128_000)) is False


def test_catalog_denies_structured_output_mirrors_the_routing_filter():
    no_structured = model_row(supports_structured_outputs=False, supported_parameters=["tools", "tool_choice"])

    assert catalog_denies_structured_output(no_structured, {"structured": True}) is True
    # The routing filter rejects the same row on the same field, which is what makes the skip
    # lossless: a `verified` verdict won by probing anyway could never have been routed.
    assert matches(no_structured, profile_requirements(structured=True)) is False

    # Declared support keeps its probes: the catalog's claim is what gets verified.
    assert catalog_denies_structured_output(model_row(), {"structured": True}) is False
    # A profile that does not require structured outputs never denies.
    assert catalog_denies_structured_output(no_structured, {}) is False
    assert catalog_denies_structured_output(no_structured, {"structured": False}) is False
    # A flag that may have been synthesized from provider defaults is not a catalog answer.
    assert (
        catalog_denies_structured_output(
            no_structured, {"structured": True}, structured_support_may_come_from_defaults=True
        )
        is False
    )
    # An absent flag means "not declared", never a denial — even though routing rejects it too.
    undeclared = model_row(supported_parameters=["tools", "tool_choice"])
    undeclared.pop("supports_structured_outputs")
    assert catalog_denies_structured_output(undeclared, {"structured": True}) is False


def test_competence_gate_uses_anti_empty_fallback_for_specialized_profiles():
    candidates = [{"id": "a"}, {"id": "b"}]

    kept, fallback = competence_gate_result(
        "ficelle/auto-tools",
        candidates,
        is_multimodal_profile=lambda _profile: False,
        is_specialized_profile=lambda _profile: True,
        multimodal_route_eligible=lambda _model: True,
        route_eligible_for_profile=lambda _model: False,
    )

    assert kept == candidates
    assert fallback is True


def test_competence_gate_keeps_benchmark_like_generic_profiles_ungated():
    candidates = [{"id": "a"}]

    kept, fallback = competence_gate_result(
        "ficelle/auto-fast",
        candidates,
        is_multimodal_profile=lambda _profile: False,
        is_specialized_profile=lambda _profile: False,
        multimodal_route_eligible=lambda _model: False,
        route_eligible_for_profile=lambda _model: False,
    )

    assert kept == candidates
    assert fallback is False


def label_for(model, profile_id, **overrides):
    callbacks = {
        "is_multimodal_profile": lambda _profile: False,
        "is_specialized_profile": lambda _profile: False,
        "multimodal_route_eligible": lambda _model: True,
        "multimodal_claimed_profiles": lambda _model: [],
        "profile_verified": lambda _profile, _model: False,
        "profile_failed": lambda _profile, _model: False,
        "capability_is_default_claimed": lambda _profile, _model: False,
        "capability_reference_routable": lambda _profile, _model: False,
    }
    callbacks.update(overrides)
    return competence_label(profile_id, model, **callbacks)


def test_competence_label_covers_specialized_and_multimodal_states():
    model = {"id": "ficelle/openrouter/model", "input_modalities": ["image"]}

    assert label_for(model, "ficelle/auto-fast") == "n/a"

    assert label_for(
        model,
        "ficelle/auto-tools",
        is_specialized_profile=lambda _profile: True,
        capability_is_default_claimed=lambda _profile, _model: True,
        capability_reference_routable=lambda _profile, _model: True,
    ) == "claimed_reference"

    assert label_for(
        model,
        "ficelle/auto-multimodal",
        is_multimodal_profile=lambda _profile: True,
        multimodal_claimed_profiles=lambda _model: ["ficelle/auto-vision"],
        profile_verified=lambda _profile, _model: True,
    ) == "verified"
