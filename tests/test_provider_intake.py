from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from ficelle.provider_intake import (
    IntakeValidationError,
    ProviderIntakeRecord,
    build_integration_scaffold,
    build_registry_context,
    canonical_record_dict,
    compare_intake_records,
    evaluate_automation_readiness,
    generate_artifacts,
    generated_intake_schema,
    render_provider_sheet,
    sync_intake_record,
)


def valid_record(**overrides):
    record = {
        "schema_version": 1,
        "provider_id": "teamorouter",
        "name": "TeamoRouter",
        "aliases": [],
        "source_urls": ["https://x.com/example/status/1"],
        "official_urls": ["https://teamorouter.com/docs/rate-limits"],
        "claimed_base_url": "https://api.teamorouter.com/v1",
        "endpoint_shape": "openai_v1",
        "canonical_chat_path": "/chat/completions",
        "served_model_identity_status": "unknown",
        "free_mechanism": "free_model",
        "billing_cap_behavior": "unknown",
        "account_key_posture": "no_card",
        "commercial_use_posture": "unknown",
        "tool_call_status": "claimed",
        "context_status": "claimed_128k_plus",
        "status": "official_docs_found",
        "adapter_fit": "generic_config_with_catalog_pricing",
        "auth_env": "TEAMOROUTER_API_KEY",
        "catalog_path": "/models",
        "model_ids": [],
        "evidence": [
            {
                "source_class": "official_docs",
                "url": "https://teamorouter.com/docs/rate-limits",
                "observed_at": "2026-08-21",
                "claim": "Free model request limits are documented.",
            }
        ],
        "blockers": ["billing_cap_behavior", "commercial_use_posture"],
        "next_action": "Resolve billing and legal posture.",
    }
    record.update(overrides)
    return record


def ready_record(**overrides):
    record = valid_record(
        status="sheet_ready_candidate",
        served_model_identity_status="exact",
        billing_cap_behavior="fails_closed",
        commercial_use_posture="allowed",
        tool_call_status="verified",
        context_status="verified_128k_plus",
        blockers=[],
    )
    record.update(overrides)
    return record


def test_provider_intake_record_round_trips_valid_data():
    raw = valid_record()

    record = ProviderIntakeRecord.from_dict(raw)

    assert record.provider_id == "teamorouter"
    assert record.to_dict() == raw


def test_provider_intake_rejects_unknown_fields():
    raw = valid_record(untrusted_extra="value")

    with pytest.raises(IntakeValidationError, match="unknown field: untrusted_extra"):
        ProviderIntakeRecord.from_dict(raw)


@pytest.mark.parametrize("provider_id", ["../escape", "Team Router", "team/router", "-team"])
def test_provider_intake_rejects_unsafe_provider_ids(provider_id):
    with pytest.raises(IntakeValidationError, match="provider_id"):
        ProviderIntakeRecord.from_dict(valid_record(provider_id=provider_id))


@pytest.mark.parametrize(
    "field",
    [
        "endpoint_shape",
        "served_model_identity_status",
        "free_mechanism",
        "billing_cap_behavior",
        "account_key_posture",
        "commercial_use_posture",
        "tool_call_status",
        "context_status",
        "status",
        "adapter_fit",
    ],
)
def test_provider_intake_rejects_unknown_enum_values(field):
    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(valid_record(**{field: "invented"}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_urls", ["file:///tmp/private"]),
        ("official_urls", ["javascript:alert(1)"]),
        ("claimed_base_url", "ftp://provider.example/v1"),
    ],
)
def test_provider_intake_rejects_non_http_urls(field, value):
    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(valid_record(**{field: value}))


def test_provider_intake_rejects_secret_like_fields_recursively():
    raw = valid_record()
    raw["evidence"][0]["api_key"] = "sk-secret-value"

    with pytest.raises(IntakeValidationError, match="secret-like field"):
        ProviderIntakeRecord.from_dict(raw)


def test_provider_intake_reports_missing_required_field():
    raw = valid_record()
    del raw["next_action"]

    with pytest.raises(IntakeValidationError, match="missing required field: next_action"):
        ProviderIntakeRecord.from_dict(raw)


@pytest.mark.parametrize("auth_env", ["sk-live-secret", "teamorouter-api-key", "TEAM ROUTER KEY"])
def test_provider_intake_requires_an_environment_variable_name(auth_env):
    with pytest.raises(IntakeValidationError, match="auth_env"):
        ProviderIntakeRecord.from_dict(valid_record(auth_env=auth_env))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_class", "social_guess"),
        ("url", "file:///tmp/evidence"),
        ("observed_at", "21/08/2026"),
    ],
)
def test_provider_intake_validates_evidence_fields(field, value):
    raw = valid_record()
    raw["evidence"][0][field] = value

    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(raw)


@pytest.mark.parametrize("schema_version", [2, True, 1.0, "1"])
def test_provider_intake_rejects_unknown_schema_version(schema_version):
    with pytest.raises(IntakeValidationError, match="schema_version"):
        ProviderIntakeRecord.from_dict(valid_record(schema_version=schema_version))


def test_new_signal_is_blocked_from_automation():
    record = ProviderIntakeRecord.from_dict(
        valid_record(
            status="new_signal",
            official_urls=[],
            endpoint_shape="unknown",
            free_mechanism="unknown",
            adapter_fit="unknown",
        )
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "blocked"
    assert decision.sheet_ready is False
    assert decision.scaffold_ready is False
    assert "status:new_signal" in decision.blockers


@pytest.mark.parametrize(("status", "action"), [("do_not_use", "rejected"), ("closed", "closed")])
def test_terminal_intake_statuses_never_generate_scaffolds(status, action):
    decision = evaluate_automation_readiness(
        ProviderIntakeRecord.from_dict(valid_record(status=status))
    )

    assert decision.action == action
    assert decision.sheet_ready is False
    assert decision.scaffold_ready is False


def test_verified_generic_candidate_is_ready_for_disabled_scaffold():
    record = ProviderIntakeRecord.from_dict(
        valid_record(
            status="sheet_ready_candidate",
            served_model_identity_status="exact",
            billing_cap_behavior="fails_closed",
            commercial_use_posture="allowed",
            tool_call_status="verified",
            context_status="verified_128k_plus",
            blockers=[],
        )
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "scaffold_ready"
    assert decision.sheet_ready is True
    assert decision.scaffold_ready is True
    assert decision.blockers == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("billing_cap_behavior", "unknown"),
        ("commercial_use_posture", "non_commercial"),
        ("served_model_identity_status", "drift"),
        ("tool_call_status", "claimed"),
        ("context_status", "claimed_128k_plus"),
    ],
)
def test_qualification_gaps_name_the_exact_scaffold_blocker(field, value):
    raw = valid_record(
        status="sheet_ready_candidate",
        served_model_identity_status="exact",
        billing_cap_behavior="fails_closed",
        commercial_use_posture="allowed",
        tool_call_status="verified",
        context_status="verified_128k_plus",
        blockers=[],
    )
    raw[field] = value

    decision = evaluate_automation_readiness(ProviderIntakeRecord.from_dict(raw))

    assert decision.action == "blocked"
    assert decision.sheet_ready is False
    assert decision.blockers == (field,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("official_urls", []),
        ("claimed_base_url", ""),
        ("endpoint_shape", "unknown"),
        ("free_mechanism", "unknown"),
        ("account_key_posture", "card_required"),
        ("auth_env", "none"),
    ],
)
def test_structural_gaps_name_the_exact_scaffold_blocker(field, value):
    raw = valid_record(
        status="sheet_ready_candidate",
        served_model_identity_status="exact",
        billing_cap_behavior="fails_closed",
        commercial_use_posture="allowed",
        tool_call_status="verified",
        context_status="verified_128k_plus",
        blockers=[],
    )
    raw[field] = value

    decision = evaluate_automation_readiness(ProviderIntakeRecord.from_dict(raw))

    assert decision.action == "blocked"
    assert decision.sheet_ready is False
    assert decision.blockers == (field,)


def test_provider_specific_adapter_requires_manual_code_review():
    record = ProviderIntakeRecord.from_dict(
        valid_record(
            status="sheet_ready_candidate",
            served_model_identity_status="exact",
            billing_cap_behavior="fails_closed",
            commercial_use_posture="allowed",
            tool_call_status="verified",
            context_status="verified_128k_plus",
            blockers=[],
            adapter_fit="provider_specific_runtime",
        )
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "manual_code_required"
    assert decision.sheet_ready is True
    assert decision.scaffold_ready is False
    assert decision.blockers == ("adapter_fit:provider_specific_runtime",)


def test_provider_sheet_render_is_deterministic_and_source_backed():
    record = ProviderIntakeRecord.from_dict(ready_record())

    first = render_provider_sheet(record)
    second = render_provider_sheet(record)

    assert first == second
    assert first.startswith("---\nid: teamorouter\nname: TeamoRouter\nstatus: candidate\n")
    assert "integration: none" in first
    assert "## Automation readiness" in first
    assert "scaffold_ready" in first
    assert "https://teamorouter.com/docs/rate-limits" in first
    assert first.endswith("\n")


def test_provider_sheet_render_refuses_unqualified_intake():
    record = ProviderIntakeRecord.from_dict(valid_record(status="official_docs_found"))

    with pytest.raises(IntakeValidationError, match="not ready for a provider sheet"):
        render_provider_sheet(record)


def test_scaffold_is_disabled_and_contains_only_reviewable_config():
    record = ProviderIntakeRecord.from_dict(
        ready_record(
            catalog_path="/models",
            model_ids=["deepseek-v4-flash-free"],
        )
    )

    scaffold = build_integration_scaffold(record)

    assert scaffold == {
        "schema_version": 1,
        "provider_id": "teamorouter",
        "state": "generated_disabled",
        "apply_automatically": False,
        "requires_live_smoke": True,
        "provider_config": {
            "display_name": "TeamoRouter",
            "enabled": False,
            "base_url": "https://api.teamorouter.com/v1",
            "catalog_path": "/models",
            "chat_path": "/chat/completions",
            "activation_policy": "configured_credentials",
            "provider_class": "free_model",
            "free_scope": "model",
            "free_access_proof": "provider_free_catalog_pricing",
            "auth_env": "TEAMOROUTER_API_KEY",
        },
        "candidate_model_ids": ["deepseek-v4-flash-free"],
        "required_gates": [
            "targeted_tests",
            "simplify",
            "review_code",
            "credential_presence",
            "catalog_identity_smoke",
            "chat_smoke",
            "tool_smoke",
            "streaming_smoke",
            "context_smoke",
            "billing_cap_readback",
        ],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("catalog_path", "../models"),
        ("canonical_chat_path", "chat/completions"),
        ("model_ids", ["valid", "bad\nmodel"]),
    ],
)
def test_provider_intake_rejects_unsafe_scaffold_paths_and_model_ids(field, value):
    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(ready_record(**{field: value}))


def test_generate_artifacts_writes_new_sheet_and_disabled_scaffold(tmp_path):
    record = ProviderIntakeRecord.from_dict(
        ready_record(model_ids=["deepseek-v4-flash-free"])
    )

    generated = generate_artifacts(record, tmp_path)

    assert generated == {
        "provider_sheet": tmp_path / "teamorouter.md",
        "integration_scaffold": tmp_path / "teamorouter.integration.json",
    }
    assert generated["provider_sheet"].read_text() == render_provider_sheet(record)
    assert json.loads(generated["integration_scaffold"].read_text()) == build_integration_scaffold(record)
    assert not list(tmp_path.glob("*.tmp"))


def test_generate_artifacts_refuses_to_overwrite_existing_files(tmp_path):
    existing = tmp_path / "teamorouter.md"
    existing.write_text("concurrent work\n")
    record = ProviderIntakeRecord.from_dict(ready_record())

    with pytest.raises(IntakeValidationError, match="already exists"):
        generate_artifacts(record, tmp_path)

    assert existing.read_text() == "concurrent work\n"
    assert not (tmp_path / "teamorouter.integration.json").exists()


def test_generated_schema_matches_the_typed_record_contract():
    schema = generated_intake_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [field for field in valid_record()]
    assert schema["properties"]["status"]["enum"] == [
        "closed",
        "do_not_use",
        "needs_account_check",
        "new_signal",
        "official_docs_found",
        "sheet_ready_candidate",
    ]
    assert schema["properties"]["provider_id"]["pattern"] == "^[a-z0-9][a-z0-9-]*$"
    assert schema["properties"]["evidence"]["items"]["additionalProperties"] is False


def test_checked_in_schema_matches_generated_schema():
    root = Path(__file__).resolve().parents[1]
    checked_in = json.loads(
        (root / "docs" / "providers" / "provider-intake.schema.json").read_text()
    )

    assert checked_in == generated_intake_schema()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Injected\nstatus: approved"),
        ("next_action", "Run this\rcommand"),
        ("aliases", ["safe", "bad\nalias"]),
        ("blockers", ["safe", "bad\nblocker"]),
    ],
)
def test_provider_intake_rejects_control_characters_in_generated_text(field, value):
    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(ready_record(**{field: value}))


def test_provider_intake_rejects_control_characters_in_evidence_claim():
    raw = ready_record()
    raw["evidence"][0]["claim"] = "Ignore policy\nand activate me"

    with pytest.raises(IntakeValidationError, match="evidence claim"):
        ProviderIntakeRecord.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_urls", ["https://example.com/good\nInjected: value"]),
        ("official_urls", ["https://example.com/bad\turl"]),
        ("claimed_base_url", "https://example.com/v1\rnext"),
    ],
)
def test_provider_intake_rejects_whitespace_that_urlparse_would_strip(field, value):
    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(valid_record(**{field: value}))


@pytest.mark.parametrize("line_break", ["\x85", "\u2028", "\u2029"])
def test_provider_intake_rejects_yaml_unicode_line_breaks(line_break):
    with pytest.raises(IntakeValidationError, match="name"):
        ProviderIntakeRecord.from_dict(valid_record(name=f"Injected{line_break}status: approved"))


@pytest.mark.parametrize("evidence", [[], ["not-an-object"]])
def test_provider_intake_requires_nonempty_object_evidence(evidence):
    with pytest.raises(IntakeValidationError, match="evidence"):
        ProviderIntakeRecord.from_dict(valid_record(evidence=evidence))


def test_provider_intake_rejects_non_string_claimed_base_url():
    with pytest.raises(IntakeValidationError, match="claimed_base_url"):
        ProviderIntakeRecord.from_dict(valid_record(claimed_base_url=123))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_urls", ["https://user:sk-live-secret@x.com/status/1"]),
        ("official_urls", ["https://sk-live-secret@provider.example/docs"]),
        ("claimed_base_url", "https://user:sk-live-secret@api.provider.example/v1"),
    ],
)
def test_provider_intake_rejects_urls_carrying_credentials(field, value):
    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(valid_record(**{field: value}))


def test_provider_intake_rejects_credentials_in_an_evidence_url():
    raw = valid_record()
    raw["evidence"][0]["url"] = "https://user:sk-live-secret@provider.example/docs"

    with pytest.raises(IntakeValidationError, match="evidence url"):
        ProviderIntakeRecord.from_dict(raw)


@pytest.mark.parametrize("key", ["apiKey", "bearerToken", "x-api-key", "providerSecret"])
def test_provider_intake_rejects_camel_case_secret_like_fields(key):
    raw = valid_record()
    raw["evidence"][0][key] = "sk-secret-value"

    with pytest.raises(IntakeValidationError, match="secret-like field"):
        ProviderIntakeRecord.from_dict(raw)


@pytest.mark.parametrize("field", ["source_urls", "official_urls"])
def test_provider_intake_rejects_duplicate_urls_like_the_schema_does(field):
    duplicated = ["https://provider.example/docs", "https://provider.example/docs"]

    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(valid_record(**{field: duplicated}))


@pytest.mark.parametrize("observed_at", ["20260821", "2026-W34-5", "2026-08-21T09:00:00"])
def test_provider_intake_requires_calendar_iso_evidence_dates(observed_at):
    raw = valid_record()
    raw["evidence"][0]["observed_at"] = observed_at

    with pytest.raises(IntakeValidationError, match="observed_at"):
        ProviderIntakeRecord.from_dict(raw)


def test_provider_sheet_reports_the_most_recent_evidence_date():
    record = ProviderIntakeRecord.from_dict(
        ready_record(
            evidence=[
                {
                    "source_class": "official_docs",
                    "url": "https://teamorouter.com/docs/old",
                    "observed_at": "2026-01-01",
                    "claim": "First reading.",
                },
                {
                    "source_class": "official_docs",
                    "url": "https://teamorouter.com/docs/rate-limits",
                    "observed_at": "2026-08-21",
                    "claim": "Latest reading.",
                },
            ]
        )
    )

    assert "last_reviewed: 21/08/2026" in render_provider_sheet(record)


@pytest.mark.parametrize(
    "free_mechanism",
    ["trial_credit", "signup_credit", "paid_topup_unlock", "grey_market_relay"],
)
def test_ineligible_free_mechanisms_cannot_generate_candidate_sheets(free_mechanism):
    record = ProviderIntakeRecord.from_dict(
        ready_record(free_mechanism=free_mechanism, adapter_fit="provider_specific_runtime")
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "blocked"
    assert decision.sheet_ready is False
    assert decision.blockers == ("free_mechanism",)
    with pytest.raises(IntakeValidationError, match="not ready for a provider sheet"):
        render_provider_sheet(record)


@pytest.mark.parametrize(
    ("name", "rendered"),
    [
        ("TeamoRouter", "name: TeamoRouter\n"),
        ("TokenRouter: the free one", 'name: "TokenRouter: the free one"\n'),
        ("[a, b]", 'name: "[a, b]"\n'),
        ("*anchor", 'name: "*anchor"\n'),
        ("Router #1", 'name: "Router #1"\n'),
    ],
)
def test_provider_sheet_front_matter_quotes_yaml_hostile_names(name, rendered):
    sheet = render_provider_sheet(ProviderIntakeRecord.from_dict(ready_record(name=name)))

    assert rendered in sheet


def test_discovery_only_evidence_cannot_reach_a_scaffold():
    record = ProviderIntakeRecord.from_dict(
        ready_record(
            evidence=[
                {
                    "source_class": "x_claim",
                    "url": "https://x.com/example/status/1",
                    "observed_at": "2026-08-21",
                    "claim": "Claims free forever.",
                },
                {
                    "source_class": "directory_claim",
                    "url": "https://directory.example/free",
                    "observed_at": "2026-08-21",
                    "claim": "Listed as free.",
                },
            ]
        )
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "blocked"
    assert decision.sheet_ready is False
    assert decision.scaffold_ready is False
    assert decision.blockers == ("evidence",)


def test_runtime_probe_evidence_satisfies_the_official_evidence_gate():
    record = ProviderIntakeRecord.from_dict(
        ready_record(
            evidence=[
                {
                    "source_class": "runtime_probe",
                    "url": "https://api.teamorouter.com/v1/models",
                    "observed_at": "2026-08-21",
                    "claim": "Unauthenticated catalog fetch returned 401.",
                }
            ]
        )
    )

    assert evaluate_automation_readiness(record).action == "scaffold_ready"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_urls", ["https://example.com/docs?api_key=sensitive-value"]),
        ("official_urls", ["https://example.com/docs?access_token=sensitive-value"]),
        ("claimed_base_url", "https://api.example.com/v1?password=sensitive-value"),
        ("source_urls", ["https://example.com/docs?key=sensitive-value"]),
        ("official_urls", ["https://example.com/docs?auth=sensitive-value"]),
        ("claimed_base_url", "https://api.example.com/v1?authorization=sensitive-value"),
    ],
)
def test_provider_intake_rejects_secret_query_parameters(field, value):
    with pytest.raises(IntakeValidationError, match=field):
        ProviderIntakeRecord.from_dict(valid_record(**{field: value}))


def test_provider_intake_rejects_secret_query_parameters_in_evidence():
    raw = valid_record()
    raw["evidence"][0]["url"] = "https://example.com/docs?bearer_token=sensitive-value"

    with pytest.raises(IntakeValidationError, match="evidence url"):
        ProviderIntakeRecord.from_dict(raw)


def test_unqualified_provider_specific_record_cannot_generate_a_candidate_sheet():
    record = ProviderIntakeRecord.from_dict(
        ready_record(
            adapter_fit="provider_specific_runtime",
            official_urls=[],
            billing_cap_behavior="unknown",
        )
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "blocked"
    assert decision.sheet_ready is False
    assert decision.scaffold_ready is False
    assert decision.blockers == ("official_urls", "billing_cap_behavior")


@pytest.mark.parametrize(
    ("adapter_fit", "action"),
    [("unknown", "blocked"), ("not_integrable", "rejected")],
)
def test_nonintegrable_adapter_states_never_become_candidate_sheets(adapter_fit, action):
    decision = evaluate_automation_readiness(
        ProviderIntakeRecord.from_dict(ready_record(adapter_fit=adapter_fit))
    )

    assert decision.action == action
    assert decision.sheet_ready is False
    assert decision.scaffold_ready is False


def test_public_shared_key_requires_do_not_use_status():
    with pytest.raises(IntakeValidationError, match="public_shared_key.*do_not_use"):
        ProviderIntakeRecord.from_dict(
            valid_record(
                status="sheet_ready_candidate",
                free_mechanism="public_shared_key",
            )
        )


def test_allowlist_scaffold_requires_at_least_one_model_id():
    record = ProviderIntakeRecord.from_dict(
        ready_record(adapter_fit="generic_config_with_allowlist", model_ids=[])
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "blocked"
    assert decision.sheet_ready is False
    assert decision.scaffold_ready is False
    assert decision.blockers == ("model_ids",)


def test_generated_schema_rejects_whitespace_only_safe_text():
    pattern = generated_intake_schema()["properties"]["name"]["pattern"]

    assert re.fullmatch(pattern, "   ") is None
    assert re.fullmatch(pattern, "TeamoRouter") is not None


def test_generated_schema_flags_common_secret_query_parameters():
    secret_pattern = generated_intake_schema()["properties"]["source_urls"]["items"]["not"]["pattern"]

    assert re.search(secret_pattern, "https://example.com/docs?api_key=sensitive-value")
    assert re.search(secret_pattern, "https://example.com/docs?BearerToken=sensitive-value")
    assert re.search(secret_pattern, "https://example.com/docs#password=sensitive-value")
    assert re.search(secret_pattern, "https://example.com/docs?key=sensitive-value")
    assert re.search(secret_pattern, "https://example.com/docs?Authorization=sensitive-value")
    assert re.search(secret_pattern, "https://example.com/docs?language=fr") is None


def test_manual_adapter_generation_writes_sheet_only(tmp_path):
    record = ProviderIntakeRecord.from_dict(
        ready_record(adapter_fit="provider_specific_runtime")
    )

    generated = generate_artifacts(record, tmp_path)

    assert generated == {"provider_sheet": tmp_path / "teamorouter.md"}
    assert generated["provider_sheet"].read_text() == render_provider_sheet(record)
    assert not (tmp_path / "teamorouter.integration.json").exists()


def test_remote_keyless_provider_uses_the_anonymous_generic_contract():
    record = ProviderIntakeRecord.from_dict(
        ready_record(account_key_posture="not_required", auth_env="none")
    )

    decision = evaluate_automation_readiness(record)
    scaffold = build_integration_scaffold(record)

    assert decision.action == "scaffold_ready"
    assert decision.sheet_ready is True
    assert decision.scaffold_ready is True
    assert decision.blockers == ()
    assert scaffold["provider_config"]["auth_mode"] == "anonymous"
    assert scaffold["provider_config"]["activation_policy"] == "always"
    assert "credential_presence" not in scaffold["required_gates"]


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("canonical_chat_path", "/api/chat", "runtime:nonstandard_chat_path"),
        ("catalog_path", "/api/models", "runtime:nonstandard_catalog_path"),
    ],
)
def test_nonstandard_generic_paths_require_runtime_code(field, value, blocker):
    record = ProviderIntakeRecord.from_dict(
        ready_record(
            endpoint_shape="openai_compatible_nonstandard_path",
            **{field: value},
        )
    )

    decision = evaluate_automation_readiness(record)

    assert decision.action == "manual_code_required"
    assert decision.sheet_ready is True
    assert decision.scaffold_ready is False
    assert decision.blockers == (blocker,)


def test_canonical_record_ignores_set_like_ordering():
    first_raw = valid_record(
        aliases=["team", "router"],
        source_urls=["https://x.com/example/status/1", "https://x.com/example/status/2"],
        official_urls=["https://teamorouter.com/docs/rate-limits", "https://teamorouter.com/docs/api"],
        model_ids=["model-b", "model-a"],
        blockers=["tools", "billing"],
        evidence=[
            {
                "source_class": "official_docs",
                "url": "https://teamorouter.com/docs/rate-limits",
                "observed_at": "2026-08-21",
                "claim": "Rate limits.",
            },
            {
                "source_class": "x_claim",
                "url": "https://x.com/example/status/1",
                "observed_at": "2026-08-20",
                "claim": "Discovery.",
            },
        ],
    )
    second_raw = dict(first_raw)
    for field in ("aliases", "source_urls", "official_urls", "model_ids", "blockers", "evidence"):
        second_raw[field] = list(reversed(first_raw[field]))
    first = ProviderIntakeRecord.from_dict(first_raw)
    second = ProviderIntakeRecord.from_dict(second_raw)

    assert canonical_record_dict(first) == canonical_record_dict(second)
    change = compare_intake_records(first, second)
    assert change.action == "unchanged"
    assert change.changed_fields == ()


@pytest.mark.parametrize(
    ("before_status", "after_status"),
    [
        ("new_signal", "official_docs_found"),
        ("official_docs_found", "needs_account_check"),
        ("new_signal", "do_not_use"),
        ("needs_account_check", "closed"),
    ],
)
def test_compare_allows_forward_and_terminal_status_transitions(before_status, after_status):
    existing = ProviderIntakeRecord.from_dict(valid_record(status=before_status))
    proposed = ProviderIntakeRecord.from_dict(valid_record(status=after_status))

    change = compare_intake_records(existing, proposed)

    assert change.action == "changed"
    assert change.changed_fields == ("status",)


@pytest.mark.parametrize(
    ("before_status", "after_status"),
    [
        ("official_docs_found", "new_signal"),
        ("needs_account_check", "official_docs_found"),
        ("do_not_use", "official_docs_found"),
        ("closed", "new_signal"),
    ],
)
def test_compare_rejects_downgrades_and_terminal_reopening(before_status, after_status):
    existing = ProviderIntakeRecord.from_dict(valid_record(status=before_status))
    proposed = ProviderIntakeRecord.from_dict(valid_record(status=after_status))

    with pytest.raises(IntakeValidationError, match="status transition"):
        compare_intake_records(existing, proposed)


def test_compare_rejects_unqualified_sheet_ready_promotion():
    existing = ProviderIntakeRecord.from_dict(valid_record(status="needs_account_check"))
    proposed = ProviderIntakeRecord.from_dict(valid_record(status="sheet_ready_candidate"))

    with pytest.raises(IntakeValidationError, match="sheet_ready_candidate is not qualified"):
        compare_intake_records(existing, proposed)


def test_compare_rejects_erasing_prior_evidence_and_sources():
    existing = ProviderIntakeRecord.from_dict(
        valid_record(
            source_urls=["https://x.com/example/status/1", "https://x.com/example/status/2"],
            model_ids=["model-a", "model-b"],
            evidence=[
                {
                    "source_class": "official_docs",
                    "url": "https://teamorouter.com/docs/rate-limits",
                    "observed_at": "2026-08-21",
                    "claim": "Rate limits.",
                },
                {
                    "source_class": "x_claim",
                    "url": "https://x.com/example/status/2",
                    "observed_at": "2026-08-20",
                    "claim": "Discovery.",
                },
            ],
        )
    )
    proposed = ProviderIntakeRecord.from_dict(valid_record())

    with pytest.raises(IntakeValidationError, match="cannot erase prior"):
        compare_intake_records(existing, proposed)


def test_sync_creates_canonical_record_and_reports_new(tmp_path):
    record = ProviderIntakeRecord.from_dict(
        valid_record(aliases=["zeta", "alpha"], model_ids=["model-b", "model-a"])
    )

    result = sync_intake_record(record, tmp_path)

    target = tmp_path / "teamorouter.json"
    assert result.action == "new"
    assert result.changed_fields == tuple(canonical_record_dict(record))
    assert result.path == target
    assert result.written is True
    assert json.loads(target.read_text()) == canonical_record_dict(record)


def test_sync_rejects_new_unqualified_sheet_ready_record(tmp_path):
    registry = tmp_path / "registry"
    record = ProviderIntakeRecord.from_dict(
        valid_record(status="sheet_ready_candidate")
    )

    with pytest.raises(IntakeValidationError, match="sheet_ready_candidate is not qualified"):
        sync_intake_record(record, registry)

    assert not registry.exists()


def test_sync_does_not_rewrite_semantically_unchanged_record(tmp_path):
    first = ProviderIntakeRecord.from_dict(
        valid_record(aliases=["alpha", "zeta"], model_ids=["model-a", "model-b"])
    )
    sync_intake_record(first, tmp_path)
    target = tmp_path / "teamorouter.json"
    before = target.read_bytes()
    second = ProviderIntakeRecord.from_dict(
        valid_record(aliases=["zeta", "alpha"], model_ids=["model-b", "model-a"])
    )

    result = sync_intake_record(second, tmp_path)

    assert result.action == "unchanged"
    assert result.changed_fields == ()
    assert result.written is False
    assert target.read_bytes() == before


def test_sync_atomically_replaces_changed_record(tmp_path):
    existing = ProviderIntakeRecord.from_dict(valid_record())
    sync_intake_record(existing, tmp_path)
    proposed = ProviderIntakeRecord.from_dict(
        valid_record(status="needs_account_check", next_action="Run an authorized account check.")
    )

    result = sync_intake_record(proposed, tmp_path)

    assert result.action == "changed"
    assert result.changed_fields == ("status", "next_action")
    assert result.written is True
    assert json.loads(result.path.read_text()) == canonical_record_dict(proposed)
    assert not list(tmp_path.glob("*.tmp"))


def test_sync_dry_run_never_creates_registry_or_record(tmp_path):
    registry = tmp_path / "missing-registry"
    record = ProviderIntakeRecord.from_dict(valid_record())

    result = sync_intake_record(record, registry, dry_run=True)

    assert result.action == "new"
    assert result.written is False
    assert not registry.exists()


def test_sync_refuses_symlink_record(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n")
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "teamorouter.json").symlink_to(outside)

    with pytest.raises(IntakeValidationError, match="symlink"):
        sync_intake_record(ProviderIntakeRecord.from_dict(valid_record()), registry)

    assert outside.read_text() == "outside\n"


def test_sync_nofollow_read_survives_a_symlink_check_race(monkeypatch, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            canonical_record_dict(ProviderIntakeRecord.from_dict(valid_record())),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    registry = tmp_path / "registry"
    registry.mkdir()
    target = registry / "teamorouter.json"
    target.symlink_to(outside)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: False if path == target else original_is_symlink(path),
    )

    with pytest.raises(IntakeValidationError, match="symlink"):
        sync_intake_record(ProviderIntakeRecord.from_dict(valid_record()), registry)


def test_sync_rejects_stale_expected_digest(tmp_path):
    record = ProviderIntakeRecord.from_dict(valid_record())
    sync_intake_record(record, tmp_path)
    proposed = ProviderIntakeRecord.from_dict(
        valid_record(status="needs_account_check", next_action="Check account.")
    )

    with pytest.raises(IntakeValidationError, match="concurrent registry change"):
        sync_intake_record(proposed, tmp_path, expected_sha256="0" * 64)


def test_registry_context_is_compact_deterministic_and_bounded(tmp_path):
    teamo = ProviderIntakeRecord.from_dict(valid_record())
    token = ProviderIntakeRecord.from_dict(
        valid_record(
            provider_id="tokenrouter",
            name="TokenRouter",
            auth_env="TOKENROUTER_API_KEY",
            next_action="Resolve free caps.",
        )
    )
    sync_intake_record(token, tmp_path)
    sync_intake_record(teamo, tmp_path)

    context = build_registry_context(tmp_path, limit=1)

    assert context["schema_version"] == 1
    assert context["missing_sha256"] == "missing"
    assert context["total"] == 2
    assert context["returned"] == 1
    assert context["truncated"] is True
    assert [row["provider_id"] for row in context["providers"]] == ["teamorouter"]
    row = context["providers"][0]
    assert row == {
        "provider_id": "teamorouter",
        "name": "TeamoRouter",
        "status": "official_docs_found",
        "blockers": ["billing_cap_behavior", "commercial_use_posture"],
        "newest_evidence_at": "2026-08-21",
        "next_action": "Resolve billing and legal posture.",
        "sha256": hashlib.sha256((tmp_path / "teamorouter.json").read_bytes()).hexdigest(),
    }
    assert "evidence" not in row
    assert "source_urls" not in row
