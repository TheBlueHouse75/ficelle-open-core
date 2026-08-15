from __future__ import annotations

import json
from typing import Any

from ficelle.providers.base import ProviderAccess
from ficelle.use_cases.provider_auth import (
    ProviderAuthPorts,
    auth_row,
    auth_status,
    invokable_provider_sources,
    provider_auth_row,
    provider_key_setup_commands,
    unconfigured_provider_sources,
    unreadable_provider_reasons,
    unusable_key_provider_block,
    unusable_key_provider_lines,
    unusable_key_provider_reasons,
)


def ports_for(
    access_by_source: dict[str, ProviderAccess],
    calls: list[tuple[str, dict[str, Any], bool]],
) -> ProviderAuthPorts:
    def provider_access(source: str, provider_cfg: dict[str, Any], require_base_url: bool) -> ProviderAccess:
        calls.append((source, provider_cfg, require_base_url))
        return access_by_source[source]

    return ProviderAuthPorts(
        provider_access=provider_access,
        credential_source_label=lambda reason: {
            "env:NVIDIA_API_KEY": "env",
            "env:KEYLESS_API_KEY": "env",
            "dotenv:NVIDIA_API_KEY": ".env",
        }.get(str(reason)),
    )


def test_auth_row_redacts_key_and_maps_key_source() -> None:
    row = auth_row(
        True,
        "secret-key",
        "env:NVIDIA_API_KEY",
        "https://nvidia.example/v1",
        credential_source_label=lambda reason: "env" if reason == "env:NVIDIA_API_KEY" else None,
    )

    assert row == {
        "invokable": True,
        "reason": "configured",
        "key_source": "env",
        "base_url": "https://nvidia.example/v1",
    }
    assert "secret-key" not in repr(row)


def test_auth_row_keeps_the_key_source_of_a_key_that_cannot_be_used() -> None:
    """"No key" and "a key that cannot be used" are different answers with different fixes.

    They used to collapse into the same row, so every reader saw `key_source: null` and
    offered `ficelle set-key` for a key that was already stored and resolving.
    """
    row = auth_row(
        False,
        "secret-key",
        "missing base_url for provider openrouter",
        None,
        credential_source_label=lambda reason: "env" if reason == "env:OPENROUTER_API_KEY" else None,
        key_reason="env:OPENROUTER_API_KEY",
    )

    assert row == {
        "invokable": False,
        # Not "configured": that word belongs to a provider a request would reach.
        "reason": "missing base_url for provider openrouter",
        "key_source": "env",
        "base_url": None,
    }
    assert "secret-key" not in json.dumps(row)
    assert "secret-key" not in repr(row)


def test_auth_row_still_reports_no_source_when_no_key_resolved() -> None:
    row = auth_row(
        False,
        None,
        "missing OPENROUTER_API_KEY",
        "https://openrouter.example/v1",
        credential_source_label=lambda _reason: "env",
        key_reason="env:OPENROUTER_API_KEY",
    )

    assert row["key_source"] is None
    assert row["reason"] == "missing OPENROUTER_API_KEY"


def test_provider_auth_row_requires_key_and_base_url_unless_adapter_marks_invokable() -> None:
    calls: list[tuple[str, dict[str, Any], bool]] = []
    config = {
        "providers": {
            "nvidia": {"base_url": "https://nvidia.example/v1"},
            "keyless": {"base_url": "http://127.0.0.1:9000/v1"},
            "localai": {"base_url": "http://127.0.0.1:8080/v1"},
        }
    }
    ports = ports_for(
        {
            "nvidia": ProviderAccess("key", "https://nvidia.example/v1", "env:NVIDIA_API_KEY"),
            # A key with no base_url is NOT invokable, whoever asks. This slot used to hold
            # openrouter with `auth_status_invokable=True` — a shape no adapter produces any
            # more, and one that spelled out under a provider's name the exemption that made
            # the status row advertise providers `invoke_model` refuses.
            "keyless": ProviderAccess("key-only", None, "env:KEYLESS_API_KEY"),
            "localai": ProviderAccess(
                None,
                "http://127.0.0.1:8080/v1",
                "keyless_local",
                auth_status_invokable=True,
            ),
        },
        calls,
    )

    assert provider_auth_row("nvidia", config, ports=ports) == {
        "invokable": True,
        "reason": "configured",
        "key_source": "env",
        "base_url": "https://nvidia.example/v1",
    }
    assert provider_auth_row("keyless", config, ports=ports)["invokable"] is False
    local = provider_auth_row("localai", config, ports=ports)
    assert local["invokable"] is True
    assert local["reason"] == "keyless_local"
    assert local["key_source"] is None
    assert calls[0] == ("nvidia", {"base_url": "https://nvidia.example/v1"}, False)


def test_provider_auth_row_names_the_store_of_a_key_the_adapter_cannot_use() -> None:
    """The adapter spends `reason` on the diagnostic, so `key_reason` carries the store.

    Without it `credential_source_label` is handed `missing base_url ...`, answers `None` by
    design, and the row goes back to claiming the provider has no key at all.
    """
    config = {"providers": {"keyless": {}}}
    ports = ports_for(
        {
            "keyless": ProviderAccess(
                "key-only",
                None,
                "missing base_url for provider keyless",
                key_reason="env:KEYLESS_API_KEY",
            )
        },
        [],
    )

    row = provider_auth_row("keyless", config, ports=ports)

    assert row == {
        "invokable": False,
        "reason": "missing base_url for provider keyless",
        "key_source": "env",
        "base_url": None,
    }
    assert "key-only" not in json.dumps(row)


def test_auth_status_preserves_rows_for_each_provider_entry() -> None:
    calls: list[tuple[str, dict[str, Any], bool]] = []
    config = {"providers": {"nvidia": {"base_url": "https://nvidia.example/v1"}, "missing": "bad"}}
    ports = ports_for(
        {
            "nvidia": ProviderAccess(None, None, "missing NVIDIA_API_KEY"),
            "missing": ProviderAccess(None, None, "missing provider"),
        },
        calls,
    )

    status = auth_status(config, ports=ports)

    assert status["nvidia"] == {
        "invokable": False,
        "reason": "missing NVIDIA_API_KEY",
        "key_source": None,
        "base_url": "https://nvidia.example/v1",
    }
    assert status["missing"]["reason"] == "missing provider"


# --- first-run credential guidance ----------------------------------------------


def test_invokable_sources_split_what_can_serve_from_what_cannot() -> None:
    """`invokable` is the servability signal, not the presence of a row or a base URL."""
    auth = {
        "openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY", "base_url": "https://o.test/v1"},
        "nous": {"invokable": True, "reason": "configured", "key_source": "env"},
        # A keyless local provider serves without a key, so it must not be advertised
        # as something the user still has to configure.
        "localai": {"invokable": True, "reason": "keyless_local", "key_source": None},
        "broken": "not a row",
    }

    assert invokable_provider_sources(auth) == ["nous", "localai"]
    assert unconfigured_provider_sources(auth) == ["openrouter", "broken"]


def test_no_provider_is_invokable_on_a_fresh_keyless_install() -> None:
    auth = {
        "openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY"},
        "nous": {"invokable": False, "reason": "missing NOUS_API_KEY"},
    }

    assert invokable_provider_sources(auth) == []
    assert unconfigured_provider_sources(auth) == ["openrouter", "nous"]


def test_an_unreadable_store_is_not_classified_as_an_unconfigured_key() -> None:
    reason = "unreadable OPENROUTER_API_KEY from wincred: CredReadW error 1312"
    auth = {
        "openrouter": {"invokable": False, "reason": reason, "key_source": None},
        "nous": {"invokable": False, "reason": "missing NOUS_API_KEY", "key_source": None},
    }

    assert unreadable_provider_reasons(auth) == {"openrouter": reason}
    assert unconfigured_provider_sources(auth) == ["nous"]


def test_a_stored_key_that_cannot_be_used_is_not_something_to_go_and_store() -> None:
    """`ficelle set-key` is the wrong answer for a key that already resolves.

    The two buckets partition "not invokable" — every non-invokable provider lands in
    exactly one of them, so no surface can lose a provider by reading only one.
    """
    auth = {
        "openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY", "key_source": None},
        "mistral": {
            "invokable": False,
            "reason": "missing base_url for provider mistral",
            "key_source": "keychain",
        },
        "nous": {"invokable": True, "reason": "configured", "key_source": "env"},
        "broken": "not a row",
    }

    assert invokable_provider_sources(auth) == ["nous"]
    assert unconfigured_provider_sources(auth) == ["openrouter", "broken"]
    assert unusable_key_provider_reasons(auth) == {"mistral": "missing base_url for provider mistral"}
    assert sorted(
        [*invokable_provider_sources(auth), *unconfigured_provider_sources(auth), *unusable_key_provider_reasons(auth)]
    ) == sorted(auth)


def test_an_unusable_key_falls_back_to_a_statement_when_the_row_has_no_reason() -> None:
    reasons = unusable_key_provider_reasons({"mistral": {"invokable": False, "key_source": "env"}})

    assert reasons == {"mistral": "the stored key cannot be used"}


def test_unusable_key_lines_align_and_quote_the_row_reason_verbatim() -> None:
    """Reworded advice is advice that drifts: the reason is what the dashboard badge shows."""
    lines = unusable_key_provider_lines(
        {
            "mistral": "missing base_url for provider mistral",
            "openrouter": "missing base_url for provider openrouter",
        }
    )

    assert lines == [
        "mistral     # missing base_url for provider mistral",
        "openrouter  # missing base_url for provider openrouter",
    ]
    assert unusable_key_provider_lines({}) == []


def test_the_unusable_key_block_is_owned_once_so_two_surfaces_cannot_drift() -> None:
    """Setup and the demo print this sentence; the keyless half's framing differs per surface."""
    assert unusable_key_provider_block({"mistral": "missing base_url for provider mistral"}) == [
        "These providers already hold a key, and something else is missing:",
        "",
        "  mistral  # missing base_url for provider mistral",
        "",
    ]
    # No heading over an empty list: a surface with nothing to report prints nothing.
    assert unusable_key_provider_block({}) == []


def test_setup_commands_take_every_url_from_the_injected_registry() -> None:
    lines = provider_key_setup_commands(
        ["openrouter", "nous"],
        {"openrouter": "https://openrouter.example/keys", "nous": "https://nous.example/"},
    )

    assert lines == [
        "ficelle set-key openrouter  # create a key at https://openrouter.example/keys",
        "ficelle set-key nous        # create a key at https://nous.example/",
    ]


def test_setup_commands_still_name_a_provider_the_registry_does_not_know() -> None:
    """The command is the actionable half; a missing link must not drop the provider."""
    assert provider_key_setup_commands(["mystery"], {}) == ["ficelle set-key mystery"]
    assert provider_key_setup_commands([], {"openrouter": "https://o.test/keys"}) == []
