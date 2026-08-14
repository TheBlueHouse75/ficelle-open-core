from __future__ import annotations

from typing import Any

import pytest

from ficelle.failures import DEFAULT_FAILURE_MARKERS
from ficelle.providers.base import (
    CatalogFetchContext,
    ProviderAccess,
    ProviderAccessContext,
    ProviderCatalogPolicy,
    ProviderNormalizedCatalogModel,
)
from ficelle.use_cases.provider_catalog import (
    ProviderCatalogPorts,
    fetch_provider_catalog,
    provider_access_result,
    provider_invocation_headers,
)


class FakeAdapter:
    source = "fake"

    def __init__(self) -> None:
        self.access_context: ProviderAccessContext | None = None
        self.fetch_context: CatalogFetchContext | None = None

    def request_headers(self) -> dict[str, str]:
        return {"X-Request": "1"}

    def invocation_headers(self) -> dict[str, str]:
        return {"X-Invoke": "1"}

    def failure_markers(self):
        return DEFAULT_FAILURE_MARKERS

    def safe_diagnostics(self, provider_cfg: dict[str, Any]) -> dict[str, Any]:
        return {"source": self.source, "catalog_url": provider_cfg.get("catalog_url")}

    def catalog_policy(self, provider_cfg: dict[str, Any]) -> ProviderCatalogPolicy:
        return ProviderCatalogPolicy(False, {}, [], [], False, {})

    def normalize_catalog_model(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> ProviderNormalizedCatalogModel:
        return ProviderNormalizedCatalogModel(model, model)

    def excludes_catalog_model(self, model: dict[str, Any], policy: ProviderCatalogPolicy) -> bool:
        return False

    def trusted_free_access(
        self,
        provider_cfg: dict[str, Any],
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> dict[str, Any] | None:
        return None

    def access(
        self,
        provider_cfg: dict[str, Any],
        context: ProviderAccessContext,
        *,
        require_base_url: bool,
    ) -> ProviderAccess:
        self.access_context = context
        key, reason = context.resolve_credentials(self.source, provider_cfg)
        external_key, _external_base_url, _external_reason = context.resolve_external_credentials(self.source, reason)
        return ProviderAccess(external_key or key, provider_cfg.get("base_url"), reason, require_base_url)

    def fetch_catalog(
        self,
        provider_cfg: dict[str, Any],
        context: CatalogFetchContext,
    ) -> tuple[list[dict[str, Any]], str | None]:
        self.fetch_context = context
        key, _reason = context.resolve_credentials(self.source, provider_cfg)
        response = context.http_get(provider_cfg["catalog_url"], timeout=context.timeout_seconds, key=key)
        return response["models"], None


def test_provider_catalog_use_case_delegates_headers_access_and_fetch_context() -> None:
    adapter = FakeAdapter()
    credential_calls: list[tuple[str, dict[str, Any]]] = []
    external_calls: list[tuple[str, str]] = []
    http_calls: list[tuple[str, dict[str, Any]]] = []

    def resolve_credentials(source: str, provider_cfg: dict[str, Any]) -> tuple[str | None, str]:
        credential_calls.append((source, provider_cfg))
        return "generic-key", "generic"

    def resolve_external_credentials(source: str, reason: str) -> tuple[str | None, str | None, str]:
        external_calls.append((source, reason))
        return "external-key", None, "external"

    def http_get(url: str, **kwargs: Any) -> dict[str, Any]:
        http_calls.append((url, kwargs))
        return {"models": [{"id": "m1"}]}

    ports = ProviderCatalogPorts(
        provider_catalog_adapter=lambda _source: adapter,
        resolve_credentials=resolve_credentials,
        resolve_external_credentials=resolve_external_credentials,
        http_get=http_get,
    )
    config = {
        "catalog_timeout_seconds": 7,
        "providers": {"fake": {"catalog_url": "https://fake.test/models", "base_url": "https://fake.test/v1"}},
    }

    assert provider_invocation_headers("fake", ports=ports) == {"X-Invoke": "1"}
    access = provider_access_result("fake", config["providers"]["fake"], require_base_url=True, ports=ports)
    models, error = fetch_provider_catalog("fake", config, ports=ports)

    assert (access.key, access.base_url, access.reason, access.auth_status_invokable) == (
        "external-key",
        "https://fake.test/v1",
        "generic",
        True,
    )
    assert models == [{"id": "m1"}]
    assert error is None
    assert credential_calls == [
        ("fake", config["providers"]["fake"]),
        ("fake", config["providers"]["fake"]),
    ]
    assert external_calls == [("fake", "generic")]
    assert http_calls == [("https://fake.test/models", {"timeout": 7.0, "key": "generic-key"})]
    assert adapter.access_context is not None
    assert adapter.fetch_context is not None


# --- Per-family catalog overrides -----------------------------------------------------------


def _adapter_for(provider_cfg):
    from ficelle.providers.openai_compatible import OpenAICompatibleCatalogAdapter

    adapter = OpenAICompatibleCatalogAdapter("groq")
    return adapter, adapter.catalog_policy(provider_cfg)


def test_family_override_narrows_only_the_rows_it_names():
    # Groq publishes no per-model supported_parameters, so the provider-wide default would
    # claim tools for every row — including compound, which documents that it accepts no
    # user-provided tools.
    provider_cfg = {
        "catalog_model_defaults": {
            "context_length": 128_000,
            "supported_parameters": ["tools", "tool_choice", "response_format"],
        },
        "catalog_model_overrides": {"groq/compound": {"supported_parameters": []}},
    }
    adapter, policy = _adapter_for(provider_cfg)

    compound = adapter.normalize_catalog_model({"id": "groq/compound"}, policy).trusted_model
    compound_mini = adapter.normalize_catalog_model({"id": "groq/compound-mini"}, policy).trusted_model
    ordinary = adapter.normalize_catalog_model({"id": "llama-3.3-70b-versatile"}, policy).trusted_model

    # The corrected family loses the claim entirely, and keeps the rest of the defaults.
    assert compound["supported_parameters"] == []
    assert compound_mini["supported_parameters"] == []
    assert compound["context_length"] == 128_000
    # Every other model keeps the provider-wide default untouched.
    assert ordinary["supported_parameters"] == ["tools", "tool_choice", "response_format"]


def test_a_models_own_declaration_still_wins_over_an_override():
    provider_cfg = {
        "catalog_model_defaults": {"supported_parameters": ["tools"]},
        "catalog_model_overrides": {"groq/compound": {"supported_parameters": []}},
    }
    adapter, policy = _adapter_for(provider_cfg)

    declared = adapter.normalize_catalog_model(
        {"id": "groq/compound", "supported_parameters": ["tools", "tool_choice"]}, policy
    ).trusted_model

    # A provider that starts publishing real per-model metadata is believed over our guess.
    assert declared["supported_parameters"] == ["tools", "tool_choice"]


def test_the_most_specific_override_wins_and_unknown_fields_are_dropped():
    provider_cfg = {
        "catalog_model_defaults": {"supported_parameters": ["tools"], "context_length": 8_000},
        "catalog_model_overrides": {
            "groq/compound": {"context_length": 32_000},
            "groq/compound-mini": {"context_length": 16_000, "api_key": "leak", "unknown_field": 1},
        },
    }
    adapter, policy = _adapter_for(provider_cfg)

    mini = adapter.normalize_catalog_model({"id": "groq/compound-mini"}, policy).trusted_model
    full = adapter.normalize_catalog_model({"id": "groq/compound"}, policy).trusted_model

    assert mini["context_length"] == 16_000, "the longer, more specific pattern wins"
    assert full["context_length"] == 32_000
    # Same whitelist as the provider-wide defaults: nothing outside trusted fields survives.
    assert "api_key" not in mini and "unknown_field" not in mini


def test_shipped_groq_pack_declares_compound_without_tools():
    # This file ships to the core-only mirror, where the closed pack is absent by
    # construction — the one pack-coupled assertion skips there instead of failing.
    provider_pack = pytest.importorskip(
        "ficelle_pro.provider_pack", reason="ficelle-pro not installed (core-only environment)"
    )

    groq = provider_pack.PROVIDER_PACK["groq"]

    assert groq["catalog_model_overrides"]["groq/compound"]["supported_parameters"] == []
