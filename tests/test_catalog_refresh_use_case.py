from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ficelle.providers.base import ProviderCatalogPolicy, ProviderNormalizedCatalogModel
from ficelle.use_cases.catalog_refresh import (
    CatalogRefreshPorts,
    CatalogRefreshRunner,
    StateMutator,
    load_or_refresh_catalog,
    refresh_catalog,
)


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _load_catalog(catalog: Any):
    def load_json(_path: Path, _default: Any) -> Any:
        return catalog

    return load_json


def _fingerprint(config: dict[str, Any]) -> str:
    return str(config.get("fingerprint") or "fp-1")


class _Adapter:
    def safe_diagnostics(self, provider_cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "adapter": "fake",
            "catalog_url": provider_cfg.get("catalog_url"),
            "source_type": "openai_compatible",
            "provider_class": "strict_zero",
            "free_mode": "catalog_free",
            "free_scope": "model",
            "activation_policy": provider_cfg.get("activation_policy", "always"),
            "quota_reset_policy": None,
        }

    def catalog_policy(self, _provider_cfg: dict[str, Any]) -> ProviderCatalogPolicy:
        return ProviderCatalogPolicy(
            has_trusted_free_access=False,
            model_defaults={},
            model_id_exclude_patterns=[],
            model_id_allowlist=[],
            requires_exact_model_allowlist=False,
            rejection_counters={
                "invalid": 0,
                "not_chat": 0,
                "not_eligible": 0,
                "unsafe_pricing": 0,
                "paid": 0,
                "no_tools": 0,
                "small_context": 0,
            },
        )

    def normalize_catalog_model(
        self,
        model: dict[str, Any],
        _policy: ProviderCatalogPolicy,
    ) -> ProviderNormalizedCatalogModel:
        return ProviderNormalizedCatalogModel(normalized_model=dict(model), trusted_model=dict(model))

    def trusted_free_access(
        self,
        _provider_cfg: dict[str, Any],
        _model: dict[str, Any],
        _policy: ProviderCatalogPolicy,
    ) -> dict[str, Any] | None:
        return None

    def excludes_catalog_model(self, _model: dict[str, Any], _policy: ProviderCatalogPolicy) -> bool:
        return False


def _ports(raw_models: list[dict[str, Any]]) -> CatalogRefreshPorts:
    return CatalogRefreshPorts(
        auth_status=lambda _config: {"fake": {"invokable": True, "reason": "configured", "base_url": "https://fake.example/v1"}},
        provider_catalog_adapter=lambda _source: _Adapter(),
        fetch_provider_catalog=lambda _source, _config: (raw_models, None),
        strict_zero_pricing=lambda _pricing: (True, "all exposed pricing fields are numeric zero", {"status": "safe"}),
        normalized_free_access=lambda _model, _pricing_ok, _pricing_reason: {
            "eligible": True,
            "mode": "catalog_free",
            "proof": "provider_pricing",
            "scope": "model",
            "status": "available",
        },
        pricing_safety_for_free_access=lambda pricing_safety, _free_access: pricing_safety,
        context_length_provenance=lambda declared, _context, _upstream: ("catalog" if declared else "default", None),
        catalog_config_fingerprint=lambda _config: "fingerprint",
        catalog_config_structural_fingerprint=lambda _config: "structural-fingerprint",
        now_iso=lambda: "2026-06-21T22:00:00+00:00",
        safe_detail=lambda value: "" if value is None else str(value),
        safe_int=lambda value, default: int(value) if value is not None else default,
        safe_float=lambda value, default: float(value) if value is not None else default,
        safe_optional_int=lambda value: int(value) if value not in (None, "") else None,
        safe_optional_bool=lambda value: value if isinstance(value, bool) else None,
        safe_string_list=lambda value: [str(item) for item in value] if isinstance(value, list) else [],
        has_tools=lambda params: "tools" in {str(item) for item in params},
        has_structured=lambda params: "response_format" in {str(item) for item in params},
        concrete_model_id=lambda source, upstream_id: f"ficelle/{source}/{upstream_id}",
        capabilities_from_defaults=lambda _model, _defaults: [],
        capabilities_from_reference=lambda _defaults, _upstream_id: [],
        capabilities_refuted_by_reference=lambda _defaults, _upstream_id: [],
        model_reference_confidence=lambda _upstream_id: "",
        provider_key_url=lambda _source: None,
    )


def test_load_or_refresh_catalog_returns_valid_cache(tmp_path):
    cached = {
        "models": [{"id": "ficelle/openrouter/free"}],
        "config_fingerprint": "fp-1",
        "generated_at": "2026-06-21T22:00:00+00:00",
    }
    refresh_calls = 0

    def refresh_catalog(_config: dict[str, Any]) -> dict[str, Any]:
        nonlocal refresh_calls
        refresh_calls += 1
        return {"models": []}

    result = load_or_refresh_catalog(
        {"fingerprint": "fp-1", "catalog_ttl_seconds": 3600},
        catalog_path=tmp_path / "catalog.json",
        load_json=_load_catalog(cached),
        refresh_catalog=refresh_catalog,
        catalog_config_fingerprint=_fingerprint,
        now_seconds=lambda: _timestamp("2026-06-21T22:10:00+00:00"),
    )

    assert result == cached
    assert refresh_calls == 0


def test_load_or_refresh_catalog_refreshes_invalid_cache(tmp_path):
    refreshed = {"models": [{"id": "fresh"}]}

    result = load_or_refresh_catalog(
        {"fingerprint": "fp-1"},
        catalog_path=tmp_path / "catalog.json",
        load_json=_load_catalog({}),
        refresh_catalog=lambda _config: refreshed,
        catalog_config_fingerprint=_fingerprint,
        now_seconds=lambda: 0.0,
    )

    assert result == refreshed


def test_load_or_refresh_catalog_refreshes_when_forced_or_stale(tmp_path):
    cached = {
        "models": [{"id": "cached"}],
        "config_fingerprint": "fp-1",
        "generated_at": "2026-06-21T22:00:00+00:00",
    }
    refreshed = {"models": [{"id": "fresh"}]}

    forced = load_or_refresh_catalog(
        {"fingerprint": "fp-1"},
        catalog_path=tmp_path / "catalog.json",
        load_json=_load_catalog(cached),
        refresh_catalog=lambda _config: refreshed,
        catalog_config_fingerprint=_fingerprint,
        now_seconds=lambda: _timestamp("2026-06-21T22:10:00+00:00"),
        force=True,
    )
    stale = load_or_refresh_catalog(
        {"fingerprint": "fp-1", "catalog_ttl_seconds": 1},
        catalog_path=tmp_path / "catalog.json",
        load_json=_load_catalog(cached),
        refresh_catalog=lambda _config: refreshed,
        catalog_config_fingerprint=_fingerprint,
        now_seconds=lambda: _timestamp("2026-06-21T23:00:00+00:00"),
    )

    assert forced == refreshed
    assert stale == refreshed


def test_load_or_refresh_catalog_refreshes_on_fingerprint_or_timestamp_mismatch(tmp_path):
    cached = {
        "models": [{"id": "cached"}],
        "config_fingerprint": "fp-1",
        "generated_at": "not-a-date",
    }
    refreshed = {"models": [{"id": "fresh"}]}

    fingerprint_changed = load_or_refresh_catalog(
        {"fingerprint": "fp-2"},
        catalog_path=tmp_path / "catalog.json",
        load_json=_load_catalog({**cached, "generated_at": "2026-06-21T22:00:00+00:00"}),
        refresh_catalog=lambda _config: refreshed,
        catalog_config_fingerprint=_fingerprint,
        now_seconds=lambda: _timestamp("2026-06-21T22:10:00+00:00"),
    )
    malformed_timestamp = load_or_refresh_catalog(
        {"fingerprint": "fp-1"},
        catalog_path=tmp_path / "catalog.json",
        load_json=_load_catalog(cached),
        refresh_catalog=lambda _config: refreshed,
        catalog_config_fingerprint=_fingerprint,
        now_seconds=lambda: _timestamp("2026-06-21T22:10:00+00:00"),
    )

    assert fingerprint_changed == refreshed
    assert malformed_timestamp == refreshed


def test_refresh_catalog_persists_catalog_and_refresh_state(tmp_path):
    writes: list[tuple[Path, dict[str, Any]]] = []
    update_reasons: list[str | None] = []
    updated_state: dict[str, Any] = {}

    def atomic_write_json(path: Path, data: Any) -> None:
        assert isinstance(data, dict)
        writes.append((path, data))

    def update_state(mutator: StateMutator, reason: str | None = None) -> dict[str, Any]:
        update_reasons.append(reason)
        mutator(updated_state)
        return updated_state

    catalog = refresh_catalog(
        {
            "allow_paid_fallback": False,
            "min_context_length": 128000,
            "providers": {
                "fake": {
                    "enabled": True,
                    "base_url": "https://fake.example/v1",
                }
            },
        },
        ports=_ports(
            [
                {
                    "id": "free-model",
                    "name": "Free Model",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 200000,
                    "supported_parameters": ["tools"],
                }
            ]
        ),
        virtual_models={"ficelle/auto-fast"},
        catalog_path=tmp_path / "catalog.json",
        atomic_write_json=atomic_write_json,
        update_state=update_state,
    )

    assert writes == [(tmp_path / "catalog.json", catalog)]
    assert update_reasons == ["refresh_catalog"]
    assert updated_state == {
        "cooldowns": {},
        "quota_cooldowns": {},
        "quota_probe_results": {},
        "quarantine": {},
        "last_catalog_refresh_at": catalog["generated_at"],
    }


def test_catalog_refresh_runner_builds_catalog_and_dedupes_model_ids():
    runner = CatalogRefreshRunner(
        _ports([
            {
                "id": "model-a",
                "name": "Model A",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 131072,
                "supported_parameters": ["tools", "response_format"],
            },
            {
                "id": "model-a",
                "name": "Model A Duplicate",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 131072,
                "supported_parameters": ["tools"],
            },
        ]),
        virtual_models={"ficelle/auto-fast"},
    )

    catalog = runner.refresh_catalog(
        {
            "allow_paid_fallback": False,
            "min_context_length": 128000,
            "providers": {"fake": {"enabled": True, "catalog_url": "https://fake.example/models"}},
        }
    )

    assert catalog["generated_at"] == "2026-06-21T22:00:00+00:00"
    assert catalog["config_fingerprint"] == "fingerprint"
    assert catalog["config_structural_fingerprint"] == "structural-fingerprint"
    assert [model["id"] for model in catalog["models"]] == ["ficelle/fake/model-a"]
    assert catalog["models"][0]["supports_structured_outputs"] is True
    assert catalog["providers"]["fake"]["raw_count"] == 2
    assert catalog["providers"]["fake"]["accepted_count"] == 1
    assert [row["status"] for row in catalog["providers"]["fake"]["models"]] == ["accepted", "accepted"]


def test_refresh_catalog_fetches_providers_concurrently():
    """A slow/down provider must not delay the others: provider catalog fetches run
    in parallel, so the refresh observes more than one fetch in flight at once.
    Regression for the startup/refresh hang caused by a single down provider."""
    import dataclasses
    import threading
    import time

    sources = [f"p{index}" for index in range(4)]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def _slow_fetch(source: str, _config: dict[str, Any]):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.2)  # stand-in for a slow/hanging provider endpoint
        with lock:
            active -= 1
        return (
            [
                {
                    "id": f"{source}-model",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 131072,
                    "supported_parameters": ["tools"],
                }
            ],
            None,
        )

    ports = dataclasses.replace(
        _ports([]),
        auth_status=lambda _config: {
            source: {"invokable": True, "reason": "configured", "base_url": "https://fake/v1"}
            for source in sources
        },
        fetch_provider_catalog=_slow_fetch,
    )
    runner = CatalogRefreshRunner(ports, virtual_models={"ficelle/auto-fast"})
    catalog = runner.refresh_catalog(
        {
            "allow_paid_fallback": False,
            "min_context_length": 128000,
            "providers": {
                source: {"enabled": True, "catalog_url": f"https://fake/{source}/models"}
                for source in sources
            },
        }
    )

    # Every provider's model still made it into the catalog...
    assert len(catalog["models"]) == len(sources)
    # ...and the fetches overlapped, proving one slow provider can't serialize the rest.
    assert max_active >= 2, f"expected concurrent provider fetches, max in-flight was {max_active}"
