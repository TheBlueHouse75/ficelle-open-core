from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ficelle.providers.base import ProviderCatalogPolicy, ProviderNormalizedCatalogModel
from ficelle.use_cases.catalog_refresh import (
    CatalogRefreshPorts,
    CatalogRefreshRunner,
    StateMutator,
    decide_catalog_publish,
    load_or_refresh_catalog,
    previous_catalog_baseline,
    publish_catalog,
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
    def __init__(self, reference_prices: dict[str, dict[str, Any]] | None = None):
        self.reference_prices = reference_prices or {}

    def reference_prices_for_free_models(self, _raw_models: list[Any]) -> dict[str, dict[str, Any]]:
        return dict(self.reference_prices)

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


def _ports(
    raw_models: list[dict[str, Any]],
    reference_prices: dict[str, dict[str, Any]] | None = None,
) -> CatalogRefreshPorts:
    return CatalogRefreshPorts(
        auth_status=lambda _config: {"fake": {"invokable": True, "reason": "configured", "base_url": "https://fake.example/v1"}},
        provider_catalog_adapter=lambda _source: _Adapter(reference_prices),
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


def test_refresh_stamps_the_adapter_reference_prices_on_accepted_rows(tmp_path):
    reference = {"google/gemma:free": {"prompt": 1.5e-06, "completion": 2e-06, "basis": "fake:google/gemma"}}
    catalog = refresh_catalog(
        {
            "allow_paid_fallback": False,
            "min_context_length": 128000,
            "providers": {"fake": {"enabled": True, "base_url": "https://fake.example/v1"}},
        },
        ports=_ports(
            [
                {
                    "id": "google/gemma:free",
                    "name": "Gemma (free)",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 200000,
                    "supported_parameters": ["tools"],
                },
                {
                    "id": "other-free",
                    "name": "Other",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 200000,
                    "supported_parameters": ["tools"],
                },
            ],
            reference_prices=reference,
        ),
        virtual_models={"ficelle/auto-fast"},
        catalog_path=tmp_path / "catalog.json",
        load_json=lambda _path, default: default,
        atomic_write_json=lambda _path, _data: None,
        update_state=lambda mutator, reason=None: mutator({}) or {},
    )

    by_upstream = {model["upstream_id"]: model for model in catalog["models"]}
    assert by_upstream["google/gemma:free"]["reference_pricing"] == reference["google/gemma:free"]
    assert by_upstream["other-free"]["reference_pricing"] is None


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
        load_json=lambda _path, default: default,
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
        # No catalog on disk to replace, so there is nothing to check a later listing
        # against. Recorded as empty rather than skipped — absent and "provider listed
        # nothing" must not be readable as the same thing.
        "previous_catalog_provider_raw_counts": {},
        "catalog_provider_raw_counts": {
            "generated_at": catalog["generated_at"],
            "counts": {"fake": 1},
        },
    }


def test_publish_records_the_sizes_of_the_catalog_it_replaces(tmp_path):
    """The catalog file is overwritten on every publish; its sizes have to survive it.

    Nothing that runs *between* two refreshes — the dashboard above all — can otherwise
    tell a provider that dropped a model from one that answered with a truncated list.
    """
    state: dict[str, Any] = {}
    disk: dict[str, Any] = {}
    path = tmp_path / "catalog.json"

    def update_state(mutator: StateMutator, _reason: str | None = None) -> dict[str, Any]:
        mutator(state)
        return state

    def publish(generated_at: str, providers: dict[str, Any]) -> dict[str, Any]:
        catalog = {"generated_at": generated_at, "providers": providers}
        publish_catalog(
            catalog,
            catalog_path=path,
            load_json=lambda p, default: disk.get(str(p), default),
            atomic_write_json=lambda p, data: disk.__setitem__(str(p), data),
            update_state=update_state,
        )
        return catalog

    first = publish("2026-06-21T22:00:00+00:00", {"openrouter": {"raw_count": 400}})
    assert previous_catalog_baseline(state, first) == {}, "the first publish replaces nothing"

    second = publish("2026-06-21T23:00:00+00:00", {"openrouter": {"raw_count": 100}})
    assert previous_catalog_baseline(state, second) == {
        "providers": {"openrouter": {"raw_count": 400}}
    }
    # The baseline belongs to one catalog generation and is refused for any other. A publish
    # writes the catalog file before the state, and readers take no catalog lock, so holding
    # a catalog the state has not caught up with is a real window — one where a truncated
    # listing measured against a two-generations-old count looks complete.
    assert previous_catalog_baseline(state, first) == {}

    # A provider that errored leaves a hole, not a zero: recording 0 would make the next
    # listing look like a recovery and re-arm every conclusion the guard exists to block.
    third = publish("2026-06-22T00:00:00+00:00", {"openrouter": {"raw_count": 0, "error": "HTTP 503"}})
    assert previous_catalog_baseline(state, third) == {
        "providers": {"openrouter": {"raw_count": 100}}
    }
    fourth = publish("2026-06-22T01:00:00+00:00", {"openrouter": {"raw_count": 100}})
    assert previous_catalog_baseline(state, fourth) == {}

    # The baseline is read off the file being replaced, not carried forward from the last
    # publish's own state entry, so a wiped state cannot leave a later refresh comparing
    # against a generation that is quietly two old.
    state.clear()
    fifth = publish("2026-06-22T02:00:00+00:00", {"openrouter": {"raw_count": 50}})
    assert previous_catalog_baseline(state, fifth) == {
        "providers": {"openrouter": {"raw_count": 100}}
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


# --- Lot 1 (synthetic-health remediation): publish decision ---------------------------------


def _decision_row(
    source: str,
    upstream_id: str,
    *,
    eligible: bool = True,
    tools: bool = True,
    context_length: int = 131072,
    stale_since: str | None = None,
) -> dict[str, Any]:
    row = {
        "id": f"ficelle/{source}/{upstream_id}",
        "source": source,
        "upstream_id": upstream_id,
        "invokable": True,
        "supports_tools": tools,
        "context_length": context_length,
        "free_access": {
            "eligible": eligible,
            "mode": "catalog_free",
            "proof": "provider_pricing",
            "status": "available",
            "scope": "model",
        },
    }
    if stale_since is not None:
        row["catalog_stale"] = True
        row["stale_since"] = stale_since
    return row


def _decision_summary(
    result_class: str,
    *,
    raw_count: int = 0,
    accepted_count: int = 0,
    error: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "result_class": result_class,
        "raw_count": raw_count,
        "accepted_count": accepted_count,
        "error": error,
        "enabled": enabled,
        "rejected": {},
        "models": [],
    }


def _decision_config(sources: list[str], *, min_context: int = 128000) -> dict[str, Any]:
    return {
        "min_context_length": min_context,
        "providers": {source: {"enabled": True} for source in sources},
    }


def test_all_transient_failures_reject_destructive_empty_publish():
    cached = {
        "generated_at": "2026-08-08T09:00:00+00:00",
        "models": [_decision_row("alpha", f"model-{index}") for index in range(225)],
    }
    refreshed = {
        "generated_at": "2026-08-08T10:00:00+00:00",
        "models": [],
        "providers": {"alpha": _decision_summary("transient_error", error="Timeout: timed out")},
    }

    decision = decide_catalog_publish(
        refreshed,
        cached,
        _decision_config(["alpha"]),
        pending_empty_listings={},
        now_iso=lambda: "2026-08-08T10:00:00+00:00",
    )

    assert decision.decision == "rejected_empty"
    assert decision.rejected_reason
    assert decision.preserved_by_source == {}


def test_kilo_stale_fallback_requires_persisted_free_flag_evidence():
    old_row = _decision_row("kilo", "stealth/ox-alpha")
    old_row["free_access"]["proof"] = "provider_free_catalog_pricing"
    verified_row = _decision_row("kilo", "verified-free")
    verified_row["free_access"].update(
        {"proof": "provider_free_catalog_pricing", "catalog_free_flag_verified": "isFree"}
    )
    refreshed = {
        "generated_at": "2026-08-21T10:00:00+00:00",
        "models": [],
        "providers": {"kilo": _decision_summary("transient_error", error="Timeout")},
    }
    config = {
        "min_context_length": 128_000,
        "providers": {
            "kilo": {
                "enabled": True,
                "provider_class": "free_model",
                "free_mode": "catalog_free",
                "free_access_proof": "provider_free_catalog_pricing",
                "catalog_free_flag_field": "isFree",
            }
        },
    }

    old_decision = decide_catalog_publish(
        refreshed,
        {"generated_at": "2026-08-21T09:00:00+00:00", "models": [old_row]},
        config,
        pending_empty_listings={},
        now_iso=lambda: "2026-08-21T10:00:00+00:00",
    )
    verified_decision = decide_catalog_publish(
        refreshed,
        {"generated_at": "2026-08-21T09:00:00+00:00", "models": [verified_row]},
        config,
        pending_empty_listings={},
        now_iso=lambda: "2026-08-21T10:00:00+00:00",
    )

    assert old_decision.decision == "promoted"
    assert old_decision.catalog["models"] == []
    assert verified_decision.decision == "rejected_empty"
    assert verified_decision.fallback_catalog is not None
    assert [row["upstream_id"] for row in verified_decision.fallback_catalog["models"]] == ["verified-free"]


def test_partial_failure_preserves_only_policy_valid_rows_of_failed_providers():
    cached = {
        "generated_at": "2026-08-08T09:00:00+00:00",
        "models": [
            _decision_row("alpha", "kept"),
            _decision_row("beta", "survivor"),
            _decision_row("beta", "not-free", eligible=False),
            _decision_row("beta", "tiny-context", context_length=4096),
            _decision_row("beta", "already-stale", stale_since="2026-08-08T05:00:00+00:00"),
        ],
    }
    refreshed = {
        "generated_at": "2026-08-08T10:00:00+00:00",
        "models": [_decision_row("alpha", "kept")],
        "providers": {
            "alpha": _decision_summary("success_nonempty", raw_count=1, accepted_count=1),
            "beta": _decision_summary("transient_error", error="ConnectionError: dns"),
        },
    }

    decision = decide_catalog_publish(
        refreshed,
        cached,
        _decision_config(["alpha", "beta"]),
        pending_empty_listings={},
        now_iso=lambda: "2026-08-08T10:00:00+00:00",
    )

    assert decision.decision == "merged_last_known_good"
    assert decision.preserved_by_source == {"beta": 2}
    preserved = {
        str(model.get("upstream_id")): model
        for model in decision.catalog["models"]
        if model.get("source") == "beta"
    }
    assert set(preserved) == {"survivor", "already-stale"}
    assert preserved["survivor"]["catalog_stale"] is True
    assert preserved["survivor"]["stale_since"] == "2026-08-08T09:00:00+00:00"
    # A row that was already stale keeps its original staleness epoch: merging again must
    # not refresh the age of its free-access proof.
    assert preserved["already-stale"]["stale_since"] == "2026-08-08T05:00:00+00:00"
    # The fresh provider's rows are untouched.
    alpha_rows = [model for model in decision.catalog["models"] if model.get("source") == "alpha"]
    assert alpha_rows and not any(model.get("catalog_stale") for model in alpha_rows)


def test_required_allowlist_change_drops_stale_cached_rows_immediately():
    cached_row = _decision_row("candidate", "prefix/verified-model")
    cached_row["free_access"].update(
        {"mode": "quota_free", "proof": "provider_free_endpoint", "scope": "provider"}
    )
    refreshed = {
        "generated_at": "2026-08-08T10:00:00+00:00",
        "models": [],
        "providers": {
            "candidate": _decision_summary("transient_error", error="ConnectionError: dns"),
        },
    }
    config = _decision_config(["candidate"])
    config["providers"]["candidate"].update(
        {
            "provider_class": "free_quota",
            "free_mode": "quota_free",
            "free_access_proof": "provider_free_endpoint",
            "require_model_id_allowlist": True,
            "model_id_allowlist": ["verified-model"],
        }
    )

    decision = decide_catalog_publish(
        refreshed,
        {"generated_at": "2026-08-08T09:00:00+00:00", "models": [cached_row]},
        config,
        pending_empty_listings={},
        now_iso=lambda: "2026-08-08T10:00:00+00:00",
    )

    assert decision.decision == "promoted"
    assert decision.catalog["models"] == []
    assert decision.fallback_catalog is None


def test_disabled_and_credentialless_providers_are_removed_immediately():
    cached = {
        "generated_at": "2026-08-08T09:00:00+00:00",
        "models": [_decision_row("gone", "model-a"), _decision_row("locked", "model-b")],
    }
    refreshed = {
        "generated_at": "2026-08-08T10:00:00+00:00",
        "models": [],
        "providers": {
            "gone": _decision_summary("disabled", enabled=False, error="disabled"),
            "locked": _decision_summary("credentials_unavailable", error="missing credentials"),
        },
    }

    decision = decide_catalog_publish(
        refreshed,
        cached,
        _decision_config(["gone", "locked"]),
        pending_empty_listings={},
        now_iso=lambda: "2026-08-08T10:00:00+00:00",
    )

    # No transient error anywhere: an empty publish here is the operator's explicit intent.
    assert decision.decision == "promoted"
    assert decision.catalog["models"] == []


def test_successful_empty_listing_needs_a_second_observation_before_pruning():
    cached = {
        "generated_at": "2026-08-08T09:00:00+00:00",
        "models": [_decision_row("alpha", "model-a")],
    }
    first_refresh = {
        "generated_at": "2026-08-08T10:00:00+00:00",
        "models": [],
        "providers": {"alpha": _decision_summary("success_empty", raw_count=0)},
    }

    first = decide_catalog_publish(
        first_refresh,
        cached,
        _decision_config(["alpha"]),
        pending_empty_listings={},
        now_iso=lambda: "2026-08-08T10:00:00+00:00",
    )

    # First observation: rows survive (stale) and the empty listing is recorded as pending.
    assert first.decision == "merged_last_known_good"
    assert first.preserved_by_source == {"alpha": 1}
    assert "alpha" in first.pending_empty_listings
    assert first.pending_empty_listings["alpha"]["catalog_generation"] == "2026-08-08T10:00:00+00:00"

    second_refresh = {
        "generated_at": "2026-08-08T11:00:00+00:00",
        "models": [],
        "providers": {"alpha": _decision_summary("success_empty", raw_count=0)},
    }

    second = decide_catalog_publish(
        second_refresh,
        first.catalog,
        _decision_config(["alpha"]),
        pending_empty_listings=first.pending_empty_listings,
        now_iso=lambda: "2026-08-08T11:00:00+00:00",
    )

    # Second, distinct observation confirms the removal.
    assert second.decision == "promoted"
    assert second.catalog["models"] == []
    assert "alpha" not in second.pending_empty_listings
    assert second.confirmed_empty_sources == ["alpha"]


def test_successful_nonempty_listing_clears_a_pending_empty_observation():
    cached = {
        "generated_at": "2026-08-08T10:00:00+00:00",
        "models": [_decision_row("alpha", "model-a", stale_since="2026-08-08T09:00:00+00:00")],
    }
    refreshed = {
        "generated_at": "2026-08-08T11:00:00+00:00",
        "models": [_decision_row("alpha", "model-a")],
        "providers": {"alpha": _decision_summary("success_nonempty", raw_count=1, accepted_count=1)},
    }

    decision = decide_catalog_publish(
        refreshed,
        cached,
        _decision_config(["alpha"]),
        pending_empty_listings={"alpha": {"observed_at": "2026-08-08T10:00:00+00:00", "catalog_generation": "2026-08-08T10:00:00+00:00"}},
        now_iso=lambda: "2026-08-08T11:00:00+00:00",
    )

    assert decision.decision == "promoted"
    assert decision.pending_empty_listings == {}
    # The fresh listing replaces the stale row: no stale marker survives a real recovery.
    assert not any(model.get("catalog_stale") for model in decision.catalog["models"])


def test_runner_tags_every_provider_summary_with_a_result_class():
    import dataclasses

    listed_model = {
        "id": "listed-model",
        "name": "Listed",
        "context_length": 200000,
        "pricing": {"prompt": 0, "completion": 0},
        "supported_parameters": ["tools"],
    }
    base = _ports([])

    def fetch(source: str, _config: dict[str, Any]):
        if source == "flaky":
            return [], "Timeout: timed out"
        if source == "hollow":
            return [], None
        return [dict(listed_model)], None

    ports = dataclasses.replace(
        base,
        auth_status=lambda _config: {
            source: {"invokable": source != "locked", "reason": "configured" if source != "locked" else "missing credentials", "base_url": "https://fake/v1"}
            for source in ("healthy", "flaky", "hollow", "gone", "locked")
        },
        fetch_provider_catalog=fetch,
    )
    runner = CatalogRefreshRunner(ports, virtual_models=set())
    catalog = runner.refresh_catalog(
        {
            "allow_paid_fallback": False,
            "min_context_length": 128000,
            "providers": {
                "healthy": {"enabled": True},
                "flaky": {"enabled": True},
                "hollow": {"enabled": True},
                "gone": {"enabled": False},
                "locked": {"enabled": True, "activation_policy": "configured_credentials"},
            },
        }
    )

    classes = {source: summary.get("result_class") for source, summary in catalog["providers"].items()}
    assert classes == {
        "healthy": "success_nonempty",
        "flaky": "transient_error",
        "hollow": "success_empty",
        "gone": "disabled",
        "locked": "credentials_unavailable",
    }
