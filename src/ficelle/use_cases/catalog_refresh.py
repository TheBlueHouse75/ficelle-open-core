from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ficelle.domain_models import RouterModel
from ficelle.providers.base import ProviderCatalogAdapter as ProviderCatalogAdapterProtocol


LoadJson = Callable[[Path, Any], Any]
AtomicWriteJson = Callable[[Path, Any], None]
RefreshCatalog = Callable[[dict[str, Any]], dict[str, Any]]
CatalogConfigFingerprint = Callable[[dict[str, Any]], str]
NowSeconds = Callable[[], float]
StateMutator = Callable[[dict[str, Any]], dict[str, Any] | None]
UpdateState = Callable[[StateMutator, str | None], dict[str, Any]]
FetchProviderCatalog = Callable[[str, dict[str, Any]], tuple[list[dict[str, Any]], str | None]]
ProviderCatalogAdapter = Callable[[str], ProviderCatalogAdapterProtocol]
AuthStatus = Callable[[dict[str, Any]], dict[str, dict[str, Any]]]
StrictZeroPricing = Callable[[Any], tuple[bool, str, dict[str, Any]]]
NormalizedFreeAccess = Callable[[dict[str, Any], bool | None, str | None], dict[str, Any]]
PricingSafetyForFreeAccess = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
ContextLengthProvenance = Callable[[int | None, int, str], tuple[str, int | None]]


@dataclass(frozen=True)
class CatalogRefreshPorts:
    auth_status: AuthStatus
    provider_catalog_adapter: ProviderCatalogAdapter
    fetch_provider_catalog: FetchProviderCatalog
    strict_zero_pricing: StrictZeroPricing
    normalized_free_access: NormalizedFreeAccess
    pricing_safety_for_free_access: PricingSafetyForFreeAccess
    context_length_provenance: ContextLengthProvenance
    catalog_config_fingerprint: CatalogConfigFingerprint
    catalog_config_structural_fingerprint: CatalogConfigFingerprint
    now_iso: Callable[[], str]
    safe_detail: Callable[[Any], str]
    safe_int: Callable[[Any, int], int]
    safe_float: Callable[[Any, float], float]
    safe_optional_int: Callable[[Any], int | None]
    safe_optional_bool: Callable[[Any], bool | None]
    safe_string_list: Callable[[Any], list[str]]
    has_tools: Callable[[Any], bool]
    has_structured: Callable[[Any], bool]
    concrete_model_id: Callable[[str, str], str]
    capabilities_from_defaults: Callable[[dict[str, Any], dict[str, Any]], list[str]]
    capabilities_from_reference: Callable[[list[str], str], list[str]]
    capabilities_refuted_by_reference: Callable[[list[str], str], list[str]]
    model_reference_confidence: Callable[[str], str]
    provider_key_url: Callable[[str], str | None]


def provider_model_inventory_row(
    *,
    model: dict[str, Any] | None,
    upstream_id: str | None,
    status: str,
    reason: str,
    source: str,
    ports: CatalogRefreshPorts,
    context_length: int | None = None,
    params: list[str] | None = None,
    free_access: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": ports.safe_detail(upstream_id),
        "name": ports.safe_detail((model or {}).get("name")) or ports.safe_detail(upstream_id),
        "owned_by": ports.safe_detail((model or {}).get("owned_by")),
        "status": status,
        "reason": reason,
        "model_id": ports.safe_detail(ports.concrete_model_id(source, upstream_id)) if upstream_id else None,
        "context_length": context_length,
        "supports_tools": ports.has_tools(params or []),
    }
    if free_access:
        row["free_mode"] = ports.safe_detail(free_access.get("mode"))
        row["free_status"] = ports.safe_detail(free_access.get("status"))
        row["free_scope"] = ports.safe_detail(free_access.get("scope"))
    return {key: value for key, value in row.items() if value is not None}


class CatalogRefreshRunner:
    def __init__(self, ports: CatalogRefreshPorts, *, virtual_models: set[str]) -> None:
        self.ports = ports
        self.virtual_models = virtual_models

    def refresh_catalog(self, config: dict[str, Any]) -> dict[str, Any]:
        if config.get("allow_paid_fallback") is not False:
            raise RuntimeError("allow_paid_fallback must stay false for the free router MVP")

        min_context = int(config.get("min_context_length") or 0)
        auth = self.ports.auth_status(config)
        all_models: list[dict[str, Any]] = []
        provider_summaries: dict[str, Any] = {}

        # Pre-fetch each provider's raw catalog concurrently so a single slow or
        # unreachable provider endpoint cannot delay the others — a down provider
        # must not block the rest of the catalog. Only the network call is
        # parallelized; all normalization/aggregation below stays sequential, so
        # there is no shared-state race.
        # A provider is skipped (with a summary) below if it is disabled or, for
        # configured-credentials providers, missing usable credentials. These two
        # predicates are the single source of truth for "should we fetch it",
        # shared by the parallel pre-fetch here and the per-provider loop below.
        def _provider_disabled(provider_cfg: dict[str, Any]) -> bool:
            return not provider_cfg.get("enabled", True)

        def _provider_missing_credentials(source: str, provider_cfg: dict[str, Any]) -> bool:
            return (
                str(provider_cfg.get("activation_policy") or "") == "configured_credentials"
                and not auth.get(source, {}).get("invokable")
            )

        fetch_sources = [
            str(source)
            for source, provider_cfg in config.get("providers", {}).items()
            if not _provider_disabled(provider_cfg)
            and not _provider_missing_credentials(str(source), provider_cfg)
        ]
        prefetched_catalogs: dict[str, tuple[list[dict[str, Any]], str | None]] = {}
        if fetch_sources:
            def _prefetch(source: str) -> tuple[str, tuple[list[dict[str, Any]], str | None]]:
                try:
                    return source, self.ports.fetch_provider_catalog(source, config)
                except Exception as exc:
                    return source, ([], f"{type(exc).__name__}: {exc}")

            with ThreadPoolExecutor(max_workers=min(16, len(fetch_sources))) as executor:
                for source, result in executor.map(_prefetch, fetch_sources):
                    prefetched_catalogs[source] = result

        for source, provider_cfg in config.get("providers", {}).items():
            auth_row = auth.get(source, {})
            provider_adapter = self.ports.provider_catalog_adapter(str(source))
            provider_diagnostics = provider_adapter.safe_diagnostics(provider_cfg)
            provider_policy = provider_adapter.catalog_policy(provider_cfg)

            def provider_summary(
                *,
                raw_count: int,
                accepted_count: int,
                invokable: bool,
                auth_reason: str | None,
                error: str | None,
                enabled: bool,
                rejected: dict[str, int],
                models: list[dict[str, Any]],
            ) -> dict[str, Any]:
                return {
                    "adapter": provider_diagnostics.get("adapter"),
                    "catalog_url": provider_diagnostics.get("catalog_url"),
                    "raw_count": raw_count,
                    "accepted_count": accepted_count,
                    "invokable": invokable,
                    "auth_reason": auth_reason,
                    "key_source": auth_row.get("key_source"),
                    "key_preview": auth_row.get("key_preview"),
                    "key_url": self.ports.provider_key_url(str(source)),
                    "error": error,
                    "enabled": enabled,
                    "source_type": provider_diagnostics.get("source_type"),
                    "provider_class": provider_diagnostics.get("provider_class"),
                    "free_mode": provider_diagnostics.get("free_mode"),
                    "free_scope": provider_diagnostics.get("free_scope"),
                    "provider_account_id": self.ports.safe_detail(provider_policy.model_defaults.get("provider_account_id")),
                    "activation_policy": provider_diagnostics.get("activation_policy"),
                    "quota_reset_policy": provider_diagnostics.get("quota_reset_policy"),
                    "notes": provider_cfg.get("notes"),
                    "rejected": rejected,
                    "models": models,
                }

            if _provider_disabled(provider_cfg):
                provider_summaries[source] = provider_summary(
                    raw_count=0,
                    accepted_count=0,
                    invokable=bool(auth_row.get("invokable")),
                    auth_reason="disabled" if auth_row.get("invokable") else auth_row.get("reason", "disabled"),
                    error="disabled",
                    enabled=False,
                    rejected=dict(provider_policy.rejection_counters),
                    models=[],
                )
                continue
            if _provider_missing_credentials(str(source), provider_cfg):
                reason = str(auth_row.get("reason") or "missing credentials")
                provider_summaries[source] = provider_summary(
                    raw_count=0,
                    accepted_count=0,
                    invokable=False,
                    auth_reason=reason,
                    error=reason,
                    enabled=True,
                    rejected=dict(provider_policy.rejection_counters),
                    models=[],
                )
                continue
            # Eligible providers (they passed both skip guards above) were all
            # pre-fetched in parallel, so read the result instead of fetching again.
            raw_models, error = prefetched_catalogs[str(source)]
            accepted: list[dict[str, Any]] = []
            rejected = dict(provider_policy.rejection_counters)
            provider_models: list[dict[str, Any]] = []

            def append_provider_model(
                model: dict[str, Any] | None,
                upstream_id: str | None,
                status: str,
                reason: str,
                *,
                context_length: int = 0,
                params: list[str] | None = None,
                free_access: dict[str, Any] | None = None,
            ) -> None:
                provider_models.append(
                    provider_model_inventory_row(
                        model=model,
                        upstream_id=upstream_id,
                        status=status,
                        reason=reason,
                        source=source,
                        ports=self.ports,
                        context_length=context_length,
                        params=params,
                        free_access=free_access,
                    )
                )

            for model in raw_models:
                if not isinstance(model, dict):
                    rejected["invalid"] += 1
                    append_provider_model(None, None, "rejected", "invalid")
                    continue
                catalog_model_result = provider_adapter.normalize_catalog_model(model, provider_policy)
                model = catalog_model_result.normalized_model
                declared_context = self.ports.safe_optional_int(model.get("context_length"))
                trusted_model = catalog_model_result.trusted_model
                raw_pricing = trusted_model.get("pricing")
                pricing = raw_pricing if isinstance(raw_pricing, dict) else {}
                params = self.ports.safe_string_list(trusted_model.get("supported_parameters"))
                context_length = self.ports.safe_int(trusted_model.get("context_length"), 0)
                pricing_ok, pricing_reason, pricing_safety = self.ports.strict_zero_pricing(raw_pricing)
                upstream_id = str(model.get("id") or "").strip()
                trusted_free_access = provider_adapter.trusted_free_access(provider_cfg, model, provider_policy)
                if trusted_free_access is not None:
                    trusted_model["free_access"] = trusted_free_access
                free_access = self.ports.normalized_free_access(trusted_model, pricing_ok, pricing_reason)
                if provider_adapter.excludes_catalog_model(model, provider_policy):
                    rejected["not_chat"] += 1
                    append_provider_model(
                        trusted_model,
                        upstream_id,
                        "rejected",
                        "not_chat",
                        context_length=context_length,
                        params=params,
                        free_access=free_access,
                    )
                    continue
                if not free_access.get("eligible") and provider_policy.has_trusted_free_access:
                    rejected["not_eligible"] += 1
                    append_provider_model(
                        trusted_model,
                        upstream_id,
                        "rejected",
                        "not_eligible",
                        context_length=context_length,
                        params=params,
                        free_access=free_access,
                    )
                    continue
                if not free_access.get("eligible"):
                    rejected["unsafe_pricing"] += 1
                    if free_access.get("mode") == "paid" or not pricing_ok:
                        rejected["paid"] += 1
                    continue
                if not self.ports.has_tools(params):
                    rejected["no_tools"] += 1
                    append_provider_model(
                        trusted_model,
                        upstream_id,
                        "rejected",
                        "no_tools",
                        context_length=context_length,
                        params=params,
                        free_access=free_access,
                    )
                    continue
                if context_length < min_context:
                    rejected["small_context"] += 1
                    append_provider_model(
                        trusted_model,
                        upstream_id,
                        "rejected",
                        "small_context",
                        context_length=context_length,
                        params=params,
                        free_access=free_access,
                    )
                    continue
                if not upstream_id:
                    rejected["invalid"] += 1
                    append_provider_model(
                        trusted_model,
                        None,
                        "rejected",
                        "invalid",
                        context_length=context_length,
                        params=params,
                        free_access=free_access,
                    )
                    continue
                architecture = trusted_model.get("architecture") if isinstance(trusted_model.get("architecture"), dict) else {}
                top_provider = trusted_model.get("top_provider") if isinstance(trusted_model.get("top_provider"), dict) else {}
                default_claimed_capabilities = self.ports.capabilities_from_defaults(model, provider_policy.model_defaults)
                context_length_source, context_length_estimate = self.ports.context_length_provenance(
                    declared_context, context_length, upstream_id
                )
                entry = RouterModel(
                    id=self.ports.concrete_model_id(source, upstream_id),
                    source=source,
                    upstream_id=upstream_id,
                    name=trusted_model.get("name"),
                    context_length=context_length,
                    supports_tools=True,
                    supports_structured_outputs=self.ports.has_structured(params),
                    pricing=dict(pricing),
                    pricing_safety=self.ports.pricing_safety_for_free_access(pricing_safety, free_access),
                    free_access=free_access,
                    provider_account_id=self.ports.safe_detail(trusted_model.get("provider_account_id")),
                    supported_parameters=[str(item) for item in params],
                    input_modalities=self.ports.safe_string_list(architecture.get("input_modalities")),
                    output_modalities=self.ports.safe_string_list(architecture.get("output_modalities")),
                    modality=str(architecture.get("modality") or "") or None,
                    tokenizer=str(architecture.get("tokenizer") or "") or None,
                    instruct_type=str(architecture.get("instruct_type") or "") or None,
                    max_completion_tokens=self.ports.safe_optional_int(top_provider.get("max_completion_tokens")),
                    is_moderated=self.ports.safe_optional_bool(top_provider.get("is_moderated")),
                    knowledge_cutoff=str(trusted_model.get("knowledge_cutoff") or "") or None,
                    description=str(trusted_model.get("description") or "") or None,
                    invokable=bool(auth_row.get("invokable")),
                    auth_reason=None if auth_row.get("invokable") else str(auth_row.get("reason") or "missing credentials"),
                    catalog_url=provider_cfg.get("catalog_url"),
                    base_url=str(auth_row.get("base_url") or provider_cfg.get("base_url") or "").rstrip("/"),
                    capabilities_from_defaults=default_claimed_capabilities,
                    capabilities_from_reference=self.ports.capabilities_from_reference(default_claimed_capabilities, upstream_id),
                    capabilities_refuted_by_reference=self.ports.capabilities_refuted_by_reference(default_claimed_capabilities, upstream_id),
                    reference_confidence=self.ports.model_reference_confidence(upstream_id),
                    context_length_source=context_length_source,
                    context_length_estimate=context_length_estimate,
                    burn_weight=(
                        self.ports.safe_float(provider_cfg.get("burn_weight"), 1.0)
                        if provider_cfg.get("burn_weight") is not None
                        else None
                    ),
                )
                accepted.append(asdict(entry))
                append_provider_model(
                    trusted_model,
                    upstream_id,
                    "accepted",
                    "accepted",
                    context_length=context_length,
                    params=params,
                    free_access=free_access,
                )
            if accepted:
                seen_ids: set[str] = set()
                deduped: list[dict[str, Any]] = []
                for accepted_entry in accepted:
                    entry_id = str(accepted_entry.get("id") or "")
                    if entry_id in seen_ids:
                        continue
                    seen_ids.add(entry_id)
                    deduped.append(accepted_entry)
                accepted = deduped
            all_models.extend(accepted)
            provider_summaries[source] = provider_summary(
                raw_count=len(raw_models),
                accepted_count=len(accepted),
                invokable=bool(auth_row.get("invokable")),
                auth_reason=auth_row.get("reason"),
                error=error,
                enabled=True,
                rejected=rejected,
                models=provider_models,
            )

        all_models.sort(
            key=lambda item: (
                not bool(item.get("invokable")),
                not bool(item.get("supports_structured_outputs")),
                -int(item.get("context_length") or 0),
                str(item.get("source")),
                str(item.get("upstream_id")),
            )
        )

        return {
            "generated_at": self.ports.now_iso(),
            "config_fingerprint": self.ports.catalog_config_fingerprint(config),
            "config_structural_fingerprint": self.ports.catalog_config_structural_fingerprint(config),
            "min_context_length": min_context,
            "allow_paid_fallback": False,
            "virtual_models": sorted(self.virtual_models),
            "providers": provider_summaries,
            "models": all_models,
        }


def load_or_refresh_catalog(
    config: dict[str, Any],
    *,
    catalog_path: Path,
    load_json: LoadJson,
    refresh_catalog: RefreshCatalog,
    catalog_config_fingerprint: CatalogConfigFingerprint,
    now_seconds: NowSeconds,
    force: bool = False,
) -> dict[str, Any]:
    catalog = load_json(catalog_path, {})
    if not isinstance(catalog, dict) or force or not catalog.get("models"):
        return refresh_catalog(config)
    if catalog.get("config_fingerprint") != catalog_config_fingerprint(config):
        return refresh_catalog(config)
    ttl = int(config.get("catalog_ttl_seconds") or 3600)
    generated_at = str(catalog.get("generated_at") or "")
    try:
        refreshed_at = datetime.fromisoformat(generated_at)
        if now_seconds() - refreshed_at.timestamp() > ttl:
            return refresh_catalog(config)
    except Exception:
        return refresh_catalog(config)
    return catalog


def catalog_listing_sizes(catalog: Any) -> dict[str, Any]:
    """How many models each provider listed, tagged with the catalog it came from.

    Only the counts are kept. The catalog itself is hundreds of KB and is overwritten on
    every publish, and a completeness check reads nothing else off it. A provider that
    errored or listed nothing is left out: it is a hole in the record, not a count of
    zero, and treating it as a baseline would make the next listing look like a recovery.
    """
    if not isinstance(catalog, dict) or not catalog.get("generated_at"):
        return {}
    counts: dict[str, int] = {}
    for source, summary in (catalog.get("providers") or {}).items():
        if not isinstance(summary, dict) or summary.get("error"):
            continue
        raw_count = summary.get("raw_count")
        if isinstance(raw_count, int) and raw_count > 0:
            counts[str(source)] = raw_count
    return {"generated_at": catalog["generated_at"], "counts": counts}


def previous_catalog_baseline(state: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """The counts recorded one publish ago, shaped like the catalog they came from.

    `prunable_catalog_sources` reads only `providers[source].raw_count` off its
    `previous` argument, so this stands in for a catalog nobody keeps a copy of — which
    is what lets a read-only view run the same truncation check the prune runs.

    The baseline is returned only when state says it is describing *this* catalog
    generation. A publish writes the catalog file first and the state second, and a
    reader takes no catalog lock, so the two can be seen one generation apart: land in
    that window with a stale baseline and a truncated listing looks complete, which is
    the false "retired" this whole path exists to avoid. Comparing `generated_at` costs
    nothing and closes the window from both sides.

    An empty result means "no baseline", never "the provider listed nothing": callers
    must not read an absent count as evidence about a provider.
    """
    recorded = state.get("catalog_provider_raw_counts")
    if not isinstance(recorded, dict) or not catalog.get("generated_at"):
        return {}
    if recorded.get("generated_at") != catalog.get("generated_at"):
        return {}
    previous = state.get("previous_catalog_provider_raw_counts")
    counts = previous.get("counts") if isinstance(previous, dict) else None
    if not isinstance(counts, dict):
        return {}
    providers = {
        str(source): {"raw_count": count}
        for source, count in counts.items()
        if isinstance(count, int) and count > 0
    }
    return {"providers": providers} if providers else {}


def publish_catalog(
    catalog: dict[str, Any],
    *,
    catalog_path: Path,
    load_json: LoadJson,
    atomic_write_json: AtomicWriteJson,
    update_state: UpdateState,
) -> None:
    # Read the listing sizes off the catalog being replaced, before overwriting it. Keeping
    # them is what lets anything running BETWEEN two refreshes tell a provider that dropped a
    # model from one that answered with a truncated list — the file itself is about to be
    # gone. They are taken from the outgoing file rather than carried forward from the last
    # publish's own state entry: state and catalog drift apart whenever state is reset or a
    # state write fails after the file was written, and a baseline that is silently one
    # generation too old is exactly how a truncated listing passes for a complete one.
    # Callers hold the catalog lock across both writes, so nothing publishes in between.
    replaced = catalog_listing_sizes(load_json(catalog_path, {}))
    atomic_write_json(catalog_path, catalog)

    def mutate(state: dict[str, Any]) -> None:
        state.setdefault("cooldowns", {})
        state.setdefault("quota_cooldowns", {})
        state.setdefault("quota_probe_results", {})
        state.setdefault("quarantine", {})
        state["last_catalog_refresh_at"] = catalog["generated_at"]
        # Each record names the catalog generation it describes, so a reader can check that
        # the pair it holds lines up before concluding anything from it.
        state["previous_catalog_provider_raw_counts"] = replaced
        state["catalog_provider_raw_counts"] = catalog_listing_sizes(catalog)

    update_state(mutate, "refresh_catalog")


def refresh_catalog(
    config: dict[str, Any],
    *,
    ports: CatalogRefreshPorts,
    virtual_models: set[str],
    catalog_path: Path,
    load_json: LoadJson,
    atomic_write_json: AtomicWriteJson,
    update_state: UpdateState,
) -> dict[str, Any]:
    catalog = CatalogRefreshRunner(
        ports,
        virtual_models=virtual_models,
    ).refresh_catalog(config)
    publish_catalog(
        catalog,
        catalog_path=catalog_path,
        load_json=load_json,
        atomic_write_json=atomic_write_json,
        update_state=update_state,
    )
    return catalog
