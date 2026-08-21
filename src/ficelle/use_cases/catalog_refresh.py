from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ficelle.domain_models import CatalogModel, RouterModel
from ficelle.providers.base import (
    TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE,
    ProviderCatalogAdapter as ProviderCatalogAdapterProtocol,
)


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
                result_class: str,
            ) -> dict[str, Any]:
                return {
                    "adapter": provider_diagnostics.get("adapter"),
                    "catalog_url": provider_diagnostics.get("catalog_url"),
                    "result_class": result_class,
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
                    result_class="disabled",
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
                    result_class="credentials_unavailable",
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

            # The pairing dialect (id convention, pricing units) is the adapter's; the
            # runner only stamps what it returns onto accepted rows.
            reference_prices = provider_adapter.reference_prices_for_free_models(raw_models)
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
                    reference_pricing=reference_prices.get(upstream_id),
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
                # A fetch error is transient by definition here: an explicit disable or a
                # credential gap took one of the earlier branches. A clean fetch that listed
                # nothing is a distinct signal from a transport failure — the publish decision
                # treats the two differently (L1-R1).
                result_class=(
                    "transient_error"
                    if error
                    else ("success_nonempty" if raw_models else "success_empty")
                ),
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


@dataclass(frozen=True)
class CatalogPublishDecision:
    """What a finished catalog rebuild is allowed to do to the active catalog (L1-R1).

    ``decision`` is one of:

    - ``promoted``: publish the rebuilt catalog as-is;
    - ``merged_last_known_good``: publish the rebuilt catalog plus still-policy-valid rows
      preserved from the cache for providers whose fetch failed transiently (each preserved
      row is marked ``catalog_stale`` with its original ``stale_since`` epoch);
    - ``rejected_empty``: do not publish at all — the rebuild accepted zero models while the
      cache was non-empty and at least one provider failed transiently, so replacing a
      working catalog with an empty one would be destructive.

    ``pending_empty_listings`` is the updated observation ledger for providers whose fetch
    succeeded but listed nothing: removal of a previously non-empty provider requires a
    second, distinct observation before its cached rows stop being preserved.
    """

    decision: str
    catalog: dict[str, Any]
    preserved_by_source: dict[str, int]
    pending_empty_listings: dict[str, dict[str, Any]]
    confirmed_empty_sources: list[str]
    rejected_reason: str | None = None
    # A policy-filtered, stale-marked view of the still-active generation for request-path
    # callers. It is populated only for rejected publication; the catalog file is untouched.
    fallback_catalog: dict[str, Any] | None = None


def _cached_row_passes_current_policy(
    row: dict[str, Any],
    enabled_sources: set[str],
    min_context: int,
    providers_cfg: dict[str, Any],
) -> bool:
    # CatalogModel.from_raw owns the row-schema coercion rules (safe ints, bools), so a
    # malformed cached row degrades to "not restorable" instead of raising here.
    model = CatalogModel.from_raw(row)
    if model is None or model.source not in enabled_sources:
        return False
    if not model.invokable or not model.supports_tools or model.context_length < min_context:
        return False
    if not model.free_access.get("eligible"):
        return False

    provider_cfg = providers_cfg.get(model.source)
    if not isinstance(provider_cfg, dict):
        return False
    upstream_id = model.upstream_id.lower()
    excluded = provider_cfg.get("model_id_exclude_patterns")
    excluded_patterns = (
        [str(pattern).strip().lower() for pattern in excluded if str(pattern).strip()]
        if isinstance(excluded, list)
        else []
    )
    if any(pattern in upstream_id for pattern in excluded_patterns):
        return False
    allowlist = provider_cfg.get("model_id_allowlist")
    allowed_patterns = (
        [str(pattern).strip().lower() for pattern in allowlist if str(pattern).strip()]
        if isinstance(allowlist, list)
        else []
    )
    official_ids = (
        [str(item).strip().lower() for item in provider_cfg.get("official_free_ids") or [] if str(item).strip()]
        if isinstance(provider_cfg.get("official_free_ids"), list)
        else []
    )
    official_suffixes = (
        [str(item).strip().lower() for item in provider_cfg.get("official_free_id_suffixes") or [] if str(item).strip()]
        if isinstance(provider_cfg.get("official_free_id_suffixes"), list)
        else []
    )
    if official_ids or official_suffixes:
        if upstream_id not in official_ids and not any(upstream_id.endswith(suffix) for suffix in official_suffixes):
            return False
    elif provider_cfg.get("require_model_id_allowlist"):
        if upstream_id not in allowed_patterns:
            return False
    elif allowed_patterns:
        provider_class = str(provider_cfg.get("provider_class") or provider_cfg.get("source_type") or "")
        if provider_class == "free_model":
            if upstream_id not in allowed_patterns:
                return False
        elif not any(pattern in upstream_id for pattern in allowed_patterns):
            return False

    # Cached pricing proof remains valid only inside the bounded stale grace. Config-owned
    # free-access proof must additionally still be authorized by the current provider class;
    # otherwise changing a provider from free_quota/local/free-model to an untrusted class
    # could keep the old eligible bit alive through a transient refresh failure.
    access_mode = str(model.free_access.get("mode") or "")
    access_proof = str(model.free_access.get("proof") or "")
    configured_flag = str(provider_cfg.get("catalog_free_flag_field") or "").strip()
    if access_proof == "provider_free_catalog_pricing" and configured_flag:
        if str(model.free_access.get("catalog_free_flag_verified") or "") != configured_flag:
            return False
    if access_mode == "catalog_free" and access_proof == "provider_pricing":
        return True
    if access_mode in TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE:
        configured_mode = str(provider_cfg.get("free_mode") or "")
        provider_class = str(provider_cfg.get("provider_class") or provider_cfg.get("source_type") or "")
        return (
            configured_mode == access_mode
            and provider_class in TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE[access_mode]
            and bool(str(provider_cfg.get("free_access_proof") or "").strip())
        )
    return False


def decide_catalog_publish(
    refreshed: dict[str, Any],
    cached: dict[str, Any],
    effective_config: dict[str, Any],
    *,
    pending_empty_listings: dict[str, dict[str, Any]],
    now_iso: Callable[[], str],
) -> CatalogPublishDecision:
    refreshed_models = [model for model in refreshed.get("models", []) if isinstance(model, dict)]
    cached_models = [
        model for model in (cached.get("models", []) if isinstance(cached, dict) else []) if isinstance(model, dict)
    ]
    summaries = refreshed.get("providers") if isinstance(refreshed.get("providers"), dict) else {}
    # `refreshed` always comes straight from this build's runner, which tags every summary;
    # an absent class stays neutral (neither transient nor empty), which is the safe default.
    classes: dict[str, str] = {
        str(source): str(summary.get("result_class") or "")
        for source, summary in summaries.items()
        if isinstance(summary, dict)
    }

    providers_cfg = effective_config.get("providers") if isinstance(effective_config.get("providers"), dict) else {}
    enabled_sources = {
        str(source)
        for source, provider_cfg in providers_cfg.items()
        if isinstance(provider_cfg, dict) and provider_cfg.get("enabled", True)
    }
    min_context = int(effective_config.get("min_context_length") or 0)
    transient_sources = {source for source, cls in classes.items() if cls == "transient_error"}
    pending = {str(key): dict(value) for key, value in pending_empty_listings.items() if isinstance(value, dict)}
    cached_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in cached_models:
        cached_by_source.setdefault(str(row.get("source") or ""), []).append(row)

    generation = str(refreshed.get("generated_at") or "")
    confirmed_empty_sources: list[str] = []
    preserve_sources: set[str] = set(transient_sources)
    for source, cls in classes.items():
        if cls == "success_nonempty":
            pending.pop(source, None)
        elif cls in {"disabled", "credentials_unavailable"}:
            # A later re-enable/recredential starts a new observation sequence; an empty
            # listing seen before the explicit removal cannot confirm a future removal.
            pending.pop(source, None)
        elif cls == "success_empty" and cached_by_source.get(source):
            observed = pending.get(source)
            if observed and str(observed.get("catalog_generation") or "") != generation:
                # Second, distinct empty listing: the removal is confirmed.
                pending.pop(source, None)
                confirmed_empty_sources.append(source)
            else:
                preserve_sources.add(source)
                pending[source] = {"observed_at": now_iso(), "catalog_generation": generation}

    known_ids = {str(model.get("id") or "") for model in refreshed_models}
    stale_epoch = str(cached.get("generated_at") or "") or now_iso()
    preserved: list[dict[str, Any]] = []
    preserved_by_source: dict[str, int] = {}
    for source in sorted(preserve_sources):
        for row in cached_by_source.get(source, []):
            if str(row.get("id") or "") in known_ids:
                continue
            if not _cached_row_passes_current_policy(row, enabled_sources, min_context, providers_cfg):
                continue
            stale_row = dict(row)
            stale_row["catalog_stale"] = True
            # Keep the original staleness epoch: merging again must not refresh the age of
            # the row's free-access proof.
            stale_row["stale_since"] = str(row.get("stale_since") or "") or stale_epoch
            preserved.append(stale_row)
            preserved_by_source[source] = preserved_by_source.get(source, 0) + 1

    # Reject only when the rebuild would remove *nothing but* rows belonging to transiently
    # failed providers and every cached row still passes today's policy. A concurrent explicit
    # disable, credential removal, allowlist change, or confirmed empty listing must take effect
    # immediately; publishing the policy-filtered merge does that without sacrificing the failed
    # providers' last-known-good rows.
    cached_sources = {str(row.get("source") or "") for row in cached_models}
    if (
        not refreshed_models
        and cached_models
        and transient_sources
        and cached_sources <= transient_sources
        and len(preserved) == len(cached_models)
    ):
        return CatalogPublishDecision(
            decision="rejected_empty",
            catalog=refreshed,
            preserved_by_source={},
            pending_empty_listings=pending,
            confirmed_empty_sources=confirmed_empty_sources,
            rejected_reason=(
                f"refresh accepted zero models while the cached catalog held {len(cached_models)}; "
                f"transient provider errors: {', '.join(sorted(transient_sources))}"
            ),
            fallback_catalog={**cached, "models": preserved},
        )

    if not preserved:
        return CatalogPublishDecision(
            decision="promoted",
            catalog=refreshed,
            preserved_by_source={},
            pending_empty_listings=pending,
            confirmed_empty_sources=confirmed_empty_sources,
        )
    return CatalogPublishDecision(
        decision="merged_last_known_good",
        catalog={**refreshed, "models": [*refreshed_models, *preserved]},
        preserved_by_source=preserved_by_source,
        pending_empty_listings=pending,
        confirmed_empty_sources=confirmed_empty_sources,
    )


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
