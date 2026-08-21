from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeAlias

from ficelle.redaction import redact_sensitive_json
from ficelle.url_security import connectable_http_url


CandidatesForProfile: TypeAlias = Callable[
    [str, list[dict[str, Any]], dict[str, Any], dict[str, Any]],
    list[dict[str, Any]],
]
RouteCompetentCandidates: TypeAlias = Callable[
    [str, list[dict[str, Any]], dict[str, Any]],
    list[dict[str, Any]],
]
CandidateRankRows: TypeAlias = Callable[
    [str, list[dict[str, Any]], dict[str, Any], dict[str, Any]],
    list[dict[str, Any]],
]
SelectedModelRow: TypeAlias = Callable[
    [str, dict[str, Any], dict[str, Any]],
    dict[str, Any],
]
SelectedModelValue: TypeAlias = Callable[
    [str, dict[str, Any], dict[str, Any]],
    Any,
]
FailedProfileEvidenceRow: TypeAlias = Callable[
    [str, dict[str, Any], dict[str, Any]],
    dict[str, Any] | None,
]
ModelHistoryRow: TypeAlias = Callable[
    [str, dict[str, Any], dict[str, Any], dict[str, Any]],
    dict[str, Any],
]
LastRouteFallbackRows: TypeAlias = Callable[[dict[str, Any]], list[dict[str, Any]]]
SafeInt: TypeAlias = Callable[[Any, int], int]
ModelOnCooldown: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str | None]]
ModelIsQuarantined: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], bool]
StateRows: TypeAlias = Callable[[dict[str, Any]], list[dict[str, Any]]]
NormalizeVirtualProfiles: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
RuntimeStateTransform: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
FusionPayload: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
RuntimePayload: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
NowIso: TypeAlias = Callable[[], str]
ApplyConfigRule: TypeAlias = Callable[[dict[str, Any]], None]
LoadCatalog: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
StaleProfileModelRows: TypeAlias = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    list[dict[str, Any]],
]
RunQuotaProbes: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
LoadRuntimeState: TypeAlias = Callable[[], dict[str, Any]]
CatalogWithScores: TypeAlias = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    dict[str, Any],
]
BuildAdminStatus: TypeAlias = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    dict[str, Any],
]
RouterSettingsPayload: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
SafeStringList: TypeAlias = Callable[[Any], list[str]]
SafeDetail: TypeAlias = Callable[[Any], str | None]
SafeStateKey: TypeAlias = Callable[[Any], str]
SafeFloat: TypeAlias = Callable[[Any, float], float]
EpochToIso: TypeAlias = Callable[[Any], str | None]
CanonicalProfileId: TypeAlias = Callable[[str], str]
BenchmarkResultMatchesCurrentTest: TypeAlias = Callable[[str, dict[str, Any]], bool]
RedactedBenchmarkRow: TypeAlias = Callable[[str, dict[str, Any]], dict[str, Any]]
ModelKey: TypeAlias = Callable[[dict[str, Any]], str]
NormalizeFreeAccess: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
QuotaCooldownScopeFromKey: TypeAlias = Callable[[str], str]
ModelCapabilityProvenance: TypeAlias = Callable[[dict[str, Any], str | None], str]
CatalogFingerprint: TypeAlias = Callable[[dict[str, Any]], str]
ProviderAuthRow: TypeAlias = Callable[[str, dict[str, Any]], dict[str, Any]]


ADMIN_STATUS_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "status",
    "reasons",
    "generated_at",
    "catalog",
    "profiles",
    "fusion",
    "critical_profiles",
    "runtime",
    "compression",
    "cooldowns",
    "model_cooldowns",
    "provider_cooldowns",
    "quota_cooldowns",
    "quota_probe_results",
    "provider_errors",
    "model_errors",
    "last_routes",
    "performance_history",
    "quarantine",
    "actions",
)


class PerformanceCandidateRankRows(Protocol):
    def __call__(
        self,
        profile_id: str,
        candidates: list[dict[str, Any]],
        profile: dict[str, Any],
        state: dict[str, Any],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class AdminStatusBuildPorts:
    model_on_cooldown: ModelOnCooldown
    model_is_quarantined: ModelIsQuarantined
    normalized_virtual_profiles: NormalizeVirtualProfiles
    redact_runtime_state: RuntimeStateTransform
    candidates_for_profile: CandidatesForProfile
    route_competent_candidates: RouteCompetentCandidates
    candidate_rank_rows: PerformanceCandidateRankRows
    verified_capability_row: SelectedModelRow
    model_score_explanation: SelectedModelRow
    model_route_competence: SelectedModelValue
    failed_profile_evidence_row: FailedProfileEvidenceRow
    active_model_cooldown_rows: StateRows
    active_provider_cooldown_rows: StateRows
    active_quota_cooldown_rows: StateRows
    quarantine_rows: StateRows
    model_history_row: ModelHistoryRow
    model_auto_score: SelectedModelValue
    fallback_attempt_rows: LastRouteFallbackRows
    compression_observability: RuntimePayload
    runtime_state_diagnostics: RuntimeStateTransform
    fusion_payload: FusionPayload
    now_iso: NowIso
    safe_int: SafeInt


@dataclass(frozen=True)
class AdminStateBuildPorts:
    load_or_refresh_catalog: LoadCatalog
    run_due_quota_probes: RunQuotaProbes
    load_runtime_state: LoadRuntimeState
    normalized_virtual_profiles: NormalizeVirtualProfiles
    build_admin_status: BuildAdminStatus
    catalog_with_auto_scores: CatalogWithScores
    fusion_payload: FusionPayload
    router_settings_payload: RouterSettingsPayload
    redact_runtime_state: RuntimeStateTransform
    stale_profile_model_rows: StaleProfileModelRows
    safe_int: SafeInt
    safe_string_list: SafeStringList


@dataclass(frozen=True)
class AdminStatusBuilder:
    ports: AdminStatusBuildPorts
    critical_profile_ids: tuple[str, ...]
    apply_verified_capability_ttl: ApplyConfigRule
    apply_route_on_capability_reference: ApplyConfigRule

    def build(self, catalog: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        self.apply_verified_capability_ttl(config)
        self.apply_route_on_capability_reference(config)
        return build_admin_status_document(
            catalog=catalog,
            config=config,
            state=state,
            critical_profile_ids=self.critical_profile_ids,
            ports=self.ports,
        )


@dataclass(frozen=True)
class AdminStateBuilder:
    ports: AdminStateBuildPorts
    default_min_context: int
    default_canary_profiles: list[str]

    def build(self, config: dict[str, Any]) -> dict[str, Any]:
        catalog = self.ports.load_or_refresh_catalog(config)
        self.ports.run_due_quota_probes(config, catalog)
        state = self.ports.load_runtime_state()
        if not isinstance(state, dict):
            state = {}
        profiles = self.ports.normalized_virtual_profiles(config)
        status_payload = self.ports.build_admin_status(catalog, config, state)
        canary_profiles = self.ports.safe_string_list(
            (config.get("canary") or {}).get("profiles")
        )
        port = self.ports.safe_int(config.get("port"), 8646)
        base_url = connectable_http_url(config.get("host"), port, "/v1")
        return {
            "catalog": self.ports.catalog_with_auto_scores(catalog, config, state),
            "virtual_profiles": profiles,
            # Ids a virtual model still points at that the catalog dropped. Sent even when
            # empty: the dashboard renders those cards from `virtual_profiles`, and without
            # this it can only say "Unknown model" — not whether the provider is down, the
            # model is gone for good, or the listing was too incomplete to tell. `state`
            # carries the previous refresh's per-provider counts, which is what that last
            # distinction rests on.
            "stale_profile_models": self.ports.stale_profile_model_rows(config, catalog, state),
            "profiles": status_payload.get("profiles") or {},
            "performance_history": status_payload.get("performance_history") or {},
            "compression": status_payload.get("compression") or {},
            "fusion": self.ports.fusion_payload(config, state),
            "settings": self.ports.router_settings_payload(config),
            "state": self.ports.redact_runtime_state(state),
            "config": {
                "min_context_length": self.ports.safe_int(
                    config.get("min_context_length"),
                    self.default_min_context,
                ),
                "allow_paid_fallback": False,
                "catalog_ttl_seconds": self.ports.safe_int(config.get("catalog_ttl_seconds"), 3600),
                "canary_profiles": canary_profiles or list(self.default_canary_profiles),
                "base_url": base_url,
                "hermes_base_url": base_url,
            },
        }


def catalog_with_auto_scores(
    catalog: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    apply_verified_capability_ttl: ApplyConfigRule,
    normalized_virtual_profiles: NormalizeVirtualProfiles,
    normalized_free_access: NormalizeFreeAccess,
    model_score_explanation: SelectedModelRow,
) -> dict[str, Any]:
    apply_verified_capability_ttl(config)
    enriched = json.loads(json.dumps(catalog))
    profiles = normalized_virtual_profiles(config)
    for model in enriched.get("models", []):
        if not isinstance(model, dict):
            continue
        model["free_access"] = normalized_free_access(model)
        details_by_profile = {
            profile_id: model_score_explanation(str(profile.get("base_profile") or profile_id), model, state)
            for profile_id, profile in profiles.items()
        }
        model["auto_scores"] = {
            profile_id: details["score_total"]
            for profile_id, details in details_by_profile.items()
        }
        model["auto_score_details"] = details_by_profile
    return enriched


def build_admin_job_row(
    job_id: str,
    metadata: dict[str, Any] | None,
    *,
    now_iso: NowIso,
    now_epoch: Callable[[], float],
) -> dict[str, Any]:
    return redact_sensitive_json(
        {
            "id": job_id,
            "status": "running",
            "started_at": now_iso(),
            "started_at_epoch": now_epoch(),
            "metadata": metadata or {},
        }
    )


def record_admin_job_start(
    state: dict[str, Any],
    key: str,
    row: dict[str, Any],
) -> None:
    jobs = state.setdefault("admin_jobs", {})
    if not isinstance(jobs, dict):
        jobs = {}
        state["admin_jobs"] = jobs
    jobs[key] = row


def admin_job_matches(source: Any, key: str, job_id: str) -> bool:
    jobs = source.get("admin_jobs") if isinstance(source, dict) else None
    row = jobs.get(key) if isinstance(jobs, dict) else None
    return isinstance(row, dict) and row.get("id") == job_id


def record_admin_job_finish(state: dict[str, Any], key: str, job_id: str) -> None:
    jobs = state.get("admin_jobs")
    if not isinstance(jobs, dict):
        return
    row = jobs.get(key)
    if isinstance(row, dict) and row.get("id") == job_id:
        jobs.pop(key, None)


def active_admin_jobs(
    source: Any,
    *,
    now_ts: float,
    max_age_seconds: float,
    safe_float: SafeFloat,
    safe_state_key: SafeStateKey,
) -> dict[str, Any]:
    jobs = source.get("admin_jobs") if isinstance(source, dict) else {}
    if not isinstance(jobs, dict):
        return {}
    active: dict[str, Any] = {}
    for key, row in jobs.items():
        if not isinstance(row, dict) or row.get("status") != "running":
            continue
        started = safe_float(row.get("started_at_epoch"), 0.0)
        if started <= 0 or now_ts - started > max_age_seconds:
            continue
        active[safe_state_key(key)] = redact_sensitive_json(row)
    return active


def load_cached_catalog_for_admin_status(
    catalog: Any,
    config: dict[str, Any],
    *,
    default_min_context: int,
) -> dict[str, Any]:
    if isinstance(catalog, dict) and isinstance(catalog.get("models"), list):
        return catalog
    return {
        "generated_at": None,
        "min_context_length": config.get("min_context_length", default_min_context),
        "providers": {},
        "models": [],
    }


def catalog_ttl_seconds(config: dict[str, Any]) -> int:
    return int(config.get("catalog_ttl_seconds") or 3600)


def catalog_age_seconds(catalog: dict[str, Any], *, now_seconds: Callable[[], float]) -> float | None:
    """Seconds since the cached catalog was generated, or None when unusable.

    The TTL verdict below reads the age through this same function, so `/health` cannot
    publish an `age_seconds` at or below the TTL while reporting `ttl_expired` — two
    parsers of one timestamp are an easy source of contradictory fields.

    Deliberately unclamped: a catalog stamped in the future yields a negative age, which
    reports a skewed clock instead of hiding it behind a floor of zero — and clamping
    would flip the TTL verdict for a negative `catalog_ttl_seconds`.
    """
    try:
        generated = datetime.fromisoformat(str(catalog.get("generated_at") or ""))
    except Exception:
        return None
    return now_seconds() - generated.timestamp()


def cached_catalog_stale_reason(
    catalog: dict[str, Any],
    config: dict[str, Any],
    *,
    now_seconds: Callable[[], float],
    catalog_config_fingerprint: CatalogFingerprint,
    catalog_config_structural_fingerprint: CatalogFingerprint,
) -> str | None:
    age = catalog_age_seconds(catalog, now_seconds=now_seconds)
    if age is None:
        return "invalid_generated_at"
    if age > catalog_ttl_seconds(config):
        return "ttl_expired"
    structural_fingerprint = catalog.get("config_structural_fingerprint")
    if structural_fingerprint and structural_fingerprint != catalog_config_structural_fingerprint(config):
        return "config_structural_fingerprint_mismatch"
    if catalog.get("config_fingerprint") != catalog_config_fingerprint(config):
        return "credential_fingerprint_mismatch" if structural_fingerprint else "config_fingerprint_mismatch"
    return None


# Stale reasons whose cached catalog is still trustworthy enough to keep its last-known
# models (admin status reports them stale/"degraded" rather than a false "0 models / fail"):
# the catalog is merely aging (ttl_expired) or only its non-structural config/credentials
# changed while the provider structure is confirmed unchanged (credential_fingerprint_mismatch
# — it carries a matching structural fingerprint). Every other reason cannot vouch for the
# models — config_structural_fingerprint_mismatch (structure changed), invalid_generated_at
# (unparseable), and config_fingerprint_mismatch (no structural fingerprint exists to confirm
# the structure) — so the catalog is emptied. Membership is the single source of truth for the
# keep-vs-empty decision, colocated with the classifier above; an unrecognised reason fails
# safe to "empty" (conservative: never over-claim availability from a catalog we can't trust).
RECOVERABLE_STALE_REASONS: frozenset[str] = frozenset(
    {"ttl_expired", "credential_fingerprint_mismatch"}
)


def stale_reason_keeps_last_known(stale_reason: str | None) -> bool:
    return stale_reason in RECOVERABLE_STALE_REASONS


def filter_cached_catalog_to_enabled_providers(
    catalog: dict[str, Any],
    config: dict[str, Any],
    *,
    provider_auth_row: ProviderAuthRow,
) -> dict[str, Any]:
    raw_configured = config.get("providers")
    configured = raw_configured if isinstance(raw_configured, dict) else {}
    enabled_sources = {
        source
        for source, provider in configured.items()
        if isinstance(provider, dict) and provider.get("enabled", True)
    }
    # Models stay off the routing surfaces until a configured_credentials
    # provider is actually invokable. Provider *cards* do not: the admin
    # Providers list is how an operator pastes the first key, so dropping
    # an unkeyed enabled row hides the only UI that can make it invokable.
    current_auth: dict[str, dict[str, Any]] = {}
    for source in enabled_sources:
        source_name = str(source)
        auth_row = provider_auth_row(source_name, config)
        current_auth[source_name] = {**auth_row, "auth_reason": auth_row.get("reason")}
    invokable_sources = {
        source
        for source in enabled_sources
        if (
            str(configured[source].get("activation_policy") or "") != "configured_credentials"
            or bool(current_auth[str(source)].get("invokable"))
        )
    }
    models = [
        model
        for model in catalog.get("models", [])
        if isinstance(model, dict) and str(model.get("source") or "") in invokable_sources
    ]
    raw_catalog_providers = catalog.get("providers") if isinstance(catalog.get("providers"), dict) else {}
    providers = {
        source: {**provider, **current_auth[str(source)]}
        for source, provider in raw_catalog_providers.items()
        if source in enabled_sources
    }
    return {**catalog, "providers": providers, "models": models}


FUSION_LAST_RUN_FIELDS = {
    "request_id",
    "run_id",
    "profile",
    "status",
    "reason_code",
    "seen_at",
    "total_latency_seconds",
    "stage_latency_seconds",
    "panel_models",
    "panel_model_details",
    "panel_requested_count",
    "panel_eligible_count",
    "panel_selected_count",
    "panel_unfilled_count",
    "panel_success_count",
    "panel_failure_count",
    "failed_panel_models",
    "panel_result_details",
    "reason_counts",
    "judge_model",
    "judge_status",
    "judge_json_valid",
    "synthesizer_model",
    "synthesizer_status",
    "draft_model",
    "blind_draft_used",
    "peer_ranking_enabled",
    "peer_ranking_rankers",
    "peer_ranking_parseable",
    "peer_ranking_used",
    "peer_ranking_aggregate",
    "degraded_flags",
    "timeout_reached",
    "call_multiplier",
    "budget_policy",
    "paid_models_excluded",
    "free_quota_models_allowed",
    "paid_or_quota_models_excluded",
    "recursion_depth",
}

ACTION_CHECK_PROVIDERS = (
    "Check provider credentials, refresh the catalog, and inspect critical profile candidate counts."
)
ACTION_MODEL_COOLDOWNS = "Inspect active model cooldown reasons before clearing or waiting for recovery."
ACTION_PROVIDER_COOLDOWNS = "Provider cooldown is active; clear it only after auth/rate-limit recovery is understood."
ACTION_QUOTA_COOLDOWNS = "Free-quota cooldown is active; Ficelle will probe automatically when the backoff expires."
ACTION_QUARANTINE = (
    "Review quarantined models; billing_or_paid quarantine should stay until the false-free cause is understood."
)
ACTION_CANARY_FAILED = "Run ficelle canary after fixing failed profiles."
ACTION_AUTO_BENCHMARK_ERROR = (
    "Inspect the auto-benchmark background error before trusting capability discovery freshness."
)
ACTION_COMPRESSION_ERRORS = (
    "Compression is failing repeatedly; keep dry-run or live-zone disabled until the route-log errors are understood."
)
ACTION_DEFAULT = "Continue dogfood and monitor route logs/canaries."


@dataclass(frozen=True)
class AdminModelAvailability:
    models: list[dict[str, Any]]
    invokable_models: list[dict[str, Any]]
    available_models: list[dict[str, Any]]


def build_admin_model_availability(
    *,
    catalog: dict[str, Any],
    state: dict[str, Any],
    model_on_cooldown: ModelOnCooldown,
    model_is_quarantined: ModelIsQuarantined,
) -> AdminModelAvailability:
    models = [model for model in catalog.get("models", []) if isinstance(model, dict)]
    invokable_models = [model for model in models if model.get("invokable")]
    available_models = [
        model
        for model in invokable_models
        if not model_on_cooldown(model, state)[0] and not model_is_quarantined(model, state)
    ]
    return AdminModelAvailability(
        models=models,
        invokable_models=invokable_models,
        available_models=available_models,
    )


@dataclass(frozen=True)
class AdminRuntimeSnapshot:
    last_routes: dict[str, Any]
    provider_errors: dict[str, Any]
    model_errors: dict[str, Any]
    quota_probe_results: dict[str, Any]
    canary: dict[str, Any]
    canary_status: str
    auto_benchmark: dict[str, Any]


def _runtime_mapping(runtime_state: dict[str, Any], key: str) -> dict[str, Any]:
    value = runtime_state.get(key)
    return value if isinstance(value, dict) else {}


def _mapping_record_count(source: dict[str, Any], key: str) -> int:
    value = source.get(key)
    return len(value) if isinstance(value, dict) else 0


def build_admin_runtime_snapshot(runtime_state: dict[str, Any]) -> AdminRuntimeSnapshot:
    canary = _runtime_mapping(runtime_state, "canary")
    raw_auto_benchmark = runtime_state.get("auto_benchmark")
    auto_benchmark = (
        redact_sensitive_json(raw_auto_benchmark)
        if isinstance(raw_auto_benchmark, dict)
        else {}
    )
    return AdminRuntimeSnapshot(
        last_routes=_runtime_mapping(runtime_state, "last_routes"),
        provider_errors=_runtime_mapping(runtime_state, "provider_errors"),
        model_errors=_runtime_mapping(runtime_state, "model_errors"),
        quota_probe_results=_runtime_mapping(runtime_state, "quota_probe_results"),
        canary=canary,
        canary_status=str(canary.get("status") or "unknown"),
        auto_benchmark=auto_benchmark,
    )


@dataclass(frozen=True)
class AdminRecoverySnapshot:
    active_model_cooldowns: list[dict[str, Any]]
    active_provider_cooldowns: list[dict[str, Any]]
    active_quota_cooldowns: list[dict[str, Any]]
    active_cooldowns: list[dict[str, Any]]
    quarantines: list[dict[str, Any]]


def build_admin_recovery_snapshot(
    *,
    state: dict[str, Any],
    active_model_cooldown_rows: StateRows,
    active_provider_cooldown_rows: StateRows,
    active_quota_cooldown_rows: StateRows,
    quarantine_rows: StateRows,
) -> AdminRecoverySnapshot:
    active_model_cooldowns = active_model_cooldown_rows(state)
    active_provider_cooldowns = active_provider_cooldown_rows(state)
    active_quota_cooldowns = active_quota_cooldown_rows(state)
    return AdminRecoverySnapshot(
        active_model_cooldowns=active_model_cooldowns,
        active_provider_cooldowns=active_provider_cooldowns,
        active_quota_cooldowns=active_quota_cooldowns,
        active_cooldowns=active_model_cooldowns + active_provider_cooldowns + active_quota_cooldowns,
        quarantines=quarantine_rows(state),
    )


def active_cooldown_rows_from_map(
    cooldowns: Any,
    now_ts: float,
    scope: str,
    *,
    safe_state_key: SafeStateKey,
    safe_detail: SafeDetail,
    epoch_to_iso: EpochToIso,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(cooldowns, dict):
        return rows
    for key, raw in cooldowns.items():
        if not isinstance(raw, dict):
            continue
        try:
            until = float(raw.get("until") or 0)
        except Exception:
            continue
        if until <= now_ts:
            continue
        row = {
            "scope": scope,
            "key": safe_state_key(key),
            "reason": safe_detail(raw.get("reason")) or "cooldown",
            "set_at": raw.get("set_at"),
            "until": until,
            "until_iso": epoch_to_iso(until),
            "seconds_remaining": max(0, int(until - now_ts)),
            "detail": safe_detail(raw.get("detail")),
        }
        if scope == "provider":
            row["source"] = safe_state_key(key)
        rows.append(row)
    return rows


def active_model_cooldown_rows(
    state: dict[str, Any],
    now_ts: float,
    *,
    safe_state_key: SafeStateKey,
    safe_detail: SafeDetail,
    safe_int: SafeInt,
    epoch_to_iso: EpochToIso,
) -> list[dict[str, Any]]:
    rows = active_cooldown_rows_from_map(
        state.get("cooldowns"),
        now_ts,
        "model",
        safe_state_key=safe_state_key,
        safe_detail=safe_detail,
        epoch_to_iso=epoch_to_iso,
    )
    rows.sort(key=lambda row: (safe_int(row.get("seconds_remaining"), 0), str(row.get("key") or "")))
    return rows


def active_provider_cooldown_rows(
    state: dict[str, Any],
    now_ts: float,
    *,
    safe_state_key: SafeStateKey,
    safe_detail: SafeDetail,
    safe_int: SafeInt,
    epoch_to_iso: EpochToIso,
) -> list[dict[str, Any]]:
    rows = active_cooldown_rows_from_map(
        state.get("provider_cooldowns"),
        now_ts,
        "provider",
        safe_state_key=safe_state_key,
        safe_detail=safe_detail,
        epoch_to_iso=epoch_to_iso,
    )
    rows.sort(key=lambda row: (safe_int(row.get("seconds_remaining"), 0), str(row.get("key") or "")))
    return rows


def safe_quota_cooldown_row(
    key: str,
    raw_value: dict[str, Any],
    now_ts: float,
    *,
    quota_cooldown_scope_from_key: QuotaCooldownScopeFromKey,
    safe_float: Callable[[Any, float], float],
    safe_state_key: SafeStateKey,
    safe_detail: SafeDetail,
    safe_int: SafeInt,
    epoch_to_iso: EpochToIso,
    include_active: bool = False,
) -> dict[str, Any]:
    until = safe_float(raw_value.get("until"), 0.0)
    next_probe_at = safe_float(raw_value.get("next_probe_at"), 0.0)
    row = {
        "scope": quota_cooldown_scope_from_key(key),
        "key": safe_state_key(key),
        "source": safe_detail(raw_value.get("source")),
        "model_id": safe_detail(raw_value.get("model_id")),
        "upstream_id": safe_detail(raw_value.get("upstream_id")),
        "reason": safe_detail(raw_value.get("reason")) or "quota_exhausted",
        "set_at": raw_value.get("set_at"),
        "until": until,
        "until_iso": epoch_to_iso(until),
        "seconds_remaining": max(0, int(until - now_ts)),
        "last_probe_at": raw_value.get("last_probe_at"),
        "next_probe_at": next_probe_at,
        "next_probe_at_iso": epoch_to_iso(next_probe_at),
        "probe_due": next_probe_at > 0 and next_probe_at <= now_ts,
        "probe_interval_seconds": safe_int(raw_value.get("probe_interval_seconds"), 0),
        "consecutive_probe_failures": safe_int(raw_value.get("consecutive_probe_failures"), 0),
        "detail": safe_detail(raw_value.get("detail")),
    }
    if include_active:
        row["active"] = until > now_ts
    return row


def active_quota_cooldown_rows(
    state: dict[str, Any],
    now_ts: float,
    *,
    quota_cooldown_scope_from_key: QuotaCooldownScopeFromKey,
    safe_float: Callable[[Any, float], float],
    safe_state_key: SafeStateKey,
    safe_detail: SafeDetail,
    safe_int: SafeInt,
    epoch_to_iso: EpochToIso,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cooldowns = state.get("quota_cooldowns") if isinstance(state.get("quota_cooldowns"), dict) else {}
    for key, raw in cooldowns.items():
        if not isinstance(raw, dict):
            continue
        until = safe_float(raw.get("until"), 0.0)
        if until <= now_ts:
            continue
        rows.append(
            safe_quota_cooldown_row(
                str(key),
                raw,
                now_ts,
                quota_cooldown_scope_from_key=quota_cooldown_scope_from_key,
                safe_float=safe_float,
                safe_state_key=safe_state_key,
                safe_detail=safe_detail,
                safe_int=safe_int,
                epoch_to_iso=epoch_to_iso,
            )
        )
    rows.sort(key=lambda row: (safe_int(row.get("seconds_remaining"), 0), str(row.get("scope") or ""), str(row.get("key") or "")))
    return rows


def quarantine_rows(
    state: dict[str, Any],
    *,
    safe_state_key: SafeStateKey,
    safe_detail: SafeDetail,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quarantine = state.get("quarantine") if isinstance(state.get("quarantine"), dict) else {}
    for key, raw in quarantine.items():
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "key": safe_state_key(key),
                "reason": safe_detail(raw.get("reason")) or "quarantined",
                "note": safe_detail(raw.get("note")),
                "set_at": raw.get("set_at"),
                "source": safe_detail(raw.get("source")) or "manual",
                "manual": bool(raw.get("manual", False)),
                "model_id": safe_detail(raw.get("model_id")),
                "upstream_id": safe_detail(raw.get("upstream_id")),
            }
        )
    rows.sort(key=lambda row: (str(row.get("reason") or ""), str(row.get("key") or "")))
    return rows


def failed_profile_evidence_row(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    *,
    canonical_virtual_model_id: CanonicalProfileId,
    verified_capability_row: SelectedModelRow,
    raw_benchmark_row: SelectedModelRow,
    benchmark_result_matches_current_test: BenchmarkResultMatchesCurrentTest,
    redacted_benchmark_row: RedactedBenchmarkRow,
    safe_detail: SafeDetail,
) -> dict[str, Any]:
    canonical_profile = canonical_virtual_model_id(profile_id)
    verified = verified_capability_row(canonical_profile, model, state)
    if str(verified.get("status") or "") == "failed":
        return redact_sensitive_json(
            {
                "profile_id": canonical_profile,
                "model_id": model.get("id"),
                "upstream_id": model.get("upstream_id"),
                "source": model.get("source"),
                "reason": "failed_capability_check",
                "message": safe_detail(verified.get("message")) or "model failed this profile capability check",
                "test_type": verified.get("test_type"),
                "failed_at": verified.get("failed_at"),
                "evidence_source": "verified_capability",
            }
        )

    benchmark_row = raw_benchmark_row(canonical_profile, model, state)
    if benchmark_result_matches_current_test(canonical_profile, benchmark_row) and benchmark_row.get("status") == "fail":
        benchmark = redacted_benchmark_row(canonical_profile, dict(benchmark_row))
        return redact_sensitive_json(
            {
                "profile_id": canonical_profile,
                "model_id": model.get("id"),
                "upstream_id": model.get("upstream_id"),
                "source": model.get("source"),
                "reason": "failed_benchmark",
                "message": safe_detail(benchmark.get("message")) or "model failed the last benchmark for this profile",
                "test_type": benchmark.get("test_type"),
                "failed_at": benchmark.get("ran_at"),
                "evidence_source": "benchmark_result",
            }
        )

    return {}


def model_history_row(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    rank_row: dict[str, Any],
    *,
    cooldown_key: ModelKey,
    normalized_free_access: NormalizeFreeAccess,
    raw_benchmark_row: SelectedModelRow,
    redacted_benchmark_row: RedactedBenchmarkRow,
    verified_capability_row: SelectedModelRow,
    failed_profile_evidence_row: FailedProfileEvidenceRow,
    safe_detail: SafeDetail,
    safe_int: SafeInt,
) -> dict[str, Any]:
    key = cooldown_key(model)
    stats_source = state.get("stats")
    stats_map: dict[str, Any] = stats_source if isinstance(stats_source, dict) else {}
    stats_raw = stats_map.get(key)
    stats: dict[str, Any] = stats_raw if isinstance(stats_raw, dict) else {}
    benchmark = redacted_benchmark_row(profile_id, raw_benchmark_row(profile_id, model, state))
    verified = verified_capability_row(profile_id, model, state)
    failed_evidence = failed_profile_evidence_row(profile_id, model, state)
    return {
        "rank": rank_row.get("rank"),
        "origin": rank_row.get("origin"),
        "manual_rank": rank_row.get("manual_rank"),
        "model_id": model.get("id"),
        "upstream_id": model.get("upstream_id"),
        "source": model.get("source"),
        "free_access": rank_row.get("free_access") or normalized_free_access(model),
        "score": rank_row.get("score"),
        "score_base": rank_row.get("score_base"),
        "score_adjustment": rank_row.get("score_adjustment"),
        "benchmark_bonus": rank_row.get("benchmark_bonus"),
        "verified_bonus": rank_row.get("verified_bonus"),
        "score_decay_factor": rank_row.get("score_decay_factor"),
        "failure_penalty": rank_row.get("failure_penalty"),
        "evidence_status": rank_row.get("evidence_status"),
        "evidence_reason": rank_row.get("evidence_reason"),
        "benchmark_evidence_status": rank_row.get("benchmark_evidence_status"),
        "benchmark_evidence_reason": rank_row.get("benchmark_evidence_reason"),
        "verified_evidence_status": rank_row.get("verified_evidence_status"),
        "verified_evidence_reason": rank_row.get("verified_evidence_reason"),
        "last_evidence_at": rank_row.get("last_evidence_at"),
        "requests": safe_int(stats.get("requests"), 0),
        "successes": safe_int(stats.get("successes"), 0),
        "failures": safe_int(stats.get("failures"), 0),
        "consecutive_failures": safe_int(stats.get("consecutive_failures"), 0),
        "latency_ewma": stats.get("latency_ewma"),
        "last_success_at": stats.get("last_success_at"),
        "last_failure_at": stats.get("last_failure_at"),
        "last_failure_reason": stats.get("last_failure_reason"),
        "benchmark_status": benchmark.get("status") or "unknown",
        "benchmark_test_type": benchmark.get("test_type"),
        "benchmark_ran_at": benchmark.get("ran_at"),
        "benchmark_latency_seconds": benchmark.get("latency_seconds"),
        "benchmark_message": safe_detail(benchmark.get("message")),
        "verified_status": verified.get("status") or "unknown",
        "verified_capability": verified.get("capability"),
        "verified_at": verified.get("verified_at"),
        "verified_failed_at": verified.get("failed_at"),
        "failed_profile_evidence": bool(failed_evidence),
        "failed_profile_evidence_reason": failed_evidence.get("reason"),
        "failed_profile_evidence_message": failed_evidence.get("message"),
        "failed_profile_evidence_at": failed_evidence.get("failed_at"),
    }


def candidate_rank_rows(
    profile_id: str,
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    state: dict[str, Any],
    *,
    canonical_virtual_model_id: CanonicalProfileId,
    model_score_explanation: SelectedModelRow,
    verified_capability_row: SelectedModelRow,
    failed_profile_evidence_row: FailedProfileEvidenceRow,
    model_route_competence: SelectedModelValue,
    gate_capability_by_profile: dict[str, str],
    model_capability_provenance: ModelCapabilityProvenance,
    normalized_free_access: NormalizeFreeAccess,
    safe_int: SafeInt,
    limit: int = 5,
) -> list[dict[str, Any]]:
    canonical_profile_id = canonical_virtual_model_id(profile_id)
    policy_profile_id = canonical_virtual_model_id(str(profile.get("base_profile") or canonical_profile_id))
    manual_ranks = {
        str(model_id): index + 1
        for index, model_id in enumerate(profile.get("models") or [])
        if str(model_id)
    }
    rows: list[dict[str, Any]] = []
    for index, model in enumerate(candidates[:limit]):
        model_id = str(model.get("id") or "")
        manual_rank = manual_ranks.get(model_id)
        origin = "auto"
        if profile.get("mode") == "manual_order":
            origin = "manual" if manual_rank is not None else "auto_tail"
        score = model_score_explanation(policy_profile_id, model, state)
        verified = verified_capability_row(policy_profile_id, model, state)
        failed_evidence = failed_profile_evidence_row(policy_profile_id, model, state)
        competence = model_route_competence(policy_profile_id, model, state)
        gate_capability = gate_capability_by_profile.get(policy_profile_id)
        rows.append(
            {
                "rank": index + 1,
                "origin": origin,
                "manual_rank": manual_rank,
                "model_id": model_id,
                "upstream_id": model.get("upstream_id"),
                "source": model.get("source"),
                "name": model.get("name"),
                "score": score["score_total"],
                "score_base": score["score_base"],
                "score_adjustment": score["score_adjustment"],
                "benchmark_bonus": score["benchmark_bonus"],
                "verified_bonus": score["verified_bonus"],
                "score_decay_factor": score["score_decay_factor"],
                "failure_penalty": score["failure_penalty"],
                "evidence_status": score["evidence_status"],
                "evidence_reason": score["evidence_reason"],
                "benchmark_evidence_status": score["benchmark_evidence_status"],
                "benchmark_evidence_reason": score["benchmark_evidence_reason"],
                "verified_evidence_status": score["verified_evidence_status"],
                "verified_evidence_reason": score["verified_evidence_reason"],
                "last_evidence_at": score["last_evidence_at"],
                "context_length": safe_int(model.get("context_length"), 0),
                "free_access": normalized_free_access(model),
                "verified_capability": verified.get("capability"),
                "verified_status": verified.get("status") or "unknown",
                "verified_at": verified.get("verified_at"),
                "failed_profile_evidence": bool(failed_evidence),
                "failed_profile_evidence_reason": failed_evidence.get("reason"),
                "failed_profile_evidence_message": failed_evidence.get("message"),
                "competence": competence,
                "capability_provenance": model_capability_provenance(model, gate_capability),
                "reference_confidence": str(model.get("reference_confidence") or ""),
                "route_eligible": competence not in {"failed", "claimed_default"},
            }
        )
    return rows


def build_admin_health_summary(
    *,
    invokable_model_count: int,
    profile_rows: dict[str, Any],
    critical_profile_ids: tuple[str, ...],
    active_model_cooldowns: list[dict[str, Any]],
    active_provider_cooldowns: list[dict[str, Any]],
    active_quota_cooldowns: list[dict[str, Any]],
    quarantines: list[dict[str, Any]],
    canary_status: str,
    auto_benchmark: dict[str, Any],
    compression: dict[str, Any],
    catalog_stale: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    status = "ok"
    if invokable_model_count <= 0:
        status = "fail"
        reasons.append("no_invokable_models")
    for profile_id in critical_profile_ids:
        row = profile_rows.get(profile_id)
        candidate_count = int(row.get("candidate_count") or 0) if isinstance(row, dict) else 0
        if candidate_count <= 0:
            status = "fail"
            reasons.append(f"critical_profile_without_candidates:{profile_id}")
    for condition, reason in (
        (active_model_cooldowns, "active_model_cooldowns"),
        (active_provider_cooldowns, "active_provider_cooldowns"),
        (active_quota_cooldowns, "active_quota_cooldowns"),
        (quarantines, "active_quarantine"),
        (canary_status == "fail", "canary_failed"),
        (isinstance(auto_benchmark.get("last_error"), dict), "auto_benchmark_background_error"),
        (compression.get("health_digest") == "compression_errors_repeated", "compression_errors_repeated"),
        (catalog_stale, "catalog_stale"),
    ):
        if condition:
            if status != "fail":
                status = "degraded"
            reasons.append(reason)

    actions: list[str] = []
    if status == "fail":
        actions.append(ACTION_CHECK_PROVIDERS)
    if active_model_cooldowns:
        actions.append(ACTION_MODEL_COOLDOWNS)
    if active_provider_cooldowns:
        actions.append(ACTION_PROVIDER_COOLDOWNS)
    if active_quota_cooldowns:
        actions.append(ACTION_QUOTA_COOLDOWNS)
    if quarantines:
        actions.append(ACTION_QUARANTINE)
    if canary_status == "fail":
        actions.append(ACTION_CANARY_FAILED)
    if isinstance(auto_benchmark.get("last_error"), dict):
        actions.append(ACTION_AUTO_BENCHMARK_ERROR)
    if compression.get("health_digest") == "compression_errors_repeated":
        actions.append(ACTION_COMPRESSION_ERRORS)
    if not actions:
        actions.append(ACTION_DEFAULT)
    return {
        "status": status,
        "reasons": reasons,
        "actions": actions,
    }


def build_admin_catalog_summary(
    catalog: dict[str, Any],
    *,
    model_count: int,
    invokable_count: int,
    available_count: int,
) -> dict[str, Any]:
    summary = {
        "generated_at": catalog.get("generated_at"),
        "min_context_length": catalog.get("min_context_length"),
        "allow_paid_fallback": False,
        "model_count": model_count,
        "invokable_count": invokable_count,
        "available_count": available_count,
        "providers": catalog.get("providers") or {},
    }
    if catalog.get("stale"):
        summary["stale"] = True
        summary["stale_reason"] = catalog.get("stale_reason") or "unknown"
    return summary


def build_catalog_refresh_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    models = [model for model in catalog.get("models", []) if isinstance(model, dict)]
    return {
        "generated_at": catalog.get("generated_at"),
        "min_context_length": catalog.get("min_context_length"),
        "allow_paid_fallback": False,
        "model_count": len(models),
        "invokable_count": sum(1 for model in models if model.get("invokable")),
        "providers": catalog.get("providers") or {},
    }


def build_admin_runtime_summary(
    *,
    active_cooldowns_count: int,
    active_model_cooldowns_count: int,
    active_provider_cooldowns_count: int,
    active_quota_cooldowns_count: int,
    quarantine_count: int,
    stats_records: int,
    provider_stats_records: int,
    provider_error_count: int,
    model_error_count: int,
    last_route_count: int,
    compression_error_count: int,
    compression_last_status: Any,
    state_diagnostics: dict[str, Any],
    canary: dict[str, Any],
    auto_benchmark: dict[str, Any],
) -> dict[str, Any]:
    return {
        "active_cooldowns_count": active_cooldowns_count,
        "active_model_cooldowns_count": active_model_cooldowns_count,
        "active_provider_cooldowns_count": active_provider_cooldowns_count,
        "active_quota_cooldowns_count": active_quota_cooldowns_count,
        "quarantine_count": quarantine_count,
        "stats_records": stats_records,
        "provider_stats_records": provider_stats_records,
        "provider_error_count": provider_error_count,
        "model_error_count": model_error_count,
        "last_route_count": last_route_count,
        "compression_error_count": compression_error_count,
        "compression_last_status": compression_last_status,
        "state_diagnostics": state_diagnostics,
        "canary": canary,
        "auto_benchmark": auto_benchmark,
    }


def build_admin_runtime_section(
    *,
    state: dict[str, Any],
    recovery: AdminRecoverySnapshot,
    runtime: AdminRuntimeSnapshot,
    compression: dict[str, Any],
    state_diagnostics: dict[str, Any],
    safe_int: SafeInt,
) -> dict[str, Any]:
    return build_admin_runtime_summary(
        active_cooldowns_count=len(recovery.active_cooldowns),
        active_model_cooldowns_count=len(recovery.active_model_cooldowns),
        active_provider_cooldowns_count=len(recovery.active_provider_cooldowns),
        active_quota_cooldowns_count=len(recovery.active_quota_cooldowns),
        quarantine_count=len(recovery.quarantines),
        stats_records=_mapping_record_count(state, "stats"),
        provider_stats_records=_mapping_record_count(state, "provider_stats"),
        provider_error_count=len(runtime.provider_errors),
        model_error_count=len(runtime.model_errors),
        last_route_count=len(runtime.last_routes),
        compression_error_count=safe_int(compression.get("error_count"), 0),
        compression_last_status=compression.get("last_status"),
        state_diagnostics=state_diagnostics,
        canary=runtime.canary,
        auto_benchmark=runtime.auto_benchmark,
    )


def build_admin_profile_row(
    *,
    profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    selected_origin: str | None,
    selected_score: dict[str, Any],
    selected_verified: dict[str, Any],
    selected_competence: Any,
    route_candidates: list[dict[str, Any]],
    policy_candidates: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
    failed_candidates: list[dict[str, Any]],
    last_route: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "ok" if candidates else "fail",
        "mode": profile.get("mode"),
        "manual_models_count": len(profile.get("models") or []),
        "auto_tail": bool(profile.get("auto_tail", True)),
        "candidate_count": len(candidates),
        "selected_model": selected.get("id") if selected else None,
        "selected_upstream": selected.get("upstream_id") if selected else None,
        "selected_origin": selected_origin,
        "selected_score": selected_score.get("score_total") if selected else None,
        "selected_score_details": selected_score,
        "selected_competence": selected_competence,
        "selected_verified_status": selected_verified.get("status") or "unknown",
        "selected_verified_capability": selected_verified.get("capability"),
        "route_candidate_ids": [str(model.get("id") or "") for model in route_candidates],
        "policy_candidate_ids": [str(model.get("id") or "") for model in policy_candidates],
        "top_candidates": top_candidates,
        "failed_profile_candidates": failed_candidates,
        "last_route": last_route,
        "requirements": profile.get("requirements") or {},
    }


def selected_profile_origin(profile: dict[str, Any], selected: dict[str, Any] | None) -> str | None:
    if selected is None:
        return None
    selected_id = selected.get("id")
    if profile.get("mode") == "manual_order":
        manual_ids = {str(item) for item in (profile.get("models") or [])}
        return "manual" if selected_id in manual_ids else "auto_tail"
    return "auto"


def build_admin_profile_rows(
    *,
    profiles: dict[str, Any],
    available_models: list[dict[str, Any]],
    state: dict[str, Any],
    last_routes: dict[str, Any],
    candidates_for_profile: CandidatesForProfile,
    route_competent_candidates: RouteCompetentCandidates,
    candidate_rank_rows: CandidateRankRows,
    verified_capability_row: SelectedModelRow,
    model_score_explanation: SelectedModelRow,
    model_route_competence: SelectedModelValue,
    failed_profile_evidence_row: FailedProfileEvidenceRow,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for profile_id, profile in profiles.items():
        policy_profile_id = str(profile.get("base_profile") or profile_id)
        candidates = candidates_for_profile(profile_id, available_models, state, profile)
        # selected reflects what routing serves (competence-gated top); the full ranked list
        # in top_candidates stays ungated so failed/unproven candidates remain visible.
        route_candidates = route_competent_candidates(policy_profile_id, candidates, state)
        policy_candidates = route_competent_candidates(
            policy_profile_id,
            available_models,
            state,
        )
        selected = route_candidates[0] if route_candidates else None
        selected_verified = verified_capability_row(policy_profile_id, selected, state) if selected else {}
        selected_score = model_score_explanation(policy_profile_id, selected, state) if selected else {}
        failed_candidates = [
            failed_profile_evidence_row(policy_profile_id, model, state)
            for model in candidates
        ]
        failed_candidates = [row for row in failed_candidates if row]
        last_route = last_routes.get(profile_id)
        rows[profile_id] = build_admin_profile_row(
            profile=profile,
            candidates=candidates,
            selected=selected,
            selected_origin=selected_profile_origin(profile, selected),
            selected_score=selected_score,
            selected_verified=selected_verified,
            selected_competence=model_route_competence(policy_profile_id, selected, state) if selected else None,
            route_candidates=route_candidates,
            policy_candidates=policy_candidates,
            top_candidates=candidate_rank_rows(profile_id, candidates, profile, state),
            failed_candidates=failed_candidates,
            last_route=last_route if isinstance(last_route, dict) else {},
        )
    return rows


def build_admin_performance_history_rows(
    *,
    profiles: dict[str, Any],
    available_models: list[dict[str, Any]],
    state: dict[str, Any],
    last_routes: dict[str, Any],
    candidates_for_profile: CandidatesForProfile,
    candidate_rank_rows: PerformanceCandidateRankRows,
    route_competent_candidates: RouteCompetentCandidates,
    model_history_row: ModelHistoryRow,
    model_auto_score: SelectedModelValue,
    model_route_competence: SelectedModelValue,
    fallback_attempt_rows: LastRouteFallbackRows,
    safe_int: SafeInt,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for profile_id, profile in profiles.items():
        policy_profile_id = str(profile.get("base_profile") or profile_id)
        candidates = candidates_for_profile(profile_id, available_models, state, profile)
        rank_rows = candidate_rank_rows(profile_id, candidates, profile, state, limit=5)
        rank_by_model = {str(row.get("model_id") or ""): row for row in rank_rows}
        # selected reflects what routing actually serves, not just the top-ranked candidate.
        route_candidates = route_competent_candidates(policy_profile_id, candidates, state)
        selected = route_candidates[0] if route_candidates else None
        last_route = last_routes.get(profile_id) if isinstance(last_routes.get(profile_id), dict) else {}
        compression = last_route.get("compression") if isinstance(last_route.get("compression"), dict) else {}
        candidate_rows = [
            model_history_row(policy_profile_id, model, state, rank_by_model.get(str(model.get("id") or ""), {}))
            for model in candidates[:5]
        ]
        rows[profile_id] = {
            "profile_id": profile_id,
            "mode": profile.get("mode"),
            "candidate_count": len(candidates),
            "selected_model": selected.get("id") if selected else None,
            "selected_upstream": selected.get("upstream_id") if selected else None,
            "selected_source": selected.get("source") if selected else None,
            "selected_score": round(model_auto_score(policy_profile_id, selected, state), 1) if selected else None,
            "selected_competence": model_route_competence(policy_profile_id, selected, state) if selected else None,
            "last_route_status": last_route.get("status"),
            "last_route_reason": last_route.get("reason"),
            "last_route_model": last_route.get("model_id"),
            "last_route_upstream": last_route.get("upstream_id"),
            "last_route_request_id": last_route.get("request_id"),
            "last_route_seen_at": last_route.get("seen_at"),
            "last_route_latency_seconds": last_route.get("latency_seconds"),
            "last_route_attempt_count": safe_int(last_route.get("attempt_count"), 0),
            "compression_status": compression.get("status") if compression else None,
            "compression_mode": compression.get("mode") if compression else None,
            "compression_estimated_saved_chars": compression.get("estimated_saved_chars") if compression else 0,
            "compression_estimated_saved_tokens": compression.get("estimated_saved_tokens") if compression else 0,
            "compression_strategy_mix": compression.get("strategies") if compression else {},
            "fallback_reasons": fallback_attempt_rows(last_route),
            "models": candidate_rows,
        }
    return rows


def build_admin_status_payload(
    *,
    status: str,
    reasons: list[str],
    generated_at: str,
    catalog: dict[str, Any],
    profiles: dict[str, Any],
    fusion: dict[str, Any],
    critical_profiles: tuple[str, ...],
    runtime: dict[str, Any],
    compression: dict[str, Any],
    cooldowns: list[dict[str, Any]],
    model_cooldowns: list[dict[str, Any]],
    provider_cooldowns: list[dict[str, Any]],
    quota_cooldowns: list[dict[str, Any]],
    quota_probe_results: dict[str, Any],
    provider_errors: dict[str, Any],
    model_errors: dict[str, Any],
    last_routes: dict[str, Any],
    performance_history: dict[str, Any],
    quarantine: list[dict[str, Any]],
    actions: list[str],
) -> dict[str, Any]:
    payload = {
        "status": status,
        "reasons": reasons,
        "generated_at": generated_at,
        "catalog": catalog,
        "profiles": profiles,
        "fusion": fusion,
        "critical_profiles": critical_profiles,
        "runtime": runtime,
        "compression": compression,
        "cooldowns": cooldowns,
        "model_cooldowns": model_cooldowns,
        "provider_cooldowns": provider_cooldowns,
        "quota_cooldowns": quota_cooldowns,
        "quota_probe_results": quota_probe_results,
        "provider_errors": provider_errors,
        "model_errors": model_errors,
        "last_routes": last_routes,
        "performance_history": performance_history,
        "quarantine": quarantine,
        "actions": actions,
    }
    return {field: payload[field] for field in ADMIN_STATUS_TOP_LEVEL_FIELDS}


def build_admin_status_document(
    *,
    catalog: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    critical_profile_ids: tuple[str, ...],
    ports: AdminStatusBuildPorts,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        state = {}
    availability = build_admin_model_availability(
        catalog=catalog,
        state=state,
        model_on_cooldown=ports.model_on_cooldown,
        model_is_quarantined=ports.model_is_quarantined,
    )
    profiles = ports.normalized_virtual_profiles(config)
    runtime_state = ports.redact_runtime_state(state)
    runtime_snapshot = build_admin_runtime_snapshot(runtime_state)
    profile_rows = build_admin_profile_rows(
        profiles=profiles,
        available_models=availability.available_models,
        state=state,
        last_routes=runtime_snapshot.last_routes,
        candidates_for_profile=ports.candidates_for_profile,
        route_competent_candidates=ports.route_competent_candidates,
        candidate_rank_rows=ports.candidate_rank_rows,
        verified_capability_row=ports.verified_capability_row,
        model_score_explanation=ports.model_score_explanation,
        model_route_competence=ports.model_route_competence,
        failed_profile_evidence_row=ports.failed_profile_evidence_row,
    )

    recovery_snapshot = build_admin_recovery_snapshot(
        state=state,
        active_model_cooldown_rows=ports.active_model_cooldown_rows,
        active_provider_cooldown_rows=ports.active_provider_cooldown_rows,
        active_quota_cooldown_rows=ports.active_quota_cooldown_rows,
        quarantine_rows=ports.quarantine_rows,
    )
    performance_history = build_admin_performance_history_rows(
        profiles=profiles,
        available_models=availability.available_models,
        state=state,
        last_routes=runtime_snapshot.last_routes,
        candidates_for_profile=ports.candidates_for_profile,
        candidate_rank_rows=ports.candidate_rank_rows,
        route_competent_candidates=ports.route_competent_candidates,
        model_history_row=ports.model_history_row,
        model_auto_score=ports.model_auto_score,
        model_route_competence=ports.model_route_competence,
        fallback_attempt_rows=ports.fallback_attempt_rows,
        safe_int=ports.safe_int,
    )
    compression = ports.compression_observability(config, runtime_state)
    state_diagnostics = ports.runtime_state_diagnostics(runtime_state)

    health_summary = build_admin_health_summary(
        invokable_model_count=len(availability.invokable_models),
        profile_rows=profile_rows,
        critical_profile_ids=critical_profile_ids,
        active_model_cooldowns=recovery_snapshot.active_model_cooldowns,
        active_provider_cooldowns=recovery_snapshot.active_provider_cooldowns,
        active_quota_cooldowns=recovery_snapshot.active_quota_cooldowns,
        quarantines=recovery_snapshot.quarantines,
        canary_status=runtime_snapshot.canary_status,
        auto_benchmark=runtime_snapshot.auto_benchmark,
        compression=compression,
        catalog_stale=bool(catalog.get("stale")),
    )

    return build_admin_status_payload(
        status=health_summary["status"],
        reasons=health_summary["reasons"],
        generated_at=ports.now_iso(),
        catalog=build_admin_catalog_summary(
            catalog,
            model_count=len(availability.models),
            invokable_count=len(availability.invokable_models),
            available_count=len(availability.available_models),
        ),
        profiles=profile_rows,
        fusion=ports.fusion_payload(config, state),
        critical_profiles=critical_profile_ids,
        runtime=build_admin_runtime_section(
            state=state,
            recovery=recovery_snapshot,
            runtime=runtime_snapshot,
            compression=compression,
            state_diagnostics=state_diagnostics,
            safe_int=ports.safe_int,
        ),
        compression=compression,
        cooldowns=recovery_snapshot.active_cooldowns,
        model_cooldowns=recovery_snapshot.active_model_cooldowns,
        provider_cooldowns=recovery_snapshot.active_provider_cooldowns,
        quota_cooldowns=recovery_snapshot.active_quota_cooldowns,
        quota_probe_results=runtime_snapshot.quota_probe_results,
        provider_errors=runtime_snapshot.provider_errors,
        model_errors=runtime_snapshot.model_errors,
        last_routes=runtime_snapshot.last_routes,
        performance_history=performance_history,
        quarantine=recovery_snapshot.quarantines,
        actions=health_summary["actions"],
    )


@dataclass(frozen=True)
class FusionAdminStatusBuilder:
    normalize_fusion_config: Callable[..., dict[str, Any]]
    fusion_call_multiplier: Callable[[dict[str, Any]], int]
    judge_history_limit: int
    reason_code_pattern: re.Pattern[str]

    def safe_reason_code(self, value: Any) -> str:
        code = str(value or "").strip()
        if self.reason_code_pattern.fullmatch(code):
            return code
        return "unsafe_reason_code" if code else "none"

    def safe_last_run(self, state: dict[str, Any]) -> dict[str, Any]:
        raw = state.get("last_fusion_run")
        if not isinstance(raw, dict):
            fusion_state = state.get("fusion") if isinstance(state.get("fusion"), dict) else {}
            raw = fusion_state.get("last_run") if isinstance(fusion_state.get("last_run"), dict) else {}
        safe = redact_sensitive_json(raw) if isinstance(raw, dict) else {}
        if not isinstance(safe, dict):
            return {}
        projected = {key: safe.get(key) for key in FUSION_LAST_RUN_FIELDS if key in safe}
        if "reason_code" in projected:
            projected["reason_code"] = self.safe_reason_code(projected.get("reason_code"))
        return projected

    def judge_failure_stats(self, state: dict[str, Any]) -> dict[str, Any]:
        source = state if isinstance(state, dict) else {}
        raw = source.get("fusion_judge_history")
        history = [item for item in raw if isinstance(item, str)] if isinstance(raw, list) else []
        samples = len(history)
        invalid_json = history.count("invalid_json")
        invalid_schema = history.count("invalid_schema")
        invalid_total = invalid_json + invalid_schema
        return {
            "samples": samples,
            "invalid_json": invalid_json,
            "invalid_schema": invalid_schema,
            "invalid_total": invalid_total,
            "invalid_rate": round(invalid_total / samples, 4) if samples else 0.0,
            "window_limit": self.judge_history_limit,
        }

    def quality_signal_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        source = state if isinstance(state, dict) else {}
        raw = source.get("fusion_quality_signal")
        if not isinstance(raw, dict):
            return {}
        return {model_id: row for model_id, row in raw.items() if isinstance(row, dict)}

    def observability(self, config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        fusion = self.normalize_fusion_config(config.get("fusion"), strict=False)
        source = state if isinstance(state, dict) else {}
        last_run = self.safe_last_run(source)
        degraded_flags = last_run.get("degraded_flags") if isinstance(last_run.get("degraded_flags"), dict) else {}
        last_status = str(last_run.get("status") or "never")
        readiness = "disabled"
        if fusion.get("enabled"):
            if last_status in {"failed", "fail", "error"}:
                readiness = "failed"
            elif any(bool(value) for value in degraded_flags.values()):
                readiness = "degraded"
            else:
                readiness = "ready"
        return {
            "readiness": readiness,
            "last_run": last_run,
            "stage_timeline": last_run.get("stage_latency_seconds")
            if isinstance(last_run.get("stage_latency_seconds"), dict)
            else {},
            "call_multiplier_estimate": self.fusion_call_multiplier(fusion),
            "budget_policy": fusion.get("budget_policy"),
            "strict_zero_only": False,
            "free_models_only": fusion.get("budget_policy") == "strict_zero",
            "paid_models_excluded": True,
            "free_quota_models_allowed": True,
            "paid_or_quota_models_excluded": False,
            "raw_output_policy": fusion.get("raw_output_policy"),
            "judge_failure_stats": self.judge_failure_stats(source),
            "peer_ranking_quality_signal": self.quality_signal_summary(source),
        }

    def fusion_payload(self, config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        fusion = self.normalize_fusion_config(config.get("fusion"), strict=False)
        return {
            "config": fusion,
            "observability": self.observability({"fusion": fusion}, state),
        }
