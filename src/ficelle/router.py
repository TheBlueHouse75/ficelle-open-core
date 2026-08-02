"""Local OpenAI-compatible strict-zero model router for AI agents.

MVP scope:
- discovers OpenRouter + Nous/RMS catalog entries that are strict-zero/free, support tools,
  and meet the configured context threshold;
- resolves provider credentials without printing secrets;
- exposes /v1/models and /v1/chat/completions for OpenAI-compatible clients;
- never falls back to paid models.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence
from urllib.parse import parse_qs, urlparse

from ficelle import license_ops, request_log, update as update_service
from ficelle.config_store import ConfigStore
try:
    from ficelle_pro.compression import (
        DEFAULT_COMPRESSION_CONFIG,
        clear as clear_compression_store,
        compression_marker,
        compress_block,
        get_original as get_compression_original,
        normalize_compression_settings,
        put_original,
        stats as compression_store_stats,
    )
except ImportError:  # pragma: no cover - exercised only in core-only installs
    # Native compression is a closed Pro engine. Only DEFAULT_COMPRESSION_CONFIG
    # (in DEFAULT_CONFIG) and normalize_compression_settings (settings/observability
    # ports) are reached on the free path, with compression resolving to mode "off";
    # the store/compress primitives stay None because the free tier never executes
    # them. See docs/prds/open-core-extraction-prd.md (Lot 1).
    from ficelle.compression_fallback import (
        DEFAULT_COMPRESSION_CONFIG,
        normalize_compression_settings,
    )

    clear_compression_store = None
    compression_marker = None
    compress_block = None
    get_compression_original = None
    put_original = None
    compression_store_stats = None
from ficelle.domain_models import (
    CatalogModel,
    SelectionPurpose,
    SelectionResult,
)
from ficelle.failures import (
    BENCHMARK_ROUTE_BLOCKING_REASONS,
    PROVIDER_ERROR_REASONS,
    PROVIDER_SCOPED_COOLDOWN_REASONS,
    bad_request_body as bad_request_body_result,
    build_upstream_failure_error as build_upstream_failure_error_result,
    classify_failure as classify_failure_result,
    cooldown_policy_for_reason,
    safe_error_body as safe_error_body_result,
    upstream_failure_actions as upstream_failure_actions_result,
)
from ficelle.provider_credentials import (
    PROVIDER_ENV_ALIASES,
    PROVIDER_KEY_VALIDATORS,
    PROVIDER_SERVICE_ALIASES,
    generic_provider_credential_aliases,
    generic_provider_credential_activation_fingerprint as generic_provider_credential_activation_fingerprint_use_case,
    is_usable_openrouter_key,
    provider_primary_service as provider_primary_service_use_case,
    remove_provider_key as remove_provider_key_use_case,
    resolve_provider_credentials,
    store_provider_key as store_provider_key_use_case,
)
from ficelle.selection import (
    SelectionPolicy,
    candidates_for_profile as candidates_for_profile_policy,
    manual_order_candidates as manual_order_candidates_policy,
    sort_available_for_virtual_model as sort_available_for_virtual_model_policy,
    typed_catalog_rows,
)
from ficelle.runtime_paths import RuntimePaths
from ficelle.routing_policy import (
    competence_gate_result as policy_competence_gate_result,
    competence_label as policy_competence_label,
    model_has_any,
    model_matches_profile_requirements as policy_model_matches_profile_requirements,
)
from ficelle.state_store import (
    StateStore,
    advisory_file_lock,
    parse_iso_timestamp,
    runtime_evidence_timestamp_value,
)
import ficelle.state_store as state_store_module
from ficelle.use_cases.benchmark import (
    BenchmarkMediaError,
    BenchmarkEvidencePorts,
    BenchmarkMediaPorts,
    BenchmarkProbeBuilder,
    BenchmarkRunner,
    ProbeRegistry,
    apply_canary_result_to_state,
    benchmark_audio_part as benchmark_audio_part_use_case,
    benchmark_file_data_url as benchmark_file_data_url_use_case,
    benchmark_media_answer as benchmark_media_answer_use_case,
    benchmark_media_settings as benchmark_media_settings_use_case,
    build_compression_benchmark_source,
    build_long_context_benchmark_source,
    benchmark_result_is_aged as benchmark_result_is_aged_use_case,
    benchmark_result_matches_current_test as benchmark_result_matches_current_test_use_case,
    benchmark_result_test_type_matches as benchmark_result_test_type_matches_use_case,
    configured_benchmark_media_path as configured_benchmark_media_path_use_case,
    expected_probe_test_type,
    extract_message_text as benchmark_extract_message_text,
    extract_reasoning_text as benchmark_extract_reasoning_text,
    extract_tool_calls as benchmark_extract_tool_calls,
    has_deliverable_message as benchmark_has_deliverable_message,
    json_fields_match as benchmark_json_fields_match,
    max_benchmark_media_bytes as max_benchmark_media_bytes_use_case,
    pick_vision_fixture as pick_vision_fixture_use_case,
    canary_status_from_benchmark_result,
    mark_stale_benchmark_row as mark_stale_benchmark_row_use_case,
    profile_ids_from_admin_body as profile_ids_from_admin_body_use_case,
    record_benchmark_failure_in_state,
    record_benchmark_result_in_state,
    redacted_benchmark_row as redacted_benchmark_row_use_case,
    run_benchmark_profiles,
    stale_score_decay_factor as stale_score_decay_factor_use_case,
    tool_call_arguments as benchmark_tool_call_arguments,
    validate_probe_response,
    vision_benchmark_fixtures as vision_benchmark_fixtures_use_case,
)
from ficelle.use_cases.capability_discovery import (
    CapabilityDiscoveryJob,
    clear_background_job_error_in_state,
    discovery_eligible_models as discovery_eligible_models_use_case,
    model_due_capabilities as model_due_capabilities_use_case,
    models_needing_discovery as models_needing_discovery_use_case,
    probeable_capability_profiles as probeable_capability_profiles_use_case,
    record_background_job_error_in_state,
)
from ficelle.use_cases.catalog_refresh import (
    CatalogRefreshRunner,
    CatalogRefreshPorts,
    load_or_refresh_catalog as load_or_refresh_catalog_use_case,
    publish_catalog as publish_catalog_use_case,
)
from ficelle.use_cases.catalog_fingerprint import (
    CatalogFingerprintPorts,
    catalog_config_fingerprint as catalog_config_fingerprint_use_case,
    catalog_config_structural_fingerprint as catalog_config_structural_fingerprint_use_case,
)
from ficelle.use_cases.chat_completion import (
    DEFAULT_CHAT_COMPLETION_MODEL,
    COMPRESSION_PENDING_MARKER as CHAT_COMPLETION_COMPRESSION_PENDING_MARKER,
    AttemptCooldownRequest,
    ChatCompletionAttemptPlan,
    ChatCompletionAttemptPorts,
    ChatCompletionRouteTelemetry,
    ChatCompletionRouter,
    CompressionRoutePlan,
    StreamingResponseStartInput,
    build_streaming_response_start,
    normalize_chat_completion_request,
    prepare_compression_route_body as prepare_chat_compression_route_body,
)
from ficelle.use_cases.admin_status import (
    AdminStateBuilder,
    AdminStateBuildPorts,
    AdminStatusBuilder,
    AdminStatusBuildPorts,
    FusionAdminStatusBuilder,
    catalog_with_auto_scores as catalog_with_auto_scores_use_case,
    candidate_rank_rows as candidate_rank_rows_use_case,
    active_admin_jobs as active_admin_jobs_use_case,
    active_quota_cooldown_rows as active_quota_cooldown_rows_use_case,
    active_model_cooldown_rows as active_model_cooldown_rows_use_case,
    active_provider_cooldown_rows as active_provider_cooldown_rows_use_case,
    admin_job_matches,
    build_admin_job_row,
    build_catalog_refresh_summary,
    cached_catalog_stale_reason as cached_catalog_stale_reason_use_case,
    stale_reason_keeps_last_known,
    failed_profile_evidence_row as failed_profile_evidence_row_use_case,
    filter_cached_catalog_to_enabled_providers as filter_cached_catalog_to_enabled_providers_use_case,
    load_cached_catalog_for_admin_status as load_cached_catalog_for_admin_status_use_case,
    model_history_row as model_history_row_use_case,
    quarantine_rows as quarantine_rows_use_case,
    record_admin_job_finish,
    record_admin_job_start,
    safe_quota_cooldown_row as safe_quota_cooldown_row_use_case,
)
from ficelle.use_cases.admin_security import (
    admin_origin_allowed,
    admin_token_matches,
    is_loopback_origin as is_loopback_origin_use_case,
)
from ficelle.use_cases.cooldowns import (
    CooldownMutationPorts,
    CooldownReadPorts,
    CooldownStatsPorts,
    CooldownSuccessPorts,
    CooldownWritePorts,
    QuarantinePorts,
    clear_all_cooldowns_in_state as clear_all_cooldowns_in_state_use_case,
    clear_model_cooldown_in_state as clear_model_cooldown_in_state_use_case,
    clear_model_error_in_state as clear_model_error_in_state_use_case,
    clear_provider_cooldown_in_state as clear_provider_cooldown_in_state_use_case,
    clear_provider_error_in_state as clear_provider_error_in_state_use_case,
    clear_quota_cooldown_in_state as clear_quota_cooldown_in_state_use_case,
    clear_quota_model_errors_for_source_in_state as clear_quota_model_errors_for_source_in_state_use_case,
    cooldown_key as cooldown_key_use_case,
    model_is_quarantined as model_is_quarantined_use_case,
    model_on_cooldown as model_on_cooldown_use_case,
    model_quarantine as model_quarantine_use_case,
    provider_on_cooldown as provider_on_cooldown_use_case,
    quarantine_row as quarantine_row_use_case,
    record_success_in_state as record_success_in_state_use_case,
    record_model_error_in_state as record_model_error_in_state_use_case,
    record_provider_error_in_state as record_provider_error_in_state_use_case,
    quota_cooldown_key as quota_cooldown_key_use_case,
    quota_cooldown_key_for_scope as quota_cooldown_key_for_scope_use_case,
    quota_cooldown_matches_model as quota_cooldown_matches_model_use_case,
    quota_probe_backoff_seconds as quota_probe_backoff_seconds_use_case,
    quota_cooldown_scope as quota_cooldown_scope_use_case,
    quota_cooldown_scope_from_key as quota_cooldown_scope_from_key_use_case,
    set_billing_quarantine_in_state as set_billing_quarantine_in_state_use_case,
    set_cooldown as set_cooldown_use_case,
    set_provider_cooldown_in_state as set_provider_cooldown_in_state_use_case,
    set_quota_cooldown_in_state as set_quota_cooldown_in_state_use_case,
    set_model_not_found_quarantine_in_state as set_model_not_found_quarantine_in_state_use_case,
    set_no_free_quota_quarantine_in_state as set_no_free_quota_quarantine_in_state_use_case,
    set_quarantine_in_state as set_quarantine_in_state_use_case,
    source_has_active_quota_cooldown as source_has_active_quota_cooldown_use_case,
    update_failure_stats as update_failure_stats_use_case,
    update_provider_failure_stats as update_provider_failure_stats_use_case,
    update_provider_success_stats as update_provider_success_stats_use_case,
    update_success_stats as update_success_stats_use_case,
)
# Fusion is NOT yet optional: its config layer (FusionConfigPorts,
# normalize_fusion_config, fusion_visible_in_model_list, fusion_call_multiplier,
# ...) runs on the free path (config normalization, model listing) even when
# fusion is disabled, so binding these to None breaks load_config(). Making fusion
# optional requires splitting its config layer (core) from the FusionRunner engine
# (closed) — that is the architecture-migration FusionRunner extraction (Lot 6) this
# PRD depends on. Until then fusion stays a hard import (a "stub" per the PRD).
# Tracked by the xfail in tests/test_open_core_boundary.py.
from ficelle.use_cases.fusion_config import (
    FusionConfigPorts,
    fusion_bool as fusion_bool_use_case,
    fusion_call_multiplier as fusion_call_multiplier_use_case,
    fusion_choice as fusion_choice_use_case,
    fusion_int as fusion_int_use_case,
    fusion_profile_id as fusion_profile_id_use_case,
    fusion_visible_in_model_list as fusion_visible_in_model_list_use_case,
    normalize_fusion_config as normalize_fusion_config_use_case,
    normalize_fusion_peer_ranking as normalize_fusion_peer_ranking_use_case,
)
try:
    from ficelle_pro.fusion import (
        FusionEligibilityPorts,
        FusionMetadataPorts,
        FusionPanelStage,
        FusionPanelPolicyPorts,
        FusionPreflightRejection,
        FusionRequestPreflight,
        FusionRequestPreflightPorts,
        FusionResponsePorts,
        FusionRunner,
        apply_fusion_run_metadata_to_state as apply_fusion_run_metadata_to_state_use_case,
        aggregate_fusion_rankings as aggregate_fusion_rankings_use_case,
        diverse_fusion_panel as diverse_fusion_panel_use_case,
        extract_fusion_judge_json as extract_fusion_judge_json_use_case,
        fusion_deadline_seconds as fusion_deadline_seconds_use_case,
        fusion_eligible_models as fusion_eligible_models_use_case,
        fusion_error_payload as fusion_error_payload_use_case,
        fusion_internal_body as fusion_internal_body_use_case,
        fusion_openai_response as fusion_openai_response_use_case,
        fusion_panel_attempt_rows,
        fusion_panel_error_rows,
        fusion_panel_heading as fusion_panel_heading_use_case,
        fusion_judge_messages as fusion_judge_messages_use_case,
        fusion_model_detail as fusion_model_detail_use_case,
        fusion_panel_result_detail as fusion_panel_result_detail_use_case,
        fusion_prompt_from_outputs as fusion_prompt_from_outputs_use_case,
        fusion_ranking_messages as fusion_ranking_messages_use_case,
        fusion_ranking_summary as fusion_ranking_summary_use_case,
        fusion_reason_counts as fusion_reason_counts_use_case,
        fusion_response_headers as fusion_response_headers_use_case,
        fusion_run_metadata as fusion_run_metadata_use_case,
        fusion_timeout_config as fusion_timeout_config_use_case,
        normalize_fusion_judge_json as normalize_fusion_judge_json_use_case,
        parse_fusion_ranking as parse_fusion_ranking_use_case,
        record_fusion_judge_outcome as record_fusion_judge_outcome_use_case,
        record_fusion_quality_signal as record_fusion_quality_signal_use_case,
        resolve_peer_ranking_outcome,
        valid_fusion_judge_json as valid_fusion_judge_json_use_case,
    )
except ImportError:  # pragma: no cover - exercised only in core-only installs
    # The fusion ENGINE (FusionRunner + panel/judge/ranking/response) is a closed Pro
    # module. Its config layer lives in fusion_config (imported above, always present),
    # so the free path (load_config, model listing, admin status) never needs these
    # symbols. Fusion ships disabled by default and the engine is only entered on an
    # explicit fusion request, so the core binds every engine symbol to None.
    # See docs/prds/open-core-extraction-prd.md (Lot 1, fusion).
    FusionEligibilityPorts = None
    FusionMetadataPorts = None
    FusionPanelStage = None
    FusionPanelPolicyPorts = None
    FusionPreflightRejection = None
    FusionRequestPreflight = None
    FusionRequestPreflightPorts = None
    FusionResponsePorts = None
    FusionRunner = None
    apply_fusion_run_metadata_to_state_use_case = None
    aggregate_fusion_rankings_use_case = None
    diverse_fusion_panel_use_case = None
    extract_fusion_judge_json_use_case = None
    fusion_deadline_seconds_use_case = None
    fusion_eligible_models_use_case = None
    fusion_error_payload_use_case = None
    fusion_internal_body_use_case = None
    fusion_openai_response_use_case = None
    fusion_panel_attempt_rows = None
    fusion_panel_error_rows = None
    fusion_panel_heading_use_case = None
    fusion_judge_messages_use_case = None
    fusion_model_detail_use_case = None
    fusion_panel_result_detail_use_case = None
    fusion_prompt_from_outputs_use_case = None
    fusion_ranking_messages_use_case = None
    fusion_ranking_summary_use_case = None
    fusion_reason_counts_use_case = None
    fusion_response_headers_use_case = None
    fusion_run_metadata_use_case = None
    fusion_timeout_config_use_case = None
    normalize_fusion_judge_json_use_case = None
    parse_fusion_ranking_use_case = None
    record_fusion_judge_outcome_use_case = None
    record_fusion_quality_signal_use_case = None
    resolve_peer_ranking_outcome = None
    valid_fusion_judge_json_use_case = None
from ficelle.use_cases.hermes_export import HermesExportBuilder
from ficelle.targets import TargetAdapter, TargetExport, TargetExportContext, target_export
from ficelle.targets.generic import GenericClientTargetAdapter
from ficelle.targets.hermes import HermesTargetAdapter, build_hermes_runtime_resolver
from ficelle.targets.openclaw import OpenClawTargetAdapter
from ficelle.targets.online import (
    ONLINE_CONTROL_PLANE_TARGET_VERSION,
    OnlineControlPlaneTargetAdapter,
    safe_online_status_payload,
)
from ficelle.use_cases.model_selection import ModelSelectionPorts, ModelSelectionRunner
from ficelle.use_cases.model_scoring import (
    ModelEvidencePorts,
    ModelScoreExplanationPorts,
    ModelScoringPorts,
    benchmark_adjustment_parts as benchmark_adjustment_parts_use_case,
    latency_score_for_model as latency_score_for_model_use_case,
    model_score_explanation as model_score_explanation_use_case,
    raw_benchmark_row as raw_benchmark_row_use_case,
    raw_verified_capability_row as raw_verified_capability_row_use_case,
    runtime_profile_row as runtime_profile_row_use_case,
    score_evidence_for_model as score_evidence_for_model_use_case,
    score_row_reason as score_row_reason_use_case,
    score_row_status as score_row_status_use_case,
    success_rate_for_model as success_rate_for_model_use_case,
)
from ficelle.use_cases.provider_auth import (
    ProviderAuthPorts,
    auth_row as auth_row_use_case,
    auth_status as auth_status_use_case,
    provider_auth_row as provider_auth_row_use_case,
)
from ficelle.use_cases.provider_catalog import (
    ProviderCatalogPorts,
    fetch_provider_catalog as fetch_provider_catalog_use_case,
    provider_access_result as provider_access_result_use_case,
    provider_invocation_headers as provider_invocation_headers_use_case,
)
from ficelle.use_cases.quota_probe import (
    QuotaProbeModelPorts,
    QuotaProbePorts,
    QuotaProbeRunner,
    QuotaProbeStatePorts,
    find_quota_probe_model as find_quota_probe_model_use_case,
    quota_probe_body as quota_probe_body_use_case,
    record_quota_probe_result as record_quota_probe_result_use_case,
    reserve_quota_probe_in_state as reserve_quota_probe_in_state_use_case,
)
from ficelle.use_cases.router_settings import (
    RouterSettingsPolicy,
    RouterSettingsSavePorts,
    auto_benchmark_enabled as auto_benchmark_enabled_use_case,
    auto_benchmark_interval_seconds as auto_benchmark_interval_seconds_use_case,
    auto_benchmark_models_per_cycle as auto_benchmark_models_per_cycle_use_case,
    normalize_router_settings as normalize_router_settings_use_case,
    normalize_settings_backoff as normalize_settings_backoff_use_case,
    route_on_capability_reference as route_on_capability_reference_use_case,
    router_settings_payload as router_settings_payload_use_case,
    save_router_settings as save_router_settings_use_case,
    settings_bool as settings_bool_use_case,
    settings_int as settings_int_use_case,
    validate_settings_payload as validate_settings_payload_use_case,
    verified_capability_ttl_seconds as verified_capability_ttl_seconds_use_case,
)
from ficelle.json_store import atomic_write_json as store_atomic_write_json
from ficelle.json_store import load_json as store_load_json
from ficelle.providers.base import (
    FREE_ACCESS_MODES,
    FREE_ACCESS_SCOPES,
    ProviderAccess,
    ProviderCatalogPolicy,
    free_access_payload,
    normalized_free_access_status,
)
from ficelle.providers.openai_compatible import NOUS_DEFAULT_BASE_URL, OPENCODE_ZEN_USER_AGENT
try:
    from ficelle_pro.provider_pack import PROVIDER_PACK, PROVIDER_PACK_KEY_URLS, PROVIDER_PACK_LABELS
except ImportError:  # pragma: no cover - core-only install without the closed pack
    # The curated provider pack (incl. the grey-market relays and their key URLs) ships
    # only in the closed Pro package. Absent → DEFAULT_CONFIG["providers"] is
    # CORE_PROVIDERS and PROVIDER_KEY_URLS holds only the reference providers.
    # PROVIDER_PACK and PROVIDER_PACK_KEY_URLS are the only pack names router uses; the
    # pack's individual constants are imported directly from ficelle_pro.provider_pack
    # by the tests that assert on them.
    PROVIDER_PACK: dict[str, Any] = {}
    PROVIDER_PACK_KEY_URLS: dict[str, str] = {}
    PROVIDER_PACK_LABELS: dict[str, str] = {}
from ficelle.providers.registry import provider_catalog_adapter
from ficelle.redaction import redact_sensitive_json, redact_sensitive_text, sanitize_error_detail

try:
    import requests
except Exception as exc:  # pragma: no cover - startup guard
    raise SystemExit(f"Missing dependency: requests ({exc})")

STATE_BACKUP_KEEP = 20
STATE_BACKUP_MIN_INTERVAL_SECONDS = 600
RUNTIME_STATE_HISTORY_KEYS = state_store_module.RUNTIME_STATE_HISTORY_KEYS

# Static admin assets shipped inside the package (not under HERMES_HOME).
PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_PATHS = RuntimePaths.from_env(package_dir=PACKAGE_DIR)
HERMES_HOME = RUNTIME_PATHS.hermes_home
ROUTER_DIR = RUNTIME_PATHS.router_dir
CATALOG_PATH = RUNTIME_PATHS.catalog_path
CATALOG_LOCK_PATH = CATALOG_PATH.with_suffix(".lock")
STATE_PATH = RUNTIME_PATHS.state_path
CONFIG_PATH = RUNTIME_PATHS.config_path
ROUTE_LOG_PATH = RUNTIME_PATHS.route_log_path
ADMIN_AUDIT_LOG_PATH = RUNTIME_PATHS.admin_audit_log_path
STATE_LOCK_PATH = RUNTIME_PATHS.state_lock_path
STATE_BACKUP_DIR = RUNTIME_PATHS.state_backup_dir
# Benchmark-vs-capability-reference contradictions (Phase A, R7): where the prior was wrong.
CAPABILITY_DISCREPANCY_LOG_PATH = RUNTIME_PATHS.capability_discrepancy_log_path
COMPRESSION_STORE_PATH = RUNTIME_PATHS.compression_store_path
REQUEST_LOG_STORE_PATH = RUNTIME_PATHS.request_log_store_path
HERMES_AGENT_DIR = RUNTIME_PATHS.hermes_agent_dir
HERMES_CONFIG_PATH = RUNTIME_PATHS.hermes_config_path
COMPRESSION_PENDING_MARKER = CHAT_COMPLETION_COMPRESSION_PENDING_MARKER

# Ficelle's runtime state may still be discovered in the legacy Hermes layout until
# setup migrates it. Credential WRITES are different: all new writes go to the
# Ficelle-owned home, while resolution below keeps read-only legacy fallbacks.
FICELLE_HOME = RUNTIME_PATHS.ficelle_home
CREDENTIAL_ENV_FILE = RUNTIME_PATHS.credential_env_file
LEGACY_CREDENTIAL_ENV_FILE = RUNTIME_PATHS.legacy_credential_env_file
FICELLE_SECRETS_KEYCHAIN = RUNTIME_PATHS.ficelle_secrets_keychain
LEGACY_FICELLE_SECRETS_KEYCHAIN = RUNTIME_PATHS.legacy_ficelle_secrets_keychain
LEGACY_HERMES_SECRETS_KEYCHAIN = RUNTIME_PATHS.legacy_hermes_secrets_keychain
_STATE_STORE = StateStore(
    STATE_PATH,
    STATE_LOCK_PATH,
    STATE_BACKUP_DIR,
    backup_keep=STATE_BACKUP_KEEP,
    backup_min_interval_seconds=STATE_BACKUP_MIN_INTERVAL_SECONDS,
    history_keys=RUNTIME_STATE_HISTORY_KEYS,
    fallback_state_path=RUNTIME_PATHS.runtime_read_dir / STATE_PATH.name,
    fallback_backup_dir=RUNTIME_PATHS.runtime_read_dir / STATE_BACKUP_DIR.name,
)
_REQUEST_LOG_LOCK = threading.Lock()
_CATALOG_PUBLISH_THREAD_LOCK = threading.RLock()
# Held from background transition-thread creation until that worker exits; it only
# deduplicates license-transition refreshes and does not serialize general catalog writes.
_LICENSE_TRANSITION_REFRESH_IN_FLIGHT_LOCK = threading.Lock()

FONTS_DIR = PACKAGE_DIR / "assets" / "fonts"
LOGOS_DIR = PACKAGE_DIR / "assets" / "logos"
ADMIN_DIR = RUNTIME_PATHS.admin_assets_dir or PACKAGE_DIR / "assets" / "admin"
# Curated cross-provider capability reference seed shipped inside the package.
CAPABILITY_REFERENCE_PATH = PACKAGE_DIR / "data" / "model_capabilities.json"
# OpenRouter capability oracle: its PUBLIC /models catalog read for capability metadata only
# (never pricing/endpoint/routing), cached under the runtime home and refreshed on a TTL. The
# route path never hits this endpoint — the fetch runs only at serve startup and `ficelle refresh`.
CAPABILITY_ORACLE_CACHE_PATH = RUNTIME_PATHS.capability_oracle_cache_path
OPENROUTER_ORACLE_URL = "https://openrouter.ai/api/v1/models"
CAPABILITY_ORACLE_TTL_SECONDS = 7 * 24 * 3600
_FONT_CACHE: dict[str, bytes] = {}
_LOGO_CACHE: dict[str, bytes] = {}
_ADMIN_ASSET_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".map": "application/json",
    ".json": "application/json",
}


def load_font_bytes(name: str) -> bytes | None:
    """Return the bytes of a packaged .woff2 admin font, or None if unknown.

    Path traversal is neutralised via ``Path(name).name`` and an explicit
    extension allowlist, so only files inside FONTS_DIR can ever be served.
    """
    safe = Path(name).name
    if not safe.endswith(".woff2"):
        return None
    cached = _FONT_CACHE.get(safe)
    if cached is not None:
        return cached
    try:
        data = (FONTS_DIR / safe).read_bytes()
    except OSError:
        return None
    _FONT_CACHE[safe] = data
    return data


def _pro_asset_dir(*parts: str) -> Path | None:
    """Resolve ``<ficelle_pro root>/<parts>``, or None on a core-only install.

    Single owner of the ``load_pro()`` + ``__file__`` resolution shared by the closed
    pack's logo and admin-view asset lookups.
    """
    from ficelle.pro import load_pro

    pro = load_pro()
    if pro is not None and pro.__file__:
        return Path(pro.__file__).resolve().parent.joinpath(*parts)
    return None


def _logo_dirs() -> list[Path]:
    """Core logos, then the closed pack's logos when ficelle_pro is installed."""
    dirs = [LOGOS_DIR]
    pro_logos = _pro_asset_dir("assets", "logos")
    if pro_logos is not None:
        dirs.append(pro_logos)
    return dirs


def _pro_admin_dir() -> Path | None:
    """The closed pack's admin-view assets dir, or None on a core-only install.

    The Fusion/Compression dashboard views ship in ficelle_pro (data-as-moat, same as
    the engines) and are served under ``/admin/static/pro/``. ``None`` here is the
    source of truth for "Pro admin surface absent" across asset serving and the
    dashboard-shell script injection (the admin endpoint gates key on the engine
    symbols instead, like the routing gates).
    """
    return _pro_asset_dir("assets", "admin")


def _load_pro_install_module() -> Any:
    from ficelle import pro_install

    return pro_install


def load_logo_bytes(name: str) -> bytes | None:
    """Return the bytes of a packaged provider .svg logo, or None if unknown.

    Core ships the open providers' marks; the closed pack's marks (incl. the grey-market
    relays) ship in ficelle_pro and are looked up there when it is installed.
    """
    safe = Path(name).name
    if not safe.endswith(".svg"):
        return None
    cached = _LOGO_CACHE.get(safe)
    if cached is not None:
        return cached
    for base in _logo_dirs():
        try:
            data = (base / safe).read_bytes()
        except OSError:
            continue
        _LOGO_CACHE[safe] = data
        return data
    return None


def load_admin_asset(relpath: str) -> tuple[bytes, str] | None:
    """Return ``(bytes, content_type)`` for a packaged dashboard asset, or None.

    Sub-directories are allowed (for future JS modules) while staying
    traversal-safe: the resolved path must live under ADMIN_DIR and carry an
    allow-listed suffix. Read fresh each time so UI edits show without a cache bust.
    """
    rel = str(relpath or "").lstrip("/")
    if not rel or rel.endswith("/"):
        return None
    base, rel = _admin_asset_base(rel)
    if base is None:
        return None
    try:
        target = (base / rel).resolve()
        target.relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    ctype = _ADMIN_ASSET_TYPES.get(target.suffix)
    if ctype is None:
        return None
    try:
        return target.read_bytes(), ctype
    except OSError:
        return None


def _admin_asset_base(rel: str) -> tuple[Path | None, str]:
    """Resolve which dir serves ``rel`` and the path within it.

    ``pro/<x>`` is served from the closed pack's admin dir (``None`` when Pro is
    absent, so the asset 404s); everything else from the core ADMIN_DIR.
    """
    if rel.startswith("pro/"):
        return _pro_admin_dir(), rel[len("pro/"):]
    return ADMIN_DIR, rel


def admin_asset_version(relpath: str) -> str:
    rel = str(relpath or "").lstrip("/")
    base, rel = _admin_asset_base(rel)
    if base is None or not rel:
        return "0"
    try:
        target = (base / rel).resolve()
        target.relative_to(base.resolve())
        return str(target.stat().st_mtime_ns)
    except (ValueError, OSError):
        return "0"

VIRTUAL_MODEL_SLUGS = {
    "auto-orchestrator",
    "auto-tools",
    "auto-json",
    "auto-compression",
    "auto-long",
    "auto-fast",
    "auto-reasoning",
    "auto-multimodal",
    "auto-vision",
    "auto-video",
    "auto-audio",
}

VIRTUAL_MODELS = {f"ficelle/{slug}" for slug in VIRTUAL_MODEL_SLUGS}
VIRTUAL_PROFILE_ID_PATTERN = re.compile(r"^ficelle/[A-Za-z0-9][A-Za-z0-9._/-]{0,120}$")
TARGET_EXPORT_VIRTUAL_MODELS = (
    "ficelle/auto-orchestrator",
    "ficelle/auto-tools",
    "ficelle/auto-json",
    "ficelle/auto-compression",
    "ficelle/auto-long",
    "ficelle/auto-fast",
    "ficelle/auto-reasoning",
    "ficelle/auto-multimodal",
    "ficelle/auto-vision",
    "ficelle/auto-video",
    "ficelle/auto-audio",
)


def is_virtual_profile_id(model_id: str) -> bool:
    return model_id.startswith("ficelle/")


def is_safe_virtual_profile_id(model_id: str) -> bool:
    candidate = str(model_id or "")
    return bool(VIRTUAL_PROFILE_ID_PATTERN.fullmatch(candidate)) and redact_sensitive_text(candidate) == candidate


def canonical_virtual_model_id(model_id: str) -> str:
    return str(model_id or "").strip()

DEFAULT_PROFILE_REQUIREMENTS = {
    "free": True,
    "tools": True,
    "min_context": 128000,
    "structured": None,
    "input_modalities": [],
    "input_modalities_any": [],
    "output_modalities": [],
    "output_modalities_any": [],
    "supported_parameters": [],
    "supported_parameters_any": [],
}

VIRTUAL_MODEL_REQUIREMENTS = {
    "ficelle/auto-orchestrator": {"structured": True},
    "ficelle/auto-fast": {"structured": True},
    "ficelle/auto-json": {"structured": True},
    "ficelle/auto-compression": {"structured": True},
    "ficelle/auto-reasoning": {"supported_parameters_any": ["reasoning", "include_reasoning"]},
    "ficelle/auto-multimodal": {"input_modalities_any": ["image", "video", "audio"]},
    "ficelle/auto-vision": {"input_modalities": ["image"]},
    "ficelle/auto-video": {"input_modalities": ["video"]},
    "ficelle/auto-audio": {"input_modalities": ["audio"]},
}

DEFAULT_QUOTA_PROBE_BACKOFF_SECONDS = [3600, 21600, 86400]
# Key-acquisition URLs for the open-core reference providers. The pack providers' URLs
# (incl. the grey-market relays) live in ficelle_pro.provider_pack and merge in when the
# closed pack is installed — so the open core never names the grey-market endpoints.
CORE_PROVIDER_KEY_URLS = {
    "openrouter": "https://openrouter.ai/keys",
    "nous": "https://portal.nousresearch.com/",
}
PROVIDER_KEY_URLS = {**CORE_PROVIDER_KEY_URLS, **PROVIDER_PACK_KEY_URLS}

# Display labels for the admin UI. The core ships the reference providers' labels; the pack
# providers' labels (incl. the grey-market relays) merge in from ficelle_pro when installed,
# so the admin overlays them and the open core never names the grey-market relays.
CORE_PROVIDER_LABELS = {"openrouter": "OpenRouter", "nous": "Nous Research"}
PROVIDER_LABELS = {**CORE_PROVIDER_LABELS, **PROVIDER_PACK_LABELS}

DEFAULT_VIRTUAL_PROFILES = {
    model_id: {
        "mode": "auto",
        "models": [],
        "excluded_models": [],
        "auto_tail": True,
        "requirements": {**DEFAULT_PROFILE_REQUIREMENTS, **VIRTUAL_MODEL_REQUIREMENTS.get(canonical_virtual_model_id(model_id), {})},
    }
    for model_id in sorted(VIRTUAL_MODELS)
}
DEFAULT_VIRTUAL_PROFILES["ficelle/auto-fast"].update(
    {
        "mode": "manual_order",
        "models": [
            "ficelle/openrouter/nex-agi/nex-n2-pro:free",
            "ficelle/openrouter/google/gemma-4-31b-it:free",
            "ficelle/nous/stepfun/step-3.7-flash:free",
        ],
    }
)
DEFAULT_VIRTUAL_PROFILES["ficelle/auto-compression"].update(
    {
        "mode": "manual_order",
        "models": [
            "ficelle/openrouter/google/gemma-4-31b-it:free",
            "ficelle/openrouter/nex-agi/nex-n2-pro:free",
            "ficelle/nous/stepfun/step-3.7-flash:free",
        ],
    }
)
DEFAULT_VIRTUAL_PROFILES["ficelle/auto-orchestrator"].update(
    {
        "mode": "manual_order",
        "models": [
            "ficelle/openrouter/nex-agi/nex-n2-pro:free",
            "ficelle/openrouter/nvidia/nemotron-3-super-120b-a12b:free",
            "ficelle/openrouter/google/gemma-4-31b-it:free",
            "ficelle/nous/stepfun/step-3.7-flash:free",
        ],
    }
)

FUSION_MODEL_ID = "ficelle/auto-fusion"
FUSION_BUDGET_POLICIES = {"strict_zero"}
FUSION_DIVERSITY_POLICIES = {"provider_and_family", "provider", "family", "score"}
FUSION_RAW_OUTPUT_POLICIES = {"none"}
FUSION_JUDGE_FORMATS = {"json", "hybrid"}
FUSION_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
DEFAULT_FUSION_CONFIG: dict[str, Any] = {
    "enabled": False,
    "visible_in_models": False,
    "panel_profile": "ficelle/auto-orchestrator",
    "judge_profile": "ficelle/auto-json",
    "synthesizer_profile": "ficelle/auto-orchestrator",
    "draft_profile": None,
    "panel_size": 4,
    "max_panel_size": 4,
    "min_successful_panel_outputs": 2,
    "budget_policy": "strict_zero",
    "max_total_calls": 9,
    "per_call_timeout_seconds": 70,
    "fusion_timeout_seconds": 200,
    "diversity_policy": "score",
    "allow_single_panel_degrade": False,
    "blind_draft": False,
    "raw_output_policy": "none",
    "anonymize_panel": True,
    "judge_format": "json",
    "peer_ranking": {
        "enabled": False,
        "min_rankers": 2,
        "replaces_judge": False,
        "feed_synthesizer": True,
        "emit_routing_signal": False,
    },
}

# Per-stage sequential fallback depth: the judge and synthesizer each try up to
# this many eligible models, moving to the next only when one fails at the
# transport level (timeout/HTTP/exception). Kept small to bound the call budget;
# fusion_call_multiplier (and its JS mirror) account for these in max_total_calls.
FUSION_JUDGE_MAX_ATTEMPTS = 2
FUSION_SYNTHESIZER_MAX_ATTEMPTS = 3
# Rolling window of recent judge outcomes, used to surface the judge's
# strict-JSON failure rate (Improvement C measurement) without storing any
# prompt, panel, or judge text. Only outcomes where the judge actually
# responded are recorded; transport failures and no-candidate runs are excluded
# so the rate reflects output-format fragility, not provider availability.
FUSION_JUDGE_HISTORY_LIMIT = 50
FUSION_JUDGE_RECORDABLE_OUTCOMES = {"ok", "invalid_json", "invalid_schema"}
# EWMA weight on the newest run when accumulating the per-model peer-ranking
# quality signal (older results decay over time). The signal is observability-only
# and INERT: it is never read by routing/scoring; wiring it into selection is
# gated on a separate evidence check (PRD Lot 5).
FUSION_QUALITY_SIGNAL_ALPHA = 0.3

BENCHMARK_VISION_FIXTURES_PATH = PACKAGE_DIR / "data" / "benchmark" / "vision.json"
BENCHMARK_VIDEO_DEFAULT_ANSWER = "73"
BENCHMARK_AUDIO_DEFAULT_ANSWER = "orange"

COMPRESSION_BENCHMARK_DEFAULT_CHARS = 18000
COMPRESSION_BENCHMARK_MIN_CHARS = 2000
COMPRESSION_BENCHMARK_MAX_CHARS = 200000
COMPRESSION_BENCHMARK_MAX_WORDS = 60

LONG_CONTEXT_BENCHMARK_DEFAULT_TOKENS = 64000
LONG_CONTEXT_BENCHMARK_MIN_TOKENS = 1000
LONG_CONTEXT_BENCHMARK_MAX_TOKENS = 256000
LONG_CONTEXT_BENCHMARK_CHARS_PER_TOKEN = 4

DEFAULT_BENCHMARK_PROFILES = list(TARGET_EXPORT_VIRTUAL_MODELS)
DEFAULT_CORE_CANARY_PROFILES = [
    "ficelle/auto-orchestrator",
    "ficelle/auto-tools",
    "ficelle/auto-json",
    "ficelle/auto-compression",
    "ficelle/auto-fast",
]

BENCHMARK_TEST_TYPES_BY_PROFILE = {
    "ficelle/auto-orchestrator": "orchestration",
    "ficelle/auto-tools": "tool_call",
    "ficelle/auto-json": "json",
    "ficelle/auto-compression": "compression",
    "ficelle/auto-fast": "fast",
    "ficelle/auto-long": "long_context",
    "ficelle/auto-reasoning": "reasoning",
    "ficelle/auto-vision": "vision",
    "ficelle/auto-video": "video",
    "ficelle/auto-audio": "audio",
}

# Profiles whose failed or unverified discriminating capability route-EXCLUDES a model (the
# proof-gate), as opposed to merely lowering its score. Kept as an explicit set, decoupled from
# BENCHMARK_TEST_TYPES_BY_PROFILE — which only declares each profile's probe shape — so a profile
# can gain a representative probe without automatically becoming a gate, and vice versa (PRD
# representative-capability-probes D2). auto-multimodal gates on an OR over the modality profiles
# and is handled directly in the route path, so it is intentionally not listed here.
GATED_PROFILES = frozenset({
    "ficelle/auto-orchestrator",
    "ficelle/auto-tools",
    "ficelle/auto-json",
    "ficelle/auto-compression",
    "ficelle/auto-long",
    "ficelle/auto-reasoning",
    "ficelle/auto-vision",
    "ficelle/auto-video",
    "ficelle/auto-audio",
})


def is_specialized_profile(profile_id: str) -> bool:
    """True for profiles whose failed/unverified capability route-EXCLUDES a model (the proof-gate).

    Membership is the explicit ``GATED_PROFILES`` set, decoupled from the benchmark test-type map
    (PRD representative-capability-probes D2): a listed profile excludes a model that failed or has
    not verified its discriminating capability, instead of only weighting capability into score.
    Generic profiles keep trusting the catalog claim. ``auto-multimodal`` gates on an OR over the
    modality profiles and is handled directly in the route path, so it is not in ``GATED_PROFILES``.
    """
    return canonical_virtual_model_id(profile_id) in GATED_PROFILES


# The single discriminating capability each specialized profile gates on, used for
# provenance: a capability claimed only via provider catalog_model_defaults must be
# benchmark-verified before routing (a real catalog declaration is trusted). Parallel
# to BENCHMARK_TEST_TYPES_BY_PROFILE but keyed to the capability token that
# capabilities_from_defaults() emits, not the benchmark test type. auto-multimodal is
# absent: it gates on an OR over the modality profiles below, not a single capability.
GATE_CAPABILITY_BY_PROFILE = {
    "ficelle/auto-tools": "tools",
    "ficelle/auto-json": "structured",
    "ficelle/auto-reasoning": "reasoning",
    "ficelle/auto-vision": "image",
    "ficelle/auto-video": "video",
    "ficelle/auto-audio": "audio",
}

# auto-multimodal's competence is the OR of these modality profiles: a pass on any one
# satisfies the gate; exclusion needs every claimed modality to have failed.
MULTIMODAL_MODALITY_PROFILES = {
    "image": "ficelle/auto-vision",
    "video": "ficelle/auto-video",
    "audio": "ficelle/auto-audio",
}


def expected_benchmark_test_type(profile_id: str) -> str:
    return expected_probe_test_type(
        profile_id,
        BENCHMARK_TEST_TYPES_BY_PROFILE,
        canonicalize=canonical_virtual_model_id,
    )


# Re-prove a verdict periodically so capability state self-heals when a provider upgrades a model
# or lifts a tier limit: a verified/failed verdict older than the active TTL is treated as stale
# and re-benchmarked rather than trusted forever. Operator-tunable via the admin Settings page
# (apply_verified_capability_ttl), with a 24h floor so benchmarks cannot run more than once a day.
DEFAULT_VERIFIED_CAPABILITY_TTL_SECONDS = 7 * 24 * 3600
MIN_VERIFIED_CAPABILITY_TTL_SECONDS = 24 * 3600
MAX_VERIFIED_CAPABILITY_TTL_SECONDS = 30 * 24 * 3600
# Active TTL, synced from config at the routing/admin/settings entry points because the staleness
# chokepoint runs deep in the selection tree where config is not threaded.
_active_verified_capability_ttl_seconds = DEFAULT_VERIFIED_CAPABILITY_TTL_SECONDS

# Phase B (cross-provider capability reference PRD, R6): an operator opt-in, default OFF, that lets
# a HIGH-confidence reference-confirmed capability open the route gate like a real catalog claim —
# bounded by: a `failed` benchmark verdict always overrides, the verified-capability TTL keeps
# re-checking, and only the confidence tier below is trusted (the hand-curated seed is "high"; the
# broad OpenRouter oracle is "medium" and stays Phase A targeting only until benchmark-agreement
# data earns more). Synced from config via apply_route_on_capability_reference at the same entry
# points as the TTL (the deep gate reads the module cache, not threaded config).
DEFAULT_ROUTE_ON_CAPABILITY_REFERENCE = False
CAPABILITY_REFERENCE_ROUTE_CONFIDENCES = frozenset({"high"})
_route_on_capability_reference = DEFAULT_ROUTE_ON_CAPABILITY_REFERENCE

# Automatic capability DISCOVERY: a bounded background loop (in serve()) probes new/unproven models
# across ALL specialized capabilities so each model is mapped to every profile it actually supports
# — instead of waiting for a manual canary or trusting the optimistic catalog_model_defaults claim.
# Each cycle takes a few not-yet-fully-discovered models and probes each on every probeable capability
# (no stop-at-first-pass), reusing the per-candidate cooldowns/quota backoff. It converges then idles,
# and a verdict re-tests itself after the verified-capability TTL. A capability probe never benches a
# model globally (probe_model_capability) — only a provider-wide block (billing/quota/auth) does.
DEFAULT_AUTO_BENCHMARK_ENABLED = True
DEFAULT_AUTO_BENCHMARK_INTERVAL_SECONDS = 3 * 3600
MIN_AUTO_BENCHMARK_INTERVAL_SECONDS = 3600
MAX_AUTO_BENCHMARK_INTERVAL_SECONDS = 7 * 24 * 3600
DEFAULT_AUTO_BENCHMARK_MODELS_PER_CYCLE = 5
MIN_AUTO_BENCHMARK_MODELS_PER_CYCLE = 1
MAX_AUTO_BENCHMARK_MODELS_PER_CYCLE = 50
AUTO_BENCHMARK_STARTUP_DELAY_SECONDS = 60


def verified_capability_ttl_seconds() -> int:
    return _active_verified_capability_ttl_seconds


def benchmark_evidence_ports() -> BenchmarkEvidencePorts:
    return BenchmarkEvidencePorts(
        expected_test_type=expected_benchmark_test_type,
        active_ttl_seconds=verified_capability_ttl_seconds,
        evidence_timestamp_value=runtime_evidence_timestamp_value,
        parse_timestamp=parse_iso_timestamp,
        now_seconds=time.time,
    )


def benchmark_result_is_aged(
    row: dict[str, Any],
    ttl_seconds: float | None = None,
    now_ts: float | None = None,
) -> bool:
    return benchmark_result_is_aged_use_case(
        row,
        ports=benchmark_evidence_ports(),
        ttl_seconds=ttl_seconds,
        now_ts=now_ts,
    )


def benchmark_result_test_type_matches(profile_id: str, row: Any) -> bool:
    return benchmark_result_test_type_matches_use_case(profile_id, row, ports=benchmark_evidence_ports())


def benchmark_result_matches_current_test(
    profile_id: str,
    row: Any,
    *,
    now_ts: float | None = None,
    ttl_seconds: float | None = None,
) -> bool:
    return benchmark_result_matches_current_test_use_case(
        profile_id,
        row,
        ports=benchmark_evidence_ports(),
        now_ts=now_ts,
        ttl_seconds=ttl_seconds,
    )


def stale_score_decay_factor(profile_id: str, row: Any, *, now_ts: float | None = None, ttl_seconds: float | None = None) -> float:
    return stale_score_decay_factor_use_case(
        profile_id,
        row,
        ports=benchmark_evidence_ports(),
        now_ts=now_ts,
        ttl_seconds=ttl_seconds,
    )


def mark_stale_benchmark_row(profile_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return mark_stale_benchmark_row_use_case(profile_id, row, ports=benchmark_evidence_ports())


def redacted_benchmark_row(profile_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return redacted_benchmark_row_use_case(
        profile_id,
        row,
        ports=benchmark_evidence_ports(),
        redact_sensitive_json=redact_sensitive_json,
    )

DEFAULT_CANARY_PROFILES = list(DEFAULT_CORE_CANARY_PROFILES)

# Reference providers that ship in the core router. The remaining curated
# providers live in ficelle_pro.provider_pack (PROVIDER_PACK) and are composed in
# after these two so DEFAULT_CONFIG["providers"] stays openrouter, nous, then
# the pack in its original order.
CORE_PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "enabled": True,
        "catalog_url": "https://openrouter.ai/api/v1/models",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "source_type": "zero_price_catalog",
    },
    "nous": {
        "enabled": True,
        "catalog_url": "https://inference-api.nousresearch.com/v1/models",
        "base_url": NOUS_DEFAULT_BASE_URL,
        "source_type": "zero_price_catalog",
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8646,
    "min_context_length": 128000,
    "catalog_ttl_seconds": 3600,
    "request_timeout_seconds": 120,
    "request_timeout_seconds_by_profile": {
        "ficelle/auto-fast": 30,
        "ficelle/auto-json": 45,
        "ficelle/auto-compression": 45,
        "ficelle/auto-tools": 60,
        "ficelle/auto-orchestrator": 75,
        "ficelle/auto-long": 90,
    },
    "max_attempts_per_request": 4,
    "catalog_timeout_seconds": 30,
    "cooldown_seconds": {
        "rate_limited": 900,
        "quota_exhausted": 3600,
        "auth_or_credit": 3600,
        "billing_or_paid": 86400,
        "benchmark_failed": 3600,
        "server_error": 180,
        "timeout": 300,
        "unavailable": 600,
    },
    "quota_probe_backoff_seconds": list(DEFAULT_QUOTA_PROBE_BACKOFF_SECONDS),
    "quota_probe_timeout_seconds": 10,
    "verified_capability_ttl_seconds": 604800,
    "route_on_capability_reference": False,
    "auto_benchmark_enabled": True,
    "auto_benchmark_interval_seconds": 10800,
    "auto_benchmark_models_per_cycle": 5,
    "benchmark_long_context_tokens": LONG_CONTEXT_BENCHMARK_DEFAULT_TOKENS,
    "benchmark_compression_chars": COMPRESSION_BENCHMARK_DEFAULT_CHARS,
    "compression": copy.deepcopy(DEFAULT_COMPRESSION_CONFIG),
    "allow_paid_fallback": False,
    "canary": {
        "enabled": True,
        "profiles": DEFAULT_CANARY_PROFILES,
    },
    "experimental": {
        "auto_fusion": False,
    },
    "fusion": DEFAULT_FUSION_CONFIG,
    "benchmark_media": {
        "audio_path": "",
        "video_path": "",
        "max_bytes": 5_000_000,
    },
    "virtual_profiles": DEFAULT_VIRTUAL_PROFILES,
    "providers": {**CORE_PROVIDERS, **PROVIDER_PACK},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    return store_load_json(path, default)


def runtime_read_path(path: Path) -> Path:
    """Resolve a canonical runtime path through the read-only compatibility root."""
    return RUNTIME_PATHS.read_path(path)


def load_runtime_json(path: Path, default: Any) -> Any:
    """Read a canonical runtime JSON file with canonical-first legacy fallback."""
    return load_json(runtime_read_path(path), default)


def load_runtime_state() -> dict[str, Any]:
    """Return the effective runtime state without ever mutating its read source."""
    return _STATE_STORE.snapshot()


def atomic_write_json(path: Path, data: Any) -> None:
    store_atomic_write_json(path, data)


@contextmanager
def runtime_state_lock() -> Iterator[None]:
    """Serialize runtime state writes across admin/background request threads and processes."""
    with _STATE_STORE.lock():
        yield


def runtime_row_timestamp(row: Any) -> float | None:
    return state_store_module.runtime_row_timestamp(row)


def runtime_state_history_counts(state: Any, key: str) -> dict[str, Any]:
    return state_store_module.runtime_state_history_counts(state, key)


def state_file_summary(path: Path) -> dict[str, Any]:
    return state_store_module.state_file_summary(path)


def state_backup_files() -> list[Path]:
    return _STATE_STORE.backup_files()


def rotate_state_backups(keep: int = STATE_BACKUP_KEEP) -> None:
    _STATE_STORE.rotate_backups(keep)


def backup_runtime_state() -> Path | None:
    return _STATE_STORE.backup()


def runtime_state_diagnostics(state: dict[str, Any] | None = None) -> dict[str, Any]:
    return _STATE_STORE.diagnostics(state)


def merge_runtime_history_rows(current: Any, incoming: Any) -> Any:
    """Merge append-only scoring evidence so stale snapshots cannot drop newer rows."""
    return state_store_module.merge_runtime_history_rows(current, incoming)


def merge_runtime_state_for_write(current: Any, incoming: dict[str, Any]) -> dict[str, Any]:
    return _STATE_STORE.merge_for_write(current, incoming)


def write_runtime_state(state: dict[str, Any]) -> None:
    """Persist runtime state while preserving accumulated benchmark/capability evidence."""
    _STATE_STORE.write(state)


def write_route_log(row: dict[str, Any], path: Path = ROUTE_LOG_PATH) -> None:
    """Append one redacted routing event. Never include prompts, messages, or credentials."""
    safe_row = redact_sensitive_json(dict(row))
    safe_row.setdefault("logged_at", now_iso())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_row, ensure_ascii=False, sort_keys=True) + "\n")


def write_admin_audit(
    action: str,
    *,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    path: Path = ADMIN_AUDIT_LOG_PATH,
) -> dict[str, Any]:
    row = {
        "id": uuid.uuid4().hex,
        "logged_at": now_iso(),
        "action": action,
        "metadata": redact_sensitive_json(metadata or {}),
    }
    if before is not None:
        row["before"] = redact_sensitive_json(before)
    if after is not None:
        row["after"] = redact_sensitive_json(after)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


def start_admin_job(job_name: str, metadata: dict[str, Any] | None = None) -> str:
    job_id = uuid.uuid4().hex
    key = safe_state_key(job_name)

    def mutate(state: dict[str, Any]) -> None:
        row = build_admin_job_row(job_id, metadata, now_iso=now_iso, now_epoch=time.time)
        record_admin_job_start(state, key, row)

    _STATE_STORE.update(mutate, reason=f"start_admin_job:{key}")
    return job_id


def finish_admin_job(job_name: str, job_id: str) -> None:
    key = safe_state_key(job_name)
    current = load_runtime_state()
    if not admin_job_matches(current, key, job_id):
        return

    def mutate(state: dict[str, Any]) -> None:
        record_admin_job_finish(state, key, job_id)

    _STATE_STORE.update(mutate, reason=f"finish_admin_job:{key}")


def active_admin_jobs(source: Any, *, now: float | None = None) -> dict[str, Any]:
    return active_admin_jobs_use_case(
        source,
        now_ts=time.time() if now is None else now,
        max_age_seconds=6 * 60 * 60,
        safe_float=safe_float,
        safe_state_key=safe_state_key,
    )


def read_admin_audit_raw(limit: int = 50, path: Path | None = None) -> list[dict[str, Any]]:
    path = runtime_read_path(ADMIN_AUDIT_LOG_PATH) if path is None else path
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return list(reversed(rows[-max(1, limit):]))


def read_admin_audit(limit: int = 50, path: Path | None = None) -> list[dict[str, Any]]:
    return [redact_sensitive_json(row) for row in read_admin_audit_raw(limit=limit, path=path)]


def find_admin_audit_entry(audit_id: str, limit: int = 500) -> dict[str, Any] | None:
    needle = str(audit_id or "").strip()
    if not needle:
        return None
    for row in read_admin_audit_raw(limit=limit):
        if row.get("id") == needle:
            return row
    return None


def ficelle_response_headers(
    request_id: str,
    requested_model: str,
    selected_model: dict[str, Any] | None = None,
    attempt_count: int = 0,
    compression: dict[str, Any] | None = None,
) -> dict[str, str]:
    headers = {
        "X-Ficelle-Request-Id": request_id,
        "X-Ficelle-Profile": safe_detail(requested_model) or "[redacted]",
        "X-Ficelle-Attempts": str(max(0, attempt_count)),
        "X-Ficelle-Fallback": "true" if attempt_count > 1 else "false",
    }
    if selected_model:
        headers["X-Ficelle-Model"] = safe_detail(selected_model.get("id")) or "[redacted]"
        headers["X-Ficelle-Upstream"] = safe_detail(selected_model.get("upstream_id")) or "[redacted]"
        headers["X-Ficelle-Source"] = safe_detail(selected_model.get("source")) or "[redacted]"
    if compression:
        status = safe_detail(compression.get("status")) or "none"
        headers["X-Ficelle-Compression"] = status if status in {"dry_run", "compressed"} else "none"
    return headers


def safe_detail(value: Any, limit: int = 250) -> str | None:
    return sanitize_error_detail(value, limit)


def safe_state_key(value: Any) -> str:
    return safe_detail(value) or "[redacted]"


def safe_error_body(exc: Exception, *, request_id: str | None = None) -> dict[str, Any]:
    return safe_error_body_result(exc, request_id=request_id)


def bad_request_body(exc: Exception, *, request_id: str | None = None) -> dict[str, Any]:
    return bad_request_body_result(exc, request_id=request_id)


def upstream_failure_actions(reason_counts: dict[str, int]) -> list[str]:
    return upstream_failure_actions_result(reason_counts)


def build_upstream_failure_error(
    requested_model: str,
    request_id: str,
    candidate_count: int,
    attempts: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return build_upstream_failure_error_result(requested_model, request_id, candidate_count, attempts, errors)


def apply_chat_attempt_cooldown(
    cooldown: AttemptCooldownRequest,
    config: dict[str, Any],
    *,
    request_id: str,
    profile_id: str,
) -> None:
    set_cooldown(
        cooldown.model,
        cooldown.reason,
        config,
        detail=cooldown.detail,
        status=cooldown.status,
        request_id=request_id,
        profile_id=profile_id,
    )


def apply_chat_route_telemetry(telemetry: ChatCompletionRouteTelemetry) -> None:
    last_route = telemetry.last_route
    record_last_route(
        last_route.safe_requested_model,
        last_route.status,
        last_route.reason,
        last_route.request_id,
        last_route.candidate_count,
        last_route.attempt_count,
        last_route.duration_seconds,
        last_route.selected_model,
        last_route.attempts,
        competence=last_route.competence,
        compression=last_route.compression,
    )
    write_route_log(telemetry.route_log)


def fusion_config_ports() -> FusionConfigPorts:
    return FusionConfigPorts(
        default_config=DEFAULT_FUSION_CONFIG,
        budget_policies=FUSION_BUDGET_POLICIES,
        diversity_policies=FUSION_DIVERSITY_POLICIES,
        raw_output_policies=FUSION_RAW_OUTPUT_POLICIES,
        judge_formats=FUSION_JUDGE_FORMATS,
        fusion_model_id=FUSION_MODEL_ID,
        judge_max_attempts=FUSION_JUDGE_MAX_ATTEMPTS,
        synthesizer_max_attempts=FUSION_SYNTHESIZER_MAX_ATTEMPTS,
        is_virtual_profile_id=is_virtual_profile_id,
        is_safe_virtual_profile_id=is_safe_virtual_profile_id,
        safe_detail=safe_detail,
        safe_int=safe_int,
    )


def fusion_bool(raw: dict[str, Any], key: str, default: bool, *, strict: bool) -> bool:
    return fusion_bool_use_case(raw, key, default, strict=strict)


def fusion_int(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    strict: bool,
) -> int:
    return fusion_int_use_case(raw, key, default, minimum=minimum, maximum=maximum, strict=strict)


def fusion_choice(raw: dict[str, Any], key: str, default: str, allowed: set[str], *, strict: bool) -> str:
    return fusion_choice_use_case(raw, key, default, allowed, strict=strict)


def fusion_profile_id(
    raw: dict[str, Any],
    key: str,
    default: str | None,
    *,
    strict: bool,
    allowed_profile_ids: set[str] | None = None,
) -> str | None:
    return fusion_profile_id_use_case(
        raw,
        key,
        default,
        ports=fusion_config_ports(),
        strict=strict,
        allowed_profile_ids=allowed_profile_ids,
    )


def fusion_call_multiplier(config: dict[str, Any]) -> int:
    return fusion_call_multiplier_use_case(config, fusion_config_ports())


def normalize_fusion_peer_ranking(raw: Any, *, strict: bool) -> dict[str, Any]:
    return normalize_fusion_peer_ranking_use_case(raw, ports=fusion_config_ports(), strict=strict)


def normalize_fusion_config(
    raw: Any,
    *,
    strict: bool = False,
    allowed_profile_ids: set[str] | None = None,
) -> dict[str, Any]:
    return normalize_fusion_config_use_case(
        raw,
        ports=fusion_config_ports(),
        strict=strict,
        allowed_profile_ids=allowed_profile_ids,
    )


def normalize_config_fusion(config: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    fusion = normalize_fusion_config(config.get("fusion"), strict=strict, allowed_profile_ids=set(configured_virtual_model_ids(config)))
    experimental = config.get("experimental") if isinstance(config.get("experimental"), dict) else {}
    config["experimental"] = {**experimental, "auto_fusion": fusion["enabled"]}
    config["fusion"] = fusion
    return config


def runtime_config_store() -> ConfigStore:
    return ConfigStore(
        CONFIG_PATH,
        DEFAULT_CONFIG,
        normalize=lambda config: normalize_config_fusion(config, strict=False),
        fallback_config_path=RUNTIME_PATHS.runtime_read_dir / CONFIG_PATH.name,
    )


def load_config() -> dict[str, Any]:
    return runtime_config_store().load()


def effective_runtime_config(
    config: dict[str, Any],
    *,
    entitled: bool | None = None,
) -> dict[str, Any]:
    is_entitled = license_ops.is_entitled() if entitled is None else entitled
    if PROVIDER_PACK and not is_entitled:
        return config_without_provider_pack(config)
    return copy.deepcopy(config)


def config_without_provider_pack(config: dict[str, Any]) -> dict[str, Any]:
    filtered = copy.deepcopy(config)
    providers = filtered.get("providers") if isinstance(filtered.get("providers"), dict) else {}
    filtered["providers"] = {
        provider_id: provider
        for provider_id, provider in providers.items()
        if provider_id not in PROVIDER_PACK
    }
    return filtered


def load_runtime_config() -> dict[str, Any]:
    return effective_runtime_config(load_config())


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def configured_virtual_model_ids(config: dict[str, Any]) -> list[str]:
    profiles = config.get("virtual_profiles") if isinstance(config.get("virtual_profiles"), dict) else {}
    configured = {str(key) for key in profiles if str(key) != FUSION_MODEL_ID and is_safe_virtual_profile_id(str(key))}
    return sorted(VIRTUAL_MODELS | configured)


# Returned by every /admin/license* route when the closed pack is absent (the free open core
# has no license to manage). One constant since four handlers share the exact body.
_LICENSE_PRO_ONLY_404 = {"error": {"message": "License management is a Ficelle Pro feature", "type": "not_found"}}


def fusion_visible_in_model_list(
    config: dict[str, Any],
    *,
    entitled: bool | None = None,
) -> bool:
    # The Fusion engine ships only in the closed ficelle_pro pack; on a core-only install
    # FusionRunner is None. Also require a valid Pro license — never advertise
    # ficelle/auto-fusion when the engine is missing or the license is inactive/expired,
    # since routing to it would otherwise return a missing-engine or license error.
    is_entitled = license_ops.is_entitled() if entitled is None else entitled
    if FusionRunner is None or not is_entitled:
        return False
    return fusion_visible_in_model_list_use_case(
        config,
        ports=fusion_config_ports(),
        configured_virtual_model_ids=configured_virtual_model_ids,
    )


def listed_virtual_model_ids(
    config: dict[str, Any],
    *,
    entitled: bool | None = None,
) -> list[str]:
    model_ids = configured_virtual_model_ids(config)
    if fusion_visible_in_model_list(config, entitled=entitled):
        model_ids = [*model_ids, FUSION_MODEL_ID]
    return sorted(dict.fromkeys(model_ids))


def safe_profile_requirements(raw: Any, config: dict[str, Any]) -> dict[str, Any]:
    requirements = raw if isinstance(raw, dict) else {}
    base_min_context = safe_int(config.get("min_context_length"), DEFAULT_PROFILE_REQUIREMENTS["min_context"])
    requested_min_context = safe_int(
        requirements.get("min_context", requirements.get("min_context_length", base_min_context)),
        base_min_context,
    )
    structured = requirements.get("structured", DEFAULT_PROFILE_REQUIREMENTS["structured"])
    return {
        "free": True,
        "tools": True,
        "min_context": max(base_min_context, requested_min_context),
        "structured": structured if isinstance(structured, bool) else None,
        "input_modalities": safe_string_list(requirements.get("input_modalities")),
        "input_modalities_any": safe_string_list(requirements.get("input_modalities_any")),
        "output_modalities": safe_string_list(requirements.get("output_modalities")),
        "output_modalities_any": safe_string_list(requirements.get("output_modalities_any")),
        "supported_parameters": safe_string_list(requirements.get("supported_parameters")),
        "supported_parameters_any": safe_string_list(requirements.get("supported_parameters_any")),
    }


def normalize_profile_model_id_list(
    profile_id: str,
    raw_model_ids: Any,
    known_model_ids: set[str] | None,
    field_name: str,
    stale_model_ids: set[str] | None = None,
) -> list[str]:
    if raw_model_ids is None:
        raw_model_ids = []
    if not isinstance(raw_model_ids, list):
        raise ValueError(f"{field_name} for {profile_id} must be a list")

    model_ids: list[str] = []
    seen: set[str] = set()
    for item in raw_model_ids:
        model_id = str(item).strip()
        if not model_id or model_id in seen:
            continue
        if known_model_ids is not None and model_id not in known_model_ids:
            # `stale_model_ids` are ids the saved config already carries: the catalog
            # moved under them, the caller did not invent them. See
            # validate_virtual_profiles_payload for why they are not an error.
            if not stale_model_ids or model_id not in stale_model_ids:
                raise ValueError(f"unknown catalog model id for {profile_id}: {model_id}")
        seen.add(model_id)
        model_ids.append(model_id)
    return model_ids


def normalize_virtual_profile(
    profile_id: str,
    raw_profile: Any,
    config: dict[str, Any],
    known_model_ids: set[str] | None = None,
    stale_model_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not is_virtual_profile_id(profile_id):
        raise ValueError(f"virtual model id must start with ficelle/: {safe_detail(profile_id) or '[redacted]'}")
    if profile_id == FUSION_MODEL_ID:
        raise ValueError(f"virtual model id is reserved for Fusion runtime: {FUSION_MODEL_ID}")
    if not is_safe_virtual_profile_id(profile_id):
        raise ValueError(f"virtual model id contains unsupported characters: {safe_detail(profile_id) or '[redacted]'}")
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    mode = str(profile.get("mode") or "auto")
    if mode not in {"auto", "manual_order"}:
        raise ValueError(f"unsupported mode for {profile_id}: {mode}")

    model_ids = normalize_profile_model_id_list(
        profile_id,
        profile.get("models", profile.get("model_ids", profile.get("order", []))),
        known_model_ids,
        "models",
        stale_model_ids,
    )
    excluded_model_ids = normalize_profile_model_id_list(
        profile_id,
        profile.get("excluded_models", []),
        known_model_ids,
        "excluded_models",
        stale_model_ids,
    )

    requirements = safe_profile_requirements(profile.get("requirements"), config)
    requirements.update(VIRTUAL_MODEL_REQUIREMENTS.get(canonical_virtual_model_id(profile_id), {}))
    return {
        "mode": mode,
        "models": model_ids,
        "excluded_models": excluded_model_ids,
        "auto_tail": bool(profile.get("auto_tail", True)),
        "requirements": requirements,
    }


def normalized_virtual_profiles(config: dict[str, Any]) -> dict[str, Any]:
    raw_profiles = config.get("virtual_profiles") if isinstance(config.get("virtual_profiles"), dict) else {}
    profiles: dict[str, Any] = {}
    for profile_id in configured_virtual_model_ids(config):
        profiles[profile_id] = normalize_virtual_profile(profile_id, raw_profiles.get(profile_id, {}), config)
    return profiles


def persisted_profile_model_ids(config: dict[str, Any]) -> set[str]:
    """Every model id the saved virtual profiles already reference, across all fields."""
    saved = config.get("virtual_profiles")
    if not isinstance(saved, dict):
        return set()
    ids: set[str] = set()
    for profile in saved.values():
        if not isinstance(profile, dict):
            continue
        # Same key aliases normalize_virtual_profile accepts, plus exclusions.
        for field in ("models", "model_ids", "order", "excluded_models"):
            raw = profile.get(field)
            if not isinstance(raw, list):
                continue
            ids.update(candidate for candidate in (str(item).strip() for item in raw) if candidate)
    return ids


def validate_virtual_profiles_payload(payload: dict[str, Any], config: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    raw_profiles = payload.get("virtual_profiles", payload.get("profiles"))
    if raw_profiles is None and all(is_virtual_profile_id(str(key)) for key in payload):
        raw_profiles = payload
    if not isinstance(raw_profiles, dict):
        raise ValueError("virtual_profiles must be an object")

    known_model_ids = {
        str(model.get("id"))
        for model in catalog.get("models", [])
        if isinstance(model, dict) and model.get("id")
    }

    # A saved id that is no longer in the catalog is the catalog moving, not a bad
    # request: free models come and go, and a provider that fails during a refresh
    # shrinks the catalog for as long as it is down. Rejecting the payload over one of
    # those blocked every other virtual model too, since a single POST carries them
    # all — and it contradicted startup, where normalized_virtual_profiles() loads the
    # very same ids with no catalog to check against. Carry them through instead of
    # dropping them, so an order survives a model that comes back; the dashboard shows
    # one as "Unknown model" with a control to remove it deliberately. Ids that were
    # never saved are still rejected, which is what catches a malformed client.
    stale_model_ids = persisted_profile_model_ids(config) - known_model_ids

    normalized: dict[str, Any] = {}
    for raw_key, raw_profile in raw_profiles.items():
        profile_id = str(raw_key).strip()
        normalized[profile_id] = normalize_virtual_profile(
            profile_id, raw_profile, config, known_model_ids, stale_model_ids
        )

    for profile_id in sorted(VIRTUAL_MODELS):
        normalized.setdefault(profile_id, normalize_virtual_profile(profile_id, {}, config))
    return dict(sorted(normalized.items()))


def save_virtual_profiles(config: dict[str, Any], profiles: dict[str, Any]) -> dict[str, Any]:
    def apply_profiles(disk_config: dict[str, Any]) -> None:
        disk_config["virtual_profiles"] = profiles

    runtime_config_store().update(apply_profiles)

    config["allow_paid_fallback"] = False
    config["virtual_profiles"] = profiles
    normalize_config_fusion(config, strict=False)
    return profiles


def rollback_virtual_profiles_from_audit(audit_id: str, config: dict[str, Any]) -> dict[str, Any]:
    needle = str(audit_id or "").strip()
    if not needle:
        raise ValueError("audit_id is required")
    row = find_admin_audit_entry(needle)
    if row is None:
        raise ValueError(f"rollback audit entry not found: {needle}")
    assert row is not None
    if row.get("action") != "admin.profiles.save":
        raise ValueError("only admin.profiles.save entries can be rolled back")
    before = row.get("before") if isinstance(row.get("before"), dict) else {}
    profiles = before.get("virtual_profiles") if isinstance(before.get("virtual_profiles"), dict) else None
    if profiles is None:
        raise ValueError("audit entry has no virtual model snapshot to restore")
    current = normalized_virtual_profiles(config)
    catalog = load_or_refresh_catalog(config)
    restored = validate_virtual_profiles_payload({"virtual_profiles": profiles}, config, catalog)
    restored = save_virtual_profiles(config, restored)
    write_admin_audit(
        "admin.rollback",
        before={"virtual_profiles": current},
        after={"virtual_profiles": restored},
        metadata={"resource": "virtual_profiles", "source_audit_id": needle},
    )
    return restored


def validate_fusion_payload(payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("fusion must be an object")
    raw = payload.get("fusion", payload.get("config", payload))
    if not isinstance(raw, dict):
        raise ValueError("fusion must be an object")
    allowed = set(configured_virtual_model_ids(config)) if isinstance(config, dict) else set(VIRTUAL_MODELS)
    return normalize_fusion_config(raw, strict=True, allowed_profile_ids=allowed)


def save_fusion_config(config: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    fusion = normalize_fusion_config(fusion, strict=True, allowed_profile_ids=set(configured_virtual_model_ids(config)))

    def apply_fusion(disk_config: dict[str, Any]) -> None:
        disk_config["fusion"] = fusion
        experimental = disk_config.get("experimental") if isinstance(disk_config.get("experimental"), dict) else {}
        disk_config["experimental"] = {**experimental, "auto_fusion": fusion["enabled"]}

    runtime_config_store().update(apply_fusion)

    config["allow_paid_fallback"] = False
    config["fusion"] = fusion
    experimental_config = config.get("experimental") if isinstance(config.get("experimental"), dict) else {}
    config["experimental"] = {**experimental_config, "auto_fusion": fusion["enabled"]}
    return fusion


def rollback_fusion_from_audit(audit_id: str, config: dict[str, Any]) -> dict[str, Any]:
    needle = str(audit_id or "").strip()
    if not needle:
        raise ValueError("audit_id is required")
    row = find_admin_audit_entry(needle)
    if row is None:
        raise ValueError(f"rollback audit entry not found: {needle}")
    assert row is not None
    if row.get("action") != "admin.fusion.save":
        raise ValueError("only admin.fusion.save entries can be rolled back")
    before = row.get("before") if isinstance(row.get("before"), dict) else {}
    raw_fusion = before.get("fusion") if isinstance(before.get("fusion"), dict) else None
    if raw_fusion is None:
        raise ValueError("audit entry has no fusion snapshot to restore")
    current = normalize_fusion_config(config.get("fusion"), strict=False)
    restored = normalize_fusion_config(raw_fusion, strict=True, allowed_profile_ids=set(configured_virtual_model_ids(config)))
    restored = save_fusion_config(config, restored)
    write_admin_audit(
        "admin.rollback",
        before={"fusion": current},
        after={"fusion": restored},
        metadata={"resource": "fusion", "source_audit_id": needle},
    )
    return restored


# Cooldown reasons tunable on the admin Settings view. Derived from the canonical
# DEFAULT_CONFIG["cooldown_seconds"] so the two can never drift; dict insertion order is
# the display order. allow_paid_fallback is deliberately not tunable: strict-zero is fixed.
SETTINGS_COOLDOWN_REASONS = tuple(DEFAULT_CONFIG["cooldown_seconds"])
SETTINGS_COOLDOWN_MIN_SECONDS = 5
SETTINGS_COOLDOWN_MAX_SECONDS = 604800  # 7 days
SETTINGS_BACKOFF_MAX_STEPS = 12


def router_settings_policy() -> RouterSettingsPolicy:
    return RouterSettingsPolicy(
        defaults=DEFAULT_CONFIG,
        cooldown_reasons=SETTINGS_COOLDOWN_REASONS,
        default_quota_probe_backoff_seconds=DEFAULT_QUOTA_PROBE_BACKOFF_SECONDS,
        cooldown_min_seconds=SETTINGS_COOLDOWN_MIN_SECONDS,
        cooldown_max_seconds=SETTINGS_COOLDOWN_MAX_SECONDS,
        backoff_max_steps=SETTINGS_BACKOFF_MAX_STEPS,
        verified_capability_ttl_min_seconds=MIN_VERIFIED_CAPABILITY_TTL_SECONDS,
        verified_capability_ttl_max_seconds=MAX_VERIFIED_CAPABILITY_TTL_SECONDS,
        auto_benchmark_interval_min_seconds=MIN_AUTO_BENCHMARK_INTERVAL_SECONDS,
        auto_benchmark_interval_max_seconds=MAX_AUTO_BENCHMARK_INTERVAL_SECONDS,
        auto_benchmark_models_min_per_cycle=MIN_AUTO_BENCHMARK_MODELS_PER_CYCLE,
        auto_benchmark_models_max_per_cycle=MAX_AUTO_BENCHMARK_MODELS_PER_CYCLE,
        benchmark_long_context_min_tokens=LONG_CONTEXT_BENCHMARK_MIN_TOKENS,
        benchmark_long_context_max_tokens=LONG_CONTEXT_BENCHMARK_MAX_TOKENS,
        benchmark_compression_min_chars=COMPRESSION_BENCHMARK_MIN_CHARS,
        benchmark_compression_max_chars=COMPRESSION_BENCHMARK_MAX_CHARS,
        normalize_compression_settings=normalize_compression_settings,
    )


def settings_int(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    strict: bool,
    label: str | None = None,
) -> int:
    return settings_int_use_case(
        raw,
        key,
        default,
        minimum=minimum,
        maximum=maximum,
        strict=strict,
        label=label,
    )


def settings_bool(raw: dict[str, Any], key: str, default: bool, *, strict: bool) -> bool:
    return settings_bool_use_case(raw, key, default, strict=strict)


def normalize_settings_backoff(raw: Any, *, strict: bool) -> list[int]:
    return normalize_settings_backoff_use_case(raw, policy=router_settings_policy(), strict=strict)


def normalize_router_settings(raw: Any, *, strict: bool = False) -> dict[str, Any]:
    return normalize_router_settings_use_case(raw, policy=router_settings_policy(), strict=strict)


def validate_settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_settings_payload_use_case(payload, policy=router_settings_policy())


def router_settings_payload(config: dict[str, Any]) -> dict[str, Any]:
    return router_settings_payload_use_case(config, policy=router_settings_policy())


def apply_verified_capability_ttl(config: dict[str, Any]) -> int:
    """Sync the active verified-capability TTL from config (operator-tunable via Settings).

    Called at the routing/admin/settings entry points (select_models, build_admin_status,
    save_router_settings); every staleness read is downstream of one of these, so the cache is
    current before the deep chokepoint reads it. Out-of-range/invalid values fall back to the
    default (matching normalize_router_settings), so the active TTL is always at least the 24h
    floor whether the value came from the validated API save path or a hand-edited config file.

    The cache is a plain module int: writes are atomic and a concurrent read across request threads
    can at worst see the operator's own just-saved value one request early, which is harmless at a
    24h+ scale (both outcomes — trust or re-benchmark — are safe), so no lock is taken.
    """
    global _active_verified_capability_ttl_seconds
    _active_verified_capability_ttl_seconds = verified_capability_ttl_seconds_use_case(
        config,
        policy=router_settings_policy(),
    )
    return _active_verified_capability_ttl_seconds


def route_on_capability_reference_enabled() -> bool:
    return _route_on_capability_reference


def apply_route_on_capability_reference(config: dict[str, Any]) -> bool:
    """Sync the Phase B opt-in from config (operator-tunable via Settings, default off).

    Mirrors apply_verified_capability_ttl: the route gate reads this module flag, not threaded
    config, so it is synced at the same entry points (select_models, build_admin_status,
    save_router_settings). Lock-free for the same reason — a one-request-early read across threads
    only flips between the gate's two safe behaviors (treat as default-claimed vs reference-routable).
    """
    global _route_on_capability_reference
    _route_on_capability_reference = route_on_capability_reference_use_case(
        config,
        policy=router_settings_policy(),
    )
    return _route_on_capability_reference


def auto_benchmark_enabled(config: dict[str, Any]) -> bool:
    return auto_benchmark_enabled_use_case(config, policy=router_settings_policy())


def auto_benchmark_interval_seconds(config: dict[str, Any]) -> int:
    return auto_benchmark_interval_seconds_use_case(config, policy=router_settings_policy())


def auto_benchmark_models_per_cycle(config: dict[str, Any]) -> int:
    return auto_benchmark_models_per_cycle_use_case(config, policy=router_settings_policy())


def save_router_settings(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return save_router_settings_use_case(
        config,
        settings,
        policy=router_settings_policy(),
        ports=RouterSettingsSavePorts(
            update_config=runtime_config_store().update,
            apply_verified_capability_ttl=apply_verified_capability_ttl,
            apply_route_on_capability_reference=apply_route_on_capability_reference,
        ),
    )


def rollback_router_settings_from_audit(audit_id: str, config: dict[str, Any]) -> dict[str, Any]:
    needle = str(audit_id or "").strip()
    if not needle:
        raise ValueError("audit_id is required")
    row = find_admin_audit_entry(needle)
    if row is None:
        raise ValueError(f"rollback audit entry not found: {needle}")
    assert row is not None
    if row.get("action") != "admin.settings.save":
        raise ValueError("only admin.settings.save entries can be rolled back")
    before = row.get("before") if isinstance(row.get("before"), dict) else {}
    raw_settings = before.get("settings") if isinstance(before.get("settings"), dict) else None
    if raw_settings is None:
        raise ValueError("audit entry has no settings snapshot to restore")
    current = normalize_router_settings(config, strict=False)
    restored = save_router_settings(config, normalize_router_settings(raw_settings, strict=True))
    write_admin_audit(
        "admin.rollback",
        before={"settings": current},
        after={"settings": restored},
        metadata={"resource": "settings", "source_audit_id": needle},
    )
    return restored


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env_file_key(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    return line.split("=", 1)[0].strip() or None


def env_file_set_key(path: Path, key: str, value: str) -> None:
    """Set ``KEY=value`` in the ``.env`` file at ``path``, replacing an existing line
    for ``KEY`` or appending. Creates the file/dir if needed and restricts permissions
    to the owner (the file holds a plaintext secret). Rejects a multi-line value: a
    newline would split into bogus lines that ``parse_env_file`` would misread."""
    if "\n" in value or "\r" in value:
        raise ValueError("credential value must be single-line for .env storage")
    lines = path.read_text(errors="ignore").splitlines() if path.exists() else []
    replaced = False
    for index, raw_line in enumerate(lines):
        if _env_file_key(raw_line) == key:
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def env_file_delete_key(path: Path, key: str) -> bool:
    """Remove the ``.env`` line for ``KEY``. Returns True when a line was removed."""
    if not path.exists():
        return False
    original = path.read_text(errors="ignore").splitlines()
    kept = [raw_line for raw_line in original if _env_file_key(raw_line) != key]
    if len(kept) == len(original):
        return False
    path.write_text(("\n".join(kept) + "\n") if kept else "")
    return True


def _unlock_dedicated_keychain(keychain_path: Path) -> Path | None:
    """Return a dedicated keychain path, but ONLY when it exists and
    could be unlocked non-interactively with the empty password it is meant to
    carry. Returning ``None`` on any unlock failure is deliberate: it stops the
    caller from running ``security find-generic-password`` against a *locked*
    keychain, which pops a blocking GUI password dialog in the launchd-managed
    daemon and never resolves (incident 2026-06-16)."""
    if not keychain_path.exists():
        return None
    try:
        result = subprocess.run(
            ["security", "unlock-keychain", "-p", "", str(keychain_path)],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return keychain_path


def _unlock_ficelle_keychain() -> Path | None:
    """Return Ficelle's canonical write keychain when it unlocks non-interactively."""
    return _unlock_dedicated_keychain(FICELLE_SECRETS_KEYCHAIN)


def _credential_env_file_paths() -> tuple[Path, ...]:
    """Canonical credential file followed by unique read-only migration fallbacks."""
    return tuple(dict.fromkeys((CREDENTIAL_ENV_FILE, LEGACY_CREDENTIAL_ENV_FILE)))


def _legacy_credential_env_file_paths() -> tuple[Path, ...]:
    """Unique read-only credential files that are not the canonical Ficelle file."""
    return tuple(
        path
        for path in _credential_env_file_paths()
        if path != CREDENTIAL_ENV_FILE
    )


def _legacy_credential_keychain_paths() -> tuple[Path, ...]:
    """Unique read-only keychains that are not Ficelle's canonical write keychain."""
    candidates = (
        LEGACY_FICELLE_SECRETS_KEYCHAIN,
        LEGACY_HERMES_SECRETS_KEYCHAIN,
    )
    return tuple(
        dict.fromkeys(
            path
            for path in candidates
            if path != FICELLE_SECRETS_KEYCHAIN
        )
    )


def _credential_keychain_paths() -> tuple[Path, ...]:
    """Canonical write keychain followed by unique read-only upgrade fallbacks."""
    return (FICELLE_SECRETS_KEYCHAIN, *_legacy_credential_keychain_paths())


def _login_keychain_path() -> Path:
    return Path.home() / "Library" / "Keychains" / "login.keychain-db"


def _read_scoped_keychain_secret(service: str, keychain: Path) -> str:
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w", str(keychain)],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def read_canonical_keychain_secret(service: str) -> str:
    """Read login then Ficelle's dedicated keychain without consulting legacy paths."""
    candidate = _read_scoped_keychain_secret(service, _login_keychain_path())
    if candidate:
        return candidate
    keychain = _unlock_ficelle_keychain()
    return _read_scoped_keychain_secret(service, keychain) if keychain is not None else ""


def unlock_and_read_legacy_keychain_secret(service: str) -> str:
    """Unlock and read unique legacy keychains without changing stored secrets."""
    for keychain_path in _legacy_credential_keychain_paths():
        keychain = _unlock_dedicated_keychain(keychain_path)
        if keychain is None:
            continue
        candidate = _read_scoped_keychain_secret(service, keychain)
        if candidate:
            return candidate
    return ""


def _scoped_keychain_secret_exists(service: str, keychain: Path) -> bool:
    """Probe a scoped keychain entry without asking `security` for its value."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, str(keychain)],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def unlock_and_list_legacy_keychain_secret_sources(
    services: Sequence[str],
) -> list[tuple[str, Path]]:
    """Unlock legacy keychains and list matching service/path pairs without values."""
    sources: list[tuple[str, Path]] = []
    for keychain_path in _legacy_credential_keychain_paths():
        keychain = _unlock_dedicated_keychain(keychain_path)
        if keychain is None:
            continue
        for service in services:
            if _scoped_keychain_secret_exists(service, keychain):
                sources.append((service, keychain))
    return sources


def read_keychain_secret(service: str) -> str:
    """Compatibility lookup across canonical then legacy scoped keychains.

    An unscoped lookup is deliberately avoided because it can prompt for locked
    keychains in a launchd-managed process.
    """
    return (
        read_canonical_keychain_secret(service)
        or unlock_and_read_legacy_keychain_secret(service)
    )


class SecretStore:
    """OS secret store accessed by service name. ``get`` reads; ``set``/``delete``
    write.

    Implementations must never raise into the caller and never trigger an
    interactive/GUI unlock prompt. ``get`` returns the stored secret, or ``None``
    when the service is absent or the store is unavailable on this host. ``label``
    names the backend in redacted source/reason strings (never the secret). A store
    that cannot write returns ``False`` from ``set``/``delete`` so the caller can fall
    back to the ``.env`` file.
    """

    label = "store"

    def get(self, service: str) -> str | None:
        raise NotImplementedError

    def get_legacy(self, service: str) -> str | None:
        return None

    def set(self, service: str, secret: str) -> bool:
        return False

    def delete(self, service: str) -> bool:
        return False


class NullSecretStore(SecretStore):
    """No OS secret store on this host; resolution falls back to env/.env only."""

    label = "null"

    def get(self, service: str) -> str | None:
        return None


def keychain_store_write(service: str, secret: str) -> bool:
    """Write a secret to the macOS login Keychain in the user session (``-U`` updates
    an existing item, enabling rotation). Returns True on success; never raises.

    CLI-only: a launchd daemon must not call this — writing to the login keychain from
    the background daemon can pop a blocking GUI prompt (incident 2026-06-16, R9). The
    server-side write path targets a non-interactive store or the ``.env`` file."""
    try:
        result = subprocess.run(
            ["security", "add-generic-password", "-a", os.getenv("USER") or "", "-s", service, "-w", secret, "-U"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def keychain_store_delete(service: str) -> bool:
    """Delete a generic-password item from the macOS login Keychain. Returns True when
    an item was removed; never raises."""
    try:
        result = subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                service,
                str(_login_keychain_path()),
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


class KeychainStore(SecretStore):
    """macOS Keychain backend over the non-interactive, scoped lookup."""

    label = "keychain"

    def get(self, service: str) -> str | None:
        return read_canonical_keychain_secret(service) or None

    def get_legacy(self, service: str) -> str | None:
        return unlock_and_read_legacy_keychain_secret(service) or None

    def set(self, service: str, secret: str) -> bool:
        return keychain_store_write(service, secret)

    def delete(self, service: str) -> bool:
        login_deleted = keychain_store_delete(service)
        dedicated_deleted = dedicated_keychain_delete(service)
        return login_deleted or dedicated_deleted


def secret_tool_lookup(service: str) -> str:
    """Linux libsecret read via the ``secret-tool`` CLI, keyed on a ``service``
    attribute. Returns "" when absent or on any error; never raises, never prompts."""
    try:
        result = subprocess.run(
            ["secret-tool", "lookup", "service", service],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def secret_tool_write(service: str, secret: str) -> bool:
    """Store a secret in Linux libsecret via ``secret-tool``, keyed on the ``service``
    attribute. The secret is passed on stdin, never on argv. Returns True on success."""
    try:
        result = subprocess.run(
            ["secret-tool", "store", "--label", f"ficelle {service}", "service", service],
            input=secret,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def secret_tool_delete(service: str) -> bool:
    """Clear a libsecret item keyed on the ``service`` attribute. Returns True on
    success; never raises."""
    try:
        result = subprocess.run(
            ["secret-tool", "clear", "service", service],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


class SecretToolStore(SecretStore):
    """Linux libsecret backend via the ``secret-tool`` CLI."""

    label = "secretservice"

    def get(self, service: str) -> str | None:
        return secret_tool_lookup(service) or None

    def set(self, service: str, secret: str) -> bool:
        return secret_tool_write(service, secret)

    def delete(self, service: str) -> bool:
        return secret_tool_delete(service)


CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2


def _windows_advapi32():
    """Load advapi32 and the CREDENTIALW ctypes structure for Win32 credential calls.

    Importable only on Windows (``ctypes.WinDLL`` is absent elsewhere), so callers wrap
    this in try/except. Factored so the read and write paths share one struct
    definition."""
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    class _Credential(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_char)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    return ctypes, wintypes, advapi32, _Credential


def windows_credential_read(service: str) -> str:
    """Windows Credential Manager read for a generic credential named ``service``.

    ``cmdkey`` can write a credential but cannot read its secret back, so the read
    goes through the Win32 ``CredReadW`` API via ``ctypes`` against ``advapi32`` —
    still the standard library, no third-party package. The blob is decoded as UTF-8
    (the write path stores UTF-8 to match). Returns "" on absence or any error; never
    raises."""
    try:
        ctypes, wintypes, advapi32, _Credential = _windows_advapi32()
        advapi32.CredReadW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(_Credential)),
        ]
        advapi32.CredReadW.restype = wintypes.BOOL
        advapi32.CredFree.argtypes = [ctypes.c_void_p]

        pointer = ctypes.POINTER(_Credential)()
        if not advapi32.CredReadW(service, CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
            return ""
        try:
            blob_size = int(pointer.contents.CredentialBlobSize)
            if blob_size <= 0:
                return ""
            blob = ctypes.string_at(pointer.contents.CredentialBlob, blob_size)
            return blob.decode("utf-8", "ignore").strip()
        finally:
            advapi32.CredFree(pointer)
    except Exception:
        return ""


def windows_credential_write(service: str, secret: str) -> bool:
    """Write a generic credential via ``CredWriteW``, storing the secret as a UTF-8
    blob to match ``windows_credential_read``. Returns True on success; never raises."""
    try:
        ctypes, wintypes, advapi32, _Credential = _windows_advapi32()
        advapi32.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
        advapi32.CredWriteW.restype = wintypes.BOOL

        blob = secret.encode("utf-8")
        buffer = ctypes.create_string_buffer(blob, len(blob))  # kept alive until the call returns
        cred = _Credential()
        cred.Type = CRED_TYPE_GENERIC
        cred.TargetName = service
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))
        cred.Persist = CRED_PERSIST_LOCAL_MACHINE
        return bool(advapi32.CredWriteW(ctypes.byref(cred), 0))
    except Exception:
        return False


def windows_credential_delete(service: str) -> bool:
    """Delete a generic credential via ``CredDeleteW``. Returns True on success; never
    raises."""
    try:
        ctypes, wintypes, advapi32, _Credential = _windows_advapi32()
        advapi32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        advapi32.CredDeleteW.restype = wintypes.BOOL
        return bool(advapi32.CredDeleteW(service, CRED_TYPE_GENERIC, 0))
    except Exception:
        return False


class WindowsCredentialStore(SecretStore):
    """Windows Credential Manager backend via the Win32 Cred* APIs."""

    label = "wincred"

    def get(self, service: str) -> str | None:
        return windows_credential_read(service) or None

    def set(self, service: str, secret: str) -> bool:
        return windows_credential_write(service, secret)

    def delete(self, service: str) -> bool:
        return windows_credential_delete(service)


def default_secret_store() -> SecretStore:
    """Return the OS secret store for this host, degrading silently to a null store.

    macOS uses the Keychain, Windows the Credential Manager, and Linux libsecret via
    ``secret-tool`` when it is installed. Any host without a usable native store
    (headless Linux, an unknown platform) falls back to ``NullSecretStore`` so
    resolution proceeds on env/.env alone — a missing store is a normal, silent
    condition, never an error.
    """
    if sys.platform == "darwin":
        return KeychainStore()
    if sys.platform == "win32":
        return WindowsCredentialStore()
    if sys.platform.startswith("linux") and shutil.which("secret-tool"):
        return SecretToolStore()
    return NullSecretStore()


def dedicated_keychain_read(service: str) -> str:
    """Read a secret from the dedicated, empty-password Ficelle keychain (macOS). Returns
    "" when the keychain is absent/not unlockable or the item is missing."""
    keychain = _unlock_ficelle_keychain()
    return _read_scoped_keychain_secret(service, keychain) if keychain is not None else ""


def dedicated_keychain_write(service: str, secret: str) -> bool:
    """Write to the dedicated, empty-password Ficelle keychain (macOS), which unlocks
    non-interactively — so the launchd daemon never pops the GUI prompt the login
    keychain would. Returns False when the keychain is absent (caller falls back to
    .env).

    The secret is passed via ``-w`` (argv): ``security add-generic-password`` has no
    stdin option for the password, so it is briefly visible to a same-user ``ps`` —
    the same caveat as ``keychain_store_write``, and no broader than what the same user
    can already read from the keychain. Linux ``secret_tool_write`` avoids this (stdin).
    """
    keychain = _unlock_ficelle_keychain()
    if keychain is None:
        return False
    try:
        result = subprocess.run(
            ["security", "add-generic-password", "-a", os.getenv("USER") or "", "-s", service, "-w", secret, "-U", str(keychain)],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def dedicated_keychain_delete(service: str) -> bool:
    """Delete an item from the dedicated Ficelle keychain (macOS). Returns False when the
    keychain is absent or the item is missing."""
    keychain = _unlock_ficelle_keychain()
    if keychain is None:
        return False
    try:
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", service, str(keychain)],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


class DedicatedKeychainStore(SecretStore):
    """macOS dedicated, empty-password Ficelle keychain. Unlike the login keychain it
    unlocks non-interactively, so the launchd daemon can write to it without a GUI
    prompt (incident 2026-06-16). Used only for server-side writes; resolution reads
    cover it already through ``read_keychain_secret``."""

    label = "keychain"

    def get(self, service: str) -> str | None:
        return dedicated_keychain_read(service) or None

    def set(self, service: str, secret: str) -> bool:
        return dedicated_keychain_write(service, secret)

    def delete(self, service: str) -> bool:
        return dedicated_keychain_delete(service)


def server_secret_store() -> SecretStore:
    """The non-interactive secret store for SERVER-SIDE writes (the launchd/systemd
    daemon and the admin web form).

    It must never touch a store that can pop a GUI prompt in the background daemon, so
    on macOS it targets the dedicated empty-password keychain rather than the login
    keychain. Windows Credential Manager (CredWrite) and Linux libsecret write without a
    prompt, so they reuse the same backends as reads. Any host where the chosen store is
    absent or fails degrades to the configured `.env` via ``store_provider_key`` — the
    write path stays cross-platform, never macOS-only.
    """
    if sys.platform == "darwin":
        return DedicatedKeychainStore()
    if sys.platform == "win32":
        return WindowsCredentialStore()
    if sys.platform.startswith("linux") and shutil.which("secret-tool"):
        return SecretToolStore()
    return NullSecretStore()


def resolve_credentials(
    env_names: list[str],
    services: list[str],
    *,
    store: SecretStore | None = None,
    validator: Callable[[Any], bool] | None = None,
) -> tuple[str | None, str]:
    """Resolve env, canonical file/store, then read-only legacy fallbacks.

    ``env_names`` are tried in order against the process environment and the
    canonical ``.env`` file; ``services`` are then tried against the canonical OS
    store. Legacy Hermes ``.env`` and keychains are consulted last. ``validator``
    optionally gates each candidate (e.g. the OpenRouter ``sk-or-`` format check).
    Returns ``(key, source)`` with redacted source/reason labels only.
    """
    store = store if store is not None else default_secret_store()
    invalid_sources: list[str] = []

    def accept(value: Any, source: str) -> bool:
        if not value:
            return False
        if validator is not None and not validator(value):
            invalid_sources.append(source)
            return False
        return True

    for env_name in env_names:
        value = os.getenv(env_name)
        source = f"env:{env_name}"
        if accept(value, source):
            return str(value).strip(), source

    canonical_env_values = parse_env_file(CREDENTIAL_ENV_FILE)
    for env_name in env_names:
        value = canonical_env_values.get(env_name)
        source = f"{CREDENTIAL_ENV_FILE}:{env_name}"
        if accept(value, source):
            return str(value).strip(), source

    for service in services:
        value = store.get(service)
        source = f"{store.label}:{service}"
        if accept(value, source):
            return str(value).strip(), source

    for credential_file in _legacy_credential_env_file_paths():
        env_values = parse_env_file(credential_file)
        for env_name in env_names:
            value = env_values.get(env_name)
            source = f"{credential_file}:{env_name}"
            if accept(value, source):
                return str(value).strip(), source

    legacy_get = getattr(store, "get_legacy", None)
    if callable(legacy_get):
        for service in services:
            value = legacy_get(service)
            source = f"{store.label}:{service}"
            if accept(value, source):
                return str(value).strip(), source

    if invalid_sources:
        return None, f"invalid {env_names[0]} from {', '.join(invalid_sources)}"
    return None, f"missing {env_names[0]}"


def legacy_provider_credential_sources(
    source: str,
    config: dict[str, Any],
) -> list[str]:
    """Return redacted, read-only labels for configured legacy credential sources."""
    provider_cfg = (config.get("providers") or {}).get(source) or {}
    env_names, services = generic_provider_credential_aliases(source, provider_cfg)
    configured: list[str] = []
    for credential_file in _legacy_credential_env_file_paths():
        env_values = parse_env_file(credential_file)
        configured.extend(
            f"{credential_file}:{env_name}"
            for env_name in env_names
            if env_values.get(env_name)
        )
    configured.extend(
        f"keychain:{keychain_path}:{service}"
        for service, keychain_path in unlock_and_list_legacy_keychain_secret_sources(
            services
        )
    )
    return configured


# Per-OS secret-store backends report their store via ``SecretStore.label``; map those
# raw labels (keyed off the classes so a label rename can't silently drift) to the names
# a user recognises in the admin UI.
_CREDENTIAL_STORE_DISPLAY = {
    KeychainStore.label: "keychain",
    SecretToolStore.label: "secret-service",
    WindowsCredentialStore.label: "credential-manager",
}


def credential_source_label(source: str | None) -> str | None:
    """Map a redacted resolution ``source`` (owned by ``resolve_credentials``) to a
    coarse store label for the admin UI — ``env``, ``.env``, an OS store name, or
    ``external`` — so a user can confirm a saved key landed in the encrypted store
    rather than plaintext ``.env``. Returns ``None`` when no key is configured. Never
    contains key material. Matches by full prefix so a Windows ``.env`` path (``C:\\...``)
    is not mis-split on its drive-letter colon."""
    if not source or source.startswith(("missing ", "invalid ")):
        return None
    if source.startswith("env:"):
        return "env"
    for credential_file in _credential_env_file_paths():
        if source.startswith(f"{credential_file}:"):
            return ".env"
    for label, display in _CREDENTIAL_STORE_DISPLAY.items():
        if source.startswith(f"{label}:"):
            return display
    return "external"


def generic_provider_credential_activation_fingerprint(source: str, provider_cfg: dict[str, Any]) -> dict[str, Any]:
    return generic_provider_credential_activation_fingerprint_use_case(
        source,
        provider_cfg,
        env_get=os.getenv,
        parse_env_file=parse_env_file,
        credential_env_files=_credential_env_file_paths(),
        keychain_paths=(
            _login_keychain_path(),
            *_credential_keychain_paths(),
        ),
    )


# External credential resolvers (R4): optional integration hooks consulted as the
# lowest-priority credential source, after env/.env/OS store. Each resolver takes a
# provider id and returns (api_key, base_url, source), or None when it does not handle it.
CredentialResolver = Callable[[str], "tuple[str | None, str | None, str] | None"]


_external_credential_resolvers: list[CredentialResolver] = [
    build_hermes_runtime_resolver(HERMES_AGENT_DIR)
]


def register_credential_resolver(resolver: CredentialResolver, *, prepend: bool = False) -> None:
    """Register an external credential resolver (R4). Resolvers are consulted in
    registration order — lowest priority, after env/.env/OS store — and the first to
    return a usable key wins. ``prepend`` places it ahead of existing resolvers (e.g.
    the shipped default)."""
    if prepend:
        _external_credential_resolvers.insert(0, resolver)
    else:
        _external_credential_resolvers.append(resolver)


def resolve_via_external_resolvers(provider: str, fallback_reason: str) -> tuple[str | None, str | None, str]:
    """Consult the registered external resolvers for ``provider`` in registration
    (priority) order. Return the first usable (api_key, base_url, source); else the
    first handled-but-unresolved result, so the highest-priority resolver's diagnostic
    is kept rather than clobbered by a lower-priority one; else
    ``(None, None, fallback_reason)`` when no resolver handles it. A resolver that
    raises is reduced to a redacted ``type: message`` reason."""
    first_unresolved: tuple[str | None, str | None, str] | None = None

    def remember(result: tuple[str | None, str | None, str]) -> None:
        nonlocal first_unresolved
        if first_unresolved is None:
            first_unresolved = result

    for resolver in list(_external_credential_resolvers):
        try:
            result = resolver(provider)
        except Exception as exc:
            remember((None, None, f"{type(exc).__name__}: {exc}"))
            continue
        if result is None:
            continue
        key, base_url, source = result
        if key:
            return key, base_url, source
        remember((None, base_url, source))
    if first_unresolved is not None:
        return first_unresolved
    return None, None, fallback_reason


def is_zero_price(value: Any) -> bool:
    try:
        return float(value or 0) == 0.0
    except Exception:
        return False


def strict_zero_pricing(pricing: Any) -> tuple[bool, str, dict[str, Any]]:
    """Return a strict free-pricing verdict for anti false-free protection."""
    if not isinstance(pricing, dict):
        return False, "pricing is not an object", {"status": "unsafe", "checked_fields": []}
    required = ("prompt", "completion")
    missing = [field for field in required if field not in pricing]
    if missing:
        return False, f"missing pricing fields: {', '.join(missing)}", {"status": "unsafe", "checked_fields": sorted(pricing)}
    checked: dict[str, str] = {}
    for key, value in pricing.items():
        if value in {None, ""}:
            return False, f"empty pricing field: {key}", {"status": "unsafe", "checked_fields": sorted(checked)}
        try:
            numeric = float(value)
        except Exception:
            return False, f"non numeric pricing field: {key}", {"status": "unsafe", "checked_fields": sorted(checked)}
        checked[str(key)] = str(value)
        if numeric != 0.0:
            return False, f"non-zero pricing field: {key}", {"status": "unsafe", "checked_fields": sorted(checked)}
    return True, "strict zero pricing", {
        "status": "strict_zero",
        "checked_fields": sorted(checked),
        "guard": "all exposed pricing fields are numeric zero",
    }


def pricing_has_non_zero_numeric(pricing: Any) -> bool:
    if not isinstance(pricing, dict):
        return False
    for value in pricing.values():
        try:
            if float(value) != 0.0:
                return True
        except Exception:
            continue
    return False


def strict_zero_free_access(pricing_reason: str) -> dict[str, Any]:
    return free_access_payload(True, "catalog_free", "provider_pricing", "model", "available", pricing_reason)


def unavailable_free_access(mode: str, proof: str, note: str) -> dict[str, Any]:
    safe_mode = mode if mode in FREE_ACCESS_MODES else "unknown"
    return free_access_payload(False, safe_mode, proof, "model", "unavailable", note)


def has_safe_free_access_proof(value: str) -> bool:
    return bool(value) and "[redacted]" not in value


# Shared mapping from a discriminating capability to the OpenAI-compatible request parameters
# that signal it, plus the input modalities tracked as capabilities. One source of truth so the
# default-provenance diff (capabilities_from_defaults) and the OpenRouter oracle
# (oracle_model_capabilities) agree on what each capability means.
CAPABILITY_PARAM_ALIASES = (
    ("tools", ("tools", "tool_choice")),
    ("structured", ("response_format", "structured_outputs")),
    ("reasoning", ("reasoning", "include_reasoning")),
)
CAPABILITY_INPUT_MODALITIES = ("image", "video", "audio")


def capabilities_from_defaults(raw_model: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    """Discriminating capabilities a catalog row claims ONLY via provider defaults.

    Sparse free-quota catalogs expose no capability signal, so Ficelle fills it from
    provider ``catalog_model_defaults``. Those filled-in claims are guesses, not provider
    declarations: the route gate requires a default-claimed capability to be
    benchmark-verified before a specialized profile will route to it, while a real catalog
    declaration is trusted as today. Returns the subset of
    {tools, structured, reasoning, image, video, audio} present only because of the defaults.
    """
    raw_params = set(safe_string_list(raw_model.get("supported_parameters")))
    default_params = set(safe_string_list(defaults.get("supported_parameters")))
    raw_arch = raw_model.get("architecture") if isinstance(raw_model.get("architecture"), dict) else {}
    default_arch = defaults.get("architecture") if isinstance(defaults.get("architecture"), dict) else {}
    raw_inputs = set(safe_string_list(raw_arch.get("input_modalities")))
    default_inputs = set(safe_string_list(default_arch.get("input_modalities")))

    defaulted: list[str] = []
    # Per-alias set difference (in the defaults, absent from the raw row). Intentionally
    # finer-grained than has_tools/has_structured, which collapse to one bool over a single
    # iterable — do not "DRY" this into those, the difference semantics would be lost.
    for capability, aliases in CAPABILITY_PARAM_ALIASES:
        if any(alias in default_params for alias in aliases) and not any(alias in raw_params for alias in aliases):
            defaulted.append(capability)
    for modality in CAPABILITY_INPUT_MODALITIES:
        if modality in default_inputs and modality not in raw_inputs:
            defaulted.append(modality)
    return defaulted


def canonical_model_key(model_id: str) -> str:
    """Normalize a provider model id to a provider-agnostic key for reference lookup.

    Conservative on purpose (R2 of the cross-provider capability reference PRD): lowercase,
    drop a ``:free``-style tag, take the bare slug after the last ``/`` (so ``meta-llama/...``
    and Gemini's ``models/...`` prefixes collapse), and unify ``_`` to ``-``. It does NOT
    strip date or serving suffixes (``-versatile``/``-2505``); those variants are mapped by the
    curated alias map, not guessed here, so a wrong inherited capability never slips in via fuzzy
    normalization.
    """
    key = str(model_id or "").strip().lower()
    if not key:
        return ""
    key = key.split(":", 1)[0]
    key = key.rsplit("/", 1)[-1]
    return key.replace("_", "-").strip()


def _normalize_capability_reference(raw: Any, *, drop_dangling_aliases: bool = True) -> dict[str, Any]:
    """Normalize a raw reference document (bundled file or test input) to {models, aliases}.

    Keys and alias endpoints are run through ``canonical_model_key`` so the seed can be written
    with vendor-qualified or differently-cased ids and still match. Capability lists are coerced
    to clean string lists; malformed entries are dropped rather than raising.

    With ``drop_dangling_aliases`` (the default for a single self-contained document), an alias is
    kept only when its target is a real model entry — dropping dangling targets and forbidding
    alias chains (the lookup is deliberately single-hop). The merge path passes False so a curated
    alias may point at an oracle-only model; the merged store is re-pruned against the merged models.
    """
    if not isinstance(raw, dict):
        return {"models": {}, "aliases": {}}
    raw_models = raw.get("models") if isinstance(raw.get("models"), dict) else {}
    models: dict[str, Any] = {}
    for key, entry in raw_models.items():
        canonical = canonical_model_key(key)
        if not canonical or not isinstance(entry, dict):
            continue
        models[canonical] = {**entry, "capabilities": safe_string_list(entry.get("capabilities"))}
    raw_aliases = raw.get("aliases") if isinstance(raw.get("aliases"), dict) else {}
    aliases: dict[str, str] = {}
    for variant, target in raw_aliases.items():
        v = canonical_model_key(variant)
        t = canonical_model_key(target)
        if v and v != t and (not drop_dangling_aliases or t in models):
            aliases[v] = t
    return {"models": models, "aliases": aliases}


_CAPABILITY_REFERENCE_CACHE: dict[str, Any] | None = None


def _curated_capability_seed_path() -> Path:
    """Resolve the curated capability seed (the high-confidence prior).

    The full curated set is data-as-moat and ships in the closed ``ficelle_pro`` pack;
    when it is installed, prefer its copy. A core-only install falls back to the thin
    bundled seed and relies on the OpenRouter oracle for its prior.
    """
    from ficelle.pro import load_pro

    pro = load_pro()
    if pro is not None and pro.__file__:
        pro_seed = Path(pro.__file__).resolve().parent / "data" / "model_capabilities.json"
        if pro_seed.is_file():
            return pro_seed
    return CAPABILITY_REFERENCE_PATH


def _build_capability_reference() -> dict[str, Any]:
    """Merge the two reference layers into one lookup store.

    The OpenRouter oracle cache (refreshable, broad) is the base layer; the curated in-repo seed
    (hand-reviewed, high confidence) overrides it on key collisions. Aliases from both are merged
    (curated wins) and then pruned against the merged models so a curated alias may legitimately
    target an oracle-only model. Both layers are capability metadata only — no pricing/endpoint.
    """
    curated = _normalize_capability_reference(load_json(_curated_capability_seed_path(), {}), drop_dangling_aliases=False)
    oracle = _normalize_capability_reference(load_runtime_json(CAPABILITY_ORACLE_CACHE_PATH, {}), drop_dangling_aliases=False)
    # Backfill the confidence tier per layer so an older oracle cache written without it still
    # reports a tier (oracle = medium, curated seed = high). reference_confidence and the Phase B
    # gate read this; setdefault preserves any tier already present in the source document.
    for entry in oracle["models"].values():
        entry.setdefault("confidence", "medium")
    for entry in curated["models"].values():
        entry.setdefault("confidence", "high")
    models = {**oracle["models"], **curated["models"]}
    aliases = {**oracle["aliases"], **curated["aliases"]}
    aliases = {variant: target for variant, target in aliases.items() if target in models}
    return {"models": models, "aliases": aliases}


def capability_reference() -> dict[str, Any]:
    """Lazily build and cache the merged capability reference store (curated seed + oracle)."""
    global _CAPABILITY_REFERENCE_CACHE
    if _CAPABILITY_REFERENCE_CACHE is None:
        _CAPABILITY_REFERENCE_CACHE = _build_capability_reference()
    return _CAPABILITY_REFERENCE_CACHE


def invalidate_capability_reference_cache() -> None:
    """Drop the merged-store cache so the next lookup rebuilds it (e.g. after an oracle refresh)."""
    global _CAPABILITY_REFERENCE_CACHE
    _CAPABILITY_REFERENCE_CACHE = None


def canonical_reference_key(model_id: str, reference: dict[str, Any]) -> str:
    """The key a model resolves to in the reference store: canonical slug + one alias hop.

    This is the identity the reference verdict is actually about (e.g. ``llama-3.3-70b-versatile``
    resolves to ``llama-3.3-70b-instruct``), so it is what the discrepancy feed should record.
    """
    key = canonical_model_key(model_id)
    if not key:
        return ""
    return reference.get("aliases", {}).get(key, key)


def reference_entry_for_model(model_id: str, reference: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a normalized reference entry for a model id via canonical key + one alias hop."""
    resolved = canonical_reference_key(model_id, reference)
    entry = reference.get("models", {}).get(resolved) if resolved else None
    return entry if isinstance(entry, dict) else None


def capabilities_from_reference(
    default_claimed: Any,
    model_id: str,
    reference: dict[str, Any] | None = None,
) -> list[str]:
    """Default-claimed capabilities that the curated cross-provider reference CONFIRMS.

    Narrows, never widens: returns only capabilities that the row already claims via provider
    ``catalog_model_defaults`` AND that the reference asserts for this model's canonical identity.
    A real catalog declaration is already trusted (it is not default-claimed), and a capability the
    row does not claim is never added. Drives provenance/observability and Phase A benchmark
    targeting; under the Phase B opt-in (off by default) a ``high``-confidence confirmation here also
    opens the route gate via ``model_capability_reference_routable`` — but a ``failed`` verdict always
    overrides and the verified-capability TTL keeps re-checking. Matching is exact canonical-key or
    curated alias; unmatched ids return ``[]`` rather than a fuzzy guess.
    """
    claimed = safe_string_list(default_claimed)
    if not claimed:
        return []
    store = capability_reference() if reference is None else _normalize_capability_reference(reference)
    entry = reference_entry_for_model(model_id, store)
    if not entry:
        return []
    # capabilities are already coerced to a clean str list by _normalize_capability_reference.
    confirmed = set(entry.get("capabilities") or [])
    return [capability for capability in claimed if capability in confirmed]


def capabilities_refuted_by_reference(
    default_claimed: Any,
    model_id: str,
    reference: dict[str, Any] | None = None,
) -> list[str]:
    """Default-claimed capabilities the cross-provider reference CONTRADICTS for a matched model.

    The mirror of :func:`capabilities_from_reference`. When the reference HAS an entry for this
    model (so it has an opinion on the base model) but does NOT list a capability the row claims
    only via ``catalog_model_defaults``, that default is an over-claim the reference refutes (e.g.
    a sparse provider stamps ``tools`` on a small model OpenRouter knows has no tool-calling).
    Returns ``[]`` when the model has no reference entry — absence is "unconfirmed", not "refuted".
    Observability/safety signal only: the benchmark stays the authority that proves or demotes.
    """
    claimed = safe_string_list(default_claimed)
    if not claimed:
        return []
    store = capability_reference() if reference is None else _normalize_capability_reference(reference)
    entry = reference_entry_for_model(model_id, store)
    if not entry:
        return []
    asserted = set(entry.get("capabilities") or [])
    return [capability for capability in claimed if capability not in asserted]


def context_length_provenance(
    declared_context: int | None,
    default_context: int,
    upstream_id: str,
) -> tuple[str, int | None]:
    """Provenance of a model's context window plus a display-only estimate.

    Returns ``(source, estimate)``. ``source`` is ``"catalog"`` when the provider catalog declared
    the window (``declared_context`` is not None), else ``"default"`` (filled from
    ``catalog_model_defaults``). ``estimate`` is a cross-provider reference prior (oracle/curated),
    populated only for a ``default`` window and only when it differs from the safe default. It is
    display-only and never feeds routing or the min-context filter — R9: the routed window is never
    inflated above a provider-declared value.
    """
    if declared_context is not None:
        return "catalog", None
    entry = reference_entry_for_model(upstream_id, capability_reference())
    estimate = safe_optional_int(entry.get("context_length")) if entry else None
    if estimate is not None and estimate != default_context:
        return "default", estimate
    return "default", None


def model_reference_confidence(model_id: str, reference: dict[str, Any] | None = None) -> str:
    """Confidence tier of a model's reference entry (``high`` curated seed / ``medium`` oracle),
    or ``""`` when the model has no entry. Phase B routing trusts only the tiers in
    CAPABILITY_REFERENCE_ROUTE_CONFIDENCES.
    """
    store = capability_reference() if reference is None else _normalize_capability_reference(reference)
    entry = reference_entry_for_model(model_id, store)
    return str(entry.get("confidence") or "") if entry else ""


def oracle_model_capabilities(raw_model: dict[str, Any]) -> list[str]:
    """Capabilities OpenRouter advertises for a model, in the shared capability vocabulary.

    Uses the same CAPABILITY_PARAM_ALIASES / CAPABILITY_INPUT_MODALITIES mapping as
    capabilities_from_defaults, so the oracle and the default-provenance diff never disagree on
    what a capability means (e.g. ``tools`` is recognized from ``tools`` or ``tool_choice``).
    """
    params = set(safe_string_list(raw_model.get("supported_parameters")))
    arch = raw_model.get("architecture") if isinstance(raw_model.get("architecture"), dict) else {}
    inputs = set(safe_string_list(arch.get("input_modalities")))
    caps: list[str] = []
    for capability, aliases in CAPABILITY_PARAM_ALIASES:
        if any(alias in params for alias in aliases):
            caps.append(capability)
    for modality in CAPABILITY_INPUT_MODALITIES:
        if modality in inputs:
            caps.append(modality)
    return caps


def oracle_identity_keys(raw_model: dict[str, Any]) -> list[str]:
    """Canonical keys to index an OpenRouter row under: its id, canonical_slug, and HF basename."""
    keys: list[str] = []
    for field_name in ("id", "canonical_slug", "hugging_face_id"):
        key = canonical_model_key(str(raw_model.get(field_name) or ""))
        if key and key not in keys:
            keys.append(key)
    return keys


def build_capability_oracle(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn a raw OpenRouter ``/models`` list into a capability-only oracle document.

    Capability metadata ONLY — pricing, endpoints, and routing data are never imported, so the
    oracle is a metadata source and never a paid-traffic destination (strict-zero). Each row is
    indexed under its id/slug/HF-basename canonical keys. When several rows map to one canonical
    key (e.g. a model's ``:free`` and paid variants, whose listed params can differ), their
    capabilities are UNIONed and the larger context window kept — the model's full advertised
    ceiling, order-independent and deterministic. Each key gets its own entry (no shared dict).
    """
    models: dict[str, Any] = {}
    for raw in data:
        if not isinstance(raw, dict):
            continue
        caps = oracle_model_capabilities(raw)
        context_length = safe_optional_int(raw.get("context_length"))
        for key in oracle_identity_keys(raw):
            existing = models.get(key)
            if existing is None:
                # "medium": the oracle describes the base model, not a provider's specific
                # deployment, so it stays Phase A targeting only — below the Phase B route floor.
                models[key] = {"capabilities": list(caps), "context_length": context_length, "source": "openrouter", "confidence": "medium"}
                continue
            for capability in caps:
                if capability not in existing["capabilities"]:
                    existing["capabilities"].append(capability)
            if context_length is not None and (existing["context_length"] is None or context_length > existing["context_length"]):
                existing["context_length"] = context_length
    return {"fetched_at": now_iso(), "source": "openrouter", "model_count": len(data), "models": models, "aliases": {}}


def fetch_capability_oracle(timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch OpenRouter's PUBLIC ``/models`` catalog (no credential) as a capability oracle.

    Returns (oracle_document, None) on success or (None, reason) on any failure so callers can
    degrade to the existing cache instead of crashing a refresh.
    """
    try:
        response = requests.get(OPENROUTER_ORACLE_URL, timeout=timeout, headers={"Accept": "application/json"})
    except Exception as exc:  # network / timeout — never let the oracle break a refresh
        return None, f"fetch error: {type(exc).__name__}"
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}"
    try:
        payload = response.json()
    except Exception:
        return None, "invalid JSON"
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None, "invalid oracle shape"
    return build_capability_oracle(data), None


def capability_oracle_is_stale(cache: Any, *, now_ts: float | None = None, ttl_seconds: int | None = None) -> bool:
    """True when the oracle cache is missing, empty, undated, or older than the TTL."""
    if not isinstance(cache, dict) or not cache.get("models"):
        return True
    fetched = parse_iso_timestamp(cache.get("fetched_at"))
    if fetched is None:
        return True
    ttl = CAPABILITY_ORACLE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    now = time.time() if now_ts is None else now_ts
    return (now - fetched) > ttl


def refresh_capability_oracle_if_stale(config: dict[str, Any]) -> dict[str, Any]:
    """Refresh the OpenRouter capability oracle cache when stale, then invalidate the merged store.

    Network happens ONLY here, and this is called only from explicit refresh points (serve startup,
    ``ficelle refresh``) — never on the route path, which reads the persisted catalog/cache offline.
    On any fetch failure the existing cache is kept (graceful degradation).
    """
    cache = load_runtime_json(CAPABILITY_ORACLE_CACHE_PATH, {})
    if not capability_oracle_is_stale(cache):
        return cache if isinstance(cache, dict) else {}
    timeout = float(config.get("catalog_timeout_seconds") or 30)
    fresh, _error = fetch_capability_oracle(timeout)
    if not fresh:
        return cache if isinstance(cache, dict) else {}
    atomic_write_json(CAPABILITY_ORACLE_CACHE_PATH, fresh)
    invalidate_capability_reference_cache()
    return fresh


def reference_capability_signal(profile_id: str, model: dict[str, Any]) -> str:
    """How the capability reference rates a model for a specialized profile's discriminating
    capability: ``confirms`` (reference asserts it), ``refutes`` (reference knows the model and it
    lacks the capability), or ``unknown`` (no reference entry, or a non-specialized profile).

    Phase A signal only (R5): it orders the benchmark queue and detects benchmark-vs-reference
    discrepancies. It never gates routing — a ``refutes`` model is deprioritized, not excluded.
    """
    capability = GATE_CAPABILITY_BY_PROFILE.get(canonical_virtual_model_id(profile_id))
    if not capability:
        return "unknown"
    entry = reference_entry_for_model(str(model.get("upstream_id") or ""), capability_reference())
    if not entry:
        return "unknown"
    return "confirms" if capability in safe_string_list(entry.get("capabilities")) else "refutes"


_BENCHMARK_REFERENCE_PRIORITY = {"confirms": 0, "unknown": 1, "refutes": 2}


def order_benchmark_candidates(profile_id: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable-sort benchmark candidates by reference signal: reference-confirmed first, refuted last
    (Phase A targeting, R5). The default benchmark stops at the first model that passes, so probing
    confirmed models first reaches a ``verified`` result with fewer live calls against tight
    free-quota limits. Ordering only — every candidate stays in the set, so a refuted-but-actually-
    capable model is still benchmarked if nothing better passes (no pool starvation).
    """
    return sorted(
        candidates,
        key=lambda model: _BENCHMARK_REFERENCE_PRIORITY.get(reference_capability_signal(profile_id, model), 1),
    )


def record_capability_discrepancy(profile_id: str, model: dict[str, Any], passed: bool) -> None:
    """Append a benchmark-vs-reference contradiction to the discrepancy feed (R7).

    Called only on a *semantic* benchmark verdict (the probe ran and was validated), never on a
    transport/quota failure — a 429/500/timeout is not evidence the capability claim was wrong, so
    its placement mirrors record_verified_capability. The benchmark is authoritative; this feed
    records only where the reference *prior* was wrong: it confirmed a capability the probe failed
    (over-claim) or refuted one the probe passed (under-claim). Agreements and unknowns are not
    logged. Privacy-safe: the row carries only canonical id, source, capability, and the two
    verdicts — never prompts, responses, or secrets (and write_route_log redacts again).
    """
    capability = GATE_CAPABILITY_BY_PROFILE.get(canonical_virtual_model_id(profile_id))
    if not capability:
        return
    signal = reference_capability_signal(profile_id, model)
    if not ((signal == "confirms" and not passed) or (signal == "refutes" and passed)):
        return
    # Reuse the shared redact + timestamped JSONL-append convention (same as routes/admin logs).
    write_route_log(
        {
            "profile_id": canonical_virtual_model_id(profile_id),
            # the post-alias key the verdict was compared against, so the feed points at the
            # reference entry to correct (not a serving-variant slug).
            "canonical_key": canonical_reference_key(str(model.get("upstream_id") or ""), capability_reference()),
            "source": model.get("source"),
            "capability": capability,
            "reference": signal,
            "benchmark": "pass" if passed else "fail",
        },
        path=CAPABILITY_DISCREPANCY_LOG_PATH,
    )


def normalized_free_access(
    model: dict[str, Any],
    pricing_ok: bool | None = None,
    pricing_reason: str | None = None,
) -> dict[str, Any]:
    """Return the normalized no-cost access metadata used by routing.

    Existing catalogs without ``free_access`` remain strict-zero only. Provider
    adapters can later mark quota/local free models without pretending their
    pricing fields are numeric zero.
    """
    raw = model.get("free_access")
    if isinstance(raw, dict):
        is_eligible = raw.get("eligible") is True
        mode = str(raw.get("mode") or "unknown")
        if mode not in FREE_ACCESS_MODES:
            mode = "unknown"
        scope = str(raw.get("scope") or "model")
        if scope not in FREE_ACCESS_SCOPES:
            scope = "model"
        proof = safe_detail(raw.get("proof")) or ""
        note = safe_detail(raw.get("note")) or ""
        status = normalized_free_access_status(raw.get("status"), "available" if is_eligible else "unavailable")

        if mode == "catalog_free":
            proof = proof or "provider_pricing"
            if proof == "provider_pricing":
                if pricing_ok is None or pricing_reason is None:
                    pricing_ok, pricing_reason, _pricing_safety = strict_zero_pricing(model.get("pricing"))
                eligible = is_eligible and pricing_ok and status == "available"
                note = note or str(pricing_reason)
            else:
                # FreeModel providers may expose a sparse catalog with no pricing fields.
                # They are trusted only through config-level proof plus an id allowlist,
                # and an explicit non-zero pricing field still fails closed.
                raw_pricing = model.get("pricing")
                if raw_pricing in (None, {}):
                    pricing_ok, pricing_reason = True, "sparse free model catalog"
                else:
                    pricing_ok, pricing_reason, _pricing_safety = strict_zero_pricing(raw_pricing)
                eligible = is_eligible and pricing_ok and has_safe_free_access_proof(proof) and status == "available"
                note = note or ("free model allowlist" if eligible else str(pricing_reason or "free model proof failed"))
            return free_access_payload(eligible, "catalog_free", proof, scope, status, note)

        if mode in {"quota_free", "local_free"}:
            eligible = is_eligible and has_safe_free_access_proof(proof) and status == "available"
            return free_access_payload(
                eligible,
                mode,
                proof or "missing_free_access_proof",
                scope,
                status,
                note or (f"{mode} access" if eligible else "missing free access proof"),
            )

        return free_access_payload(
            False,
            mode,
            proof or "not_free",
            scope,
            "unavailable",
            note or "model is not eligible for free routing",
        )

    if pricing_ok is None or pricing_reason is None:
        pricing_ok, pricing_reason, _pricing_safety = strict_zero_pricing(model.get("pricing"))

    if pricing_ok:
        return strict_zero_free_access(str(pricing_reason))

    mode = "paid" if pricing_has_non_zero_numeric(model.get("pricing")) else "unknown"
    return unavailable_free_access(mode, "provider_pricing", str(pricing_reason or "pricing is not strict zero"))


def pricing_safety_for_free_access(pricing_safety: dict[str, Any], free_access: dict[str, Any]) -> dict[str, Any]:
    mode = str(free_access.get("mode") or "")
    proof = str(free_access.get("proof") or "")
    if free_access.get("eligible") and mode == "catalog_free" and proof != "provider_pricing":
        return {
            "status": "not_applicable",
            "checked_fields": [],
            "guard": f"{mode} access is tracked by {proof}",
        }
    if free_access.get("eligible") and mode in {"quota_free", "local_free"}:
        return {
            "status": "not_applicable",
            "checked_fields": [],
            "guard": f"{mode} access is tracked by free_access metadata",
        }
    return pricing_safety


def has_tools(supported_parameters: Iterable[Any]) -> bool:
    params = {str(item) for item in supported_parameters}
    return "tools" in params or "tool_choice" in params


def has_structured(supported_parameters: Iterable[Any]) -> bool:
    params = {str(item) for item in supported_parameters}
    return "structured_outputs" in params or "response_format" in params


def safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def safe_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def safe_optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def concrete_model_id(source: str, upstream_id: str) -> str:
    return f"ficelle/{source}/{upstream_id}"


def provider_catalog_ports() -> ProviderCatalogPorts:
    return ProviderCatalogPorts(
        provider_catalog_adapter=provider_catalog_adapter,
        resolve_credentials=resolve_generic_provider_credentials,
        resolve_external_credentials=resolve_via_external_resolvers,
        http_get=requests.get,
    )


def provider_invocation_headers(source: str) -> dict[str, str]:
    return provider_invocation_headers_use_case(source, ports=provider_catalog_ports())


def provider_access_result(
    source: str,
    provider_cfg: dict[str, Any],
    *,
    require_base_url: bool,
) -> ProviderAccess:
    return provider_access_result_use_case(
        source,
        provider_cfg,
        require_base_url=require_base_url,
        ports=provider_catalog_ports(),
    )


def fetch_provider_catalog(source: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    return fetch_provider_catalog_use_case(source, config, ports=provider_catalog_ports())


def resolve_generic_provider_credentials(source: str, provider_cfg: dict[str, Any]) -> tuple[str | None, str]:
    env_names, services = generic_provider_credential_aliases(source, provider_cfg)
    return resolve_credentials(env_names, services, validator=PROVIDER_KEY_VALIDATORS.get(source))


def provider_primary_service(source: str, config: dict[str, Any]) -> str:
    """The canonical service name a provider's key is written under — its primary env
    alias (e.g. ``OPENROUTER_API_KEY``). Reads pick it up as the first service tried."""
    return provider_primary_service_use_case(source, config)


def store_provider_key(source: str, config: dict[str, Any], secret: str, *, store: SecretStore | None = None) -> str:
    """Write a provider's API key to the highest-trust writable store on this host,
    under the provider's primary service name, falling back to the configured ``.env``
    (plaintext). Never writes to a process env var. Returns a redacted target label
    (no secret). Refreshing the router so the provider activates is the caller's job.

    ``store`` defaults to ``default_secret_store()`` (the interactive/CLI target — the
    macOS login Keychain). The server-side write path passes a non-interactive store
    so the launchd daemon never touches the login keychain (R9)."""
    store = store if store is not None else default_secret_store()
    return store_provider_key_use_case(
        source,
        config,
        secret,
        store=store,
        credential_env_file=CREDENTIAL_ENV_FILE,
        env_file_set_key=env_file_set_key,
    )


def remove_provider_key(source: str, config: dict[str, Any], *, store: SecretStore | None = None) -> list[str]:
    """Delete a provider's key from Ficelle's canonical OS store and ``.env`` file.

    Returns the redacted targets cleared. Process environment variables and read-only
    legacy fallbacks are intentionally left untouched; callers report any remaining
    legacy sources separately.
    """
    store = store if store is not None else default_secret_store()
    return remove_provider_key_use_case(
        source,
        config,
        store=store,
        credential_env_file=CREDENTIAL_ENV_FILE,
        env_file_delete_key=env_file_delete_key,
    )


def _auth_row(invokable: bool, key: Any, reason: str, base_url: Any) -> dict[str, Any]:
    """Build a redacted provider auth row. ``reason`` doubles as the resolution source
    when ``key`` is set, so ``key_source`` exposes the coarse store (keychain/.env/…)
    without ever leaking the secret."""
    return auth_row_use_case(
        invokable,
        key,
        reason,
        base_url,
        credential_source_label=credential_source_label,
    )


def provider_auth_ports() -> ProviderAuthPorts:
    return ProviderAuthPorts(
        provider_access=lambda source, provider_cfg, require_base_url: provider_access_result(
            source,
            provider_cfg,
            require_base_url=require_base_url,
        ),
        credential_source_label=credential_source_label,
    )


def provider_auth_row(source: str, config: dict[str, Any]) -> dict[str, Any]:
    return provider_auth_row_use_case(source, config, ports=provider_auth_ports())


def auth_status(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return auth_status_use_case(config, ports=provider_auth_ports())


def safe_quota_cooldown_row(key: str, raw_value: dict[str, Any], now_ts: float, *, include_active: bool = False) -> dict[str, Any]:
    return safe_quota_cooldown_row_use_case(
        key,
        raw_value,
        now_ts,
        quota_cooldown_scope_from_key=quota_cooldown_scope_from_key,
        safe_float=safe_float,
        safe_state_key=safe_state_key,
        safe_detail=safe_detail,
        safe_int=safe_int,
        epoch_to_iso=epoch_to_iso,
        include_active=include_active,
    )


def redact_runtime_state(state: Any) -> dict[str, Any]:
    source = state if isinstance(state, dict) else {}
    now = time.time()
    cooldowns: dict[str, Any] = {}
    for key, raw_value in (source.get("cooldowns") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        until = float(raw_value.get("until") or 0)
        cooldowns[safe_state_key(key)] = {
            "reason": safe_detail(raw_value.get("reason")) or "cooldown",
            "set_at": raw_value.get("set_at"),
            "until": until,
            "active": until > now,
            "seconds_remaining": max(0, int(until - now)),
        }

    provider_cooldowns: dict[str, Any] = {}
    for key, raw_value in (source.get("provider_cooldowns") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        until = float(raw_value.get("until") or 0)
        provider_cooldowns[safe_state_key(key)] = {
            "reason": safe_detail(raw_value.get("reason")) or "cooldown",
            "set_at": raw_value.get("set_at"),
            "until": until,
            "active": until > now,
            "seconds_remaining": max(0, int(until - now)),
            "detail": safe_detail(raw_value.get("detail")),
        }

    quota_cooldowns: dict[str, Any] = {}
    for key, raw_value in (source.get("quota_cooldowns") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        quota_cooldowns[safe_state_key(key)] = safe_quota_cooldown_row(str(key), raw_value, now, include_active=True)

    quota_probe_results: dict[str, Any] = {}
    for key, raw_value in (source.get("quota_probe_results") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        quota_probe_results[safe_state_key(key)] = {
            "status": safe_detail(raw_value.get("status")) or "unknown",
            "reason": safe_detail(raw_value.get("reason")),
            "detail": safe_detail(raw_value.get("detail")),
            "probed_at": raw_value.get("probed_at"),
            "source": safe_detail(raw_value.get("source")),
            "model_id": safe_detail(raw_value.get("model_id")),
            "upstream_id": safe_detail(raw_value.get("upstream_id")),
        }

    successes: dict[str, Any] = {}
    for key, raw_value in (source.get("successes") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        successes[safe_state_key(key)] = {
            "last_success_at": raw_value.get("last_success_at"),
            "latency_seconds": raw_value.get("latency_seconds"),
        }

    stats: dict[str, Any] = {}
    for key, raw_value in (source.get("stats") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        stats[safe_state_key(key)] = {
            "requests": safe_int(raw_value.get("requests"), 0),
            "successes": safe_int(raw_value.get("successes"), 0),
            "failures": safe_int(raw_value.get("failures"), 0),
            "consecutive_failures": safe_int(raw_value.get("consecutive_failures"), 0),
            "latency_ewma": raw_value.get("latency_ewma"),
            "last_success_at": raw_value.get("last_success_at"),
            "last_failure_at": raw_value.get("last_failure_at"),
            "last_failure_reason": safe_detail(raw_value.get("last_failure_reason")),
        }

    provider_stats: dict[str, Any] = {}
    for key, raw_value in (source.get("provider_stats") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        provider_stats[safe_state_key(key)] = {
            "requests": safe_int(raw_value.get("requests"), 0),
            "successes": safe_int(raw_value.get("successes"), 0),
            "failures": safe_int(raw_value.get("failures"), 0),
            "consecutive_failures": safe_int(raw_value.get("consecutive_failures"), 0),
            "latency_ewma": raw_value.get("latency_ewma"),
            "last_success_at": raw_value.get("last_success_at"),
            "last_failure_at": raw_value.get("last_failure_at"),
            "last_failure_reason": safe_detail(raw_value.get("last_failure_reason")),
        }

    provider_errors: dict[str, Any] = {}
    for key, raw_value in (source.get("provider_errors") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        reason = str(raw_value.get("reason") or "unknown")
        if reason not in PROVIDER_ERROR_REASONS:
            continue
        provider_errors[safe_state_key(key)] = {
            "reason": safe_detail(reason) or "unknown",
            "detail": safe_detail(raw_value.get("detail")),
            "status": raw_value.get("status"),
            "model_id": safe_detail(raw_value.get("model_id")),
            "upstream_id": safe_detail(raw_value.get("upstream_id")),
            "request_id": safe_detail(raw_value.get("request_id")),
            "seen_at": raw_value.get("seen_at"),
        }

    model_errors: dict[str, Any] = {}
    for key, raw_value in (source.get("model_errors") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        model_errors[safe_state_key(key)] = {
            "reason": safe_detail(raw_value.get("reason")) or "unknown",
            "detail": safe_detail(raw_value.get("detail")),
            "status": raw_value.get("status"),
            "model_id": safe_detail(raw_value.get("model_id")),
            "upstream_id": safe_detail(raw_value.get("upstream_id")),
            "source": safe_detail(raw_value.get("source")),
            "profile_id": safe_detail(raw_value.get("profile_id")),
            "request_id": safe_detail(raw_value.get("request_id")),
            "seen_at": raw_value.get("seen_at"),
        }

    compression_errors: dict[str, Any] = {}
    for key, raw_value in (source.get("compression_errors") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        compression_errors[safe_state_key(key)] = {
            "consecutive_errors": safe_int(raw_value.get("consecutive_errors"), 0),
            "last_error_at": raw_value.get("last_error_at"),
            "last_status": safe_detail(raw_value.get("last_status")) or "unknown",
            "last_error_type": safe_detail(raw_value.get("last_error_type")),
        }

    compression_retrievals: dict[str, Any] = {}
    raw_retrievals = source.get("compression_retrievals") if isinstance(source.get("compression_retrievals"), dict) else {}
    compression_retrievals["success_count"] = safe_int(raw_retrievals.get("success_count"), 0)
    compression_retrievals["missing_count"] = safe_int(raw_retrievals.get("missing_count"), 0)
    compression_retrievals["last_success_at"] = raw_retrievals.get("last_success_at")
    compression_retrievals["last_missing_at"] = raw_retrievals.get("last_missing_at")

    last_routes: dict[str, Any] = {}
    for key, raw_value in (source.get("last_routes") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        route_row = {
            "status": str(raw_value.get("status") or "unknown"),
            "reason": safe_detail(raw_value.get("reason")) or "",
            "request_id": safe_detail(raw_value.get("request_id")),
            "model_id": safe_detail(raw_value.get("model_id")),
            "upstream_id": safe_detail(raw_value.get("upstream_id")),
            "source": safe_detail(raw_value.get("source")),
            "attempt_count": safe_int(raw_value.get("attempt_count"), 0),
            "candidate_count": safe_int(raw_value.get("candidate_count"), 0),
            "latency_seconds": raw_value.get("latency_seconds"),
            "attempts": safe_attempt_rows(raw_value.get("attempts")) if isinstance(raw_value.get("attempts"), list) else [],
            "seen_at": raw_value.get("seen_at"),
        }
        compression = raw_value.get("compression")
        if isinstance(compression, dict):
            route_row["compression"] = safe_compression_metadata(compression)
        last_routes[safe_state_key(key)] = route_row

    quarantine: dict[str, Any] = {}
    for key, raw_value in (source.get("quarantine") or {}).items():
        if not isinstance(raw_value, dict):
            continue
        quarantine[safe_state_key(key)] = {
            "reason": safe_detail(raw_value.get("reason")) or "quarantined",
            "note": safe_detail(raw_value.get("note")),
            "set_at": raw_value.get("set_at"),
            "source": safe_detail(raw_value.get("source")) or "manual",
            "manual": bool(raw_value.get("manual", False)),
            "model_id": safe_detail(raw_value.get("model_id")),
            "upstream_id": safe_detail(raw_value.get("upstream_id")),
        }

    benchmark_results: dict[str, Any] = {}
    for key, raw_profiles in (source.get("benchmark_results") or {}).items():
        if not isinstance(raw_profiles, dict):
            continue
        safe_key = safe_state_key(key)
        safe_profiles = benchmark_results.setdefault(safe_key, {})
        for profile_id, raw_value in raw_profiles.items():
            if not isinstance(raw_value, dict):
                continue
            row = {
                "status": str(raw_value.get("status") or "unknown"),
                "test_type": raw_value.get("test_type"),
                "ran_at": raw_value.get("ran_at"),
                "latency_seconds": raw_value.get("latency_seconds"),
                "model_id": safe_detail(raw_value.get("model_id")),
                "upstream_id": safe_detail(raw_value.get("upstream_id")),
                "message": safe_detail(raw_value.get("message")),
                "text_preview": safe_detail(raw_value.get("text_preview")),
            }
            safe_profiles[safe_state_key(profile_id)] = redacted_benchmark_row(str(profile_id), row)

    verified_capabilities: dict[str, Any] = {}
    for key, raw_profiles in (source.get("verified_capabilities") or {}).items():
        if not isinstance(raw_profiles, dict):
            continue
        safe_key = safe_state_key(key)
        safe_profiles = verified_capabilities.setdefault(safe_key, {})
        for profile_id, raw_value in raw_profiles.items():
            if not isinstance(raw_value, dict):
                continue
            row = {
                "status": str(raw_value.get("status") or "unknown"),
                "capability": str(raw_value.get("capability") or "unknown"),
                "test_type": raw_value.get("test_type"),
                "verified_at": raw_value.get("verified_at"),
                "failed_at": raw_value.get("failed_at"),
                "model_id": safe_detail(raw_value.get("model_id")),
                "upstream_id": safe_detail(raw_value.get("upstream_id")),
                "message": safe_detail(raw_value.get("message")),
            }
            safe_profiles[safe_state_key(profile_id)] = redacted_benchmark_row(str(profile_id), row)

    return {
        "last_catalog_refresh_at": source.get("last_catalog_refresh_at"),
        "cooldowns": cooldowns,
        "provider_cooldowns": provider_cooldowns,
        "quota_cooldowns": quota_cooldowns,
        "quota_probe_results": quota_probe_results,
        "successes": successes,
        "stats": stats,
        "provider_stats": provider_stats,
        "provider_errors": provider_errors,
        "model_errors": model_errors,
        "compression_errors": compression_errors,
        "compression_retrievals": compression_retrievals,
        "last_routes": last_routes,
        "quarantine": quarantine,
        "benchmark_results": benchmark_results,
        "verified_capabilities": verified_capabilities,
        "admin_jobs": active_admin_jobs(source),
        "canary": redact_sensitive_json(source.get("canary")) if isinstance(source.get("canary"), dict) else {},
        "auto_benchmark": redact_sensitive_json(source.get("auto_benchmark")) if isinstance(source.get("auto_benchmark"), dict) else {},
    }


def catalog_with_auto_scores(catalog: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return catalog_with_auto_scores_use_case(
        catalog,
        config,
        state,
        apply_verified_capability_ttl=apply_verified_capability_ttl,
        normalized_virtual_profiles=normalized_virtual_profiles,
        normalized_free_access=normalized_free_access,
        model_score_explanation=model_score_explanation,
    )


def build_fusion_admin_status_builder() -> FusionAdminStatusBuilder:
    return FusionAdminStatusBuilder(
        normalize_fusion_config=normalize_fusion_config,
        fusion_call_multiplier=fusion_call_multiplier,
        judge_history_limit=FUSION_JUDGE_HISTORY_LIMIT,
        reason_code_pattern=FUSION_REASON_CODE_PATTERN,
    )


def safe_fusion_reason_code(value: Any) -> str:
    return build_fusion_admin_status_builder().safe_reason_code(value)


def safe_fusion_last_run(state: dict[str, Any]) -> dict[str, Any]:
    return build_fusion_admin_status_builder().safe_last_run(state)


def fusion_judge_failure_stats(state: dict[str, Any]) -> dict[str, Any]:
    return build_fusion_admin_status_builder().judge_failure_stats(state)


def fusion_quality_signal_summary(state: dict[str, Any]) -> dict[str, Any]:
    return build_fusion_admin_status_builder().quality_signal_summary(state)


def fusion_observability(config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return build_fusion_admin_status_builder().observability(config, state)


def fusion_admin_payload(config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return build_fusion_admin_status_builder().fusion_payload(config, state)


def build_admin_state_builder() -> AdminStateBuilder:
    return AdminStateBuilder(
        AdminStateBuildPorts(
            load_or_refresh_catalog=load_or_refresh_catalog,
            run_due_quota_probes=run_due_quota_probes,
            load_runtime_state=load_runtime_state,
            normalized_virtual_profiles=normalized_virtual_profiles,
            build_admin_status=build_admin_status,
            catalog_with_auto_scores=catalog_with_auto_scores,
            fusion_payload=fusion_admin_payload,
            router_settings_payload=router_settings_payload,
            redact_runtime_state=redact_runtime_state,
            safe_int=safe_int,
            safe_string_list=safe_string_list,
        ),
        default_min_context=DEFAULT_PROFILE_REQUIREMENTS["min_context"],
        default_canary_profiles=list(DEFAULT_CANARY_PROFILES),
    )


def admin_state(config: dict[str, Any]) -> dict[str, Any]:
    from ficelle.pro import pro_installed

    payload = build_admin_state_builder().build(config)
    payload["provider_labels"] = dict(PROVIDER_LABELS)
    # The admin UI reads this to hide the Fusion/Compression action buttons on a
    # core-only install (the views themselves gate structurally — pro-views.js ships
    # and renders them only when the pack is present). False on a core-only install.
    payload["pro_installed"] = pro_installed()
    payload["update"] = update_service.public_update_status()
    return payload


def catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    return build_catalog_refresh_summary(catalog)


CRITICAL_PROFILE_IDS = [
    "ficelle/auto-orchestrator",
    "ficelle/auto-tools",
    "ficelle/auto-json",
]


def epoch_to_iso(value: Any) -> str | None:
    try:
        timestamp = float(value)
    except Exception:
        return None
    if timestamp <= 0:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def active_model_cooldown_rows(state: dict[str, Any], now: float | None = None) -> list[dict[str, Any]]:
    now_ts = time.time() if now is None else now
    return active_model_cooldown_rows_use_case(
        state,
        now_ts,
        safe_state_key=safe_state_key,
        safe_detail=safe_detail,
        safe_int=safe_int,
        epoch_to_iso=epoch_to_iso,
    )


def active_provider_cooldown_rows(state: dict[str, Any], now: float | None = None) -> list[dict[str, Any]]:
    now_ts = time.time() if now is None else now
    return active_provider_cooldown_rows_use_case(
        state,
        now_ts,
        safe_state_key=safe_state_key,
        safe_detail=safe_detail,
        safe_int=safe_int,
        epoch_to_iso=epoch_to_iso,
    )


def active_quota_cooldown_rows(state: dict[str, Any], now: float | None = None) -> list[dict[str, Any]]:
    now_ts = time.time() if now is None else now
    return active_quota_cooldown_rows_use_case(
        state,
        now_ts,
        quota_cooldown_scope_from_key=quota_cooldown_scope_from_key,
        safe_float=safe_float,
        safe_state_key=safe_state_key,
        safe_detail=safe_detail,
        safe_int=safe_int,
        epoch_to_iso=epoch_to_iso,
    )


def quarantine_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    return quarantine_rows_use_case(
        state,
        safe_state_key=safe_state_key,
        safe_detail=safe_detail,
    )


def failed_profile_evidence_row(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return failed_profile_evidence_row_use_case(
        profile_id,
        model,
        state,
        canonical_virtual_model_id=canonical_virtual_model_id,
        verified_capability_row=verified_capability_row,
        raw_benchmark_row=raw_benchmark_row,
        benchmark_result_matches_current_test=benchmark_result_matches_current_test,
        redacted_benchmark_row=redacted_benchmark_row,
        safe_detail=safe_detail,
    )


def candidates_for_profile(
    profile_id: str,
    available: list[dict[str, Any]],
    state: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    return candidates_for_profile_policy(
        profile_id,
        available,
        state,
        profile,
        canonical_virtual_model_id=canonical_virtual_model_id,
        model_matches_profile_requirements=model_matches_profile_requirements,
        manual_order_candidates=manual_order_candidates,
        sort_available_for_virtual_model=sort_available_for_virtual_model,
    )


def candidate_rank_rows(
    profile_id: str,
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    state: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    return candidate_rank_rows_use_case(
        profile_id,
        candidates,
        profile,
        state,
        canonical_virtual_model_id=canonical_virtual_model_id,
        model_score_explanation=model_score_explanation,
        verified_capability_row=verified_capability_row,
        failed_profile_evidence_row=failed_profile_evidence_row,
        model_route_competence=model_route_competence,
        gate_capability_by_profile=GATE_CAPABILITY_BY_PROFILE,
        model_capability_provenance=model_capability_provenance,
        normalized_free_access=normalized_free_access,
        safe_int=safe_int,
        limit=limit,
    )


def model_history_row(profile_id: str, model: dict[str, Any], state: dict[str, Any], rank_row: dict[str, Any]) -> dict[str, Any]:
    return model_history_row_use_case(
        profile_id,
        model,
        state,
        rank_row,
        cooldown_key=cooldown_key,
        normalized_free_access=normalized_free_access,
        raw_benchmark_row=raw_benchmark_row,
        redacted_benchmark_row=redacted_benchmark_row,
        verified_capability_row=verified_capability_row,
        failed_profile_evidence_row=failed_profile_evidence_row,
        safe_detail=safe_detail,
        safe_int=safe_int,
    )


def fallback_attempt_rows(last_route: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attempts = last_route.get("attempts") if isinstance(last_route.get("attempts"), list) else []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        row = safe_attempt_row(attempt)
        reason = str(row.get("reason") or "")
        status = str(row.get("status") or "")
        if reason == "ok" or status in {"200", "ok"}:
            continue
        rows.append(row)
    return rows[:5]


def estimated_tokens_for_chars(char_count: Any) -> int:
    chars = safe_int(char_count, 0)
    if chars <= 0:
        return 0
    return max(1, math.ceil(chars / 4))


def safe_compression_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    strategies = raw.get("strategies") if isinstance(raw.get("strategies"), dict) else {}
    outcomes = raw.get("outcomes") if isinstance(raw.get("outcomes"), dict) else {}
    row: dict[str, Any] = {
        "mode": safe_state_key(raw.get("mode")),
        "status": safe_state_key(raw.get("status")),
        "block_count": safe_int(raw.get("block_count"), 0),
        "compressed_block_count": safe_int(raw.get("compressed_block_count"), 0),
        "estimated_original_chars": safe_int(raw.get("estimated_original_chars"), 0),
        "estimated_compressed_chars": safe_int(raw.get("estimated_compressed_chars"), 0),
        "estimated_saved_chars": safe_int(raw.get("estimated_saved_chars"), 0),
        "estimated_saved_tokens": estimated_tokens_for_chars(raw.get("estimated_saved_chars")),
        "savings_ratio": round(safe_float(raw.get("savings_ratio"), 0.0), 4),
        "needs_original_storage_count": safe_int(raw.get("needs_original_storage_count"), 0),
        "strategies": {safe_state_key(key): safe_int(value, 0) for key, value in strategies.items()},
        "outcomes": {safe_state_key(key): safe_int(value, 0) for key, value in outcomes.items()},
    }
    if raw.get("error_type"):
        row["error_type"] = safe_detail(raw.get("error_type"))
    return redact_sensitive_json(row)


def compression_observability(config: dict[str, Any], runtime_state: dict[str, Any]) -> dict[str, Any]:
    compression_config = normalize_compression_settings(config.get("compression"), strict=False)
    last_routes = runtime_state.get("last_routes") if isinstance(runtime_state.get("last_routes"), dict) else {}
    status_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    strategy_mix: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    route_count = 0
    compressed_route_count = 0
    dry_run_route_count = 0
    error_count = 0
    repeated_error_count = 0
    estimated_original_chars = 0
    estimated_compressed_chars = 0
    estimated_saved_chars = 0
    last_seen_at: str | None = None
    last_profile_id: str | None = None
    last_status: str | None = None

    for profile_id, last_route in last_routes.items():
        if not isinstance(last_route, dict):
            continue
        compression = last_route.get("compression") if isinstance(last_route.get("compression"), dict) else {}
        if not compression:
            continue
        route_count += 1
        status_value = str(compression.get("status") or "unknown")
        mode_value = str(compression.get("mode") or "unknown")
        _increment_count(status_counts, status_value)
        _increment_count(mode_counts, mode_value)
        if status_value == "compressed":
            compressed_route_count += 1
        if status_value == "dry_run":
            dry_run_route_count += 1
        if status_value == "error_original_forwarded":
            error_count += 1
        estimated_original_chars += safe_int(compression.get("estimated_original_chars"), 0)
        estimated_compressed_chars += safe_int(compression.get("estimated_compressed_chars"), 0)
        estimated_saved_chars += safe_int(compression.get("estimated_saved_chars"), 0)
        for key, value in (compression.get("strategies") or {}).items():
            strategy_mix[str(key)] = strategy_mix.get(str(key), 0) + safe_int(value, 0)
        for key, value in (compression.get("outcomes") or {}).items():
            outcome_counts[str(key)] = outcome_counts.get(str(key), 0) + safe_int(value, 0)
        seen_at = str(last_route.get("seen_at") or "")
        if seen_at and (last_seen_at is None or seen_at > last_seen_at):
            last_seen_at = seen_at
            last_profile_id = safe_state_key(profile_id)
            last_status = status_value

    compression_errors = runtime_state.get("compression_errors") if isinstance(runtime_state.get("compression_errors"), dict) else {}
    for raw_error in compression_errors.values():
        if not isinstance(raw_error, dict):
            continue
        repeated_error_count = max(repeated_error_count, safe_int(raw_error.get("consecutive_errors"), 0))
    compression_retrievals = (
        runtime_state.get("compression_retrievals") if isinstance(runtime_state.get("compression_retrievals"), dict) else {}
    )

    try:
        store = compression_store_stats(store_path=COMPRESSION_STORE_PATH)
    except Exception as exc:
        store = {"entry_count": 0, "oldest_created_at": None, "newest_expires_at": None, "error_type": safe_detail(type(exc).__name__)}

    savings_ratio = round(estimated_saved_chars / estimated_original_chars, 4) if estimated_original_chars > 0 else 0.0
    mode = str(compression_config.get("mode") or "off")
    digest = "compression_errors_repeated" if mode != "off" and max(error_count, repeated_error_count) >= 2 else "ok"
    return redact_sensitive_json(
        {
            "config": compression_config,
            "configured_mode": mode,
            "dry_run_enabled": mode == "dry_run",
            "live_zone_enabled": mode == "live_zone",
            "observed_route_count": route_count,
            "compressed_route_count": compressed_route_count,
            "dry_run_route_count": dry_run_route_count,
            "error_count": error_count,
            "repeated_error_count": repeated_error_count,
            "retrieval_success_count": safe_int(compression_retrievals.get("success_count"), 0),
            "retrieval_missing_count": safe_int(compression_retrievals.get("missing_count"), 0),
            "retrieval_last_success_at": compression_retrievals.get("last_success_at"),
            "retrieval_last_missing_at": compression_retrievals.get("last_missing_at"),
            "status_counts": status_counts,
            "mode_counts": mode_counts,
            "strategy_mix": strategy_mix,
            "outcome_counts": outcome_counts,
            "estimated_original_chars": estimated_original_chars,
            "estimated_compressed_chars": estimated_compressed_chars,
            "estimated_saved_chars": estimated_saved_chars,
            "estimated_saved_tokens": estimated_tokens_for_chars(estimated_saved_chars),
            "estimated_savings_ratio": savings_ratio,
            "store": {
                "entry_count": safe_int(store.get("entry_count"), 0),
                "oldest_created_at": store.get("oldest_created_at"),
                "newest_expires_at": store.get("newest_expires_at"),
                "error_type": store.get("error_type"),
            },
            "last_status": last_status,
            "last_profile_id": last_profile_id,
            "last_seen_at": last_seen_at,
            "health_digest": digest,
        }
    )


def compression_route_event(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    compression = raw.get("compression") if isinstance(raw.get("compression"), dict) else None
    if compression is None:
        return None
    attempts = raw.get("attempts") if isinstance(raw.get("attempts"), list) else []
    selected = raw.get("selected_upstream") or raw.get("selected_model")
    final_status = raw.get("final_status")
    event = {
        "logged_at": safe_detail(raw.get("logged_at"), 80),
        "request_id": safe_detail(raw.get("request_id"), 80),
        "requested_model": safe_detail(raw.get("requested_model"), 120),
        "selected_upstream": safe_detail(selected, 160),
        "selected_source": safe_detail(raw.get("selected_source"), 80),
        "final_status": final_status if type(final_status) is int else safe_detail(final_status, 40),
        "final_reason": safe_detail(raw.get("final_reason"), 120),
        "candidate_count": safe_int(raw.get("candidate_count"), 0),
        "attempt_count": safe_int(raw.get("attempt_count"), len(attempts)),
        "duration_seconds": round(safe_float(raw.get("duration_seconds"), 0.0), 4),
        "stream": bool(raw.get("stream")),
        "compression": safe_compression_metadata(compression),
    }
    return redact_sensitive_json(event)


def read_compression_route_events(limit: int = 50, path: Path | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(safe_int(limit, 50), 200))
    path = runtime_read_path(ROUTE_LOG_PATH) if path is None else path
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in reversed(deque(handle, maxlen=1000)):
                if len(events) >= limit:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                event = compression_route_event(row)
                if event is not None:
                    events.append(event)
    except OSError:
        return []
    return events


def compression_admin_payload(config: dict[str, Any], limit: int = 50) -> dict[str, Any]:
    runtime_state = load_runtime_state()
    observability = compression_observability(config, runtime_state)
    events = read_compression_route_events(limit=limit)
    return redact_sensitive_json(
        {
            "generated_at": now_iso(),
            "observability": observability,
            "events": events,
            "event_count": len(events),
        }
    )


def admin_query_limit(path: str, default: int = 50) -> int:
    values = parse_qs(urlparse(path).query, keep_blank_values=True).get("limit")
    return safe_int(values[0], default) if values else default


def request_log_query_payload(path: str) -> dict[str, Any]:
    """Filtered list of recent routed requests from the derived index."""
    params = parse_qs(urlparse(path).query, keep_blank_values=True)

    def first(key: str) -> str | None:
        values = params.get(key)
        return values[0] if values and values[0] != "" else None

    try:
        with _REQUEST_LOG_LOCK:
            rows = request_log.query(
                store_path=REQUEST_LOG_STORE_PATH,
                source_path=runtime_read_path(ROUTE_LOG_PATH),
                limit=admin_query_limit(path, default=request_log.DEFAULT_QUERY_LIMIT),
                profile=first("profile"),
                source=first("source"),
                reason=first("reason"),
                status=first("status"),
                q=first("q"),
            )
        return {"generated_at": now_iso(), "entries": rows}
    except Exception as exc:
        return {"generated_at": now_iso(), "entries": [], "error_type": type(exc).__name__}


def request_log_summary_payload(path: str) -> dict[str, Any]:
    """Aggregate request-health metrics over a window from the derived index."""
    params = parse_qs(urlparse(path).query, keep_blank_values=True)
    window_values = params.get("window")
    window = request_log.window_seconds_from_label(window_values[0] if window_values else None)
    try:
        with _REQUEST_LOG_LOCK:
            return request_log.summary(
                store_path=REQUEST_LOG_STORE_PATH,
                source_path=runtime_read_path(ROUTE_LOG_PATH),
                window_seconds=window,
            )
    except Exception as exc:
        return {"generated_at": now_iso(), "window_seconds": window, "total": 0, "error_type": type(exc).__name__}


def admin_status_ports() -> AdminStatusBuildPorts:
    return AdminStatusBuildPorts(
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=model_is_quarantined,
        normalized_virtual_profiles=normalized_virtual_profiles,
        redact_runtime_state=redact_runtime_state,
        candidates_for_profile=candidates_for_profile,
        route_competent_candidates=route_competent_candidates,
        candidate_rank_rows=candidate_rank_rows,
        verified_capability_row=verified_capability_row,
        model_score_explanation=model_score_explanation,
        model_route_competence=model_route_competence,
        failed_profile_evidence_row=failed_profile_evidence_row,
        active_model_cooldown_rows=active_model_cooldown_rows,
        active_provider_cooldown_rows=active_provider_cooldown_rows,
        active_quota_cooldown_rows=active_quota_cooldown_rows,
        quarantine_rows=quarantine_rows,
        model_history_row=model_history_row,
        model_auto_score=model_auto_score,
        fallback_attempt_rows=fallback_attempt_rows,
        compression_observability=compression_observability,
        runtime_state_diagnostics=runtime_state_diagnostics,
        fusion_payload=fusion_admin_payload,
        now_iso=now_iso,
        safe_int=safe_int,
    )


def build_admin_status_builder() -> AdminStatusBuilder:
    return AdminStatusBuilder(
        ports=admin_status_ports(),
        critical_profile_ids=tuple(CRITICAL_PROFILE_IDS),
        apply_verified_capability_ttl=apply_verified_capability_ttl,
        apply_route_on_capability_reference=apply_route_on_capability_reference,
    )


def build_admin_status(catalog: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return build_admin_status_builder().build(catalog, config, state)


def load_cached_catalog_for_admin_status(config: dict[str, Any]) -> dict[str, Any]:
    catalog = load_runtime_json(CATALOG_PATH, {})
    return load_cached_catalog_for_admin_status_use_case(
        catalog,
        config,
        default_min_context=DEFAULT_CONFIG["min_context_length"],
    )


def cached_catalog_stale_reason(catalog: dict[str, Any], config: dict[str, Any]) -> str | None:
    return cached_catalog_stale_reason_use_case(
        catalog,
        config,
        now_seconds=time.time,
        catalog_config_fingerprint=catalog_config_fingerprint,
        catalog_config_structural_fingerprint=catalog_config_structural_fingerprint,
    )


def filter_cached_catalog_to_enabled_providers(catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return filter_cached_catalog_to_enabled_providers_use_case(
        catalog,
        config,
        provider_auth_row=provider_auth_row,
    )


def license_variant_structural_fingerprints(config: dict[str, Any]) -> set[str]:
    return {
        catalog_config_structural_fingerprint(config),
        catalog_config_structural_fingerprint(config_without_provider_pack(config)),
    }


def admin_status(config: dict[str, Any], *, run_quota_probes: bool = True) -> dict[str, Any]:
    effective_config = effective_runtime_config(config)
    catalog = load_cached_catalog_for_admin_status(effective_config)
    stale_reason = cached_catalog_stale_reason(catalog, effective_config)
    if stale_reason is None:
        if run_quota_probes:
            run_due_quota_probes(effective_config, catalog)
    else:
        filtered_catalog = filter_cached_catalog_to_enabled_providers(catalog, effective_config)
        catalog = {**filtered_catalog, "stale": True, "stale_reason": stale_reason}
        cached_structural_fingerprint = catalog.get("config_structural_fingerprint")
        license_transition = (
            stale_reason == "config_structural_fingerprint_mismatch"
            and cached_structural_fingerprint in license_variant_structural_fingerprints(config)
        )
        # A catalog that is merely aging or whose non-structural config/credentials changed
        # still describes a valid set of models, so keep the last-known catalog (reported
        # stale/"degraded" downstream, not a false "0 models / fail" that makes an idle-but-
        # healthy instance look down). A structurally changed or unparseable catalog is
        # emptied — the recoverable-reason policy lives next to the classifier.
        if not stale_reason_keeps_last_known(stale_reason) and not license_transition:
            catalog["models"] = []
            catalog["providers"] = {}
    state = load_runtime_state()
    return build_admin_status(
        catalog,
        effective_config,
        state if isinstance(state, dict) else {},
    )


def admin_page_html() -> str:
    """Return the admin dashboard shell.

    The dashboard HTML/CSS/JS live as static files under ``assets/admin`` and are
    served via ``/admin/static/``; this keeps router.py focused on the backend.
    """
    asset = load_admin_asset("index.html")
    if asset is None:
        return "<!doctype html><meta charset=utf-8><title>Ficelle</title><body>admin assets missing</body>"
    html = asset[0].decode("utf-8")
    token_meta = f'<meta name="ficelle-admin-token" content="{admin_token()}">'
    html = (
        html
        .replace('/admin/static/admin.css"', f'/admin/static/admin.css?v={admin_asset_version("admin.css")}"')
        .replace('/admin/static/app.js"', f'/admin/static/app.js?v={admin_asset_version("app.js")}"')
        .replace("</head>", f"{token_meta}\n</head>", 1)
    )
    # Inject the closed pack's Fusion/Compression views only when ficelle_pro ships them.
    # On a core-only install the script is absent and the core renders upsell placeholders.
    pro_admin = _pro_admin_dir()
    if pro_admin is not None and (pro_admin / "pro-views.js").is_file():
        pro_src = f'/admin/static/pro/pro-views.js?v={admin_asset_version("pro/pro-views.js")}'
        html = html.replace("</body>", f'  <script type="module" src="{pro_src}"></script>\n</body>', 1)
    return html


def admin_token() -> str:
    """Return the local admin auth token, creating it (0600) on first use.

    The token guards the admin credential-write endpoint against cross-site browser
    requests: it is embedded in the same-origin admin page (see ``admin_page_html``),
    so a page from another origin cannot read it and cannot forge the write."""
    path = ROUTER_DIR / "admin-token"
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except OSError:
        pass
    token = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token)
        path.chmod(0o600)
    except OSError:
        pass
    return token


def is_loopback_origin(origin: str) -> bool:
    """True when an HTTP ``Origin`` header points at this loopback host. Used to reject
    cross-site browser POSTs to the admin write endpoint (CSRF defense)."""
    return is_loopback_origin_use_case(origin)


def catalog_config_fingerprint(config: dict[str, Any], *, include_credentials: bool = True) -> str:
    return catalog_config_fingerprint_use_case(
        config,
        ports=catalog_fingerprint_ports(),
        include_credentials=include_credentials,
    )


def catalog_config_structural_fingerprint(config: dict[str, Any]) -> str:
    return catalog_config_structural_fingerprint_use_case(config, ports=catalog_fingerprint_ports())


def catalog_fingerprint_ports() -> CatalogFingerprintPorts:
    return CatalogFingerprintPorts(
        safe_int=safe_int,
        redact_sensitive_json=redact_sensitive_json,
        credential_activation_fingerprint=generic_provider_credential_activation_fingerprint,
    )


def catalog_refresh_ports() -> CatalogRefreshPorts:
    return CatalogRefreshPorts(
        auth_status=auth_status,
        provider_catalog_adapter=provider_catalog_adapter,
        fetch_provider_catalog=fetch_provider_catalog,
        strict_zero_pricing=strict_zero_pricing,
        normalized_free_access=normalized_free_access,
        pricing_safety_for_free_access=pricing_safety_for_free_access,
        context_length_provenance=context_length_provenance,
        catalog_config_fingerprint=catalog_config_fingerprint,
        catalog_config_structural_fingerprint=catalog_config_structural_fingerprint,
        now_iso=now_iso,
        safe_detail=safe_detail,
        safe_int=safe_int,
        safe_float=safe_float,
        safe_optional_int=safe_optional_int,
        safe_optional_bool=safe_optional_bool,
        safe_string_list=safe_string_list,
        has_tools=has_tools,
        has_structured=has_structured,
        concrete_model_id=concrete_model_id,
        capabilities_from_defaults=capabilities_from_defaults,
        capabilities_from_reference=capabilities_from_reference,
        capabilities_refuted_by_reference=capabilities_refuted_by_reference,
        model_reference_confidence=model_reference_confidence,
        provider_key_url=lambda source: PROVIDER_KEY_URLS.get(source),
    )


def _build_catalog_for_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    return CatalogRefreshRunner(
        catalog_refresh_ports(),
        virtual_models=VIRTUAL_MODELS,
    ).refresh_catalog(config)


def _publish_catalog_unlocked(catalog: dict[str, Any]) -> None:
    publish_catalog_use_case(
        catalog,
        catalog_path=CATALOG_PATH,
        atomic_write_json=atomic_write_json,
        update_state=_STATE_STORE.update,
    )


def _publish_catalog(catalog: dict[str, Any]) -> None:
    with advisory_file_lock(CATALOG_LOCK_PATH, _CATALOG_PUBLISH_THREAD_LOCK):
        _publish_catalog_unlocked(catalog)


def _refresh_catalog_for_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    catalog = _build_catalog_for_effective_config(config)
    _publish_catalog(catalog)
    return catalog


def refresh_catalog(config: dict[str, Any]) -> dict[str, Any]:
    return _refresh_catalog_for_effective_config(effective_runtime_config(config))


def _load_or_refresh_catalog_for_effective_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    catalog = load_or_refresh_catalog_use_case(
        config,
        catalog_path=CATALOG_PATH,
        load_json=load_runtime_json,
        refresh_catalog=_refresh_catalog_for_effective_config,
        catalog_config_fingerprint=catalog_config_fingerprint,
        now_seconds=time.time,
        force=force,
    )
    return filter_cached_catalog_to_enabled_providers(catalog, config)


def _preserve_cached_models_for_failed_providers(
    refreshed: dict[str, Any],
    cached: dict[str, Any],
    effective_config: dict[str, Any],
) -> dict[str, Any]:
    providers = refreshed.get("providers")
    if not isinstance(providers, dict):
        return refreshed
    failed_sources = {
        source
        for source, summary in providers.items()
        if isinstance(summary, dict) and summary.get("error")
    }
    if not failed_sources:
        return refreshed
    cached = filter_cached_catalog_to_enabled_providers(cached, effective_config)
    refreshed_models = [
        model for model in refreshed.get("models", []) if isinstance(model, dict)
    ]
    known_ids = {str(model.get("id") or "") for model in refreshed_models}
    preserved = [
        model
        for model in cached.get("models", [])
        if (
            isinstance(model, dict)
            and model.get("source") in failed_sources
            and str(model.get("id") or "") not in known_ids
        )
    ]
    if not preserved:
        return refreshed
    return {**refreshed, "models": [*refreshed_models, *preserved]}


def _run_catalog_license_transition_refresh(
    effective_config: dict[str, Any],
    cached: dict[str, Any],
) -> None:
    try:
        refreshed = _build_catalog_for_effective_config(effective_config)
        catalog_to_publish = _preserve_cached_models_for_failed_providers(
            refreshed,
            cached,
            effective_config,
        )
        with advisory_file_lock(CATALOG_LOCK_PATH, _CATALOG_PUBLISH_THREAD_LOCK):
            current = load_runtime_json(CATALOG_PATH, {})
            cache_is_unchanged = (
                isinstance(current, dict)
                and current.get("generated_at") == cached.get("generated_at")
                and current.get("config_fingerprint") == cached.get("config_fingerprint")
                and current.get("config_structural_fingerprint")
                == cached.get("config_structural_fingerprint")
            )
            if cache_is_unchanged:
                _publish_catalog_unlocked(catalog_to_publish)
    except Exception:
        # Keep serving the last-known filtered catalog. Because its fingerprint still
        # represents the previous license variant, the next request can retry.
        pass
    finally:
        _LICENSE_TRANSITION_REFRESH_IN_FLIGHT_LOCK.release()


def _start_catalog_license_transition_refresh(
    effective_config: dict[str, Any],
    cached: dict[str, Any],
) -> None:
    if not _LICENSE_TRANSITION_REFRESH_IN_FLIGHT_LOCK.acquire(blocking=False):
        return
    worker = threading.Thread(
        target=_run_catalog_license_transition_refresh,
        args=(copy.deepcopy(effective_config), copy.deepcopy(cached)),
        name="ficelle-catalog-license-refresh",
        daemon=True,
    )
    try:
        worker.start()
    except Exception:
        _LICENSE_TRANSITION_REFRESH_IN_FLIGHT_LOCK.release()
        raise


def _cached_catalog_for_license_transition(
    canonical_config: dict[str, Any],
    effective_config: dict[str, Any],
) -> dict[str, Any] | None:
    catalog = load_runtime_json(CATALOG_PATH, {})
    if not isinstance(catalog, dict) or not catalog.get("models"):
        return None
    cached_structural_fingerprint = catalog.get("config_structural_fingerprint")
    effective_fingerprint = catalog_config_structural_fingerprint(effective_config)
    if (
        cached_structural_fingerprint == effective_fingerprint
        or cached_structural_fingerprint
        not in license_variant_structural_fingerprints(canonical_config)
    ):
        return None
    filtered = filter_cached_catalog_to_enabled_providers(catalog, effective_config)
    fallback = {
        **filtered,
        "stale": True,
        "stale_reason": "config_structural_fingerprint_mismatch",
    }
    _start_catalog_license_transition_refresh(effective_config, fallback)
    return fallback


def load_or_refresh_catalog(
    config: dict[str, Any],
    force: bool = False,
    *,
    entitled: bool | None = None,
) -> dict[str, Any]:
    effective_config = effective_runtime_config(config, entitled=entitled)
    if force:
        return _load_or_refresh_catalog_for_effective_config(effective_config, force=True)
    cached_transition = _cached_catalog_for_license_transition(config, effective_config)
    if cached_transition is not None:
        return cached_transition
    return _load_or_refresh_catalog_for_effective_config(effective_config)


def cooldown_key(model: dict[str, Any]) -> str:
    return cooldown_key_use_case(model)


def cooldown_read_ports() -> CooldownReadPorts:
    return CooldownReadPorts(
        normalized_free_access=normalized_free_access,
        safe_detail=safe_detail,
        safe_float=safe_float,
        now_seconds=time.time,
        free_access_scopes=set(FREE_ACCESS_SCOPES),
    )


def quota_cooldown_scope(model: dict[str, Any]) -> str:
    return quota_cooldown_scope_use_case(model, ports=cooldown_read_ports())


def quota_cooldown_key(model: dict[str, Any]) -> str:
    return quota_cooldown_key_use_case(model, ports=cooldown_read_ports())


def quota_cooldown_key_for_scope(model: dict[str, Any], scope: str) -> str:
    return quota_cooldown_key_for_scope_use_case(model, scope, ports=cooldown_read_ports())


def quota_cooldown_scope_from_key(key: str) -> str:
    return quota_cooldown_scope_from_key_use_case(key, ports=cooldown_read_ports())


def quota_cooldown_matches_model(key: str, model: dict[str, Any]) -> bool:
    return quota_cooldown_matches_model_use_case(key, model, ports=cooldown_read_ports())


def provider_on_cooldown(source: str, state: dict[str, Any]) -> tuple[bool, str | None]:
    return provider_on_cooldown_use_case(source, state, ports=cooldown_read_ports())


def model_on_cooldown(model: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str | None]:
    return model_on_cooldown_use_case(model, state, ports=cooldown_read_ports())


def model_quarantine(model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    return model_quarantine_use_case(model, state)


def model_is_quarantined(model: dict[str, Any], state: dict[str, Any]) -> bool:
    return model_is_quarantined_use_case(model, state)


def quarantine_ports() -> QuarantinePorts:
    return QuarantinePorts(
        safe_detail=safe_detail,
        now_iso=now_iso,
        cooldown_key=cooldown_key,
    )


def quarantine_row(model: dict[str, Any], reason: str, note: str | None, source: str, manual: bool) -> dict[str, Any]:
    return quarantine_row_use_case(model, reason, note, source, manual, ports=quarantine_ports())


def set_quarantine(model: dict[str, Any], reason: str, note: str | None = None, source: str = "manual", manual: bool = True) -> dict[str, Any]:
    row = quarantine_row(model, reason, note, source, manual)
    key = cooldown_key(model)

    def mutate(state: dict[str, Any]) -> None:
        quarantine = state.setdefault("quarantine", {})
        if not isinstance(quarantine, dict):
            quarantine = {}
            state["quarantine"] = quarantine
        quarantine[key] = row

    _STATE_STORE.update(mutate, reason=f"set_quarantine:{reason}")
    return row


def set_quarantine_in_state(state: dict[str, Any], model: dict[str, Any], reason: str, detail: str | None, source: str, fallback_note: str | None = None) -> None:
    set_quarantine_in_state_use_case(
        state,
        model,
        reason,
        detail,
        source,
        fallback_note,
        ports=quarantine_ports(),
    )


def set_billing_quarantine_in_state(state: dict[str, Any], model: dict[str, Any], detail: str | None, source: str, fallback_note: str | None = None) -> None:
    set_billing_quarantine_in_state_use_case(
        state,
        model,
        detail,
        source,
        fallback_note,
        ports=quarantine_ports(),
    )


def set_no_free_quota_quarantine_in_state(state: dict[str, Any], model: dict[str, Any], detail: str | None, source: str, fallback_note: str | None = None) -> None:
    set_no_free_quota_quarantine_in_state_use_case(
        state,
        model,
        detail,
        source,
        fallback_note,
        ports=quarantine_ports(),
    )


def set_model_not_found_quarantine_in_state(state: dict[str, Any], model: dict[str, Any], detail: str | None, source: str, fallback_note: str | None = None) -> None:
    set_model_not_found_quarantine_in_state_use_case(
        state,
        model,
        detail,
        source,
        fallback_note,
        ports=quarantine_ports(),
    )


def clear_quarantine(model: dict[str, Any]) -> bool:
    key = cooldown_key(model)
    existed = False

    def mutate(state: dict[str, Any]) -> None:
        nonlocal existed
        quarantine = state.setdefault("quarantine", {})
        if not isinstance(quarantine, dict):
            quarantine = {}
            state["quarantine"] = quarantine
        existed = key in quarantine
        quarantine.pop(key, None)

    _STATE_STORE.update(mutate, reason="clear_quarantine")
    return existed


def cooldown_stats_ports() -> CooldownStatsPorts:
    return CooldownStatsPorts(
        safe_int=safe_int,
        safe_detail=safe_detail,
        now_iso=now_iso,
        cooldown_key=cooldown_key,
        canonical_virtual_model_id=canonical_virtual_model_id,
    )


def update_failure_stats(state: dict[str, Any], model: dict[str, Any], reason: str) -> None:
    update_failure_stats_use_case(state, model, reason, ports=cooldown_stats_ports())


def update_success_stats(state: dict[str, Any], model: dict[str, Any], latency_seconds: float) -> None:
    update_success_stats_use_case(state, model, latency_seconds, ports=cooldown_stats_ports())


def record_model_error_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    detail: str | None = None,
    status: int | str | None = None,
    request_id: str | None = None,
    profile_id: str | None = None,
) -> None:
    record_model_error_in_state_use_case(
        state,
        model,
        reason,
        detail,
        status,
        request_id,
        profile_id,
        ports=cooldown_stats_ports(),
    )


def clear_model_error_in_state(state: dict[str, Any], model: dict[str, Any]) -> None:
    clear_model_error_in_state_use_case(state, model, ports=cooldown_stats_ports())


def record_provider_error_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    detail: str | None = None,
    status: int | str | None = None,
    request_id: str | None = None,
) -> None:
    record_provider_error_in_state_use_case(
        state,
        model,
        reason,
        detail,
        status,
        request_id,
        ports=cooldown_stats_ports(),
    )


def clear_provider_error_in_state(state: dict[str, Any], source: str) -> None:
    clear_provider_error_in_state_use_case(state, source)


def safe_attempt_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model": safe_detail(raw.get("model")),
        "upstream": safe_detail(raw.get("upstream")),
        "source": safe_detail(raw.get("source")),
        "status": raw.get("status"),
        "reason": safe_detail(raw.get("reason")),
        "latency_seconds": raw.get("latency_seconds"),
    }
    if raw.get("error_type"):
        row["error_type"] = safe_detail(raw.get("error_type"))
    if raw.get("stream_started") is not None:
        row["stream_started"] = bool(raw.get("stream_started"))
    return row


def safe_attempt_rows(attempts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in attempts or []:
        if isinstance(raw, dict):
            rows.append(safe_attempt_row(raw))
    return rows[:8]


def _increment_count(counts: dict[str, int], key: Any) -> None:
    name = safe_state_key(key)
    counts[name] = counts.get(name, 0) + 1


def prepare_compression_route_body(
    body: dict[str, Any],
    config: dict[str, Any],
    *,
    entitled: bool | None = None,
) -> CompressionRoutePlan:
    # Compression is a closed Pro engine. When the license is inactive/expired, fail closed by
    # behaving exactly as a core-only install does: no compression metadata or error counters.
    is_entitled = license_ops.is_entitled() if entitled is None else entitled
    if not is_entitled:
        return CompressionRoutePlan(body=body, metadata=None)
    return prepare_chat_compression_route_body(
        body,
        config,
        store_path=COMPRESSION_STORE_PATH,
        compress=compress_block,
        write_original=put_original,
    )


def chat_failure_response_headers(
    request_id: str,
    requested_model: str,
    attempt_count: int,
    compression: dict[str, Any] | None,
) -> dict[str, str]:
    return ficelle_response_headers(
        request_id,
        requested_model,
        attempt_count=attempt_count,
        compression=compression,
    )


def chat_success_response_headers(
    request_id: str,
    requested_model: str,
    selected_model: dict[str, Any],
    attempt_count: int,
    compression: dict[str, Any] | None,
) -> dict[str, str]:
    return ficelle_response_headers(
        request_id,
        requested_model,
        selected_model,
        attempt_count,
        compression=compression,
    )


def compression_retrieve_payload(hash_value: Any, query: Any = None) -> dict[str, Any]:
    if not isinstance(hash_value, str) or not hash_value.strip():
        raise ValueError("hash is required")
    if query is not None and not isinstance(query, str):
        raise ValueError("query must be a string")
    query_text = query.strip() if isinstance(query, str) and query.strip() else None
    entry = get_compression_original(
        hash_value,
        query=query_text,
        store_path=COMPRESSION_STORE_PATH,
    )
    if entry is None:
        record_compression_retrieval("missing")
        raise FileNotFoundError("compressed original not found or expired")
    record_compression_retrieval("success")
    return {
        "hash": entry.hash,
        "strategy": entry.strategy,
        "original_chars": entry.original_chars,
        "compressed_chars": entry.compressed_chars,
        "content": entry.content,
    }


def record_compression_retrieval(status: str) -> None:
    try:
        def mutate(state: dict[str, Any]) -> None:
            retrievals = state.setdefault("compression_retrievals", {})
            if not isinstance(retrievals, dict):
                retrievals = {}
                state["compression_retrievals"] = retrievals
            now = now_iso()
            if status == "success":
                retrievals["success_count"] = safe_int(retrievals.get("success_count"), 0) + 1
                retrievals["last_success_at"] = now
            else:
                retrievals["missing_count"] = safe_int(retrievals.get("missing_count"), 0) + 1
                retrievals["last_missing_at"] = now

        _STATE_STORE.update(mutate, reason=f"record_compression_retrieval:{status}")
    except Exception:
        return


def update_compression_error_stats(state: dict[str, Any], requested_model: str, compression: dict[str, Any] | None) -> None:
    errors = state.setdefault("compression_errors", {})
    if not isinstance(errors, dict):
        errors = {}
        state["compression_errors"] = errors
    profile_id = safe_state_key(requested_model)
    status = str((compression or {}).get("status") or "")
    if status == "error_original_forwarded":
        record = errors.setdefault(profile_id, {})
        if not isinstance(record, dict):
            record = {}
            errors[profile_id] = record
        record["consecutive_errors"] = safe_int(record.get("consecutive_errors"), 0) + 1
        record["last_error_at"] = now_iso()
        record["last_status"] = status
        if compression and compression.get("error_type"):
            record["last_error_type"] = safe_detail(compression.get("error_type"))
        return
    errors.pop(profile_id, None)


def record_last_route(
    requested_model: str,
    status: str,
    reason: str,
    request_id: str,
    candidate_count: int,
    attempt_count: int,
    latency_seconds: float,
    model: dict[str, Any] | None = None,
    attempts: list[dict[str, Any]] | None = None,
    competence: str | None = None,
    compression: dict[str, Any] | None = None,
) -> None:
    def mutate(state: dict[str, Any]) -> None:
        last_routes = state.setdefault("last_routes", {})
        row = {
            "status": status,
            "reason": reason,
            "request_id": request_id,
            "candidate_count": candidate_count,
            "attempt_count": attempt_count,
            "latency_seconds": round(latency_seconds, 4),
            "attempts": safe_attempt_rows(attempts),
            "seen_at": now_iso(),
        }
        if model is not None:
            row.update({
                "model_id": model.get("id"),
                "upstream_id": model.get("upstream_id"),
                "source": model.get("source"),
            })
        if competence:
            row["competence"] = competence
        if compression:
            row["compression"] = safe_compression_metadata(compression)
        compression_metadata = row.get("compression") if isinstance(row.get("compression"), dict) else None
        update_compression_error_stats(state, str(requested_model), compression_metadata)
        last_routes[str(requested_model)] = row

    _STATE_STORE.update(mutate, reason="record_last_route")


def update_provider_failure_stats(state: dict[str, Any], source: str, reason: str) -> None:
    update_provider_failure_stats_use_case(state, source, reason, ports=cooldown_stats_ports())


def update_provider_success_stats(state: dict[str, Any], source: str, latency_seconds: float) -> None:
    update_provider_success_stats_use_case(state, source, latency_seconds, ports=cooldown_stats_ports())


def cooldown_mutation_ports() -> CooldownMutationPorts:
    return CooldownMutationPorts(
        safe_int=safe_int,
        safe_detail=safe_detail,
        now_seconds=time.time,
        now_iso=now_iso,
        quota_cooldown_key=quota_cooldown_key,
        quota_cooldown_scope=quota_cooldown_scope,
        quota_cooldown_scope_from_key=quota_cooldown_scope_from_key,
        update_provider_failure_stats=update_provider_failure_stats,
        default_quota_probe_backoff_seconds=tuple(DEFAULT_QUOTA_PROBE_BACKOFF_SECONDS),
    )


def set_provider_cooldown_in_state(state: dict[str, Any], source: str, reason: str, config: dict[str, Any], detail: str | None = None) -> None:
    set_provider_cooldown_in_state_use_case(state, source, reason, config, detail, ports=cooldown_mutation_ports())


def quota_probe_backoff_seconds(config: dict[str, Any], consecutive_failures: int) -> int:
    return quota_probe_backoff_seconds_use_case(config, consecutive_failures, ports=cooldown_mutation_ports())


def set_quota_cooldown_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    config: dict[str, Any],
    detail: str | None = None,
    *,
    probe_failed: bool = False,
    cooldown_key_override: str | None = None,
) -> None:
    set_quota_cooldown_in_state_use_case(
        state,
        model,
        config,
        detail,
        ports=cooldown_mutation_ports(),
        probe_failed=probe_failed,
        cooldown_key_override=cooldown_key_override,
    )


def clear_model_cooldown(model: dict[str, Any]) -> bool:
    key = cooldown_key(model)
    existed = False

    def mutate(state: dict[str, Any]) -> None:
        nonlocal existed
        existed = clear_model_cooldown_in_state_use_case(state, key)

    _STATE_STORE.update(mutate, reason="clear_model_cooldown")
    return existed


def clear_provider_cooldown(source: str) -> bool:
    existed = False

    def mutate(state: dict[str, Any]) -> None:
        nonlocal existed
        existed = clear_provider_cooldown_in_state_use_case(state, source)

    _STATE_STORE.update(mutate, reason="clear_provider_cooldown")
    return existed


def clear_all_cooldowns(scope: str = "all") -> int:
    changed = 0

    def mutate(state: dict[str, Any]) -> None:
        nonlocal changed
        changed = clear_all_cooldowns_in_state_use_case(state, scope)

    _STATE_STORE.update(mutate, reason=f"clear_all_cooldowns:{scope}")
    return changed


def cooldown_write_ports() -> CooldownWritePorts:
    return CooldownWritePorts(
        update_state=_STATE_STORE.update,
        update_failure_stats=update_failure_stats,
        record_model_error_in_state=record_model_error_in_state,
        cooldown_policy_for_reason=lambda reason, source: cooldown_policy_for_reason(reason, source=source),
        record_provider_error_in_state=record_provider_error_in_state,
        set_quota_cooldown_in_state=set_quota_cooldown_in_state,
        set_provider_cooldown_in_state=set_provider_cooldown_in_state,
        set_billing_quarantine_in_state=set_billing_quarantine_in_state,
        set_no_free_quota_quarantine_in_state=set_no_free_quota_quarantine_in_state,
        set_model_not_found_quarantine_in_state=set_model_not_found_quarantine_in_state,
        cooldown_key=cooldown_key,
        now_seconds=time.time,
        now_iso=now_iso,
        safe_detail=safe_detail,
    )


def set_cooldown(
    model: dict[str, Any],
    reason: str,
    config: dict[str, Any],
    detail: str | None = None,
    status: int | str | None = None,
    request_id: str | None = None,
    profile_id: str | None = None,
) -> None:
    set_cooldown_use_case(
        model,
        reason,
        config,
        ports=cooldown_write_ports(),
        detail=detail,
        status=status,
        request_id=request_id,
        profile_id=profile_id,
    )


def cooldown_success_ports() -> CooldownSuccessPorts:
    return CooldownSuccessPorts(
        safe_float=safe_float,
        now_seconds=time.time,
        now_iso=now_iso,
        cooldown_key=cooldown_key,
        quota_cooldown_matches_model=quota_cooldown_matches_model,
        clear_provider_error_in_state=clear_provider_error_in_state,
        clear_model_error_in_state=clear_model_error_in_state,
        update_success_stats=update_success_stats,
        update_provider_success_stats=update_provider_success_stats,
    )


def record_success(model: dict[str, Any], latency_seconds: float) -> None:
    def mutate(state: dict[str, Any]) -> None:
        record_success_in_state_use_case(state, model, latency_seconds, ports=cooldown_success_ports())

    _STATE_STORE.update(mutate, reason="record_success")


def model_scoring_ports() -> ModelScoringPorts:
    return ModelScoringPorts(cooldown_key=cooldown_key, safe_int=safe_int)


def success_rate_for_model(model: dict[str, Any], state: dict[str, Any]) -> float:
    return success_rate_for_model_use_case(model, state, ports=model_scoring_ports())


def latency_score_for_model(model: dict[str, Any], state: dict[str, Any]) -> float:
    return latency_score_for_model_use_case(model, state, ports=model_scoring_ports())


def model_evidence_ports() -> ModelEvidencePorts:
    return ModelEvidencePorts(
        cooldown_key=cooldown_key,
        canonical_virtual_model_id=canonical_virtual_model_id,
        benchmark_result_matches_current_test=benchmark_result_matches_current_test,
        benchmark_result_test_type_matches=benchmark_result_test_type_matches,
        benchmark_result_is_aged=benchmark_result_is_aged,
        runtime_evidence_timestamp_value=runtime_evidence_timestamp_value,
        parse_iso_timestamp=parse_iso_timestamp,
        stale_score_decay_factor=stale_score_decay_factor,
    )


def profile_benchmark_adjustment(requested_model: str, model: dict[str, Any], state: dict[str, Any]) -> float:
    """Profile-specific quality adjustment from real canary/benchmark outcomes.

    Catalog capabilities say what a provider claims. Benchmark rows say whether this
    concrete upstream actually works for the destination profile Ficelle is routing.
    """
    return benchmark_adjustment_parts(requested_model, model, state)["score_adjustment"]


def runtime_profile_row(section: str, profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return runtime_profile_row_use_case(section, profile_id, model, state, ports=model_evidence_ports())


def raw_benchmark_row(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return raw_benchmark_row_use_case(profile_id, model, state, ports=model_evidence_ports())


def raw_verified_capability_row(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return raw_verified_capability_row_use_case(profile_id, model, state, ports=model_evidence_ports())


def score_row_status(profile_id: str, row: dict[str, Any]) -> str:
    return score_row_status_use_case(profile_id, row, ports=model_evidence_ports())


def score_row_reason(profile_id: str, row: dict[str, Any]) -> str | None:
    return score_row_reason_use_case(profile_id, row, ports=model_evidence_ports())


def score_evidence_for_model(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return score_evidence_for_model_use_case(profile_id, model, state, ports=model_evidence_ports())


def benchmark_adjustment_parts(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return benchmark_adjustment_parts_use_case(profile_id, model, state, ports=model_evidence_ports())


def model_auto_score(requested_model: str, model: dict[str, Any], state: dict[str, Any]) -> float:
    return model_score_explanation(requested_model, model, state)["score_total_raw"]


def model_score_explanation(requested_model: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return model_score_explanation_use_case(
        requested_model,
        model,
        state,
        ports=ModelScoreExplanationPorts(
            scoring=model_scoring_ports(),
            evidence=model_evidence_ports(),
            model_has_any=model_has_any,
        ),
    )


def model_matches_profile_requirements(model: dict[str, Any], profile: dict[str, Any]) -> bool:
    return policy_model_matches_profile_requirements(
        model,
        profile,
        default_min_context=DEFAULT_PROFILE_REQUIREMENTS["min_context"],
        free_access_eligible=lambda row: bool(normalized_free_access(row).get("eligible")),
    )


def sort_available_for_virtual_model(requested_model: str, available: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    return sort_available_for_virtual_model_policy(
        requested_model,
        available,
        state,
        model_auto_score=model_auto_score,
        safe_int=safe_int,
        safe_float=safe_float,
        cooldown_key=cooldown_key,
    )


def manual_order_candidates(
    requested_model: str,
    eligible: list[dict[str, Any]],
    state: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    return manual_order_candidates_policy(
        requested_model,
        eligible,
        state,
        profile,
        sort_available_for_virtual_model=sort_available_for_virtual_model,
    )


def quota_probe_body(model: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return quota_probe_body_use_case(model, config)


def quota_probe_model_ports() -> QuotaProbeModelPorts:
    return QuotaProbeModelPorts(
        safe_float=safe_float,
        now_seconds=time.time,
        model_is_quarantined=model_is_quarantined,
        provider_on_cooldown=provider_on_cooldown,
        cooldown_key=cooldown_key,
        normalized_free_access=normalized_free_access,
        quota_cooldown_matches_model=quota_cooldown_matches_model,
    )


def find_quota_probe_model(catalog: dict[str, Any], key: str, state: dict[str, Any]) -> dict[str, Any] | None:
    return find_quota_probe_model_use_case(catalog, key, state, ports=quota_probe_model_ports())


def quota_probe_state_ports() -> QuotaProbeStatePorts:
    return QuotaProbeStatePorts(
        safe_detail=safe_detail,
        safe_float=safe_float,
        now_seconds=time.time,
        now_iso=now_iso,
        update_state=_STATE_STORE.update,
    )


def record_quota_probe_result(state: dict[str, Any], key: str, model: dict[str, Any], status: str, reason: str, detail: str | None = None) -> None:
    record_quota_probe_result_use_case(state, key, model, status, reason, detail, ports=quota_probe_state_ports())


def clear_quota_cooldown_in_state(state: dict[str, Any], key: str, model: dict[str, Any]) -> None:
    clear_quota_cooldown_in_state_use_case(state, key, model, ports=cooldown_success_ports())


def clear_quota_model_errors_for_source_in_state(state: dict[str, Any], source: str) -> None:
    clear_quota_model_errors_for_source_in_state_use_case(state, source)


def source_has_active_quota_cooldown(state: dict[str, Any], source: str) -> bool:
    return source_has_active_quota_cooldown_use_case(state, source, ports=cooldown_success_ports())


def fresh_runtime_state() -> dict[str, Any]:
    return load_runtime_state()


def probe_quota_cooldown(key: str, model: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:

    def record_failed_probe(
        reason: str,
        cooldown_detail: str,
        result_detail: str,
        apply_failure_guard: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        def mutate(state: dict[str, Any]) -> None:
            if apply_failure_guard is not None:
                apply_failure_guard(state)
            set_quota_cooldown_in_state(state, model, config, cooldown_detail, probe_failed=True, cooldown_key_override=key)
            record_quota_probe_result(state, key, model, "fail", reason, result_detail)

        _STATE_STORE.update(mutate, reason=f"probe_quota_cooldown:{reason}")
        return {"status": "fail", "reason": reason, "key": key, "model_id": model.get("id")}

    started = time.time()
    try:
        timeout = max(1.0, safe_float(config.get("quota_probe_timeout_seconds"), 10.0))
        response = invoke_model(model, quota_probe_body(model, config), config, timeout_seconds=timeout)
        latency = time.time() - started
    except requests.exceptions.Timeout as exc:
        return record_failed_probe("timeout", f"quota probe timeout: {type(exc).__name__}", f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        return record_failed_probe("unavailable", f"quota probe error: {type(exc).__name__}: {exc}", f"{type(exc).__name__}: {exc}")

    if 200 <= response.status_code < 300:
        try:
            payload = response.json()
        except ValueError as exc:
            return record_failed_probe("unavailable", f"quota probe invalid JSON: {exc}", f"{type(exc).__name__}: {exc}")
        if not has_deliverable_message(payload):
            return record_failed_probe("unavailable", "quota probe returned no deliverable assistant message", "empty assistant response")
        def mutate(state: dict[str, Any]) -> None:
            clear_quota_cooldown_in_state(state, key, model)
            record_quota_probe_result(state, key, model, "pass", "ok", "quota probe succeeded")
            update_success_stats(state, model, latency)
            source = str(model.get("source") or "").strip()
            if source:
                update_provider_success_stats(state, source, latency)

        _STATE_STORE.update(mutate, reason="probe_quota_cooldown:pass")
        return {"status": "pass", "reason": "ok", "key": key, "model_id": model.get("id")}

    text = response.text[:1000]
    reason = classify_failure(response.status_code, text, model)
    detail = f"quota probe HTTP {response.status_code}: {text[:250]}"
    apply_failure_guard: Callable[[dict[str, Any]], None] | None = None
    if reason == "no_free_quota":
        def apply_no_free_quota_guard(state: dict[str, Any]) -> None:
            set_no_free_quota_quarantine_in_state(state, model, detail, "quota_probe_guard")

        apply_failure_guard = apply_no_free_quota_guard
    elif reason == "model_not_found":
        def apply_model_not_found_guard(state: dict[str, Any]) -> None:
            set_model_not_found_quarantine_in_state(state, model, detail, "quota_probe_guard")

        apply_failure_guard = apply_model_not_found_guard
    elif reason == "billing_or_paid":
        def apply_billing_guard(state: dict[str, Any]) -> None:
            set_billing_quarantine_in_state(state, model, detail, "quota_probe_guard")

        apply_failure_guard = apply_billing_guard
    elif reason == "auth_or_credit":
        def apply_auth_guard(state: dict[str, Any]) -> None:
            record_provider_error_in_state(state, model, reason, detail, response.status_code)
            set_provider_cooldown_in_state(state, str(model.get("source") or ""), reason, config, detail)

        apply_failure_guard = apply_auth_guard
    return record_failed_probe(reason, detail, detail, apply_failure_guard)


def reserve_quota_probe_in_state(state: dict[str, Any], key: str, row: dict[str, Any], config: dict[str, Any]) -> bool:
    return reserve_quota_probe_in_state_use_case(state, key, row, config, ports=quota_probe_state_ports())


def quota_probe_runner() -> QuotaProbeRunner:
    return QuotaProbeRunner(
        QuotaProbePorts(
            fresh_runtime_state=fresh_runtime_state,
            safe_float=safe_float,
            safe_string_list=safe_string_list,
            normalized_virtual_profiles=normalized_virtual_profiles,
            canonical_virtual_model_id=canonical_virtual_model_id,
            model_matches_profile_requirements=model_matches_profile_requirements,
            quota_cooldown_matches_model=quota_cooldown_matches_model,
            find_quota_probe_model=find_quota_probe_model,
            reserve_quota_probe=reserve_quota_probe_in_state,
            probe_quota_cooldown=probe_quota_cooldown,
            virtual_models=set(VIRTUAL_MODELS),
            now_seconds=time.time,
        )
    )


def run_due_quota_probes(
    config: dict[str, Any],
    catalog: dict[str, Any],
    *,
    force: bool = False,
    source: str | None = None,
    keys: set[str] | None = None,
) -> dict[str, Any]:
    return quota_probe_runner().run_due_quota_probes(
        config,
        catalog,
        force=force,
        source=source,
        keys=keys,
    )


def due_quota_probe_keys_for_request(requested_model: str, models: list[dict[str, Any]], config: dict[str, Any], state: dict[str, Any]) -> list[str]:
    return quota_probe_runner().due_quota_probe_keys_for_request(requested_model, models, config, state)


def known_provider_sources(config: dict[str, Any], catalog: dict[str, Any]) -> set[str]:
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    sources = {str(source) for source in providers}
    for model in catalog.get("models", []) or []:
        if isinstance(model, dict) and model.get("source"):
            sources.add(str(model.get("source")))
    return sources


def model_capability_is_default_claimed(profile_id: str, model: dict[str, Any]) -> bool:
    """True when the profile's discriminating capability was claimed only via provider
    defaults (a guess), not declared by the provider catalog."""
    capability = GATE_CAPABILITY_BY_PROFILE.get(canonical_virtual_model_id(profile_id))
    if not capability:
        return False
    return capability in safe_string_list(model.get("capabilities_from_defaults"))


def model_capability_provenance(model: dict[str, Any], capability: str | None) -> str:
    """Provenance of a claimed discriminating capability, for observability only.

    ``reference`` (a cross-provider prior confirms a value the row only claimed via provider
    defaults), ``default`` (provider-default guess, still unproven), or ``catalog`` (a real provider
    declaration). ``n/a`` when the profile has no single discriminating capability (generic and
    auto-multimodal profiles). This is data lineage, reported for ALL confidence tiers and
    independent of the Phase B setting — distinct from the routing label ``model_route_competence``,
    which only treats a ``reference`` claim as routable (``claimed_reference``) when it is
    ``high``-confidence AND the opt-in is on. So a row can read provenance ``reference`` while still
    being gated out (e.g. a ``medium`` oracle prior); the row's ``reference_confidence`` explains why.
    """
    if not capability:
        return "n/a"
    if capability in safe_string_list(model.get("capabilities_from_reference")):
        return "reference"
    if capability in safe_string_list(model.get("capabilities_from_defaults")):
        return "default"
    return "catalog"


def model_capability_reference_routable(profile_id: str, model: dict[str, Any]) -> bool:
    """Phase B (R6): may a default-claimed capability route on the reference prior alone?

    True only when the operator opt-in is ON and the model's discriminating capability is
    reference-confirmed at a trusted confidence tier (the hand-curated ``high`` seed; the broad
    ``medium`` oracle is excluded). Identity matching is the same conservative canonical-key/alias
    path, so no fuzzy match qualifies. Callers still apply the ``failed`` override and the
    verified-capability TTL, so this only widens an *unverified* default-claim, never overrides a
    benchmark verdict.
    """
    if not route_on_capability_reference_enabled():
        return False
    capability = GATE_CAPABILITY_BY_PROFILE.get(canonical_virtual_model_id(profile_id))
    if not capability or capability not in safe_string_list(model.get("capabilities_from_reference")):
        return False
    return str(model.get("reference_confidence") or "") in CAPABILITY_REFERENCE_ROUTE_CONFIDENCES


def route_eligible_for_profile(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> bool:
    """Route eligibility of one model for one specialized profile (R4 + Phase B R6).

    ``failed`` -> excluded (always, even under Phase B); ``verified`` -> eligible; an unverified
    capability is eligible when it is a real catalog declaration, or — only when the Phase B opt-in
    is on — a high-confidence reference-confirmed claim (model_capability_reference_routable). A
    capability claimed only via provider defaults and not reference-routable must still be
    benchmark-verified before routing, which is what stops sparse providers from flooding
    specialized pools with unproven models.
    """
    if model_failed_profile_evidence(profile_id, model, state):
        return False
    if verified_capability_row(profile_id, model, state).get("status") == "verified":
        return True
    if not model_capability_is_default_claimed(profile_id, model):
        return True
    return model_capability_reference_routable(profile_id, model)


def multimodal_route_eligible(model: dict[str, Any], state: dict[str, Any]) -> bool:
    """auto-multimodal gate (R7): a pass on any one claimed modality satisfies it.

    Competent when the model is ``verified`` on any modality it claims; excluded only when
    every claimed modality has failed evidence. Modalities are never default-claimed (the
    defaults inject text only), so an unverified-and-not-failed modality is trusted.
    """
    claimed = claimed_multimodal_profiles(model)
    if not claimed:
        return True
    if any(verified_capability_row(profile, model, state).get("status") == "verified" for profile in claimed):
        return True
    return not all(model_failed_profile_evidence(profile, model, state) for profile in claimed)


def claimed_multimodal_profiles(model: dict[str, Any]) -> list[str]:
    return [
        profile
        for modality, profile in MULTIMODAL_MODALITY_PROFILES.items()
        if model_has_any(model, "input_modalities", [modality])
    ]


def route_competent_candidates(
    profile_id: str,
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Narrow route candidates to those competent for a specialized profile.

    A specialized profile (its whole value is one discriminating capability) routes only to
    models that are ``verified`` for it or declare the capability in a real catalog;
    ``failed`` and default-claimed-unverified models are excluded (route_eligible_for_profile,
    R4). auto-multimodal uses the OR-over-modalities gate (R7). Generic profiles are
    unaffected and keep capability as a score signal only.

    The pool is never emptied: if the gate would exclude every candidate, the full ordered
    set is returned unchanged, because benchmark/verified-capability state must not empty
    routing pools (docs/components/router.md). Such a served model is then labelled
    ``claimed_default``/``claimed_catalog`` in the route-log ``competence`` field
    (model_route_competence), so the fallback is observable rather than silent.
    """
    return route_competence_gate_result(profile_id, candidates, state)[0]


def route_competence_gate_result(
    profile_id: str,
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Return routed candidates and whether an anti-empty competence fallback was used.

    The boolean is true only when a competence gate would have emptied a non-empty
    candidate pool, so the original pool was returned for availability.
    """
    canonical = canonical_virtual_model_id(profile_id)
    return policy_competence_gate_result(
        canonical,
        candidates,
        is_multimodal_profile=lambda profile: profile == "ficelle/auto-multimodal",
        is_specialized_profile=is_specialized_profile,
        multimodal_route_eligible=lambda model: multimodal_route_eligible(model, state),
        route_eligible_for_profile=lambda model: route_eligible_for_profile(canonical, model, state),
    )


def model_route_competence(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> str:
    """Competence label of a model for a profile, for route logs and admin status.

    ``failed``/``verified`` come from benchmark evidence; an unverified specialized capability is
    ``claimed_catalog`` (real provider declaration), ``claimed_reference`` (Phase B on and a
    high-confidence reference prior is opening the gate), or ``claimed_default`` (provider-default
    guess, still gated out); generic profiles are ``n/a``. A *served* model labelled ``claimed_*``
    means routing fell back to an unproven candidate because no proven one was available.
    """
    canonical = canonical_virtual_model_id(profile_id)
    return policy_competence_label(
        canonical,
        model,
        is_multimodal_profile=lambda profile: profile == "ficelle/auto-multimodal",
        is_specialized_profile=is_specialized_profile,
        multimodal_route_eligible=lambda candidate: multimodal_route_eligible(candidate, state),
        multimodal_claimed_profiles=claimed_multimodal_profiles,
        profile_verified=lambda profile, candidate: verified_capability_row(profile, candidate, state).get("status") == "verified",
        profile_failed=lambda profile, candidate: model_failed_profile_evidence(profile, candidate, state),
        capability_is_default_claimed=model_capability_is_default_claimed,
        capability_reference_routable=model_capability_reference_routable,
    )


def selection_policy() -> SelectionPolicy:
    return SelectionPolicy(
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=model_is_quarantined,
        normalized_virtual_profiles=normalized_virtual_profiles,
        canonical_virtual_model_id=canonical_virtual_model_id,
        model_matches_profile_requirements=model_matches_profile_requirements,
        manual_order_candidates=manual_order_candidates,
        sort_available_for_virtual_model=sort_available_for_virtual_model,
        route_competence_gate_result=route_competence_gate_result,
        virtual_models=set(VIRTUAL_MODELS),
    )


def model_selection_runner() -> ModelSelectionRunner:
    return ModelSelectionRunner(
        ModelSelectionPorts(
            load_config=load_runtime_config,
            fresh_runtime_state=fresh_runtime_state,
            due_quota_probe_keys_for_request=due_quota_probe_keys_for_request,
            run_due_quota_probes=run_due_quota_probes,
            safe_int=safe_int,
            apply_verified_capability_ttl=apply_verified_capability_ttl,
            apply_route_on_capability_reference=apply_route_on_capability_reference,
            selection_policy=selection_policy,
        )
    )


def select_models_result(
    requested_model: str,
    catalog: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    purpose: SelectionPurpose = "route",
) -> SelectionResult:
    runtime_config = config if config is not None else load_runtime_config()
    return model_selection_runner().select_models_result(
        requested_model,
        catalog,
        runtime_config,
        purpose=purpose,
    )


def select_models_result_from_state(
    requested_model: str,
    catalog: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    purpose: SelectionPurpose = "route",
) -> SelectionResult:
    """Select candidates from explicit catalog/config/state without runtime state reads or quota probes."""
    return model_selection_runner().select_models_result_from_state(
        requested_model,
        catalog,
        config,
        state,
        purpose=purpose,
    )


def select_models_result_from_typed_rows(
    requested_model: str,
    typed_rows: list[tuple[CatalogModel, dict[str, Any]]],
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    purpose: SelectionPurpose = "route",
) -> SelectionResult:
    return model_selection_runner().select_models_result_from_typed_rows(requested_model, typed_rows, config, state, purpose=purpose)


def select_models(
    requested_model: str,
    catalog: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    purpose: SelectionPurpose = "route",
) -> list[dict[str, Any]]:
    return select_models_result(requested_model, catalog, config, purpose=purpose).as_legacy_models()


def provider_credentials(source: str, config: dict[str, Any]) -> tuple[str | None, str | None, str]:
    return resolve_provider_credentials(
        source,
        config,
        lambda provider_source, provider_cfg: provider_access_result(
            provider_source,
            provider_cfg,
            require_base_url=True,
        ),
    )


def classify_failure(status_code: int, text: str, model: dict[str, Any] | None = None) -> str:
    access = normalized_free_access(model) if isinstance(model, dict) else None
    source = str(model.get("source") or "") if isinstance(model, dict) else ""
    markers = provider_catalog_adapter(source).failure_markers() if source else None
    return classify_failure_result(status_code, text, normalized_free_access=access, markers=markers)


def success_error_payload_failure(payload: Any, model: dict[str, Any] | None = None) -> tuple[str, str, int | None] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    error = payload["error"]
    message = safe_detail(error.get("message")) or "provider returned an error payload with HTTP 200"
    raw_code = error.get("code") or error.get("status") or error.get("status_code")
    status_code: int | None = None
    if isinstance(raw_code, int):
        status_code = raw_code
    elif isinstance(raw_code, str):
        try:
            status_code = int(raw_code)
        except ValueError:
            status_code = None
    reason = classify_failure(status_code or 200, json.dumps(error, ensure_ascii=False)[:1000], model)
    return reason, message, status_code


def request_timeout_seconds_for_profile(requested_model: str, config: dict[str, Any]) -> float:
    default_timeout = float(config.get("request_timeout_seconds") or 120)
    per_profile = config.get("request_timeout_seconds_by_profile")
    if isinstance(per_profile, dict):
        raw_value = per_profile.get(canonical_virtual_model_id(requested_model))
        if raw_value is not None:
            try:
                return max(1.0, float(raw_value))
            except Exception:
                return default_timeout
    return default_timeout


def max_attempts_for_request(requested_model: str, config: dict[str, Any], candidate_count: int) -> int:
    if canonical_virtual_model_id(requested_model) not in configured_virtual_model_ids(config):
        return candidate_count
    configured = safe_int(config.get("max_attempts_per_request"), candidate_count)
    if configured <= 0:
        return candidate_count
    return max(1, min(candidate_count, configured))


def benchmark_media_ports() -> BenchmarkMediaPorts:
    return BenchmarkMediaPorts(
        safe_int=safe_int,
        env_get=os.getenv,
        default_media_path=default_benchmark_media_path,
        load_json=load_json,
        vision_fixtures_path=BENCHMARK_VISION_FIXTURES_PATH,
        now_seconds=time.time,
    )


def benchmark_media_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    return benchmark_media_settings_use_case(config)


def configured_benchmark_media_path(config: dict[str, Any] | None, kind: str) -> Path | None:
    return configured_benchmark_media_path_use_case(config, kind, ports=benchmark_media_ports())


def max_benchmark_media_bytes(config: dict[str, Any] | None) -> int:
    return max_benchmark_media_bytes_use_case(config, ports=benchmark_media_ports())


def benchmark_file_data_url(config: dict[str, Any] | None, kind: str, default_mime: str) -> str:
    return benchmark_file_data_url_use_case(config, kind, default_mime, ports=benchmark_media_ports())


def benchmark_audio_part(config: dict[str, Any] | None) -> dict[str, Any]:
    return benchmark_audio_part_use_case(config, ports=benchmark_media_ports())


def default_benchmark_media_path(kind: str) -> Path:
    suffix = {"audio": "audio.mp3", "video": "video.mp4"}.get(kind, f"{kind}.bin")
    return PACKAGE_DIR / "data" / "benchmark" / suffix


def benchmark_media_answer(config: dict[str, Any] | None, kind: str, default: str) -> str:
    return benchmark_media_answer_use_case(config, kind, default)


def vision_benchmark_fixtures() -> dict[str, str]:
    return vision_benchmark_fixtures_use_case(ports=benchmark_media_ports())


def pick_vision_fixture(index: int | None = None) -> tuple[str, str]:
    return pick_vision_fixture_use_case(index, ports=benchmark_media_ports())


def invoke_model(
    model: dict[str, Any],
    body: dict[str, Any],
    config: dict[str, Any],
    *,
    timeout_seconds: float | int | None = None,
    extra_headers: dict[str, str] | None = None,
) -> requests.Response:
    source = str(model.get("source"))
    key, base_url, reason = provider_credentials(source, config)
    if not base_url or (not key and reason != "keyless_local"):
        raise RuntimeError(f"credentials unavailable: {reason}")
    requested_model = normalize_chat_completion_request(body).requested_model
    payload = dict(body)
    payload.pop("_ficelle_timeout_seconds", None)
    payload.pop("_ficelle_extra_headers", None)
    payload["model"] = model["upstream_id"]
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json" if not payload.get("stream") else "text/event-stream",
        **provider_invocation_headers(source),
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if isinstance(extra_headers, dict):
        headers.update({str(key): str(value) for key, value in extra_headers.items()})
    try:
        timeout = float(timeout_seconds) if timeout_seconds is not None else request_timeout_seconds_for_profile(requested_model, config)
    except Exception:
        timeout = request_timeout_seconds_for_profile(requested_model, config)
    return requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
        stream=bool(payload.get("stream")),
    )


def model_family(model: dict[str, Any]) -> str:
    upstream = str(model.get("upstream_id") or model.get("id") or "")
    return upstream.split("/", 1)[0].split(":", 1)[0] or "unknown"


def provider_isolation_policy(model: dict[str, Any]) -> str:
    source = str(model.get("source") or "")
    if source in {"openrouter", "nous"}:
        return "api_only"
    return "no_hard_isolation"


def model_is_strict_zero(model: dict[str, Any]) -> bool:
    pricing_ok, _, _ = strict_zero_pricing(model.get("pricing"))
    return pricing_ok


def model_failed_profile_evidence(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> bool:
    return bool(failed_profile_evidence_row(profile_id, model, state))


def provider_allowed_for_fusion(model: dict[str, Any], config: dict[str, Any]) -> bool:
    access = normalized_free_access(model)
    mode = str(access.get("mode") or "")
    if access.get("eligible") is not True or mode not in {"catalog_free", "quota_free"}:
        return False
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    provider = providers.get(str(model.get("source") or "")) if isinstance(providers, dict) else None
    provider_config = provider if isinstance(provider, dict) else {}
    source_type = str(provider_config.get("source_type") or model.get("source_type") or "")
    provider_class = str(provider_config.get("provider_class") or model.get("provider_class") or "")
    free_mode = str(provider_config.get("free_mode") or model.get("free_mode") or "")
    quota_markers = {"free_quota_candidate", "free_quota", "quota_free", "trial", "quota_trial"}
    provider_has_quota_marker = any(value in quota_markers for value in (source_type, provider_class, free_mode))
    if provider_has_quota_marker and mode != "quota_free":
        return False
    return True


def fusion_metadata_ports() -> FusionMetadataPorts:
    return FusionMetadataPorts(
        safe_detail=safe_detail,
        model_family=model_family,
        model_is_strict_zero=model_is_strict_zero,
        normalized_free_access=normalized_free_access,
        provider_isolation_policy=provider_isolation_policy,
        safe_fusion_reason_code=safe_fusion_reason_code,
        safe_int=safe_int,
        fusion_call_multiplier=fusion_call_multiplier,
        now_iso=now_iso,
        now_seconds=time.time,
        fusion_model_id=FUSION_MODEL_ID,
        default_panel_size=DEFAULT_FUSION_CONFIG["panel_size"],
    )


def fusion_model_detail(model: dict[str, Any]) -> dict[str, Any]:
    return fusion_model_detail_use_case(model, fusion_metadata_ports())


def fusion_reason_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return fusion_reason_counts_use_case(results)


def fusion_panel_result_detail(result: dict[str, Any]) -> dict[str, Any]:
    return fusion_panel_result_detail_use_case(result, fusion_metadata_ports())


def fusion_eligibility_ports() -> FusionEligibilityPorts:
    return FusionEligibilityPorts(
        select_models=select_models,
        load_state=load_runtime_state,
        provider_allowed_for_fusion=provider_allowed_for_fusion,
        model_failed_profile_evidence=model_failed_profile_evidence,
        fusion_model_id=FUSION_MODEL_ID,
    )


def fusion_eligible_models(profile_id: str, catalog: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    return fusion_eligible_models_use_case(profile_id, catalog, config, ports=fusion_eligibility_ports())


def fusion_panel_policy_ports() -> FusionPanelPolicyPorts:
    return FusionPanelPolicyPorts(
        safe_int=safe_int,
        model_family=model_family,
        fusion_model_id=FUSION_MODEL_ID,
        default_panel_size=DEFAULT_FUSION_CONFIG["panel_size"],
        default_diversity_policy=DEFAULT_FUSION_CONFIG["diversity_policy"],
    )


def fusion_deadline_seconds(deadline: float) -> float:
    return fusion_deadline_seconds_use_case(deadline, time.time)


def fusion_timeout_config(fusion: dict[str, Any], deadline: float) -> dict[str, Any]:
    return fusion_timeout_config_use_case(
        fusion,
        deadline,
        default_per_call_timeout_seconds=DEFAULT_FUSION_CONFIG["per_call_timeout_seconds"],
        now_seconds=time.time,
    )


def diverse_fusion_panel(candidates: list[dict[str, Any]], fusion: dict[str, Any]) -> list[dict[str, Any]]:
    return diverse_fusion_panel_use_case(candidates, fusion, ports=fusion_panel_policy_ports())


def fusion_internal_body(body: dict[str, Any], profile_id: str, fusion: dict[str, Any], *, allow_response_format: bool = False) -> dict[str, Any]:
    return fusion_internal_body_use_case(body, profile_id, allow_response_format=allow_response_format)


def build_fusion_runner() -> FusionRunner:
    return FusionRunner(
        invoke_model=invoke_model,
        classify_failure=classify_failure,
        extract_message_text=extract_message_text,
        success_error_payload_failure=success_error_payload_failure,
        fusion_internal_body=fusion_internal_body,
        fusion_timeout_config=fusion_timeout_config,
        fusion_deadline_seconds=fusion_deadline_seconds,
        fusion_ranking_messages=fusion_ranking_messages,
        fusion_judge_messages=fusion_judge_messages,
        fusion_prompt_from_outputs=fusion_prompt_from_outputs,
        fusion_eligible_models=fusion_eligible_models,
        extract_fusion_judge_json=extract_fusion_judge_json,
        normalize_fusion_judge_json=normalize_fusion_judge_json,
        parse_fusion_ranking=parse_fusion_ranking,
        aggregate_fusion_rankings=aggregate_fusion_rankings,
        record_success=record_success,
        set_cooldown=set_cooldown,
        is_timeout_exception=lambda exc: isinstance(exc, requests.exceptions.Timeout),
        fusion_model_id=FUSION_MODEL_ID,
        now=time.time,
    )


def build_fusion_request_preflight() -> FusionRequestPreflight:
    return FusionRequestPreflight(
        FusionRequestPreflightPorts(
            normalize_fusion_config=normalize_fusion_config,
            configured_virtual_model_ids=configured_virtual_model_ids,
            ficelle_response_headers=ficelle_response_headers,
            fusion_error_payload=fusion_error_payload,
            fusion_run_metadata=fusion_run_metadata,
            persist_fusion_run=persist_fusion_run,
            fusion_response_headers=fusion_response_headers,
            uuid_hex=lambda: uuid.uuid4().hex,
        ),
        fusion_model_id=FUSION_MODEL_ID,
        default_timeout_seconds=float(DEFAULT_FUSION_CONFIG["fusion_timeout_seconds"]),
    )


def build_fusion_panel_stage() -> FusionPanelStage:
    return FusionPanelStage(
        fusion_eligible_models=fusion_eligible_models,
        diverse_fusion_panel=diverse_fusion_panel,
        run_fusion_panel_calls=run_fusion_panel_calls,
        record_success=record_success,
        set_cooldown=set_cooldown,
        fusion_model_id=FUSION_MODEL_ID,
        now=time.time,
    )


def fusion_model_call(
    model: dict[str, Any],
    body: dict[str, Any],
    config: dict[str, Any],
    profile_id: str,
    fusion: dict[str, Any],
    *,
    allow_response_format: bool = False,
) -> dict[str, Any]:
    return build_fusion_runner().model_call(
        model,
        body,
        config,
        profile_id,
        fusion,
        allow_response_format=allow_response_format,
    )


def run_fusion_panel_calls(
    candidates: list[dict[str, Any]],
    body: dict[str, Any],
    config: dict[str, Any],
    profile_id: str,
    fusion: dict[str, Any],
    deadline: float,
) -> tuple[list[dict[str, Any]], bool]:
    return build_fusion_runner().panel_calls(candidates, body, config, profile_id, fusion, deadline)


def fusion_panel_heading(index: int, result: dict[str, Any], anonymize: bool) -> str:
    return fusion_panel_heading_use_case(index, result, anonymize)


def fusion_prompt_from_outputs(original_messages: Any, panel_results: list[dict[str, Any]], judge: dict[str, Any] | None = None, draft_text: str | None = None, *, anonymize: bool = True, ranking_summary: str | None = None) -> list[dict[str, Any]]:
    return fusion_prompt_from_outputs_use_case(
        original_messages,
        panel_results,
        judge,
        draft_text,
        anonymize=anonymize,
        ranking_summary=ranking_summary,
    )


def fusion_judge_messages(original_messages: Any, panel_results: list[dict[str, Any]], *, anonymize: bool = True) -> list[dict[str, Any]]:
    return fusion_judge_messages_use_case(original_messages, panel_results, anonymize=anonymize)


def fusion_ranking_messages(original_messages: Any, panel_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return fusion_ranking_messages_use_case(original_messages, panel_results)


def parse_fusion_ranking(text: str, valid_labels: set[str]) -> list[str]:
    return parse_fusion_ranking_use_case(text, valid_labels)


def aggregate_fusion_rankings(parsed_rankings: list[list[str]]) -> list[dict[str, Any]]:
    return aggregate_fusion_rankings_use_case(parsed_rankings)


def fusion_ranking_summary(aggregate: list[dict[str, Any]]) -> str:
    return fusion_ranking_summary_use_case(aggregate)


def run_fusion_ranking_round(
    successes: list[dict[str, Any]],
    body: dict[str, Any],
    config: dict[str, Any],
    fusion: dict[str, Any],
    deadline: float,
) -> dict[str, Any]:
    # Reuse the concurrent panel machinery: each successful panel model ranks the
    # anonymized answers. The rankers are the panel models, so no new selection,
    # no new providers, and the recursion-depth header is set per call. Stats and
    # cooldowns are intentionally not updated here: these models were already
    # recorded in the panel round, so re-recording would double-count them.
    return build_fusion_runner().ranking_round(
        successes,
        body,
        config,
        fusion,
        deadline,
        panel_calls=run_fusion_panel_calls,
    )


def extract_fusion_judge_json(text: str) -> Any:
    return extract_fusion_judge_json_use_case(text)


def normalize_fusion_judge_json(value: Any) -> dict[str, Any] | None:
    return normalize_fusion_judge_json_use_case(value)


def valid_fusion_judge_json(value: Any) -> bool:
    return valid_fusion_judge_json_use_case(value)


def fusion_response_ports() -> FusionResponsePorts:
    return FusionResponsePorts(
        ficelle_response_headers=ficelle_response_headers,
        safe_detail=safe_detail,
        fusion_model_id=FUSION_MODEL_ID,
        now_seconds=time.time,
    )


def fusion_openai_response(request_id: str, content: str) -> dict[str, Any]:
    return fusion_openai_response_use_case(request_id, content, ports=fusion_response_ports())


def fusion_response_headers(request_id: str, selected_model: dict[str, Any] | None, attempt_count: int, metadata: dict[str, Any]) -> dict[str, str]:
    return fusion_response_headers_use_case(
        request_id,
        selected_model,
        attempt_count,
        metadata,
        ports=fusion_response_ports(),
    )


def record_fusion_judge_outcome(state: dict[str, Any], metadata: dict[str, Any]) -> None:
    return record_fusion_judge_outcome_use_case(
        state,
        metadata,
        history_limit=FUSION_JUDGE_HISTORY_LIMIT,
        recordable_outcomes=FUSION_JUDGE_RECORDABLE_OUTCOMES,
    )


def record_fusion_quality_signal(state: dict[str, Any], metadata: dict[str, Any]) -> None:
    return record_fusion_quality_signal_use_case(state, metadata, alpha=FUSION_QUALITY_SIGNAL_ALPHA)


def persist_fusion_run(metadata: dict[str, Any]) -> None:
    def mutate(state: dict[str, Any]) -> None:
        apply_fusion_run_metadata_to_state_use_case(
            state,
            metadata,
            safe_last_run=safe_fusion_last_run,
            history_limit=FUSION_JUDGE_HISTORY_LIMIT,
            recordable_outcomes=FUSION_JUDGE_RECORDABLE_OUTCOMES,
            quality_signal_alpha=FUSION_QUALITY_SIGNAL_ALPHA,
        )

    _STATE_STORE.update(mutate, reason="persist_fusion_run")


def fusion_error_payload(message: str, request_id: str, error_type: str = "bad_request") -> dict[str, Any]:
    return fusion_error_payload_use_case(message, request_id, error_type, safe_detail=safe_detail)


def fusion_run_metadata(
    fusion: dict[str, Any],
    request_id: str,
    run_id: str,
    status: str,
    reason_code: str,
    request_started: float,
    stage_latency_seconds: dict[str, float],
    panel_candidates: list[dict[str, Any]],
    panel_results: list[dict[str, Any]] | None = None,
    degraded_flags: dict[str, bool] | None = None,
    recursion_depth: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return fusion_run_metadata_use_case(
        fusion,
        request_id,
        run_id,
        status,
        reason_code,
        request_started,
        stage_latency_seconds,
        panel_candidates,
        ports=fusion_metadata_ports(),
        panel_results=panel_results,
        degraded_flags=degraded_flags,
        recursion_depth=recursion_depth,
        extra=extra,
    )


def run_fusion_chat_completion(
    body: dict[str, Any],
    config: dict[str, Any],
    catalog: dict[str, Any],
    request_id: str,
    request_started: float,
    recursion_depth: int,
    *,
    entitled: bool | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    # Fusion is a closed Pro engine (ficelle_pro). On a core-only install the runner
    # and preflight symbols are None, so reject cleanly instead of crashing with a
    # NoneType call if a client targets ficelle/auto-fusion directly.
    if FusionRunner is None:
        return (
            404,
            {"error": {"message": f"model not found: {FUSION_MODEL_ID} (Fusion requires Ficelle Pro)", "type": "not_found"}},
            {},
        )
    is_entitled = license_ops.is_entitled() if entitled is None else entitled
    if not is_entitled:
        # Pack present but the license is inactive/expired/past-grace: fail closed with an
        # explicit license error (R10), not a generic upstream/model failure.
        return (
            402,
            {"error": {"message": "Ficelle Pro license is inactive or expired — run `ficelle license status` and refresh billing.", "type": "license_required"}},
            {},
        )
    preflight = build_fusion_request_preflight().prepare(
        body,
        config,
        request_id,
        request_started,
        recursion_depth,
    )
    if isinstance(preflight, FusionPreflightRejection):
        return preflight.status, preflight.payload, preflight.headers
    fusion = preflight.fusion
    deadline = preflight.deadline

    run_id = uuid.uuid4().hex
    panel_stage = build_fusion_panel_stage().run(
        body,
        config,
        catalog,
        fusion,
        deadline,
        request_id,
    )
    panel_candidates = panel_stage.candidates
    panel_results = panel_stage.results
    successes = panel_stage.successes
    failures = panel_stage.failures
    panel_latency = panel_stage.selection_latency_seconds
    panel_call_latency = panel_stage.call_latency_seconds
    timeout_reached = panel_stage.timeout_reached
    degraded_flags = dict(panel_stage.degraded_flags)
    panel_selection_metadata = {"panel_eligible_count": panel_stage.eligible_count}
    if not panel_candidates:
        metadata = fusion_run_metadata(
            fusion,
            request_id,
            run_id,
            "failed",
            "no_panel_candidates",
            request_started,
            {"selection": round(panel_latency, 4)},
            [],
            recursion_depth=recursion_depth,
            extra=panel_selection_metadata,
        )
        persist_fusion_run(metadata)
        record_last_route(FUSION_MODEL_ID, "fail", "no_panel_candidates", request_id, 0, 0, time.time() - request_started, attempts=[])
        write_route_log({"request_id": request_id, "requested_model": FUSION_MODEL_ID, "final_status": 503, "final_reason": "no_panel_candidates", "candidate_count": 0, "attempt_count": 0, "attempts": [], "stream": False})
        return 503, fusion_error_payload("no eligible strict-zero panel candidates for ficelle/auto-fusion", request_id, "no_available_model"), fusion_response_headers(request_id, None, 0, metadata)

    min_success = safe_int(fusion.get("min_successful_panel_outputs"), 2)
    single_panel_degraded = bool(fusion.get("allow_single_panel_degrade") and len(successes) == 1 and min_success > 1)
    if single_panel_degraded:
        degraded_flags["single_panel_degraded"] = True
    if len(successes) < min_success and not single_panel_degraded:
        errors = fusion_panel_error_rows(failures)
        panel_attempts = fusion_panel_attempt_rows(panel_results)
        metadata = fusion_run_metadata(
            fusion,
            request_id,
            run_id,
            "failed",
            "insufficient_panel_success",
            request_started,
            {"panel": round(panel_call_latency, 4)},
            panel_candidates,
            panel_results,
            degraded_flags=degraded_flags,
            recursion_depth=recursion_depth,
            extra={**panel_selection_metadata, "timeout_reached": timeout_reached},
        )
        persist_fusion_run(metadata)
        record_last_route(FUSION_MODEL_ID, "fail", "upstream_failure", request_id, len(panel_candidates), len(panel_results), time.time() - request_started, attempts=panel_attempts)
        write_route_log({"request_id": request_id, "requested_model": FUSION_MODEL_ID, "final_status": 502, "final_reason": "insufficient_panel_success", "candidate_count": len(panel_candidates), "attempt_count": len(panel_results), "attempts": panel_attempts, "stream": False})
        return 502, build_upstream_failure_error(FUSION_MODEL_ID, request_id, len(panel_candidates), panel_attempts, errors), fusion_response_headers(request_id, None, len(panel_results), metadata)

    peer_ranking_config = fusion.get("peer_ranking") if isinstance(fusion.get("peer_ranking"), dict) else {}
    peer_ranking_enabled = bool(peer_ranking_config.get("enabled")) and bool(successes)
    replaces_judge_active = peer_ranking_enabled and bool(peer_ranking_config.get("replaces_judge"))

    # Run the peer-ranking round concurrently with the judge: both are independent
    # over the same panel outputs, so overlapping them saves roughly the judge
    # stage's latency instead of paying judge + ranking sequentially. The ranking
    # round touches no shared mutable state (it does not record success/cooldown),
    # so it is safe to run while the judge writes its own results.
    ranking_executor = None
    ranking_future = None
    ranking_started = time.time()
    if peer_ranking_enabled:
        ranking_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        ranking_future = ranking_executor.submit(run_fusion_ranking_round, successes, body, config, fusion, deadline)

    fusion_runner = build_fusion_runner()
    judge_result = fusion_runner.judge_round(
        successes,
        body,
        config,
        catalog,
        fusion,
        deadline,
        request_id,
        replaces_judge_active=replaces_judge_active,
        max_judge_attempts=FUSION_JUDGE_MAX_ATTEMPTS,
    )
    judge_json = judge_result["judge_json"]
    judge_model = judge_result["judge_model"]
    judge_status = str(judge_result["judge_status"])
    judge_attempts = judge_result["judge_attempts"]
    degraded_flags.update(judge_result["degraded_flags"])
    judge_latency = float(judge_result["latency_seconds"])

    draft_text: str | None = None
    draft_model: dict[str, Any] | None = None
    draft_attempts: list[dict[str, Any]] = []
    draft_latency = 0.0
    if fusion.get("blind_draft"):
        draft_result = fusion_runner.draft_round(
            body,
            config,
            catalog,
            fusion,
            deadline,
            request_id,
        )
        draft_text = draft_result["draft_text"]
        draft_model = draft_result["draft_model"]
        draft_attempts = draft_result["draft_attempts"]
        degraded_flags.update(draft_result["degraded_flags"])
        draft_latency = float(draft_result["latency_seconds"])

    ranking_aggregate: list[dict[str, Any]] = []
    ranking_metadata: dict[str, Any] = {}
    ranking_latency = 0.0
    if ranking_future is not None:
        try:
            # Defensive timeout: the round is already deadline-bounded internally,
            # but never block the request past the remaining fusion budget.
            ranking_result = ranking_future.result(timeout=max(1.0, fusion_deadline_seconds(deadline)))
        except Exception:
            ranking_result = None
        finally:
            if ranking_executor is not None:
                ranking_executor.shutdown(wait=False)
        # Wall-clock since the ranking was submitted (before the judge); it overlaps
        # the judge stage, so it is an upper bound, not additive with judge latency.
        ranking_latency = time.time() - ranking_started
        ranking_outcome = resolve_peer_ranking_outcome(
            ranking_result,
            peer_ranking_config,
            safe_int=safe_int,
            default_min_rankers=DEFAULT_FUSION_CONFIG["peer_ranking"]["min_rankers"],
        )
        ranking_aggregate = ranking_outcome.aggregate
        ranking_metadata = ranking_outcome.metadata
        degraded_flags.update(ranking_outcome.degraded_flags)

    ranking_summary_for_synth = (
        fusion_ranking_summary(ranking_aggregate)
        if (ranking_aggregate and peer_ranking_config.get("feed_synthesizer"))
        else None
    )
    synthesizer_result = fusion_runner.synthesizer_round(
        successes,
        body,
        config,
        catalog,
        fusion,
        deadline,
        request_id,
        judge_json,
        draft_text,
        ranking_summary_for_synth,
        max_synthesizer_attempts=FUSION_SYNTHESIZER_MAX_ATTEMPTS,
    )
    final_text = str(synthesizer_result["final_text"])
    selected_synth = synthesizer_result["selected_synth"]
    synth_status = str(synthesizer_result["synth_status"])
    synth_attempts = synthesizer_result["synth_attempts"]
    synth_errors = synthesizer_result["synth_errors"]
    degraded_flags.update(synthesizer_result["degraded_flags"])
    synth_latency = float(synthesizer_result["latency_seconds"])
    reason_code = str(synthesizer_result["reason_code"])
    synth_candidate_count = int(synthesizer_result["candidate_count"])
    metadata = fusion_run_metadata(
        fusion,
        request_id,
        run_id,
        "ok" if final_text else "failed",
        reason_code,
        request_started,
        {"selection": round(panel_latency, 4), "panel": round(panel_call_latency, 4), "judge": round(judge_latency, 4), "ranking": round(ranking_latency, 4), "draft": round(draft_latency, 4), "synthesizer": round(synth_latency, 4)},
        panel_candidates,
        panel_results,
        degraded_flags=degraded_flags,
        recursion_depth=recursion_depth,
        extra={
            **panel_selection_metadata,
            **ranking_metadata,
            "judge_model": judge_model.get("id") if judge_model else None,
            "judge_status": judge_status,
            "judge_json_valid": judge_status == "ok",
            "synthesizer_model": selected_synth.get("id") if selected_synth else None,
            "synthesizer_status": synth_status,
            "draft_model": draft_model.get("id") if draft_model else None,
            "blind_draft_used": bool(draft_text),
        },
    )
    persist_fusion_run(metadata)
    attempts = fusion_panel_attempt_rows(panel_results)
    attempts.extend(judge_attempts)
    attempts.extend(draft_attempts)
    attempts.extend(synth_attempts)
    if not final_text:
        record_last_route(FUSION_MODEL_ID, "fail", reason_code, request_id, len(panel_candidates), len(attempts), time.time() - request_started, attempts=attempts)
        write_route_log({"request_id": request_id, "requested_model": FUSION_MODEL_ID, "final_status": 502, "final_reason": reason_code, "candidate_count": len(panel_candidates), "attempt_count": len(attempts), "attempts": attempts, "fusion": metadata, "stream": False})
        synth_failure_errors = synth_errors or [{"reason": "synthesizer_unavailable"}]
        synth_failure_candidate_count = max(synth_candidate_count, len(synth_failure_errors))
        return 502, build_upstream_failure_error(FUSION_MODEL_ID, request_id, synth_failure_candidate_count, synth_failure_errors, synth_failure_errors), fusion_response_headers(request_id, None, len(attempts), metadata)
    record_last_route(FUSION_MODEL_ID, "ok", "ok", request_id, len(panel_candidates), len(attempts), time.time() - request_started, selected_synth, attempts)
    write_route_log({"request_id": request_id, "requested_model": FUSION_MODEL_ID, "selected_model": selected_synth.get("id") if selected_synth else None, "final_status": 200, "final_reason": "ok", "candidate_count": len(panel_candidates), "attempt_count": len(attempts), "attempts": attempts, "fusion": metadata, "stream": False})
    return 200, fusion_openai_response(request_id, final_text), fusion_response_headers(request_id, selected_synth, len(attempts), metadata)


def benchmark_compression_source(config: dict[str, Any] | None = None) -> str:
    return build_compression_benchmark_source(
        config,
        safe_int=safe_int,
        default_chars=COMPRESSION_BENCHMARK_DEFAULT_CHARS,
        min_chars=COMPRESSION_BENCHMARK_MIN_CHARS,
        max_chars=COMPRESSION_BENCHMARK_MAX_CHARS,
    )


def benchmark_long_context_source(config: dict[str, Any] | None = None) -> str:
    return build_long_context_benchmark_source(
        config,
        safe_int=safe_int,
        default_tokens=LONG_CONTEXT_BENCHMARK_DEFAULT_TOKENS,
        min_tokens=LONG_CONTEXT_BENCHMARK_MIN_TOKENS,
        max_tokens=LONG_CONTEXT_BENCHMARK_MAX_TOKENS,
        chars_per_token=LONG_CONTEXT_BENCHMARK_CHARS_PER_TOKEN,
    )


def build_benchmark_probe_builder() -> BenchmarkProbeBuilder:
    return BenchmarkProbeBuilder(
        canonical_virtual_model_id=canonical_virtual_model_id,
        compression_source=benchmark_compression_source,
        long_context_source=benchmark_long_context_source,
        pick_vision_fixture=pick_vision_fixture,
        benchmark_file_data_url=benchmark_file_data_url,
        benchmark_audio_part=benchmark_audio_part,
        benchmark_media_answer=benchmark_media_answer,
        video_default_answer=BENCHMARK_VIDEO_DEFAULT_ANSWER,
        audio_default_answer=BENCHMARK_AUDIO_DEFAULT_ANSWER,
    )


def benchmark_body(profile_id: str, config: dict[str, Any] | None = None) -> tuple[dict[str, Any], str, str]:
    return build_benchmark_probe_builder().build(profile_id, config)


def extract_message_text(payload: Any) -> str:
    return benchmark_extract_message_text(payload)


def extract_reasoning_text(payload: Any) -> str:
    """Return the reasoning trace from a chat completion, or "" when absent.

    OpenAI-compatible providers expose the trace as ``message.reasoning`` (string,
    OpenRouter) or ``message.reasoning_content`` (string or block list). An empty
    result means the model did not emit a reasoning trace — used to fail a model
    that accepted the reasoning parameter but did not honor it.
    """
    return benchmark_extract_reasoning_text(payload)


def extract_tool_calls(payload: Any) -> list[dict[str, Any]]:
    return benchmark_extract_tool_calls(payload)


def has_deliverable_message(payload: Any) -> bool:
    return benchmark_has_deliverable_message(payload)


def sse_error_event(code: str, detail: str | None = None) -> bytes:
    """A terminal SSE error frame for a mid-stream upstream failure.

    Once bytes are streamed the HTTP status is already 200 and no fallback is possible, so the
    failure is surfaced to the client as an explicit error event before the connection closes —
    otherwise a truncated stream is indistinguishable from a complete response. Shape mirrors an
    OpenAI-style streamed error; ``detail`` is already redacted by ``safe_detail``.
    """
    payload = {"error": {"message": detail or code, "type": "stream_error", "code": code}}
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def stream_chunks_to_writer(chunks: Iterable[bytes], write_chunk: Any, flush: Any) -> dict[str, Any]:
    response_committed = False
    chunk_count = 0
    bytes_sent = 0
    stream_tail = b""
    try:
        for chunk in chunks:
            if not chunk:
                continue
            # The writer may emit HTTP headers before attempting the first body write. Mark
            # the response as committed before calling it so a body failure cannot trigger a
            # fallback on an already-started HTTP 200; update byte/tail metrics only after the
            # chunk itself was accepted.
            response_committed = True
            write_chunk(chunk)
            chunk_count += 1
            bytes_sent += len(chunk)
            stream_tail = (stream_tail + chunk)[-2:]
            flush()
    except Exception as exc:
        reason = "mid_stream_failure" if response_committed else "pre_stream_failure"
        detail = safe_detail(exc)
        if response_committed:
            # The HTTP response may already be committed (headers sent, no retry possible):
            # emit an explicit terminal error frame so the client sees a failure instead of a
            # silently truncated stream.
            # Best-effort — the client may already be gone (broken pipe); the failure is still
            # reported to the caller via the returned dict regardless.
            try:
                if stream_tail.endswith(b"\n\n"):
                    boundary = b""
                elif stream_tail.endswith(b"\n"):
                    boundary = b"\n"
                else:
                    boundary = b"\n\n"
                write_chunk(boundary + sse_error_event(reason, detail))
                flush()
            except Exception:
                pass
        return {
            "status": "fail",
            "reason": reason,
            "stream_started": response_committed,
            "chunk_count": chunk_count,
            "bytes_sent": bytes_sent,
            "error_type": type(exc).__name__,
            "message": detail,
        }
    if not response_committed:
        return {
            "status": "fail",
            "reason": "empty_stream",
            "stream_started": False,
            "chunk_count": 0,
            "bytes_sent": 0,
        }
    return {
        "status": "ok",
        "reason": "ok",
        "stream_started": True,
        "chunk_count": chunk_count,
        "bytes_sent": bytes_sent,
    }


def capability_for_benchmark(profile_id: str, test_type: str) -> str:
    canonical_profile_id = canonical_virtual_model_id(profile_id)
    if test_type == "tool_call":
        return "tool_call"
    if test_type == "orchestration":
        return "orchestration"
    if test_type in {"json", "fast"}:
        return "structured_json"
    if test_type == "compression":
        return "compression"
    if test_type == "exact_text" and canonical_profile_id == "ficelle/auto-orchestrator":
        return "instruction_following"
    if test_type == "vision":
        return "vision"
    if test_type == "reasoning":
        return "reasoning"
    if test_type == "long_context":
        return "long_context"
    return test_type or "benchmark"


def verified_capability_row(profile_id: str, model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    row = raw_verified_capability_row(profile_id, model, state)
    if not benchmark_result_matches_current_test(profile_id, row):
        return {}
    return row


def record_verified_capability(profile_id: str, model: dict[str, Any], result: dict[str, Any]) -> None:
    def mutate(state: dict[str, Any]) -> None:
        key = cooldown_key(model)
        verified = state.setdefault("verified_capabilities", {})
        if not isinstance(verified, dict):
            verified = {}
            state["verified_capabilities"] = verified
        by_profile = verified.setdefault(key, {})
        if not isinstance(by_profile, dict):
            by_profile = {}
            verified[key] = by_profile
        status = str(result.get("status") or "unknown")
        row = {
            "status": "verified" if status == "pass" else "failed",
            "capability": capability_for_benchmark(profile_id, str(result.get("test_type") or "")),
            "test_type": result.get("test_type"),
            "model_id": model.get("id"),
            "upstream_id": model.get("upstream_id"),
            "message": safe_detail(result.get("message")),
        }
        if status == "pass":
            row["verified_at"] = result.get("ran_at") or now_iso()
        else:
            row["failed_at"] = result.get("ran_at") or now_iso()
        by_profile[canonical_virtual_model_id(profile_id)] = redact_sensitive_json(row)

    _STATE_STORE.update(mutate, reason="record_verified_capability")


def tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    return benchmark_tool_call_arguments(call)


def json_fields_match(want: dict[str, Any], got: dict[str, Any]) -> tuple[bool, str]:
    """Compare expected typed fields against a parsed JSON object. Numbers must stay numbers (not
    strings), booleans/null exact, arrays compared as case-insensitive sets, strings trimmed and
    case-folded. A None expectation means "absent or null" (not invented) — only a concrete value
    fails. Tolerant on formatting, strict on type and value."""
    return benchmark_json_fields_match(want, got)


def validate_benchmark_response(test_type: str, expected: str, text: str, payload: Any | None = None) -> tuple[bool, str]:
    return validate_probe_response(
        test_type,
        expected,
        text,
        payload,
        compression_max_words=COMPRESSION_BENCHMARK_MAX_WORDS,
    )


def build_benchmark_runner() -> BenchmarkRunner:
    return BenchmarkRunner(
        load_or_refresh_catalog=load_or_refresh_catalog,
        select_models=select_models,
        order_benchmark_candidates=order_benchmark_candidates,
        benchmark_body=benchmark_body,
        invoke_model=invoke_model,
        classify_failure=classify_failure,
        set_cooldown=set_cooldown,
        record_benchmark_failure=record_benchmark_failure,
        record_benchmark_result=record_benchmark_result,
        record_success=record_success,
        record_verified_capability=record_verified_capability,
        record_capability_discrepancy=record_capability_discrepancy,
        extract_message_text=extract_message_text,
        safe_detail=safe_detail,
        redact_sensitive_json=redact_sensitive_json,
        now_iso=now_iso,
        route_blocking_reasons=BENCHMARK_ROUTE_BLOCKING_REASONS,
        media_error_types=(BenchmarkMediaError,),
        registry=ProbeRegistry(compression_max_words=COMPRESSION_BENCHMARK_MAX_WORDS),
    )


def build_capability_discovery_job() -> CapabilityDiscoveryJob:
    return CapabilityDiscoveryJob(
        benchmark_body=benchmark_body,
        invoke_model=invoke_model,
        classify_failure=classify_failure,
        set_cooldown=set_cooldown,
        record_benchmark_result=record_benchmark_result,
        record_verified_capability=record_verified_capability,
        record_success=record_success,
        extract_message_text=extract_message_text,
        safe_detail=safe_detail,
        now_iso=now_iso,
        route_blocking_reasons=BENCHMARK_ROUTE_BLOCKING_REASONS,
        registry=ProbeRegistry(compression_max_words=COMPRESSION_BENCHMARK_MAX_WORDS),
    )


def record_benchmark_result(profile_id: str, model: dict[str, Any], result: dict[str, Any]) -> None:
    def mutate(state: dict[str, Any]) -> None:
        record_benchmark_result_in_state(
            state,
            profile_id,
            model,
            result,
            cooldown_key=cooldown_key,
            redact_sensitive_json=redact_sensitive_json,
        )

    _STATE_STORE.update(mutate, reason="record_benchmark_result")


def record_benchmark_failure(model: dict[str, Any], reason: str, profile_id: str | None = None, detail: str | None = None) -> None:
    def mutate(state: dict[str, Any]) -> None:
        record_benchmark_failure_in_state(
            state,
            model,
            reason,
            detail=detail,
            profile_id=profile_id,
            update_failure_stats=update_failure_stats,
            record_model_error=record_model_error_in_state,
        )

    _STATE_STORE.update(mutate, reason=f"record_benchmark_failure:{reason}")


def benchmark_profile(profile_id: str, config: dict[str, Any], *, all_candidates: bool = False, only_model_id: str | None = None) -> dict[str, Any]:
    return build_benchmark_runner().benchmark_profile(
        profile_id,
        config,
        all_candidates=all_candidates,
        only_model_id=only_model_id,
    )


def run_benchmarks(profile_ids: list[str], config: dict[str, Any], *, all_candidates: bool = False, only_model_id: str | None = None) -> dict[str, Any]:
    profiles = normalized_virtual_profiles(config)
    known = set(profiles) | VIRTUAL_MODELS
    return run_benchmark_profiles(
        profile_ids,
        config,
        benchmark_profile=benchmark_profile,
        known_profile_ids=known,
        default_profile_ids=list(DEFAULT_BENCHMARK_PROFILES),
        safe_int=safe_int,
        now_iso=now_iso,
        all_candidates=all_candidates,
        only_model_id=only_model_id,
    )


def probeable_capability_profiles(config: dict[str, Any]) -> list[str]:
    """Specialized capabilities discovery can actually probe right now.

    The modality profiles drop out when their media fixtures are missing (``benchmark_body`` raises),
    so a model is never kept in the discovery queue forever for a capability it can't be tested on.
    """
    return probeable_capability_profiles_use_case(
        config,
        profile_ids=list(BENCHMARK_TEST_TYPES_BY_PROFILE),
        benchmark_body=benchmark_body,
    )


def model_due_capabilities(model: dict[str, Any], profile_ids: list[str], state: dict[str, Any]) -> list[str]:
    """Probeable capabilities for which this model has no fresh verified/failed verdict yet (TTL-aware)."""
    return model_due_capabilities_use_case(
        model,
        profile_ids,
        state,
        verified_capability_row=verified_capability_row,
    )


def discovery_eligible_models(catalog: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Invokable catalog models that aren't on cooldown or quarantined (same eligibility as routing).

    Model-centric on purpose: discovery does NOT apply a per-capability shape filter, so it can probe
    a capability a sparse provider's defaults never claimed (e.g. reasoning/vision) — that is how
    hidden capabilities are found rather than left as gaps.
    """
    return discovery_eligible_models_use_case(
        catalog,
        state,
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=model_is_quarantined,
    )


def models_needing_discovery(catalog: dict[str, Any], state: dict[str, Any], profile_ids: list[str]) -> list[dict[str, Any]]:
    return models_needing_discovery_use_case(
        catalog,
        state,
        profile_ids,
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=model_is_quarantined,
        verified_capability_row=verified_capability_row,
    )


def probe_model_capability(model: dict[str, Any], profile_id: str, config: dict[str, Any]) -> str:
    """Probe ONE model for ONE capability and record a per-(model, profile) verdict.

    Discovery-specific failure policy — deliberately different from ``benchmark_profile`` — because we
    are testing capabilities the model may not have, so a capability probe must NEVER bench the model
    globally: a clear client rejection (HTTP 400/422) or a 2xx response that fails validation is a
    per-profile ``failed`` verdict; a transient error (429/5xx/timeout/exception) records nothing and is
    retried next cycle; only a provider-wide block (billing/quota/auth) cools the model. Returns one of
    ``verified`` / ``failed`` / ``skip`` / ``blocked``.
    """
    return build_capability_discovery_job().probe_model_capability(model, profile_id, config)


def run_auto_benchmark_cycle(config: dict[str, Any]) -> dict[str, Any]:
    """One bounded capability-DISCOVERY pass.

    Takes up to ``auto_benchmark_models_per_cycle`` not-yet-fully-discovered models and probes each on
    ALL its still-unverdicted probeable capabilities (no stop-at-first-pass), so over time every model
    is mapped to every profile it supports. A capability probe never benches a model globally (see
    ``probe_model_capability``); a provider-wide block stops that model for the cycle. Converges then
    idles; the verified-capability TTL re-opens verdicts for re-discovery. Records a privacy-safe
    marker in state. Never raises out (the loop also guards).
    """
    result = build_capability_discovery_job().run_auto_benchmark_cycle(
        config,
        auto_benchmark_enabled=auto_benchmark_enabled,
        load_or_refresh_catalog=load_or_refresh_catalog,
        load_state=load_runtime_state,
        write_runtime_state=write_runtime_state,
        probeable_capability_profiles=probeable_capability_profiles,
        models_needing_discovery=models_needing_discovery,
        model_due_capabilities=model_due_capabilities,
        auto_benchmark_models_per_cycle=auto_benchmark_models_per_cycle,
        probe_model_capability=probe_model_capability,
    )
    if result.get("status") in {"disabled", "idle", "ran"}:
        clear_background_job_error("auto_benchmark")
    return result


def record_background_job_error(job_name: str, exc: BaseException, *, state_key: str = "auto_benchmark") -> None:
    def mutate(state: dict[str, Any]) -> None:
        record_background_job_error_in_state(
            state,
            job_name,
            exc,
            state_key=state_key,
            safe_detail=safe_detail,
            now_iso=now_iso,
        )

    _STATE_STORE.update(mutate, reason=f"record_background_job_error:{state_key}")


def clear_background_job_error(state_key: str = "auto_benchmark") -> bool:
    state = load_runtime_state()
    section = state.get(state_key)
    if not isinstance(section, dict) or "last_error" not in section:
        return False

    def mutate(next_state: dict[str, Any]) -> None:
        clear_background_job_error_in_state(next_state, state_key=state_key)

    _STATE_STORE.update(mutate, reason=f"clear_background_job_error:{state_key}")
    return True


def handle_background_job_error(job_name: str, exc: BaseException, *, state_key: str = "auto_benchmark") -> bool:
    try:
        record_background_job_error(job_name, exc, state_key=state_key)
    except Exception:
        return False
    return True


def auto_benchmark_loop(config: dict[str, Any]) -> None:
    """Background daemon loop driving auto-benchmarking. Re-reads the live config each iteration so
    the admin Settings toggle/interval take effect without a restart; a disabled cycle is a no-op.
    """
    time.sleep(AUTO_BENCHMARK_STARTUP_DELAY_SECONDS)
    while True:
        try:
            run_auto_benchmark_cycle(config)
        except Exception as exc:
            handle_background_job_error("auto_benchmark", exc)
        time.sleep(auto_benchmark_interval_seconds(config))


def run_canary(profile_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    configured = safe_string_list((config.get("canary") or {}).get("profiles"))
    selected = profile_ids or configured or list(DEFAULT_CANARY_PROFILES)
    run_due_quota_probes(config, load_or_refresh_catalog(config))
    result = run_benchmarks(selected, config)
    status = canary_status_from_benchmark_result(result, safe_int=safe_int)

    def mutate(state: dict[str, Any]) -> None:
        apply_canary_result_to_state(state, result, status)

    _STATE_STORE.update(mutate, reason="run_canary")
    result["canary_status"] = status
    return result


def profile_ids_from_admin_body(body: dict[str, Any]) -> list[str]:
    return profile_ids_from_admin_body_use_case(body)


def find_catalog_model(catalog: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    needle = str(model_id or "").strip()
    for model in catalog.get("models", []):
        if not isinstance(model, dict):
            continue
        if needle in {str(model.get("id") or ""), str(model.get("upstream_id") or "")}:
            return model
    return None


def build_hermes_export_builder() -> HermesExportBuilder:
    return HermesExportBuilder(
        picker_virtual_models=TARGET_EXPORT_VIRTUAL_MODELS,
        fusion_model_id=FUSION_MODEL_ID,
        fusion_visible_in_model_list=fusion_visible_in_model_list,
        safe_int=safe_int,
    )


def build_target_registry() -> dict[str, TargetAdapter]:
    hermes = HermesTargetAdapter(export_builder=build_hermes_export_builder())
    openclaw = OpenClawTargetAdapter(
        virtual_models=TARGET_EXPORT_VIRTUAL_MODELS,
        fusion_model_id=FUSION_MODEL_ID,
        fusion_visible_in_model_list=fusion_visible_in_model_list,
    )
    generic = GenericClientTargetAdapter(
        virtual_models=TARGET_EXPORT_VIRTUAL_MODELS,
        fusion_model_id=FUSION_MODEL_ID,
        fusion_visible_in_model_list=fusion_visible_in_model_list,
    )
    online = OnlineControlPlaneTargetAdapter()
    return {hermes.target_id: hermes, openclaw.target_id: openclaw, generic.target_id: generic, online.target_id: online}


def target_legacy_export_payload(target_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
    export = target_export(build_target_registry(), target_id, config)
    if export is None:
        return None
    return export.legacy_payload()


def target_export_public_fields(export: TargetExport) -> dict[str, Any]:
    return {
        "target_id": export.target_id,
        "base_url": export.base_url,
        "models": list(export.models),
        "presets": [dict(preset) for preset in export.presets],
        "warnings": list(export.warnings),
        "required_assets": list(export.required_assets),
        "verification_commands": [list(command) for command in export.verification_commands],
        "redaction_status": export.redaction_status,
    }


def target_public_export_payload(target_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
    export = target_export(build_target_registry(), target_id, config)
    if export is None:
        return None
    payload = target_export_public_fields(export)
    if export.config_text is not None:
        payload["yaml"] = export.config_text
    if export.config is not None:
        payload["config"] = dict(export.config)
    return payload


def target_summary_payload(adapter: TargetAdapter, config: dict[str, Any]) -> dict[str, Any]:
    export = adapter.export_config(TargetExportContext(config=config))
    return {
        **target_export_public_fields(export),
        "display_name": adapter.display_name,
        "kind": adapter.kind,
        "supports_plugin_install": adapter.supports_plugin_install,
        "supports_config_export": adapter.supports_config_export,
        "supports_health_check": adapter.supports_health_check,
    }


def targets_contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    registry = build_target_registry()
    return {
        "targets": [target_summary_payload(registry[target_id], config) for target_id in sorted(registry)],
        "exports": {target_id: f"/admin/export/{target_id}" for target_id in sorted(registry)},
        "service": {
            "status_endpoint": "/admin/status.json",
            "state_endpoint": "/admin/state",
            "settings_endpoint": "/admin/settings",
            "credential_key_endpoint_template": "/admin/providers/{provider}/key",
        },
    }


def online_safe_status(config: dict[str, Any]) -> dict[str, Any]:
    return safe_online_status_payload(
        status=admin_status(config, run_quota_probes=False),
        config_fingerprint=catalog_config_fingerprint(config, include_credentials=False),
        config_structural_fingerprint=catalog_config_structural_fingerprint(config),
        target_version=ONLINE_CONTROL_PLANE_TARGET_VERSION,
    )


def hermes_config_export(config: dict[str, Any]) -> dict[str, Any]:
    export = target_legacy_export_payload("hermes", config)
    if export is None:
        raise RuntimeError("Hermes target adapter is not registered")
    return export


class RouterHandler(BaseHTTPRequestHandler):
    server_version = "Ficelle/0.1"

    def _send_json(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, status: int, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_export_payload(self, export: dict[str, Any]) -> None:
        query = urlparse(self.path).query
        if "format=yaml" in query and "yaml" in export:
            self._send_text(200, str(export["yaml"]), "text/yaml; charset=utf-8")
        else:
            self._send_json(200, export)

    def _send_bytes(self, status: int, data: bytes, content_type: str, cache_seconds: int = 0, no_cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if cache_seconds:
            self.send_header("Cache-Control", f"public, max-age={cache_seconds}, immutable")
        elif no_cache:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _admin_origin_ok(self) -> bool:
        """CSRF guard for admin mutations. A browser always sends ``Origin`` on a POST;
        accept it only when it is this exact loopback server (host AND port), which
        blocks both cross-site pages and other local apps on a different port. A request
        with no ``Origin`` (curl/CLI) is allowed — it cannot be driven cross-site."""
        return admin_origin_allowed(
            self.headers.get("Origin"),
            self.server.server_address[1],  # type: ignore[attr-defined]
        )

    def _admin_write_authorized(self) -> bool:
        """Guard secret-bearing admin writes: the CSRF origin check plus the local admin
        token (the token gates the credential-write endpoint specifically)."""
        if not self._admin_origin_ok():
            return False
        presented = self.headers.get("X-Ficelle-Admin-Token") or ""
        # Constant-time compare to avoid a token-recovery timing side-channel.
        return admin_token_matches(presented, admin_token())

    @property
    def config(self) -> dict[str, Any]:
        return self.server.config  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in {"/health", "/v1/health"}:
                catalog = load_or_refresh_catalog(self.config)
                self._send_json(200, {"status": "ok", "generated_at": catalog.get("generated_at"), "model_count": len(catalog.get("models") or [])})
                return
            if path == "/admin":
                self._send_html(200, admin_page_html())
                return
            if path == "/admin/state":
                self._send_json(200, admin_state(self.config))
                return
            if path == "/admin/status.json":
                self._send_json(200, admin_status(self.config))
                return
            if path == "/admin/update":
                self._send_json(200, update_service.public_update_status())
                return
            if path == "/admin/targets":
                self._send_json(200, targets_contract_payload(self.config))
                return
            if path == "/admin/online/status.json":
                self._send_json(200, online_safe_status(self.config))
                return
            if path == "/admin/state-diagnostics":
                self._send_json(200, runtime_state_diagnostics())
                return
            if path == "/admin/fusion":
                if FusionRunner is None:
                    self._send_json(404, {"error": {"message": "Fusion is a Ficelle Pro feature", "type": "not_found"}})
                    return
                state = load_runtime_state()
                self._send_json(200, fusion_admin_payload(self.config, state))
                return
            if path == "/admin/settings":
                self._send_json(200, router_settings_payload(self.config))
                return
            if path == "/admin/compression":
                if compress_block is None:
                    self._send_json(404, {"error": {"message": "Compression is a Ficelle Pro feature", "type": "not_found"}})
                    return
                self._send_json(200, compression_admin_payload(self.config, limit=admin_query_limit(self.path)))
                return
            if path == "/admin/license":
                # Gated on pack presence, NOT on entitlement: the License page must stay reachable
                # while a license is expired/past-due so the user can see why and refresh (R10).
                try:
                    payload = license_ops.current_status()
                except license_ops.LicenseNotInstalled:
                    self._send_json(404, _LICENSE_PRO_ONLY_404)
                    return
                payload["service_url"] = license_ops.service_url()
                self._send_json(200, payload)
                return
            if path == "/admin/license/install":
                pro_install = _load_pro_install_module()
                self._send_json(200, pro_install.read_install_status())
                return
            if path == "/admin/audit":
                self._send_json(200, {"entries": read_admin_audit(limit=admin_query_limit(self.path))})
                return
            if path == "/admin/requests":
                self._send_json(200, request_log_query_payload(self.path))
                return
            if path == "/admin/requests/summary":
                self._send_json(200, request_log_summary_payload(self.path))
                return
            if path.startswith("/admin/fonts/"):
                data = load_font_bytes(path[len("/admin/fonts/"):])
                if data is None:
                    self._send_json(404, {"error": {"message": "font not found", "type": "not_found"}})
                else:
                    self._send_bytes(200, data, "font/woff2", cache_seconds=31536000)
                return
            if path.startswith("/admin/logos/"):
                data = load_logo_bytes(path[len("/admin/logos/"):])
                if data is None:
                    self._send_json(404, {"error": {"message": "logo not found", "type": "not_found"}})
                else:
                    self._send_bytes(200, data, "image/svg+xml", cache_seconds=31536000)
                return
            if path.startswith("/admin/static/"):
                asset = load_admin_asset(path[len("/admin/static/"):])
                if asset is None:
                    self._send_json(404, {"error": {"message": "asset not found", "type": "not_found"}})
                else:
                    self._send_bytes(200, asset[0], asset[1], no_cache=True)
                return
            if path == "/admin/export/hermes-config":
                export = hermes_config_export(self.config)
                self._send_export_payload(export)
                return
            if path.startswith("/admin/export/"):
                target_id = path.removeprefix("/admin/export/")
                if target_id == "hermes":
                    export = target_legacy_export_payload(target_id, self.config)
                else:
                    export = target_public_export_payload(target_id, self.config)
                if export is None:
                    self._send_json(404, {"error": {"message": f"unknown target export: {target_id}", "type": "not_found"}})
                    return
                self._send_export_payload(export)
                return
            if path == "/v1/models":
                entitled = license_ops.is_entitled()
                effective_config = effective_runtime_config(self.config, entitled=entitled)
                catalog = load_or_refresh_catalog(self.config, entitled=entitled)
                data = []
                now = int(time.time())
                for mid in listed_virtual_model_ids(effective_config, entitled=entitled):
                    data.append({"id": mid, "object": "model", "created": now, "owned_by": "ficelle"})
                for model in catalog.get("models", []):
                    if not isinstance(model, dict):
                        continue
                    data.append({
                        "id": model.get("id"),
                        "object": "model",
                        "created": now,
                        "owned_by": f"ficelle/{model.get('source')}",
                        "context_length": model.get("context_length"),
                        "upstream_id": model.get("upstream_id"),
                        "invokable": model.get("invokable"),
                        "supports_tools": model.get("supports_tools"),
                        "supports_structured_outputs": model.get("supports_structured_outputs"),
                    })
                self._send_json(200, {"object": "list", "data": data})
                return
            if path in {"/status", "/v1/status"}:
                catalog = load_or_refresh_catalog(self.config)
                redacted = dict(catalog)
                self._send_json(200, redacted)
                return
            self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})
        except Exception as exc:
            self._send_json(500, safe_error_body(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        # CSRF guard for the whole admin mutation surface: a cross-origin browser POST is
        # rejected here. /v1/* (the inference API) is intentionally not gated.
        if path.startswith("/admin/") and not self._admin_origin_ok():
            self._send_json(403, {"error": {"code": "forbidden", "message": "cross-origin admin request rejected"}})
            return
        if path == "/admin/refresh":
            try:
                catalog = refresh_catalog(self.config)
                summary = catalog_summary(catalog)
                write_admin_audit("admin.refresh", after={"summary": summary})
                self._send_json(200, {"summary": summary})
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/update":
            try:
                body = self._read_body()
                action = str(body.get("action") or "install").strip().lower()
                if action == "check":
                    status = update_service.check_for_updates(force=True)
                    self._send_json(200, update_service.public_update_status(status))
                    return
                if action != "install":
                    raise ValueError("action must be check or install")
                update_service.spawn_update()
                write_admin_audit("admin.update.install", metadata={"triggered": True})
                self._send_json(202, {"status": "queued", "restart_expected": True})
            except update_service.UpdateInProgress as exc:
                self._send_json(409, {"error": {"message": str(exc), "type": "update_in_progress"}})
            except update_service.UpdateUnavailable as exc:
                self._send_json(409, {"error": {"message": str(exc), "type": "update_unavailable"}})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except update_service.UpdateError as exc:
                self._send_json(500, {"error": {"message": str(exc), "type": "update_error"}})
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        # License management (Pro). Shared with `ficelle license` via ficelle.license_ops; the
        # audit records only the resulting status/plan, never the license key (security section).
        if path == "/admin/license/activate":
            try:
                body = self._read_body()
                license_key = str(body.get("license_key") or body.get("key") or "").strip()
                if not license_key:
                    self._send_json(400, {"error": {"message": "license_key is required", "type": "invalid_request"}})
                    return
                entitlement = license_ops.activate(license_key)
                write_admin_audit("admin.license.activate", after={"status": entitlement.status, "plan": entitlement.plan})
                self._send_json(200, license_ops.status_dict(entitlement))
            except license_ops.LicenseNotInstalled:
                self._send_json(404, _LICENSE_PRO_ONLY_404)
            except license_ops.LicenseOperationError as exc:
                self._send_json(400, {"error": {"message": str(exc), "type": "license_error"}})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/license/refresh":
            try:
                entitlement = license_ops.refresh()
                write_admin_audit("admin.license.refresh", after={"status": entitlement.status})
                self._send_json(200, license_ops.status_dict(entitlement))
            except license_ops.LicenseNotInstalled:
                self._send_json(404, _LICENSE_PRO_ONLY_404)
            except license_ops.LicenseOperationError as exc:
                self._send_json(400, {"error": {"message": str(exc), "type": "license_error"}})
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/license/deactivate":
            try:
                service_ok = license_ops.deactivate()
                write_admin_audit("admin.license.deactivate", metadata={"service_ok": service_ok})
                self._send_json(200, {"deactivated": True, "service_ok": service_ok})
            except license_ops.LicenseNotInstalled:
                self._send_json(404, _LICENSE_PRO_ONLY_404)
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/license/install":
            # The in-app Pro unlock (paste your key, no terminal). Unlike the other /admin/license
            # routes this is available on a core-only install, since it IS the installer. It spawns
            # `ficelle install-pro` DETACHED so the helper outlives the restart it will trigger (the
            # daemon can't boot itself out and relaunch). The key rides in the child's environment —
            # never argv or logs — and the audit records only that an install was triggered.
            try:
                body = self._read_body()
                license_key = str(body.get("license_key") or body.get("key") or "").strip()
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
                return
            if not license_key:
                self._send_json(
                    400,
                    {
                        "error": {
                            "message": "license_key is required",
                            "type": "invalid_request",
                        }
                    },
                )
                return
            try:
                pro_install = _load_pro_install_module()
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
                return

            try:
                pro_install.queue_install()
                try:
                    subprocess.Popen(
                        pro_install.detached_install_command(),
                        env={
                            **os.environ,
                            "FICELLE_LICENSE_KEY": license_key,
                            "FICELLE_INSTALL_QUEUED": "1",
                        },
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception:
                    pro_install.best_effort_write_install_status("failed", "The Ficelle Pro installer could not start.")
                    raise
                write_admin_audit("admin.license.install", metadata={"triggered": True})
                self._send_json(202, {"status": "installing", "restart_expected": True})
            except pro_install.ProInstallInProgress:
                self._send_json(
                    409,
                    {
                        "error": {
                            "message": "A Ficelle Pro installation is already running.",
                            "type": "install_in_progress",
                        }
                    },
                )
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/benchmark":
            job_id: str | None = None
            try:
                body = self._read_body()
                profile_ids = profile_ids_from_admin_body(body)
                all_candidates = bool(body.get("all_candidates") or body.get("retest_all_candidates"))
                only_model_id = body.get("model_id")
                if only_model_id is not None and (not isinstance(only_model_id, str) or not only_model_id.strip()):
                    raise ValueError("model_id must be a non-empty string")
                only_model_id = only_model_id.strip() if isinstance(only_model_id, str) else None
                job_id = start_admin_job("benchmark", {"profiles": profile_ids, "all_candidates": all_candidates, "model_id": only_model_id})
                result = run_benchmarks(profile_ids, self.config, all_candidates=all_candidates, only_model_id=only_model_id)
                write_admin_audit("admin.benchmark", after={"summary": result.get("summary") or {}}, metadata={"profiles": profile_ids, "all_candidates": all_candidates, "model_id": only_model_id})
                self._send_json(200, result)
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            finally:
                if job_id:
                    finish_admin_job("benchmark", job_id)
            return

        if path == "/admin/canary":
            try:
                body = self._read_body()
                profile_ids = profile_ids_from_admin_body(body)
                result = run_canary(profile_ids, self.config)
                write_admin_audit("admin.canary", after={"summary": result.get("summary") or {}, "status": result.get("canary_status")}, metadata={"profiles": profile_ids})
                self._send_json(200, result)
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/cooldowns":
            try:
                body = self._read_body()
                action = str(body.get("action") or "clear").strip().lower()
                scope = str(body.get("scope") or "model").strip().lower()
                if action in {"clear_all", "remove_all"}:
                    if scope not in {"all", "model", "models", "provider", "providers", "quota", "quotas"}:
                        raise ValueError("scope must be all/model/provider/quota")
                    changed = clear_all_cooldowns(scope)
                    write_admin_audit("admin.cooldowns.clear_all", after={"changed": changed}, metadata={"scope": scope})
                    self._send_json(200, {"action": "clear_all", "scope": scope, "changed": changed})
                    return
                if action not in {"clear", "remove"}:
                    raise ValueError("action must be clear or clear_all")
                if scope in {"provider", "providers"}:
                    source = str(body.get("source") or body.get("provider") or "").strip()
                    if not source:
                        raise ValueError("source is required for provider cooldown clear")
                    changed = clear_provider_cooldown(source)
                    write_admin_audit(
                        "admin.cooldowns.clear_provider",
                        after={"changed": changed},
                        metadata={"scope": "provider", "source": source},
                    )
                    self._send_json(200, {"action": "clear", "scope": "provider", "source": source, "changed": changed})
                    return
                model_id = str(body.get("model_id") or body.get("id") or "").strip()
                if not model_id:
                    raise ValueError("model_id is required for model cooldown clear")
                catalog = load_or_refresh_catalog(self.config)
                model = find_catalog_model(catalog, model_id)
                if not model:
                    raise ValueError(f"unknown model: {model_id}")
                changed = clear_model_cooldown(model)
                write_admin_audit(
                    "admin.cooldowns.clear_model",
                    after={"changed": changed},
                    metadata={"scope": "model", "model_id": model.get("id"), "source": model.get("source")},
                )
                self._send_json(200, {"action": "clear", "scope": "model", "model_id": model.get("id"), "changed": changed})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/providers/probe":
            try:
                body = self._read_body()
                source = str(body.get("source") or body.get("provider") or "").strip()
                if not source:
                    raise ValueError("source is required")
                runtime_config = effective_runtime_config(self.config)
                cached_catalog = load_runtime_json(CATALOG_PATH, {})
                cached_catalog = cached_catalog if isinstance(cached_catalog, dict) else {}
                cached_catalog = filter_cached_catalog_to_enabled_providers(cached_catalog, runtime_config)
                if source not in known_provider_sources(runtime_config, cached_catalog):
                    raise ValueError(f"unknown provider: {source}")
                catalog = _load_or_refresh_catalog_for_effective_config(runtime_config)
                if source not in known_provider_sources(runtime_config, catalog):
                    raise ValueError(f"unknown provider: {source}")
                result = run_due_quota_probes(runtime_config, catalog, force=True, source=source)
                write_admin_audit("admin.providers.probe", after={"result": result}, metadata={"source": source})
                self._send_json(200, result)
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path.startswith("/admin/providers/") and path.endswith("/key"):
            try:
                if not self._admin_write_authorized():
                    self._send_json(403, {"error": {"code": "forbidden", "message": "admin token required"}})
                    return
                source = path[len("/admin/providers/"):-len("/key")].strip("/")
                body = self._read_body()
                catalog = load_or_refresh_catalog(self.config)
                if source not in known_provider_sources(self.config, catalog):
                    raise ValueError(f"unknown provider: {source}")
                # Server-side writes run in the launchd daemon, so they target the
                # platform's non-interactive store (dedicated keychain / Credential
                # Manager / libsecret), never the login keychain, and fall back to .env
                # where none is available (R9). Cross-platform, never macOS-only.
                if body.get("remove"):
                    cleared = remove_provider_key(source, self.config, store=server_secret_store())
                    remaining_legacy_sources = legacy_provider_credential_sources(
                        source,
                        self.config,
                    )
                    refresh_catalog(self.config)
                    write_admin_audit(
                        "admin.providers.remove_key",
                        after={
                            "cleared": cleared,
                            "remaining_legacy_sources": remaining_legacy_sources,
                        },
                        metadata={"source": source},
                    )
                    self._send_json(
                        200,
                        {
                            "source": source,
                            "removed": cleared,
                            "remaining_legacy_sources": remaining_legacy_sources,
                            "auth": provider_auth_row(source, self.config),
                        },
                    )
                    return
                key = str(body.get("key") or "").strip()
                if not key:
                    raise ValueError("key is required")
                validator = PROVIDER_KEY_VALIDATORS.get(source)
                if validator and not validator(key):
                    raise ValueError(f"invalid {source} API key format")
                target = store_provider_key(source, self.config, key, store=server_secret_store())
                refresh_catalog(self.config)
                write_admin_audit("admin.providers.set_key", after={"stored": True}, metadata={"source": source, "target": target})
                self._send_json(200, {"source": source, "stored": True, "target": target, "auth": provider_auth_row(source, self.config)})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/quarantine":
            try:
                body = self._read_body()
                model_id = str(body.get("model_id") or body.get("id") or "").strip()
                action = str(body.get("action") or "add").strip().lower()
                if not model_id:
                    raise ValueError("model_id is required")
                catalog = load_or_refresh_catalog(self.config)
                model = find_catalog_model(catalog, model_id)
                if not model:
                    raise ValueError(f"unknown model: {model_id}")
                if action in {"remove", "clear", "release", "unquarantine"}:
                    existed = clear_quarantine(model)
                    write_admin_audit(
                        "admin.quarantine.remove",
                        after={"changed": existed},
                        metadata={"model_id": model.get("id"), "source": model.get("source")},
                    )
                    self._send_json(200, {"model_id": model.get("id"), "quarantined": False, "changed": existed})
                elif action in {"add", "set", "quarantine"}:
                    reason = str(body.get("reason") or "manual")
                    note = str(body.get("note") or "manual quarantine from admin UI")
                    row = set_quarantine(model, reason, note, "manual", True)
                    write_admin_audit(
                        "admin.quarantine.add",
                        after={"record": row},
                        metadata={"model_id": model.get("id"), "source": model.get("source"), "reason": reason},
                    )
                    self._send_json(200, {"model_id": model.get("id"), "quarantined": True, "record": row})
                else:
                    raise ValueError("action must be add/remove")
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/fusion":
            if FusionRunner is None:
                self._send_json(404, {"error": {"message": "Fusion is a Ficelle Pro feature", "type": "not_found"}})
                return
            try:
                body = self._read_body()
                before = normalize_fusion_config(self.config.get("fusion"), strict=False)
                fusion = validate_fusion_payload(body, self.config)
                saved = save_fusion_config(self.config, fusion)
                audit = write_admin_audit(
                    "admin.fusion.save",
                    before={"fusion": before},
                    after={"fusion": saved},
                    metadata={"enabled": saved["enabled"], "visible_in_models": saved["visible_in_models"]},
                )
                state = load_runtime_state()
                self._send_json(200, {"fusion": saved, "observability": fusion_observability({"fusion": saved}, state), "audit_id": audit["id"]})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/fusion/rollback":
            if FusionRunner is None:
                self._send_json(404, {"error": {"message": "Fusion is a Ficelle Pro feature", "type": "not_found"}})
                return
            try:
                body = self._read_body()
                audit_id = str(body.get("audit_id") or body.get("id") or "").strip()
                fusion = rollback_fusion_from_audit(audit_id, self.config)
                state = load_runtime_state()
                self._send_json(200, {"fusion": fusion, "observability": fusion_observability({"fusion": fusion}, state)})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/settings":
            try:
                body = self._read_body()
                before = normalize_router_settings(self.config, strict=False)
                settings = validate_settings_payload(body)
                saved = save_router_settings(self.config, settings)
                audit = write_admin_audit(
                    "admin.settings.save",
                    before={"settings": before},
                    after={"settings": saved},
                    metadata={
                        "max_attempts_per_request": saved["max_attempts_per_request"],
                        "catalog_timeout_seconds": saved["catalog_timeout_seconds"],
                        "verified_capability_ttl_seconds": saved["verified_capability_ttl_seconds"],
                        "route_on_capability_reference": saved["route_on_capability_reference"],
                        "auto_benchmark_enabled": saved["auto_benchmark_enabled"],
                        "auto_benchmark_interval_seconds": saved["auto_benchmark_interval_seconds"],
                        "auto_benchmark_models_per_cycle": saved["auto_benchmark_models_per_cycle"],
                        "compression_mode": saved["compression"]["mode"],
                    },
                )
                self._send_json(200, {"settings": {"config": saved}, "audit_id": audit["id"]})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/settings/rollback":
            try:
                body = self._read_body()
                audit_id = str(body.get("audit_id") or body.get("id") or "").strip()
                settings = rollback_router_settings_from_audit(audit_id, self.config)
                self._send_json(200, {"settings": {"config": settings}})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/compression/retrieve":
            if compress_block is None:
                self._send_json(404, {"error": {"message": "Compression is a Ficelle Pro feature", "type": "not_found"}})
                return
            try:
                body = self._read_body()
                self._send_json(200, compression_retrieve_payload(body.get("hash"), body.get("query")))
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except FileNotFoundError as exc:
                self._send_json(404, {"error": {"code": "not_found", "message": str(exc)}})
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/compression/clear":
            if compress_block is None:
                self._send_json(404, {"error": {"message": "Compression is a Ficelle Pro feature", "type": "not_found"}})
                return
            try:
                removed = clear_compression_store(store_path=COMPRESSION_STORE_PATH)
                write_admin_audit("admin.compression.clear", after={"removed": removed}, metadata={"removed": removed})
                self._send_json(200, {"removed": removed})
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/profiles":
            try:
                body = self._read_body()
                catalog = load_or_refresh_catalog(self.config)
                before = normalized_virtual_profiles(self.config)
                profiles = validate_virtual_profiles_payload(body, self.config, catalog)
                saved = save_virtual_profiles(self.config, profiles)
                audit = write_admin_audit(
                    "admin.profiles.save",
                    before={"virtual_profiles": before},
                    after={"virtual_profiles": saved},
                    metadata={"profile_count": len(saved)},
                )
                self._send_json(200, {"virtual_profiles": saved, "audit_id": audit["id"]})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path == "/admin/profiles/rollback":
            try:
                body = self._read_body()
                audit_id = str(body.get("audit_id") or body.get("id") or "").strip()
                profiles = rollback_virtual_profiles_from_audit(audit_id, self.config)
                self._send_json(200, {"virtual_profiles": profiles})
            except ValueError as exc:
                self._send_json(400, bad_request_body(exc))
            except Exception as exc:
                self._send_json(500, safe_error_body(exc))
            return

        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        request_id = uuid.uuid4().hex
        requested_model = DEFAULT_CHAT_COMPLETION_MODEL
        request_started = time.time()
        try:
            body = self._read_body()
            chat_request = normalize_chat_completion_request(body)
            requested_model = chat_request.requested_model
            safe_requested_model = chat_request.safe_requested_model
            entitled = license_ops.is_entitled()
            effective_config = effective_runtime_config(self.config, entitled=entitled)
            chat_router = ChatCompletionRouter(
                config=effective_config,
                load_catalog=lambda _config: load_or_refresh_catalog(
                    self.config,
                    entitled=entitled,
                ),
                select_candidates=select_models,
                is_fusion_profile=lambda model: canonical_virtual_model_id(model) == FUSION_MODEL_ID,
                is_virtual_model=lambda model: model in configured_virtual_model_ids(effective_config),
                response_headers=ficelle_response_headers,
                record_last_route=record_last_route,
                write_route_log=write_route_log,
                max_attempts_for_request=max_attempts_for_request,
                prepare_compression_route_body=lambda request_body, config: prepare_compression_route_body(
                    request_body,
                    config,
                    entitled=entitled,
                ),
                now=time.time,
            )
            def attempt_ports_factory(attempt_plan: ChatCompletionAttemptPlan) -> ChatCompletionAttemptPorts:
                compression_metadata = attempt_plan.compression_metadata

                def stream_response(response: Any, model: dict[str, Any], attempt_count: int) -> dict[str, Any]:
                    headers_sent = False

                    def write_stream_chunk(chunk: bytes) -> None:
                        nonlocal headers_sent
                        if not headers_sent:
                            stream_start = build_streaming_response_start(
                                StreamingResponseStartInput(
                                    request_id=request_id,
                                    safe_requested_model=safe_requested_model,
                                    selected_model=model,
                                    attempt_count=attempt_count,
                                    response_status=response.status_code,
                                    response_headers=response.headers,
                                    compression=compression_metadata,
                                ),
                                build_headers=chat_success_response_headers,
                            )
                            self.send_response(stream_start.status)
                            self.send_header("Content-Type", stream_start.content_type)
                            for key, value in stream_start.headers.items():
                                self.send_header(key, value)
                            self.end_headers()
                            headers_sent = True
                        self.wfile.write(chunk)

                    return stream_chunks_to_writer(response.iter_content(chunk_size=None), write_stream_chunk, self.wfile.flush)

                return ChatCompletionAttemptPorts(
                    invoke_model=invoke_model,
                    apply_cooldown=lambda cooldown: apply_chat_attempt_cooldown(
                        cooldown,
                        effective_config,
                        request_id=request_id,
                        profile_id=safe_requested_model,
                    ),
                    record_success=record_success,
                    resolve_competence=lambda profile, model: model_route_competence(profile, model, fresh_runtime_state()),
                    record_telemetry=apply_chat_route_telemetry,
                    stream_response=stream_response,
                    detect_success_error=success_error_payload_failure,
                    has_deliverable=has_deliverable_message,
                    classify_failure=classify_failure,
                    is_timeout_exception=lambda exc: isinstance(exc, requests.exceptions.Timeout),
                    build_success_headers=chat_success_response_headers,
                    build_failure_error=build_upstream_failure_error,
                    build_failure_headers=chat_failure_response_headers,
                )

            chat_result = chat_router.handle(
                body,
                request_id=request_id,
                request_started=request_started,
                ports_factory=attempt_ports_factory,
                request=chat_request,
            )
            chat_route = chat_result.start
            catalog = chat_route.catalog
            if chat_route.is_fusion_request:
                recursion_depth = safe_int(self.headers.get("X-Ficelle-Fusion-Depth"), 0)
                status, payload, headers = run_fusion_chat_completion(
                    body,
                    effective_config,
                    catalog,
                    request_id,
                    request_started,
                    recursion_depth,
                    entitled=entitled,
                )
                self._send_json(status, payload, headers=headers)
                return
            if chat_route.response is not None:
                self._send_json(chat_route.response.status, chat_route.response.payload, headers=chat_route.response.headers)
                return
            attempt_result = chat_result.attempt_result
            if attempt_result is None:
                raise RuntimeError("chat completion route completed without a response")
            if attempt_result.raw_response is not None:
                success_response = attempt_result.raw_response
                self.send_response(success_response.status)
                self.send_header("Content-Type", success_response.content_type)
                for key, value in success_response.headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(success_response.content)))
                self.end_headers()
                self.wfile.write(success_response.content)
                return
            if attempt_result.json_response is not None:
                failure_response = attempt_result.json_response
                self._send_json(failure_response.status, failure_response.payload, headers=failure_response.headers)
                return
            return
        except ValueError as exc:
            self._send_json(
                400,
                bad_request_body(exc, request_id=request_id),
                headers=ficelle_response_headers(request_id, requested_model),
            )
        except Exception as exc:
            self._send_json(
                500,
                safe_error_body(exc, request_id=request_id),
                headers=ficelle_response_headers(request_id, requested_model),
            )


def print_summary(catalog: dict[str, Any]) -> None:
    providers = catalog.get("providers") or {}
    print(f"generated_at={catalog.get('generated_at')}")
    print(f"allow_paid_fallback={catalog.get('allow_paid_fallback')}")
    print(f"min_context_length={catalog.get('min_context_length')}")
    print(f"models_total={len(catalog.get('models') or [])}")
    print(f"models_invokable={sum(1 for m in catalog.get('models') or [] if m.get('invokable'))}")
    for source, row in providers.items():
        print(
            f"provider={source} raw={row.get('raw_count')} accepted={row.get('accepted_count')} "
            f"invokable={row.get('invokable')} error={row.get('error')} auth={row.get('auth_reason')}"
        )


def catalog_refresh_interval_seconds(config: dict[str, Any]) -> int:
    """How often the background catalog refresh runs: ~120s before the catalog TTL so the
    cached catalog never goes stale, with a 60s floor to avoid hammering providers, but
    always kept strictly below the TTL so the margin holds even for a small or malformed
    ``catalog_ttl_seconds`` (a bad value must not silently disable the refresh daemon).
    Re-derived from the live config each cycle."""
    try:
        ttl = int(config.get("catalog_ttl_seconds") or 3600)
    except (TypeError, ValueError):
        ttl = 3600
    if ttl <= 1:
        return 1
    return min(max(60, ttl - 120), ttl - 1)


def catalog_refresh_loop(config: dict[str, Any]) -> None:
    """Background daemon that keeps the cached catalog fresh. An idle instance gets no
    routing requests to trigger an on-demand refresh, so without this the catalog crosses
    its TTL and ``admin_status`` reports it as empty ("0 models / fail") even though
    routing self-heals on demand. Warm once, then force a refresh well within the TTL so
    the catalog never goes stale. Re-reads the live ``config`` each cycle; daemon so it
    dies with the process."""
    try:
        refresh_capability_oracle_if_stale(config)
        load_or_refresh_catalog(config)
    except Exception as exc:  # never let the warm crash the serving process
        print(f"initial catalog warm failed: {exc}")
    while True:
        time.sleep(catalog_refresh_interval_seconds(config))
        try:
            refresh_capability_oracle_if_stale(config)
            load_or_refresh_catalog(config, force=True)
        except Exception as exc:
            print(f"periodic catalog refresh failed: {exc}")


def serve(config: dict[str, Any]) -> None:
    host = str(config.get("host") or "127.0.0.1")
    port = int(config.get("port") or 8646)
    # Strict-zero fail-fast: refuse to start if paid fallback was somehow enabled.
    # The catalog warm runs in a background thread below (where this guard would be
    # swallowed), so assert it synchronously here before binding — cheap, no network.
    if config.get("allow_paid_fallback") is not False:
        raise RuntimeError("allow_paid_fallback must stay false for the free router MVP")
    # Bind and start serving BEFORE warming the catalog. A slow or hanging provider
    # catalog fetch must never keep the HTTP server from binding — otherwise the whole
    # service is unreachable at startup (`ficelle restart` reports "did not become
    # ready"). Admin status/health read the *cached* catalog, so they stay responsive
    # while the warm runs in the background; the live catalog populates when ready.
    server = ThreadingHTTPServer((host, port), RouterHandler)
    server.config = config  # type: ignore[attr-defined]

    # Keep the catalog fresh in the background — warm once, then refresh within the TTL so
    # an idle instance never lets it go stale (a stale catalog is reported as empty by
    # admin status: a false "0 models / fail"). Daemon so it dies with the process.
    threading.Thread(target=catalog_refresh_loop, args=(config,), name="ficelle-catalog-refresh", daemon=True).start()
    # Background capability qualification: the loop reads the live `config` each cycle (same dict the
    # admin mutates), so it is always started and just idles when `auto_benchmark_enabled` is off —
    # letting the Settings toggle take effect without a restart. Daemon so it dies with the process.
    threading.Thread(target=auto_benchmark_loop, args=(config,), name="ficelle-auto-benchmark", daemon=True).start()
    threading.Thread(target=update_service.update_check_loop, name="ficelle-update-check", daemon=True).start()
    print(f"Ficelle listening on http://{host}:{port}/v1")
    server.serve_forever()


def main_args(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ficelle local strict-zero model router")
    parser.add_argument("--refresh", action="store_true", help="Refresh catalog and print redacted summary")
    parser.add_argument("--auth-status", action="store_true", help="Print credential availability without secrets")
    parser.add_argument("--serve", action="store_true", help="Run the OpenAI-compatible HTTP server")
    parser.add_argument("--canary", action="store_true", help="Run configured canary benchmarks and print redacted JSON")
    parser.add_argument("--probe-provider", help="Run due quota probes for a provider and print redacted JSON")
    parser.add_argument("--host", help="Override host")
    parser.add_argument("--port", type=int, help="Override port")
    args = parser.parse_args(argv)

    config = load_config()
    if args.host:
        config["host"] = args.host
    if args.port:
        config["port"] = args.port

    if args.auth_status:
        print(json.dumps(auth_status(config), indent=2, ensure_ascii=False))
        return 0
    if args.refresh:
        refresh_capability_oracle_if_stale(config)
        catalog = refresh_catalog(config)
        print_summary(catalog)
        return 0
    if args.canary:
        print(json.dumps(run_canary([], config), indent=2, ensure_ascii=False))
        return 0
    if args.probe_provider:
        catalog = load_or_refresh_catalog(config)
        print(json.dumps(run_due_quota_probes(config, catalog, force=True, source=args.probe_provider), indent=2, ensure_ascii=False))
        return 0
    if args.serve:
        serve(config)
        return 0

    parser.print_help()
    return 0


def main() -> int:
    return main_args()


if __name__ == "__main__":
    raise SystemExit(main())
