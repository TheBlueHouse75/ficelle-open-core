from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ficelle.failures import (
    CALLER_CAUSED_FAILURE_REASONS,
    MODEL_NOT_SERVEABLE_NOTE,
    PROVIDER_SCOPED_COOLDOWN_REASONS,
    CooldownPolicy,
)


StateMutator = Callable[[dict[str, Any]], dict[str, Any] | None]
UpdateState = Callable[[StateMutator, str | None], dict[str, Any]]
NormalizeFreeAccess = Callable[[dict[str, Any]], dict[str, Any]]
SafeDetail = Callable[[Any], str | None]
SafeFloat = Callable[[Any, float], float]
NowSeconds = Callable[[], float]
NowIso = Callable[[], str]
SafeInt = Callable[[Any, int], int]
CanonicalProfileId = Callable[[str], str]


@dataclass(frozen=True)
class CooldownWritePorts:
    update_state: UpdateState
    update_failure_stats: Callable[[dict[str, Any], dict[str, Any], str], None]
    record_model_error_in_state: Callable[
        [dict[str, Any], dict[str, Any], str, str | None, int | str | None, str | None, str | None],
        None,
    ]
    cooldown_policy_for_reason: Callable[[str, str], CooldownPolicy]
    record_provider_error_in_state: Callable[
        [dict[str, Any], dict[str, Any], str, str | None, int | str | None, str | None],
        None,
    ]
    # Both return the key they blocked — the quota pool's, at its declared scope, and the
    # provider's `source` — so `set_cooldown` can report it rather than re-derive it.
    set_quota_cooldown_in_state: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str | None], str]
    set_provider_cooldown_in_state: Callable[[dict[str, Any], str, str, dict[str, Any], str | None], str]
    set_billing_quarantine_in_state: Callable[[dict[str, Any], dict[str, Any], str | None, str, str], None]
    set_no_free_quota_quarantine_in_state: Callable[[dict[str, Any], dict[str, Any], str | None, str, str], None]
    set_model_not_found_quarantine_in_state: Callable[[dict[str, Any], dict[str, Any], str | None, str, str], None]
    cooldown_key: Callable[[dict[str, Any]], str]
    now_seconds: Callable[[], float]
    now_iso: Callable[[], str]
    safe_detail: Callable[[Any], str]


@dataclass(frozen=True)
class AppliedCooldown:
    """The keys one cooldown write blocked BEYOND the candidate that earned it.

    A model cooldown and a quarantine are deliberately absent: they block the one model that just
    failed, which its caller is not going to try again anyway. What is here is what also rules out
    *other* candidates — a whole provider, or whichever slice of a quota pool the provider declares
    — and it is reported by the write instead of re-derived from the reason, so a caller acting on
    it cannot drift from what the state actually says.
    """

    provider_source: str = ""
    quota_key: str = ""


@dataclass(frozen=True)
class CooldownReadPorts:
    normalized_free_access: NormalizeFreeAccess
    safe_detail: SafeDetail
    safe_float: SafeFloat
    now_seconds: NowSeconds
    free_access_scopes: set[str]


@dataclass(frozen=True)
class QuarantinePorts:
    safe_detail: SafeDetail
    now_iso: NowIso
    cooldown_key: Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class CooldownStatsPorts:
    safe_int: SafeInt
    safe_detail: SafeDetail
    now_iso: NowIso
    now_epoch: Callable[[], float]
    cooldown_key: Callable[[dict[str, Any]], str]
    canonical_virtual_model_id: CanonicalProfileId


@dataclass(frozen=True)
class CooldownMutationPorts:
    safe_int: SafeInt
    safe_detail: SafeDetail
    now_seconds: NowSeconds
    now_iso: NowIso
    quota_cooldown_key: Callable[[dict[str, Any]], str]
    quota_cooldown_scope: Callable[[dict[str, Any]], str]
    quota_cooldown_scope_from_key: Callable[[str], str]
    update_provider_failure_stats: Callable[[dict[str, Any], str, str], None]
    default_quota_probe_backoff_seconds: tuple[int, ...]


@dataclass(frozen=True)
class CooldownSuccessPorts:
    safe_float: SafeFloat
    now_seconds: NowSeconds
    now_iso: NowIso
    cooldown_key: Callable[[dict[str, Any]], str]
    quota_cooldown_matches_model: Callable[[str, dict[str, Any]], bool]
    clear_provider_error_in_state: Callable[[dict[str, Any], str], None]
    clear_model_error_in_state: Callable[[dict[str, Any], dict[str, Any]], None]
    update_success_stats: Callable[..., None]
    update_provider_success_stats: Callable[[dict[str, Any], str, float], None]


def cooldown_key(model: dict[str, Any]) -> str:
    return f"{model.get('source')}::{model.get('upstream_id')}"


def quota_cooldown_scope(model: dict[str, Any], *, ports: CooldownReadPorts) -> str:
    access = ports.normalized_free_access(model)
    scope = str(access.get("scope") or "provider")
    return scope if scope in ports.free_access_scopes else "provider"


def quota_cooldown_key(model: dict[str, Any], *, ports: CooldownReadPorts) -> str:
    return quota_cooldown_key_for_scope(model, quota_cooldown_scope(model, ports=ports), ports=ports)


def quota_cooldown_key_for_scope(model: dict[str, Any], scope: str, *, ports: CooldownReadPorts) -> str:
    source = str(model.get("source") or "").strip()
    upstream_id = str(model.get("upstream_id") or "").strip()
    if scope == "model":
        return f"model:{source}::{upstream_id}"
    if scope == "account":
        account_id = ports.safe_detail(model.get("provider_account_id")) or source
        return f"account:{source}::{account_id}"
    if scope == "shared_account":
        account_id = ports.safe_detail(model.get("provider_account_id")) or source
        return f"shared_account:{account_id}"
    return f"provider:{source}"


def quota_cooldown_scope_from_key(key: str, *, ports: CooldownReadPorts) -> str:
    scope = str(key).split(":", 1)[0]
    return scope if scope in ports.free_access_scopes else "provider"


def quota_cooldown_matches_model(key: str, model: dict[str, Any], *, ports: CooldownReadPorts) -> bool:
    return key == quota_cooldown_key_for_scope(model, quota_cooldown_scope_from_key(key, ports=ports), ports=ports)


def provider_on_cooldown(source: str, state: dict[str, Any], *, ports: CooldownReadPorts) -> tuple[bool, str | None]:
    provider_cooldowns = state.get("provider_cooldowns") if isinstance(state.get("provider_cooldowns"), dict) else {}
    cd = (provider_cooldowns.get(str(source)) or {}) if isinstance(provider_cooldowns, dict) else {}
    until = float(cd.get("until") or 0)
    if until > ports.now_seconds():
        return True, str(cd.get("reason") or "provider_cooldown")
    return False, None


def cooldown_block_scope(reason: str | None) -> str:
    """The blocking scope encoded in a `model_on_cooldown` reason string.

    `model_on_cooldown` prefixes provider-wide blocks with `provider:` and quota-pool
    blocks with `quota:`; everything else is a model-scoped cooldown. This is the single
    place that convention is decoded — selection exclusions and the synthetic-health
    eligibility records both read it from here, so the API refusal and the harness can
    never disagree about a block's scope.
    """
    reason_text = str(reason or "")
    if reason_text.startswith("provider:"):
        return "provider"
    if reason_text.startswith("quota:"):
        return "quota"
    return "model"


def model_on_cooldown(model: dict[str, Any], state: dict[str, Any], *, ports: CooldownReadPorts) -> tuple[bool, str | None]:
    provider_blocked, provider_reason = provider_on_cooldown(str(model.get("source") or ""), state, ports=ports)
    if provider_blocked:
        return True, f"provider:{provider_reason or 'cooldown'}"
    quota_cooldowns = state.get("quota_cooldowns") if isinstance(state.get("quota_cooldowns"), dict) else {}
    for key, raw in quota_cooldowns.items():
        if not isinstance(raw, dict) or not quota_cooldown_matches_model(str(key), model, ports=ports):
            continue
        until = ports.safe_float(raw.get("until"), 0.0)
        if until > 0:
            return True, "quota:quota_exhausted"
    cd = ((state.get("cooldowns") or {}).get(cooldown_key(model)) or {})
    until = float(cd.get("until") or 0)
    if until > ports.now_seconds():
        return True, str(cd.get("reason") or "cooldown")
    return False, None


def model_quarantine(model: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    row = ((state.get("quarantine") or {}).get(cooldown_key(model)) or {}) if isinstance(state, dict) else {}
    return row if isinstance(row, dict) and row else None


def model_is_quarantined(model: dict[str, Any], state: dict[str, Any]) -> bool:
    return model_quarantine(model, state) is not None


def quarantine_row(
    model: dict[str, Any],
    reason: str,
    note: str | None,
    source: str,
    manual: bool,
    *,
    ports: QuarantinePorts,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "note": ports.safe_detail(note),
        "set_at": ports.now_iso(),
        "source": source,
        "manual": manual,
        "model_id": model.get("id"),
        "upstream_id": model.get("upstream_id"),
    }


def set_quarantine_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    detail: str | None,
    source: str,
    fallback_note: str | None = None,
    *,
    ports: QuarantinePorts,
) -> None:
    quarantine = state.setdefault("quarantine", {})
    quarantine[ports.cooldown_key(model)] = quarantine_row(
        model,
        reason,
        detail or fallback_note,
        source,
        False,
        ports=ports,
    )


def set_billing_quarantine_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    detail: str | None,
    source: str,
    fallback_note: str | None = None,
    *,
    ports: QuarantinePorts,
) -> None:
    set_quarantine_in_state(state, model, "billing_or_paid", detail, source, fallback_note, ports=ports)


def set_no_free_quota_quarantine_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    detail: str | None,
    source: str,
    fallback_note: str | None = None,
    *,
    ports: QuarantinePorts,
) -> None:
    set_quarantine_in_state(
        state,
        model,
        "no_free_quota",
        detail,
        source,
        fallback_note or "provider reports a zero free-tier allocation for this model",
        ports=ports,
    )


def set_model_not_found_quarantine_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    detail: str | None,
    source: str,
    fallback_note: str | None = None,
    *,
    ports: QuarantinePorts,
) -> None:
    set_quarantine_in_state(
        state,
        model,
        "model_not_found",
        detail,
        source,
        fallback_note or MODEL_NOT_SERVEABLE_NOTE,
        ports=ports,
    )



# Half-life for the counters that feed scoring. Sized against 52 days of this install's own
# traffic: 240 models carry stats, half of them last active 48 days ago, yet those cumulative
# counters still fully decided their score. At 7 days a 48-day-old record keeps 0.9% of its
# weight (effectively neutral) while a model used weekly keeps 50% of its history — and 7 days
# is already the project's "evidence stays meaningful" horizon (verified_capability_ttl_seconds),
# so this is consistency with an existing constant rather than a fresh guess.
SCORE_DECAY_HALF_LIFE_SECONDS = 7 * 86_400


def _decay_scored_counters(record: dict[str, Any], now_epoch: float) -> None:
    """Age the scoring counters before folding in a new observation.

    `successes`/`failures` stay as lifetime totals for the admin card; scoring reads the decayed
    pair instead, so a model that degraded stops coasting on old wins and one that recovered
    stops being punished for them.
    """
    previous = record.get("scored_at")
    scored_successes = _safe_number(record.get("scored_successes"))
    scored_failures = _safe_number(record.get("scored_failures"))
    if previous is not None and (scored_successes or scored_failures):
        try:
            elapsed = max(0.0, now_epoch - float(previous))
        except (TypeError, ValueError):
            elapsed = 0.0
        if elapsed > 0:
            factor = 0.5 ** (elapsed / SCORE_DECAY_HALF_LIFE_SECONDS)
            scored_successes *= factor
            scored_failures *= factor
    record["scored_successes"] = round(scored_successes, 6)
    record["scored_failures"] = round(scored_failures, 6)
    record["scored_at"] = now_epoch


def _safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def _record_failure(record: dict[str, Any], reason: str, *, ports: CooldownStatsPorts) -> None:
    record["requests"] = ports.safe_int(record.get("requests"), 0) + 1
    record["failures"] = ports.safe_int(record.get("failures"), 0) + 1
    _decay_scored_counters(record, ports.now_epoch())
    if reason not in CALLER_CAUSED_FAILURE_REASONS:
        # A caller's own bad request is visible in the lifetime totals but must not drag the
        # score, the same exemption the consecutive-failure streak already makes.
        record["scored_failures"] = round(_safe_number(record.get("scored_failures")) + 1.0, 6)
    # The streak drives a cumulative score penalty and the selection tie-break, so it stays reserved
    # for failures the model is answerable for. See CALLER_CAUSED_FAILURE_REASONS.
    if reason not in CALLER_CAUSED_FAILURE_REASONS:
        record["consecutive_failures"] = ports.safe_int(record.get("consecutive_failures"), 0) + 1
    record["last_failure_at"] = ports.now_iso()
    record["last_failure_reason"] = reason
    reasons = record.setdefault("failure_reasons", {})
    if isinstance(reasons, dict):
        reasons[reason] = ports.safe_int(reasons.get(reason), 0) + 1


def _latency_ewma(previous_latency: Any, latency_seconds: float) -> float:
    try:
        return float(previous_latency) * 0.7 + latency_seconds * 0.3
    except Exception:
        return latency_seconds


def _record_success(
    record: dict[str, Any],
    latency_seconds: float,
    *,
    ports: CooldownStatsPorts,
    representative_latency: bool = True,
) -> None:
    record["requests"] = ports.safe_int(record.get("requests"), 0) + 1
    record["successes"] = ports.safe_int(record.get("successes"), 0) + 1
    record["consecutive_failures"] = 0
    _decay_scored_counters(record, ports.now_epoch())
    record["scored_successes"] = round(_safe_number(record.get("scored_successes")) + 1.0, 6)
    if representative_latency:
        record["latency_ewma"] = _latency_ewma(record.get("latency_ewma"), latency_seconds)
    else:
        # A capability probe asks for a handful of tokens, so its latency says nothing about how
        # long this model takes on a real request — but it fed the same EWMA the latency score
        # reads, making a model that has only ever been probed look the fastest in the pool. The
        # success itself still counts: answering a probe is evidence the model works.
        record["probe_latency_ewma"] = _latency_ewma(record.get("probe_latency_ewma"), latency_seconds)
    record["last_success_at"] = ports.now_iso()


def update_failure_stats(state: dict[str, Any], model: dict[str, Any], reason: str, *, ports: CooldownStatsPorts) -> None:
    stats = state.setdefault("stats", {})
    key = ports.cooldown_key(model)
    record = stats.setdefault(key, {})
    _record_failure(record, reason, ports=ports)


def update_success_stats(
    state: dict[str, Any],
    model: dict[str, Any],
    latency_seconds: float,
    *,
    ports: CooldownStatsPorts,
    representative_latency: bool = True,
) -> None:
    stats = state.setdefault("stats", {})
    key = ports.cooldown_key(model)
    record = stats.setdefault(key, {})
    _record_success(record, latency_seconds, ports=ports, representative_latency=representative_latency)


def record_model_error_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    detail: str | None = None,
    status: int | str | None = None,
    request_id: str | None = None,
    profile_id: str | None = None,
    *,
    ports: CooldownStatsPorts,
) -> None:
    errors = state.setdefault("model_errors", {})
    errors[ports.cooldown_key(model)] = {
        "reason": reason,
        "detail": ports.safe_detail(detail),
        "status": status,
        "model_id": model.get("id"),
        "upstream_id": model.get("upstream_id"),
        "source": model.get("source"),
        "profile_id": ports.canonical_virtual_model_id(profile_id) if profile_id else None,
        "request_id": request_id,
        "seen_at": ports.now_iso(),
    }


def clear_model_error_in_state(state: dict[str, Any], model: dict[str, Any], *, ports: CooldownStatsPorts) -> None:
    errors = state.get("model_errors")
    if isinstance(errors, dict):
        errors.pop(ports.cooldown_key(model), None)


def record_provider_error_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    detail: str | None = None,
    status: int | str | None = None,
    request_id: str | None = None,
    *,
    ports: CooldownStatsPorts,
) -> None:
    source = str(model.get("source") or "").strip()
    if not source:
        return
    errors = state.setdefault("provider_errors", {})
    errors[source] = {
        "reason": reason,
        "detail": ports.safe_detail(detail),
        "status": status,
        "model_id": model.get("id"),
        "upstream_id": model.get("upstream_id"),
        "request_id": request_id,
        "seen_at": ports.now_iso(),
    }


def clear_provider_error_in_state(state: dict[str, Any], source: str) -> None:
    errors = state.get("provider_errors")
    if isinstance(errors, dict) and source:
        errors.pop(source, None)


def update_provider_failure_stats(
    state: dict[str, Any],
    source: str,
    reason: str,
    *,
    ports: CooldownStatsPorts,
) -> None:
    provider_stats = state.setdefault("provider_stats", {})
    record = provider_stats.setdefault(source, {})
    _record_failure(record, reason, ports=ports)


def update_provider_success_stats(
    state: dict[str, Any],
    source: str,
    latency_seconds: float,
    *,
    ports: CooldownStatsPorts,
) -> None:
    provider_stats = state.setdefault("provider_stats", {})
    record = provider_stats.setdefault(source, {})
    _record_success(record, latency_seconds, ports=ports)


def set_provider_cooldown_in_state(
    state: dict[str, Any],
    source: str,
    reason: str,
    config: dict[str, Any],
    detail: str | None = None,
    *,
    ports: CooldownMutationPorts,
) -> str:
    """Cool a whole provider, and return the key that now blocks it (its `source`), or `""`.

    Reported rather than inferred so a caller can act on what was *written*: the attempt loop
    diverts the rest of its window off exactly the scopes its own failures blocked.
    """
    source = str(source or "").strip()
    if not source:
        return ""
    ports.update_provider_failure_stats(state, source, reason)
    cooldowns = state.setdefault("provider_cooldowns", {})
    seconds_map = config.get("cooldown_seconds") or {}
    seconds = int(seconds_map.get(reason) or seconds_map.get("unavailable") or 600)
    cooldowns[source] = {
        "until": ports.now_seconds() + seconds,
        "reason": reason,
        "detail": ports.safe_detail(detail),
        "set_at": ports.now_iso(),
    }
    return source


def quota_probe_backoff_seconds(config: dict[str, Any], consecutive_failures: int, *, ports: CooldownMutationPorts) -> int:
    raw_values = config.get("quota_probe_backoff_seconds")
    values = [ports.safe_int(value, 0) for value in raw_values] if isinstance(raw_values, list) else []
    values = [value for value in values if value > 0]
    if not values:
        values = list(ports.default_quota_probe_backoff_seconds)
    index = max(0, min(len(values) - 1, consecutive_failures))
    return values[index]


def set_quota_cooldown_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    config: dict[str, Any],
    detail: str | None = None,
    *,
    ports: CooldownMutationPorts,
    probe_failed: bool = False,
    cooldown_key_override: str | None = None,
) -> str:
    """Cool a quota pool, and return the key that now blocks it.

    That key is NOT the source: it is whichever of the four scopes the provider declares
    (`quota_cooldown_key_for_scope`) — `model:`, `account:`, `shared_account:` or `provider:` — so
    it can block one model, or span several sources sharing one account. Reported for the same
    reason the provider cooldown reports its own: what blocks is what the caller must act on, and
    only the write knows it (`cooldown_key_override` included).
    """
    source = str(model.get("source") or "").strip()
    if source:
        ports.update_provider_failure_stats(state, source, "quota_exhausted")
    cooldowns = state.setdefault("quota_cooldowns", {})
    key = cooldown_key_override or ports.quota_cooldown_key(model)
    previous_raw = cooldowns.get(key)
    previous = previous_raw if isinstance(previous_raw, dict) else {}
    previous_failures = ports.safe_int(previous.get("consecutive_probe_failures"), 0)
    consecutive_probe_failures = previous_failures + 1 if probe_failed else previous_failures
    interval = quota_probe_backoff_seconds(config, consecutive_probe_failures, ports=ports)
    now_ts = ports.now_seconds()
    set_at = ports.now_iso()
    last_probe_at = set_at if probe_failed else previous.get("last_probe_at")
    cooldowns[key] = {
        "scope": ports.quota_cooldown_scope_from_key(key) if cooldown_key_override else ports.quota_cooldown_scope(model),
        "source": source,
        "model_id": model.get("id"),
        "upstream_id": model.get("upstream_id"),
        "reason": "quota_exhausted",
        "until": now_ts + interval,
        "set_at": set_at,
        "last_probe_at": last_probe_at,
        "next_probe_at": now_ts + interval,
        "probe_interval_seconds": interval,
        "consecutive_probe_failures": consecutive_probe_failures,
        "detail": ports.safe_detail(detail),
    }
    return key


def clear_model_cooldown_in_state(state: dict[str, Any], cooldown_key: str) -> bool:
    cooldowns = state.setdefault("cooldowns", {})
    if not isinstance(cooldowns, dict):
        cooldowns = {}
        state["cooldowns"] = cooldowns
    existed = cooldown_key in cooldowns
    cooldowns.pop(cooldown_key, None)
    return existed


def clear_provider_cooldown_in_state(state: dict[str, Any], source: str) -> bool:
    key = str(source or "").strip()
    provider_cooldowns = state.setdefault("provider_cooldowns", {})
    if not isinstance(provider_cooldowns, dict):
        provider_cooldowns = {}
        state["provider_cooldowns"] = provider_cooldowns
    existed = key in provider_cooldowns
    provider_cooldowns.pop(key, None)
    return existed


def clear_all_cooldowns_in_state(state: dict[str, Any], scope: str = "all") -> int:
    changed = 0
    if scope in {"all", "model", "models"}:
        cooldowns = state.setdefault("cooldowns", {})
        changed += len(cooldowns) if isinstance(cooldowns, dict) else 0
        state["cooldowns"] = {}
    if scope in {"all", "provider", "providers"}:
        provider_cooldowns = state.setdefault("provider_cooldowns", {})
        changed += len(provider_cooldowns) if isinstance(provider_cooldowns, dict) else 0
        state["provider_cooldowns"] = {}
    if scope in {"all", "quota", "quotas"}:
        quota_cooldowns = state.setdefault("quota_cooldowns", {})
        changed += len(quota_cooldowns) if isinstance(quota_cooldowns, dict) else 0
        state["quota_cooldowns"] = {}
    return changed


def source_has_active_quota_cooldown(state: dict[str, Any], source: str, *, ports: CooldownSuccessPorts) -> bool:
    cooldowns = state.get("quota_cooldowns") if isinstance(state.get("quota_cooldowns"), dict) else {}
    now_ts = ports.now_seconds()
    for raw in cooldowns.values():
        if not isinstance(raw, dict):
            continue
        if str(raw.get("source") or "") == source and ports.safe_float(raw.get("until"), 0.0) > now_ts:
            return True
    return False


def record_success_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    latency_seconds: float,
    *,
    ports: CooldownSuccessPorts,
    representative_latency: bool = True,
) -> None:
    key = ports.cooldown_key(model)
    successes = state.setdefault("successes", {})
    successes[key] = {
        "last_success_at": ports.now_iso(),
        "latency_seconds": latency_seconds,
    }
    cooldowns = state.get("cooldowns")
    if isinstance(cooldowns, dict):
        cooldowns.pop(key, None)
    provider_cooldowns = state.get("provider_cooldowns")
    source = str(model.get("source") or "").strip()
    if isinstance(provider_cooldowns, dict) and source:
        provider_cooldowns.pop(source, None)
    quota_cooldowns = state.get("quota_cooldowns")
    if isinstance(quota_cooldowns, dict):
        for quota_key, raw in list(quota_cooldowns.items()):
            if isinstance(raw, dict) and ports.quota_cooldown_matches_model(str(quota_key), model):
                quota_cooldowns.pop(quota_key, None)
    if not source_has_active_quota_cooldown(state, source, ports=ports):
        ports.clear_provider_error_in_state(state, source)
    quarantine = state.get("quarantine")
    if isinstance(quarantine, dict):
        row = quarantine.get(key)
        # A model that just answered is demonstrably serveable again: the catalog-drift
        # quarantine self-heals on its own success (L4-R3). Billing and manual quarantines
        # require explicit resolution and stay untouched.
        if isinstance(row, dict) and str(row.get("reason") or "") == "model_not_found":
            quarantine.pop(key, None)
    ports.clear_model_error_in_state(state, model)
    ports.update_success_stats(state, model, latency_seconds, representative_latency=representative_latency)
    if source:
        ports.update_provider_success_stats(state, source, latency_seconds)


def clear_quota_model_errors_for_source_in_state(state: dict[str, Any], source: str) -> None:
    errors = state.get("model_errors")
    if not isinstance(errors, dict):
        return
    for key, row in list(errors.items()):
        if not isinstance(row, dict):
            continue
        if row.get("reason") == "quota_exhausted" and str(row.get("source") or "") == source:
            errors.pop(key, None)


def clear_quota_cooldown_in_state(
    state: dict[str, Any],
    key: str,
    model: dict[str, Any],
    *,
    ports: CooldownSuccessPorts,
) -> None:
    cooldowns = state.get("quota_cooldowns")
    if isinstance(cooldowns, dict):
        cooldowns.pop(key, None)
    source = str(model.get("source") or "")
    if not source_has_active_quota_cooldown(state, source, ports=ports):
        provider_errors = state.get("provider_errors") if isinstance(state.get("provider_errors"), dict) else {}
        provider_error = provider_errors.get(source) or {}
        if isinstance(provider_error, dict) and provider_error.get("reason") == "quota_exhausted":
            ports.clear_provider_error_in_state(state, source)
        clear_quota_model_errors_for_source_in_state(state, source)
    ports.clear_model_error_in_state(state, model)


def prune_provider_scoped_model_cooldowns_in_state(state: dict[str, Any]) -> None:
    """Remove legacy model cooldowns whose reason is provider-scoped (L2-R4 migration).

    Under the one-blocking-scope contract these reasons never write a model cooldown:
    while the provider cooldown is active the row is redundant, and after it expires the
    row wrongly keeps one model blocked past provider recovery. Independent model-scoped
    reasons (`rate_limited_upstream`, `request_too_large`, timeouts, server errors) and
    quarantines are untouched."""
    cooldowns = state.get("cooldowns") if isinstance(state.get("cooldowns"), dict) else {}
    for key in list(cooldowns):
        row = cooldowns.get(key)
        if isinstance(row, dict) and str(row.get("reason") or "") in PROVIDER_SCOPED_COOLDOWN_REASONS:
            del cooldowns[key]


def set_cooldown(
    model: dict[str, Any],
    reason: str,
    config: dict[str, Any],
    *,
    ports: CooldownWritePorts,
    detail: str | None = None,
    status: int | str | None = None,
    request_id: str | None = None,
    profile_id: str | None = None,
) -> AppliedCooldown:
    """Write the cooldown policy for this failure, and report what it blocked beyond this model."""
    provider_source = ""
    quota_key = ""

    def mutate(state: dict[str, Any]) -> None:
        nonlocal provider_source, quota_key
        # Every write converges the state toward the one-blocking-scope contract, so a
        # legacy row cannot outlive the first cooldown written after an upgrade.
        prune_provider_scoped_model_cooldowns_in_state(state)
        ports.update_failure_stats(state, model, reason)
        ports.record_model_error_in_state(state, model, reason, detail, status, request_id, profile_id)
        source = str(model.get("source") or "").strip()
        policy = ports.cooldown_policy_for_reason(reason, source)
        if policy.record_provider_error:
            ports.record_provider_error_in_state(state, model, reason, detail, status, request_id)
        if policy.quota_cooldown:
            quota_key = ports.set_quota_cooldown_in_state(state, model, config, detail) or ""
        if policy.provider_cooldown:
            provider_source = ports.set_provider_cooldown_in_state(
                state, policy.provider_cooldown_source, reason, config, detail
            ) or ""
        quarantine = policy.quarantine
        if quarantine and quarantine.reason == "billing_or_paid":
            ports.set_billing_quarantine_in_state(
                state,
                model,
                detail,
                quarantine.source,
                quarantine.fallback_note,
            )
        if quarantine and quarantine.reason == "no_free_quota":
            ports.set_no_free_quota_quarantine_in_state(
                state,
                model,
                detail,
                quarantine.source,
                quarantine.fallback_note,
            )
        if quarantine and quarantine.reason == "model_not_found":
            ports.set_model_not_found_quarantine_in_state(
                state,
                model,
                detail,
                quarantine.source,
                quarantine.fallback_note,
            )
        if not policy.model_cooldown:
            return
        cooldowns = state.setdefault("cooldowns", {})
        seconds_map = config.get("cooldown_seconds") or {}
        seconds = int(seconds_map.get(reason) or seconds_map.get("unavailable") or 600)
        cooldowns[ports.cooldown_key(model)] = {
            "until": ports.now_seconds() + seconds,
            "reason": reason,
            "detail": ports.safe_detail(detail),
            "set_at": ports.now_iso(),
        }

    ports.update_state(mutate, f"set_cooldown:{reason}")
    return AppliedCooldown(provider_source=provider_source, quota_key=quota_key)
