from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ficelle.failures import (
    NO_MODEL_FAULT_FAILURE_REASONS,
    MODEL_NOT_SERVEABLE_NOTE,
    PROVIDER_SCOPED_COOLDOWN_REASONS,
    CooldownPolicy,
)
from ficelle.state_store import parse_iso_timestamp


StateMutator = Callable[[dict[str, Any]], dict[str, Any] | None]
UpdateState = Callable[[StateMutator, str | None], dict[str, Any]]
NormalizeFreeAccess = Callable[[dict[str, Any]], dict[str, Any]]
SafeDetail = Callable[[Any], str | None]
SafeFloat = Callable[[Any, float], float]
NowSeconds = Callable[[], float]
NowIso = Callable[[], str]
SafeInt = Callable[[Any, int], int]
CanonicalProfileId = Callable[[str], str]
TRANSIENT_QUOTA_PROBE_BACKOFF_SECONDS = (60, 300, 900)


@dataclass(frozen=True)
class CooldownWritePorts:
    update_state: UpdateState
    # `(state, model, reason, *, attempt_reason)` — the attempt reason keys the backoff window.
    update_failure_stats: Callable[..., None]
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
    until = ports.safe_float(cd.get("until"), 0.0)
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
    until = ports.safe_float(cd.get("until"), 0.0)
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


def score_decay_factor(elapsed_seconds: float) -> float:
    """The half-life weight an observation `elapsed_seconds` old still carries.

    Shared with the read path (`success_rate_for_model`): the counters are aged when a new
    observation is folded in AND again for the time since that write, so one clock and one curve
    govern both and an idle model cannot keep a frozen rate for weeks.
    """
    try:
        elapsed = float(elapsed_seconds)
    except (TypeError, ValueError):
        return 1.0
    if elapsed <= 0:
        return 1.0
    return 0.5 ** (elapsed / SCORE_DECAY_HALF_LIFE_SECONDS)


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
            factor = score_decay_factor(elapsed)
            scored_successes *= factor
            scored_failures *= factor
    record["scored_successes"] = round(scored_successes, 6)
    record["scored_failures"] = round(scored_failures, 6)
    record["scored_at"] = now_epoch


def _safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _safe_timestamp(value: Any) -> float | None:
    """Read a persisted timestamp without rejecting the Unix epoch itself."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


# How long two failures of the same kind still count as the same episode. Long enough to catch a
# model failing on a request every half hour (`minimax-m3` timed out 31 times that way), short
# enough that a model quiet for a working day starts its escalation over.
COOLDOWN_BACKOFF_WINDOW_SECONDS = 6 * 3600
# The escalation itself: base x 2^(k-1), capped at 4x the base and never above an hour, so a short
# by-design cooldown (`request_too_large`, 120s) stays short and no reason can be silently turned
# into a day-long block by repetition.
COOLDOWN_BACKOFF_MAX_SECONDS = 3600

# One memory of a model's failures: `failure_ledger` in its `state.stats` record, one row per
# ATTEMPT reason. A row carries the `origin` of the failures accumulated since it was last
# rebutted (`request` outranks `probe`), a `status`, a recency-weighted count (`weight`, aged on
# the scored counters' half-life) and the count inside the current 6-hour episode (`episode`) —
# two readings of the same failures at two horizons: the score asks how much a model has failed
# lately, the escalation asks whether it is failing right now — and `last_at`. Three derivations
# read it: `failure_penalty_weight` (the score), `recent_failure_reason_count` (the escalation) and
# `_settle_ledger_on_success` (the rebuttal rule). It replaced `consecutive_failures` (a streak any
# success zeroed, a 5-token probe included) and `recent_failure_reasons` (the repeat windows the
# escalation read); why, and the evidence, are in docs/components/router.md § Candidate scoring.
FAILURE_LEDGER_KEY = "failure_ledger"
FAILURE_ORIGIN_REQUEST = "request"
FAILURE_ORIGIN_PROBE = "probe"
# `open`: the model still answers for these failures. `retest_due`: a probe answered since, which
# proves the model responds, not that a real request would succeed, so the penalty collapses to
# one failure's worth (enough to rank behind clean peers, not enough to stay buried) until a real
# request closes it. `rebutted`: a success of the same standing answered; the row stays only as
# escalation memory, because a model failing one request in two never lacks a success.
LEDGER_STATUS_OPEN = "open"
LEDGER_STATUS_RETEST_DUE = "retest_due"
LEDGER_STATUS_REBUTTED = "rebutted"
LEDGER_STATUSES = frozenset({LEDGER_STATUS_OPEN, LEDGER_STATUS_RETEST_DUE, LEDGER_STATUS_REBUTTED})
# Rows per record. A model fails a handful of distinct ways and a row nothing reads any more is
# dropped on the next write, so the cap only guards the state file against a pathological reason
# vocabulary; when it bites, rebutted rows go first, then the lightest.
FAILURE_LEDGER_MAX_ROWS = 12
# Below this a row's recency-weighted count no longer says anything (about four half-lives).
FAILURE_LEDGER_MIN_WEIGHT = 0.05
# The score penalty per unit of open ledger weight, i.e. per recent failure the model has not
# answered for. Unchanged from the streak it replaced.
FAILURE_PENALTY_POINTS = 12.0


def _ledger_row(origin: str, status: str, weight: float, last_at: float, episode: int) -> dict[str, Any]:
    return {
        "origin": FAILURE_ORIGIN_PROBE if origin == FAILURE_ORIGIN_PROBE else FAILURE_ORIGIN_REQUEST,
        "status": status if status in LEDGER_STATUSES else LEDGER_STATUS_OPEN,
        "weight": round(_safe_number(weight), 6),
        "last_at": float(last_at),
        "episode": max(1, int(episode)),
    }


def _decayed_weight(row: dict[str, Any], now_epoch: float) -> float:
    return _safe_number(row.get("weight")) * score_decay_factor(now_epoch - _safe_number(row.get("last_at")))


def _legacy_ledger_rows(record: dict[str, Any], now_epoch: float) -> dict[str, dict[str, Any]]:
    """The ledger a pre-ledger record implies, so an existing install keeps its standing.

    The reason windows become rebutted rows (they only ever fed the escalation, and their count
    is the episode count), and a live streak becomes one open row under the last failure reason,
    dated by `last_failure_at` when that parses and by now otherwise: the penalty it carried is
    kept, not re-derived.
    """
    rows: dict[str, dict[str, Any]] = {}
    windows = record.get("recent_failure_reasons")
    if isinstance(windows, dict):
        for name, raw in windows.items():
            if not isinstance(raw, dict):
                continue
            count = _safe_number(raw.get("count"))
            last_at = _safe_timestamp(raw.get("last_at"))
            if count <= 0 or last_at is None:
                continue
            rows[str(name)] = _ledger_row(FAILURE_ORIGIN_REQUEST, LEDGER_STATUS_REBUTTED, count, last_at, int(count))
    streak = int(_safe_number(record.get("consecutive_failures")))
    if streak > 0:
        reason = str(record.get("last_failure_reason") or "").strip() or "unknown"
        last_at = parse_iso_timestamp(record.get("last_failure_at"))
        if last_at is None:
            last_at = now_epoch
        previous = rows.get(reason)
        weight = max(streak, previous["weight"] if previous else 0)
        rows[reason] = _ledger_row(FAILURE_ORIGIN_REQUEST, LEDGER_STATUS_OPEN, weight, last_at, previous["episode"] if previous else 1)
    return rows


def failure_ledger(record: Any, now_epoch: float) -> dict[str, dict[str, Any]]:
    """The record's ledger as read: normalized row copies, migrated from the pre-ledger fields when
    the record has none, malformed rows dropped. Never raises on hand-edited or older state."""
    if not isinstance(record, dict):
        return {}
    stored = record.get(FAILURE_LEDGER_KEY)
    if not isinstance(stored, dict):
        return _legacy_ledger_rows(record, now_epoch)
    rows: dict[str, dict[str, Any]] = {}
    for reason, raw in stored.items():
        if not isinstance(raw, dict):
            continue
        weight = _safe_number(raw.get("weight"))
        last_at = _safe_timestamp(raw.get("last_at"))
        if weight <= 0 or last_at is None:
            continue
        rows[str(reason)] = _ledger_row(
            str(raw.get("origin") or ""), str(raw.get("status") or ""), weight, last_at, int(_safe_number(raw.get("episode")))
        )
    return rows


def _write_ledger(record: dict[str, Any], rows: dict[str, dict[str, Any]], now_epoch: float) -> None:
    kept: dict[str, dict[str, Any]] = {}
    for reason, row in rows.items():
        if _decayed_weight(row, now_epoch) < FAILURE_LEDGER_MIN_WEIGHT:
            continue
        # Neither the penalty nor the escalation reads a rebutted row past the episode window.
        if row["status"] == LEDGER_STATUS_REBUTTED and (now_epoch - row["last_at"]) > COOLDOWN_BACKOFF_WINDOW_SECONDS:
            continue
        kept[reason] = row
    if len(kept) > FAILURE_LEDGER_MAX_ROWS:
        ranked = sorted(
            kept.items(),
            key=lambda item: (item[1]["status"] != LEDGER_STATUS_REBUTTED, _decayed_weight(item[1], now_epoch)),
        )
        for reason, _row in ranked[: len(kept) - FAILURE_LEDGER_MAX_ROWS]:
            kept.pop(reason)
    if kept:
        record[FAILURE_LEDGER_KEY] = kept
    else:
        record.pop(FAILURE_LEDGER_KEY, None)
    # The fields the ledger replaced: dropped on the first write after an upgrade, so a record is
    # migrated once and no reader can find a streak that disagrees with the ledger.
    record.pop("consecutive_failures", None)
    record.pop("recent_failure_reasons", None)


def _record_ledger_failure(record: dict[str, Any], reason: str, origin: str, now_epoch: float) -> None:
    rows = failure_ledger(record, now_epoch)
    previous = rows.get(reason)
    if previous is None:
        rows[reason] = _ledger_row(origin, LEDGER_STATUS_OPEN, 1.0, now_epoch, 1)
    else:
        # The origin describes the failures since the last rebuttal: a rebutted row starts over
        # with this failure's, and a request failure outranks a probe's because only a request
        # success may rebut it.
        if previous["status"] == LEDGER_STATUS_REBUTTED or origin == FAILURE_ORIGIN_REQUEST:
            merged_origin = origin
        else:
            merged_origin = previous["origin"]
        # A quiet working day ends the episode; the weight carries across it, aged.
        same_episode = (now_epoch - previous["last_at"]) <= COOLDOWN_BACKOFF_WINDOW_SECONDS
        rows[reason] = _ledger_row(
            merged_origin,
            LEDGER_STATUS_OPEN,
            # A rebuttal keeps only the episode memory. Its failures no longer belong to the
            # penalty, so a fresh failure starts that weight from one rather than reactivating it.
            (0.0 if previous["status"] == LEDGER_STATUS_REBUTTED else _decayed_weight(previous, now_epoch)) + 1.0,
            now_epoch,
            previous["episode"] + 1 if same_episode else 1,
        )
    _write_ledger(record, rows, now_epoch)


def _settle_ledger_on_success(record: dict[str, Any], origin: str, now_epoch: float) -> None:
    """The rebuttal rule: a request success rebuts every row; a probe success rebuts the rows
    probes wrote and only marks a request-origin row due for a re-test."""
    rows = failure_ledger(record, now_epoch)
    for row in rows.values():
        if row["status"] == LEDGER_STATUS_REBUTTED:
            continue
        if origin == FAILURE_ORIGIN_REQUEST or row["origin"] == FAILURE_ORIGIN_PROBE:
            row["status"] = LEDGER_STATUS_REBUTTED
        else:
            row["status"] = LEDGER_STATUS_RETEST_DUE
    _write_ledger(record, rows, now_epoch)


def _row_penalty_weight(row: dict[str, Any], now_epoch: float) -> float:
    """What one row adds to the penalty: its aged weight while open, at most one failure's worth
    once a probe has answered for it, nothing once rebutted."""
    if row["status"] == LEDGER_STATUS_REBUTTED:
        return 0.0
    weight = _decayed_weight(row, now_epoch)
    return min(weight, 1.0) if row["status"] == LEDGER_STATUS_RETEST_DUE else weight


def failure_penalty_weight(record: Any, now_epoch: float) -> float:
    """The recency-weighted count of failures this record still answers for; the score subtracts
    `FAILURE_PENALTY_POINTS` per unit."""
    return sum(_row_penalty_weight(row, now_epoch) for row in failure_ledger(record, now_epoch).values())


def failure_ledger_summary(record: Any, now_epoch: float) -> dict[str, Any]:
    """What the admin shows of the ledger, under the names the projection and dashboard use: the
    penalty weight, the reasons still open, and whether a probe has already answered for some."""
    open_reasons: dict[str, float] = {}
    penalty = 0.0
    retest_due = False
    for reason, row in failure_ledger(record, now_epoch).items():
        if row["status"] == LEDGER_STATUS_REBUTTED:
            continue
        open_reasons[reason] = round(_decayed_weight(row, now_epoch), 2)
        retest_due = retest_due or row["status"] == LEDGER_STATUS_RETEST_DUE
        penalty += _row_penalty_weight(row, now_epoch)
    return {
        "failure_penalty_weight": round(penalty, 2),
        "open_failure_reasons": open_reasons,
        "failure_retest_due": retest_due,
    }


def recent_failure_reason_count(record: Any, reason: str, now_epoch: float) -> int:
    """How many times `reason` failed inside the current episode (0 when the episode is over).

    A row whose last failure is older than the episode window counts for nothing — a quiet
    working day starts the escalation over — and a success does not end an episode: the row
    survives it as escalation memory, which is what the streak this replaced got wrong.
    """
    row = failure_ledger(record, now_epoch).get(reason)
    if row is None or (now_epoch - row["last_at"]) > COOLDOWN_BACKOFF_WINDOW_SECONDS:
        return 0
    return row["episode"]


def escalated_cooldown_seconds(base_seconds: int, repeats: int) -> int:
    """`base x 2^(repeats-1)`, capped — the model-cooldown backoff.

    The cooldown table is flat, so a model that times out every time it is picked was cooled for the
    same 300s and returned to the pool for the next request, over and over. Escalation is driven by
    the ledger's episode count, NOT by a streak: a streak is zeroed by the first success, which is
    exactly what a model failing one request in two never lacks. Providers already escalate this
    way for quota probes (`quota_probe_backoff_seconds`).
    """
    base = max(0, int(base_seconds))
    if repeats <= 1 or base >= COOLDOWN_BACKOFF_MAX_SECONDS:
        return base
    # Two doublings at most (4x the base), and never above an hour.
    return int(min(base * 2 ** min(repeats - 1, 2), COOLDOWN_BACKOFF_MAX_SECONDS))


def _record_failure(
    record: dict[str, Any],
    reason: str,
    *,
    ports: CooldownStatsPorts,
    attempt_reason: str = "",
    origin: str = FAILURE_ORIGIN_REQUEST,
) -> None:
    now_epoch = ports.now_epoch()
    record["requests"] = ports.safe_int(record.get("requests"), 0) + 1
    record["failures"] = ports.safe_int(record.get("failures"), 0) + 1
    _decay_scored_counters(record, now_epoch)
    # A failure the model is not answerable for — a caller's own bad request, Ficelle's own deadline
    # or restart — stays visible in the lifetime totals but must drag neither the score nor the
    # ledger: none of these reasons cools, so there is no escalation to feed either. See
    # NO_MODEL_FAULT_FAILURE_REASONS.
    if reason not in NO_MODEL_FAULT_FAILURE_REASONS:
        record["scored_failures"] = round(_safe_number(record.get("scored_failures")) + 1.0, 6)
        # The ledger row is keyed on the ATTEMPT reason when the caller observed one. Several
        # distinct endings share one cooldown reason — an empty assistant message, a bad
        # `finish_reason` and a transport exception all cool as `unavailable` — and counting them
        # together escalated each other's block on evidence about none of them.
        _record_ledger_failure(record, attempt_reason or reason, origin, now_epoch)
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
    origin: str = FAILURE_ORIGIN_REQUEST,
) -> None:
    now_epoch = ports.now_epoch()
    record["requests"] = ports.safe_int(record.get("requests"), 0) + 1
    record["successes"] = ports.safe_int(record.get("successes"), 0) + 1
    _settle_ledger_on_success(record, origin, now_epoch)
    _decay_scored_counters(record, now_epoch)
    record["scored_successes"] = round(_safe_number(record.get("scored_successes")) + 1.0, 6)
    if origin == FAILURE_ORIGIN_REQUEST:
        record["latency_ewma"] = _latency_ewma(record.get("latency_ewma"), latency_seconds)
    else:
        # A capability probe asks for a handful of tokens, so its latency says nothing about how
        # long this model takes on a real request — but it fed the same EWMA the latency score
        # reads, making a model that has only ever been probed look the fastest in the pool. The
        # success itself still counts: answering a probe is evidence the model works.
        record["probe_latency_ewma"] = _latency_ewma(record.get("probe_latency_ewma"), latency_seconds)
    record["last_success_at"] = ports.now_iso()


def update_failure_stats(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    *,
    ports: CooldownStatsPorts,
    attempt_reason: str = "",
    origin: str = FAILURE_ORIGIN_REQUEST,
) -> None:
    stats = state.setdefault("stats", {})
    key = ports.cooldown_key(model)
    record = stats.setdefault(key, {})
    _record_failure(record, reason, ports=ports, attempt_reason=attempt_reason, origin=origin)


def update_success_stats(
    state: dict[str, Any],
    model: dict[str, Any],
    latency_seconds: float,
    *,
    ports: CooldownStatsPorts,
    origin: str = FAILURE_ORIGIN_REQUEST,
) -> None:
    stats = state.setdefault("stats", {})
    key = ports.cooldown_key(model)
    record = stats.setdefault(key, {})
    _record_success(record, latency_seconds, ports=ports, origin=origin)


def completion_tokens_per_second(
    completion_tokens: Any,
    latency_seconds: Any,
    first_byte_seconds: Any = None,
) -> float | None:
    """Generation throughput of one answered request, or None when it cannot be measured.

    Wall-clock latency conflates queueing with generation: a model that waits 40s and then writes
    900 tokens in 10s is not the same upstream as one that starts instantly and crawls, yet both
    read as 50s. Subtracting time-to-first-byte isolates the part the model controls. Telemetry
    only for now — nothing scores on it.
    """
    tokens = _safe_number(completion_tokens)
    latency = _safe_number(latency_seconds)
    if tokens <= 0 or latency <= 0:
        return None
    first_byte = _safe_number(first_byte_seconds)
    generation = latency - first_byte if 0 < first_byte < latency else latency
    if generation <= 0:
        return None
    return round(tokens / generation, 4)


def update_throughput_stats(
    state: dict[str, Any],
    model: dict[str, Any],
    tokens_per_second: float | None,
    *,
    ports: CooldownStatsPorts,
) -> None:
    """Fold one measured throughput into the per-upstream EWMA, if there is one to fold."""
    if tokens_per_second is None or tokens_per_second <= 0:
        return
    stats = state.setdefault("stats", {})
    record = stats.setdefault(ports.cooldown_key(model), {})
    record["completion_tokens_per_second"] = round(
        _latency_ewma(record.get("completion_tokens_per_second"), float(tokens_per_second)), 4
    )


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
    probe_ambiguous: bool = False,
    retry_after_seconds: int | None = None,
    retry_after_source: str | None = None,
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
    if source and not probe_ambiguous:
        ports.update_provider_failure_stats(state, source, "quota_exhausted")
    cooldowns = state.setdefault("quota_cooldowns", {})
    key = cooldown_key_override or ports.quota_cooldown_key(model)
    previous_raw = cooldowns.get(key)
    previous = previous_raw if isinstance(previous_raw, dict) else {}
    previous_failures = ports.safe_int(previous.get("consecutive_probe_failures"), 0)
    consecutive_probe_failures = previous_failures + 1 if probe_failed else previous_failures
    previous_ambiguous = ports.safe_int(previous.get("consecutive_ambiguous_probes"), 0)
    consecutive_ambiguous = previous_ambiguous + 1 if probe_ambiguous else 0
    if retry_after_seconds is not None:
        interval = max(1, min(86_400, ports.safe_int(retry_after_seconds, 1)))
    elif probe_ambiguous:
        index = min(len(TRANSIENT_QUOTA_PROBE_BACKOFF_SECONDS) - 1, max(0, consecutive_ambiguous - 1))
        interval = TRANSIENT_QUOTA_PROBE_BACKOFF_SECONDS[index]
    else:
        interval = quota_probe_backoff_seconds(config, consecutive_probe_failures, ports=ports)
    now_ts = ports.now_seconds()
    set_at = ports.now_iso()
    last_probe_at = set_at if probe_failed or probe_ambiguous else previous.get("last_probe_at")
    stored_detail = detail
    if retry_after_seconds is not None:
        stored_detail = f"retry after {interval}s via {retry_after_source or 'provider'}; {detail or 'quota probe'}"
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
        "consecutive_ambiguous_probes": consecutive_ambiguous,
        "probe_outcome": "ambiguous" if probe_ambiguous else "confirmed" if probe_failed else "initial",
        "detail": ports.safe_detail(stored_detail),
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
    origin: str = FAILURE_ORIGIN_REQUEST,
) -> None:
    """Record a success, and — unless the caller is a probe — lift the blocks it disproves.

    ``origin`` is the one fact a caller states about a success, because everything that follows
    from being a probe follows together. A benchmark or discovery call is a few tokens on a
    private body, so:

    - it does not lift a route-blocking cooldown. A 300s `timeout` earned by an actual
      600s-budget request was erased by the next auto-benchmark cycle seconds later, and the
      model went straight back into the pool;
    - its timing never reaches the latency EWMA that ranks models for real requests
      (`probe_latency_ewma` instead);
    - in the failure ledger it rebuts the failures probes wrote and only marks a request's
      failure due for a re-test, where a request success rebuts everything
      (`_settle_ledger_on_success`).

    Everything else the success proves still applies — stats, scoring, the recorded
    model/provider error, the quota block the probe genuinely re-tested, and the
    `model_not_found` quarantine a served completion self-heals.
    """
    key = ports.cooldown_key(model)
    clear_cooldowns = origin == FAILURE_ORIGIN_REQUEST
    source = str(model.get("source") or "").strip()
    successes = state.setdefault("successes", {})
    successes[key] = {
        "last_success_at": ports.now_iso(),
        "latency_seconds": latency_seconds,
    }
    if clear_cooldowns:
        cooldowns = state.get("cooldowns")
        if isinstance(cooldowns, dict):
            cooldowns.pop(key, None)
        provider_cooldowns = state.get("provider_cooldowns")
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
    ports.update_success_stats(state, model, latency_seconds, origin=origin)
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


def set_cooldown_in_state(
    state: dict[str, Any],
    model: dict[str, Any],
    reason: str,
    config: dict[str, Any],
    *,
    ports: CooldownWritePorts,
    detail: str | None = None,
    status: int | str | None = None,
    request_id: str | None = None,
    profile_id: str | None = None,
    retry_after_seconds: int | None = None,
    retry_after_source: str | None = None,
    attempt_reason: str = "",
    origin: str = FAILURE_ORIGIN_REQUEST,
) -> AppliedCooldown:
    """Apply the cooldown policy for this failure to `state`, and report what it blocked.

    Separate from `set_cooldown` so a caller with more than one write to make for the same failure
    can fold them into a single state cycle instead of taking the lock, parsing and rewriting the
    whole state twice.
    """
    provider_source = ""
    quota_key = ""
    effective_config = config
    # A provider that named a delay outranks Ficelle's own escalation: it knows when it will answer
    # again, and doubling on top of it would only strand a model the provider offered back.
    explicit_retry_after = retry_after_seconds is not None and retry_after_seconds > 0
    if explicit_retry_after:
        seconds = max(1, min(86_400, int(retry_after_seconds or 0)))
        effective_config = {
            **config,
            "cooldown_seconds": {**(config.get("cooldown_seconds") or {}), reason: seconds},
            "quota_probe_backoff_seconds": [seconds],
        }
        detail = f"retry after {seconds}s via {retry_after_source or 'provider'}; {detail or reason}"

    # Every write converges the state toward the one-blocking-scope contract, so a
    # legacy row cannot outlive the first cooldown written after an upgrade.
    prune_provider_scoped_model_cooldowns_in_state(state)
    ports.update_failure_stats(state, model, reason, attempt_reason=attempt_reason, origin=origin)
    ports.record_model_error_in_state(state, model, reason, detail, status, request_id, profile_id)
    source = str(model.get("source") or "").strip()
    policy = ports.cooldown_policy_for_reason(reason, source)
    if policy.record_provider_error:
        ports.record_provider_error_in_state(state, model, reason, detail, status, request_id)
    if policy.quota_cooldown:
        quota_key = ports.set_quota_cooldown_in_state(state, model, effective_config, detail) or ""
    if policy.provider_cooldown:
        provider_source = ports.set_provider_cooldown_in_state(
            state, policy.provider_cooldown_source, reason, effective_config, detail
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
    applied = AppliedCooldown(provider_source=provider_source, quota_key=quota_key)
    if not policy.model_cooldown:
        return applied
    cooldowns = state.setdefault("cooldowns", {})
    key = ports.cooldown_key(model)
    seconds_map = effective_config.get("cooldown_seconds") or {}
    base_seconds = int(seconds_map.get(reason) or seconds_map.get("unavailable") or 600)
    now_ts = ports.now_seconds()
    stats = state.get("stats") if isinstance(state.get("stats"), dict) else {}
    # `update_failure_stats` above already wrote this failure to the ledger, so the episode count
    # IS k. It is keyed on the ATTEMPT reason when there is one, so the escalation follows the
    # failure the caller actually observed rather than the cooldown bucket several of them share.
    repeats = (
        0
        if explicit_retry_after
        else recent_failure_reason_count(stats.get(key), attempt_reason or reason, now_ts)
    )
    seconds = escalated_cooldown_seconds(base_seconds, repeats)
    row: dict[str, Any] = {
        "until": now_ts + seconds,
        "reason": reason,
        "detail": ports.safe_detail(detail),
        "set_at": ports.now_iso(),
    }
    if seconds != base_seconds:
        # Stated in the row so an operator reading state sees an escalation rather than a
        # cooldown table that disagrees with the config.
        row["base_seconds"] = base_seconds
        row["reason_repeats"] = repeats
    cooldowns[key] = row
    return applied


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
    retry_after_seconds: int | None = None,
    retry_after_source: str | None = None,
    attempt_reason: str = "",
    origin: str = FAILURE_ORIGIN_REQUEST,
) -> AppliedCooldown:
    """Write the cooldown policy for this failure, and report what it blocked beyond this model.

    `origin` is `"probe"` for a benchmark or discovery probe: its failure enters the ledger as one
    a later probe success may rebut, while a request's failure waits for a request success.
    """
    applied = AppliedCooldown()

    def mutate(state: dict[str, Any]) -> None:
        nonlocal applied
        applied = set_cooldown_in_state(
            state,
            model,
            reason,
            config,
            ports=ports,
            detail=detail,
            status=status,
            request_id=request_id,
            profile_id=profile_id,
            retry_after_seconds=retry_after_seconds,
            retry_after_source=retry_after_source,
            attempt_reason=attempt_reason,
            origin=origin,
        )

    ports.update_state(mutate, f"set_cooldown:{reason}")
    return applied
