"""Bounded, redacted operator quality evidence for routed model/profile pairs.

The request log remains the attribution authority. This module deliberately stores
only a hash of that request id in a stable event slot, so the runtime ledger can
drive observability and optional ranking without becoming another request log.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ficelle.request_log import MAX_AGE_SECONDS
from ficelle.state_store import parse_iso_timestamp


QUALITY_FEEDBACK_MAX_SLOTS = 256
QUALITY_HALF_LIFE_SECONDS = 7 * 86_400
QUALITY_FEEDBACK_STATE_KEY = "quality_feedback"
_QUALITY_FEEDBACK_CONFIG_KEY = "_quality_feedback_config"
_ALLOWED_BODY_KEYS = frozenset({"request_id", "outcome", "severity", "reason", "validator", "expected_revision"})
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_VALIDATOR_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_OUTCOMES = frozenset({"pass", "fail"})
_FAIL_SEVERITIES = frozenset({"minor", "major", "critical"})
_REASONS = frozenset(
    {
        "empty_output",
        "invalid_json",
        "tool_loop",
        "unit_mismatch",
        "internal_contradiction",
        "critical_omission",
        "language_contamination",
        "unsafe_action",
        "manual_rejection",
    }
)
_FAILURE_WEIGHTS = {"minor": 1 / 3, "major": 1.0, "critical": 2.0}


class QualityFeedbackError(ValueError):
    """A closed error code suitable for the local admin API."""

    def __init__(self, code: str, http_status: int) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status


@dataclass(frozen=True)
class QualityFeedbackInput:
    request_id: str
    outcome: str
    severity: str
    reason: str
    validator: str
    expected_revision: int


def parse_quality_feedback_payload(payload: Any) -> QualityFeedbackInput:
    """Validate the intentionally closed feedback body before any attribution lookup."""
    if not isinstance(payload, dict) or set(payload) != _ALLOWED_BODY_KEYS:
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    request_id = payload.get("request_id")
    outcome = payload.get("outcome")
    severity = payload.get("severity")
    reason = payload.get("reason")
    validator = payload.get("validator")
    expected_revision = payload.get("expected_revision")
    if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    if not isinstance(outcome, str) or outcome not in _OUTCOMES:
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    if not isinstance(severity, str):
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    # A pass is a confirmation, not a severity claim. `none` makes the body equally explicit
    # in both cases and prevents a caller from smuggling a fail weight into a pass event.
    if (outcome == "pass" and severity != "none") or (outcome == "fail" and severity not in _FAIL_SEVERITIES):
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    if not isinstance(reason, str) or reason not in _REASONS:
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    if not isinstance(validator, str) or not _VALIDATOR_PATTERN.fullmatch(validator):
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or not 0 <= expected_revision <= 1_000_000:
        raise QualityFeedbackError("invalid_quality_feedback", 400)
    return QualityFeedbackInput(request_id, outcome, severity, reason, validator, expected_revision)


def quality_feedback_scoring_state(
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    canonical_profile_id: Callable[[str], str],
) -> dict[str, Any]:
    """Attach normalized, ephemeral scoring configuration without persisting config in state."""
    source = config.get("quality_feedback") if isinstance(config.get("quality_feedback"), dict) else {}
    raw_profiles = source.get("enabled_profiles") if isinstance(source, dict) else None
    profiles = raw_profiles if isinstance(raw_profiles, list) else ["ficelle/auto-orchestrator"]
    enabled_profiles = tuple(
        dict.fromkeys(
            canonical
            for value in profiles
            if isinstance(value, str)
            and (canonical := canonical_profile_id(value))
        )
    )
    view = dict(state) if isinstance(state, dict) else {}
    view[_QUALITY_FEEDBACK_CONFIG_KEY] = {
        "routing_enabled": bool(source.get("routing_enabled")) if isinstance(source, dict) else False,
        "enabled_profiles": enabled_profiles,
    }
    return view


def quality_feedback_routing_enabled(profile_id: str, state: dict[str, Any]) -> bool:
    settings = state.get(_QUALITY_FEEDBACK_CONFIG_KEY) if isinstance(state, dict) else None
    if not isinstance(settings, dict) or not settings.get("routing_enabled"):
        return False
    profiles = settings.get("enabled_profiles")
    return isinstance(profiles, (tuple, list, set)) and profile_id in profiles


def quality_feedback_summary_for_model(
    profile_id: str,
    model: dict[str, Any],
    state: dict[str, Any],
    now_epoch: Callable[[], float],
) -> dict[str, Any]:
    source = model.get("source") if isinstance(model, dict) else None
    upstream_id = model.get("upstream_id") if isinstance(model, dict) else None
    if not isinstance(source, str) or not source or not isinstance(upstream_id, str) or not upstream_id:
        return _empty_summary()
    return quality_feedback_summary(state, f"{source}::{upstream_id}", profile_id, now_epoch=now_epoch)


def quality_feedback_summary(
    state: dict[str, Any],
    model_key: str,
    profile_id: str,
    *,
    now_epoch: Callable[[], float],
) -> dict[str, Any]:
    """Compute recency-weighted quality evidence directly from durable event slots."""
    now = now_epoch()
    if not math.isfinite(now):
        return _empty_summary()
    slots = _slots_for(state, model_key, profile_id)
    sample_count = 0
    active_fail_weight = 0.0
    active_pass_weight = 0.0
    latest: dict[str, Any] | None = None
    latest_stamp: float | None = None
    for row in slots.values():
        if not isinstance(row, dict):
            continue
        expires_at = parse_iso_timestamp(row.get("expires_at"))
        recorded_at = parse_iso_timestamp(row.get("recorded_at"))
        outcome = row.get("outcome")
        severity = row.get("severity")
        reason = row.get("reason")
        validator = row.get("validator")
        if (
            expires_at is None
            or expires_at <= now
            or recorded_at is None
            or not isinstance(outcome, str)
            or not isinstance(severity, str)
            or outcome not in _OUTCOMES
            or not isinstance(reason, str)
            or reason not in _REASONS
            or not isinstance(validator, str)
            or not _VALIDATOR_PATTERN.fullmatch(validator)
        ):
            continue
        if outcome == "pass" and severity != "none":
            continue
        if outcome == "fail" and severity not in _FAIL_SEVERITIES:
            continue
        decay = 0.5 ** (max(0.0, now - recorded_at) / QUALITY_HALF_LIFE_SECONDS)
        sample_count += 1
        if outcome == "pass":
            active_pass_weight += decay
        else:
            active_fail_weight += _FAILURE_WEIGHTS[str(severity)] * decay
        if latest_stamp is None or recorded_at >= latest_stamp:
            latest = row
            latest_stamp = recorded_at
    if sample_count == 0 or latest is None:
        return _empty_summary()
    failure_penalty = min(24.0, 12.0 * active_fail_weight)
    pass_credit = min(4.0, active_pass_weight)
    adjustment = max(-24.0, min(4.0, pass_credit - failure_penalty))
    return {
        "sample_count": sample_count,
        "quality_adjustment": adjustment,
        "last_reason": latest["reason"],
        "last_severity": latest["severity"],
        "last_recorded_at": latest["recorded_at"],
        "status": latest["outcome"],
    }


def quality_feedback_status(state: dict[str, Any], *, now_epoch: Callable[[], float]) -> dict[str, Any]:
    """Redacted runtime projection: aggregates only, never slot ids or request ids."""
    feedback = state.get(QUALITY_FEEDBACK_STATE_KEY) if isinstance(state, dict) else None
    if not isinstance(feedback, dict):
        return {}
    rows: dict[str, Any] = {}
    for model_key, profiles in feedback.items():
        if not isinstance(model_key, str) or not isinstance(profiles, dict):
            continue
        profile_rows: dict[str, Any] = {}
        for profile_id in profiles:
            if not isinstance(profile_id, str):
                continue
            summary = quality_feedback_summary(state, model_key, profile_id, now_epoch=now_epoch)
            if summary["sample_count"]:
                profile_rows[profile_id] = summary
        if profile_rows:
            rows[model_key] = profile_rows
    return rows


def record_quality_feedback(
    state: dict[str, Any],
    feedback: QualityFeedbackInput,
    attribution: dict[str, Any],
    *,
    now_epoch: Callable[[], float],
) -> dict[str, Any]:
    """CAS-write one stable slot, pruning expired evidence only on a successful write."""
    now = now_epoch()
    if not math.isfinite(now):
        raise QualityFeedbackError("feedback_request_unattributable", 422)
    profile_id, model_key, expires_at = _attribution_parts(attribution, now)
    working = dict(state) if isinstance(state, dict) else {}
    if QUALITY_FEEDBACK_STATE_KEY in working:
        working[QUALITY_FEEDBACK_STATE_KEY] = copy.deepcopy(working[QUALITY_FEEDBACK_STATE_KEY])
    _prune_expired_slots(working, now)
    ledger = _ledger_for_write(working, model_key, profile_id)
    slots = ledger["event_slots"]
    slot_key = hashlib.sha256(f"{feedback.request_id}|{feedback.validator}".encode("utf-8")).hexdigest()
    existing = slots.get(slot_key)
    if isinstance(existing, dict):
        revision = _revision(existing)
        if revision is None:
            raise QualityFeedbackError("feedback_revision_conflict", 409)
        if _same_event(existing, feedback):
            # A retry carries the revision it compared against before this write: zero for a
            # create, then the prior revision for a correction. The current revision itself is
            # also an explicit no-op correction. Older revisions stay conflicts.
            if feedback.expected_revision not in {max(0, revision - 1), revision}:
                raise QualityFeedbackError("feedback_revision_conflict", 409)
            _commit(state, working)
            return _write_result("replayed", revision, working, model_key, profile_id, now)
        if feedback.expected_revision != revision:
            raise QualityFeedbackError("feedback_revision_conflict", 409)
        next_revision = revision + 1
        status = "corrected"
    else:
        if feedback.expected_revision != 0:
            raise QualityFeedbackError("feedback_revision_conflict", 409)
        if len(slots) >= QUALITY_FEEDBACK_MAX_SLOTS:
            raise QualityFeedbackError("feedback_capacity_exhausted", 409)
        next_revision = 1
        status = "created"
    slots[slot_key] = {
        "revision": next_revision,
        "outcome": feedback.outcome,
        "severity": feedback.severity,
        "reason": feedback.reason,
        "validator": feedback.validator,
        "recorded_at": _iso_timestamp(now),
        # The expiry belongs to the attributed route, not to its later correction.
        "expires_at": expires_at,
    }
    _commit(state, working)
    return _write_result(status, next_revision, working, model_key, profile_id, now)


def _attribution_parts(attribution: dict[str, Any], now: float) -> tuple[str, str, str]:
    profile = attribution.get("profile") if isinstance(attribution, dict) else None
    source = attribution.get("source") if isinstance(attribution, dict) else None
    upstream_id = attribution.get("upstream_id") if isinstance(attribution, dict) else None
    model_id = attribution.get("model_id") if isinstance(attribution, dict) else None
    timestamp = attribution.get("timestamp") if isinstance(attribution, dict) else None
    if (
        not isinstance(profile, str)
        or not profile
        or not isinstance(source, str)
        or not source
        or not isinstance(upstream_id, str)
        or not upstream_id
        or not isinstance(model_id, str)
        or not model_id
        or isinstance(timestamp, bool)
    ):
        raise QualityFeedbackError("feedback_request_unattributable", 422)
    try:
        request_timestamp = float(timestamp)
    except (TypeError, ValueError):
        raise QualityFeedbackError("feedback_request_unattributable", 422) from None
    if not math.isfinite(request_timestamp) or request_timestamp > now:
        raise QualityFeedbackError("feedback_request_unattributable", 422)
    expiry = request_timestamp + MAX_AGE_SECONDS
    if expiry <= now:
        raise QualityFeedbackError("feedback_request_unattributable", 422)
    return profile, f"{source}::{upstream_id}", _iso_timestamp(expiry)


def _slots_for(state: dict[str, Any], model_key: str, profile_id: str) -> dict[str, Any]:
    feedback = state.get(QUALITY_FEEDBACK_STATE_KEY) if isinstance(state, dict) else None
    models = feedback.get(model_key) if isinstance(feedback, dict) else None
    ledger = models.get(profile_id) if isinstance(models, dict) else None
    slots = ledger.get("event_slots") if isinstance(ledger, dict) else None
    return slots if isinstance(slots, dict) else {}


def _ledger_for_write(state: dict[str, Any], model_key: str, profile_id: str) -> dict[str, Any]:
    feedback = state.setdefault(QUALITY_FEEDBACK_STATE_KEY, {})
    if not isinstance(feedback, dict):
        feedback = {}
        state[QUALITY_FEEDBACK_STATE_KEY] = feedback
    models = feedback.setdefault(model_key, {})
    if not isinstance(models, dict):
        models = {}
        feedback[model_key] = models
    ledger = models.setdefault(profile_id, {})
    if not isinstance(ledger, dict):
        ledger = {}
        models[profile_id] = ledger
    slots = ledger.setdefault("event_slots", {})
    if not isinstance(slots, dict):
        ledger["event_slots"] = {}
    return ledger


def _prune_expired_slots(state: dict[str, Any], now: float) -> None:
    feedback = state.get(QUALITY_FEEDBACK_STATE_KEY)
    if not isinstance(feedback, dict):
        return
    for model_key, profiles in list(feedback.items()):
        if not isinstance(profiles, dict):
            feedback.pop(model_key, None)
            continue
        for profile_id, ledger in list(profiles.items()):
            slots = ledger.get("event_slots") if isinstance(ledger, dict) else None
            if not isinstance(slots, dict):
                profiles.pop(profile_id, None)
                continue
            active = {
                slot_key: row
                for slot_key, row in slots.items()
                if isinstance(row, dict)
                and (expires_at := parse_iso_timestamp(row.get("expires_at"))) is not None
                and expires_at > now
            }
            if active:
                ledger["event_slots"] = active
            else:
                profiles.pop(profile_id, None)
        if not profiles:
            feedback.pop(model_key, None)
    if not feedback:
        state.pop(QUALITY_FEEDBACK_STATE_KEY, None)


def _same_event(existing: dict[str, Any], feedback: QualityFeedbackInput) -> bool:
    return all(
        existing.get(key) == value
        for key, value in (
            ("outcome", feedback.outcome),
            ("severity", feedback.severity),
            ("reason", feedback.reason),
            ("validator", feedback.validator),
        )
    )


def _revision(row: dict[str, Any]) -> int | None:
    revision = row.get("revision")
    return revision if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 1 else None


def _write_result(status: str, revision: int, state: dict[str, Any], model_key: str, profile_id: str, now: float) -> dict[str, Any]:
    return {
        "status": status,
        "revision": revision,
        "quality": quality_feedback_summary(state, model_key, profile_id, now_epoch=lambda: now),
    }


def _commit(target: dict[str, Any], source: dict[str, Any]) -> None:
    target.clear()
    target.update(source)


def _empty_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "quality_adjustment": 0.0,
        "last_reason": None,
        "last_severity": None,
        "last_recorded_at": None,
        "status": "none",
    }


def _iso_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()
