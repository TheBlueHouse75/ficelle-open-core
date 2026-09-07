from __future__ import annotations

import copy
import hashlib

import pytest

from ficelle import request_log
from ficelle.use_cases.quality_feedback import (
    QUALITY_FEEDBACK_MAX_SLOTS,
    QualityFeedbackError,
    parse_quality_feedback_payload,
    quality_feedback_summary,
    record_quality_feedback,
)


NOW = 1_000_000.0


def _attribution(*, timestamp: float = NOW - 10) -> dict[str, object]:
    return {
        "profile": "ficelle/auto-orchestrator",
        "source": "openrouter",
        "upstream_id": "google/gemma",
        "model_id": "ficelle/openrouter/gemma:free",
        "timestamp": timestamp,
    }


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "request-123",
        "outcome": "fail",
        "severity": "major",
        "reason": "invalid_json",
        "validator": "operator.review",
        "expected_revision": 0,
    }
    payload.update(overrides)
    return payload


def _record(state: dict[str, object], payload: dict[str, object], *, now: float = NOW) -> dict[str, object]:
    return record_quality_feedback(
        state,
        parse_quality_feedback_payload(payload),
        _attribution(),
        now_epoch=lambda: now,
    )


def test_feedback_payload_is_closed_and_pass_uses_none_severity() -> None:
    parsed = parse_quality_feedback_payload(_payload(outcome="pass", severity="none", reason="empty_output"))

    assert parsed.outcome == "pass"
    assert parsed.severity == "none"

    for invalid in (
        _payload(extra="caller supplied attribution"),
        _payload(outcome="pass", severity="minor"),
        _payload(outcome="fail", severity="none"),
        _payload(validator="operator review"),
        _payload(reason="free text is forbidden"),
        _payload(expected_revision=True),
    ):
        with pytest.raises(QualityFeedbackError, match="invalid_quality_feedback"):
            parse_quality_feedback_payload(invalid)


def test_event_slots_are_idempotent_cas_replacements_without_raw_request_ids() -> None:
    unrelated = {"nested": ["must-not-be-copied"]}
    state: dict[str, object] = {"unrelated": unrelated}
    created = _record(state, _payload())
    before_replay = copy.deepcopy(state)

    replayed = _record(state, _payload())
    corrected = _record(
        state,
        _payload(outcome="pass", severity="none", reason="empty_output", expected_revision=1),
    )
    before_conflict = copy.deepcopy(state)

    assert created["status"] == "created"
    assert created["revision"] == 1
    assert state["unrelated"] is unrelated
    assert replayed == {**created, "status": "replayed"}
    assert before_replay == {
        "unrelated": {"nested": ["must-not-be-copied"]},
        "quality_feedback": {
            "openrouter::google/gemma": {
                "ficelle/auto-orchestrator": {
                    "event_slots": {
                        hashlib.sha256(b"request-123|operator.review").hexdigest(): {
                            "revision": 1,
                            "outcome": "fail",
                            "severity": "major",
                            "reason": "invalid_json",
                            "validator": "operator.review",
                            "recorded_at": "1970-01-12T13:46:40+00:00",
                            "expires_at": "1970-02-11T13:46:30+00:00",
                        }
                    }
                }
            }
        }
    }
    assert "request-123" not in repr(state)
    assert corrected["status"] == "corrected"
    assert corrected["revision"] == 2
    assert _record(
        state,
        _payload(outcome="pass", severity="none", reason="empty_output", expected_revision=1),
    )["status"] == "replayed"
    assert len(
        state["quality_feedback"]["openrouter::google/gemma"]["ficelle/auto-orchestrator"]["event_slots"]
    ) == 1

    with pytest.raises(QualityFeedbackError, match="feedback_revision_conflict"):
        _record(state, _payload(reason="tool_loop", expected_revision=1))
    assert state == before_conflict
    assert state["unrelated"] is unrelated


def test_expired_attribution_is_rejected_and_expired_slots_are_pruned_before_capacity_check() -> None:
    state: dict[str, object] = {}
    expired_attribution = _attribution(timestamp=NOW - request_log.MAX_AGE_SECONDS - 1)
    with pytest.raises(QualityFeedbackError, match="feedback_request_unattributable"):
        record_quality_feedback(
            state,
            parse_quality_feedback_payload(_payload()),
            expired_attribution,
            now_epoch=lambda: NOW,
        )
    assert state == {}

    slot_key = hashlib.sha256(b"expired-request|operator.expired").hexdigest()
    active_slots = {
        f"active-{index}": {
            "revision": 1,
            "outcome": "pass",
            "severity": "none",
            "reason": "empty_output",
            "validator": f"operator.active-{index}",
            "recorded_at": "1970-01-12T13:46:40+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
        for index in range(QUALITY_FEEDBACK_MAX_SLOTS - 1)
    }
    state = {
        "quality_feedback": {
            "openrouter::google/gemma": {
                "ficelle/auto-orchestrator": {
                    "event_slots": {
                        slot_key: {
                            "revision": 1,
                            "outcome": "fail",
                            "severity": "critical",
                            "reason": "unsafe_action",
                            "validator": "operator.expired",
                            "recorded_at": "1970-01-01T00:00:00+00:00",
                            "expires_at": "1970-01-01T00:00:01+00:00",
                        },
                        **active_slots,
                    }
                }
            }
        }
    }

    result = _record(state, _payload(request_id="request-new", validator="operator.new"))

    slots = state["quality_feedback"]["openrouter::google/gemma"]["ficelle/auto-orchestrator"]["event_slots"]
    assert slot_key not in slots
    assert hashlib.sha256(b"request-new|operator.new").hexdigest() in slots
    assert result["status"] == "created"
    assert len(slots) == QUALITY_FEEDBACK_MAX_SLOTS
    assert sum(slot["expires_at"] > "1970-01-01T00:00:00+00:00" for slot in slots.values()) == QUALITY_FEEDBACK_MAX_SLOTS


def test_capacity_rejects_new_slot_without_mutation_but_allows_replay_and_correction() -> None:
    expires_at = "2099-01-01T00:00:00+00:00"
    slots = {
        f"slot-{index}": {
            "revision": 1,
            "outcome": "pass",
            "severity": "none",
            "reason": "empty_output",
            "validator": f"validator.{index}",
            "recorded_at": "1970-01-01T00:00:00+00:00",
            "expires_at": expires_at,
        }
        for index in range(QUALITY_FEEDBACK_MAX_SLOTS - 1)
    }
    stable_key = hashlib.sha256(b"request-123|operator.review").hexdigest()
    slots[stable_key] = {
        "revision": 1,
        "outcome": "fail",
        "severity": "major",
        "reason": "invalid_json",
        "validator": "operator.review",
        "recorded_at": "1970-01-01T00:00:00+00:00",
        "expires_at": expires_at,
    }
    state: dict[str, object] = {
        "quality_feedback": {"openrouter::google/gemma": {"ficelle/auto-orchestrator": {"event_slots": slots}}}
    }
    before = copy.deepcopy(state)

    with pytest.raises(QualityFeedbackError, match="feedback_capacity_exhausted"):
        _record(state, _payload(validator="operator.other"))
    assert state == before

    assert _record(state, _payload())["status"] == "replayed"
    assert _record(state, _payload(reason="tool_loop", expected_revision=1))["status"] == "corrected"


def test_quality_summary_uses_seven_day_half_life_and_redacts_slots() -> None:
    state: dict[str, object] = {}
    _record(state, _payload())
    _record(
        state,
        _payload(
            request_id="request-456",
            outcome="pass",
            severity="none",
            reason="empty_output",
            validator="operator.pass",
        ),
    )

    current = quality_feedback_summary(
        state,
        "openrouter::google/gemma",
        "ficelle/auto-orchestrator",
        now_epoch=lambda: NOW,
    )
    week_later = quality_feedback_summary(
        state,
        "openrouter::google/gemma",
        "ficelle/auto-orchestrator",
        now_epoch=lambda: NOW + 7 * 86_400,
    )

    assert current == {
        "sample_count": 2,
        "quality_adjustment": -11.0,
        "last_reason": "empty_output",
        "last_severity": "none",
        "last_recorded_at": "1970-01-12T13:46:40+00:00",
        "status": "pass",
    }
    assert week_later["sample_count"] == 2
    assert week_later["quality_adjustment"] == -5.5
    assert "event_slots" not in current
    assert "request-123" not in repr(current)


def test_quality_summary_ignores_non_string_outcome_and_severity() -> None:
    state: dict[str, object] = {
        "quality_feedback": {
            "openrouter::google/gemma": {
                "ficelle/auto-orchestrator": {
                    "event_slots": {
                        "list-outcome": {
                            "revision": 1,
                            "outcome": [],
                            "severity": {},
                            "reason": "invalid_json",
                            "validator": "operator.review",
                            "recorded_at": "1970-01-12T13:46:40+00:00",
                            "expires_at": "2099-01-01T00:00:00+00:00",
                        },
                        "dict-outcome": {
                            "revision": 1,
                            "outcome": {},
                            "severity": [],
                            "reason": "invalid_json",
                            "validator": "operator.review",
                            "recorded_at": "1970-01-12T13:46:40+00:00",
                            "expires_at": "2099-01-01T00:00:00+00:00",
                        },
                    }
                }
            }
        }
    }

    assert quality_feedback_summary(
        state,
        "openrouter::google/gemma",
        "ficelle/auto-orchestrator",
        now_epoch=lambda: NOW,
    ) == {
        "sample_count": 0,
        "quality_adjustment": 0.0,
        "last_reason": None,
        "last_severity": None,
        "last_recorded_at": None,
        "status": "none",
    }
