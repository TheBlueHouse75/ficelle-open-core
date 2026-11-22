from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ficelle import request_log as rl


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _line(request_id: str, *, now: float, offset: float = 0.0, **overrides) -> dict:
    row = {
        "request_id": request_id,
        "requested_model": "ficelle/auto-orchestrator",
        "selected_model": "ficelle/openrouter/gemma:free",
        "selected_upstream": "google/gemma",
        "selected_source": "openrouter",
        "competence": "verified",
        "final_status": 200,
        "final_reason": "ok",
        "candidate_count": 3,
        "attempt_count": 1,
        "attempts": [{"model": "ficelle/openrouter/gemma:free", "source": "openrouter", "status": 200, "reason": "ok", "latency_seconds": 1.0}],
        "duration_seconds": 1.2,
        "stream": False,
        "compression": None,
        "logged_at": _iso(now - offset),
    }
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count(db: Path, table: str = "requests") -> int:
    """Row count straight from the DB — does not trigger ingestion (unlike query())."""
    with sqlite3.connect(db) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


FIXTURES = Path(__file__).parent / "fixtures" / "request_log"


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "routes.jsonl", tmp_path / "requests.sqlite"


def test_ingest_maps_fields_and_orders_newest_first(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("r1", now=now, offset=10),
        _line("r2", now=now, offset=20, requested_model="ficelle/auto-tools", selected_source=None, selected_model=None,
              selected_upstream=None, final_status=502, final_reason="upstream_failure", attempt_count=2,
              attempts=[{"model": "a", "source": "s1", "status": 429, "reason": "rate_limited"},
                        {"model": "b", "source": "s2", "status": "timeout", "reason": "timeout", "error_type": "Timeout", "latency_seconds": 5.0}],
              duration_seconds=7.3),
    ])
    rows = rl.query(store_path=db, source_path=log, limit=10)
    assert [r["request_id"] for r in rows] == ["r1", "r2"]  # newest (r1) first
    r2 = next(r for r in rows if r["request_id"] == "r2")
    assert r2["error_type"] == "Timeout"
    assert r2["fallback"] is True
    assert r2["status"] == 502
    assert r2["reason"] == "upstream_failure"
    assert len(r2["attempts"]) == 2
    r1 = next(r for r in rows if r["request_id"] == "r1")
    assert r1["fallback"] is False
    assert r1["source"] == "openrouter"


def test_ingestion_is_idempotent(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now), _line("r2", now=now)])
    rl.query(store_path=db, source_path=log)
    rl.query(store_path=db, source_path=log)
    _write(log, [_line("r3", now=now)])
    rl.query(store_path=db, source_path=log)
    assert _count(db) == 3


def test_quality_feedback_attribution_uses_exact_request_id_and_only_final_successes(paths):
    log, db = paths
    now = time.time()
    _write(
        log,
        [
            _line("request-good", now=now),
            _line("request-failed", now=now, final_status=502, final_reason="upstream_failure"),
            _line("request-unattributed", now=now, selected_source=None, selected_upstream=None, selected_model=None),
        ],
    )

    attributed = rl.resolve_quality_feedback_attribution(
        "request-good",
        store_path=db,
        source_path=log,
        now=now,
    )

    assert attributed == {
        "profile": "ficelle/auto-orchestrator",
        "source": "openrouter",
        "upstream_id": "google/gemma",
        "model_id": "ficelle/openrouter/gemma:free",
        "timestamp": pytest.approx(now),
        # Only rows logged since the routing-time base profile was recorded carry one.
        "policy_profile": None,
    }
    # This is deliberately not a list-filter search: wildcard-looking input remains a literal
    # request id and cannot attribute a neighboring route.
    assert rl.resolve_quality_feedback_attribution("request-%", store_path=db, source_path=log, now=now) is None
    assert rl.resolve_quality_feedback_attribution("request-failed", store_path=db, source_path=log, now=now) is None
    assert rl.resolve_quality_feedback_attribution("request-unattributed", store_path=db, source_path=log, now=now) is None


def test_delivery_update_corrects_a_logged_success_without_rewriting_history(paths):
    log, db = paths
    now = time.time()
    _write(
        log,
        [
            _line("r1", now=now),
            {
                "event": "request_delivery_update",
                "request_id": "r1",
                "final_reason": "client_disconnected",
                "error_type": "BrokenPipeError",
                "logged_at": _iso(now),
            },
        ],
    )

    row = rl.query(store_path=db, source_path=log)[0]
    totals = rl.summary(store_path=db, source_path=log, now=now)

    assert row["status"] == 200
    assert row["reason"] == "client_disconnected"
    assert row["error_type"] == "BrokenPipeError"
    assert totals["total"] == 1
    assert totals["ok"] == 0


def test_live_tail_requests_a_resync_after_a_delivery_update(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now)])
    cursor = rl.tail(store_path=db, source_path=log)["cursor"]
    _write(
        log,
        [
            {
                "event": "request_delivery_update",
                "request_id": "r1",
                "final_reason": "client_disconnected",
                "error_type": "ConnectionResetError",
                "logged_at": _iso(now),
            }
        ],
    )

    update = rl.tail(cursor=cursor, store_path=db, source_path=log)

    assert update == {"cursor": cursor + 1, "entries": [], "resync": True}


def test_live_tail_keeps_delivery_resync_after_another_read_ingested_it(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now)])
    cursor = rl.tail(store_path=db, source_path=log)["cursor"]
    _write(
        log,
        [
            {
                "event": "request_delivery_update",
                "request_id": "r1",
                "final_reason": "client_disconnected",
                "error_type": "BrokenPipeError",
                "logged_at": _iso(now),
            }
        ],
    )

    # Summary polling can win the ingestion race with the Requests SSE subscriber.
    rl.summary(store_path=db, source_path=log, now=now)
    update = rl.tail(cursor=cursor, store_path=db, source_path=log)

    assert update == {"cursor": cursor + 1, "entries": [], "resync": True}


def test_partial_trailing_line_is_not_ingested_until_complete(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now)])
    # Append a partial line (a concurrent writer mid-append): no trailing newline.
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id":"r2","final_reason":"ok","attempt_count":1,"attempts":[]')
    assert [r["request_id"] for r in rl.query(store_path=db, source_path=log)] == ["r1"]
    # Complete the line; the offset stayed before it, so it ingests now.
    with log.open("a", encoding="utf-8") as handle:
        handle.write("}\n")
    assert sorted(r["request_id"] for r in rl.query(store_path=db, source_path=log)) == ["r1", "r2"]


def test_corrupted_line_is_skipped(paths):
    log, db = paths
    now = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_line("r1", now=now)) + "\n")
        handle.write("this is not json\n")
        handle.write(json.dumps(_line("r2", now=now)) + "\n")
    assert sorted(r["request_id"] for r in rl.query(store_path=db, source_path=log)) == ["r1", "r2"]


def test_truncation_or_rotation_rebuilds_without_duplicates(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now), _line("r2", now=now)])
    rl.query(store_path=db, source_path=log)
    # External rotation: file replaced by a smaller one (size < stored offset).
    log.write_text(json.dumps(_line("r3", now=now)) + "\n", encoding="utf-8")
    rows = sorted(r["request_id"] for r in rl.query(store_path=db, source_path=log))
    assert rows == ["r1", "r2", "r3"]  # old rows kept (INSERT OR IGNORE), new row added


def test_rows_appended_between_the_last_ingest_and_a_rotation_are_still_indexed(paths):
    # Ingestion is lazy and the dashboard is usually closed, so a 32 MB rotation can fire between
    # two ingests. The previous live file is then `routes.jsonl.1`; reading only the new file
    # silently dropped everything written since the last visit.
    from ficelle.json_store import rotate_if_oversized

    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now, offset=60), _line("r2", now=now, offset=50)])
    rl.query(store_path=db, source_path=log)
    _write(log, [_line("r3", now=now, offset=40), _line("r4", now=now, offset=30)])
    rotate_if_oversized(log, 1)
    _write(log, [_line("r5", now=now, offset=20)])

    ids = {row["request_id"] for row in rl.query(store_path=db, source_path=log, limit=10)}

    assert ids == {"r1", "r2", "r3", "r4", "r5"}
    # Re-ingesting nothing new keeps the index stable: the rotated segment is not read again.
    assert _count(db) == 5
    assert rl.summary(store_path=db, source_path=log, now=now)["total"] == 5


def test_rows_survive_several_rotations_between_ingests_in_order(paths):
    from ficelle.json_store import rotate_if_oversized

    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now, offset=90)])
    rl.query(store_path=db, source_path=log)
    _write(log, [_line("r2", now=now, offset=80)])
    rotate_if_oversized(log, 1)
    _write(log, [_line("r3", now=now, offset=70)])
    rotate_if_oversized(log, 1)
    _write(log, [_line("r4", now=now, offset=60)])

    ids = {row["request_id"] for row in rl.query(store_path=db, source_path=log, limit=10)}

    assert ids == {"r1", "r2", "r3", "r4"}


def test_segments_still_kept_are_indexed_after_the_indexed_file_rotated_out_of_retention(paths):
    # Four rotations between two visits push the indexed file past `.3`: its inode matched no
    # segment, so the three surviving segments were skipped for good.
    from ficelle.json_store import rotate_if_oversized

    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now, offset=90)])
    rl.query(store_path=db, source_path=log)
    for index in range(2, 6):
        rotate_if_oversized(log, 1)
        _write(log, [_line(f"r{index}", now=now, offset=90 - index * 10)])

    ids = {row["request_id"] for row in rl.query(store_path=db, source_path=log, limit=10)}

    # r1 was indexed before; r2..r4 sit in `.3`..`.1` and r5 in the live file.
    assert ids == {"r1", "r2", "r3", "r4", "r5"}


def test_a_reused_inode_does_not_pass_for_the_indexed_file(paths):
    # ext4 hands the inode freed by a rotation to the next file: the stored `dev:ino` then names
    # the new live file, and its rows (and every rotated segment) were silently skipped.
    from ficelle.json_store import rotate_if_oversized

    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now, offset=90)])
    rl.query(store_path=db, source_path=log)
    for index in range(2, 6):
        rotate_if_oversized(log, 1)
        _write(log, [_line(f"r{index}", now=now, offset=90 - index * 10)])
    live = log.stat()
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'ingest_source_identity'",
            (f"{log.resolve()}:{live.st_dev}:{live.st_ino}",),
        )

    ids = {row["request_id"] for row in rl.query(store_path=db, source_path=log, limit=10)}

    assert ids == {"r1", "r2", "r3", "r4", "r5"}


def test_a_schema_rebuild_backfills_the_rotated_segments_too(paths):
    # A version bump drops the derived table; reading only the live file lost every row already
    # rotated to `routes.jsonl.N`, although those are still inside the retention window.
    from ficelle.json_store import rotate_if_oversized

    log, db = paths
    now = time.time()
    streamed = {"stream": True, "attempts": [_stream_attempt("openrouter", 0.5)]}
    _write(log, [_line("r1", now=now, offset=90, **streamed)])
    rotate_if_oversized(log, 1)
    _write(log, [_line("r2", now=now, offset=80, **streamed)])
    rotate_if_oversized(log, 1)
    _write(log, [_line("r3", now=now, offset=70, **streamed)])
    rl.query(store_path=db, source_path=log)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(rl.SCHEMA_VERSION - 1),))

    ids = {row["request_id"] for row in rl.query(store_path=db, source_path=log, limit=10)}

    assert ids == {"r1", "r2", "r3"}
    # The per-attempt rows are rebuilt with them, once: the replay must not double the TTFT sample.
    assert rl.summary(store_path=db, source_path=log, now=now)["ttft"]["overall"]["n"] == 3


def test_a_rotated_segment_that_was_never_the_indexed_log_is_not_ingested(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now)])
    rl.query(store_path=db, source_path=log)
    # A leftover `.1` from some other lineage has a different inode than the file we indexed.
    _write(log.with_name("routes.jsonl.1"), [_line("foreign", now=now)])
    # Created before the old file is replaced, so the new log cannot reuse the old inode.
    replacement = log.with_name("routes.jsonl.new")
    _write(replacement, [_line("r2", now=now)])
    replacement.replace(log)

    ids = {row["request_id"] for row in rl.query(store_path=db, source_path=log, limit=10)}

    assert ids == {"r1", "r2"}


def test_source_switch_resets_offset_and_ingests_canonical_from_start(tmp_path):
    legacy_log = tmp_path / "legacy" / "routes.jsonl"
    canonical_log = tmp_path / "canonical" / "routes.jsonl"
    db = tmp_path / "canonical" / "requests.sqlite"
    now = time.time()
    _write(legacy_log, [_line("legacy-1", now=now), _line("legacy-2", now=now)])

    rl.query(store_path=db, source_path=legacy_log)
    _write(
        canonical_log,
        [
            _line("canonical-1", now=now, selected_upstream="canonical/one"),
            _line("canonical-2", now=now, selected_upstream="canonical/two"),
            _line("canonical-3", now=now, selected_upstream="canonical/three"),
        ],
    )

    rows = rl.query(store_path=db, source_path=canonical_log, limit=10)

    assert {row["request_id"] for row in rows} == {
        "legacy-1",
        "legacy-2",
        "canonical-1",
        "canonical-2",
        "canonical-3",
    }
    with sqlite3.connect(db) as conn:
        source_identity = conn.execute(
            "SELECT value FROM meta WHERE key = 'ingest_source_identity'"
        ).fetchone()[0]
    assert str(canonical_log.resolve()) in source_identity


def test_filters(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("r1", now=now),
        _line("r2", now=now, requested_model="ficelle/auto-tools", selected_source="nous",
              final_status=502, final_reason="upstream_failure"),
        _line("r3", now=now, requested_model="ficelle/auto-tools", selected_upstream="x/y", final_status=503,
              final_reason="no_available_model"),
    ])
    q = lambda **kw: sorted(r["request_id"] for r in rl.query(store_path=db, source_path=log, **kw))
    assert q(profile="ficelle/auto-tools") == ["r2", "r3"]
    assert q(source="nous") == ["r2"]
    assert q(reason="ok") == ["r1"]
    assert q(status="502") == ["r2"]
    assert q(q="y") == ["r3"]  # upstream_id LIKE %y%


def test_failed_run_is_attributed_to_its_last_attempt(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        # No candidate delivered, so the router wrote no `selected_*`: the row must still name the
        # provider and model that produced final_reason, or every failure is invisible in the
        # provider column, the provider filter, the search box and the per-provider summary.
        _line("fail", now=now, selected_source=None, selected_model=None, selected_upstream=None,
              final_status=400, final_reason="bad_upstream_request", attempt_count=1,
              attempts=[{"model": "ficelle/openrouter/gemma:free", "upstream": "google/gemma-4",
                         "source": "openrouter", "status": 400, "reason": "bad_upstream_request"}]),
        # Refused before any attempt: nothing to attribute it to, so the columns stay empty.
        _line("early", now=now, selected_source=None, selected_model=None, selected_upstream=None,
              final_status=400, final_reason="malformed_tool_call", attempt_count=0, attempts=[]),
    ])
    rows = {r["request_id"]: r for r in rl.query(store_path=db, source_path=log)}
    assert rows["fail"]["source"] == "openrouter"
    assert rows["fail"]["upstream_id"] == "google/gemma-4"
    assert rows["fail"]["model_id"] == "ficelle/openrouter/gemma:free"
    assert rows["early"]["source"] is None
    assert rows["early"]["upstream_id"] is None
    assert rows["early"]["model_id"] is None
    assert [r["request_id"] for r in rl.query(store_path=db, source_path=log, source="openrouter")] == ["fail"]
    by_source = {r["source"]: r["total"] for r in rl.summary(store_path=db, source_path=log, now=now)["by_source"]}
    assert by_source["openrouter"] == 1


def test_a_failed_run_is_never_attributed_to_an_attempt_that_succeeded(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        # A Fusion panel: its attempts carry the successful panelists too, in completion order, so
        # the LAST one is the slow model that answered correctly. Charging it with the run's
        # failure would blame the only provider that worked.
        _line("panel", now=now, requested_model="ficelle/auto-fusion", selected_source=None,
              selected_model=None, selected_upstream=None, final_status=502,
              final_reason="insufficient_panel_success", attempt_count=3,
              attempts=[{"model": "a", "upstream": "groq/a", "source": "groq", "status": 429, "reason": "rate_limited"},
                        {"model": "b", "upstream": "nvidia/b", "source": "nvidia", "status": 400, "reason": "bad_upstream_request"},
                        {"model": "c", "upstream": "or/c", "source": "openrouter", "status": 200, "reason": "ok"}]),
        # Every attempt succeeded yet the run failed (a synth stage with no candidate to run):
        # no model is at fault, so the row names none.
        _line("synth", now=now, requested_model="ficelle/auto-fusion", selected_source=None,
              selected_model=None, selected_upstream=None, final_status=502,
              final_reason="synthesizer_failed", attempt_count=1,
              attempts=[{"model": "judge", "upstream": "x/judge", "source": "cerebras", "status": 200, "reason": "ok"}]),
        # A stage that did fail is named even when the terminal verdict came from a later stage
        # that never ran: the column answers "who failed on this request", and the row's own
        # `reason` already says what ended it.
        _line("synth-unavailable", now=now, requested_model="ficelle/auto-fusion",
              selected_source=None, selected_model=None, selected_upstream=None,
              final_status=502, final_reason="synthesizer_failed", attempt_count=2,
              attempts=[{"model": "panel-bad", "upstream": "x/panel-bad", "source": "groq",
                         "status": 503, "reason": "server_error"},
                        {"model": "judge", "upstream": "x/judge", "source": "cerebras",
                         "status": 200, "reason": "ok"}],
              fusion={"degraded_flags": {"synthesizer_unavailable": True}}),
        # If the synthesizer did run and fail, its attempt remains the correct attribution.
        _line("synth-failed", now=now, requested_model="ficelle/auto-fusion",
              selected_source=None, selected_model=None, selected_upstream=None,
              final_status=502, final_reason="synthesizer_failed", attempt_count=2,
              attempts=[{"model": "panel", "upstream": "x/panel", "source": "groq",
                         "status": 200, "reason": "ok"},
                        {"model": "synth", "upstream": "x/synth", "source": "nvidia",
                         "status": 503, "reason": "server_error"}],
              fusion={"degraded_flags": {}}),
    ])
    rows = {r["request_id"]: r for r in rl.query(store_path=db, source_path=log)}
    assert rows["panel"]["source"] == "nvidia"
    assert rows["panel"]["upstream_id"] == "nvidia/b"
    assert rows["synth"]["source"] is None
    assert rows["synth"]["upstream_id"] is None
    assert rows["synth-unavailable"]["source"] == "groq"
    assert rows["synth-unavailable"]["upstream_id"] == "x/panel-bad"
    assert rows["synth-failed"]["source"] == "nvidia"
    assert rows["synth-failed"]["upstream_id"] == "x/synth"
    by_source = {r["source"]: r["total"] for r in rl.summary(store_path=db, source_path=log, now=now)["by_source"]}
    assert "openrouter" not in by_source
    assert "cerebras" not in by_source
    assert by_source["nvidia"] == 2


def test_attempt_without_an_explicit_failure_reason_is_not_used_for_attribution(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("prior-failure", now=now, selected_source=None, selected_model=None,
              selected_upstream=None, final_status=502, final_reason="upstream_failure",
              attempt_count=3,
              attempts=[{"model": "failed", "upstream": "nous/failed", "source": "nous",
                         "status": 503, "reason": "server_error"},
                        {"model": "unknown", "upstream": "other/unknown", "source": "other",
                         "status": 200},
                        {"ignored_by_whitelist": True}]),
        _line("unknown-only", now=now, selected_source=None, selected_model=None,
              selected_upstream=None, final_status=502, final_reason="upstream_failure",
              attempt_count=1,
              attempts=[{"model": "unknown", "upstream": "other/unknown", "source": "other",
                         "status": 200}]),
    ])

    rows = {r["request_id"]: r for r in rl.query(store_path=db, source_path=log)}
    assert rows["prior-failure"]["source"] == "nous"
    assert rows["prior-failure"]["upstream_id"] == "nous/failed"
    assert rows["prior-failure"]["model_id"] == "failed"
    assert rows["unknown-only"]["source"] is None
    assert rows["unknown-only"]["upstream_id"] is None
    assert rows["unknown-only"]["model_id"] is None


def test_a_recorded_selection_always_wins_over_the_attempt_fallback(paths):
    log, db = paths
    now = time.time()
    # A run that failed over and then succeeded: `selected_*` names the winner, and the failed
    # attempt must not be what the row reports.
    _write(log, [
        _line("won", now=now, attempt_count=2,
              attempts=[{"model": "other", "upstream": "other/model", "source": "nous", "status": 429, "reason": "rate_limited"},
                        {"model": "ficelle/openrouter/gemma:free", "source": "openrouter", "status": 200, "reason": "ok"}]),
    ])
    row = rl.query(store_path=db, source_path=log)[0]
    assert row["source"] == "openrouter"
    assert row["upstream_id"] == "google/gemma"
    assert row["model_id"] == "ficelle/openrouter/gemma:free"


def test_summary_success_is_reason_ok_not_http_status(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("ok1", now=now, duration_seconds=1.0),
        # mid_stream_failure returns HTTP 200 but is an error — must not count as success.
        _line("mid", now=now, final_status=200, final_reason="mid_stream_failure", stream=True, duration_seconds=3.0),
        _line("err", now=now, final_status=502, final_reason="upstream_failure", duration_seconds=7.0),
    ])
    s = rl.summary(store_path=db, source_path=log, window_seconds=86400, now=now)
    assert s["total"] == 3
    assert s["ok"] == 1
    assert s["errors"] == 2
    assert s["success_rate"] == round(1 / 3, 4)
    assert s["latency_p50"] is not None and s["latency_p95"] is not None
    assert len(s["timeline"]) >= 1
    # by_reason drives the Outcome filter, so it must carry the terminal reasons.
    reasons = {r["reason"]: r for r in s["by_reason"]}
    assert reasons["ok"]["total"] == 1
    assert "upstream_failure" in reasons and "mid_stream_failure" in reasons
    statuses = {r["status"] for r in s["by_status"]}
    assert {200, 502} <= statuses


def test_summary_timeline_covers_every_slot_including_idle_ones(paths):
    log, db = paths
    now = time.time()
    # Two requests half an hour apart, so the middle of the window is genuinely idle.
    _write(log, [
        _line("early", now=now, offset=1800),
        _line("late", now=now),
    ])
    s = rl.summary(store_path=db, source_path=log, window_seconds=3600, now=now)
    bucket = s["bucket_seconds"]
    starts = [row["bucket"] for row in s["timeline"]]
    # A windowed series is defined by its window: contiguous slots, no holes.
    assert starts == list(range(starts[0], starts[-1] + bucket, bucket))
    assert starts[-1] == int(now // bucket) * bucket
    assert starts[0] == int((now - 3600) // bucket) * bucket
    # The timeline and the window totals share one cutoff, so they must never drift.
    assert sum(row["total"] for row in s["timeline"]) == s["total"] == 2
    assert any(row["total"] == 0 for row in s["timeline"]), "idle slots must be reported as zeros"


def test_summary_timeline_is_all_zeros_when_nothing_was_logged(paths):
    log, db = paths
    now = time.time()
    _write(log, [])
    s = rl.summary(store_path=db, source_path=log, window_seconds=3600, now=now)
    assert s["total"] == 0
    # The admin view keys its "no requests yet" empty state off the counts, not the
    # length — so assert the slots are actually there before asserting they are zero.
    assert len(s["timeline"]) == 3600 // s["bucket_seconds"] + 1
    assert all(row["total"] == 0 for row in s["timeline"])


def test_summary_timeline_uses_floor_buckets_before_unix_epoch(paths):
    log, db = paths
    reference = -1.0
    cutoff = reference - 60
    # Insert directly because catch-up retention correctly removes 1969 log rows.
    rl.summary(store_path=db, source_path=log, window_seconds=60, now=reference)
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO requests (request_id, ts, reason) VALUES (?, ?, 'ok')",
            [("at-cutoff", cutoff), ("at-reference", reference)],
        )

    s = rl.summary(store_path=db, source_path=log, window_seconds=60, now=reference)
    assert s["timeline"] == [{"bucket": -300, "total": 2, "ok": 2}]
    assert sum(row["total"] for row in s["timeline"]) == s["total"] == 2


def test_summary_excludes_rows_after_the_reference_time(paths):
    log, db = paths
    reference = 1_000.0
    cutoff = reference - 300
    rl.summary(store_path=db, source_path=log, window_seconds=300, now=reference)
    with sqlite3.connect(db) as conn:
        conn.executemany(
            """
            INSERT INTO requests (request_id, ts, reason, source, latency_seconds)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("at-cutoff", cutoff, "ok", "inside", 1.0),
                ("at-reference", reference, "failed", "inside", 2.0),
                ("future", reference + 500, "ok", "future", 100.0),
            ],
        )

    s = rl.summary(store_path=db, source_path=log, window_seconds=300, now=reference)
    assert sum(row["total"] for row in s["timeline"]) == s["total"] == 2
    assert s["ok"] == 1
    assert s["by_source"] == [{"source": "inside", "total": 2, "ok": 1}]
    assert s["latency_p50"] == 1.0
    assert s["latency_p95"] == 2.0


def test_summary_latency_percentiles_are_nearest_rank_over_rows_that_have_a_latency(paths):
    log, db = paths
    now = time.time()
    # 20 timed rows (1..20 s, written out of order) plus rows that never recorded a latency,
    # which must neither be counted nor shift the ranks.
    timed = [_line(f"t{index}", now=now, offset=index, duration_seconds=float(21 - index)) for index in range(1, 21)]
    untimed = [_line(f"u{index}", now=now, offset=index, duration_seconds=None) for index in range(1, 6)]
    _write(log, timed + untimed)

    s = rl.summary(store_path=db, source_path=log, now=now)

    assert s["total"] == 25
    assert (s["latency_p50"], s["latency_p95"]) == (10.0, 19.0)


def test_summary_window_excludes_old_rows(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("recent", now=now, offset=60),
        _line("old", now=now, offset=10 * 86400),
    ])
    s = rl.summary(store_path=db, source_path=log, window_seconds=86400, now=now)
    assert s["total"] == 1  # only the recent one is inside the 24h window


def test_retention_by_count(paths, monkeypatch):
    log, db = paths
    now = time.time()
    monkeypatch.setattr(rl, "MAX_ROWS", 3)
    _write(log, [_line(f"r{i}", now=now, offset=i, stream=True, attempts=[_stream_attempt("openrouter", i + 1.0)])
                 for i in range(6)])
    rl.query(store_path=db, source_path=log)
    assert _count(db) == 3
    # The oldest rows go, whatever order they were appended in.
    assert {row["request_id"] for row in rl.query(store_path=db, source_path=log)} == {"r0", "r1", "r2"}
    # Their attempts go with them: a trimmed request must not keep counting in the TTFT.
    ttft = rl.summary(store_path=db, source_path=log, now=now)["ttft"]["overall"]
    assert (ttft["n"], ttft["p95"]) == (3, 3.0)


def test_retention_by_count_trims_only_the_excess_when_already_at_the_cap(paths, monkeypatch):
    log, db = paths
    now = time.time()
    monkeypatch.setattr(rl, "MAX_ROWS", 3)
    _write(log, [_line(f"r{i}", now=now, offset=10 - i) for i in range(3)])
    rl.query(store_path=db, source_path=log)
    _write(log, [_line("r3", now=now, offset=0), _line("r4", now=now, offset=-1)])

    ids = [row["request_id"] for row in rl.query(store_path=db, source_path=log)]

    assert ids == ["r4", "r3", "r2"]


def test_retention_by_age(paths, monkeypatch):
    log, db = paths
    now = time.time()
    monkeypatch.setattr(rl, "MAX_AGE_SECONDS", 100)
    _write(log, [
        _line("fresh", now=now, offset=10),
        _line("stale", now=now, offset=10_000),
    ])
    rl.query(store_path=db, source_path=log)
    assert [r["request_id"] for r in rl.query(store_path=db, source_path=log)] == ["fresh"]
    assert _count(db, "request_attempts") == 1  # the stale request's attempt went with it


def test_only_whitelisted_fields_are_stored(paths):
    log, db = paths
    now = time.time()
    # Source line carries extra fields — top-level AND nested inside an attempt —
    # that must never reach the index.
    _write(log, [_line(
        "r1", now=now,
        prompt="SUPER SECRET PROMPT", messages=[{"role": "user", "content": "SECRET"}],
        authorization="Bearer SECRETTOKEN",
        attempts=[{"model": "m", "reason": "timeout", "error_type": "Timeout",
                   "raw_error": "token=SECRETTOKEN", "headers": {"authorization": "SECRETHDR"}}],
    )])
    rows = rl.query(store_path=db, source_path=log)
    assert rows and "prompt" not in rows[0] and "messages" not in rows[0]
    # The whitelisted attempt keeps only known keys; the nested secret fields are dropped.
    assert rows[0]["attempts"] == [{"model": "m", "reason": "timeout", "error_type": "Timeout"}]
    # Inspect the raw table: columns are the fixed whitelist, no prompt/secret values anywhere.
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)").fetchall()}
        dump = json.dumps([dict(zip([c[0] for c in conn.execute("SELECT * FROM requests LIMIT 1").description],
                                    conn.execute("SELECT * FROM requests LIMIT 1").fetchone()))])
    assert "prompt" not in cols and "messages" not in cols and "authorization" not in cols
    assert "SECRET" not in dump and "SECRETTOKEN" not in dump and "SECRETHDR" not in dump


def test_error_detail_survives_ingestion_on_a_bad_upstream_request_attempt(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line(
        "r1", now=now,
        attempts=[{
            "model": "m",
            "reason": "bad_upstream_request",
            "status": 400,
            "error_detail": "invalid request: missing field [redacted]",
        }],
    )])
    rows = rl.query(store_path=db, source_path=log)
    assert rows[0]["attempts"] == [{
        "model": "m",
        "reason": "bad_upstream_request",
        "status": 400,
        "error_detail": "invalid request: missing field [redacted]",
    }]


def test_prompt_size_route_fields_survive_ingestion(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("sized", now=now, prompt_tokens_estimate=266645, excluded_for_context=4),
        _line("older", now=now, offset=-5.0),
    ])
    rows = {row["request_id"]: row for row in rl.query(store_path=db, source_path=log)}
    assert rows["sized"]["prompt_tokens_estimate"] == 266645
    assert rows["sized"]["excluded_for_context"] == 4
    # A row written before these fields existed stays readable, with nothing invented.
    assert rows["older"]["prompt_tokens_estimate"] is None
    assert rows["older"]["excluded_for_context"] is None


def test_timeout_diagnostics_are_whitelisted_inside_attempts(paths):
    log, db = paths
    now = time.time()
    _write(
        log,
        [
            _line(
                "r1",
                now=now,
                attempts=[
                    {
                        "model": "m",
                        "reason": "request_deadline_exceeded",
                        "read_timeout_seconds": 18.25,
                        "connect_timeout_seconds": 5.0,
                        "read_timeout_source": "request_deadline",
                        "request_budget_remaining_seconds": 20.25,
                        "timeout_phase": "response_headers",
                    }
                ],
            )
        ],
    )

    attempt = rl.query(store_path=db, source_path=log)[0]["attempts"][0]

    assert attempt == {
        "model": "m",
        "reason": "request_deadline_exceeded",
        "read_timeout_seconds": 18.25,
        "connect_timeout_seconds": 5.0,
        "read_timeout_source": "request_deadline",
        "request_budget_remaining_seconds": 20.25,
        "timeout_phase": "response_headers",
    }


def test_completion_chars_survives_ingestion_on_a_runaway_output_attempt(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line(
        "r1", now=now,
        attempts=[{
            "model": "m",
            "reason": "runaway_output",
            "status": 200,
            "stream_started": True,
            "completion_chars": 400123,
        }],
    )])

    attempt = rl.query(store_path=db, source_path=log)[0]["attempts"][0]

    assert attempt == {
        "model": "m",
        "reason": "runaway_output",
        "status": 200,
        "stream_started": True,
        "completion_chars": 400123,
    }


def test_missing_source_file_is_safe(paths):
    log, db = paths
    rows = rl.query(store_path=db, source_path=log)
    assert rows == []
    summary = rl.summary(store_path=db, source_path=log, window_seconds=3600)
    assert summary["total"] == 0


def test_tail_primes_without_replaying_history(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("a", now=now, offset=2), _line("b", now=now, offset=1)])

    primed = rl.tail(store_path=db, source_path=log)

    # Opening the page must not push rows the list already showed.
    assert primed == {"cursor": 2, "entries": [], "resync": False}


def test_tail_returns_only_rows_ingested_after_the_cursor(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("a", now=now, offset=2)])
    primed = rl.tail(store_path=db, source_path=log)
    _write(log, [_line("b", now=now, offset=1), _line("c", now=now)])

    batch = rl.tail(cursor=primed["cursor"], store_path=db, source_path=log)

    assert [row["request_id"] for row in batch["entries"]] == ["b", "c"]  # oldest first
    assert batch["cursor"] == 3
    assert rl.tail(cursor=batch["cursor"], store_path=db, source_path=log)["entries"] == []


def test_tail_cursor_advances_past_rows_the_filter_dropped(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("a", now=now, offset=3)])
    primed = rl.tail(store_path=db, source_path=log, source="openrouter")
    _write(log, [
        _line("other", now=now, offset=2, selected_source="nous"),
        _line("match", now=now, offset=1),
    ])

    batch = rl.tail(cursor=primed["cursor"], store_path=db, source_path=log, source="openrouter")

    # Without the skip the non-matching row would be re-scanned on every poll, forever.
    assert [row["request_id"] for row in batch["entries"]] == ["match"]
    assert batch["cursor"] == 3


def test_tail_caps_one_batch_and_resumes_from_the_last_row(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("a", now=now, offset=3)])
    primed = rl.tail(store_path=db, source_path=log)
    _write(log, [_line("b", now=now, offset=2), _line("c", now=now, offset=1), _line("d", now=now)])

    first = rl.tail(cursor=primed["cursor"], store_path=db, source_path=log, limit=2)
    assert [row["request_id"] for row in first["entries"]] == ["b", "c"]
    assert first["cursor"] == 3

    second = rl.tail(cursor=first["cursor"], store_path=db, source_path=log, limit=2)
    assert [row["request_id"] for row in second["entries"]] == ["d"]


def test_tail_asks_for_a_resync_when_the_subscriber_fell_too_far_behind(paths):
    log, db = paths
    now = time.time()
    total = rl.MAX_TAIL_REPLAY + 5
    _write(log, [_line(f"r{i}", now=now, offset=total - i) for i in range(total)])

    batch = rl.tail(cursor=0, store_path=db, source_path=log)

    assert batch == {"cursor": total, "entries": [], "resync": True}


def test_tail_asks_for_a_resync_when_the_index_was_rebuilt_below_the_cursor(paths):
    log, db = paths
    _write(log, [_line("a", now=time.time())])

    batch = rl.tail(cursor=999, store_path=db, source_path=log)

    assert batch == {"cursor": 1, "entries": [], "resync": True}


def test_source_signature_moves_only_when_the_log_grows(paths):
    log, _db = paths
    assert rl.source_signature(log) is None

    _write(log, [_line("a", now=time.time())])
    first = rl.source_signature(log)
    assert first is not None
    assert rl.source_signature(log) == first

    _write(log, [_line("b", now=time.time())])
    assert rl.source_signature(log) != first


def test_ingest_records_token_usage_and_summary_sums_the_recorded_savings(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        # The reference price is a fact recorded at route time, so the summary is
        # immutable history: no read-time catalog join can change it.
        _line("u1", now=now, offset=10, usage={"prompt_tokens": 1000, "completion_tokens": 500},
              reference_pricing={"prompt": 1e-06, "completion": 2e-06}),
        _line("u2", now=now, offset=20, usage={"prompt_tokens": 200, "completion_tokens": 100},
              selected_model="ficelle/openrouter/unpriced:free"),
        _line("u3", now=now, offset=30),  # no usage reported
    ])

    payload = rl.summary(store_path=db, source_path=log, window_seconds=3600, now=now)

    assert payload["usage"] == {"requests": 2, "prompt_tokens": 1200, "completion_tokens": 600}
    assert payload["savings"]["priced_requests"] == 1
    assert payload["savings"]["unpriced_requests"] == 1
    assert payload["savings"]["estimated_saved_usd"] == pytest.approx(1000 * 1e-06 + 500 * 2e-06, abs=2e-06)
    by_id = {row["request_id"]: row for row in rl.query(store_path=db, source_path=log, limit=10)}
    assert by_id["u1"]["prompt_tokens"] == 1000
    assert by_id["u1"]["completion_tokens"] == 500
    assert by_id["u3"]["prompt_tokens"] is None


def test_ingest_defends_against_poisoned_usage_and_price_values(paths):
    # The success writer clamps at the source, but the ingest must hold on its own:
    # a foreign or hand-edited log line with inf prices or negative counts would
    # otherwise poison the window SUM (and inf makes the payload unserializable).
    log, db = paths
    now = time.time()
    _write(log, [
        _line("p1", now=now, offset=10,
              usage={"prompt_tokens": -50, "completion_tokens": True},
              reference_pricing={"prompt": "Infinity", "completion": True}),
        _line("p2", now=now, offset=20, usage={"prompt_tokens": 10, "completion_tokens": 5},
              reference_pricing={"prompt": 1e-06, "completion": 1e-06}),
    ])

    payload = rl.summary(store_path=db, source_path=log, window_seconds=3600, now=now)

    assert payload["usage"] == {"requests": 1, "prompt_tokens": 10, "completion_tokens": 5}
    assert payload["savings"]["priced_requests"] == 1
    # The micro-dollar floor may understate by up to 1e-06 (here: 14.9999… → 14 µ$) —
    # understating is the direction the estimate is allowed to err in.
    assert payload["savings"]["estimated_saved_usd"] == pytest.approx(1.4e-05)


def test_refusal_facts_and_handler_error_type_survive_ingestion(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line(
            "refused", now=now, final_status=503, final_reason="no_available_model", attempts=[],
            attempt_count=0, candidate_count=0, selected_model=None, selected_upstream=None, selected_source=None,
            refusal={"excluded": {"not_invokable": 20, "cooldown": 1, "bad": "str", "nested": {"a": 1}},
                     "scope": "provider", "block_reason": "cooldown", "prompt": "SECRET"},
            retry_after_seconds=300,
        ),
        _line(
            "crashed", now=now, offset=1.0, final_status=500, final_reason="internal_error", attempts=[],
            attempt_count=0, error_type="KeyError",
        ),
        _line("plain", now=now, offset=2.0),
    ])
    rows = {row["request_id"]: row for row in rl.query(store_path=db, source_path=log)}
    assert rows["refused"]["refusal"] == {
        "excluded": {"not_invokable": 20, "cooldown": 1}, "scope": "provider", "block_reason": "cooldown",
    }
    assert rows["refused"]["retry_after_seconds"] == 300
    assert rows["crashed"]["error_type"] == "KeyError"
    assert rows["plain"]["refusal"] is None and rows["plain"]["retry_after_seconds"] is None


def test_index_built_before_the_refusal_column_rebuilds_from_the_route_log(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("refused", now=now, final_status=503, attempts=[], retry_after_seconds=60)])
    rl.query(store_path=db, source_path=log)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value = '5' WHERE key = 'schema_version'")
        conn.execute("ALTER TABLE requests DROP COLUMN detail_json")
    rows = rl.query(store_path=db, source_path=log)
    assert [(r["request_id"], r["retry_after_seconds"]) for r in rows] == [("refused", 60)]


def test_window_seconds_from_label():
    assert rl.window_seconds_from_label("1h") == 3600
    assert rl.window_seconds_from_label("7d") == 604800
    assert rl.window_seconds_from_label("bogus") == rl.WINDOW_SECONDS[rl.DEFAULT_WINDOW]
    assert rl.window_seconds_from_label(None) == rl.WINDOW_SECONDS[rl.DEFAULT_WINDOW]


def _stream_attempt(source: str, delta: float | None, *, reason: str = "ok", byte: float | None = None, **extra) -> dict:
    attempt = {"model": f"ficelle/{source}/m", "source": source, "status": 200 if reason == "ok" else 502,
               "reason": reason, "latency_seconds": 3.0, **extra}
    if delta is not None:
        attempt["first_delta_seconds"] = delta
    if byte is not None:
        attempt["first_byte_seconds"] = byte
    return attempt


def test_ingest_persists_the_attempt_and_route_timing_fields(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line(
        "timed", now=now, stream=True, first_byte_seconds=0.31, first_delta_seconds=0.42,
        route_prep_seconds=0.002, router_overhead_seconds=0.004, workload="aux:title_generation",
        attempts=[_stream_attempt("openrouter", 0.42, byte=0.31, router_pre_dispatch_seconds=0.001)],
    )])
    row = rl.query(store_path=db, source_path=log)[0]
    assert (row["first_byte_seconds"], row["first_delta_seconds"]) == (0.31, 0.42)
    assert (row["route_prep_seconds"], row["router_overhead_seconds"]) == (0.002, 0.004)
    assert row["workload"] == "aux:title_generation" and row["synthetic"] is False
    assert row["attempts"][0]["first_delta_seconds"] == 0.42
    assert row["attempts"][0]["first_byte_seconds"] == 0.31
    assert row["attempts"][0]["router_pre_dispatch_seconds"] == 0.001


def test_ingest_drops_non_finite_or_negative_timings(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("bad", now=now, stream=True, first_delta_seconds=-1.0, first_byte_seconds=float("inf"),
                       router_overhead_seconds="slow")])
    row = rl.query(store_path=db, source_path=log)[0]
    assert (row["first_delta_seconds"], row["first_byte_seconds"], row["router_overhead_seconds"]) == (None, None, None)


def test_summary_ttft_is_nearest_rank_per_attempt_and_attributed_to_each_provider(paths):
    log, db = paths
    now = time.time()
    rows = []
    # Ten single-attempt streamed requests on "fast": 0.1 .. 1.0 s.
    for index in range(1, 11):
        rows.append(_line(f"fast{index}", now=now, offset=index, stream=True, selected_source="fast",
                          attempts=[_stream_attempt("fast", index / 10)]))
    # A failed-over request: "slow" failed before any delta (no observation), "mid" streamed a
    # delta and then errored (one observation, its own), "fast" answered (one observation).
    rows.append(_line("failover", now=now, offset=1, stream=True, selected_source="fast", attempt_count=3, attempts=[
        _stream_attempt("slow", None, reason="timeout"),
        _stream_attempt("mid", 9.0, reason="upstream_stream_error"),
        _stream_attempt("fast", 2.0),
    ]))
    _write(log, rows)

    ttft = rl.summary(store_path=db, source_path=log, now=now)["ttft"]
    by_source = {entry["source"]: entry for entry in ttft["by_source"]}

    assert by_source["fast"]["n"] == 11  # the failover's winning attempt counts for fast, not for the last row's source
    # sorted: 0.1..1.0 plus 2.0 -> nearest-rank p50 = ceil(.5*11)=6th, p75 = 9th, p95 = 11th
    assert (by_source["fast"]["p50"], by_source["fast"]["p75"], by_source["fast"]["p95"]) == (0.6, 0.9, 2.0)
    assert by_source["mid"]["n"] == 1 and by_source["mid"]["p50"] == 9.0
    # "slow" never produced a delta: present with an explicit state, never a 0 or a fast-looking null.
    assert by_source["slow"] == {"source": "slow", "n": 0, "p50": None, "p75": None, "p95": None,
                                 "state": "no_streaming_data"}
    assert ttft["overall"]["n"] == 12


def test_summary_ttft_ignores_non_streamed_synthetic_and_errored_rows(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("real", now=now, stream=True, attempts=[_stream_attempt("openrouter", 0.5, byte=0.4)]),
        # first_byte equals the whole latency on a buffered answer: must never read as a fast TTFT.
        _line("buffered", now=now, stream=False, attempts=[_stream_attempt("openrouter", None, byte=7.0)]),
        _line("probe", now=now, stream=True, synthetic_case_id="case-1",
              attempts=[_stream_attempt("openrouter", 99.0)]),
        _line("bench", now=now, stream=True, workload="benchmark:coding",
              attempts=[_stream_attempt("openrouter", 98.0)]),
        _line("failed-pre-commit", now=now, stream=True, final_status=502, final_reason="upstream_failure",
              selected_source=None, selected_model=None, selected_upstream=None,
              attempts=[_stream_attempt("openrouter", None, reason="timeout", byte=0.2)]),
    ])

    ttft = rl.summary(store_path=db, source_path=log, now=now)["ttft"]

    assert ttft["by_source"] == [{"source": "openrouter", "n": 1, "p50": 0.5, "p75": 0.5, "p95": 0.5, "state": "ok"}]
    assert ttft["by_profile"][0]["n"] == 1


def test_summary_ttft_is_explicitly_empty_without_streaming_data(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("buffered", now=now, stream=False)])

    ttft = rl.summary(store_path=db, source_path=log, now=now)["ttft"]

    assert ttft["overall"] == {"n": 0, "p50": None, "p75": None, "p95": None, "state": "no_streaming_data"}
    assert ttft["by_source"] == [
        {"source": "openrouter", "n": 0, "p50": None, "p75": None, "p95": None, "state": "no_streaming_data"}
    ]


def test_summary_ttft_groups_by_profile(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line("a", now=now, stream=True, requested_model="ficelle/auto-fast", attempts=[_stream_attempt("p", 0.2)]),
        _line("b", now=now, stream=True, requested_model="ficelle/auto-long", attempts=[_stream_attempt("p", 4.0)]),
    ])
    by_profile = {e["profile"]: e for e in rl.summary(store_path=db, source_path=log, now=now)["ttft"]["by_profile"]}
    assert (by_profile["ficelle/auto-fast"]["p50"], by_profile["ficelle/auto-long"]["p50"]) == (0.2, 4.0)


def test_summary_is_unchanged_from_schema_9_on_a_recorded_route_log(tmp_path):
    # `summary_v9.json` is what schema 9 returned for this log: one scan per aggregate and every
    # row's attempts_json parsed in Python. Schema 10 reads per-attempt rows and folds the
    # breakdowns into one scan; for the same log, every figure, order and tie must match. The
    # log mixes random traffic with malformed attempts, a replayed request id, a row without a
    # timestamp, a provider that only failed before a delta and delivery corrections; `now` is
    # far enough ahead that the real-clock age trim keeps every row.
    golden = json.loads((FIXTURES / "summary_v9.json").read_text(encoding="utf-8"))
    db = tmp_path / "requests.sqlite"
    for label, seconds in rl.WINDOW_SECONDS.items():
        result = rl.summary(store_path=db, source_path=FIXTURES / "routes.jsonl", window_seconds=seconds,
                            now=4_000_000_000.0)
        result.pop("generated_at")
        assert json.loads(json.dumps(result)) == golden[label], label


def test_summary_router_overhead_percentiles_skip_untimed_and_synthetic_rows(paths):
    log, db = paths
    now = time.time()
    timed = [_line(f"o{index}", now=now, offset=index, router_overhead_seconds=index / 1000) for index in range(1, 21)]
    _write(log, timed + [
        _line("legacy", now=now),  # logged before the field existed: not a zero
        _line("probe", now=now, synthetic_case_id="case-9", router_overhead_seconds=5.0),
    ])

    overhead = rl.summary(store_path=db, source_path=log, now=now)["router_overhead"]

    assert overhead == {"n": 20, "p50": 0.01, "p95": 0.019}


def test_index_built_before_the_timing_columns_rebuilds_with_history(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("old", now=now, stream=True, first_delta_seconds=1.5,
                       attempts=[_stream_attempt("openrouter", 1.5)])])
    rl.query(store_path=db, source_path=log)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(rl.SCHEMA_VERSION - 1),))
        conn.execute("ALTER TABLE requests DROP COLUMN first_delta_seconds")
        # A real pre-8 file lacks the columns the summary index covers, and so the index itself.
        conn.execute("DROP INDEX idx_requests_summary")
        conn.execute("ALTER TABLE requests DROP COLUMN router_overhead_seconds")
    assert rl.summary(store_path=db, source_path=log, now=now)["ttft"]["overall"]["n"] == 1




def test_context_filter_route_fields_survive_ingestion(paths):
    log, db = paths
    now = time.time()
    _write(log, [
        _line(
            "filtered", now=now,
            context_required_tokens=230_011, context_largest_in_pool=1_000_000, demoted_for_context=2,
            prompt_estimate_tokens=200_009, context_changed_selection=True,
        ),
        _line("quiet", now=now, offset=1, context_changed_selection=False, demoted_for_context=0),
        _line("older", now=now, offset=2),
    ])
    rows = {row["request_id"]: row for row in rl.query(store_path=db, source_path=log)}

    assert rows["filtered"]["context_required_tokens"] == 230_011
    assert rows["filtered"]["context_largest_in_pool"] == 1_000_000
    assert rows["filtered"]["demoted_for_context"] == 2
    assert rows["filtered"]["prompt_estimate_tokens"] == 200_009
    assert rows["filtered"]["context_changed_selection"] is True
    assert rows["quiet"]["context_changed_selection"] is False
    # A row from before these fields existed reads as unknown, not as "the filter did nothing".
    assert rows["older"]["context_changed_selection"] is None
    assert rows["older"]["demoted_for_context"] is None


def test_context_estimate_summary_is_nearest_rank_over_rows_with_both_figures(paths):
    log, db = paths
    now = time.time()
    # estimate / actual: 0.8 (under), 1.0, 1.1, 1.25 -> nearest rank over 4 samples.
    pairs = [(8_000, 10_000), (10_000, 10_000), (11_000, 10_000), (12_500, 10_000)]
    rows = [
        _line(
            f"r{i}", now=now, offset=i,
            prompt_estimate_tokens=estimate, usage={"prompt_tokens": actual, "completion_tokens": 5},
        )
        for i, (estimate, actual) in enumerate(pairs)
    ]
    rows += [
        # Excluded, each for its own reason: no upstream usage, synthetic probe, tiny prompt.
        _line("no-usage", now=now, offset=10, prompt_estimate_tokens=50_000),
        _line("probe", now=now, offset=11, prompt_estimate_tokens=1_500, synthetic_case_id="case-1",
              usage={"prompt_tokens": 1_000, "completion_tokens": 5}),
        _line("tiny", now=now, offset=12, prompt_estimate_tokens=10, usage={"prompt_tokens": 40, "completion_tokens": 5}),
        _line("legacy", now=now, offset=13, usage={"prompt_tokens": 9_000, "completion_tokens": 5}),
    ]
    _write(log, rows)

    estimate = rl.summary(store_path=db, source_path=log, window_seconds=86400, now=now)["context_estimate"]

    assert estimate["n"] == 4
    assert estimate["ratio_p50"] == 1.0
    assert estimate["ratio_p90"] == 1.25
    assert estimate["ratio_p99"] == 1.25
    assert estimate["ratio_p1"] == 0.8
    assert estimate["underestimate_rate"] == 0.25


def test_context_estimate_summary_has_an_honest_empty_state(paths):
    log, db = paths
    now = time.time()
    _write(log, [_line("r1", now=now)])

    estimate = rl.summary(store_path=db, source_path=log, window_seconds=86400, now=now)["context_estimate"]

    assert estimate["n"] == 0
    assert estimate["ratio_p50"] is None and estimate["underestimate_rate"] is None
