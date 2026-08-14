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


def _count(db: Path) -> int:
    """Row count straight from the DB — does not trigger ingestion (unlike query())."""
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]


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
    _write(log, [_line(f"r{i}", now=now, offset=i) for i in range(6)])
    rl.query(store_path=db, source_path=log)
    assert _count(db) == 3


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


def test_window_seconds_from_label():
    assert rl.window_seconds_from_label("1h") == 3600
    assert rl.window_seconds_from_label("7d") == 604800
    assert rl.window_seconds_from_label("bogus") == rl.WINDOW_SECONDS[rl.DEFAULT_WINDOW]
    assert rl.window_seconds_from_label(None) == rl.WINDOW_SECONDS[rl.DEFAULT_WINDOW]
