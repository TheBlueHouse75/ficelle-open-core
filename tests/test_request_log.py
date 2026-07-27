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


def test_window_seconds_from_label():
    assert rl.window_seconds_from_label("1h") == 3600
    assert rl.window_seconds_from_label("7d") == 604800
    assert rl.window_seconds_from_label("bogus") == rl.WINDOW_SECONDS[rl.DEFAULT_WINDOW]
    assert rl.window_seconds_from_label(None) == rl.WINDOW_SECONDS[rl.DEFAULT_WINDOW]
