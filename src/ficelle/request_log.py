"""Derived request-log index over ``routes.jsonl``.

``write_route_log`` (in ``router.py``) already appends one redacted JSON line per
chat-completion route to the Ficelle runtime log — never prompts
or secrets. This module builds a small, reconstructible SQLite index from that
file so the admin dashboard can list and aggregate request health without
touching the routing hot path.

Design:

- ``routes.jsonl`` stays the append-only source of truth. The SQLite file is a
  derived cache: delete it and it rebuilds on the next admin query.
- Ingestion is lazy ("catch-up"): each public call reads the new tail of the log
  from a persisted byte offset and inserts the rows. There is no hook in the
  hot-path writer.
- ``_row_from_log_line`` is the redaction boundary: only whitelisted fields ever
  reach the database.

The module is intentionally decoupled from ``router.py`` (it imports nothing from
it) so it can be unit-tested in isolation, mirroring ``compression.py``.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ficelle.runtime_paths import RuntimePaths

SCHEMA_VERSION = 3

# Retention bounds for the derived index (the JSONL source keeps everything).
MAX_ROWS = 100_000
MAX_AGE_SECONDS = 30 * 86_400

# Ingestion / query bounds.
INGEST_BATCH = 1000
DEFAULT_QUERY_LIMIT = 50
MAX_QUERY_LIMIT = 200

# Live-tail bounds. MAX_TAIL_REPLAY caps how far behind a resuming subscriber may be
# before ``tail`` gives up on replaying and asks it to reload from scratch — a browser
# tab that reconnects after hours must not stream a day of history one batch at a time.
DEFAULT_TAIL_LIMIT = 100
MAX_TAIL_REPLAY = 200

# Summary windows exposed to the dashboard (label -> seconds).
WINDOW_SECONDS: dict[str, int] = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
    "30d": 2_592_000,
}
DEFAULT_WINDOW = "24h"

_RUNTIME_PATHS = RuntimePaths.from_env()
DEFAULT_DB_PATH = _RUNTIME_PATHS.request_log_store_path
DEFAULT_ROUTE_LOG_PATH = _RUNTIME_PATHS.route_log_path

# Column order shared by the schema and the INSERT statement.
_COLUMNS = (
    "request_id",
    "ts",
    "logged_at",
    "profile",
    "source",
    "upstream_id",
    "model_id",
    "status",
    "reason",
    "error_type",
    "latency_seconds",
    "candidate_count",
    "attempt_count",
    "stream",
    "fallback",
    "competence",
    "compression_status",
    "prompt_tokens",
    "completion_tokens",
    "prompt_price",
    "completion_price",
    "attempts_json",
)

_INSERT_SQL = (
    "INSERT OR IGNORE INTO requests ("
    + ", ".join(_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in _COLUMNS)
    + ")"
)

# Per-attempt keys kept inside attempts_json. This is the redaction whitelist for
# the one verbatim blob: anything else in an attempt dict is dropped on ingest.
_ATTEMPT_KEYS = (
    "model",
    "upstream",
    "source",
    "status",
    "reason",
    "error_type",
    "latency_seconds",
    "stream_started",
    "stream_chunk_count",
    "stream_bytes_sent",
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def query(
    *,
    store_path: Path | None = None,
    source_path: Path | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
    profile: str | None = None,
    source: str | None = None,
    reason: str | None = None,
    status: Any = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent request rows (most recent first), after a catch-up ingest."""
    bounded = max(1, min(_safe_int(limit, DEFAULT_QUERY_LIMIT) or DEFAULT_QUERY_LIMIT, MAX_QUERY_LIMIT))
    with closing(_connect(store_path)) as conn:
        _catch_up(conn, source_path)
        where, params = _filter_conditions(profile, source, reason, status, q)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(
            f"SELECT * FROM requests{clause} ORDER BY ts DESC, rowid DESC LIMIT ?",
            (*params, bounded),
        ).fetchall()
        return [_public_row(row) for row in rows]


def tail(
    *,
    cursor: Any = None,
    store_path: Path | None = None,
    source_path: Path | None = None,
    limit: int = DEFAULT_TAIL_LIMIT,
    profile: str | None = None,
    source: str | None = None,
    reason: str | None = None,
    status: Any = None,
    q: str | None = None,
) -> dict[str, Any]:
    """Return rows ingested after ``cursor`` (oldest first) and the cursor to poll next.

    The cursor is the SQLite ``rowid``, which follows the append-only ingest order, so a
    subscriber never re-reads a row it already has. ``cursor=None`` primes a live tail:
    it catches up and hands back the current high-water mark with no rows, so opening the
    page does not replay history.

    ``resync`` asks the caller to reload the list instead of applying a delta — either the
    subscriber fell further behind than ``MAX_TAIL_REPLAY``, or the index was rebuilt under
    it and the rowids restarted below its cursor.
    """
    bounded = max(1, min(_safe_int(limit, DEFAULT_TAIL_LIMIT) or DEFAULT_TAIL_LIMIT, MAX_QUERY_LIMIT))
    with closing(_connect(store_path)) as conn:
        _catch_up(conn, source_path)
        head = _head(conn)
        start = _safe_int(cursor, None)
        if start is None:
            return {"cursor": head, "entries": [], "resync": False}
        if start > head or head - start > MAX_TAIL_REPLAY:
            return {"cursor": head, "entries": [], "resync": True}
        where, params = _filter_conditions(profile, source, reason, status, q)
        clause = " AND ".join(["rowid > ?", *where])
        rows = conn.execute(
            f"SELECT rowid AS row_id, * FROM requests WHERE {clause} ORDER BY rowid LIMIT ?",
            (start, *params, bounded),
        ).fetchall()
        # Fewer rows than the cap means the scan reached the head, so the cursor may skip
        # the rows the filter dropped. A full batch means more may remain behind it.
        next_cursor = int(rows[-1]["row_id"]) if len(rows) >= bounded else head
        return {"cursor": next_cursor, "entries": [_public_row(row) for row in rows], "resync": False}


def head_cursor(*, store_path: Path | None = None) -> int:
    """Current high-water rowid, without ingesting.

    Pairs a list snapshot with the cursor a live tail must start from: without it the
    subscriber primes at whatever the head is when its stream opens, and any request
    logged between the two silently never reaches the page.
    """
    with closing(_connect(store_path)) as conn:
        return _head(conn)


def _head(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COALESCE(MAX(rowid), 0) AS head FROM requests").fetchone()["head"])


def source_signature(source_path: Path | None = None) -> tuple[int, int] | None:
    """Cheap change detector for the route log: ``(size, mtime_ns)``, or ``None`` if absent.

    A live tail polls this instead of opening the index, so an idle router costs one
    ``stat()`` per tick per subscriber and touches SQLite only when the log actually grew.
    """
    path = (
        source_path
        if source_path is not None
        else _RUNTIME_PATHS.read_path(DEFAULT_ROUTE_LOG_PATH)
    )
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def _filter_conditions(
    profile: str | None,
    source: str | None,
    reason: str | None,
    status: Any,
    q: str | None,
) -> tuple[list[str], list[Any]]:
    """Build the shared WHERE conditions for the list and the live tail, and their params.

    Returns the conditions unjoined: the tail prepends its own cursor condition, and each
    caller owns where the ``WHERE`` keyword goes.
    """
    where: list[str] = []
    params: list[Any] = []
    if profile:
        where.append("profile = ?")
        params.append(profile)
    if source:
        where.append("source = ?")
        params.append(source)
    if reason:
        where.append("reason = ?")
        params.append(reason)
    if status is not None and status != "":
        where.append("status = ?")
        params.append(_safe_int(status, -1))
    if q:
        where.append("(request_id LIKE ? OR model_id LIKE ? OR upstream_id LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    return where, params


def summary(
    *,
    store_path: Path | None = None,
    source_path: Path | None = None,
    window_seconds: int = WINDOW_SECONDS[DEFAULT_WINDOW],
    now: float | None = None,
) -> dict[str, Any]:
    """Return aggregate request-health metrics over the given window.

    The savings figure is a pure aggregation of what each success row recorded at
    route time (token counts × the USD-per-token reference stamped on the row), so
    the estimate is immutable history: a later catalog price change cannot inflate
    it, and rebuilding the index from ``routes.jsonl`` reproduces the same numbers.
    """
    window = max(60, _safe_int(window_seconds, WINDOW_SECONDS[DEFAULT_WINDOW]) or WINDOW_SECONDS[DEFAULT_WINDOW])
    reference = float(now) if now is not None else time.time()
    cutoff = reference - window
    bucket = _bucket_seconds(window)
    first_bucket = int(cutoff // bucket) * bucket
    last_bucket = int(reference // bucket) * bucket
    with closing(_connect(store_path)) as conn:
        _catch_up(conn, source_path)
        totals = conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN reason = 'ok' THEN 1 ELSE 0 END) AS ok,
              SUM(CASE WHEN fallback = 1 THEN 1 ELSE 0 END) AS fallback,
              SUM(CASE WHEN stream = 1 THEN 1 ELSE 0 END) AS stream
            FROM requests WHERE ts >= ? AND ts <= ?
            """,
            (cutoff, reference),
        ).fetchone()
        total = int(totals["total"] or 0)
        ok = int(totals["ok"] or 0)
        errors = total - ok
        by_source = _grouped(conn, "source", cutoff, reference)
        by_reason = _grouped(conn, "reason", cutoff, reference)
        by_status = _grouped(conn, "status", cutoff, reference)
        counted = {
            int(row["b"]): (int(row["t"]), int(row["ok"] or 0))
            for row in conn.execute(
                """
                SELECT CAST((ts - ?) / ? AS INT) * ? + ? AS b,
                       COUNT(*) AS t,
                       SUM(CASE WHEN reason = 'ok' THEN 1 ELSE 0 END) AS ok
                FROM requests WHERE ts >= ? AND ts <= ? AND ts IS NOT NULL
                GROUP BY b ORDER BY b
                """,
                (first_bucket, bucket, bucket, first_bucket, cutoff, reference),
            ).fetchall()
        }
        # Emit every slot in the window, idle ones included: a windowed series is defined
        # by its window, so an idle bucket is a zero, not an absence. A bare GROUP BY
        # drops those rows, leaving callers unable to tell quiet apart from missing.
        timeline = []
        for start in range(first_bucket, last_bucket + bucket, bucket):
            slot_total, slot_ok = counted.get(start, (0, 0))
            timeline.append({"bucket": start, "total": slot_total, "ok": slot_ok})
        latency_count = int(conn.execute(
            "SELECT COUNT(*) AS c FROM requests "
            "WHERE ts >= ? AND ts <= ? AND latency_seconds IS NOT NULL",
            (cutoff, reference),
        ).fetchone()["c"] or 0)
        usage_totals = conn.execute(
            """
            SELECT COUNT(*) AS c,
                   SUM(COALESCE(prompt_tokens, 0)) AS p,
                   SUM(COALESCE(completion_tokens, 0)) AS o,
                   SUM(CASE WHEN prompt_price IS NOT NULL OR completion_price IS NOT NULL
                       THEN 1 ELSE 0 END) AS priced,
                   SUM(COALESCE(prompt_tokens, 0) * COALESCE(prompt_price, 0)
                       + COALESCE(completion_tokens, 0) * COALESCE(completion_price, 0)) AS saved
            FROM requests
            WHERE ts >= ? AND ts <= ?
              AND (prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL)
            """,
            (cutoff, reference),
        ).fetchone()
        usage_requests = int(usage_totals["c"] or 0)
        priced_requests = int(usage_totals["priced"] or 0)
        return {
            "generated_at": _now_iso(),
            "window_seconds": window,
            "bucket_seconds": bucket,
            "total": total,
            "ok": ok,
            "errors": errors,
            "success_rate": round(ok / total, 4) if total else None,
            "fallback_count": int(totals["fallback"] or 0),
            "stream_count": int(totals["stream"] or 0),
            "latency_p50": _percentile(conn, cutoff, reference, latency_count, 50),
            "latency_p95": _percentile(conn, cutoff, reference, latency_count, 95),
            "by_source": by_source,
            "by_reason": by_reason,
            "by_status": by_status,
            "timeline": timeline,
            "usage": {
                "requests": usage_requests,
                "prompt_tokens": int(usage_totals["p"] or 0),
                "completion_tokens": int(usage_totals["o"] or 0),
            },
            "savings": {
                # Floored at micro-dollar precision, never rounded: half-up rounding
                # could overstate, the one direction the estimate must not err in.
                "estimated_saved_usd": math.floor(float(usage_totals["saved"] or 0.0) * 1e6) / 1e6,
                "priced_requests": priced_requests,
                "unpriced_requests": usage_requests - priced_requests,
            },
        }


def window_seconds_from_label(label: Any) -> int:
    """Map a window label ('1h'/'24h'/'7d'/'30d') to seconds, defaulting to 24h."""
    if isinstance(label, str) and label in WINDOW_SECONDS:
        return WINDOW_SECONDS[label]
    return WINDOW_SECONDS[DEFAULT_WINDOW]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _catch_up(conn: sqlite3.Connection, source_path: Path | None) -> int:
    """Ingest new complete lines from the route log since the stored byte offset."""
    path = (
        source_path
        if source_path is not None
        else _RUNTIME_PATHS.read_path(DEFAULT_ROUTE_LOG_PATH)
    )
    if not path.exists():
        return 0
    try:
        source_stat = path.stat()
    except OSError:
        return 0
    current_size = source_stat.st_size
    source_identity = (
        f"{path.resolve()}:{source_stat.st_dev}:{source_stat.st_ino}"
    )
    previous_source_identity = _meta_text(conn, "ingest_source_identity")
    if previous_source_identity != source_identity:
        # A canonical log superseding a legacy source (or an external rotation)
        # must be read from byte zero. The SQLite index is derived and request_id
        # uniqueness makes replay safe.
        offset = 0
        _set_meta(conn, "ingest_source_identity", source_identity)
    else:
        offset = _meta_int(conn, "ingest_offset", 0)
    if current_size < offset:
        # File was truncated or rotated externally; re-read from the start.
        # UNIQUE(request_id) + INSERT OR IGNORE keeps this from duplicating rows.
        offset = 0
    if current_size == offset:
        # Nothing new to ingest, but still enforce retention so age-based trim
        # runs on a quiescent log (not only when new lines arrive).
        _trim(conn)
        conn.commit()
        return 0
    inserted = 0
    pending: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    # Partial trailing line (a concurrent writer mid-append).
                    # Stop before it and leave the offset so we re-read it later.
                    break
                offset = handle.tell()
                row = _parse_line(raw_line)
                if row is not None:
                    pending.append(row)
                if len(pending) >= INGEST_BATCH:
                    inserted += _insert_rows(conn, pending)
                    _set_meta(conn, "ingest_offset", str(offset))
                    conn.commit()
                    pending = []
    except OSError:
        if pending:
            inserted += _insert_rows(conn, pending)
        _set_meta(conn, "ingest_offset", str(offset))
        conn.commit()
        return inserted
    if pending:
        inserted += _insert_rows(conn, pending)
    _set_meta(conn, "ingest_offset", str(offset))
    _trim(conn)
    conn.commit()
    return inserted


def _parse_line(raw_line: bytes) -> dict[str, Any] | None:
    try:
        decoded = raw_line.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        payload = json.loads(decoded)
    except (ValueError, TypeError):
        return None
    return _row_from_log_line(payload)


def _row_from_log_line(raw: Any) -> dict[str, Any] | None:
    """Whitelist and normalize one parsed route-log line. The redaction boundary.

    Only the fields listed here reach the database — both the scalar columns and,
    inside ``attempts_json``, the per-attempt keys in ``_ATTEMPT_KEYS``; anything
    else in the source line is dropped, so an unexpected field can never leak into
    the index. The field names mirror what ``write_route_log`` emits in
    ``router.py`` — if that writer renames a field, update the mapping here.
    """
    if not isinstance(raw, dict):
        return None
    request_id = raw.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return None

    raw_attempts = raw.get("attempts")
    attempts = (
        [{k: a.get(k) for k in _ATTEMPT_KEYS if k in a} for a in raw_attempts if isinstance(a, dict)]
        if isinstance(raw_attempts, list)
        else []
    )
    last_attempt = attempts[-1] if attempts else None
    error_type: str | None = None
    if last_attempt is not None:
        candidate = last_attempt.get("error_type")
        if isinstance(candidate, str) and candidate:
            error_type = candidate

    # `selected_*` is written only once an attempt delivered, so a run where no candidate
    # succeeded carries none of it — and these three fields are the list's provider column,
    # the provider filter and the `q` search. Left NULL, a failure names nobody in the very
    # views built to spot a misbehaving provider. Fall back to the last attempt that FAILED,
    # not the last attempt: a Fusion panel runs in parallel and its rows carry the successful
    # panelists too (`reason: "ok"`), in completion order — a fast rejection lands first and a
    # real generation last, so plain `attempts[-1]` would routinely charge a failed run to the
    # one provider that answered correctly. An attempt has to state a failure to be picked: one
    # whose `reason` is absent or not a string says nothing about its outcome (the whitelist can
    # also reduce an unrecognized attempt to `{}`), and a row must not name a provider on a
    # guess. A row with no stated failure at all names nobody, which is the honest answer. Only
    # the last failure is attributed: earlier ones stay in `attempts`, since one row has one
    # provider column. All three fields move together or none do, so a row never mixes one
    # attempt's provider with another's model.
    selected_source = _str_or_none(raw.get("selected_source"))
    selected_upstream = _str_or_none(raw.get("selected_upstream"))
    selected_model = _str_or_none(raw.get("selected_model"))
    if selected_source is None and selected_upstream is None and selected_model is None:
        failed_attempt = None
        for attempt in reversed(attempts):
            attempt_reason = attempt.get("reason")
            if isinstance(attempt_reason, str) and attempt_reason and attempt_reason != "ok":
                failed_attempt = attempt
                break
        if failed_attempt is not None:
            selected_source = _str_or_none(failed_attempt.get("source"))
            selected_upstream = _str_or_none(failed_attempt.get("upstream"))
            selected_model = _str_or_none(failed_attempt.get("model"))

    compression = raw.get("compression")
    compression_status = compression.get("status") if isinstance(compression, dict) else None

    # Token counts and the USD-per-token reference recorded at route time — the only
    # usage facts the success writer emits.
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    reference = raw.get("reference_pricing") if isinstance(raw.get("reference_pricing"), dict) else {}

    attempt_count = _safe_int(raw.get("attempt_count"), 0) or 0
    logged_at = raw.get("logged_at") if isinstance(raw.get("logged_at"), str) else None

    return {
        "request_id": request_id,
        "ts": _parse_ts(logged_at),
        "logged_at": logged_at,
        "profile": _str_or_none(raw.get("requested_model")),
        "source": selected_source,
        "upstream_id": selected_upstream,
        "model_id": selected_model,
        "status": _safe_int(raw.get("final_status"), None),
        "reason": _str_or_none(raw.get("final_reason")),
        "error_type": error_type,
        "latency_seconds": _safe_float(raw.get("duration_seconds")),
        "candidate_count": _safe_int(raw.get("candidate_count"), None),
        "attempt_count": attempt_count,
        "stream": 1 if raw.get("stream") else 0,
        "fallback": 1 if attempt_count > 1 else 0,
        "competence": _str_or_none(raw.get("competence")),
        "compression_status": _str_or_none(compression_status),
        "prompt_tokens": _token_count(usage.get("prompt_tokens")),
        "completion_tokens": _token_count(usage.get("completion_tokens")),
        "prompt_price": _price_per_token(reference.get("prompt")),
        "completion_price": _price_per_token(reference.get("completion")),
        "attempts_json": json.dumps(attempts, ensure_ascii=False, sort_keys=True),
    }


def _token_count(value: Any) -> int | None:
    """A non-negative token count, or None. The success writer clamps at the source,
    but the ingest defends on its own: a foreign or hand-edited log line must not
    subtract from the window totals."""
    if isinstance(value, bool):
        return None
    count = _safe_int(value, None)
    return count if count is not None and count >= 0 else None


def _price_per_token(value: Any) -> float | None:
    """A finite, non-negative USD-per-token rate, or None. Bools coerce to 1.0 and
    "Infinity" parses to inf — either would poison the window's savings SUM and can
    make the summary payload unserializable, so values earn the same whitelist
    discipline as field names."""
    if isinstance(value, bool):
        return None
    price = _safe_float(value)
    return price if price is not None and price >= 0 and math.isfinite(price) else None


def _insert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(_INSERT_SQL, [tuple(row[col] for col in _COLUMNS) for row in rows])
    return conn.total_changes - before


def _trim(conn: sqlite3.Connection, now: float | None = None) -> None:
    # The count-based trim (sort + NOT IN) is the expensive one, so only run it when the
    # table is actually over the cap. The age purge is sargable on idx_requests_ts, so it
    # runs cheaply on every catch-up — enforcing retention even on a quiescent log.
    total = int(conn.execute("SELECT COUNT(*) AS c FROM requests").fetchone()["c"] or 0)
    if total > MAX_ROWS:
        conn.execute(
            """
            DELETE FROM requests WHERE rowid NOT IN (
              SELECT rowid FROM requests ORDER BY ts DESC, rowid DESC LIMIT ?
            )
            """,
            (MAX_ROWS,),
        )
    cutoff = (now if now is not None else time.time()) - MAX_AGE_SECONDS
    conn.execute("DELETE FROM requests WHERE ts IS NOT NULL AND ts < ?", (cutoff,))


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _grouped(
    conn: sqlite3.Connection,
    column: str,
    cutoff: float,
    reference: float,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT {column} AS key,
               COUNT(*) AS total,
               SUM(CASE WHEN reason = 'ok' THEN 1 ELSE 0 END) AS ok
        FROM requests WHERE ts >= ? AND ts <= ? AND {column} IS NOT NULL
        GROUP BY {column} ORDER BY total DESC
        """,
        (cutoff, reference),
    ).fetchall()
    return [
        {column: row["key"], "total": int(row["total"]), "ok": int(row["ok"] or 0)}
        for row in rows
    ]


def _percentile(
    conn: sqlite3.Connection,
    cutoff: float,
    reference: float,
    count: int,
    pct: int,
) -> float | None:
    if count == 0:
        return None
    # Nearest-rank: 1-based ceil(pct/100 * count) converted to a 0-based offset.
    # Avoids the upper bias of floor() (e.g. p50 of two samples returning the max).
    offset = max(0, min((count * pct + 99) // 100 - 1, count - 1))
    row = conn.execute(
        "SELECT latency_seconds FROM requests "
        "WHERE ts >= ? AND ts <= ? AND latency_seconds IS NOT NULL "
        "ORDER BY latency_seconds LIMIT 1 OFFSET ?",
        (cutoff, reference, offset),
    ).fetchone()
    return float(row["latency_seconds"]) if row else None


def _bucket_seconds(window_seconds: int) -> int:
    if window_seconds <= 3_600:
        return 300
    if window_seconds <= 86_400:
        return 3_600
    if window_seconds <= 604_800:
        return 21_600
    return 86_400


# --------------------------------------------------------------------------- #
# Connection / schema
# --------------------------------------------------------------------------- #
def _connect(store_path: Path | None = None) -> sqlite3.Connection:
    path = store_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    path.chmod(0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    _create_requests_table(conn)
    _ensure_schema_version(conn)
    return conn


def _create_requests_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
          request_id TEXT,
          ts REAL,
          logged_at TEXT,
          profile TEXT,
          source TEXT,
          upstream_id TEXT,
          model_id TEXT,
          status INTEGER,
          reason TEXT,
          error_type TEXT,
          latency_seconds REAL,
          candidate_count INTEGER,
          attempt_count INTEGER,
          stream INTEGER,
          fallback INTEGER,
          competence TEXT,
          compression_status TEXT,
          prompt_tokens INTEGER,
          completion_tokens INTEGER,
          prompt_price REAL,
          completion_price REAL,
          attempts_json TEXT
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_requests_request_id ON requests(request_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts)")


def _ensure_schema_version(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = _safe_int(row["value"], None) if row is not None else None
    if current == SCHEMA_VERSION:
        return
    if current is not None:
        # Derived data; rebuild rather than migrate.
        conn.execute("DROP TABLE IF EXISTS requests")
        _create_requests_table(conn)
        conn.execute("DELETE FROM meta WHERE key = 'ingest_offset'")
    _set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    conn.commit()


def _meta_int(conn: sqlite3.Connection, key: str, default: int) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    value = _safe_int(row["value"], None)
    return value if value is not None else default


def _meta_text(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row is not None else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))


# --------------------------------------------------------------------------- #
# Small value coercers (module-local; keeps this decoupled from router.py)
# --------------------------------------------------------------------------- #
def _public_row(row: sqlite3.Row) -> dict[str, Any]:
    attempts: list[Any] = []
    raw = row["attempts_json"]
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                attempts = parsed
        except ValueError:
            attempts = []
    return {
        "request_id": row["request_id"],
        "ts": row["ts"],
        "logged_at": row["logged_at"],
        "profile": row["profile"],
        "source": row["source"],
        "upstream_id": row["upstream_id"],
        "model_id": row["model_id"],
        "status": row["status"],
        "reason": row["reason"],
        "error_type": row["error_type"],
        "latency_seconds": row["latency_seconds"],
        "candidate_count": row["candidate_count"],
        "attempt_count": row["attempt_count"],
        "stream": bool(row["stream"]),
        "fallback": bool(row["fallback"]),
        "competence": row["competence"],
        "compression_status": row["compression_status"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "attempts": attempts,
    }


def _parse_ts(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value)


def _safe_int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
