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
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ficelle.runtime_paths import RuntimePaths

SCHEMA_VERSION = 10

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
    "prompt_tokens_estimate",
    "excluded_for_context",
    "demoted_for_context",
    "context_required_tokens",
    "context_largest_in_pool",
    "prompt_estimate_tokens",
    "context_changed_selection",
    "attempts_json",
    "detail_json",
    "delivery_corrected",
    "first_byte_seconds",
    "first_delta_seconds",
    "route_prep_seconds",
    "router_overhead_seconds",
    "workload",
    "synthetic",
    "policy_profile",
)

_INSERT_SQL = (
    "INSERT OR IGNORE INTO requests ("
    + ", ".join(_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in _COLUMNS)
    + ")"
)

# One row per attempt in ``request_attempts``, filled at ingest beside ``attempts_json`` so
# ``summary()`` reads TTFT observations from columns instead of re-parsing every row's blob. ``ts``, ``profile``
# and ``organic_stream`` (the request streamed and is not synthetic) are copied from the request:
# a delivery correction moves the request's rowid and changes only its reason, none of these.
_ATTEMPT_COLUMNS = (
    "request_id",
    "position",
    "ts",
    "profile",
    "organic_stream",
    "source",
    "upstream_id",
    "model_id",
    "reason",
    "first_byte_seconds",
    "first_delta_seconds",
)

_INSERT_ATTEMPT_SQL = (
    "INSERT INTO request_attempts ("
    + ", ".join(_ATTEMPT_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in _ATTEMPT_COLUMNS)
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
    "error_detail",
    "error_errno",
    "latency_seconds",
    "first_byte_seconds",
    "first_delta_seconds",
    "router_pre_dispatch_seconds",
    "read_timeout_seconds",
    "connect_timeout_seconds",
    "read_timeout_source",
    "request_budget_remaining_seconds",
    "timeout_phase",
    "transport_retry",
    "stale_latency_seconds",
    "stream_started",
    "stream_chunk_count",
    "stream_bytes_sent",
    "completion_chars",
)

# Traffic that measures the router rather than serving a client. Synthetic-health runs carry a
# validated `synthetic_case_id`; a loopback client may also label itself with one of these
# workload families. The TTFT, router-overhead and estimator aggregates leave both out so a probe
# cannot skew what users felt; the headline latency_p50/p95, totals and timeline count every row.
_NON_ORGANIC_WORKLOAD = re.compile(r"(benchmark|canary|synthetic)(?![a-z0-9])")


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


def resolve_quality_feedback_attribution(
    request_id: Any,
    *,
    store_path: Path | None = None,
    source_path: Path | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Resolve one attributable final success for local operator quality feedback.

    This is intentionally a literal equality lookup over the existing redacted derived index.
    The caller receives only trusted route provenance, never the request id it supplied, and
    failed, delivery-corrected, incomplete, future or expired rows fail closed as ``None``.
    """
    if not isinstance(request_id, str) or not request_id:
        return None
    reference = time.time() if now is None else now
    if not math.isfinite(reference):
        return None
    with closing(_connect(store_path)) as conn:
        _catch_up(conn, source_path)
        row = conn.execute(
            """
            SELECT profile, source, upstream_id, model_id, status, reason, ts, policy_profile
            FROM requests WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
    if row is None:
        return None
    timestamp = row["ts"]
    if (
        not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or float(timestamp) > reference
        or float(timestamp) + MAX_AGE_SECONDS <= reference
        or not isinstance(row["status"], int)
        or not 200 <= row["status"] < 300
        or row["reason"] != "ok"
    ):
        return None
    fields = {key: row[key] for key in ("profile", "source", "upstream_id", "model_id")}
    if not all(isinstance(value, str) and value for value in fields.values()):
        return None
    return {**fields, "timestamp": float(timestamp), "policy_profile": row["policy_profile"]}


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
        _, delivery_updated = _catch_up(conn, source_path)
        head = _head(conn)
        start = _safe_int(cursor, None)
        if start is None:
            return {"cursor": head, "entries": [], "resync": False}
        if delivery_updated:
            # The current tail ingested a delivery correction. Force one list reload instead
            # of applying that existing request as a new row.
            return {"cursor": head, "entries": [], "resync": True}
        if start > head or head - start > MAX_TAIL_REPLAY:
            return {"cursor": head, "entries": [], "resync": True}
        corrected = conn.execute(
            "SELECT 1 FROM requests WHERE rowid > ? AND delivery_corrected = 1 LIMIT 1",
            (start,),
        ).fetchone()
        if corrected is not None:
            # Another admin read may have ingested the correction before this tail. Its
            # moved rowid persists the signal so every subscriber behind it still reloads.
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
        # One scan for the totals and the usage figures: each separate query re-reads the whole
        # window, and the window's rows are wide. A row counts toward usage only when it recorded
        # token counts; the COALESCEd sums need no such guard because the others add zero.
        totals = conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN reason = 'ok' THEN 1 ELSE 0 END) AS ok,
              SUM(CASE WHEN fallback = 1 THEN 1 ELSE 0 END) AS fallback,
              SUM(CASE WHEN stream = 1 THEN 1 ELSE 0 END) AS stream,
              SUM(CASE WHEN prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL
                  THEN 1 ELSE 0 END) AS usage_requests,
              SUM(COALESCE(prompt_tokens, 0)) AS prompt_tokens,
              SUM(COALESCE(completion_tokens, 0)) AS completion_tokens,
              SUM(CASE WHEN (prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL)
                        AND (prompt_price IS NOT NULL OR completion_price IS NOT NULL)
                  THEN 1 ELSE 0 END) AS priced,
              SUM(COALESCE(prompt_tokens, 0) * COALESCE(prompt_price, 0)
                  + COALESCE(completion_tokens, 0) * COALESCE(completion_price, 0)) AS saved
            FROM requests WHERE ts >= ? AND ts <= ?
            """,
            (cutoff, reference),
        ).fetchone()
        total = int(totals["total"] or 0)
        ok = int(totals["ok"] or 0)
        errors = total - ok
        facets = _facets(conn, cutoff, reference, first_bucket, bucket)
        by_source = facets["source"]
        # Emit every slot in the window, idle ones included: a windowed series is defined
        # by its window, so an idle bucket is a zero, not an absence. A bare GROUP BY
        # drops those rows, leaving callers unable to tell quiet apart from missing.
        timeline = []
        for start in range(first_bucket, last_bucket + bucket, bucket):
            slot_total, slot_ok = facets["timeline"].get(start, (0, 0))
            timeline.append({"bucket": start, "total": slot_total, "ok": slot_ok})
        latencies, overhead, ratios = _distributions(conn, cutoff, reference)
        usage_requests = int(totals["usage_requests"] or 0)
        priced_requests = int(totals["priced"] or 0)
        ttft = _ttft_summary(
            conn,
            cutoff,
            reference,
            known_sources=[row["source"] for row in by_source],
            known_profiles=[row["profile"] for row in facets["profile"]],
        )
        context_estimate = _context_estimate(ratios)
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
            "latency_p50": _percentile(latencies, 50),
            "latency_p95": _percentile(latencies, 95),
            "ttft": ttft,
            # Ficelle's own share of a request (route preparation, pre-dispatch work, residual),
            # not the provider's. Rows logged before the field existed are absent, never zero.
            "router_overhead": {
                "n": len(overhead),
                "p50": _percentile(overhead, 50),
                "p95": _percentile(overhead, 95),
            },
            "by_source": by_source,
            "by_reason": facets["reason"],
            "by_status": facets["status"],
            "timeline": timeline,
            "context_estimate": context_estimate,
            "usage": {
                "requests": usage_requests,
                "prompt_tokens": int(totals["prompt_tokens"] or 0),
                "completion_tokens": int(totals["completion_tokens"] or 0),
            },
            "savings": {
                # Floored at micro-dollar precision, never rounded: half-up rounding
                # could overstate, the one direction the estimate must not err in.
                "estimated_saved_usd": math.floor(float(totals["saved"] or 0.0) * 1e6) / 1e6,
                "priced_requests": priced_requests,
                "unpriced_requests": usage_requests - priced_requests,
            },
        }


# Below this many upstream-reported prompt tokens the ratio measures the provider's fixed chat
# template overhead, not the request-body heuristic, and a small prompt never decides a context fit.
ESTIMATE_RATIO_MIN_PROMPT_TOKENS = 1_000


def _context_estimate(ratios: list[float]) -> dict[str, Any]:
    """How the pre-call prompt estimate compares with what upstreams reported.

    ``ratio`` is ``prompt_estimate_tokens / prompt_tokens`` per request that has both (a row without
    upstream usage is excluded, never counted as zero). A ratio below 1 is an underestimate, the
    direction the safety factor exists to cover, so the low percentiles are the ones that justify
    the factor: a factor F covers a request when ``ratio * F >= 1``. Synthetic health probes are
    excluded. ``n == 0`` yields nulls, never a made-up distribution. ``ratios`` arrives sorted.
    """
    count = len(ratios)

    def at(pct: int) -> float | None:
        value = _percentile(ratios, pct)
        return round(value, 4) if value is not None else None

    return {
        "n": count,
        "min_prompt_tokens": ESTIMATE_RATIO_MIN_PROMPT_TOKENS,
        "ratio_p1": at(1),
        "ratio_p10": at(10),
        "ratio_p50": at(50),
        "ratio_p90": at(90),
        "ratio_p99": at(99),
        "underestimate_rate": round(sum(1 for ratio in ratios if ratio < 1) / count, 4) if count else None,
    }


def window_seconds_from_label(label: Any) -> int:
    """Map a window label ('1h'/'24h'/'7d'/'30d') to seconds, defaulting to 24h."""
    if isinstance(label, str) and label in WINDOW_SECONDS:
        return WINDOW_SECONDS[label]
    return WINDOW_SECONDS[DEFAULT_WINDOW]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _catch_up(conn: sqlite3.Connection, source_path: Path | None) -> tuple[int, bool]:
    """Ingest new complete lines and report whether an existing row was corrected."""
    path = (
        source_path
        if source_path is not None
        else _RUNTIME_PATHS.read_path(DEFAULT_ROUTE_LOG_PATH)
    )
    if not path.exists():
        return 0, False
    try:
        source_stat = path.stat()
    except OSError:
        return 0, False
    current_size = source_stat.st_size
    source_identity = (
        f"{path.resolve()}:{source_stat.st_dev}:{source_stat.st_ino}"
    )
    previous_source_identity = _meta_text(conn, "ingest_source_identity")
    previous_head = _meta_text(conn, "ingest_source_head") or ""
    current_head = _head_fingerprint(path)
    inserted = 0
    delivery_updated = False
    if previous_source_identity != source_identity or not current_head.startswith(previous_head):
        # A canonical log superseding a legacy source (or an external rotation)
        # must be read from byte zero. The SQLite index is derived and request_id
        # uniqueness makes replay safe.
        inserted, delivery_updated = _ingest_rotated_segments(
            conn, path, previous_source_identity, _meta_int(conn, "ingest_offset", 0), previous_head
        )
        offset = 0
        _set_meta(conn, "ingest_source_identity", source_identity)
    else:
        offset = _meta_int(conn, "ingest_offset", 0)
    # Refreshed on every pass: a head stored while the file held a partial first line keeps
    # growing into the full prefix, which `startswith` accepts.
    _set_meta(conn, "ingest_source_head", current_head)
    if current_size < offset:
        # File was truncated or rotated externally; re-read from the start.
        # UNIQUE(request_id) + INSERT OR IGNORE keeps this from duplicating rows.
        offset = 0
    if current_size == offset:
        # Nothing new to ingest, but still enforce retention so age-based trim
        # runs on a quiescent log (not only when new lines arrive).
        _trim(conn)
        conn.commit()
        return inserted, delivery_updated

    def checkpoint(read_offset: int) -> None:
        _set_meta(conn, "ingest_offset", str(read_offset))
        conn.commit()

    changed, updated, offset, read_ok = _ingest_lines(conn, path, offset, checkpoint)
    inserted += changed
    delivery_updated = delivery_updated or updated
    _set_meta(conn, "ingest_offset", str(offset))
    if read_ok:
        _trim(conn)
    conn.commit()
    return inserted, delivery_updated


def _ingest_lines(
    conn: sqlite3.Connection,
    path: Path,
    offset: int,
    checkpoint: Callable[[int], None] | None = None,
) -> tuple[int, bool, int, bool]:
    """Insert every complete line of `path` from `offset`, `INGEST_BATCH` rows at a time.

    Returns the rows inserted, whether a delivery update applied, the offset just past the last
    complete line read, and whether the read finished without an OSError. A partial trailing line
    (a concurrent writer mid-append) stops the read before it, so the next catch-up re-reads it.
    `checkpoint` runs with that offset after each full batch, which lets the live file persist
    its progress as it goes.
    """
    inserted = 0
    delivery_updated = False
    pending: list[dict[str, Any]] = []
    read_ok = True

    def flush() -> None:
        nonlocal inserted, delivery_updated, pending
        changed, updated = _insert_rows(conn, pending)
        inserted += changed
        delivery_updated = delivery_updated or updated
        pending = []

    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                raw_line = handle.readline()
                if not raw_line or not raw_line.endswith(b"\n"):
                    break
                offset = handle.tell()
                row = _parse_line(raw_line)
                if row is not None:
                    pending.append(row)
                if len(pending) >= INGEST_BATCH:
                    flush()
                    if checkpoint is not None:
                        checkpoint(offset)
    except OSError:
        read_ok = False
    flush()
    return inserted, delivery_updated, offset, read_ok


# Matches `rotate_if_oversized`'s default `keep`, which is what the router's route log uses.
ROTATED_SEGMENTS = 3


# Leading bytes kept with the `dev:ino` identity. Linux filesystems reuse a freed inode number
# at once, so after a rotation drops the oldest segment the new live file (or a newer segment)
# can carry the inode of the file indexed last; its first bytes (a timestamped JSON line) differ.
_HEAD_FINGERPRINT_BYTES = 128


def _head_fingerprint(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return handle.read(_HEAD_FINGERPRINT_BYTES).hex()
    except OSError:
        return ""


def _ingest_rotated_segments(
    conn: sqlite3.Connection,
    path: Path,
    previous_identity: str | None,
    previous_offset: int,
    previous_head: str = "",
) -> tuple[int, bool]:
    """Ingest what the log gained between the last ingest and the rotation that replaced it.

    Ingestion is lazy and the dashboard is usually closed, so the file indexed last time can
    have been rotated to `<log>.N` since. A rename keeps the inode, so the stored `dev:ino` finds
    it: read it from the stored offset, then every newer segment in full. The live file, which the
    caller reads next, follows them. No previous identity at all (a first run, or a schema rebuild,
    which forgets it) means nothing was indexed yet: every surviving segment is read, oldest first.
    A previous identity that matches no segment while every slot is taken means the indexed file
    rotated out of retention: every surviving segment is newer, so all are read, oldest first.
    Otherwise (a source switch, a stray leftover segment) nothing is ingested here.
    """
    segments = [path.with_name(f"{path.name}.{index}") for index in range(1, ROTATED_SEGMENTS + 1)]
    if not previous_identity:
        reads = [(segment, 0) for segment in reversed(segments) if segment.exists()]
    else:
        reads = []
        stored_path, _, stored_inode = previous_identity.rpartition(":")
        stored_path, _, stored_device = stored_path.rpartition(":")
        if stored_path != str(path.resolve()):
            return 0, False
        for position, segment in enumerate(segments):
            try:
                stat = segment.stat()
            except OSError:
                continue
            if (str(stat.st_dev), str(stat.st_ino)) != (stored_device, stored_inode):
                continue
            if not _head_fingerprint(segment).startswith(previous_head):
                # Same inode number, different file: ext4 hands a freed inode to the next file.
                continue
            # `segments[position]` is the previously indexed file; lower positions are newer.
            reads = [
                (segments[index], previous_offset if index == position else 0)
                for index in range(position, -1, -1)
            ]
            break
        else:
            # Every slot taken and none of them the indexed file: it rotated out of retention,
            # and every surviving segment is newer than it. A lone leftover `.1` is not that.
            if all(segment.exists() for segment in segments):
                reads = [(segment, 0) for segment in reversed(segments)]
    inserted = 0
    delivery_updated = False
    # Rotated files take no more appends, so their progress needs no checkpoint.
    for segment, offset in reads:
        changed, updated, _, _ = _ingest_lines(conn, segment, offset)
        inserted += changed
        delivery_updated = delivery_updated or updated
    return inserted, delivery_updated


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
    if raw.get("event") == "request_delivery_update":
        # The normal success row is written before a non-streaming body is handed to the
        # caller. If that final socket write fails, this append-only correction preserves
        # the real delivery verdict without rewriting routes.jsonl.
        if raw.get("final_reason") != "client_disconnected":
            return None
        return {
            "_event": "request_delivery_update",
            "request_id": request_id,
            "reason": "client_disconnected",
            "error_type": _str_or_none(raw.get("error_type")),
        }

    raw_attempts = raw.get("attempts")
    attempts = (
        [{k: a.get(k) for k in _ATTEMPT_KEYS if k in a} for a in raw_attempts if isinstance(a, dict)]
        if isinstance(raw_attempts, list)
        else []
    )
    # (position, source, upstream, model, reason, first byte, first delta): the request-level
    # columns of `_ATTEMPT_COLUMNS` are added at insert. Only a non-empty string names a
    # provider, and timings pass the same finite, non-negative filter as the request's own.
    attempt_rows = [
        (
            position,
            *(
                value if isinstance(value, str) and value else None
                for value in (attempt.get("source"), attempt.get("upstream"), attempt.get("model"), attempt.get("reason"))
            ),
            _seconds(attempt.get("first_byte_seconds")),
            _seconds(attempt.get("first_delta_seconds")),
        )
        for position, attempt in enumerate(attempts)
    ]
    last_attempt = attempts[-1] if attempts else None
    error_type: str | None = None
    if last_attempt is not None:
        candidate = last_attempt.get("error_type")
        if isinstance(candidate, str) and candidate:
            error_type = candidate
    if error_type is None:
        # A handler failure (400/500/503) never reached an attempt: the exception class is on the row itself.
        error_type = _str_or_none(raw.get("error_type"))

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
    workload = _str_or_none(raw.get("workload"))
    synthetic = bool(raw.get("synthetic_case_id")) or (
        workload is not None and _NON_ORGANIC_WORKLOAD.match(workload.lower()) is not None
    )

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
        "prompt_tokens_estimate": _safe_int(raw.get("prompt_tokens_estimate"), None),
        "excluded_for_context": _safe_int(raw.get("excluded_for_context"), None),
        "demoted_for_context": _safe_int(raw.get("demoted_for_context"), None),
        "context_required_tokens": _token_count(raw.get("context_required_tokens")),
        "context_largest_in_pool": _token_count(raw.get("context_largest_in_pool")),
        "prompt_estimate_tokens": _token_count(raw.get("prompt_estimate_tokens")),
        "context_changed_selection": (
            None if raw.get("context_changed_selection") is None else int(bool(raw.get("context_changed_selection")))
        ),
        "attempts_json": json.dumps(attempts, ensure_ascii=False, sort_keys=True),
        "_attempt_rows": attempt_rows,
        "detail_json": _refusal_detail_json(raw),
        "delivery_corrected": 0,
        "first_byte_seconds": _seconds(raw.get("first_byte_seconds")),
        "first_delta_seconds": _seconds(raw.get("first_delta_seconds")),
        "route_prep_seconds": _seconds(raw.get("route_prep_seconds")),
        "router_overhead_seconds": _seconds(raw.get("router_overhead_seconds")),
        "workload": workload,
        "synthetic": 1 if synthetic else 0,
        "policy_profile": _str_or_none(raw.get("policy_profile")),
    }


def _seconds(value: Any) -> float | None:
    """A finite, non-negative duration, or None. Same defence as ``_price_per_token``: a
    hand-edited or foreign line must not put inf or a negative into a percentile."""
    if isinstance(value, bool):
        return None
    seconds = _safe_float(value)
    return seconds if seconds is not None and seconds >= 0 and math.isfinite(seconds) else None


def _refusal_detail_json(raw: dict[str, Any]) -> str | None:
    """Why a router-owned refusal happened, rebuilt field by field: a refusal row has no
    attempts, so without this the Requests page could only repeat the bare reason code.
    Same redaction discipline as the attempts: known keys and scalar types only."""
    detail: dict[str, Any] = {}
    refusal = raw.get("refusal")
    if isinstance(refusal, dict):
        kept: dict[str, Any] = {}
        for key in ("scope", "block_reason"):
            if isinstance(refusal.get(key), str) and refusal[key]:
                kept[key] = refusal[key]
        excluded = refusal.get("excluded")
        if isinstance(excluded, dict):
            counts = {
                str(reason): count
                for reason, count in excluded.items()
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            }
            if counts:
                kept["excluded"] = counts
        if kept:
            detail["refusal"] = kept
    retry_after = raw.get("retry_after_seconds")
    if isinstance(retry_after, int) and not isinstance(retry_after, bool) and retry_after > 0:
        detail["retry_after_seconds"] = retry_after
    return json.dumps(detail, ensure_ascii=False, sort_keys=True) if detail else None


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


def _insert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> tuple[int, bool]:
    if not rows:
        return 0, False
    before = conn.total_changes
    attempt_rows: list[tuple[Any, ...]] = []
    delivery_updated = False
    # Preserve source order: a route and its delivery correction can land in the same
    # ingest batch, and the correction must see the row that immediately precedes it.
    for row in rows:
        if row.get("_event") == "request_delivery_update":
            update_before = conn.total_changes
            conn.execute(
                """
                UPDATE requests
                SET rowid = (SELECT COALESCE(MAX(rowid), 0) + 1 FROM requests),
                    reason = ?, error_type = ?, delivery_corrected = 1
                WHERE request_id = ? AND (reason IS NOT ? OR error_type IS NOT ?)
                """,
                (
                    row["reason"],
                    row["error_type"],
                    row["request_id"],
                    row["reason"],
                    row["error_type"],
                ),
            )
            delivery_updated = delivery_updated or conn.total_changes > update_before
            continue
        insert_before = conn.total_changes
        conn.execute(_INSERT_SQL, tuple(row[col] for col in _COLUMNS))
        if conn.total_changes > insert_before:
            # Only for the row that was kept: a replayed request_id is ignored, attempts included.
            organic_stream = 1 if row["stream"] and not row["synthetic"] else 0
            attempt_rows.extend(
                (row["request_id"], position, row["ts"], row["profile"], organic_stream, *rest)
                for position, *rest in row["_attempt_rows"]
            )
    changed = conn.total_changes - before
    conn.executemany(_INSERT_ATTEMPT_SQL, attempt_rows)
    return changed, delivery_updated


def _trim(conn: sqlite3.Connection, now: float | None = None) -> None:
    # The count-based trim only runs when the table is over the cap. It deletes exactly the
    # excess, oldest first, straight off idx_requests_ts: once the index sits at the cap every
    # ingest is over it by a few rows, and keeping the newest MAX_ROWS with a NOT IN sub-select
    # sorted all 100k of them each time. The age purge is sargable on the same index, so it
    # runs cheaply on every catch-up — enforcing retention even on a quiescent log.
    # A request's attempt rows go first, selected through the parent's own indexes, so both
    # tables always hold the same requests.
    total = int(conn.execute("SELECT COUNT(*) AS c FROM requests").fetchone()["c"] or 0)
    if total > MAX_ROWS:
        oldest = "SELECT {} FROM requests ORDER BY ts ASC, rowid ASC LIMIT ?"
        conn.execute(
            f"DELETE FROM request_attempts WHERE request_id IN ({oldest.format('request_id')})",
            (total - MAX_ROWS,),
        )
        conn.execute(f"DELETE FROM requests WHERE rowid IN ({oldest.format('rowid')})", (total - MAX_ROWS,))
    cutoff = (now if now is not None else time.time()) - MAX_AGE_SECONDS
    conn.execute(
        "DELETE FROM request_attempts WHERE request_id IN ("
        "SELECT request_id FROM requests WHERE ts IS NOT NULL AND ts < ?)",
        (cutoff,),
    )
    conn.execute("DELETE FROM requests WHERE ts IS NOT NULL AND ts < ?", (cutoff,))


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _facets(
    conn: sqlite3.Connection,
    cutoff: float,
    reference: float,
    first_bucket: int,
    bucket: int,
) -> dict[str, Any]:
    """The per-source/reason/status/profile breakdowns and the timeline, from one scan.

    One GROUP BY over every facet at once instead of a scan per facet: the window holds few
    distinct combinations (they are correlated, and a window has at most ~30 buckets), so
    folding them here is cheaper than re-reading the window five times. A NULL facet value is
    left out of that facet's breakdown but still counts in the others. Equal totals rank by
    descending key, the order the former per-facet ``GROUP BY ... ORDER BY total DESC`` returned.
    """
    groups: dict[str, dict[Any, list[int]]] = {name: {} for name in ("source", "reason", "status", "profile")}
    timeline: dict[int, tuple[int, int]] = {}
    for row in conn.execute(
        """
        SELECT source, reason, status, profile,
               CAST((ts - ?) / ? AS INT) * ? + ? AS b,
               COUNT(*) AS total
        FROM requests WHERE ts >= ? AND ts <= ?
        GROUP BY source, reason, status, profile, b
        """,
        (first_bucket, bucket, bucket, first_bucket, cutoff, reference),
    ):
        total = int(row["total"])
        ok = total if row["reason"] == "ok" else 0
        for name, counts in groups.items():
            key = row[name]
            if key is None:
                continue
            entry = counts.setdefault(key, [0, 0])
            entry[0] += total
            entry[1] += ok
        slot_total, slot_ok = timeline.get(int(row["b"]), (0, 0))
        timeline[int(row["b"])] = (slot_total + total, slot_ok + ok)
    facets: dict[str, Any] = {}
    for name, counts in groups.items():
        # Two stable passes: descending key, then descending total.
        ranked = sorted(sorted(counts.items(), reverse=True), key=lambda item: -item[1][0])
        facets[name] = [{name: key, "total": total, "ok": ok} for key, (total, ok) in ranked]
    facets["timeline"] = timeline
    return facets


def _distributions(
    conn: sqlite3.Connection, cutoff: float, reference: float
) -> tuple[list[float], list[float], list[float]]:
    """The window's latencies, router overheads and estimate ratios, each sorted, from one scan.

    Each column is the row's value, or NULL where the row is outside that distribution, so one
    pass over ``idx_requests_summary`` replaces a scan per distribution. Ranked in Python: SQLite's
    window functions measured slower than handing these narrow tuples over, and they would raise
    the SQLite floor. CAST, because both token columns are INTEGER and SQLite would otherwise
    divide them as integers.
    """
    latencies: list[float] = []
    overhead: list[float] = []
    ratios: list[float] = []
    cursor = conn.cursor()
    cursor.row_factory = None
    for latency, router_overhead, ratio in cursor.execute(
        """
        SELECT latency_seconds,
               CASE WHEN synthetic = 0 THEN router_overhead_seconds END,
               CASE WHEN synthetic = 0 AND prompt_estimate_tokens > 0 AND prompt_tokens >= ?
                    THEN CAST(prompt_estimate_tokens AS REAL) / prompt_tokens END
        FROM requests WHERE ts >= ? AND ts <= ?
        """,
        (ESTIMATE_RATIO_MIN_PROMPT_TOKENS, cutoff, reference),
    ):
        if latency is not None:
            latencies.append(latency)
        if router_overhead is not None:
            overhead.append(router_overhead)
        if ratio is not None:
            ratios.append(ratio)
    latencies.sort()
    overhead.sort()
    ratios.sort()
    return latencies, overhead, ratios


def _ttft_summary(
    conn: sqlite3.Connection,
    cutoff: float,
    reference: float,
    *,
    known_sources: list[str],
    known_profiles: list[str],
) -> dict[str, Any]:
    """Time-to-first-token percentiles, one observation per streamed attempt that produced a delta.

    TTFT is ``first_delta_seconds``: the first substantive delta relayed to the client, reasoning
    included, measured from the provider send. It exists only on the streaming path, so a buffered
    answer (whose first upstream byte is its whole latency) and an attempt that failed before any
    delta contribute nothing. Attribution is per attempt: a failed-over request counts for each
    provider that streamed a delta, not for the provider that finally answered. A provider seen
    in the window without any observation stays at ``n == 0``, shown as "no streaming data"
    rather than as a number that would read as fast.
    """
    per_source: dict[str, list[float]] = {}
    per_profile: dict[str, list[float]] = {profile: [] for profile in known_profiles}
    overall: list[float] = []
    # One row per attempt of an organic streamed request, straight off the covering
    # idx_request_attempts_ttft: nothing re-parses attempts_json. Every such attempt registers its
    # provider, so one that only ever failed before a delta still appears (as "no streaming data")
    # instead of vanishing from the table. Every profile of the window is already registered.
    cursor = conn.cursor()
    cursor.row_factory = None
    for source, profile, delta in cursor.execute(
        "SELECT source, profile, first_delta_seconds FROM request_attempts "
        "WHERE organic_stream = 1 AND ts >= ? AND ts <= ?",
        (cutoff, reference),
    ):
        observed = per_source.setdefault(source, []) if source is not None else None
        if delta is None:
            continue
        overall.append(delta)
        if observed is not None:
            observed.append(delta)
        if profile is not None:
            per_profile.setdefault(profile, []).append(delta)
    for source in known_sources:
        per_source.setdefault(source, [])

    def entries(groups: dict[str, list[float]], key: str) -> list[dict[str, Any]]:
        ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        return [{key: name, **_ttft_stats(values)} for name, values in ranked]

    return {
        "overall": _ttft_stats(overall),
        "by_source": entries(per_source, "source"),
        "by_profile": entries(per_profile, "profile"),
    }


def _ttft_stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": _percentile(ordered, 50),
        "p75": _percentile(ordered, 75),
        "p95": _percentile(ordered, 95),
        "state": "ok" if ordered else "no_streaming_data",
    }


def _percentile(sorted_values: list[float], pct: int) -> float | None:
    count = len(sorted_values)
    if count == 0:
        return None
    # Nearest-rank: 1-based ceil(pct/100 * count) converted to a 0-based offset.
    # Avoids the upper bias of floor() (e.g. p50 of two samples returning the max).
    offset = max(0, min((count * pct + 99) // 100 - 1, count - 1))
    return float(sorted_values[offset])


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
    # Version first: an older file can lack columns the current indexes cover.
    _ensure_schema_version(conn)
    _create_requests_table(conn)
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
          prompt_tokens_estimate INTEGER,
          excluded_for_context INTEGER,
          demoted_for_context INTEGER,
          context_required_tokens INTEGER,
          context_largest_in_pool INTEGER,
          prompt_estimate_tokens INTEGER,
          context_changed_selection INTEGER,
          attempts_json TEXT,
          detail_json TEXT,
          delivery_corrected INTEGER NOT NULL DEFAULT 0,
          first_byte_seconds REAL,
          first_delta_seconds REAL,
          route_prep_seconds REAL,
          router_overhead_seconds REAL,
          workload TEXT,
          synthetic INTEGER NOT NULL DEFAULT 0,
          policy_profile TEXT
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_requests_request_id ON requests(request_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts)")
    # Covers every column summary() reads, so its window scans never touch the wide rows
    # (attempts_json dominates their size).
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_requests_summary ON requests(
          ts, reason, source, status, profile, fallback, stream, synthetic, latency_seconds,
          router_overhead_seconds, prompt_tokens, completion_tokens, prompt_price, completion_price,
          prompt_estimate_tokens
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS request_attempts (
          request_id TEXT NOT NULL,
          position INTEGER NOT NULL,
          ts REAL,
          profile TEXT,
          organic_stream INTEGER NOT NULL,
          source TEXT,
          upstream_id TEXT,
          model_id TEXT,
          reason TEXT,
          first_byte_seconds REAL,
          first_delta_seconds REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_request_attempts_request_id ON request_attempts(request_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_request_attempts_ttft "
        "ON request_attempts(organic_stream, ts, source, profile, first_delta_seconds)"
    )


def _ensure_schema_version(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = _safe_int(row["value"], None) if row is not None else None
    if current == SCHEMA_VERSION:
        return
    # Derived data; rebuild rather than migrate (a new file has nothing to drop). The caller
    # creates the tables afresh.
    conn.execute("DROP TABLE IF EXISTS requests")
    conn.execute("DROP TABLE IF EXISTS request_attempts")
    # Forgetting the identity too makes the next ingest read the rotated segments as well.
    conn.execute(
        "DELETE FROM meta WHERE key IN ('ingest_offset', 'ingest_source_identity', 'ingest_source_head')"
    )
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
    detail: dict[str, Any] = {}
    if isinstance(row["detail_json"], str) and row["detail_json"]:
        try:
            parsed_detail = json.loads(row["detail_json"])
            if isinstance(parsed_detail, dict):
                detail = parsed_detail
        except ValueError:
            detail = {}
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
        "prompt_tokens_estimate": row["prompt_tokens_estimate"],
        "excluded_for_context": row["excluded_for_context"],
        "first_byte_seconds": row["first_byte_seconds"],
        "first_delta_seconds": row["first_delta_seconds"],
        "route_prep_seconds": row["route_prep_seconds"],
        "router_overhead_seconds": row["router_overhead_seconds"],
        "workload": row["workload"],
        "synthetic": bool(row["synthetic"]),
        "demoted_for_context": row["demoted_for_context"],
        "context_required_tokens": row["context_required_tokens"],
        "context_largest_in_pool": row["context_largest_in_pool"],
        "prompt_estimate_tokens": row["prompt_estimate_tokens"],
        "context_changed_selection": (
            None if row["context_changed_selection"] is None else bool(row["context_changed_selection"])
        ),
        "attempts": attempts,
        "refusal": detail.get("refusal"),
        "retry_after_seconds": detail.get("retry_after_seconds"),
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
