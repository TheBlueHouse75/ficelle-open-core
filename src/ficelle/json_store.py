from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO


# The runtime home holds credentials, routing history and admin-action history. Directories were
# created at the process umask (0755 in practice) and the JSONL logs at 0644, so any other local
# account could read them — while the JSON files were already private, because
# NamedTemporaryFile creates at 0600 and os.replace carries that mode across.
RUNTIME_DIR_MODE = 0o700
RUNTIME_FILE_MODE = 0o600


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=RUNTIME_DIR_MODE)
    # `mode=` only applies when mkdir actually creates, so an install predating this keeps its
    # old permissions; tighten it in place rather than leaving history world-readable forever.
    if (path.stat().st_mode & 0o777) != RUNTIME_DIR_MODE:
        path.chmod(RUNTIME_DIR_MODE)


def rotate_if_oversized(path: Path, max_bytes: int, keep: int = 3) -> None:
    """Roll `path` to `path.1` (and shift the rest) once it passes `max_bytes`.

    The append-only JSONL logs had no rotation at all — only the derived SQLite index was
    bounded — so a busy agent host grew them without limit until the disk filled, at which point
    state writes start failing too. The reader tolerates this: the request index detects
    truncation and identity changes and re-reads from the start.
    """
    if max_bytes <= 0:
        return
    try:
        if not path.exists() or path.stat().st_size <= max_bytes:
            return
    except OSError:
        return
    try:
        oldest = path.with_name(f"{path.name}.{keep}")
        if oldest.exists():
            oldest.unlink()
        for index in range(keep - 1, 0, -1):
            source = path.with_name(f"{path.name}.{index}")
            if source.exists():
                source.replace(path.with_name(f"{path.name}.{index + 1}"))
        path.replace(path.with_name(f"{path.name}.1"))
    except OSError:
        # Losing a rotation is survivable; losing the write that follows it is not.
        return


def open_private_append(path: Path) -> TextIO:
    """Open an append-only runtime log that must not be world-readable.

    Creating at 0600 rather than creating-then-chmod'ing closes the window where the file exists
    with default permissions, and re-tightening covers logs an earlier version already created.
    """
    ensure_private_dir(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, RUNTIME_FILE_MODE)
    try:
        if (os.fstat(fd).st_mode & 0o777) != RUNTIME_FILE_MODE:
            os.fchmod(fd, RUNTIME_FILE_MODE)
        return os.fdopen(fd, "a", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise


def write_private_text(path: Path, text: str) -> None:
    """Replace a file's contents without ever exposing them at the process umask.

    Raises on a failed permission change rather than swallowing it: the callers are writing
    secrets, and a silent fallback leaves a readable credential file with no warning.
    """
    ensure_private_dir(path.parent)
    # Do not pass O_TRUNC here: an older 0644 credential file must be tightened *before* its
    # replacement contents are visible. O_CREAT's mode is ignored for existing files.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, RUNTIME_FILE_MODE)
    try:
        if (os.fstat(fd).st_mode & 0o777) != RUNTIME_FILE_MODE:
            os.fchmod(fd, RUNTIME_FILE_MODE)
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1  # The TextIO wrapper now owns and closes this descriptor.
        with handle:
            handle.seek(0)
            handle.truncate()
            handle.write(text)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise


def sweep_orphan_temp_files(directory: Path, *, max_age_seconds: float = 3600.0) -> int:
    """Delete `*.tmp` leftovers from a write that was killed mid-flight.

    `atomic_write_json` unlinks its temp file on an in-process exception, but a SIGKILL or a power
    loss between create and rename leaves one behind for good. Only files older than an hour are
    swept, so a write in flight in another process is never touched.
    """
    removed = 0
    try:
        candidates = list(directory.glob("*.tmp"))
    except OSError:
        return 0
    cutoff = time.time() - max_age_seconds
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    ensure_private_dir(path.parent)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # Write to a *unique* temp file in the same directory, then atomically rename.
    # A fixed `<name>.tmp` path would let concurrent writers of the same file (e.g.
    # the startup catalog warm thread and a request-triggered refresh) clobber each
    # other's temp file and promote a truncated/interleaved JSON, or fail the second
    # rename. A per-write temp name makes concurrent writes safe (last writer wins).
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
