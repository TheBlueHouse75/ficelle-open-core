from __future__ import annotations

import errno
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt


@contextmanager
def file_lock(
    path: Path,
    *,
    blocking: bool = False,
    timeout_seconds: float = 0.0,
) -> Iterator[bool]:
    """Acquire a process-safe advisory lock and yield whether it was obtained.

    ``blocking=True`` without a timeout waits indefinitely on both platforms. Windows offers no
    such primitive: ``msvcrt.LK_LOCK`` retries for about ten seconds and then fails with
    ``EDEADLOCK``, which is neither a wait nor the ``EACCES`` a caller could treat as "held by
    someone else". The retry loop below already provides an unbounded wait, so the Windows branch
    always takes the non-blocking mode and lets the loop do the waiting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    try:
        while True:
            try:
                if fcntl is not None:
                    flags = fcntl.LOCK_EX
                    if not blocking or timeout_seconds > 0:
                        flags |= fcntl.LOCK_NB
                    fcntl.flock(descriptor, flags)
                else:  # pragma: no cover - exercised on Windows
                    if os.fstat(descriptor).st_size == 0:
                        os.write(descriptor, b"0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if not blocking or (deadline is not None and time.monotonic() >= deadline):
                    break
                time.sleep(0.1)
        yield acquired
    finally:
        if acquired:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            else:  # pragma: no cover - exercised on Windows
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)


def probe_lock_path(ficelle_home: Path) -> Path:
    return ficelle_home / "locks" / "provider-probes.lock"


def synthetic_health_lock_path(ficelle_home: Path) -> Path:
    return ficelle_home / "locks" / "synthetic-health.lock"


# The synthetic-correlation contract shared by the router (which validates the header) and
# the harness (which sends it). One home so the two sides can never drift: a case id the
# harness emits is by construction one the router's gate accepts (L2-R3).
SYNTHETIC_CASE_ID_HEADER = "X-Ficelle-Synthetic-Case-Id"
SYNTHETIC_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def synthetic_health_lock_held(ficelle_home: Path) -> bool:
    """True while a synthetic-health run holds this FICELLE_HOME's lock.

    Probes by trying a non-blocking acquire: failing to get the lock means the harness
    process holds it; getting it (released immediately by the context manager) means no
    run is active. The exists() pre-check keeps a router-side probe from creating the lock
    file on every headered request."""
    lock_path = synthetic_health_lock_path(ficelle_home)
    if not lock_path.exists():
        return False
    try:
        with file_lock(lock_path, blocking=False) as acquired:
            return not acquired
    except OSError:
        return False
