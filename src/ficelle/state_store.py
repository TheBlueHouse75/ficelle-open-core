from __future__ import annotations

import copy
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from ficelle.json_store import atomic_write_json, ensure_private_dir, load_json

try:
    import fcntl
except Exception:  # pragma: no cover - Windows fallback for local imports/tests
    fcntl = None


RUNTIME_STATE_HISTORY_KEYS = ("benchmark_results", "verified_capabilities")


@contextmanager
def advisory_file_lock(path: Path, thread_lock: threading.RLock) -> Iterator[None]:
    """Serialize a critical section across threads and, where supported, processes."""
    with thread_lock:
        ensure_private_dir(path.parent)
        with path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_iso_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    # Ficelle writes UTC timestamps; legacy or hand-edited naive values should stay UTC
    # instead of being reinterpreted in the host's local timezone.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def runtime_evidence_timestamp_value(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    value = row.get("verified_at") or row.get("failed_at") or row.get("ran_at")
    return str(value) if value else None


def runtime_row_timestamp(row: Any) -> float | None:
    return parse_iso_timestamp(runtime_evidence_timestamp_value(row))


def state_file_summary(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False, "size_bytes": 0, "modified_at": None}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def runtime_state_history_counts(state: Any, key: str) -> dict[str, Any]:
    history = state.get(key) if isinstance(state, dict) else {}
    if not isinstance(history, dict):
        return {"model_count": 0, "unique_profile_count": 0, "evidence_row_count": 0, "latest_evidence_at": None}
    profile_ids: set[str] = set()
    evidence_row_count = 0
    latest_timestamp: float | None = None
    latest_evidence_at: str | None = None
    for raw_profiles in history.values():
        if not isinstance(raw_profiles, dict):
            continue
        evidence_row_count += len(raw_profiles)
        for profile_id, row in raw_profiles.items():
            profile_ids.add(str(profile_id))
            evidence_at = runtime_evidence_timestamp_value(row)
            stamp = parse_iso_timestamp(evidence_at)
            if stamp is not None and (latest_timestamp is None or stamp > latest_timestamp):
                latest_timestamp = stamp
                latest_evidence_at = evidence_at
    return {
        "model_count": len(history),
        "unique_profile_count": len(profile_ids),
        "evidence_row_count": evidence_row_count,
        "latest_evidence_at": latest_evidence_at,
    }


def merge_runtime_history_rows(current: Any, incoming: Any) -> Any:
    """Merge two history levels, newest evidence winning, copying each surviving row once.

    The previous shape deep-copied *every* current row up front and then overwrote most of them
    with a copy of the incoming row, so the losing copy was pure waste — 17.4 ms of a 35.9 ms
    state write on a real 949 KB state. Deciding first and copying once is the same result:
    every branch below reproduces the original outcome, including the two that keep the current
    row (incoming is older, or incoming is not a dict).
    """
    if not (isinstance(current, dict) and isinstance(incoming, dict)):
        return copy.deepcopy(incoming)
    merged: dict[Any, Any] = {}
    for key, current_value in current.items():
        if key not in incoming:
            merged[key] = copy.deepcopy(current_value)
            continue
        incoming_value = incoming[key]
        if isinstance(current_value, dict) and isinstance(incoming_value, dict):
            current_stamp = runtime_row_timestamp(current_value)
            incoming_stamp = runtime_row_timestamp(incoming_value)
            if current_stamp is not None or incoming_stamp is not None:
                if incoming_stamp is not None and (current_stamp is None or incoming_stamp >= current_stamp):
                    merged[key] = copy.deepcopy(incoming_value)
                else:
                    merged[key] = copy.deepcopy(current_value)
            else:
                # Neither side is an evidence row (this is the model level); recurse to reach the
                # per-profile rows that do carry timestamps.
                merged[key] = merge_runtime_history_rows(current_value, incoming_value)
        elif isinstance(current_value, dict):
            # Incoming is not a row-shaped value: the stored history wins rather than being
            # replaced by whatever this writer happened to hold.
            merged[key] = copy.deepcopy(current_value)
        else:
            merged[key] = copy.deepcopy(incoming_value)
    for key, incoming_value in incoming.items():
        if key not in current:
            merged[key] = copy.deepcopy(incoming_value)
    return merged


def merge_runtime_state_for_write(current: Any, incoming: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(current, dict):
        return incoming
    # Shallow, not deep: the only writes below replace whole top-level history keys, so nothing
    # nested inside `incoming` is ever mutated and the caller's object is safe either way. The
    # deep copy this replaces duplicated the entire state — ~5 ms of every write — to protect
    # against a mutation that does not happen. The result is serialized, never mutated further.
    merged = dict(incoming)
    for key in RUNTIME_STATE_HISTORY_KEYS:
        current_value = current.get(key)
        incoming_value = incoming.get(key)
        if isinstance(current_value, dict):
            if isinstance(incoming_value, dict):
                merged[key] = merge_runtime_history_rows(current_value, incoming_value)
            else:
                merged[key] = copy.deepcopy(current_value)
    return merged


@dataclass
class StateStore:
    state_path: Path
    lock_path: Path
    backup_dir: Path
    backup_keep: int = 20
    backup_min_interval_seconds: int = 600
    history_keys: tuple[str, ...] = RUNTIME_STATE_HISTORY_KEYS
    fallback_state_path: Path | None = None
    fallback_backup_dir: Path | None = None
    thread_lock: threading.RLock = field(default_factory=threading.RLock)
    last_update_reason: str | None = field(default=None, init=False)

    @contextmanager
    def lock(self) -> Iterator[None]:
        with advisory_file_lock(self.lock_path, self.thread_lock):
            yield

    def read_state_path(self) -> Path:
        if self.state_path.exists() or self.fallback_state_path is None:
            return self.state_path
        return (
            self.fallback_state_path
            if self.fallback_state_path.exists()
            else self.state_path
        )

    def load(self, default: Any | None = None) -> Any:
        fallback = {} if default is None else default
        path = self.read_state_path()
        loaded = load_json(path, fallback)
        if loaded is not fallback or not path.exists():
            return loaded
        # `load_json` cannot tell "absent" from "unparseable", and resetting to {} here does not
        # merely lose this read: the next write backs up the *reset* state, and with 20 kept
        # backups at a 600 s floor the pre-corruption ones are rotated away within hours. So a
        # disk fault or a hand edit silently erased every cooldown, quarantine and benchmark row.
        recovered = self._restore_from_backup()
        if recovered is not None:
            return recovered
        return loaded

    def _restore_from_backup(self) -> dict[str, Any] | None:
        """Newest parseable backup, or None when there is nothing to fall back to."""
        for candidate in reversed(self.backup_files()):
            restored = load_json(candidate, None)
            if isinstance(restored, dict):
                print(
                    f"ficelle: {self.read_state_path()} was unreadable; recovered runtime state "
                    f"from {candidate}.",
                    flush=True,
                )
                return restored
        print(
            f"ficelle: {self.read_state_path()} was unreadable and no parseable backup was "
            "found; starting from empty runtime state.",
            flush=True,
        )
        return None

    def snapshot(self) -> dict[str, Any]:
        state = self.load({})
        return state if isinstance(state, dict) else {}

    @staticmethod
    def _backup_files(directory: Path) -> list[Path]:
        try:
            candidates = list(directory.iterdir())
        except OSError:
            return []
        files: list[tuple[float, str, Path]] = []
        for path in candidates:
            if not path.name.startswith("state-") or path.suffix != ".json":
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, path.name, path))
        return [path for _, _, path in sorted(files)]

    def canonical_backup_files(self) -> list[Path]:
        return self._backup_files(self.backup_dir)

    def fallback_backup_files(self) -> list[Path]:
        if self.fallback_backup_dir is None or self.fallback_backup_dir == self.backup_dir:
            return []
        return self._backup_files(self.fallback_backup_dir)

    def backup_files(self) -> list[Path]:
        canonical = self.canonical_backup_files()
        return canonical if canonical else self.fallback_backup_files()

    def rotate_backups(self, keep: int | None = None) -> None:
        files = self.canonical_backup_files()
        for path in files[: max(0, len(files) - (self.backup_keep if keep is None else keep))]:
            try:
                path.unlink()
            except OSError:
                pass

    def backup(self) -> Path | None:
        if not self.state_path.exists():
            return None
        latest_backup = self.canonical_backup_files()[-1:] or []
        if latest_backup:
            try:
                age = time.time() - latest_backup[0].stat().st_mtime
            except OSError:
                age = self.backup_min_interval_seconds
            if age < self.backup_min_interval_seconds:
                return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.backup_dir / f"state-{stamp}-{uuid.uuid4().hex[:8]}.json"
        try:
            ensure_private_dir(self.backup_dir)
            shutil.copy2(self.state_path, path)
            path.touch()
            self.rotate_backups()
        except OSError:
            return None
        return path

    def merge_for_write(self, current: Any, incoming: dict[str, Any]) -> dict[str, Any]:
        if self.history_keys == RUNTIME_STATE_HISTORY_KEYS:
            return merge_runtime_state_for_write(current, incoming)
        merged = merge_runtime_state_for_write(current, incoming)
        if not isinstance(current, dict):
            return merged
        for key in self.history_keys:
            if key in RUNTIME_STATE_HISTORY_KEYS:
                continue
            current_value = current.get(key)
            incoming_value = incoming.get(key)
            if isinstance(current_value, dict):
                merged[key] = copy.deepcopy(incoming_value) if isinstance(incoming_value, dict) else copy.deepcopy(current_value)
        return merged

    def write(self, state: dict[str, Any]) -> None:
        with self.lock():
            self.backup()
            current = self.load({})
            atomic_write_json(self.state_path, self.merge_for_write(current, state))

    def update(
        self,
        mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
        reason: str | None = None,
    ) -> dict[str, Any]:
        with self.lock():
            self.backup()
            current = self.load({})
            state = copy.deepcopy(current) if isinstance(current, dict) else {}
            updated = mutator(state)
            next_state = updated if isinstance(updated, dict) else state
            merged = self.merge_for_write(current, next_state)
            atomic_write_json(self.state_path, merged)
            self.last_update_reason = reason
            return merged

    def diagnostics(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime_state = self.snapshot() if state is None else state
        if not isinstance(runtime_state, dict):
            runtime_state = {}
        backups = self.backup_files()
        selected_backup_dir = backups[0].parent if backups else self.backup_dir
        latest_backup = backups[-1] if backups else None
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": state_file_summary(self.read_state_path()),
            "protected_history_keys": list(self.history_keys),
            "benchmark_results": runtime_state_history_counts(runtime_state, "benchmark_results"),
            "verified_capabilities": runtime_state_history_counts(runtime_state, "verified_capabilities"),
            "backups": {
                "dir": str(selected_backup_dir),
                "keep": self.backup_keep,
                "min_interval_seconds": self.backup_min_interval_seconds,
                "count": len(backups),
                "latest": state_file_summary(latest_backup) if latest_backup else None,
            },
        }
