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

from ficelle.json_store import atomic_write_json, load_json

try:
    import fcntl
except Exception:  # pragma: no cover - Windows fallback for local imports/tests
    fcntl = None


RUNTIME_STATE_HISTORY_KEYS = ("benchmark_results", "verified_capabilities")


@contextmanager
def advisory_file_lock(path: Path, thread_lock: threading.RLock) -> Iterator[None]:
    """Serialize a critical section across threads and, where supported, processes."""
    with thread_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
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
    if isinstance(current, dict) and isinstance(incoming, dict):
        merged = {key: copy.deepcopy(value) for key, value in current.items()}
        for key, incoming_value in incoming.items():
            current_value = merged.get(key)
            if isinstance(current_value, dict) and isinstance(incoming_value, dict):
                current_stamp = runtime_row_timestamp(current_value)
                incoming_stamp = runtime_row_timestamp(incoming_value)
                if current_stamp is not None or incoming_stamp is not None:
                    if incoming_stamp is not None and (current_stamp is None or incoming_stamp >= current_stamp):
                        merged[key] = copy.deepcopy(incoming_value)
                    continue
                merged[key] = merge_runtime_history_rows(current_value, incoming_value)
            elif isinstance(current_value, dict) and not isinstance(incoming_value, dict):
                continue
            else:
                merged[key] = copy.deepcopy(incoming_value)
        return merged
    return copy.deepcopy(incoming)


def merge_runtime_state_for_write(current: Any, incoming: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(current, dict):
        return incoming
    merged = copy.deepcopy(incoming)
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
    thread_lock: threading.RLock = field(default_factory=threading.RLock)
    last_update_reason: str | None = field(default=None, init=False)

    @contextmanager
    def lock(self) -> Iterator[None]:
        with advisory_file_lock(self.lock_path, self.thread_lock):
            yield

    def load(self, default: Any | None = None) -> Any:
        return load_json(self.state_path, {} if default is None else default)

    def snapshot(self) -> dict[str, Any]:
        state = self.load({})
        return state if isinstance(state, dict) else {}

    def backup_files(self) -> list[Path]:
        try:
            candidates = list(self.backup_dir.iterdir())
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

    def rotate_backups(self, keep: int | None = None) -> None:
        files = self.backup_files()
        for path in files[: max(0, len(files) - (self.backup_keep if keep is None else keep))]:
            try:
                path.unlink()
            except OSError:
                pass

    def backup(self) -> Path | None:
        if not self.state_path.exists():
            return None
        latest_backup = self.backup_files()[-1:] or []
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
            self.backup_dir.mkdir(parents=True, exist_ok=True)
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
            current = load_json(self.state_path, {})
            atomic_write_json(self.state_path, self.merge_for_write(current, state))

    def update(
        self,
        mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
        reason: str | None = None,
    ) -> dict[str, Any]:
        with self.lock():
            self.backup()
            current = load_json(self.state_path, {})
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
        latest_backup = backups[-1] if backups else None
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": state_file_summary(self.state_path),
            "protected_history_keys": list(self.history_keys),
            "benchmark_results": runtime_state_history_counts(runtime_state, "benchmark_results"),
            "verified_capabilities": runtime_state_history_counts(runtime_state, "verified_capabilities"),
            "backups": {
                "dir": str(self.backup_dir),
                "keep": self.backup_keep,
                "min_interval_seconds": self.backup_min_interval_seconds,
                "count": len(backups),
                "latest": state_file_summary(latest_backup) if latest_backup else None,
            },
        }
