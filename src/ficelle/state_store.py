from __future__ import annotations

import copy
import json
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
from ficelle.probe_lock import file_lock

try:
    import fcntl
except Exception:  # pragma: no cover - Windows fallback for local imports/tests
    fcntl = None


RUNTIME_STATE_HISTORY_KEYS = ("benchmark_results", "verified_capabilities", "quality_feedback")

# Per-model runtime sections, all keyed by `cooldown_key` (`source::upstream_id`), that the
# catalog-presence prune deletes for a model gone from its provider for good. `quarantine` is
# deliberately absent: it records a billing/paid verdict, and a model that comes back under the
# same id must not silently lose it. `quality_feedback` expires on its own TTL.
PRUNABLE_MODEL_SECTIONS = (
    "stats",
    "successes",
    "model_errors",
    "cooldowns",
    "benchmark_results",
    "verified_capabilities",
    "request_token_limits",
    "unsupported_parameters",
)
# Owned by `StateStore.reconcile_model_presence` (and `clear_listed_model_presence`, which every
# catalog publish runs for the models it lists): `first_absent` (key -> ISO time a full
# listing of the model's provider first lacked it) and `pruned` (key -> ISO time it was pruned,
# the tombstone that stops a stale whole-state write from resurrecting the deleted rows).
MODEL_PRESENCE_KEY = "model_presence"
MODEL_ABSENCE_RETENTION_SECONDS = 30 * 24 * 3600
# A stale snapshot older than this is not a writer still in flight.
MODEL_PRUNE_TOMBSTONE_SECONDS = 7 * 24 * 3600


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
    value = row.get("verified_at") or row.get("failed_at") or row.get("ran_at") or row.get("recorded_at")
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


def merge_quality_feedback_history(current: Any, incoming: Any, *, now: float | None = None) -> dict[str, Any]:
    """Preserve active feedback slots from stale writers while letting expiry pruning stick.

    Quality feedback is a revisioned slot ledger, unlike benchmark history's timestamped
    per-profile rows. Its entries must merge by revision and omit expired slots, otherwise a
    stale whole-state repair could either drop a correction or resurrect evidence a write pruned.
    """
    reference = time.time() if now is None else now
    current_models = current if isinstance(current, dict) else {}
    incoming_models = incoming if isinstance(incoming, dict) else {}
    merged: dict[str, Any] = {}
    for model_key in set(current_models) | set(incoming_models):
        current_profiles = current_models.get(model_key)
        incoming_profiles = incoming_models.get(model_key)
        if not isinstance(current_profiles, dict) and not isinstance(incoming_profiles, dict):
            continue
        profiles: dict[str, Any] = {}
        current_profiles = current_profiles if isinstance(current_profiles, dict) else {}
        incoming_profiles = incoming_profiles if isinstance(incoming_profiles, dict) else {}
        for profile_id in set(current_profiles) | set(incoming_profiles):
            current_slots = _quality_feedback_slots(current_profiles.get(profile_id), reference)
            incoming_slots = _quality_feedback_slots(incoming_profiles.get(profile_id), reference)
            slots: dict[str, Any] = {}
            for slot_key in set(current_slots) | set(incoming_slots):
                current_row = current_slots.get(slot_key)
                incoming_row = incoming_slots.get(slot_key)
                if current_row is None:
                    slots[str(slot_key)] = copy.deepcopy(incoming_row)
                    continue
                if incoming_row is None:
                    slots[str(slot_key)] = copy.deepcopy(current_row)
                    continue
                current_revision = _quality_feedback_revision(current_row)
                incoming_revision = _quality_feedback_revision(incoming_row)
                if incoming_revision is not None and (current_revision is None or incoming_revision > current_revision):
                    slots[str(slot_key)] = copy.deepcopy(incoming_row)
                else:
                    slots[str(slot_key)] = copy.deepcopy(current_row)
            if slots:
                profiles[str(profile_id)] = {"event_slots": slots}
        if profiles:
            merged[str(model_key)] = profiles
    return merged


def _quality_feedback_slots(ledger: Any, now: float) -> dict[Any, dict[str, Any]]:
    slots = ledger.get("event_slots") if isinstance(ledger, dict) else None
    if not isinstance(slots, dict):
        return {}
    active: dict[Any, dict[str, Any]] = {}
    for slot_key, row in slots.items():
        expires_at = parse_iso_timestamp(row.get("expires_at")) if isinstance(row, dict) else None
        if expires_at is not None and expires_at > now:
            active[slot_key] = row
    return active


def _quality_feedback_revision(row: dict[str, Any]) -> int | None:
    revision = row.get("revision")
    return revision if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 1 else None


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
        if key == "quality_feedback":
            combined = merge_quality_feedback_history(current_value, incoming_value)
            if combined:
                merged[key] = combined
            else:
                merged.pop(key, None)
            continue
        if current_value == incoming_value:
            # A mutator that never touched this history key hands back exactly what is stored, and
            # merging two equal values reproduces them. C-speed equality is far cheaper than the
            # row-by-row merge that most writes (a spend counter, a cooldown) do not need.
            continue
        if isinstance(current_value, dict):
            if isinstance(incoming_value, dict):
                merged[key] = merge_runtime_history_rows(current_value, incoming_value)
            else:
                merged[key] = copy.deepcopy(current_value)
    _keep_model_presence_ledger(current, merged)
    return merged


def _keep_model_presence_ledger(current: dict[str, Any], merged: dict[str, Any]) -> None:
    """The presence ledger and its tombstones survive any write but the prune and the publish-time clear.

    Without this, the history merge above would re-add every pruned benchmark row the moment a
    writer holding a pre-prune snapshot wrote it back, and a stale ledger could restore an
    absence clock a relisting had already cleared. Only a key the store holds no row for is
    dropped, so a writer adding to a section never loses an unrelated model.
    """
    presence = current.get(MODEL_PRESENCE_KEY)
    if not isinstance(presence, dict):
        merged.pop(MODEL_PRESENCE_KEY, None)
        return
    merged[MODEL_PRESENCE_KEY] = presence
    pruned = presence.get("pruned")
    if not isinstance(pruned, dict) or not pruned:
        return
    for section in PRUNABLE_MODEL_SECTIONS:
        incoming_rows = merged.get(section)
        if not isinstance(incoming_rows, dict):
            continue
        current_rows = current.get(section)
        current_rows = current_rows if isinstance(current_rows, dict) else {}
        resurrected = {key for key in pruned if key in incoming_rows and key not in current_rows}
        if resurrected:
            merged[section] = {key: value for key, value in incoming_rows.items() if key not in resurrected}


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
    _shared: tuple[tuple[str, int, int, int], dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _shared_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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

    def shared_snapshot(self) -> dict[str, Any]:
        """Parse the state file at most once per version of it, for readers that never mutate.

        A request reads the state four times (token caps, selection, per-attempt parameters,
        competence); each was a full `json.loads`. Keyed on the file's identity because writes
        publish by rename. The top level is a fresh dict per call, but nested values are shared,
        so callers must treat them as read-only, as with the catalog document cache. Anything
        that is not a clean parse of the real file (absent, corrupt, backup recovery) is not cached.
        """
        path = self.read_state_path()
        try:
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            return self.snapshot()
        with self._shared_lock:
            cached = self._shared
        if cached is not None and cached[0] == signature:
            return dict(cached[1])
        try:
            parsed = json.loads(path.read_text())
        except Exception:
            return self.snapshot()
        if not isinstance(parsed, dict):
            return self.snapshot()
        with self._shared_lock:
            self._shared = (signature, parsed)
        return dict(parsed)

    def _publish(self, state: dict[str, Any]) -> None:
        """The one write of the state file, which also drops the parsed-state cache.

        Same-size rewrites inside one mtime tick can share a file signature on some filesystems,
        so a writer in this process never relies on the signature alone.
        """
        atomic_write_json(self.state_path, state, compact=True)
        with self._shared_lock:
            self._shared = None

    def _load_for_update(self) -> tuple[Any, dict[str, Any]]:
        """The stored state plus an independent working copy for the mutator.

        Parsing the text twice costs about a third of `copy.deepcopy` on a ~1 MB state, and the
        stored copy is what the merge compares against, so the two must not alias.
        """
        path = self.read_state_path()
        try:
            text = path.read_text()
            current = json.loads(text)
            return current, json.loads(text) if isinstance(current, dict) else {}
        except Exception:
            # Absent or unreadable: `load` owns the backup recovery and the empty default.
            current = self.load({})
            return current, copy.deepcopy(current) if isinstance(current, dict) else {}

    def write(self, state: dict[str, Any]) -> None:
        with self.lock():
            self.backup()
            current = self.load({})
            self._publish(self.merge_for_write(current, state))

    def update(
        self,
        mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
        reason: str | None = None,
    ) -> dict[str, Any]:
        with self.lock():
            self.backup()
            current, state = self._load_for_update()
            updated = mutator(state)
            next_state = updated if isinstance(updated, dict) else state
            merged = self.merge_for_write(current, next_state)
            self._publish(merged)
            self.last_update_reason = reason
            return merged

    def reconcile_model_presence(
        self,
        *,
        observed_sources: set[str],
        listed_keys: set[str],
        now: float | None = None,
        retention_seconds: float = MODEL_ABSENCE_RETENTION_SECONDS,
    ) -> list[str]:
        """Delete the runtime entries of models gone from their provider for ``retention_seconds``.

        ``observed_sources`` are the providers whose listing this catalog refresh can trust to be
        complete (answered, non-empty, not truncated); ``listed_keys`` the ``source::upstream_id``
        keys they still list. A model of any other provider — failed, disabled, uncredentialed,
        removed from config — is unobserved: its absence clock neither starts nor clears and it is
        never pruned, so a refresh that failed or came back empty or partial deletes nothing. A
        relisting clears the clock, so a temporary catalog gap never adds up to the window.

        Applied without the history merge, which exists to stop stale writers from dropping rows
        and would otherwise put the deleted ones straight back. Returns the pruned keys.
        """
        if not observed_sources:
            return []
        reference = time.time() if now is None else now
        stamp = datetime.fromtimestamp(reference, timezone.utc).isoformat()
        with self.lock():
            current, state = self._load_for_update()
            if not isinstance(current, dict):
                return []
            raw_presence = state.get(MODEL_PRESENCE_KEY)
            presence = raw_presence if isinstance(raw_presence, dict) else {}
            first_absent = presence.get("first_absent") if isinstance(presence.get("first_absent"), dict) else {}
            tombstones = presence.get("pruned") if isinstance(presence.get("pruned"), dict) else {}
            known: set[str] = set(first_absent)
            for section in PRUNABLE_MODEL_SECTIONS:
                rows = state.get(section)
                if isinstance(rows, dict):
                    known.update(str(key) for key in rows)
            pruned: list[str] = []
            next_absent: dict[str, Any] = {}
            for key in sorted(known):
                if key.split("::", 1)[0] not in observed_sources:
                    if key in first_absent:
                        next_absent[key] = first_absent[key]
                    continue
                if key in listed_keys:
                    continue
                since = parse_iso_timestamp(first_absent.get(key))
                if since is None:
                    next_absent[key] = stamp
                elif reference - since >= retention_seconds:
                    pruned.append(key)
                else:
                    next_absent[key] = first_absent[key]
            next_tombstones = {
                key: value
                for key, value in tombstones.items()
                if key not in listed_keys
                and (parse_iso_timestamp(value) or 0.0) > reference - MODEL_PRUNE_TOMBSTONE_SECONDS
            }
            for key in pruned:
                next_tombstones[key] = stamp
                for section in PRUNABLE_MODEL_SECTIONS:
                    rows = state.get(section)
                    if isinstance(rows, dict):
                        rows.pop(key, None)
            if next_absent or next_tombstones:
                state[MODEL_PRESENCE_KEY] = {"first_absent": next_absent, "pruned": next_tombstones}
            else:
                state.pop(MODEL_PRESENCE_KEY, None)
            if not pruned and state.get(MODEL_PRESENCE_KEY) == current.get(MODEL_PRESENCE_KEY):
                return []
            self.backup()
            self._publish(state)
            self.last_update_reason = "reconcile_model_presence"
            return pruned

    def clear_listed_model_presence(self, listed_keys: set[str]) -> bool:
        """Drop the absence clock and tombstone of every model a catalog about to be published lists.

        Runs on every catalog publish, observed or not: a listed model is present whatever the
        pruning trust in its provider's listing, and a tombstone left on it would make the
        merge silently drop each stats, cooldown, benchmark or verified row written for it
        until a fully observed refresh happened to clear it. Like the prune, it is applied
        without the history merge, which would put the cleared ledger straight back. Returns
        whether anything was cleared; a publish with nothing to clear writes nothing.
        """

        def listed_marks(state: Any) -> bool:
            presence = state.get(MODEL_PRESENCE_KEY) if isinstance(state, dict) else None
            if not isinstance(presence, dict):
                return False
            return any(
                isinstance(presence.get(part), dict) and not listed_keys.isdisjoint(presence[part])
                for part in ("first_absent", "pruned")
            )

        # The cached snapshot answers the common no-op publish without taking the lock.
        if not listed_keys or not listed_marks(self.snapshot()):
            return False
        with self.lock():
            current, state = self._load_for_update()
            if not listed_marks(current):
                return False
            presence = state[MODEL_PRESENCE_KEY]
            next_presence: dict[str, dict[str, Any]] = {}
            for part in ("first_absent", "pruned"):
                rows = presence.get(part)
                next_presence[part] = (
                    {key: value for key, value in rows.items() if key not in listed_keys} if isinstance(rows, dict) else {}
                )
            if any(next_presence.values()):
                state[MODEL_PRESENCE_KEY] = next_presence
            else:
                state.pop(MODEL_PRESENCE_KEY, None)
            self.backup()
            self._publish(state)
            self.last_update_reason = "clear_listed_model_presence"
            return True

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


SPEND_METER_DOCUMENT_KEY = "meters"


@dataclass
class SpendMeterStore:
    """Provider spend meters, kept out of ``state.json`` so the per-request write stays tiny.

    Every upstream call is counted before it leaves (``router.record_upstream_spend``): a request
    that never answers, or a router that crashes mid-call, has still spent its allowance, and
    other processes see it through the shared file. That write used to be a full read/merge/
    rewrite of the whole runtime state for a counter of a few hundred bytes, so its cost grew with
    the state. This file holds only the meters, behind its own cross-process lock
    (``probe_lock.file_lock``, so Windows is excluded across processes too), written with the
    same private permissions and atomic rename as the state. An unreadable file reads as empty:
    the meters are per-window counts, so the worst case is one window of undercounting, never a
    failed request.
    """

    path: Path
    lock_path: Path
    thread_lock: threading.RLock = field(default_factory=threading.RLock)

    def load(self) -> dict[str, Any]:
        document = load_json(self.path, {})
        meters = document.get(SPEND_METER_DOCUMENT_KEY) if isinstance(document, dict) else None
        return meters if isinstance(meters, dict) else {}

    def update(self, mutator: Callable[[dict[str, Any]], None]) -> None:
        ensure_private_dir(self.lock_path.parent)
        with self.thread_lock, file_lock(self.lock_path, blocking=True):
            meters = self.load()
            mutator(meters)
            atomic_write_json(self.path, {SPEND_METER_DOCUMENT_KEY: meters}, compact=True)
