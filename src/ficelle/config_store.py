from __future__ import annotations

import copy
import json
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ficelle.json_store import atomic_write_json, load_json
from ficelle.state_store import advisory_file_lock


class ConfigFileUnreadable(RuntimeError):
    """The stored config exists but cannot be parsed, so it must not be overwritten.

    `config.json` is documented as user-editable, and `load_json` answers a syntax error with the
    defaults. Writing those defaults back — which any admin save, or the prune that runs during a
    refresh, would do — replaces the user's providers, profiles and port with stock values and
    loses them for good, since config has no backups the way state does.
    """


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# One lock for every ConfigStore in the process — see `ConfigStore.thread_lock`.
_CONFIG_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class ConfigStore:
    config_path: Path
    defaults: dict[str, Any]
    normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    fallback_config_path: Path | None = None
    # Unlike StateStore this had no lock at all, while admin saves run in concurrent handler
    # threads and `ficelle set-key --refresh` writes from a second process against the live
    # daemon. Two overlapping saves each built a config from the pre-mutation file, so one
    # writer's fields were silently dropped.
    #
    # Module-level, not per-instance: `runtime_config_store()` builds a fresh (frozen) store on
    # every call, so a `default_factory` lock is a different object for each caller and two
    # threads never meet on it. Where `fcntl` exists the `flock` half still serializes them and
    # the thread lock is belt to that brace — but `advisory_file_lock` degrades to the thread
    # lock alone where it does not (Windows), and there a per-call lock guards nothing at all.
    thread_lock: threading.RLock = field(default=_CONFIG_WRITE_LOCK, repr=False, compare=False)

    def read_path(self) -> Path:
        if self.config_path.exists() or self.fallback_config_path is None:
            return self.config_path
        return (
            self.fallback_config_path
            if self.fallback_config_path.exists()
            else self.config_path
        )

    def lock_path(self) -> Path:
        return self.config_path.with_name(self.config_path.name + ".lock")

    def backup_path(self) -> Path:
        return self.config_path.with_name(self.config_path.name + ".bak")

    def broken_path(self) -> Path:
        return self.config_path.with_name(self.config_path.name + ".broken")

    def stored_config_error(self) -> str | None:
        """Why the stored config cannot be trusted, or None when it parses (or is absent)."""
        path = self.read_path()
        try:
            if not path.exists():
                return None
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"unreadable: {exc}"
        if not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            return f"invalid JSON: {exc}"
        if not isinstance(parsed, dict):
            return f"expected a JSON object, found {type(parsed).__name__}"
        return None

    def merged(self, existing: Any | None = None) -> dict[str, Any]:
        config = deep_merge(self.defaults, existing if isinstance(existing, dict) else {})
        config["allow_paid_fallback"] = False
        if self.normalize is not None:
            config = self.normalize(config)
        return config

    def load(self) -> dict[str, Any]:
        read_path = self.read_path()
        error = self.stored_config_error()
        if error is not None:
            # Still serving on purpose: a broken config must not take the router down. What it
            # must no longer do is look like an *absent* config, because the next write would
            # then make the defaults permanent. `save` refuses while this holds.
            self._preserve_broken_copy()
            print(
                f"ficelle: {read_path} could not be read ({error}); running on defaults and "
                f"refusing to overwrite it. A copy is at {self.broken_path()}.",
                flush=True,
            )
            return self.merged({})
        existing = load_json(read_path, {})
        config = self.merged(existing)
        if not read_path.exists():
            atomic_write_json(self.config_path, config)
        return config

    def _preserve_broken_copy(self) -> None:
        try:
            source = self.read_path()
            if source.exists() and not self.broken_path().exists():
                shutil.copy2(source, self.broken_path())
        except OSError:
            pass

    def _backup_current(self) -> None:
        try:
            if self.config_path.exists():
                shutil.copy2(self.config_path, self.backup_path())
        except OSError:
            pass

    def save(self, config: dict[str, Any]) -> dict[str, Any]:
        with advisory_file_lock(self.lock_path(), self.thread_lock):
            return self._save_locked(config)

    def _save_locked(self, config: dict[str, Any]) -> dict[str, Any]:
        error = self.stored_config_error()
        if error is not None:
            self._preserve_broken_copy()
            raise ConfigFileUnreadable(
                f"refusing to overwrite {self.read_path()} ({error}); "
                f"a copy is at {self.broken_path()}"
            )
        normalized = self.merged(config)
        self._backup_current()
        atomic_write_json(self.config_path, normalized)
        return normalized

    def update(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        # Read, mutate and write inside one lock, or two concurrent savers each write a config
        # built from the pre-mutation file and the later one silently wins.
        with advisory_file_lock(self.lock_path(), self.thread_lock):
            existing = load_json(self.read_path(), {})
            config = self.merged(existing)
            mutator(config)
            return self._save_locked(config)
