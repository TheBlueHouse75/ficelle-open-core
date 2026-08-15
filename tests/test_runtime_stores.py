from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ficelle.config_store import ConfigStore
from ficelle.json_store import atomic_write_json, load_json
from ficelle.runtime_paths import RuntimePaths
from ficelle.state_store import StateStore


def file_snapshot(path: Path) -> tuple[bytes, int, int]:
    stat = path.stat()
    return path.read_bytes(), stat.st_mode, stat.st_mtime_ns


def test_runtime_paths_resolve_from_explicit_environment(tmp_path):
    hermes_home = tmp_path / "hermes"
    ficelle_home = tmp_path / "ficelle-home"
    runtime_dir = tmp_path / "runtime"
    package_dir = tmp_path / "package"

    paths = RuntimePaths.from_env(
        environ={
            "HERMES_HOME": str(hermes_home),
            "FICELLE_HOME": str(ficelle_home),
            "FICELLE_RUNTIME_DIR": str(runtime_dir),
        },
        package_dir=package_dir,
    )

    assert paths.hermes_home == hermes_home
    router_dir = ficelle_home
    logs_dir = router_dir / "logs"
    assert paths.ficelle_home == ficelle_home
    assert paths.router_dir == router_dir
    assert paths.runtime_read_dir == runtime_dir
    assert paths.config_path == router_dir / "config.json"
    assert paths.state_path == router_dir / "state.json"
    assert paths.catalog_path == router_dir / "catalog.json"
    assert paths.route_log_path == logs_dir / "routes.jsonl"
    assert paths.request_log_store_path == router_dir / "requests.sqlite"
    assert paths.admin_audit_log_path == logs_dir / "admin-actions.jsonl"
    assert paths.state_lock_path == router_dir / "state.lock"
    assert paths.state_backup_dir == router_dir / "state-backups"
    assert paths.capability_discrepancy_log_path == logs_dir / "capability-discrepancies.jsonl"
    assert paths.compression_store_path == router_dir / "compression.sqlite"
    assert paths.capability_oracle_cache_path == router_dir / "capability_oracle.json"
    assert paths.hermes_agent_dir == hermes_home / "hermes-agent"
    assert paths.hermes_config_path == hermes_home / "config.yaml"
    assert paths.credential_env_file == ficelle_home / ".env"
    assert paths.legacy_credential_env_file == hermes_home / ".env"
    assert paths.ficelle_secrets_keychain == ficelle_home / "ficelle-secrets.keychain-db"
    assert paths.legacy_ficelle_secrets_keychain == ficelle_home / "hermes-secrets.keychain-db"
    assert paths.legacy_hermes_secrets_keychain == hermes_home / "hermes-secrets.keychain-db"
    assert paths.admin_assets_dir == package_dir / "assets" / "admin"


def test_runtime_paths_default_to_standalone_ficelle_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    paths = RuntimePaths.from_env(environ={})

    assert paths.ficelle_home == tmp_path / ".ficelle"
    assert paths.router_dir == tmp_path / ".ficelle"
    assert paths.runtime_read_dir == tmp_path / ".ficelle"
    assert paths.hermes_home == tmp_path / ".hermes"
    assert paths.legacy_ficelle_secrets_keychain == (
        tmp_path / ".ficelle" / "hermes-secrets.keychain-db"
    )


def test_runtime_paths_discover_legacy_layout_without_mutating_it(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    legacy_router_dir = tmp_path / ".hermes" / "ficelle"
    legacy_router_dir.mkdir(parents=True)
    legacy_state = legacy_router_dir / "state.json"
    legacy_state.write_text('{"preserved": true}', encoding="utf-8")
    legacy_snapshot = file_snapshot(legacy_state)

    paths = RuntimePaths.from_env(environ={})

    assert paths.ficelle_home == tmp_path / ".ficelle"
    assert paths.router_dir == tmp_path / ".ficelle"
    assert paths.runtime_read_dir == legacy_router_dir
    assert paths.credential_env_file == tmp_path / ".ficelle" / ".env"
    assert paths.legacy_credential_env_file == tmp_path / ".hermes" / ".env"
    assert paths.request_log_store_path == tmp_path / ".ficelle" / "requests.sqlite"
    assert paths.read_path(paths.state_path) == legacy_state
    assert file_snapshot(legacy_state) == legacy_snapshot
    assert not (tmp_path / ".ficelle").exists()

    paths.state_path.parent.mkdir(parents=True)
    paths.state_path.write_text('{"canonical": true}', encoding="utf-8")

    assert paths.read_path(paths.state_path) == paths.state_path
    assert file_snapshot(legacy_state) == legacy_snapshot


def test_runtime_paths_prefer_existing_standalone_home_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    (tmp_path / ".ficelle").mkdir()
    (tmp_path / ".ficelle" / "state.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".hermes" / "ficelle").mkdir(parents=True)

    paths = RuntimePaths.from_env(environ={})

    assert paths.ficelle_home == tmp_path / ".ficelle"
    assert paths.router_dir == tmp_path / ".ficelle"
    assert paths.runtime_read_dir == tmp_path / ".hermes" / "ficelle"
    assert paths.read_path(paths.state_path) == paths.state_path


def test_runtime_paths_explicit_ficelle_home_wins_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    ficelle_home = tmp_path / "custom-credentials"
    ficelle_home.mkdir()
    (ficelle_home / "hermes-secrets.keychain-db").write_text(
        "legacy-keychain",
        encoding="utf-8",
    )
    legacy_router_dir = tmp_path / ".hermes" / "ficelle"
    legacy_router_dir.mkdir(parents=True)
    (legacy_router_dir / "state.json").write_text('{"legacy": true}', encoding="utf-8")

    paths = RuntimePaths.from_env(environ={"FICELLE_HOME": str(ficelle_home)})

    assert paths.ficelle_home == ficelle_home
    assert paths.router_dir == ficelle_home
    assert paths.runtime_read_dir == ficelle_home
    assert paths.legacy_ficelle_secrets_keychain == (
        ficelle_home / "hermes-secrets.keychain-db"
    )


def test_runtime_paths_default_legacy_keychain_is_credential_only(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    ficelle_home = tmp_path / ".ficelle"
    ficelle_home.mkdir()
    (ficelle_home / "hermes-secrets.keychain-db").write_text(
        "legacy-keychain",
        encoding="utf-8",
    )
    legacy_router_dir = tmp_path / ".hermes" / "ficelle"
    legacy_router_dir.mkdir(parents=True)
    (legacy_router_dir / "state.json").write_text('{"legacy": true}', encoding="utf-8")

    paths = RuntimePaths.from_env(environ={})

    assert paths.ficelle_home == ficelle_home
    assert paths.router_dir == ficelle_home
    assert paths.runtime_read_dir == legacy_router_dir
    assert paths.read_path(paths.state_path) == legacy_router_dir / "state.json"
    assert paths.legacy_ficelle_secrets_keychain == (
        ficelle_home / "hermes-secrets.keychain-db"
    )


def test_runtime_paths_explicit_runtime_dir_wins_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    legacy_router_dir = tmp_path / ".hermes" / "ficelle"
    legacy_router_dir.mkdir(parents=True)
    runtime_dir = tmp_path / "selected-runtime"

    paths = RuntimePaths.from_env(
        environ={"FICELLE_RUNTIME_DIR": str(runtime_dir)},
    )

    assert paths.ficelle_home == tmp_path / ".ficelle"
    assert paths.router_dir == tmp_path / ".ficelle"
    assert paths.runtime_read_dir == runtime_dir


def test_credential_write_does_not_switch_unmigrated_legacy_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    legacy_router_dir = tmp_path / ".hermes" / "ficelle"
    legacy_router_dir.mkdir(parents=True)
    (legacy_router_dir / "state.json").write_text('{"legacy": true}', encoding="utf-8")

    before = RuntimePaths.from_env(environ={})
    before.credential_env_file.parent.mkdir(parents=True)
    before.credential_env_file.write_text("OPENROUTER_API_KEY=redacted\n", encoding="utf-8")
    before.ficelle_secrets_keychain.touch()
    after = RuntimePaths.from_env(environ={})

    assert before.ficelle_home == after.ficelle_home == tmp_path / ".ficelle"
    assert before.router_dir == after.router_dir == tmp_path / ".ficelle"
    assert before.runtime_read_dir == after.runtime_read_dir == legacy_router_dir
    assert after.read_path(after.state_path) == legacy_router_dir / "state.json"
    assert after.credential_env_file == tmp_path / ".ficelle" / ".env"
    assert after.ficelle_secrets_keychain == tmp_path / ".ficelle" / "ficelle-secrets.keychain-db"


def test_runtime_read_path_returns_unknown_paths_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    legacy_router_dir = tmp_path / ".hermes" / "ficelle"
    legacy_router_dir.mkdir(parents=True)
    outside_path = tmp_path / "outside.json"
    outside_path.write_text("{}", encoding="utf-8")

    paths = RuntimePaths.from_env(environ={})

    assert paths.read_path(outside_path) == outside_path
    assert paths.read_path(paths.router_dir / ".." / "outside.json") == (
        paths.router_dir / ".." / "outside.json"
    )


def test_config_stores_share_one_thread_lock(tmp_path):
    """Two stores for the same config must meet on the same in-process lock.

    `runtime_config_store()` builds a fresh frozen store per call, so a per-instance
    `default_factory` lock is a different object for every caller and serializes nothing
    between them. `flock` covers that on POSIX — but `advisory_file_lock` degrades to the
    thread lock alone where `fcntl` is absent (Windows), and there the guard would be gone
    entirely. Cheap to assert, and invisible on this platform otherwise.
    """
    config_path = tmp_path / "ficelle" / "config.json"
    first = ConfigStore(config_path, {"allow_paid_fallback": False})
    second = ConfigStore(config_path, {"allow_paid_fallback": False})
    assert first.thread_lock is second.thread_lock

    # And it really excludes: held by one store, a *second thread* going through the other
    # one waits. Checked from another thread because an RLock re-entered by its own holder
    # would say yes to anything.
    acquired: list[bool] = []
    with first.thread_lock:
        probe = threading.Thread(
            target=lambda: acquired.append(second.thread_lock.acquire(timeout=0.2))
        )
        probe.start()
        probe.join(timeout=5)
    assert acquired == [False]


def test_config_store_load_save_and_update_preserve_strict_zero(tmp_path):
    config_path = tmp_path / "ficelle" / "config.json"
    defaults = {
        "allow_paid_fallback": False,
        "nested": {"enabled": True, "limit": 4},
        "fusion": {"enabled": False},
    }
    store = ConfigStore(
        config_path,
        defaults,
        normalize=lambda config: {**config, "normalized": True},
    )

    loaded = store.load()
    assert loaded["allow_paid_fallback"] is False
    assert loaded["nested"] == {"enabled": True, "limit": 4}
    assert loaded["normalized"] is True
    assert config_path.exists()

    config_path.write_text('{"allow_paid_fallback": true, "nested": {"limit": 2}}', encoding="utf-8")
    loaded = store.load()
    assert loaded["allow_paid_fallback"] is False
    assert loaded["nested"] == {"enabled": True, "limit": 2}

    saved = store.update(lambda config: config["nested"].update({"limit": 9}))
    assert saved["allow_paid_fallback"] is False
    assert saved["nested"]["limit"] == 9
    assert load_json(config_path, {})["allow_paid_fallback"] is False


def test_config_store_reads_legacy_without_materializing_canonical(tmp_path):
    canonical_path = tmp_path / ".ficelle" / "config.json"
    legacy_path = tmp_path / ".hermes" / "ficelle" / "config.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text('{"nested": {"limit": 7}}', encoding="utf-8")
    legacy_path.chmod(0o640)
    legacy_snapshot = file_snapshot(legacy_path)
    store = ConfigStore(
        canonical_path,
        {
            "allow_paid_fallback": False,
            "nested": {"enabled": True, "limit": 4},
        },
        fallback_config_path=legacy_path,
    )

    loaded = store.load()

    assert loaded["nested"] == {"enabled": True, "limit": 7}
    assert not canonical_path.exists()
    assert file_snapshot(legacy_path) == legacy_snapshot


def test_config_store_first_update_is_canonical_copy_on_write(tmp_path):
    canonical_path = tmp_path / ".ficelle" / "config.json"
    legacy_path = tmp_path / ".hermes" / "ficelle" / "config.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        '{"allow_paid_fallback": true, "nested": {"enabled": true, "limit": 7}}',
        encoding="utf-8",
    )
    legacy_path.chmod(0o640)
    legacy_snapshot = file_snapshot(legacy_path)
    store = ConfigStore(
        canonical_path,
        {
            "allow_paid_fallback": False,
            "nested": {"enabled": False, "limit": 4},
        },
        fallback_config_path=legacy_path,
    )

    saved = store.update(lambda config: config["nested"].update({"limit": 9}))

    assert saved["nested"] == {"enabled": True, "limit": 9}
    assert load_json(canonical_path, {}) == saved
    assert file_snapshot(legacy_path) == legacy_snapshot


def test_config_store_existing_malformed_canonical_wins_over_legacy(tmp_path):
    canonical_path = tmp_path / ".ficelle" / "config.json"
    legacy_path = tmp_path / ".hermes" / "ficelle" / "config.json"
    canonical_path.parent.mkdir(parents=True)
    legacy_path.parent.mkdir(parents=True)
    canonical_path.write_text('{"nested":', encoding="utf-8")
    legacy_path.write_text('{"nested": {"limit": 7}}', encoding="utf-8")
    store = ConfigStore(
        canonical_path,
        {
            "allow_paid_fallback": False,
            "nested": {"enabled": True, "limit": 4},
        },
        fallback_config_path=legacy_path,
    )

    loaded = store.load()

    assert loaded["nested"] == {"enabled": True, "limit": 4}
    assert canonical_path.read_text(encoding="utf-8") == '{"nested":'


def test_state_store_update_uses_isolated_paths_and_preserves_evidence(tmp_path):
    state_path = tmp_path / "state.json"
    store = StateStore(
        state_path,
        tmp_path / "state.lock",
        tmp_path / "state-backups",
        backup_min_interval_seconds=0,
    )
    atomic_write_json(
        state_path,
        {
            "benchmark_results": {
                "openrouter::a": {
                    "ficelle/auto-fast": {
                        "status": "pass",
                        "ran_at": "2026-06-19T01:00:00+00:00",
                    }
                }
            },
            "verified_capabilities": {
                "openrouter::a": {
                    "ficelle/auto-fast": {
                        "status": "verified",
                        "verified_at": "2026-06-19T01:00:00+00:00",
                    }
                }
            },
        },
    )

    def mutate(state):
        state["benchmark_results"] = {
            "openrouter::b": {
                "ficelle/auto-fast": {
                    "status": "pass",
                    "ran_at": "2026-06-19T02:00:00+00:00",
                }
            }
        }
        state["verified_capabilities"] = {
            "openrouter::b": {
                "ficelle/auto-fast": {
                    "status": "verified",
                    "verified_at": "2026-06-19T02:00:00+00:00",
                }
            }
        }
        state["last_routes"] = {"ficelle/auto-fast": {"model": "b"}}

    returned = store.update(mutate)

    state = load_json(state_path, {})
    assert returned == state
    assert set(state["benchmark_results"]) == {"openrouter::a", "openrouter::b"}
    assert set(state["verified_capabilities"]) == {"openrouter::a", "openrouter::b"}
    assert state["last_routes"]["ficelle/auto-fast"]["model"] == "b"
    assert len(store.backup_files()) == 1
    diagnostics = store.diagnostics()
    assert diagnostics["state"]["path"] == str(state_path)
    assert diagnostics["benchmark_results"]["evidence_row_count"] == 2
    assert diagnostics["verified_capabilities"]["evidence_row_count"] == 2


def test_state_store_update_returns_persisted_state_when_mutator_returns_partial(tmp_path):
    state_path = tmp_path / "state.json"
    store = StateStore(
        state_path,
        tmp_path / "state.lock",
        tmp_path / "state-backups",
        backup_min_interval_seconds=0,
    )
    atomic_write_json(
        state_path,
        {
            "benchmark_results": {
                "openrouter::a": {
                    "ficelle/auto-fast": {
                        "status": "pass",
                        "ran_at": "2026-06-19T01:00:00+00:00",
                    }
                }
            },
        },
    )

    returned = store.update(lambda _state: {"last_routes": {"ficelle/auto-fast": {"model": "b"}}})
    written = load_json(state_path, {})

    assert returned == written
    assert set(returned) == {"benchmark_results", "last_routes"}
    assert returned["benchmark_results"]["openrouter::a"]["ficelle/auto-fast"]["status"] == "pass"
    assert returned["last_routes"]["ficelle/auto-fast"]["model"] == "b"


def test_state_store_update_accepts_non_persisted_reason(tmp_path):
    state_path = tmp_path / "state.json"
    store = StateStore(
        state_path,
        tmp_path / "state.lock",
        tmp_path / "state-backups",
        backup_min_interval_seconds=0,
    )

    returned = store.update(
        lambda state: state.update({"last_routes": {"ficelle/auto-fast": {"model": "b"}}}),
        reason="route_attempt",
    )
    written = load_json(state_path, {})

    assert returned == written
    assert store.last_update_reason == "route_attempt"
    assert "reason" not in written
    assert "last_update_reason" not in written


def test_state_store_reads_legacy_then_writes_and_locks_canonical_only(tmp_path):
    canonical_dir = tmp_path / ".ficelle"
    legacy_dir = tmp_path / ".hermes" / "ficelle"
    canonical_state = canonical_dir / "state.json"
    legacy_state = legacy_dir / "state.json"
    legacy_state.parent.mkdir(parents=True)
    legacy_state.write_text(
        '{"benchmark_results": {"legacy": {}}, "last_routes": {"old": true}}',
        encoding="utf-8",
    )
    legacy_state.chmod(0o640)
    legacy_snapshot = file_snapshot(legacy_state)
    store = StateStore(
        canonical_state,
        canonical_dir / "state.lock",
        canonical_dir / "state-backups",
        backup_min_interval_seconds=0,
        fallback_state_path=legacy_state,
        fallback_backup_dir=legacy_dir / "state-backups",
    )

    assert store.snapshot()["last_routes"] == {"old": True}
    assert not canonical_dir.exists()

    updated = store.update(
        lambda state: state.update({"last_routes": {"new": True}}),
        reason="legacy-copy-on-write",
    )

    assert updated["last_routes"] == {"new": True}
    assert "legacy" in updated["benchmark_results"]
    assert load_json(canonical_state, {}) == updated
    assert (canonical_dir / "state.lock").exists()
    assert not (legacy_dir / "state.lock").exists()
    assert not (canonical_dir / "state-backups").exists()
    assert file_snapshot(legacy_state) == legacy_snapshot


def test_state_store_existing_canonical_wins_even_when_malformed(tmp_path):
    canonical_state = tmp_path / ".ficelle" / "state.json"
    legacy_state = tmp_path / ".hermes" / "ficelle" / "state.json"
    canonical_state.parent.mkdir(parents=True)
    legacy_state.parent.mkdir(parents=True)
    canonical_state.write_text('{"broken":', encoding="utf-8")
    legacy_state.write_text('{"legacy": true}', encoding="utf-8")
    store = StateStore(
        canonical_state,
        canonical_state.with_suffix(".lock"),
        canonical_state.parent / "state-backups",
        fallback_state_path=legacy_state,
    )

    assert store.load({"default": True}) == {"default": True}
    assert store.read_state_path() == canonical_state


def test_state_store_reads_fallback_backups_but_never_rotates_them(tmp_path):
    canonical_dir = tmp_path / ".ficelle"
    legacy_dir = tmp_path / ".hermes" / "ficelle"
    legacy_state = legacy_dir / "state.json"
    legacy_backups = legacy_dir / "state-backups"
    legacy_backups.mkdir(parents=True)
    legacy_state.write_text('{"generation": 1}', encoding="utf-8")
    first_legacy_backup = legacy_backups / "state-20260101T000000.000000Z-a.json"
    second_legacy_backup = legacy_backups / "state-20260102T000000.000000Z-b.json"
    first_legacy_backup.write_text('{"generation": 0}', encoding="utf-8")
    second_legacy_backup.write_text('{"generation": 1}', encoding="utf-8")
    legacy_snapshots = {
        path: file_snapshot(path)
        for path in (legacy_state, first_legacy_backup, second_legacy_backup)
    }
    store = StateStore(
        canonical_dir / "state.json",
        canonical_dir / "state.lock",
        canonical_dir / "state-backups",
        backup_keep=1,
        backup_min_interval_seconds=0,
        fallback_state_path=legacy_state,
        fallback_backup_dir=legacy_backups,
    )

    assert store.backup_files() == [first_legacy_backup, second_legacy_backup]
    assert store.diagnostics()["backups"]["dir"] == str(legacy_backups)

    store.rotate_backups(keep=0)

    assert {
        path: file_snapshot(path)
        for path in (legacy_state, first_legacy_backup, second_legacy_backup)
    } == legacy_snapshots

    store.update(lambda state: state.update({"generation": 2}))
    store.update(lambda state: state.update({"generation": 3}))

    canonical_backups = store.canonical_backup_files()
    assert len(canonical_backups) == 1
    assert store.backup_files() == canonical_backups
    assert {
        path: file_snapshot(path)
        for path in (legacy_state, first_legacy_backup, second_legacy_backup)
    } == legacy_snapshots


def test_state_store_diagnostics_prefers_canonical_dir_when_fallback_is_empty(
    tmp_path,
):
    canonical_dir = tmp_path / ".ficelle"
    legacy_backups = tmp_path / ".hermes" / "ficelle" / "state-backups"
    store = StateStore(
        canonical_dir / "state.json",
        canonical_dir / "state.lock",
        canonical_dir / "state-backups",
        fallback_backup_dir=legacy_backups,
    )

    diagnostics = store.diagnostics({})

    assert diagnostics["backups"]["dir"] == str(canonical_dir / "state-backups")
    assert diagnostics["backups"]["count"] == 0


def _merge_history_rows_reference(current, incoming):
    """The pre-optimization merge, kept verbatim as the oracle for the rewrite below.

    `merge_runtime_history_rows` used to deep-copy every current row up front and then overwrite
    most of them, which cost 17.4 ms of a 35.9 ms state write. Deciding first and copying once is
    a behaviour-preserving change *only* if it agrees with this on every shape, so the agreement
    is asserted rather than argued.
    """
    import copy

    from ficelle.state_store import runtime_row_timestamp

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
                merged[key] = _merge_history_rows_reference(current_value, incoming_value)
            elif isinstance(current_value, dict) and not isinstance(incoming_value, dict):
                continue
            else:
                merged[key] = copy.deepcopy(incoming_value)
        return merged
    return copy.deepcopy(incoming)


def test_history_merge_matches_the_pre_optimization_reference_on_generated_shapes():
    import random

    from ficelle.state_store import merge_runtime_history_rows

    rnd = random.Random(1234)
    stamps = [None, "2026-01-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00", "not-a-date"]

    def value(depth=0):
        kind = rnd.random()
        if depth < 2 and kind < 0.35:
            return {f"p{index}": value(depth + 1) for index in range(rnd.randint(0, 3))}
        if kind < 0.75:
            row = {"status": rnd.choice(["verified", "failed"]), "n": rnd.randint(0, 9)}
            stamp = rnd.choice(stamps)
            if stamp:
                row[rnd.choice(["verified_at", "failed_at", "ran_at"])] = stamp
            return row
        return rnd.choice([None, 5, "text", [1, 2]])

    for _ in range(2000):
        keys = [f"m{index}" for index in range(rnd.randint(0, 6))]
        current = {key: value() for key in keys if rnd.random() < 0.8}
        incoming = {key: value() for key in keys if rnd.random() < 0.8}
        if rnd.random() < 0.1:
            current = rnd.choice([None, 5, "x"])
        if rnd.random() < 0.05:
            incoming = rnd.choice([None, 5, "x"])
        assert merge_runtime_history_rows(current, incoming) == _merge_history_rows_reference(current, incoming), (
            current,
            incoming,
        )


def test_history_merge_explicitly_covers_evidence_ties_and_nested_edge_shapes():
    """Keep the timestamp branches deterministic: random generation must not be their only proof."""
    from ficelle.state_store import merge_runtime_history_rows

    equal_timestamp = "2026-06-01T00:00:00+00:00"
    newer_timestamp = "2026-08-01T00:00:00+00:00"
    older_timestamp = "2026-01-01T00:00:00+00:00"
    for timestamp_field in ("verified_at", "failed_at", "ran_at"):
        current = {"model": {"profile": {"status": "current", timestamp_field: equal_timestamp}}}
        incoming = {"model": {"profile": {"status": "incoming", timestamp_field: equal_timestamp}}}

        merged = merge_runtime_history_rows(current, incoming)

        # The old `>=` rule makes an equally recent incoming row win for every accepted field.
        assert merged == _merge_history_rows_reference(current, incoming)
        assert merged["model"]["profile"]["status"] == "incoming"

    current = {
        "only_current": {"level_one": {"level_two": {"row": "current"}}},
        "newer_current": {"profile": {"status": "current", "verified_at": newer_timestamp}},
        "current_dict": {"profile": {"status": "current"}},
        "current_scalar": "current",
        "unparseable": {"profile": {"status": "current", "ran_at": "not-a-date"}},
    }
    incoming = {
        "only_incoming": {"level_one": {"level_two": {"row": "incoming"}}},
        "newer_current": {"profile": {"status": "incoming", "failed_at": older_timestamp}},
        "current_dict": "not-a-row",
        "current_scalar": {"profile": {"status": "incoming"}},
        "unparseable": {"profile": {"status": "incoming", "ran_at": "not-a-date"}},
    }

    merged = merge_runtime_history_rows(current, incoming)
    assert merged == _merge_history_rows_reference(current, incoming)
    assert merged["newer_current"]["profile"]["status"] == "current"
    assert merge_runtime_history_rows(None, incoming) == _merge_history_rows_reference(None, incoming)
    assert merge_runtime_history_rows(current, None) == _merge_history_rows_reference(current, None)


def test_merge_for_write_does_not_mutate_incoming_and_aliases_non_history_values():
    # The whole-state deep copy was replaced by a shallow one, which is only safe because the
    # merge assigns top-level history keys and never reaches into the caller's nested objects.
    from ficelle.state_store import merge_runtime_state_for_write

    current = {"benchmark_results": {"m": {"p": {"status": "verified", "verified_at": "2026-08-01T00:00:00+00:00"}}}}
    incoming = {
        "benchmark_results": {"m": {"p": {"status": "failed", "failed_at": "2026-01-01T00:00:00+00:00"}}},
        "stats": {"m": {"requests": 1}},
    }
    incoming_before = copy.deepcopy(incoming)

    merged = merge_runtime_state_for_write(current, incoming)

    assert incoming == incoming_before, "the caller's state must come back untouched"
    # Stored evidence is newer, so it survives a writer holding a stale row.
    assert merged["benchmark_results"]["m"]["p"]["status"] == "verified"
    # The shallow copy deliberately shares non-history values; persistence occurs before callers
    # receive control again, so this alias cannot alter the bytes already written by StateStore.
    merged["stats"]["m"]["requests"] = 99
    assert incoming["stats"]["m"]["requests"] == 99, (
        "non-history values are shared by design; if this ever needs to stop being true, "
        "the shallow copy in merge_runtime_state_for_write is what to revisit"
    )


def test_stale_writer_cannot_erase_newer_evidence(tmp_path):
    """The invariant merge_for_write exists for, asserted end to end through StateStore.

    Losing a row here does not crash anything: it deletes routing knowledge the router then has
    to re-earn by spending real free-tier quota re-probing. That is why the C3 copy optimization
    is fenced by this rather than by its own micro-tests alone.
    """
    store = StateStore(state_path=tmp_path / "state.json", lock_path=tmp_path / "state.lock", backup_dir=tmp_path / "backups")
    atomic_write_json(
        tmp_path / "state.json",
        {
            "benchmark_results": {"m1": {"auto-tools": {"status": "verified", "verified_at": "2026-08-01T00:00:00+00:00"}}},
            "verified_capabilities": {"m1": {"tools": {"status": "verified", "verified_at": "2026-08-01T00:00:00+00:00"}}},
        },
    )

    # A writer whose snapshot predates the evidence entirely must not drop it.
    store.write({"stats": {"m1": {"requests": 1}}})
    assert store.snapshot()["benchmark_results"]["m1"]["auto-tools"]["status"] == "verified"
    assert store.snapshot()["verified_capabilities"]["m1"]["tools"]["status"] == "verified"

    # A writer holding an older row loses to the stored one.
    store.write({"benchmark_results": {"m1": {"auto-tools": {"status": "failed", "failed_at": "2026-01-01T00:00:00+00:00"}}}})
    assert store.snapshot()["benchmark_results"]["m1"]["auto-tools"]["status"] == "verified"

    # But a genuinely newer row still has to win, or evidence would freeze.
    store.write({"benchmark_results": {"m1": {"auto-tools": {"status": "failed", "failed_at": "2026-09-01T00:00:00+00:00"}}}})
    assert store.snapshot()["benchmark_results"]["m1"]["auto-tools"]["status"] == "failed"


def test_concurrent_writers_each_keep_their_evidence(tmp_path):
    import threading

    store = StateStore(state_path=tmp_path / "state.json", lock_path=tmp_path / "state.lock", backup_dir=tmp_path / "backups")
    atomic_write_json(tmp_path / "state.json", {})

    def record(index: int) -> None:
        def mutate(state):
            state.setdefault("benchmark_results", {})[f"m{index}"] = {
                "p": {"status": "verified", "verified_at": f"2026-08-0{index + 1}T00:00:00+00:00"}
            }

        store.update(mutate)

    threads = [threading.Thread(target=record, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(store.snapshot()["benchmark_results"]) == 8


def test_corrupt_config_is_served_as_defaults_but_never_overwritten(tmp_path, capsys):
    # config.json is documented as user-editable, and load_json answers a syntax error with the
    # defaults. Writing those back — any admin save, or the prune during a refresh — replaced the
    # user's providers, profiles and port with stock values, with no backup to recover from.
    from ficelle.config_store import ConfigFileUnreadable, ConfigStore

    store = ConfigStore(config_path=tmp_path / "config.json", defaults={"port": 8646, "providers": {}})
    (tmp_path / "config.json").write_text('{"providers": {"mine": {"enabled": tru', encoding="utf-8")

    # The router keeps serving: a broken config must not take it down.
    assert store.load()["port"] == 8646
    assert "could not be read" in capsys.readouterr().out

    with pytest.raises(ConfigFileUnreadable):
        store.save({"port": 9999})

    assert "tru" in (tmp_path / "config.json").read_text(encoding="utf-8"), "the original must survive"
    assert (tmp_path / "config.json.broken").exists(), "a copy is kept for the operator"


def test_concurrent_config_updates_do_not_drop_fields(tmp_path):
    import threading

    from ficelle.config_store import ConfigStore

    store = ConfigStore(config_path=tmp_path / "config.json", defaults={"port": 8646, "providers": {}})
    store.save({"port": 8646})

    def add(index: int) -> None:
        store.update(lambda config: config.setdefault("providers", {}).update({f"p{index}": {"enabled": True}}))

    threads = [threading.Thread(target=add, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(store.load()["providers"]) == 12


def test_unreadable_state_recovers_from_the_newest_parseable_backup(tmp_path, capsys):
    # Resetting to {} did not only lose that read: the next write backed up the *reset* state, so
    # with 20 kept backups the pre-corruption ones rotated away within hours of active use.
    store = StateStore(
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "state.lock",
        backup_dir=tmp_path / "backups",
        backup_min_interval_seconds=0,
    )
    store.update(lambda state: state.setdefault("cooldowns", {}).update({"m1": {"until": 999}}))
    store.update(lambda state: state.setdefault("cooldowns", {}).update({"m2": {"until": 999}}))

    (tmp_path / "state.json").write_text('{"cooldowns": {"m1"', encoding="utf-8")

    recovered = store.snapshot()
    assert recovered.get("cooldowns"), "a corrupt state must not silently become empty"
    assert "recovered runtime state" in capsys.readouterr().out


def test_unreadable_state_without_backups_still_degrades_to_empty(tmp_path):
    store = StateStore(state_path=tmp_path / "state.json", lock_path=tmp_path / "state.lock", backup_dir=tmp_path / "backups")
    (tmp_path / "state.json").write_text("{not json", encoding="utf-8")

    assert store.snapshot() == {}


def test_oversized_jsonl_log_rotates_and_keeps_a_bounded_number_of_segments(tmp_path):
    # The append-only logs had no rotation at all — only the derived SQLite index was bounded —
    # so a busy agent host grew them until the disk filled, at which point state writes start
    # failing too.
    from ficelle.json_store import open_private_append, rotate_if_oversized

    path = tmp_path / "routes.jsonl"
    for _ in range(400):
        rotate_if_oversized(path, 4096, keep=3)
        with open_private_append(path) as handle:
            handle.write("x" * 200 + "\n")

    segments = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("routes.jsonl"))
    assert segments == ["routes.jsonl", "routes.jsonl.1", "routes.jsonl.2", "routes.jsonl.3"]
    assert not (tmp_path / "routes.jsonl.4").exists(), "retention must be bounded"
    for name in segments:
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o600
        assert (tmp_path / name).stat().st_size <= 8192


def test_rotation_is_skipped_below_the_threshold_and_never_raises(tmp_path):
    from ficelle.json_store import rotate_if_oversized

    path = tmp_path / "small.jsonl"
    path.write_text("one line\n", encoding="utf-8")
    rotate_if_oversized(path, 4096)
    assert path.read_text(encoding="utf-8") == "one line\n"
    assert not (tmp_path / "small.jsonl.1").exists()

    # A missing file, and an unwritable directory, must both be survivable: losing a rotation is
    # acceptable, losing the write that follows it is not.
    rotate_if_oversized(tmp_path / "absent.jsonl", 1)


def test_state_backups_are_as_private_as_the_state_itself(tmp_path):
    # Backups hold full copies of runtime state. `logs/` was tightened but this directory was
    # still created at the process umask — found by listing a live instance's runtime home, not
    # by a test, which is why the check now lives here.
    store = StateStore(
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "state.lock",
        backup_dir=tmp_path / "state-backups",
        backup_min_interval_seconds=0,
    )
    store.update(lambda state: state.setdefault("cooldowns", {}).update({"m": {"until": 1}}))
    store.update(lambda state: state.setdefault("cooldowns", {}).update({"m2": {"until": 1}}))

    assert (tmp_path / "state-backups").stat().st_mode & 0o777 == 0o700
    for backup in (tmp_path / "state-backups").iterdir():
        assert backup.stat().st_mode & 0o777 == 0o600, backup


@pytest.fixture
def platform_without_posix_modes(monkeypatch):
    """Make the writer modules see the mode behavior reported by Windows.

    Deleting `fchmod` alone does not reproduce the failure: on POSIX
    `os.open(..., O_CREAT, 0o600)` really does produce 0o600, so the guard short-circuits and
    the missing function is never reached. Windows reports 0o666 for a writable file whatever
    mode was requested, which is what drives the call in the first place. Both halves are
    needed, and CI only ever runs on Linux — which is how this reached a real Windows host.

    Keep the simulation local to the two modules under test. Patching the process-wide
    ``os.fstat`` also changes what pytest, tempfile, and file wrappers can observe, and rebuilding
    an ``os.stat_result`` portably would require preserving platform-specific fields.
    """
    from ficelle import json_store, synthetic_health

    class WindowsLikeOs:
        def __getattr__(self, name):
            if name == "fchmod":
                raise AttributeError(name)
            return getattr(os, name)

        def fstat(self, fd):
            result = os.fstat(fd)
            return SimpleNamespace(st_mode=(result.st_mode & ~0o777) | 0o666)

    windows_like_os = WindowsLikeOs()
    monkeypatch.setattr(json_store, "os", windows_like_os)
    monkeypatch.setattr(synthetic_health, "os", windows_like_os)


def test_private_text_is_written_where_the_platform_has_no_fchmod(tmp_path, platform_without_posix_modes):
    from ficelle.json_store import write_private_text

    path = tmp_path / "credentials.env"
    write_private_text(path, "OPENROUTER_API_KEY=value\n")
    assert path.read_text(encoding="utf-8") == "OPENROUTER_API_KEY=value\n"

    # Replacing an existing file takes the same path, and is how `set-key` rewrites credentials.
    write_private_text(path, "OPENROUTER_API_KEY=rotated\n")
    assert path.read_text(encoding="utf-8") == "OPENROUTER_API_KEY=rotated\n"


def test_jsonl_append_survives_a_platform_without_fchmod(tmp_path, platform_without_posix_modes):
    # The route log is appended on every request, so this path failing meant the router could
    # answer /health and list models while every completion raised AttributeError.
    from ficelle.json_store import open_private_append

    path = tmp_path / "routes.jsonl"
    for event in ("first", "second"):
        with open_private_append(path) as handle:
            handle.write('{"event":"%s"}\n' % event)

    assert path.read_text(encoding="utf-8").splitlines() == ['{"event":"first"}', '{"event":"second"}']


def test_synthetic_health_state_survives_a_platform_without_fchmod(tmp_path, platform_without_posix_modes):
    from ficelle.synthetic_health import atomic_write_text

    path = tmp_path / "synthetic-state.json"
    atomic_write_text(path, '{"runs": 1}')
    assert path.read_text(encoding="utf-8") == '{"runs": 1}'


def test_a_real_fchmod_failure_is_not_swallowed(tmp_path, monkeypatch):
    """The Windows guard must not turn a genuine permission failure into a silent success.

    These writers carry credentials, so a fallback that leaves a readable file with no warning
    is worse than a crash.
    """
    from ficelle import json_store

    class OsWithFailingFchmod:
        def __getattr__(self, name):
            return getattr(os, name)

        def fstat(self, fd):
            return SimpleNamespace(st_mode=0o100666)

        def fchmod(self, fd, mode):
            raise PermissionError("refused")

    monkeypatch.setattr(json_store, "os", OsWithFailingFchmod())

    with pytest.raises(PermissionError, match="refused"):
        json_store.write_private_text(tmp_path / "credentials.env", "OPENROUTER_API_KEY=value\n")
