from __future__ import annotations

from pathlib import Path

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
