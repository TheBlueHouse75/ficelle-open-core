from __future__ import annotations

from pathlib import Path

from ficelle.config_store import ConfigStore
from ficelle.json_store import atomic_write_json, load_json
from ficelle.runtime_paths import RuntimePaths
from ficelle.state_store import StateStore


def test_runtime_paths_resolve_from_explicit_environment(tmp_path):
    hermes_home = tmp_path / "hermes"
    ficelle_home = tmp_path / "ficelle-home"
    package_dir = tmp_path / "package"

    paths = RuntimePaths.from_env(
        environ={
            "HERMES_HOME": str(hermes_home),
            "FICELLE_HOME": str(ficelle_home),
        },
        package_dir=package_dir,
    )

    assert paths.hermes_home == hermes_home
    router_dir = hermes_home / "ficelle"
    logs_dir = router_dir / "logs"
    assert paths.router_dir == router_dir
    assert paths.config_path == router_dir / "config.json"
    assert paths.state_path == router_dir / "state.json"
    assert paths.catalog_path == router_dir / "catalog.json"
    assert paths.route_log_path == logs_dir / "routes.jsonl"
    assert paths.admin_audit_log_path == logs_dir / "admin-actions.jsonl"
    assert paths.state_lock_path == router_dir / "state.lock"
    assert paths.state_backup_dir == router_dir / "state-backups"
    assert paths.capability_discrepancy_log_path == logs_dir / "capability-discrepancies.jsonl"
    assert paths.compression_store_path == router_dir / "compression.sqlite"
    assert paths.capability_oracle_cache_path == router_dir / "capability_oracle.json"
    assert paths.hermes_agent_dir == hermes_home / "hermes-agent"
    assert paths.hermes_config_path == hermes_home / "config.yaml"
    assert paths.ficelle_home == ficelle_home
    assert paths.credential_env_file == ficelle_home / ".env"
    assert paths.hermes_secrets_keychain == ficelle_home / "hermes-secrets.keychain-db"
    assert paths.admin_assets_dir == package_dir / "assets" / "admin"


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
