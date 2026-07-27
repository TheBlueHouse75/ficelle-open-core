"""Open-core extraction boundary guardrails.

See ``docs/prds/open-core-extraction-prd.md``. The public ``ficelle`` core must
import AND run the free-tier path without the closed Pro assets. Importing alone is
not enough: a missing engine's config-layer symbols would be dereferenced in
``load_config()`` (config normalization), so each check exercises the real free path
in a fresh interpreter with an isolated ``HERMES_HOME``.

Optional closed assets:

- Native compression falls back to ``compression_fallback`` (mode ``"off"``).
- The fusion *engine* (``FusionRunner`` + panel/judge/ranking) is optional because
  its config layer was split into ``fusion_config`` (always in the core); when absent
  fusion is simply not visible in the model list.
- The curated provider pack (``ficelle_pro.provider_pack``) is optional; when absent
  ``DEFAULT_CONFIG["providers"]`` reduces to ``CORE_PROVIDERS`` (the reference
  providers), so a core-only install never ships the grey-market relays.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Exercise the free path, not just the import: load_config() is where a missing
# engine's config-layer symbols would be dereferenced, and normalize_compression_settings
# is the compression symbol reached on the settings/observability path (it must resolve
# to the "off" default when the native engine is absent).
_FREE_PATH_BODY = (
    "cfg = router.load_config(); "
    "router.listed_virtual_model_ids(cfg); "
    "assert router.normalize_compression_settings(cfg.get('compression'))['mode'] == 'off'"
)

# Closed Pro assets, individually and together.
_COMPRESSION = ("ficelle_pro.compression",)
_FUSION_ENGINE = ("ficelle_pro.fusion",)
_PROVIDER_PACK = ("ficelle_pro.provider_pack",)
_ALL_CLOSED = _COMPRESSION + _FUSION_ENGINE + _PROVIDER_PACK

# Subprocesses do NOT inherit pytest's `pythonpath` ini setting, so point them at this
# worktree's core + closed-pack sources explicitly — robust to how pytest is invoked
# (plain `pytest` or with a manual PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATHS = (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "ficelle-pro" / "src"))


def _run_core_with_modules_blocked(
    blocked: tuple[str, ...], home: Path, body: str = _FREE_PATH_BODY
) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter with ``blocked`` modules unavailable.

    ``sys.modules[name] = None`` makes any ``import name`` raise ModuleNotFoundError,
    faithfully simulating a core-only install. ``HERMES_HOME`` is isolated so
    ``load_config`` never touches the real runtime config.
    """
    code = textwrap.dedent(
        f"""
        import sys
        for _name in {list(blocked)!r}:
            sys.modules[_name] = None
        import ficelle.router as router
        {body}
        """
    )
    existing = os.environ.get("PYTHONPATH")
    pythonpath = os.pathsep.join([*_SRC_PATHS, existing] if existing else _SRC_PATHS)
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": pythonpath}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_core_runs_free_path_without_native_compression(tmp_path: Path) -> None:
    result = _run_core_with_modules_blocked(_COMPRESSION, tmp_path)
    assert result.returncode == 0, result.stderr


def test_core_runs_free_path_without_fusion_engine(tmp_path: Path) -> None:
    result = _run_core_with_modules_blocked(_FUSION_ENGINE, tmp_path)
    assert result.returncode == 0, result.stderr


def test_core_runs_free_path_without_provider_pack(tmp_path: Path) -> None:
    result = _run_core_with_modules_blocked(_PROVIDER_PACK, tmp_path)
    assert result.returncode == 0, result.stderr


def test_core_providers_reduce_to_reference_set_without_pack(tmp_path: Path) -> None:
    body = (
        "assert list(router.DEFAULT_CONFIG['providers']) == list(router.CORE_PROVIDERS), "
        "list(router.DEFAULT_CONFIG['providers'])"
    )
    result = _run_core_with_modules_blocked(_PROVIDER_PACK, tmp_path, body)
    assert result.returncode == 0, result.stderr


def test_core_runs_free_path_without_any_closed_asset(tmp_path: Path) -> None:
    result = _run_core_with_modules_blocked(_ALL_CLOSED, tmp_path)
    assert result.returncode == 0, result.stderr


def test_core_never_lists_auto_fusion_without_engine(tmp_path: Path) -> None:
    # Even when the config tries to enable + expose Fusion, a core-only install must
    # not advertise ficelle/auto-fusion: the engine is absent and routing to it would
    # otherwise crash. Visibility is gated on the engine, not just the config.
    body = (
        "cfg = router.load_config(); "
        "cfg['fusion'] = {'enabled': True, 'visible_in_models': True}; "
        "assert router.fusion_visible_in_model_list(cfg) is False; "
        "assert router.FUSION_MODEL_ID not in router.listed_virtual_model_ids(cfg)"
    )
    result = _run_core_with_modules_blocked(_FUSION_ENGINE, tmp_path, body)
    assert result.returncode == 0, result.stderr


def test_core_fusion_request_rejects_cleanly_without_engine(tmp_path: Path) -> None:
    # A client that targets ficelle/auto-fusion directly must get a clean 404, never a
    # 'NoneType object is not callable' crash from the absent FusionRunner/preflight.
    body = (
        "status, payload, _ = router.run_fusion_chat_completion("
        "{'model': router.FUSION_MODEL_ID, 'messages': []}, router.load_config(), {}, 'req', 0.0, 0); "
        "assert status == 404, status; "
        "assert payload['error']['type'] == 'not_found', payload"
    )
    result = _run_core_with_modules_blocked(_FUSION_ENGINE, tmp_path, body)
    assert result.returncode == 0, result.stderr


def test_admin_state_reports_pro_absent_without_pack(tmp_path: Path) -> None:
    # The admin UI gates Fusion/Compression rendering on this flag; it must be False on
    # a core-only install so the UI shows the upsell placeholder, not hollow controls.
    # Block the whole ficelle_pro package (not just submodules) to model a real
    # core-only install, where pro_installed() — i.e. `import ficelle_pro` — fails.
    body = "assert router.admin_state(router.load_config())['pro_installed'] is False"
    result = _run_core_with_modules_blocked(("ficelle_pro",), tmp_path, body)
    assert result.returncode == 0, result.stderr


def test_core_lists_auto_profiles_without_pro(tmp_path: Path) -> None:
    # AC#1: a core-only install must still list every ficelle/auto-* virtual profile
    # (the free tier routes through these), while never advertising auto-fusion.
    body = (
        "cfg = router.load_config(); ids = set(router.listed_virtual_model_ids(cfg)); "
        "missing = router.VIRTUAL_MODELS - ids; "
        "assert not missing, sorted(missing); "
        "assert router.FUSION_MODEL_ID not in ids"
    )
    result = _run_core_with_modules_blocked(("ficelle_pro",), tmp_path, body)
    assert result.returncode == 0, result.stderr


def test_core_does_not_serve_pro_admin_assets(tmp_path: Path) -> None:
    # The Fusion/Compression view assets ship in ficelle_pro and are served under
    # /admin/static/pro/. A core-only install must not resolve them, and the path
    # must stay traversal-safe (no escaping the pack dir).
    body = (
        "assert router._pro_admin_dir() is None; "
        "assert router.load_admin_asset('pro/pro-views.js') is None; "
        "assert router.load_admin_asset('pro/../../router.py') is None"
    )
    result = _run_core_with_modules_blocked(("ficelle_pro",), tmp_path, body)
    assert result.returncode == 0, result.stderr


def test_core_dashboard_omits_pro_script(tmp_path: Path) -> None:
    # The dashboard shell must not inject the Pro-views script on a core-only install.
    body = "assert '/admin/static/pro/' not in router.admin_page_html()"
    result = _run_core_with_modules_blocked(("ficelle_pro",), tmp_path, body)
    assert result.returncode == 0, result.stderr
