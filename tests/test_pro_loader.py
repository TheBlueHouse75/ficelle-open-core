"""Tests for the optional ``ficelle_pro`` discovery hook.

See ``docs/prds/open-core-extraction-prd.md``. The monorepo dev/test environment has
``ficelle-pro`` on the path (it holds the closed assets), so the core detects it; a
core-only install is simulated by blocking the ``ficelle_pro`` import in a subprocess.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ficelle.pro import load_pro, pro_installed

# Subprocesses do NOT inherit pytest's `pythonpath` ini setting, so point them at this
# worktree's sources explicitly (so they import this branch's code, not an installed copy).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATHS = (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "ficelle-pro" / "src"))


def _subprocess_env() -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH")
    paths = [*_SRC_PATHS, existing] if existing else list(_SRC_PATHS)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}


def _run_pro_subprocess(
    code: str,
    *,
    hermes_home: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _subprocess_env()
    if hermes_home is not None:
        env["HERMES_HOME"] = str(hermes_home)
        env["FICELLE_HOME"] = str(hermes_home / ".ficelle")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )


def test_pro_present_in_dev_environment() -> None:
    # The monorepo dev/test env has ficelle-pro on the path. A genuinely core-only
    # environment (e.g. the public mirror) legitimately lacks it, so skip there.
    if not pro_installed():
        pytest.skip("ficelle-pro not installed (core-only environment)")
    assert load_pro() is not None


def test_pro_admin_views_served_and_injected(tmp_path: Path) -> None:
    # When the pack is present, its Fusion/Compression views are served under
    # /admin/static/pro/ and the dashboard shell injects the script. (Dev/test env
    # has the pack on the path; a real core-only mirror legitimately lacks it.)
    if not pro_installed():
        pytest.skip("ficelle-pro not installed (core-only environment)")
    code = (
        "import ficelle.router as router; "
        "a = router.load_admin_asset('pro/pro-views.js'); "
        "assert a is not None and a[1].startswith('application/javascript'), a; "
        "assert '/admin/static/pro/pro-views.js' in router.admin_page_html()"
    )
    result = _run_pro_subprocess(code, hermes_home=tmp_path)
    assert result.returncode == 0, result.stderr


def test_admin_state_exposes_pro_installed_flag(tmp_path: Path) -> None:
    # admin_state must surface pro_installed for the frontend gating. Run with isolated
    # Ficelle and Hermes homes so it never touches the real runtime config; the dev/test
    # path has the pack, so the flag is True and matches pro_installed().
    code = (
        "import os; "
        "from pathlib import Path; "
        "import ficelle.router as router; "
        "from ficelle.pro import pro_installed; "
        "assert router.FICELLE_HOME == Path(os.environ['FICELLE_HOME']); "
        "assert router.admin_state(router.load_config())['pro_installed'] == pro_installed()"
    )
    result = _run_pro_subprocess(code, hermes_home=tmp_path)
    assert result.returncode == 0, result.stderr


def test_pro_present_provides_full_pool_and_engines(tmp_path: Path) -> None:
    # AC#2: with the pack installed, the curated provider pool is composed into the
    # default config and both closed engines (fusion + native compression) are present.
    if not pro_installed():
        pytest.skip("ficelle-pro not installed (core-only environment)")
    code = (
        "import ficelle.router as router; "
        "from ficelle_pro.provider_pack import PROVIDER_PACK; "
        "assert PROVIDER_PACK, 'curated pack is empty'; "
        "providers = set(router.DEFAULT_CONFIG['providers']); "
        # Real composition: the pool is the core reference set PLUS the curated pack,
        # so it must contain both and be strictly larger than core-only.
        "assert set(router.CORE_PROVIDERS) | set(PROVIDER_PACK) <= providers; "
        "assert len(providers) > len(router.CORE_PROVIDERS); "
        "assert router.FusionRunner is not None; "
        "assert router.compress_block is not None"
    )
    result = _run_pro_subprocess(code, hermes_home=tmp_path)
    assert result.returncode == 0, result.stderr


def test_pro_absent_is_detected() -> None:
    # Simulate a core-only install: block the ficelle_pro import in a fresh interpreter.
    code = (
        "import sys; sys.modules['ficelle_pro'] = None; "
        "from ficelle.pro import load_pro, pro_installed; "
        "assert load_pro() is None; assert pro_installed() is False; print('ok')"
    )
    result = _run_pro_subprocess(code)
    assert result.returncode == 0, result.stderr
