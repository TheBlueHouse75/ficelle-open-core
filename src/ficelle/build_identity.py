"""Build identity of the importable ``ficelle`` package of the current process.

Shared by three consumers that must agree on the shape: the synthetic-health
harness records its own identity in ``run.json``, the router snapshots and
serves its identity from ``/health`` and ``/admin/status.json``, and
``ficelle doctor`` compares both. The identity of the *harness* process says
nothing about the *service* on the loopback port: on 2026-08-10 a LaunchAgent
venv served a week-old wheel while the checkout carried the fixes under
validation, and both sides reported the same ``ficelle_version`` — only the
runtime content hash discriminates.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from ficelle import __version__

_CACHED_IDENTITY: dict[str, Any] | None = None


def package_build_identity() -> dict[str, Any]:
    """Identify the exact build this process imports (L1-R2): version, package path,
    install mode, git commit and dirtiness for editable installs, plus a content hash
    of the runtime files actually importable. ``complete`` gates baseline acceptance —
    a reference baseline whose build cannot be reproduced is not comparable.

    Computed once per process (two git subprocesses plus a hash of every package file):
    the snapshot must describe what this process imported, not whatever a later edit
    leaves on disk — which is also why the router warms it at startup."""
    global _CACHED_IDENTITY
    if _CACHED_IDENTITY is None:
        _CACHED_IDENTITY = _collect_identity()
    return dict(_CACHED_IDENTITY)


def _collect_identity() -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "ficelle_version": __version__,
        "package_dir": str(package_dir),
    }
    repo_root = package_dir.parent.parent
    editable = (repo_root / ".git").exists() and (repo_root / "src" / "ficelle").exists()
    identity["install_mode"] = "editable" if editable else "distribution"
    identity["git_commit"] = None
    identity["git_dirty"] = None
    if editable:
        try:
            identity["git_commit"] = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            identity["git_dirty"] = bool(
                subprocess.run(
                    ["git", "-C", str(repo_root), "status", "--porcelain"],
                    capture_output=True, text=True, timeout=10, check=True,
                ).stdout.strip()
            )
        except Exception:
            identity["git_commit"] = None
            identity["git_dirty"] = None
    hasher = hashlib.sha256()
    try:
        for path in sorted(package_dir.rglob("*.py")):
            hasher.update(path.relative_to(package_dir).as_posix().encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(path.read_bytes())
        identity["runtime_content_hash"] = hasher.hexdigest()
    except OSError:
        identity["runtime_content_hash"] = None
    identity["complete"] = bool(
        identity["runtime_content_hash"]
        and (not editable or identity["git_commit"] is not None)
    )
    return identity
