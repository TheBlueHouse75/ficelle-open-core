"""Optional discovery of the closed ``ficelle_pro`` commercial pack.

The open core runs standalone (the free tier). When the licensed ``ficelle-pro``
package is installed it provides the closed assets (the curated provider pack, the
fusion engine, the native compression engine), which the core's per-asset optional
imports pick up. These helpers are the canonical "is Pro present" check — used to
decide whether to surface Pro-only controls (e.g. hide the fusion / native-compression
admin surfaces on a core-only install). See ``docs/prds/open-core-extraction-prd.md``.
"""
from __future__ import annotations

from types import ModuleType


def load_pro() -> ModuleType | None:
    """Return the installed ``ficelle_pro`` module, or ``None`` on a core-only install."""
    try:
        import ficelle_pro
    except ImportError:
        return None
    return ficelle_pro


def pro_installed() -> bool:
    """True when the closed ``ficelle-pro`` pack is installed alongside the core."""
    return load_pro() is not None
