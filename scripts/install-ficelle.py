#!/usr/bin/env python3
"""Install Ficelle from a source checkout.

This wrapper keeps the user-facing script stable while the implementation lives
inside the package as `ficelle.install` / `ficelle-setup`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from ficelle.install import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
