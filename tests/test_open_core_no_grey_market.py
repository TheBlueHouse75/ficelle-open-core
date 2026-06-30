"""Guard: the open core (``src/ficelle/``) must contain no grey-market relay identifiers.

The reseller relays and their endpoints live only in the closed ``ficelle_pro`` pack,
and the forbidden-identifier list itself is **owned by the pack**
(``ficelle_pro.provider_pack.GREY_MARKET_IDENTIFIERS``) — so the public open-core tree
never names them in plaintext (naming them here would itself be the leak this guards
against, open-core-extraction-prd.md R9). This test fails if any of those identifiers
leaks back into the open-core tree, which becomes the public BSL mirror.

On a core-only checkout (the public mirror) the pack is absent, so there is nothing to
guard against and the test skips — the mirror is generated clean by the assembler
(``scripts/prepare-public-mirror.py``), which runs the same scan with the pack present.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "ficelle"

try:
    from ficelle_pro.provider_pack import GREY_MARKET_IDENTIFIERS
except ImportError:
    GREY_MARKET_IDENTIFIERS = ()


def test_open_core_has_no_grey_market_identifiers() -> None:
    if not GREY_MARKET_IDENTIFIERS:
        pytest.skip("ficelle_pro absent (core-only / public mirror) — the closed pack owns the forbidden list")
    forbidden = re.compile("|".join(re.escape(name) for name in GREY_MARKET_IDENTIFIERS), re.IGNORECASE)
    # Scan every text asset, not just code: marks (.svg), docs (.md) and config can leak too.
    text_suffixes = {".py", ".json", ".md", ".svg", ".txt", ".toml", ".cfg", ".yaml", ".yml", ".js", ".html", ".css"}
    offenders: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if forbidden.search(line):
                rel = path.relative_to(CORE_ROOT.parents[1])
                offenders.append(f"{rel}:{lineno}: {line.strip()[:80]}")
    assert not offenders, "grey-market identifiers leaked into the open core:\n" + "\n".join(offenders)
