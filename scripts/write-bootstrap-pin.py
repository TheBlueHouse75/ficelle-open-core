#!/usr/bin/env python3
"""Write the Hermes catalog plugin's bootstrap pin from the documented bootstrap pin.

The catalog copy of the Ficelle provider downloads `scripts/bootstrap-ficelle.py` from the public
mirror and runs it only if its SHA-256 matches this pin. The pin is not a second source of truth:
it repeats the immutable mirror commit already used by every documented bootstrap command (the
bootstrap docstring) and the verify-before-run checksum in `docs/install.md`, both refreshed for
each release as described in `docs/client-release-runbook.md` §1. Run this right after that refresh.

    python scripts/write-bootstrap-pin.py            # write the pin
    python scripts/write-bootstrap-pin.py --check    # exit 1 if the pin is missing or stale

The pin never hashes the local bootstrap: between releases that file is ahead of the published
mirror commit. It also stays out of the Core wheel (see MANIFEST.in).

The compatible Core range defaults to the pinned release's own minor series (`0.3.x` for 0.3.13):
the plugin only talks to Core's stable OpenAI-compatible surface, and a new minor is where a
compatibility decision should be made on purpose.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap-ficelle.py"
INSTALL_DOC = REPO_ROOT / "docs" / "install.md"
PIN_PATH = REPO_ROOT / "src" / "ficelle" / "assets" / "hermes-plugin" / "ficelle" / "bootstrap-pin.json"

# A full commit SHA, never a tag: a tag can be moved, and the plugin refuses anything else.
URL_RE = re.compile(
    r"https://raw\.githubusercontent\.com/TheBlueHouse75/ficelle-open-core/[0-9a-f]{40}/scripts/bootstrap-ficelle\.py"
)
RELEASE_RE = re.compile(r"pinned to public mirror release v(\d+)\.(\d+)\.(\d+)")
CHECKSUM_RE = re.compile(r"'([0-9a-f]{64})' bootstrap-ficelle\.py")


def _only(found: list, what: str, source: Path):
    distinct = sorted(set(found))
    if len(distinct) != 1:
        raise SystemExit(f"{source}: expected exactly one {what}, found {len(distinct)}")
    return distinct[0]


def build_pin(
    bootstrap: Path, install_doc: Path, min_core: str | None, max_core: str | None
) -> dict[str, object]:
    """The pin implied by the documented bootstrap command.

    `bootstrap_sha256` is absent when `install_doc` does not exist: the public mirror omits that
    page, so a check there can only compare the commit and the release.
    """
    text = bootstrap.read_text(encoding="utf-8")
    major, minor, patch = _only(RELEASE_RE.findall(text), "pinned mirror release", bootstrap)
    pin: dict[str, object] = {
        "release_tag": f"v{major}.{minor}.{patch}",
        "bootstrap_url": _only(URL_RE.findall(text), "commit-pinned bootstrap URL", bootstrap),
        "core_version_range": {
            "min": min_core or f"{major}.{minor}.0",
            "max_exclusive": max_core or f"{major}.{int(minor) + 1}.0",
        },
    }
    if install_doc.is_file():
        pin["bootstrap_sha256"] = _only(
            CHECKSUM_RE.findall(install_doc.read_text(encoding="utf-8")), "bootstrap checksum", install_doc
        )
    return pin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Verify the existing pin instead of writing it.")
    parser.add_argument("--min-core", help="Lowest compatible Core version (default: X.Y.0 of the pinned release).")
    parser.add_argument("--max-core", help="First incompatible Core version (default: next minor).")
    parser.add_argument("--bootstrap", type=Path, default=BOOTSTRAP, help=argparse.SUPPRESS)
    parser.add_argument("--install-doc", type=Path, default=INSTALL_DOC, help=argparse.SUPPRESS)
    parser.add_argument("--pin", type=Path, default=PIN_PATH, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    expected = build_pin(args.bootstrap, args.install_doc, args.min_core, args.max_core)
    if args.check:
        try:
            current = json.loads(args.pin.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = None
        # The range is a release decision (`--min-core`/`--max-core`), not derived from the
        # documented pin, so a deliberately widened range must not read as stale unless one is given.
        ignored = set() if (args.min_core or args.max_core) else {"core_version_range"}
        if not isinstance(current, dict) or any(
            current.get(key) != value for key, value in expected.items() if key not in ignored
        ):
            print(f"{args.pin}: missing or stale; run scripts/write-bootstrap-pin.py", file=sys.stderr)
            return 1
        return 0
    if "bootstrap_sha256" not in expected:
        raise SystemExit(f"{args.install_doc}: not found; the pin is written from the private repository")
    args.pin.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.pin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
