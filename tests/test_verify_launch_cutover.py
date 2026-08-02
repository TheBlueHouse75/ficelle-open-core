from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify-launch-cutover.py"
SPEC = importlib.util.spec_from_file_location("verify_launch_cutover", SCRIPT)
assert SPEC and SPEC.loader
canary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = canary
SPEC.loader.exec_module(canary)


def _fetcher(values: dict[str, canary.Snapshot]):
    def fetch(url: str) -> canary.Snapshot:
        return values.get(url, canary.Snapshot(404, b""))

    return fetch


def _health() -> canary.Snapshot:
    return canary.Snapshot(
        200,
        json.dumps(
            {
                "ok": True,
                "signing_key_configured": True,
                "commerce_configured": True,
                "wheel_available": True,
                "dev_stub_enabled": False,
            }
        ).encode(),
    )


def test_canary_and_bootstrap_pin_the_same_core_hash() -> None:
    bootstrap = (SCRIPT.parent / "bootstrap-ficelle.py").read_text(encoding="utf-8")
    match = re.search(r'DEFAULT_CORE_SHA256 = \(\s*"([0-9a-f]{64})"', bootstrap)

    assert match is not None
    assert match.group(1) == canary.CORE_SHA256


def test_private_phase_requires_anonymous_repo_and_release_404() -> None:
    values = {
        "https://api.github.com/repos/TheBlueHouse75/ficelle-open-core":
            canary.Snapshot(404, b""),
        f"https://github.com/TheBlueHouse75/ficelle-open-core/releases/download/"
        f"v{canary.VERSION}/ficelle_router-{canary.VERSION}-py3-none-any.whl":
            canary.Snapshot(404, b""),
        "https://site.example/": canary.Snapshot(200, b"ready"),
        "https://license.example/healthz": _health(),
    }

    assert canary.verify(
        "private",
        website_url="https://site.example",
        license_url="https://license.example",
        fetcher=_fetcher(values),
    ) == []


def test_public_phase_checks_release_hash_files_bootstrap_and_site(
    monkeypatch,
) -> None:
    wheel = b"wheel"
    expected_hash = hashlib.sha256(wheel).hexdigest()
    monkeypatch.setattr(canary, "CORE_SHA256", expected_hash)
    repository = "TheBlueHouse75/ficelle-open-core"
    release_url = (
        f"https://github.com/{repository}/releases/download/"
        f"v{canary.VERSION}/ficelle_router-{canary.VERSION}-py3-none-any.whl"
    )
    values = {
        f"https://api.github.com/repos/{repository}":
            canary.Snapshot(200, b'{"private": false}'),
        release_url: canary.Snapshot(200, wheel),
        f"https://raw.githubusercontent.com/{repository}/v{canary.VERSION}/"
        "scripts/bootstrap-ficelle.py":
            canary.Snapshot(
                200,
                (
                    f'CORE_VERSION = "{canary.VERSION}"\n'
                    f'CORE_SHA = "{canary.CORE_SHA256}"'
                ).encode(),
            ),
        "https://site.example/": canary.Snapshot(
            200,
            f"/TheBlueHouse75/ficelle-open-core/v{canary.VERSION}"
            "/scripts/bootstrap-ficelle.py".encode(),
        ),
        "https://license.example/healthz": _health(),
    }
    for filename in ("README.md", "LICENSE", "SECURITY.md", "CONTRIBUTING.md"):
        values[
            f"https://raw.githubusercontent.com/{repository}/main/{filename}"
        ] = canary.Snapshot(200, b"ok")

    failures = canary.verify(
        "public",
        website_url="https://site.example",
        license_url="https://license.example",
        fetcher=_fetcher(values),
    )

    assert failures == []
