#!/usr/bin/env python3
"""Anonymous launch preflight for the private guard and final public cutover."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

REPOSITORY = "TheBlueHouse75/ficelle-open-core"
VERSION = "0.1.5"
CORE_WHEEL = f"ficelle_router-{VERSION}-py3-none-any.whl"
CORE_SHA256 = "335a5edd6e662dee386a46a12575a4b7fc75dac7cf85a62e4287498d6102d7ed"
BOOTSTRAP_PATH = f"/{REPOSITORY}/v{VERSION}/scripts/bootstrap-ficelle.py"


@dataclass(frozen=True)
class Snapshot:
    status: int
    body: bytes


Fetcher = Callable[[str], Snapshot]


def fetch(url: str) -> Snapshot:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ficelle-launch-canary/0.1.5"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return Snapshot(int(response.status), response.read())
    except urllib.error.HTTPError as exc:
        return Snapshot(exc.code, exc.read())


def _health_is_ready(snapshot: Snapshot) -> bool:
    if snapshot.status != 200:
        return False
    try:
        payload = json.loads(snapshot.body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        payload.get("ok") is True
        and payload.get("signing_key_configured") is True
        and payload.get("commerce_configured") is True
        and payload.get("wheel_available") is True
        and payload.get("dev_stub_enabled") is False
    )


def verify(
    phase: str,
    *,
    website_url: str,
    license_url: str,
    fetcher: Fetcher = fetch,
) -> list[str]:
    failures: list[str] = []
    api_url = f"https://api.github.com/repos/{REPOSITORY}"
    release_url = (
        f"https://github.com/{REPOSITORY}/releases/download/v{VERSION}/{CORE_WHEEL}"
    )
    bootstrap_url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/v{VERSION}/"
        "scripts/bootstrap-ficelle.py"
    )

    repository = fetcher(api_url)
    release = fetcher(release_url)
    website = fetcher(website_url.rstrip("/") + "/")
    health = fetcher(license_url.rstrip("/") + "/healthz")

    if website.status != 200:
        failures.append(f"website returned HTTP {website.status}")
    if not _health_is_ready(health):
        failures.append("license service delivery readiness failed")

    if phase == "private":
        if repository.status != 404:
            failures.append(
                f"repository must remain anonymous-404 before cutover, got {repository.status}"
            )
        if release.status != 404:
            failures.append(
                f"release must remain anonymous-404 before cutover, got {release.status}"
            )
        return failures

    if repository.status != 200:
        failures.append(f"repository returned HTTP {repository.status}")
    else:
        try:
            repository_payload = json.loads(repository.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            failures.append("repository API returned invalid JSON")
        else:
            if repository_payload.get("private") is not False:
                failures.append("repository API still reports private")

    if release.status != 200:
        failures.append(f"Core release wheel returned HTTP {release.status}")
    elif hashlib.sha256(release.body).hexdigest() != CORE_SHA256:
        failures.append("Core release wheel SHA-256 does not match the pinned installer")

    bootstrap = fetcher(bootstrap_url)
    if bootstrap.status != 200:
        failures.append(f"versioned bootstrap returned HTTP {bootstrap.status}")
    else:
        text = bootstrap.body.decode("utf-8", errors="replace")
        if CORE_SHA256 not in text or f'CORE_VERSION = "{VERSION}"' not in text:
            failures.append("versioned bootstrap does not pin the release wheel and version")

    for filename in ("README.md", "LICENSE", "SECURITY.md", "CONTRIBUTING.md"):
        raw_url = (
            f"https://raw.githubusercontent.com/{REPOSITORY}/main/{filename}"
        )
        snapshot = fetcher(raw_url)
        if snapshot.status != 200:
            failures.append(f"{filename} returned HTTP {snapshot.status}")

    website_text = website.body.decode("utf-8", errors="replace")
    if BOOTSTRAP_PATH not in website_text:
        failures.append("website does not advertise the versioned public bootstrap")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("private", "public"))
    parser.add_argument(
        "--website-url",
        default="https://ficelle-website.netlify.app",
    )
    parser.add_argument(
        "--license-url",
        default="https://ficelle-license-service.onrender.com",
    )
    args = parser.parse_args()
    failures = verify(
        args.phase,
        website_url=args.website_url,
        license_url=args.license_url,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"Ficelle {args.phase} launch canary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
