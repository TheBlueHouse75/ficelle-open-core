#!/usr/bin/env python3
"""Audit the Core release artifacts before they reach a public index.

Publishing is irreversible: PyPI refuses to re-use a filename even after a delete, and a file
that was downloadable for one minute is downloadable forever. Two mistakes are unrecoverable
here, so this script exists to make both of them fail loudly instead of silently succeeding:

1. uploading the paid pack. It is served only by the license service after an entitlement
   check, and publishing it would give the commercial product away;
2. uploading artifacts built from the **private monorepo** rather than the public mirror. That
   is not a hypothetical: a monorepo sdist used to carry 112 files under ``docs/`` (the
   provider dossiers, the GTM plan, the Pro licensing PRD), ``prepare-public-mirror.py``,
   ``dev-license-service.py`` and the Pro-coupled tests, and a monorepo wheel embeds the
   private README as its long description.

Every check below is structural, digest-based, or a comparison between the two artifacts.
None of them names a closed identifier, because this script ships in the public mirror and a
guard that has to spell out the secret it protects is not a guard.

    python scripts/audit-release-artifacts.py dist/

Exit status is 0 when every artifact is publishable and 1 otherwise, with each failure printed
on its own line — a release manager reading CI logs should not have to re-run to find the
second problem.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The distribution name is fixed. Anything else in the upload directory is either the paid pack
# or a stray artifact, and both must stop the publish rather than ride along with it.
PROJECT_NAME = "ficelle-router"
DIST_STEM = "ficelle_router"

# Suffixes pip and twine would treat as an uploadable distribution. Checksums, signatures and
# release notes are ignored; anything installable has to be recognised.
DIST_SUFFIXES = (".whl", ".tar.gz", ".zip", ".egg")

# A wheel holds the import package and its metadata directory. Nothing else has ever been in
# one here, so an unexpected top-level entry means the build tree was not the mirror.
WHEEL_TOP_LEVEL = {"ficelle"}

# The sdist ships `src/` plus packaging metadata — see MANIFEST.in for why `docs/`, `scripts/`,
# `plugins/` and `tests/` are pruned. Listing the permitted set rather than the forbidden one
# means a newly added private directory fails closed. Entries are optional: releases built from
# a mirror that carried no MANIFEST.in have neither `CHANGELOG.md` nor `MANIFEST.in`.
SDIST_TOP_LEVEL = {
    "CHANGELOG.md",
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
    "src",
}
SDIST_SRC_ENTRIES = {"ficelle", f"{DIST_STEM}.egg-info"}

# Up to and including this release the mirror shipped no MANIFEST.in, so setuptools added
# `tests/` to the sdist on its own. Those suites are the *mirror's* — core-only and
# grey-market-clean by construction — so tolerating them is safe for artifacts already built.
# The tolerance must not outlive them: from the next release MANIFEST.in prunes `tests/`, so a
# tests directory would mean the sdist came from a tree that bypassed it, and the monorepo's
# suite is exactly what must never ship. Bounding it here makes it expire on its own instead of
# becoming a standing hole nobody remembers to close.
LAST_RELEASE_WITH_TESTS_IN_SDIST = (0, 1, 7)


def pinned_release(bootstrap: Path) -> tuple[str, str]:
    """Return the (version, wheel SHA-256) the installer verifies against.

    Reusing the installer's own pin is the point: Ficelle distributes one wheel through three
    channels (the GitHub Release, the bootstrap, and now PyPI), and they are only the same
    artifact if they are the same bytes. Parsed rather than imported so this stays runnable
    against a checkout of any tag.
    """
    text = bootstrap.read_text(encoding="utf-8")
    digests = re.findall(r'"([0-9a-f]{64})"', text)
    versions = re.findall(r'^CORE_VERSION = "([^"]+)"', text, re.MULTILINE)
    if len(digests) != 1 or len(versions) != 1:
        raise SystemExit(
            f"{bootstrap}: expected exactly one CORE_VERSION and one digest, "
            f"found {len(versions)} and {len(digests)}"
        )
    return versions[0], digests[0]


def _release_predates_the_tests_prune(version: str) -> bool:
    """Whether ``tests/`` in this version's sdist is the known-clean legacy shape.

    Only the plain ``X.Y.Z`` scheme this project releases under is understood. Anything else —
    a pre-release, a local version, a typo — reads as newer, so the tolerance is withheld and
    the audit fails closed rather than guessing in the permissive direction.
    """
    try:
        parsed = tuple(int(part) for part in version.split("."))
    except ValueError:
        return False
    return parsed <= LAST_RELEASE_WITH_TESTS_IN_SDIST


def _long_description(headers: str) -> str:
    """Return the long description carried by a METADATA / PKG-INFO blob."""
    message = Parser().parsestr(headers)
    # setuptools writes the readme as the message body, but older metadata versions used a
    # folded `Description:` header. Accept either so this does not depend on the build's age.
    payload = message.get_payload()
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return (message.get("Description") or "").strip()


def audit_wheel(path: Path, version: str, expected_sha256: str, failures: list[str]) -> str:
    """Check the wheel and return its long description for the cross-artifact comparison."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        # This is the strongest check in the file. The pinned digest is only ever produced by a
        # mirror build of the released tag, so a mismatch means these are not the bytes the
        # installer verifies — a rebuild, a different tree, or a tampered asset.
        failures.append(
            f"{path.name}: sha256 {digest} does not match the pinned {expected_sha256}. "
            "PyPI would serve a different artifact than the bootstrap verifies."
        )

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        dist_info = f"{DIST_STEM}-{version}.dist-info"
        unexpected = sorted(
            {name.split("/", 1)[0] for name in names} - WHEEL_TOP_LEVEL - {dist_info}
        )
        if unexpected:
            failures.append(f"{path.name}: unexpected top-level entries {unexpected}")
        try:
            metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        except KeyError:
            failures.append(f"{path.name}: no {dist_info}/METADATA")
            return ""

    message = Parser().parsestr(metadata)
    if (message.get("Name") or "").strip().lower().replace("_", "-") != PROJECT_NAME:
        failures.append(f"{path.name}: metadata name is {message.get('Name')!r}, not {PROJECT_NAME!r}")
    if (message.get("Version") or "").strip() != version:
        failures.append(f"{path.name}: metadata version is {message.get('Version')!r}, not {version!r}")
    description = _long_description(metadata)
    if not description:
        # The PyPI project page is the long description; an empty one ships a blank page and
        # cannot be fixed without cutting a new version.
        failures.append(f"{path.name}: empty long description — the PyPI page would be blank")
    return description


def audit_sdist(path: Path, version: str, wheel_description: str, failures: list[str]) -> None:
    root = f"{DIST_STEM}-{version}"
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            # A release tarball is unpacked by pip on someone else's machine. Links and paths
            # that escape the root are how that turns into a write outside the build directory.
            if member.issym() or member.islnk():
                failures.append(f"{path.name}: contains a link member {member.name!r}")
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                failures.append(f"{path.name}: contains an escaping path {member.name!r}")

        names = [member.name for member in members]
        roots = {name.split("/", 1)[0] for name in names}
        if roots != {root}:
            failures.append(f"{path.name}: expected a single {root}/ root, found {sorted(roots)}")
            return

        permitted = set(SDIST_TOP_LEVEL)
        if _release_predates_the_tests_prune(version):
            permitted.add("tests")
        top_level = {
            name.split("/")[1] for name in names if name.count("/") >= 1 and name != root
        } - {""}
        unexpected = sorted(top_level - permitted)
        if unexpected:
            failures.append(
                f"{path.name}: unexpected top-level entries {unexpected} — a private directory "
                "reached the sdist (see MANIFEST.in)"
            )
        src = {
            name.split("/")[2]
            for name in names
            if name.startswith(f"{root}/src/") and name.count("/") >= 2
        } - {""}
        unexpected_src = sorted(src - SDIST_SRC_ENTRIES)
        if unexpected_src:
            failures.append(f"{path.name}: unexpected src/ entries {unexpected_src}")

        extracted = archive.extractfile(f"{root}/PKG-INFO")
        if extracted is None:
            failures.append(f"{path.name}: no PKG-INFO")
            return
        pkg_info = extracted.read().decode("utf-8")

    # The wheel is digest-verified above, so it is known to come from the mirror. Requiring the
    # sdist to carry the same long description transfers that guarantee: the private tree's
    # README differs from the public one, so a monorepo-built sdist cannot pass this.
    if wheel_description and _long_description(pkg_info) != wheel_description:
        failures.append(
            f"{path.name}: long description differs from the wheel's — the two artifacts were "
            "not built from the same tree"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="directory holding the artifacts to publish")
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=REPO_ROOT / "scripts" / "bootstrap-ficelle.py",
        help="installer script carrying the pinned CORE_VERSION and wheel digest",
    )
    args = parser.parse_args(argv)

    version, expected_sha256 = pinned_release(args.bootstrap)
    wheel_name = f"{DIST_STEM}-{version}-py3-none-any.whl"
    sdist_name = f"{DIST_STEM}-{version}.tar.gz"

    if not args.directory.is_dir():
        raise SystemExit(f"{args.directory}: not a directory")

    failures: list[str] = []
    dists: dict[str, Path] = {}
    for entry in sorted(args.directory.iterdir()):
        if not entry.is_file():
            continue
        if not entry.name.endswith(DIST_SUFFIXES):
            print(f"ignored (not a distribution): {entry.name}")
            continue
        if entry.name not in {wheel_name, sdist_name}:
            # Deliberately blunt. The only way the paid pack reaches this directory is by
            # mistake, and the mistake is not recoverable once the upload succeeds.
            failures.append(
                f"{entry.name}: not a {PROJECT_NAME} {version} artifact. Only {wheel_name} and "
                f"{sdist_name} may be published; never publish the paid pack."
            )
            continue
        dists[entry.name] = entry

    wheel = dists.get(wheel_name)
    if wheel is None:
        failures.append(f"{args.directory}: {wheel_name} is missing")
        description = ""
    else:
        description = audit_wheel(wheel, version, expected_sha256, failures)

    sdist = dists.get(sdist_name)
    if sdist is None:
        failures.append(f"{args.directory}: {sdist_name} is missing")
    else:
        audit_sdist(sdist, version, description, failures)

    if failures:
        print(f"\nREFUSING to publish {PROJECT_NAME} {version}:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\n{PROJECT_NAME} {version}: {len(dists)} artifact(s) cleared for publication")
    print(f"  wheel sha256 {expected_sha256} matches the installer pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
