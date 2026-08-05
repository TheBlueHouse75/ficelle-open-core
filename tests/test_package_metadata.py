from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import ficelle

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hermes reads these manifests to render `hermes plugins list`, and they ship inside the
# wheel as static assets, so a literal left behind at bump time advertises a version the
# installed artifact does not have (0.1.6 shipped a manifest still claiming 0.1.3). They
# stay literal rather than build-generated: the wheel builds with stock setuptools and no
# build hook, and an editable checkout would never regenerate a build-time value.
PACKAGED_PLUGIN_MANIFESTS = (
    "src/ficelle/assets/hermes-plugin/ficelle/plugin.yaml",
    "src/ficelle/assets/hermes-plugin/ficelle-compression/plugin.yaml",
)


def _manifest_version(path: Path) -> str:
    # Hand-rolled rather than PyYAML: the manifests are four lines we own, and the runtime
    # has no YAML dependency to borrow. Only unindented keys count, so a nested `version:`
    # is ignored the way a root-key lookup would ignore it. Duplicates are rejected instead
    # of resolved, because a reader taking the last one would silently disagree with a
    # reader taking the first — and a guard that disagrees with Hermes is worse than none.
    versions = [
        line.split(":", 1)[1].split("#", 1)[0].strip().strip("\"'")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("version:")
    ]
    if len(versions) != 1:
        raise AssertionError(f"expected exactly one root version key in {path}, found {versions}")
    return versions[0]


def test_runtime_version_matches_project_metadata():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())

    assert ficelle.__version__ == metadata["project"]["version"]


@pytest.mark.parametrize("relative_path", PACKAGED_PLUGIN_MANIFESTS)
def test_packaged_hermes_plugin_version_matches_release(relative_path: str):
    assert _manifest_version(REPO_ROOT / relative_path) == ficelle.__version__


def test_manifest_version_reads_the_root_key_only(tmp_path):
    manifest = tmp_path / "plugin.yaml"

    manifest.write_text('name: ficelle\nversion: "9.9.9"  # release\nnested:\n  version: 0.0.1\n')
    assert _manifest_version(manifest) == "9.9.9"

    manifest.write_text("version: 9.9.9\nversion: 0.0.1\n")
    with pytest.raises(AssertionError):
        _manifest_version(manifest)

    manifest.write_text("name: ficelle\n")
    with pytest.raises(AssertionError):
        _manifest_version(manifest)


# The pin below is the one bump nothing else catches. `pyproject.toml`, `__init__.py` and the
# manifests are all read by tests or by the runtime, but a stale `ficelle-router==` only fails
# when pip actually resolves Core and Pro together — that is, in the buyer's terminal, as
# `ResolutionImpossible`, after they have paid. 0.1.7 shipped with it stale and was caught by a
# manual install, not by this suite.
#
# It reads the closed pack, and the rest of this file is core, so the mirror ships the file
# without `ficelle-pro/` to read. Skip there rather than fail: the monorepo is the only place
# the pin exists to be checked, and a red suite in the mirror is a red badge on the public repo.
PRO_PYPROJECT = REPO_ROOT / "ficelle-pro/pyproject.toml"


@pytest.mark.skipif(not PRO_PYPROJECT.exists(), reason="closed pack absent (core-only checkout)")
def test_pro_pins_the_core_version_it_ships_with() -> None:
    metadata = tomllib.loads(PRO_PYPROJECT.read_text(encoding="utf-8"))
    pins = [
        dep
        for dep in metadata["project"]["dependencies"]
        if dep.replace(" ", "").startswith("ficelle-router")
    ]
    assert len(pins) == 1, f"expected exactly one ficelle-router pin, found {pins}"
    assert pins[0].replace(" ", "") == f"ficelle-router=={ficelle.__version__}", (
        f"ficelle-pro pins {pins[0]!r} while Core is {ficelle.__version__} — "
        "every Core+Pro install would fail pip resolution"
    )


# The installer verifies the Core wheel against this digest, so the two constants that carry it
# must agree. They live in different scripts and 0.1.7 shipped them briefly out of step, which
# fails the install with an integrity error rather than anything readable.
def test_the_two_pinned_core_digests_agree() -> None:
    def _pinned(relative_path: str) -> tuple[str, str]:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        digests = re.findall(r'"([0-9a-f]{64})"', text)
        versions = re.findall(r'^(?:CORE_VERSION|VERSION) = "([^"]+)"', text, re.MULTILINE)
        assert len(digests) == 1, f"{relative_path}: expected one digest, found {len(digests)}"
        assert len(versions) == 1, f"{relative_path}: expected one version, found {len(versions)}"
        return versions[0], digests[0]

    bootstrap = _pinned("scripts/bootstrap-ficelle.py")
    canary = _pinned("scripts/verify-launch-cutover.py")
    assert bootstrap == canary, (
        f"bootstrap pins {bootstrap} but the launch canary pins {canary}"
    )
    assert bootstrap[0] == ficelle.__version__
