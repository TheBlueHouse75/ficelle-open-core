from __future__ import annotations

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
