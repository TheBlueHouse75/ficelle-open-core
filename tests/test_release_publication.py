"""Guards on the path that uploads Core to a public index.

A PyPI upload cannot be undone: the filename is burned even after a delete, and the paid pack
is served only by the license service after an entitlement check. So the two things worth
testing are that the workflow cannot authenticate any way except trusted publishing, and that
the audit it runs actually refuses the artifacts it is supposed to refuse.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit-release-artifacts.py"
# The filename is part of PyPI's trusted-publisher configuration: PyPI mints a credential for
# this exact workflow file, so renaming it silently breaks every future release.
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"

SPEC = importlib.util.spec_from_file_location("audit_release_artifacts", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

VERSION = "0.1.7"
DESCRIPTION = "# Ficelle\n\nLocal strict-zero router.\n"


def _metadata(description: str = DESCRIPTION, version: str = VERSION) -> str:
    return (
        "Metadata-Version: 2.1\n"
        "Name: ficelle-router\n"
        f"Version: {version}\n"
        "Description-Content-Type: text/markdown\n"
        f"\n{description}"
    )


def _wheel(directory: Path, description: str = DESCRIPTION, version: str = VERSION) -> Path:
    path = directory / f"ficelle_router-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ficelle/__init__.py", f'__version__ = "{version}"\n')
        archive.writestr(
            f"ficelle_router-{version}.dist-info/METADATA", _metadata(description, version)
        )
    return path


def _sdist(
    directory: Path,
    description: str = DESCRIPTION,
    extra: str | None = None,
    version: str = VERSION,
) -> Path:
    path = directory / f"ficelle_router-{version}.tar.gz"
    root = f"ficelle_router-{version}"
    with tarfile.open(path, "w:gz") as archive:
        for name, text in [
            (f"{root}/PKG-INFO", _metadata(description, version)),
            (f"{root}/pyproject.toml", "[project]\n"),
            (f"{root}/src/ficelle/__init__.py", f'__version__ = "{version}"\n'),
        ] + ([(f"{root}/{extra}", "private\n")] if extra else []):
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _bootstrap(directory: Path, wheel: Path, version: str = VERSION) -> Path:
    """Write a stand-in installer pinning the digest of ``wheel``."""
    path = directory / "bootstrap.py"
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    path.write_text(f'CORE_VERSION = "{version}"\nDEFAULT_CORE_SHA256 = (\n    "{digest}"\n)\n')
    return path


def _run(dist: Path, bootstrap: Path) -> int:
    return audit.main([str(dist), "--bootstrap", str(bootstrap)])


@pytest.fixture()
def dist(tmp_path: Path) -> Path:
    directory = tmp_path / "dist"
    directory.mkdir()
    return directory


def test_a_matching_pair_is_cleared(dist: Path, tmp_path: Path) -> None:
    wheel = _wheel(dist)
    _sdist(dist)

    assert _run(dist, _bootstrap(tmp_path, wheel)) == 0


def test_the_paid_pack_is_refused(dist: Path, tmp_path: Path) -> None:
    wheel = _wheel(dist)
    _sdist(dist)
    bootstrap = _bootstrap(tmp_path, wheel)
    (dist / f"ficelle_pro-{VERSION}-py3-none-any.whl").write_bytes(wheel.read_bytes())

    assert _run(dist, bootstrap) == 1


def test_a_wheel_that_is_not_the_pinned_one_is_refused(dist: Path, tmp_path: Path) -> None:
    """A rebuild, another tree, or a swapped asset — all of them change the digest."""
    wheel = _wheel(dist)
    _sdist(dist)
    bootstrap = _bootstrap(tmp_path, wheel)
    wheel.write_bytes(wheel.read_bytes() + b"\0")

    assert _run(dist, bootstrap) == 1


def test_an_sdist_from_another_tree_is_refused(dist: Path, tmp_path: Path) -> None:
    """The monorepo's README is not the public one, and the long description carries it."""
    wheel = _wheel(dist)
    _sdist(dist, description="# Ficelle\n\nPrivate monorepo readme.\n")

    assert _run(dist, _bootstrap(tmp_path, wheel)) == 1


def test_a_private_directory_in_the_sdist_is_refused(dist: Path, tmp_path: Path) -> None:
    wheel = _wheel(dist)
    _sdist(dist, extra="docs/gtm-launch-plan.md")

    assert _run(dist, _bootstrap(tmp_path, wheel)) == 1


def test_the_legacy_tests_directory_is_tolerated_only_up_to_its_last_release(
    dist: Path, tmp_path: Path
) -> None:
    """0.1.7 and earlier were built from a mirror without MANIFEST.in, so setuptools added
    `tests/` on its own. From the next release the prune is in effect, and a tests directory
    means the sdist came from a tree that bypassed it — which is how the monorepo's Pro-coupled
    suite would ship. The tolerance has to expire rather than stand forever.
    """
    legacy = ".".join(str(part) for part in audit.LAST_RELEASE_WITH_TESTS_IN_SDIST)
    wheel = _wheel(dist, version=legacy)
    _sdist(dist, extra="tests/test_pro_licensing.py", version=legacy)
    assert _run(dist, _bootstrap(tmp_path, wheel, version=legacy)) == 0

    major, minor, patch = audit.LAST_RELEASE_WITH_TESTS_IN_SDIST
    later = f"{major}.{minor}.{patch + 1}"
    for stale in dist.iterdir():
        stale.unlink()
    wheel = _wheel(dist, version=later)
    _sdist(dist, extra="tests/test_pro_licensing.py", version=later)
    assert _run(dist, _bootstrap(tmp_path, wheel, version=later)) == 1


def test_an_unparsable_version_withholds_the_legacy_tolerance(dist: Path, tmp_path: Path) -> None:
    """Fail closed: a pre-release or a typo must not read as "older than 0.1.7"."""
    wheel = _wheel(dist, version="0.1.7rc1")
    _sdist(dist, extra="tests/test_pro_licensing.py", version="0.1.7rc1")

    assert _run(dist, _bootstrap(tmp_path, wheel, version="0.1.7rc1")) == 1


def test_a_missing_artifact_is_refused(dist: Path, tmp_path: Path) -> None:
    wheel = _wheel(dist)

    assert _run(dist, _bootstrap(tmp_path, wheel)) == 1


def test_the_workflow_authenticates_only_through_trusted_publishing() -> None:
    """No stored token, and the OIDC permission the exchange needs.

    An API token would be a long-lived secret that publishes under this project name from
    anywhere it leaks. Trusted publishing scopes the credential to this repository, this
    workflow file and a named environment, and it expires in minutes.
    """
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in text
    for forbidden in ("password:", "PYPI_API_TOKEN", "TWINE_PASSWORD", "secrets.PYPI"):
        assert forbidden not in text, f"{WORKFLOW.name} references {forbidden}"


def test_the_workflow_only_ever_runs_in_the_public_mirror() -> None:
    """The monorepo receives this file through the mirror assembler and must never publish."""
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.repository == 'TheBlueHouse75/ficelle-open-core'" in text


def test_the_workflow_audits_before_it_publishes() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "scripts/audit-release-artifacts.py" in text
    # Every publish job reads the artifact the audit job produced, so none can reach an index
    # with anything the audit did not clear. Counting both sides rather than hardcoding a number
    # keeps the invariant true if an index is ever added.
    assert text.count("pypa/gh-action-pypi-publish") == text.count("needs: audit") > 0
