"""Shared pytest fixtures and suite-wide isolation for Ficelle.

Two concerns live here:

- OS secret-store isolation: no test may shell out to the real macOS Keychain
  (``security``) or Linux libsecret (``secret-tool``). The Windows Credential Manager is
  reached through the ctypes ``Cred*W`` API rather than a CLI, so it is outside this
  guard's reach — harmless because the suite runs on macOS/Linux. See
  ``block_real_secret_store``.
- Running lock tests on both platform branches: CI is ``ubuntu-latest`` only, so the
  ``msvcrt`` half of ``ficelle.probe_lock.file_lock`` would otherwise never execute. See
  ``lock_platform``.
"""
from __future__ import annotations

import errno
import importlib.util
import os
import subprocess
import threading

import pytest

# Tests that IMPORT the closed ``ficelle_pro`` pack at module load would crash collection
# where it is absent (a core-only checkout). Skip just those so a plain ``pytest`` does
# not error out. Behaviour tests that merely assume the full pack (provider adapters,
# cooldowns, doctor, ...) are a separate concern handled by the public-mirror curation,
# not here — this list is only the import-coupled modules.
collect_ignore: list[str] = []
if importlib.util.find_spec("ficelle_pro") is None:
    collect_ignore = [
        "test_router.py",
        "test_compression.py",
        "test_fusion_use_case.py",
        "test_router_settings_use_case.py",
        "test_prepare_public_mirror.py",  # imports the mirror assembler, which imports the pack
    ]

# Command-line front-ends for the host's secret store. Credential resolution reads
# through these — one ``security``/``secret-tool`` subprocess per provider/service it
# probes. Tests must never reach the real store: doing so (a) reads the developer's
# actual credentials as a side effect, and (b) spawns a subprocess per lookup whose
# latency, under system load, pushed admin key-write requests past the test HTTP
# client's 5s timeout (it is the auth re-resolution those requests trigger that shells
# out, not the write itself) — the root cause of the flaky tests (2026-06-21).
# Blocking them at the subprocess boundary fixes both without changing what the store
# factories return, so backend-selection tests still see the real
# ``KeychainStore``/``SecretToolStore``/``WindowsCredentialStore``.
_SECRET_STORE_CLIS = frozenset({"security", "secret-tool", "cmdkey"})


def _is_secret_store_cli(args: object) -> bool:
    # router/install always invoke these CLIs as an argv list; the basename strip keeps
    # the guard robust if a call ever uses an absolute path (e.g. /usr/bin/security).
    if not isinstance(args, (list, tuple)) or not args:
        return False
    return str(args[0]).rsplit("/", 1)[-1] in _SECRET_STORE_CLIS


@pytest.fixture(autouse=True)
def block_real_secret_store(monkeypatch):
    """Neutralize the OS secret-store CLIs for the whole suite.

    ``security``/``secret-tool``/``cmdkey`` invocations are answered with a synthetic
    "absent" result (exit code 1, empty output), so credential resolution falls back
    to env/.env exactly as it would on a host without a usable store. Every other
    subprocess call is delegated to the real ``subprocess.run`` untouched.

    The patch targets the global ``subprocess`` module, so it survives the
    ``importlib.reload`` that ``load_router`` performs. A test that needs to *simulate*
    a populated store keeps overriding ``subprocess.run`` (or the store factory)
    itself; that override is applied after this fixture and therefore wins.
    """
    real_run = subprocess.run

    def guarded_run(args, *call_args, **call_kwargs):
        if _is_secret_store_cli(args):
            return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="")
        return real_run(args, *call_args, **call_kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


class FakeMsvcrt:
    """The three ``msvcrt`` symbols ``probe_lock`` uses, with Windows' ownership rule.

    A byte-range lock on Windows belongs to the file *handle*, so a second descriptor on the
    same file conflicts even inside one process — the same exclusion ``flock`` gives us. Keying
    the held set by (device, inode) reproduces that on a POSIX test host.
    """

    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.held: set[tuple[int, int]] = set()
        # The real thing arbitrates in the kernel; tests contend from threads, so the
        # check-then-take below has to be atomic or a waiter could squeeze in beside a holder.
        self._guard = threading.Lock()

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        stat = os.fstat(fd)
        key = (stat.st_dev, stat.st_ino)
        with self._guard:
            if mode == self.LK_UNLCK:
                self.held.discard(key)
                return
            if key in self.held:
                raise OSError(errno.EACCES, "Permission denied")
            self.held.add(key)


@pytest.fixture(params=["posix", "windows"])
def lock_platform(request, monkeypatch):
    """Run a lock test twice: once natively, once with `fcntl` absent as on Windows.

    A module-scope `import fcntl` is POSIX-only, so it makes the whole importing module
    unimportable on Windows — how `ficelle install-pro` and `POST /admin/license/install` came
    to be broken there while the rest of the service ran fine. CI is `ubuntu-latest` only and
    the "Windows" tests in `test_service.py` inject `platform_name="win32"` without ever
    reaching the I/O layer, so nothing caught it.

    Substituting `probe_lock`'s own module-level names keeps the simulation inside the module
    under test; patching `fcntl` or `os` process-wide would also change what pytest and
    `tempfile` observe. Returns the fake on the Windows leg and None on the POSIX one, so a
    test can assert on the calls when it cares and ignore it otherwise.
    """
    if request.param == "posix":
        return None

    from ficelle import probe_lock

    fake = FakeMsvcrt()
    monkeypatch.setattr(probe_lock, "fcntl", None)
    monkeypatch.setattr(probe_lock, "msvcrt", fake, raising=False)
    return fake


def unreleasable_version(current: str) -> str:
    """A version string that cannot be a real release of ``current``'s line.

    Fixtures that need "the *other* version" — an incompatible Pro wheel, a mismatched pin —
    used to spell one out. A literal works right up until it ships: two Pro compatibility
    guards stood on `0.2.0`, chosen because it differed from the version of the day, and
    cutting 0.2.0 made both "incompatible" wheels compatible. The guards stopped raising and
    the tests went green while asserting nothing.

    Bumping the major cannot collide with any later release of the current line, and deriving
    it means the caller passes whichever version it is actually guarding (`bootstrap`'s pin and
    the installed Core version are separate sources of truth that a release can disagree on).
    """
    major, _, _ = current.partition(".")
    return f"{int(major) + 1}.0.0"
