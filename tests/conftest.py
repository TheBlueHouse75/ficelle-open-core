"""Shared pytest fixtures and suite-wide isolation for Ficelle.

The single concern handled here today is OS secret-store isolation: no test may
shell out to the real macOS Keychain (``security``) or Linux libsecret
(``secret-tool``). The Windows Credential Manager is reached through the ctypes
``Cred*W`` API rather than a CLI, so it is outside this guard's reach — harmless
because the suite runs on macOS/Linux. See ``block_real_secret_store`` for the rest.
"""
from __future__ import annotations

import importlib.util
import subprocess

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
