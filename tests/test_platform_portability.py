"""Guard: no module-scope import of a platform-only stdlib module.

An `import fcntl` at the top of `pro_install.py` did not fail a single test here — CI runs on
`ubuntu-latest` only, and the "Windows" tests in `tests/test_service.py` inject
`platform_name="win32"` without ever importing the module on a host that lacks `fcntl`. The
failure only surfaced on a real Windows box, as an `ImportError` that took `ficelle install-pro`
and `POST /admin/license/install` down with it.

Behavioural coverage is the real net (`lock_platform` in `tests/conftest.py` runs the lock tests
with `fcntl` absent); this is the cheap one that catches the next occurrence at the import line,
on any platform, before anybody has to plug in a Windows host to find out.

Sibling guard, same corpus, different invariant: `tests/test_open_core_import_direction.py`
checks that `ficelle_pro` imports sit behind a `try`/`except ImportError` seam.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Stdlib modules that exist on only one family of platforms. Importing one unconditionally at
# module scope makes the whole module unimportable everywhere else.
PLATFORM_ONLY_MODULES = frozenset(
    {
        # POSIX-only
        "fcntl",
        "grp",
        "posix",
        "pty",
        "pwd",
        "resource",
        "syslog",
        "termios",
        "tty",
        # Windows-only
        "_winapi",
        "msvcrt",
        "nt",
        "winreg",
        "winsound",
    }
)


def scanned_files() -> list[Path]:
    """Everything shipped or run outside a test: the package, the entry point, and the scripts.

    `scripts/` earns its place — the installers are the code most likely to reach for `winreg`,
    and they run on the user's machine before any of the package's own guards apply.

    Filtered to what the checkout actually holds, because this test ships to the public mirror
    and the mirror carries a subset: `ficelle_router.py` is in neither its allowlisted dirs nor
    its allowlisted files, and only four of `scripts/` are mirrored. Scanning a fixed list would
    fail the mirror's CI on a missing file rather than on a real finding.
    """
    candidates = [
        *(REPO_ROOT / "src" / "ficelle").rglob("*.py"),
        REPO_ROOT / "ficelle_router.py",
        *(REPO_ROOT / "scripts").glob("*.py"),
    ]
    return sorted(path for path in candidates if path.is_file())


def unguarded_platform_imports(source: str) -> list[tuple[str, int]]:
    """Return (module, line) for each platform-only import reached on plain module execution.

    Only top-level statements count. An import nested in a function or class is lazy, and one
    nested in a `try` (the `except ImportError` probe) or an `if sys.platform ...` is the portable
    form — none of them run unconditionally at import time, which is the only thing that breaks.
    """
    return [
        (name, statement.lineno)
        for statement in ast.parse(source).body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
        for name in imported_names(statement)
        if name in PLATFORM_ONLY_MODULES
    ]


def imported_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.level:  # a relative import can never name a stdlib module
        return []
    return [(node.module or "").split(".")[0]]


def test_no_shipped_module_imports_a_platform_only_stdlib_module() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line} ({name})"
        for path in scanned_files()
        for name, line in unguarded_platform_imports(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "platform-only stdlib import at module scope — this makes the whole module unimportable "
        "on every other platform. Move it inside the function that needs it, guard it with "
        "try/ImportError, or use ficelle.probe_lock.file_lock for locking:\n"
        + "\n".join(offenders)
    )


def test_the_guard_catches_the_import_it_was_written_for() -> None:
    # The bug verbatim: `import fcntl` next to the other stdlib imports at the top of a module.
    assert unguarded_platform_imports("import hashlib\nimport fcntl\nimport os\n") == [("fcntl", 2)]
    assert unguarded_platform_imports("from fcntl import flock\n") == [("fcntl", 1)]

    # And leaves the three portable shapes alone.
    assert unguarded_platform_imports("try:\n    import fcntl\nexcept ImportError:\n    fcntl = None\n") == []
    assert unguarded_platform_imports("import sys\nif sys.platform == 'win32':\n    import msvcrt\n") == []
    assert unguarded_platform_imports("def lock():\n    import fcntl\n") == []
