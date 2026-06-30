"""Guard: the open core depends one-directionally on the closed pack.

open-core-extraction-prd.md R6 / AC#4: the core may reference ``ficelle_pro`` only
through the guarded optional-import seam (``try: import ficelle_pro … except
ImportError``), never as a hard dependency. An unguarded import would make the
core-only wheel fail at module load on a public/free install where the pack is absent.

This test parses every core module's AST and fails if a ``ficelle_pro`` import sits
outside the body of a ``try`` whose ``except`` catches ``ImportError`` (or broader).
"""
from __future__ import annotations

import ast
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "ficelle"

# An except clause guards the optional import when it catches one of these.
_GUARDING_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _references_ficelle_pro(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "ficelle_pro"
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "ficelle_pro" for alias in node.names)
    return False


def _guards_import_error(handlers: list[ast.ExceptHandler]) -> bool:
    for handler in handlers:
        if handler.type is None:  # bare ``except:``
            return True
        candidates = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        if any(isinstance(c, ast.Name) and c.id in _GUARDING_EXCEPTIONS for c in candidates):
            return True
    return False


def test_core_imports_ficelle_pro_only_through_guarded_optional_imports() -> None:
    offenders: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Imports inside the BODY of a try-guarded-by-ImportError are the allowed seam.
        # (Imports in the ``except`` fallback are not the pack — they import the core stub.)
        guarded: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and _guards_import_error(node.handlers):
                for stmt in node.body:
                    for child in ast.walk(stmt):
                        if _references_ficelle_pro(child):
                            guarded.add(id(child))
        for node in ast.walk(tree):
            if _references_ficelle_pro(node) and id(node) not in guarded:
                rel = path.relative_to(CORE_ROOT.parents[1])
                offenders.append(f"{rel}:{getattr(node, 'lineno', 0)}")
    assert not offenders, (
        "unguarded ficelle_pro import in the open core — the core must import the closed "
        "pack only through a try/except ImportError seam (R6/AC#4):\n" + "\n".join(offenders)
    )
