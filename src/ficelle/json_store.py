from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # Write to a *unique* temp file in the same directory, then atomically rename.
    # A fixed `<name>.tmp` path would let concurrent writers of the same file (e.g.
    # the startup catalog warm thread and a request-triggered refresh) clobber each
    # other's temp file and promote a truncated/interleaved JSON, or fail the second
    # rename. A per-write temp name makes concurrent writes safe (last writer wins).
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
