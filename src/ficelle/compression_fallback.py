"""Core-only fallbacks for the closed native-compression engine.

The native first-party compression engine (``ficelle.compression``) is a closed
Pro module (see ``docs/prds/open-core-extraction-prd.md`` and
``docs/open-core-split.md``). When it is absent from a core-only install, the
open engine must still import and run with native compression **off**.

This module supplies the *only* compression symbols the core needs at module
load when the real engine is missing:

- ``DEFAULT_COMPRESSION_CONFIG`` — the same default config the real engine
  exposes, with ``mode == "off"``. It feeds ``DEFAULT_CONFIG["compression"]`` in
  ``router.py`` and the savings/TTL defaults in ``use_cases/chat_completion.py``.
- ``normalize_compression_settings`` — a no-op normalizer that always resolves to
  the "off" default. Free-tier callers (the chat planner, the settings policy,
  the observability surface) read ``mode`` from the result and short-circuit, so
  returning the default config preserves the free-tier "compression off" behavior.

The compression *primitives* (``compress_block``, ``put_original``,
``plan_chat_compression``, ``compression_marker``, ``get_original``, ``stats``,
``clear``) are deliberately **not** mirrored here: the free-tier code paths guard
on ``mode == "off"`` before reaching them, so the core binds them to ``None``.

When ``ficelle.compression`` IS present, this module is never imported and
behavior is byte-for-byte unchanged.
"""
from __future__ import annotations

import copy
from typing import Any

# Mirrors ficelle.compression.DEFAULT_COMPRESSION_CONFIG (mode "off" by default).
DEFAULT_COMPRESSION_CONFIG: dict[str, Any] = {
    "mode": "off",
    "min_chars": 2000,
    "min_savings_ratio": 0.15,
    "max_compressed_blocks": 4,
    "store_ttl_seconds": 900,
    "store_max_entries": 1000,
    "allow_streaming": False,
    "strategies": {
        "json_array": True,
        "log": True,
        "diff": True,
        "search_results": True,
        "text_excerpt": False,
        "code": False,
    },
}


def normalize_compression_settings(raw_config: Any, *, strict: bool = False) -> dict[str, Any]:
    """No-op normalizer used when the native compression engine is absent.

    The native engine is the only thing that can turn compression on, so without
    it the resolved configuration is always the "off" default. ``raw_config`` and
    ``strict`` are accepted to match the real signature but ignored.
    """
    return copy.deepcopy(DEFAULT_COMPRESSION_CONFIG)
