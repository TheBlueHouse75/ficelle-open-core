"""Ficelle compression retrieval tool for Hermes."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

DEFAULT_FICELLE_ROUTER_URL = "http://127.0.0.1:8646"
COMPRESSION_MARKER_RE = re.compile(r"^<<ficelle:compressed:([a-f0-9]{24})>>$")
COMPRESSION_HASH_RE = re.compile(r"^[a-f0-9]{24}$")

FICELLE_RETRIEVE_SCHEMA: dict[str, Any] = {
    "name": "ficelle_retrieve_compressed",
    "description": (
        "Retrieve the local original content for a Ficelle compression marker. "
        "Use this when the prompt contains <<ficelle:compressed:<hash>>> and "
        "the omitted original content is needed. The marker is not a file path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hash_or_marker": {
                "type": "string",
                "description": "Full Ficelle compression marker or bare 24-character lowercase hash.",
            },
            "query": {
                "type": "string",
                "description": "Optional substring filter to return only matching original lines.",
            },
        },
        "required": ["hash_or_marker"],
    },
}


def _compression_hash(hash_or_marker: str) -> str:
    value = str(hash_or_marker or "").strip()
    marker = COMPRESSION_MARKER_RE.match(value)
    if marker:
        return marker.group(1)
    if COMPRESSION_HASH_RE.match(value):
        return value
    raise ValueError("Provide a Ficelle compression marker or 24-character lowercase hash, not a file path.")


def _router_url_from_openai_url(value: Any) -> str:
    url = str(value or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/v1"):
        return url[:-3]
    return url


def _configured_router_url() -> str:
    env_url = _router_url_from_openai_url(os.environ.get("FICELLE_ROUTER_URL"))
    if env_url:
        return env_url
    try:
        from hermes_cli.config import load_config  # type: ignore[import-not-found]

        config = load_config()
    except Exception:
        return DEFAULT_FICELLE_ROUTER_URL
    providers = config.get("providers") if isinstance(config, dict) else {}
    ficelle = providers.get("ficelle") if isinstance(providers, dict) else {}
    if isinstance(ficelle, dict):
        for key in ("api", "base_url", "url"):
            config_url = _router_url_from_openai_url(ficelle.get(key))
            if config_url:
                return config_url
    return DEFAULT_FICELLE_ROUTER_URL


def ficelle_retrieve_compressed(hash_or_marker: str, query: str | None = None) -> dict[str, Any]:
    """Retrieve a Ficelle-compressed original by marker/hash from the local router."""
    try:
        digest = _compression_hash(hash_or_marker)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    payload = {"hash": digest}
    if query:
        payload["query"] = query
    request = urllib.request.Request(
        _configured_router_url() + "/admin/compression/retrieve",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=8.0) as response:
            result = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            error_payload = {"error": {"message": str(exc)}}
        message = error_payload.get("error", {}).get("message") if isinstance(error_payload, dict) else str(exc)
        return {"ok": False, "hash": digest, "error": message or "compressed original could not be retrieved"}
    except Exception as exc:
        return {"ok": False, "hash": digest, "error": f"Ficelle router is not reachable: {type(exc).__name__}"}
    if not isinstance(result, dict):
        return {"ok": False, "hash": digest, "error": "Ficelle returned an invalid retrieval response"}
    result["ok"] = True
    return result


def _handle_ficelle_retrieve(args: dict[str, Any], **_: Any) -> str:
    result = ficelle_retrieve_compressed(
        str(args.get("hash_or_marker", "")),
        query=str(args["query"]) if args.get("query") else None,
    )
    return json.dumps(result, ensure_ascii=False)


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="ficelle_retrieve_compressed",
        toolset="ficelle",
        schema=FICELLE_RETRIEVE_SCHEMA,
        handler=_handle_ficelle_retrieve,
        description="Retrieve a locally stored original for a Ficelle compression marker.",
        emoji="C",
    )
