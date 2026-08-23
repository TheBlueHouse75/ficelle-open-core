"""Ficelle-owned registry for optional client connectors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ficelle.json_store import write_private_text


REGISTRY_FILENAME = "connectors.json"
SUPPORTED_CONNECTORS = ("hermes", "openclaw")


def registry_path(ficelle_home: Path) -> Path:
    return ficelle_home / REGISTRY_FILENAME


def load_connectors(ficelle_home: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(registry_path(ficelle_home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    connectors = payload.get("connectors") if isinstance(payload, dict) else None
    if not isinstance(connectors, dict):
        return {}
    return {
        connector_id: dict(record)
        for connector_id, record in connectors.items()
        if connector_id in SUPPORTED_CONNECTORS and isinstance(record, dict)
    }


def save_connectors(ficelle_home: Path, connectors: dict[str, dict[str, Any]]) -> None:
    payload = {"schema_version": 1, "connectors": connectors}
    write_private_text(
        registry_path(ficelle_home),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def register_connector(
    ficelle_home: Path,
    connector_id: str,
    *,
    client_home: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if connector_id not in SUPPORTED_CONNECTORS:
        raise ValueError(f"unsupported connector: {connector_id}")
    connectors = load_connectors(ficelle_home)
    existing_record = connectors.get(connector_id)
    record = dict(existing_record or {})
    record["installed"] = True
    if client_home is not None:
        record["client_home"] = str(client_home)
    if metadata:
        existing_metadata = record.get("metadata")
        merged_metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
        merged_metadata.update(metadata)
        record["metadata"] = merged_metadata
    if record == existing_record:
        return
    connectors[connector_id] = record
    save_connectors(ficelle_home, connectors)


def unregister_connector(ficelle_home: Path, connector_id: str) -> bool:
    connectors = load_connectors(ficelle_home)
    removed = connectors.pop(connector_id, None) is not None
    if removed:
        save_connectors(ficelle_home, connectors)
    return removed
