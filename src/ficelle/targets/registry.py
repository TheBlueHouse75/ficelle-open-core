from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ficelle.targets.base import TargetAdapter, TargetExport, TargetExportContext


def registered_target_ids(registry: Mapping[str, TargetAdapter]) -> tuple[str, ...]:
    return tuple(sorted(registry))


def get_target_adapter(registry: Mapping[str, TargetAdapter], target_id: str) -> TargetAdapter | None:
    return registry.get(target_id.strip().lower())


def target_export(registry: Mapping[str, TargetAdapter], target_id: str, config: Mapping[str, Any]) -> TargetExport | None:
    adapter = get_target_adapter(registry, target_id)
    if adapter is None:
        return None
    return adapter.export_config(TargetExportContext(config=config))
