"""Target adapter contracts and registry."""

from ficelle.targets.base import (
    SmokeCheck,
    TargetAdapter,
    TargetExport,
    TargetExportContext,
    TargetInstallContext,
    TargetInstallResult,
    TargetKind,
    TargetSmokeContext,
)
from ficelle.targets.registry import get_target_adapter, registered_target_ids, target_export

__all__ = [
    "SmokeCheck",
    "TargetAdapter",
    "TargetExport",
    "TargetExportContext",
    "TargetInstallContext",
    "TargetInstallResult",
    "TargetKind",
    "TargetSmokeContext",
    "get_target_adapter",
    "registered_target_ids",
    "target_export",
]
