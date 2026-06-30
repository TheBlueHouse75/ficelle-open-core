from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ficelle.targets.base import (
    SmokeCheck,
    TargetExport,
    TargetExportContext,
    TargetInstallContext,
    TargetInstallResult,
    TargetKind,
    TargetSmokeContext,
    target_base_url,
)


ONLINE_CONTROL_PLANE_TARGET_VERSION = "control-plane-v1"


def safe_online_status_payload(
    *,
    status: Mapping[str, Any],
    config_fingerprint: str,
    config_structural_fingerprint: str,
    target_version: str,
) -> dict[str, Any]:
    catalog = status.get("catalog") if isinstance(status.get("catalog"), Mapping) else {}
    runtime = status.get("runtime") if isinstance(status.get("runtime"), Mapping) else {}
    profiles = status.get("profiles") if isinstance(status.get("profiles"), Mapping) else {}
    compression = status.get("compression") if isinstance(status.get("compression"), Mapping) else {}
    counts = {
        "profiles": len(profiles),
        "catalog_models": _safe_int(catalog.get("model_count")),
        "catalog_available": _safe_int(catalog.get("available_count")),
        "active_cooldowns": _safe_int(runtime.get("active_cooldowns_count")),
        "provider_error_count": _safe_int(runtime.get("provider_error_count")),
        "last_route_count": _safe_int(runtime.get("last_route_count")),
        "compression_errors": _safe_int(compression.get("error_count")),
    }
    return {
        "status": str(status.get("status") or "unknown"),
        "generated_at": str(status.get("generated_at") or ""),
        "target_version": target_version,
        "config_fingerprint": config_fingerprint,
        "config_structural_fingerprint": config_structural_fingerprint,
        "counts": counts,
    }


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


@dataclass(frozen=True)
class OnlineControlPlaneTargetAdapter:
    target_id: str = "online"
    display_name: str = "Online control plane"
    kind: TargetKind = "control_plane"
    default_base_url: str = "http://127.0.0.1:8646/v1"
    supports_plugin_install: bool = False
    supports_config_export: bool = True
    supports_health_check: bool = True
    target_version: str = ONLINE_CONTROL_PLANE_TARGET_VERSION

    def export_config(self, context: TargetExportContext) -> TargetExport:
        base_url = target_base_url(context.config)
        return TargetExport(
            target_id=self.target_id,
            base_url=base_url,
            models=(),
            config={
                "mode": "distribution_control_plane_reporting",
                "local_service": {
                    "base_url": base_url,
                    "health": "/health",
                    "status": "/admin/online/status.json",
                },
                "reporting": {
                    "payload": "safe_online_status_payload",
                    "upload": "opt_in",
                },
                "blocked": {
                    "hosted_inference_proxy": True,
                    "cloud_secret_custody": True,
                    "remote_v1_chat_completions": True,
                },
                "target_version": self.target_version,
            },
            warnings=(
                "Online v1 is distribution/control-plane/reporting only; it does not proxy inference.",
                "Do not upload provider credentials, prompts, completions, route rows, or raw runtime state.",
            ),
            verification_commands=(
                ("ficelle", "health"),
                ("ficelle", "models"),
            ),
        )

    def install_assets(self, context: TargetInstallContext) -> TargetInstallResult:
        warning = "No assets installed; online control-plane packaging is blocked pending a separate cloud PRD."
        if context.dry_run:
            warning = "Dry run only. " + warning
        return TargetInstallResult(target_id=self.target_id, warnings=(warning,))

    def smoke_checks(self, context: TargetSmokeContext) -> Sequence[SmokeCheck]:
        return (
            SmokeCheck("service-health", ("ficelle", "health"), "Core service responds locally."),
            SmokeCheck("safe-status", ("ficelle", "doctor", "--json"), "Local status can be summarized safely."),
        )
