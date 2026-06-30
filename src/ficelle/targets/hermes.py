from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ficelle.targets.base import (
    SmokeCheck,
    TargetExport,
    TargetExportContext,
    TargetInstallContext,
    TargetInstallResult,
    TargetKind,
    TargetSmokeContext,
)
from ficelle.use_cases.hermes_export import HermesExportBuilder


@dataclass(frozen=True)
class HermesTargetAdapter:
    export_builder: HermesExportBuilder
    target_id: str = "hermes"
    display_name: str = "Hermes"
    kind: TargetKind = "agent_host"
    default_base_url: str = "http://127.0.0.1:8646/v1"
    supports_plugin_install: bool = True
    supports_config_export: bool = True
    supports_health_check: bool = True

    def export_config(self, context: TargetExportContext) -> TargetExport:
        config = dict(context.config)
        hermes_export = self.export_builder.export(config)
        payload = hermes_export.payload
        return TargetExport(
            target_id=self.target_id,
            base_url=str(payload["base_url"]),
            models=tuple(hermes_export.picker_models),
            config_text=str(payload["yaml"]),
            presets=tuple(dict(preset) for preset in payload["presets"]),
            required_assets=("ficelle", "ficelle-compression"),
            verification_commands=(
                ("ficelle", "health"),
                ("ficelle", "models"),
                ("ficelle", "export"),
            ),
        )

    def install_assets(self, context: TargetInstallContext) -> TargetInstallResult:
        warnings = ("dry-run only; use ficelle-setup for Hermes asset installation",) if context.dry_run else ()
        return TargetInstallResult(
            target_id=self.target_id,
            installed_assets=("ficelle", "ficelle-compression"),
            warnings=warnings,
        )

    def smoke_checks(self, context: TargetSmokeContext) -> Sequence[SmokeCheck]:
        checks: list[SmokeCheck] = [
            SmokeCheck("service-health", ("ficelle", "health"), "Core service responds locally."),
            SmokeCheck("model-list", ("ficelle", "models"), "Core service exposes virtual models."),
            SmokeCheck("hermes-export", ("ficelle", "export"), "Hermes export renders without secrets."),
        ]
        if context.credentials_expected:
            checks.append(
                SmokeCheck(
                    "live-route",
                    ("curl", "-s", f"{context.base_url}/chat/completions"),
                    "Optional live route smoke when provider credentials are configured.",
                    requires_credentials=True,
                )
            )
        return tuple(checks)
