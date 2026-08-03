from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    visible_model_ids,
)


@dataclass(frozen=True)
class GenericClientTargetAdapter:
    virtual_models: Sequence[str]
    fusion_model_id: str | None = None
    fusion_visible_in_model_list: Callable[[Mapping[str, Any]], bool] | None = None
    target_id: str = "generic"
    display_name: str = "Generic OpenAI-compatible client"
    kind: TargetKind = "generic_client"
    default_base_url: str = "http://127.0.0.1:8646/v1"
    supports_plugin_install: bool = False
    supports_config_export: bool = True
    supports_health_check: bool = True

    def export_config(self, context: TargetExportContext) -> TargetExport:
        config = dict(context.config)
        base_url = target_base_url(config)
        # Both forms, because clients disagree on what an "endpoint" field means: some append
        # the route to a base, others send the configured value verbatim. Exporting only the
        # base leaves the second kind with nothing to paste, so the user hand-builds a URL and
        # lands on `POST /v1` — a 404 that reads as a router failure.
        chat_completions_url = f"{base_url}/chat/completions"
        models = self._model_ids(config)
        primary = "ficelle/auto-fast" if "ficelle/auto-fast" in models else models[0]
        return TargetExport(
            target_id=self.target_id,
            base_url=base_url,
            models=models,
            config={
                "base_url": base_url,
                "chat_completions_url": chat_completions_url,
                "api_key": "ficelle-local",
                "models": list(models),
            },
            presets=(
                {
                    "name": "Generic chat client",
                    "base_url": base_url,
                    "chat_completions_url": chat_completions_url,
                    "model": primary,
                    "api_key": "ficelle-local",
                },
            ),
            warnings=(
                "Use the placeholder api_key only for clients that require a non-empty key for local loopback calls.",
                "Paste base_url into clients that append the route themselves, chat_completions_url into clients whose endpoint field wants the full chat URL.",
            ),
            verification_commands=(
                ("ficelle", "health"),
                ("ficelle", "models"),
            ),
        )

    def install_assets(self, context: TargetInstallContext) -> TargetInstallResult:
        warning = "No assets installed; copy the exported base URL and model id into the target client."
        if context.dry_run:
            warning = "Dry run only. " + warning
        return TargetInstallResult(target_id=self.target_id, warnings=(warning,))

    def smoke_checks(self, context: TargetSmokeContext) -> Sequence[SmokeCheck]:
        return (
            SmokeCheck("service-health", ("ficelle", "health"), "Core service responds locally."),
            SmokeCheck("model-list", ("ficelle", "models"), "Core service exposes virtual models."),
        )

    def _model_ids(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        return visible_model_ids(
            self.virtual_models, self.fusion_model_id, self.fusion_visible_in_model_list, config
        )
