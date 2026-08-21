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


DEFAULT_OPENCLAW_CONTEXT_WINDOW = 64000
DEFAULT_OPENCLAW_MAX_TOKENS = 8192


def openclaw_model_ref(provider_id: str, core_model_id: str) -> str:
    return f"{provider_id}/{core_model_id}"


def openclaw_model_alias(core_model_id: str) -> str:
    return "Ficelle " + core_model_id.removeprefix("ficelle/").replace("-", " ").title()


def openclaw_model_inputs(core_model_id: str) -> list[str]:
    if core_model_id == "ficelle/auto-vision":
        return ["text", "image"]
    if core_model_id == "ficelle/auto-video":
        return ["text", "video"]
    if core_model_id == "ficelle/auto-audio":
        return ["text", "audio"]
    return ["text"]


@dataclass(frozen=True)
class OpenClawTargetAdapter:
    virtual_models: Sequence[str]
    fusion_model_id: str | None = None
    fusion_visible_in_model_list: Callable[[Mapping[str, Any]], bool] | None = None
    model_policy_ids: Mapping[str, str] | None = None
    target_id: str = "openclaw"
    display_name: str = "OpenClaw"
    kind: TargetKind = "agent_host"
    default_base_url: str = "http://127.0.0.1:8646/v1"
    provider_id: str = "ficelle"
    supports_plugin_install: bool = False
    supports_config_export: bool = True
    supports_health_check: bool = True

    def export_config(self, context: TargetExportContext) -> TargetExport:
        config = dict(context.config)
        base_url = target_base_url(config)
        model_ids = self._model_ids(config)
        primary = "ficelle/auto-orchestrator" if "ficelle/auto-orchestrator" in model_ids else model_ids[0]
        fallbacks = tuple(model_id for model_id in model_ids if model_id != primary)
        payload = self._openclaw_config(base_url, primary, fallbacks)
        return TargetExport(
            target_id=self.target_id,
            base_url=base_url,
            models=model_ids,
            config=payload,
            presets=(
                {
                    "name": "OpenClaw primary agent",
                    "model": openclaw_model_ref(self.provider_id, primary),
                    "core_model": primary,
                },
            ),
            warnings=(
                "Supported experimental target; re-run OpenClaw local model-runner smoke before release promotion.",
            ),
            verification_commands=(
                ("ficelle", "health"),
                ("ficelle", "models"),
                ("openclaw", "models", "list"),
            ),
        )

    def install_assets(self, context: TargetInstallContext) -> TargetInstallResult:
        warnings = (
            "No assets installed; merge the exported JSON into the OpenClaw config manually during experimental validation.",
        )
        if context.dry_run:
            warnings = ("Dry run only. " + warnings[0],)
        return TargetInstallResult(target_id=self.target_id, warnings=warnings)

    def smoke_checks(self, context: TargetSmokeContext) -> Sequence[SmokeCheck]:
        checks: list[SmokeCheck] = [
            SmokeCheck("service-health", ("ficelle", "health"), "Core service responds locally."),
            SmokeCheck("model-list", ("ficelle", "models"), "Core service exposes virtual models."),
            SmokeCheck("openclaw-model-list", ("openclaw", "models", "list"), "OpenClaw can list configured models."),
        ]
        if context.credentials_expected:
            checks.append(
                SmokeCheck(
                    "openclaw-agent-smoke",
                    (
                        "openclaw",
                        "infer",
                        "model",
                        "run",
                        "--local",
                        "--model",
                        openclaw_model_ref(self.provider_id, "ficelle/auto-fast"),
                        "--prompt",
                        "Reply exactly: ficelle-openclaw-smoke-ok",
                        "--json",
                    ),
                    "Optional live target smoke once OpenClaw is installed and configured.",
                    requires_credentials=True,
                )
            )
        return tuple(checks)

    def _model_ids(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        return visible_model_ids(
            self.virtual_models, self.fusion_model_id, self.fusion_visible_in_model_list, config
        )

    def _openclaw_config(self, base_url: str, primary: str, fallbacks: Sequence[str]) -> dict[str, Any]:
        ordered_model_ids = (primary, *fallbacks)
        policy_ids = self.model_policy_ids or {}
        allowlist = {
            openclaw_model_ref(self.provider_id, model_id): {
                "alias": openclaw_model_alias(model_id),
                "params": {"ficelleCoreModel": model_id},
            }
            for model_id in ordered_model_ids
        }
        provider_models = [
            {
                "id": model_id,
                "name": openclaw_model_alias(model_id),
                "reasoning": policy_ids.get(model_id, model_id) == "ficelle/auto-reasoning",
                "input": openclaw_model_inputs(policy_ids.get(model_id, model_id)),
                "contextWindow": DEFAULT_OPENCLAW_CONTEXT_WINDOW,
                "maxTokens": DEFAULT_OPENCLAW_MAX_TOKENS,
            }
            for model_id in ordered_model_ids
        ]
        return {
            "agents": {
                "defaults": {
                    "models": allowlist,
                },
            },
            "models": {
                "providers": {
                    self.provider_id: {
                        "baseUrl": base_url,
                        "apiKey": "ficelle-local",
                        "api": "openai-completions",
                        "timeoutSeconds": 300,
                        "models": provider_models,
                    },
                },
            },
        }
