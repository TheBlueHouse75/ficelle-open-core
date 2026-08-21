from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ficelle.url_security import connectable_http_url


SafeInt = Callable[[Any, int], int]
FusionVisibleInModelList = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class HermesExport:
    payload: dict[str, Any]
    picker_models: Sequence[str]


@dataclass(frozen=True)
class HermesExportBuilder:
    picker_virtual_models: Sequence[str]
    fusion_model_id: str
    fusion_visible_in_model_list: FusionVisibleInModelList
    safe_int: SafeInt

    def picker_models(self, config: dict[str, Any]) -> list[str]:
        return hermes_picker_models(
            config,
            picker_virtual_models=self.picker_virtual_models,
            fusion_model_id=self.fusion_model_id,
            fusion_visible_in_model_list=self.fusion_visible_in_model_list,
        )

    def export(self, config: dict[str, Any]) -> HermesExport:
        picker_models = self.picker_models(config)
        payload = build_hermes_config_export(
            config,
            picker_models=picker_models,
            safe_int=self.safe_int,
        )
        return HermesExport(payload=payload, picker_models=tuple(picker_models))

    def build(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.export(config).payload


def hermes_picker_models(
    config: dict[str, Any],
    *,
    picker_virtual_models: Sequence[str],
    fusion_model_id: str,
    fusion_visible_in_model_list: FusionVisibleInModelList,
) -> list[str]:
    picker_models = list(picker_virtual_models)
    if fusion_visible_in_model_list(config):
        picker_models.append(fusion_model_id)
    return picker_models


def build_hermes_config_export(
    config: dict[str, Any],
    *,
    safe_int: SafeInt,
    picker_models: Sequence[str],
) -> dict[str, Any]:
    base_url = connectable_http_url(
        config.get("host"),
        safe_int(config.get("port"), 8646),
        "/v1",
    )
    provider_yaml = "\n".join([
        "# Ficelle is installed as a Hermes user provider plugin under ~/.hermes/plugins/model-providers/ficelle.",
        "# Paste the desired slots into ~/.hermes/config.yaml and restart the gateway for Discord/Telegram sessions.",
        f"# Local endpoint: {base_url}",
        "",
        "plugins:",
        "  enabled:",
        "    - \"ficelle-compression\"",
        "",
        "toolsets:",
        "  - \"ficelle\"",
        "",
        "providers:",
        "  ficelle:",
        "    name: \"Ficelle FREE\"",
        f"    api: \"{base_url}\"",
        "    transport: openai_chat",
        "    models:",
        *[f"      - \"{model_id}\"" for model_id in picker_models],
        "",
        "# Ficelle is the main provider; auxiliary slots reuse its specialized profiles.",
        "# Long context is for huge prompts; it is not the default Hermes compression choice.",
        "auxiliary:",
        "  title_generation:",
        "    provider: \"ficelle\"",
        "    model: \"ficelle/auto-fast\"",
        "  compression:",
        "    provider: \"ficelle\"",
        "    model: \"ficelle/auto-compression\"",
        "  web_extract:",
        "    provider: \"ficelle\"",
        "    model: \"ficelle/auto-json\"",
        "",
        "# Ficelle also handles fallback through its tool-capable profile.",
        "fallback_providers:",
        "  - provider: \"ficelle\"",
        "    model: \"ficelle/auto-tools\"",
        "",
        "# Ficelle is the active main model route. Hermes 0.17 treats slash-containing model IDs",
        "# as aggregator slugs in doctor, so use its local custom transport until that validator",
        "# recognizes provider-plugin model IDs.",
        "model:",
        "  provider: \"custom\"",
        f"  base_url: \"{base_url}\"",
        "  key_env: \"FICELLE_API_KEY\"",
        "  model: \"ficelle/auto-orchestrator\"",
    ])
    presets = [
        {
            "model": "ficelle/auto-orchestrator",
            "benefit": "main Hermes orchestrator with strict-zero routing",
            "hermes_slot": "model.provider/model.model",
        },
        {
            "model": "ficelle/auto-fast",
            "benefit": "cheap quick replies for titles and small drafts",
            "hermes_slot": "auxiliary.title_generation",
        },
        {
            "model": "ficelle/auto-json",
            "benefit": "structured extraction when exact JSON matters",
            "hermes_slot": "auxiliary.web_extract or custom scripts",
        },
        {
            "model": "ficelle/auto-compression",
            "benefit": "bounded summarization for Hermes compaction",
            "hermes_slot": "auxiliary.compression",
        },
        {
            "model": "ficelle/auto-long",
            "benefit": "very large prompts when context window is the constraint",
            "hermes_slot": "long-document experiments",
        },
        {
            "model": "ficelle/auto-tools",
            "benefit": "general tool-capable fallback",
            "hermes_slot": "fallback_providers",
        },
        {
            "model": "ficelle/auto-vision",
            "benefit": "free image understanding smoke-tested by canary",
            "hermes_slot": "vision experiments only",
        },
    ]
    return {"base_url": base_url, "yaml": provider_yaml, "presets": presets}
