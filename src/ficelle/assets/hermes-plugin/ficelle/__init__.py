"""Ficelle FREE local router provider profile for Hermes."""
from __future__ import annotations

import json
import urllib.request

from providers import register_provider  # type: ignore[import-not-found]
from providers.base import ProviderProfile  # type: ignore[import-not-found]


VIRTUAL_MODELS = (
    "ficelle/auto-orchestrator",
    "ficelle/auto-tools",
    "ficelle/auto-json",
    "ficelle/auto-compression",
    "ficelle/auto-long",
    "ficelle/auto-fast",
    "ficelle/auto-reasoning",
    "ficelle/auto-multimodal",
    "ficelle/auto-vision",
    "ficelle/auto-video",
    "ficelle/auto-audio",
    "ficelle/auto-coding",
)
FICELLE_ROUTER_URL = "http://127.0.0.1:8646"

# Fusion is an optional virtual profile: the gateway only advertises
# ficelle/auto-fusion in /v1/models when fusion is enabled and exposed. Mirror
# that by letting it through the live filter while keeping it out of the offline
# fallback, so we never advertise it when the gateway is unreachable.
FUSION_MODEL = "ficelle/auto-fusion"
LISTABLE_MODELS = (*VIRTUAL_MODELS, FUSION_MODEL)
CUSTOM_MODEL_PREFIX = "ficelle/custom/"
WORKLOAD_BY_MODEL = {
    "ficelle/auto-fast": "aux:title_generation",
    "ficelle/auto-json": "aux:web_extract",
    "ficelle/auto-compression": "aux:compression",
    "ficelle/auto-vision": "aux:vision",
}


class FicelleProfile(ProviderProfile):
    def build_api_kwargs_extras(
        self,
        *,
        model: str | None = None,
        reasoning_config: dict[str, object] | None = None,
        supports_reasoning: bool = False,
        **context: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        extra_body: dict[str, object] = {}
        if supports_reasoning and reasoning_config:
            if reasoning_config.get("enabled") is False:
                extra_body["reasoning"] = {"enabled": False}
            else:
                extra_body["reasoning"] = {
                    "enabled": True,
                    "effort": reasoning_config.get("effort") or "medium",
                }
        workload = WORKLOAD_BY_MODEL.get(str(model or ""))
        if not workload:
            return extra_body, {}
        return extra_body, {"extra_headers": {"X-Ficelle-Workload": workload}}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        try:
            endpoint = (base_url or self.base_url).rstrip("/") + "/models"
            request = urllib.request.Request(endpoint)
            if api_key:
                request.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:
            return list(self.fallback_models)
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return list(self.fallback_models)
        ids = [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]
        virtual_ids = [model_id for model_id in LISTABLE_MODELS if model_id in ids]
        virtual_ids.extend(model_id for model_id in ids if model_id.startswith(CUSTOM_MODEL_PREFIX))
        return virtual_ids or list(self.fallback_models)


ficelle = FicelleProfile(
    name="ficelle",
    aliases=("Ficelle", "Ficelle FREE"),
    # Hermes skips an api_key provider that declares no env var, so ficelle
    # must declare one even though it needs no auth.
    # See docs/components/hermes-integration.md.
    env_vars=("FICELLE_API_KEY",),
    display_name="Ficelle FREE",
    description="Local strict-zero/free OpenAI-compatible model router",
    signup_url=FICELLE_ROUTER_URL + "/admin",
    base_url=FICELLE_ROUTER_URL + "/v1",
    models_url=FICELLE_ROUTER_URL + "/v1/models",
    fallback_models=VIRTUAL_MODELS,
)

register_provider(ficelle)
