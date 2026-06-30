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
)
FICELLE_ROUTER_URL = "http://127.0.0.1:8646"

# Fusion is an optional virtual profile: the gateway only advertises
# ficelle/auto-fusion in /v1/models when fusion is enabled and exposed. Mirror
# that by letting it through the live filter while keeping it out of the offline
# fallback, so we never advertise it when the gateway is unreachable.
FUSION_MODEL = "ficelle/auto-fusion"
LISTABLE_MODELS = (*VIRTUAL_MODELS, FUSION_MODEL)


class FicelleProfile(ProviderProfile):
    def fetch_models(self, *, api_key: str | None = None, timeout: float = 8.0) -> list[str] | None:
        try:
            with urllib.request.urlopen(self.base_url.rstrip("/") + "/models", timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:
            return list(self.fallback_models)
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return list(self.fallback_models)
        ids = [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]
        virtual_ids = [model_id for model_id in LISTABLE_MODELS if model_id in ids]
        return virtual_ids or list(self.fallback_models)


ficelle = FicelleProfile(
    name="ficelle",
    aliases=("Ficelle", "Ficelle FREE"),
    env_vars=(),
    display_name="Ficelle FREE",
    description="Local strict-zero/free OpenAI-compatible model router",
    signup_url=FICELLE_ROUTER_URL + "/admin",
    base_url=FICELLE_ROUTER_URL + "/v1",
    models_url=FICELLE_ROUTER_URL + "/v1/models",
    fallback_models=VIRTUAL_MODELS,
)

register_provider(ficelle)
