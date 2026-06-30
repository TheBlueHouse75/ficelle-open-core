from __future__ import annotations

from functools import lru_cache

from ficelle.providers.base import ProviderCatalogAdapter
from ficelle.providers.openai_compatible import OpenAICompatibleCatalogAdapter


@lru_cache(maxsize=None)
def provider_catalog_adapter(source: str) -> ProviderCatalogAdapter:
    return OpenAICompatibleCatalogAdapter(source)
