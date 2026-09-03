from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ficelle.providers.base import (
    CatalogFetchContext,
    ProviderAccess,
    ProviderAccessContext,
    ProviderCatalogAdapter,
    missing_provider_parameter,
)


@dataclass(frozen=True)
class ProviderCatalogPorts:
    provider_catalog_adapter: Callable[[str], ProviderCatalogAdapter]
    resolve_credentials: Callable[[str, dict[str, Any]], tuple[str | None, str]]
    resolve_external_credentials: Callable[[str, str], tuple[str | None, str | None, str]]
    http_get: Callable[..., Any]
    resolve_provider_parameter: Callable[[str], str | None] = missing_provider_parameter


def provider_invocation_headers(source: str, *, ports: ProviderCatalogPorts) -> dict[str, str]:
    return ports.provider_catalog_adapter(source).invocation_headers()


def provider_access_result(
    source: str,
    provider_cfg: dict[str, Any],
    *,
    require_base_url: bool,
    ports: ProviderCatalogPorts,
) -> ProviderAccess:
    if not provider_cfg.get("enabled", True):
        base_url = str(provider_cfg.get("base_url") or "").strip().rstrip("/") or None
        return ProviderAccess(None, base_url, "disabled")
    context = ProviderAccessContext(
        resolve_credentials=ports.resolve_credentials,
        resolve_external_credentials=ports.resolve_external_credentials,
        resolve_provider_parameter=ports.resolve_provider_parameter,
    )
    return ports.provider_catalog_adapter(source).access(
        provider_cfg,
        context,
        require_base_url=require_base_url,
    )


def fetch_provider_catalog(
    source: str,
    config: dict[str, Any],
    *,
    ports: ProviderCatalogPorts,
) -> tuple[list[dict[str, Any]], str | None]:
    provider_cfg = config["providers"][source]
    context = CatalogFetchContext(
        timeout_seconds=float(config.get("catalog_timeout_seconds") or 30),
        http_get=ports.http_get,
        resolve_credentials=ports.resolve_credentials,
        resolve_provider_parameter=ports.resolve_provider_parameter,
    )
    return ports.provider_catalog_adapter(source).fetch_catalog(provider_cfg, context)
