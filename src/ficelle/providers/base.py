from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ficelle.failures import FailureMarkers
from ficelle.redaction import sanitize_error_detail


CredentialResolver = Callable[[str, dict[str, Any]], tuple[str | None, str]]
ExternalCredentialResolver = Callable[[str, str], tuple[str | None, str | None, str]]
HttpGet = Callable[..., Any]

FREE_ACCESS_MODES = {"catalog_free", "quota_free", "local_free", "paid", "unknown"}
FREE_ACCESS_SCOPES = {"model", "provider", "account", "shared_account"}
FREE_ACCESS_STATUSES = {"available", "unavailable", "cooling_down", "unknown"}
TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE = {
    "catalog_free": {"free_model"},
    "quota_free": {"free_quota", "trial_credit"},
    "local_free": {"local"},
}
TRUSTED_PROVIDER_MODEL_FIELDS = {
    "architecture",
    "context_length",
    "description",
    "knowledge_cutoff",
    "name",
    "provider_account_id",
    "supported_parameters",
    "top_provider",
}


def normalized_free_access_status(value: Any, default: str) -> str:
    status = sanitize_error_detail(value, 64) or default
    return status if status in FREE_ACCESS_STATUSES else "unknown"


def free_access_output_status(status: str, eligible: bool) -> str:
    if status != "available":
        return status
    return "available" if eligible else "unavailable"


def free_access_payload(eligible: bool, mode: str, proof: str, scope: str, status: str, note: str) -> dict[str, Any]:
    return {
        "eligible": eligible,
        "mode": mode,
        "proof": proof,
        "scope": scope,
        "status": free_access_output_status(status, eligible),
        "note": note,
    }


@dataclass(frozen=True)
class ProviderAccess:
    key: str | None
    base_url: str | None
    reason: str
    auth_status_invokable: bool = False


@dataclass(frozen=True)
class ProviderAccessContext:
    resolve_credentials: CredentialResolver
    resolve_external_credentials: ExternalCredentialResolver


@dataclass(frozen=True)
class CatalogFetchContext:
    timeout_seconds: float
    http_get: HttpGet
    resolve_credentials: CredentialResolver


@dataclass(frozen=True)
class ProviderCatalogPolicy:
    has_trusted_free_access: bool
    model_defaults: dict[str, Any]
    model_id_exclude_patterns: list[str]
    model_id_allowlist: list[str]
    requires_exact_model_allowlist: bool
    rejection_counters: dict[str, int]


@dataclass(frozen=True)
class ProviderNormalizedCatalogModel:
    normalized_model: dict[str, Any]
    trusted_model: dict[str, Any]


class ProviderCatalogAdapter(Protocol):
    source: str

    def request_headers(self) -> dict[str, str]:
        ...

    def invocation_headers(self) -> dict[str, str]:
        ...

    def failure_markers(self) -> FailureMarkers:
        ...

    def safe_diagnostics(self, provider_cfg: dict[str, Any]) -> dict[str, Any]:
        ...

    def catalog_policy(self, provider_cfg: dict[str, Any]) -> ProviderCatalogPolicy:
        ...

    def normalize_catalog_model(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> ProviderNormalizedCatalogModel:
        ...

    def excludes_catalog_model(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> bool:
        ...

    def trusted_free_access(
        self,
        provider_cfg: dict[str, Any],
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> dict[str, Any] | None:
        ...

    def access(
        self,
        provider_cfg: dict[str, Any],
        context: ProviderAccessContext,
        *,
        require_base_url: bool,
    ) -> ProviderAccess:
        ...

    def fetch_catalog(
        self,
        provider_cfg: dict[str, Any],
        context: CatalogFetchContext,
    ) -> tuple[list[dict[str, Any]], str | None]:
        ...
