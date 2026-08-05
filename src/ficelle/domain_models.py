from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal


SelectionPurpose = Literal["route", "benchmark"]

_SENSITIVE_JSON_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "accesstoken",
    "refresh_token",
    "refreshtoken",
    "token",
}
_SENSITIVE_KEY_PARTS = ("authorization", "bearer", "credential", "password", "secret", "token")


def _safe_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _safe_dict(value: Any) -> dict[str, Any]:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    compact = normalized.replace("_", "")
    return (
        normalized in _SENSITIVE_JSON_KEYS
        or compact in _SENSITIVE_JSON_KEYS
        or normalized.endswith("_api_key")
        or normalized.endswith("_access_token")
        or normalized.endswith("_refresh_token")
        or compact.endswith("apikey")
        or compact.endswith("accesstoken")
        or compact.endswith("refreshtoken")
        or any(part in normalized for part in _SENSITIVE_KEY_PARTS)
    )


def _redact_extra_value(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = str(key)
            safe[safe_key] = "[redacted]" if _is_sensitive_key(key) and item not in (None, "") else _redact_extra_value(item)
        return safe
    if isinstance(value, list):
        return [_redact_extra_value(item) for item in value]
    return copy.deepcopy(value)


def _redact_unknown_fields(raw: dict[str, Any], known_fields: set[str]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        safe_key = str(key)
        if safe_key in known_fields:
            continue
        if _is_sensitive_key(safe_key):
            extra[safe_key] = "[redacted]"
        else:
            extra[safe_key] = _redact_extra_value(value)
    return extra


@dataclass(frozen=True)
class CatalogModel:
    id: str
    source: str
    upstream_id: str
    context_length: int
    supports_tools: bool
    supports_structured_outputs: bool
    invokable: bool
    pricing: dict[str, Any] = field(default_factory=dict, repr=False)
    pricing_safety: dict[str, Any] = field(default_factory=dict, repr=False)
    free_access: dict[str, Any] = field(default_factory=dict, repr=False)
    supported_parameters: list[str] = field(default_factory=list)
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    provider_account_id: str | None = None
    name: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_raw(cls, row: Any) -> "CatalogModel | None":
        if not isinstance(row, dict):
            return None
        known = set(cls.__dataclass_fields__) - {"extra_fields", "raw"}
        model_id = _safe_string(row.get("id"))
        source = _safe_string(row.get("source"))
        upstream_id = _safe_string(row.get("upstream_id"))
        if not upstream_id:
            source_prefix = f"ficelle/{source}/" if source else ""
            upstream_id = model_id.removeprefix(source_prefix) if model_id and source_prefix else model_id
        return cls(
            id=model_id,
            source=source,
            upstream_id=upstream_id,
            context_length=max(0, _safe_int(row.get("context_length"), 0)),
            supports_tools=bool(row.get("supports_tools")),
            supports_structured_outputs=bool(row.get("supports_structured_outputs")),
            invokable=bool(row.get("invokable")),
            pricing=_safe_dict(row.get("pricing")),
            pricing_safety=_safe_dict(row.get("pricing_safety")),
            free_access=_safe_dict(row.get("free_access")),
            supported_parameters=_safe_string_list(row.get("supported_parameters")),
            input_modalities=_safe_string_list(row.get("input_modalities")),
            output_modalities=_safe_string_list(row.get("output_modalities")),
            provider_account_id=_safe_string(row.get("provider_account_id")) or None,
            name=_safe_string(row.get("name")) or None,
            extra_fields=_redact_unknown_fields(row, known),
            raw=copy.deepcopy(row),
        )

    def as_legacy_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)


@dataclass
class RouterModel:
    id: str
    source: str
    upstream_id: str
    name: str | None
    context_length: int
    supports_tools: bool
    supports_structured_outputs: bool
    pricing: dict[str, Any]
    pricing_safety: dict[str, Any]
    free_access: dict[str, Any]
    provider_account_id: str | None
    supported_parameters: list[str]
    input_modalities: list[str]
    output_modalities: list[str]
    modality: str | None
    tokenizer: str | None
    instruct_type: str | None
    max_completion_tokens: int | None
    is_moderated: bool | None
    knowledge_cutoff: str | None
    description: str | None
    invokable: bool
    auth_reason: str | None = None
    catalog_url: str | None = None
    base_url: str | None = None
    capabilities_from_defaults: list[str] = field(default_factory=list)
    capabilities_from_reference: list[str] = field(default_factory=list)
    capabilities_refuted_by_reference: list[str] = field(default_factory=list)
    reference_confidence: str = ""
    context_length_source: str = "catalog"
    context_length_estimate: int | None = None
    burn_weight: float | None = None


@dataclass(frozen=True)
class ProviderBudget:
    """One account's allowance for one pool of upstream capacity, as the provider states it.

    A budget is a stock with a refill schedule, not a bare number: providers disagree on the unit
    (requests, tokens, dollars, Neurons) and on the window (daily for most, monthly for Cohere), so
    both travel with the amount instead of being baked into field names and state keys. Only
    ``requests`` budgets are enforced today — that is the unit Ficelle's own meter records — but a
    budget in any unit can be stored and surfaced without a schema change.
    """

    amount: float
    unit: str = "requests"
    period: str = "day"


@dataclass(frozen=True)
class SelectionRequest:
    requested_model: str
    canonical_model: str
    purpose: SelectionPurpose = "route"


@dataclass(frozen=True)
class SelectionResult:
    request: SelectionRequest
    candidates: list[CatalogModel]
    excluded_reasons: dict[str, str] = field(default_factory=dict)
    anti_empty_fallback: bool = False

    def as_legacy_models(self) -> list[dict[str, Any]]:
        return [candidate.as_legacy_dict() for candidate in self.candidates]
