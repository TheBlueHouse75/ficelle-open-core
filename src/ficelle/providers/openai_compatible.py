from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ficelle.config_store import deep_merge
from ficelle.domain_models import ProviderBudget
from ficelle.failures import DEFAULT_FAILURE_MARKERS, FailureMarkers
from ficelle.providers.base import (
    CatalogFetchContext,
    FREE_ACCESS_SCOPES,
    ProviderAccess,
    ProviderAccessContext,
    ProviderCatalogPolicy,
    ProviderNormalizedCatalogModel,
    TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE,
    TRUSTED_PROVIDER_MODEL_FIELDS,
    free_access_payload,
    normalized_free_access_status,
)
from ficelle.redaction import redact_sensitive_json, sanitize_error_detail


OPENCODE_ZEN_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
CONTEXT_LENGTH_CATALOG_ALIASES = ("context_window", "max_context_length")


@dataclass(frozen=True)
class OpenAICompatibleCatalogAdapter:
    source: str

    def request_headers(self) -> dict[str, str]:
        if self.source == "opencode_zen":
            return {"User-Agent": OPENCODE_ZEN_USER_AGENT}
        return {}

    def invocation_headers(self) -> dict[str, str]:
        headers = self.request_headers()
        if self.source == "openrouter":
            return {
                **headers,
                "HTTP-Referer": "http://127.0.0.1:8646",
                "X-Title": "Ficelle",
            }
        return headers

    def failure_markers(self) -> FailureMarkers:
        return DEFAULT_FAILURE_MARKERS

    def resolve_budget(self, context: CatalogFetchContext, provider_cfg: dict[str, Any]) -> ProviderBudget | None:
        """This ACCOUNT's budget, read from the provider, or None if it cannot say.

        A tier that differs between two users of the same build cannot be shipped as a constant, and
        guessing it wrong is worse than not knowing: too low throttles a paying account, too high
        defeats the meter. Providers that expose nothing return None and are counted but never
        capped. Never raises — a router does not fail to start because an account endpoint is down.
        """
        if self.source != "openrouter":
            return None
        try:
            # Resolved inside the guard: on macOS this shells out to the keychain, which is one more
            # thing that can fail for reasons that have nothing to do with the account's tier.
            key, _reason = context.resolve_credentials(self.source, provider_cfg)
            base_url = self._base_url(provider_cfg)
            if not key or not base_url:
                return None
            timeout = context.timeout_seconds
            response = context.http_get(
                f"{base_url}/credits",
                headers={"Accept": "application/json", "Authorization": f"Bearer {key}", **self.request_headers()},
                timeout=(min(5.0, timeout), timeout),
            )
            if not (200 <= int(response.status_code) < 300):
                return None
            purchased = float(((response.json() or {}).get("data") or {}).get("total_credits"))
        except Exception:
            return None
        # OpenRouter's published thresholds for `:free` models, keyed on LIFETIME PURCHASED credits —
        # `total_credits`, not the per-key spend cap `/api/v1/key` reports as `limit`, which an
        # operator sets and which says nothing about the tier.
        return ProviderBudget(1000.0 if purchased >= 10 else 50.0)

    def safe_diagnostics(self, provider_cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "adapter": "openai_compatible",
            "source": self.source,
            "catalog_url": provider_cfg.get("catalog_url"),
            "source_type": provider_cfg.get("source_type"),
            "provider_class": provider_cfg.get("provider_class"),
            "free_mode": provider_cfg.get("free_mode"),
            "free_scope": provider_cfg.get("free_scope"),
            "activation_policy": provider_cfg.get("activation_policy"),
            "quota_reset_policy": provider_cfg.get("quota_reset_policy"),
        }

    def catalog_policy(self, provider_cfg: dict[str, Any]) -> ProviderCatalogPolicy:
        has_trusted_free_access = self._has_trusted_free_access_config(provider_cfg)
        common_rejections = {"no_tools": 0, "small_context": 0, "not_chat": 0, "invalid": 0}
        if has_trusted_free_access:
            rejection_counters = {"not_eligible": 0, **common_rejections}
        else:
            rejection_counters = {"paid": 0, "unsafe_pricing": 0, **common_rejections}
        return ProviderCatalogPolicy(
            has_trusted_free_access=has_trusted_free_access,
            model_defaults=self._safe_provider_model_defaults(provider_cfg),
            model_id_exclude_patterns=self._safe_model_id_patterns(provider_cfg, "model_id_exclude_patterns"),
            model_id_allowlist=self._safe_model_id_patterns(provider_cfg, "model_id_allowlist"),
            requires_exact_model_allowlist=self._requires_exact_model_allowlist(provider_cfg),
            rejection_counters=rejection_counters,
        )

    def normalize_catalog_model(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> ProviderNormalizedCatalogModel:
        normalized_model = self._catalog_row_with_normalized_context(model)
        trusted_model = deep_merge(policy.model_defaults, normalized_model)
        default_params = self._safe_string_list(policy.model_defaults.get("supported_parameters"))
        model_params = self._safe_string_list(normalized_model.get("supported_parameters"))
        if "supported_parameters" in normalized_model and not model_params:
            trusted_model["supported_parameters"] = []
        elif default_params or model_params:
            trusted_model["supported_parameters"] = list(dict.fromkeys([*default_params, *model_params]))
        # Raw provider free_access claims are not trusted; config-derived free_access is added later.
        trusted_model.pop("free_access", None)
        return ProviderNormalizedCatalogModel(normalized_model=normalized_model, trusted_model=trusted_model)

    def excludes_catalog_model(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> bool:
        return (
            self._catalog_row_declares_non_chat(model)
            or self._catalog_row_excluded_by_id(model, policy.model_id_exclude_patterns)
            or self._catalog_row_excluded_by_allowlist(
                model,
                policy.model_id_allowlist,
                exact=policy.requires_exact_model_allowlist,
            )
        )

    def trusted_free_access(
        self,
        provider_cfg: dict[str, Any],
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> dict[str, Any] | None:
        mode = str(provider_cfg.get("free_mode") or "")
        upstream_id = str(model.get("id") or "").strip()
        if mode not in TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE:
            return None
        if not policy.has_trusted_free_access:
            return None
        if not upstream_id:
            return None
        proof = sanitize_error_detail(provider_cfg.get("free_access_proof"), 250) or ""
        if mode == "catalog_free":
            if not proof:
                return None
            if not self._catalog_row_matches_allowlist(
                model,
                policy.model_id_allowlist,
                exact=policy.requires_exact_model_allowlist,
            ):
                return None

        scope = str(provider_cfg.get("free_scope") or "provider")
        if scope not in FREE_ACCESS_SCOPES:
            scope = "provider"
        return free_access_payload(
            True,
            mode,
            proof or "provider_free_endpoint",
            scope,
            normalized_free_access_status(provider_cfg.get("free_status"), "available"),
            sanitize_error_detail(provider_cfg.get("free_note"), 250) or f"{mode} provider catalog",
        )

    def access(
        self,
        provider_cfg: dict[str, Any],
        context: ProviderAccessContext,
        *,
        require_base_url: bool,
    ) -> ProviderAccess:
        if self.source == "nous":
            return self._nous_access(provider_cfg, context)
        key, reason = context.resolve_credentials(self.source, provider_cfg)
        base_url = self._base_url(provider_cfg)
        if not base_url:
            # No base URL means nothing can be sent, whoever is asking. openrouter and mistral
            # used to be exempted here so a key alone read as "configured" — harmless back when
            # their base_url was guaranteed by config, but it let the status row advertise a
            # provider `invoke_model` refuses with this very message, and `selection.py` keeps
            # its models in the routing pool on the strength of that row.
            # `key_reason` carries the store the key came from past this replaced `reason`,
            # so a reader can tell "no key" from "a key nothing can send yet".
            if key or require_base_url:
                return ProviderAccess(
                    key,
                    None,
                    f"missing base_url for provider {self.source}",
                    key_reason=reason,
                )
            return ProviderAccess(None, None, reason)
        if not key and self._allows_keyless_local(provider_cfg):
            return ProviderAccess(None, base_url, "keyless_local", auth_status_invokable=True)
        return ProviderAccess(key, base_url, reason)

    def _nous_access(
        self,
        provider_cfg: dict[str, Any],
        context: ProviderAccessContext,
    ) -> ProviderAccess:
        """Nous always has a base URL, so its answer does not depend on ``require_base_url``."""
        generic_key, generic_reason = context.resolve_credentials(self.source, provider_cfg)
        base_url = self._base_url(provider_cfg) or NOUS_DEFAULT_BASE_URL
        if generic_key:
            return ProviderAccess(generic_key, base_url, generic_reason)
        external_key, external_base_url, external_reason = context.resolve_external_credentials(
            self.source,
            "missing NOUS_API_KEY",
        )
        # The configured Nous URL backs an external key exactly as it backs no key at all. It
        # used to be withheld when require_base_url was False, so the status row called a key
        # "not invokable" that invocation — resolving with True — used without trouble, and
        # `selection.py`, which drops models whose provider is not invokable, kept a working
        # provider out of routing.
        return ProviderAccess(external_key or None, external_base_url or base_url, external_reason)

    def _allows_keyless_local(self, provider_cfg: dict[str, Any]) -> bool:
        provider_class = str(provider_cfg.get("provider_class") or provider_cfg.get("source_type") or "")
        return provider_class == "local" and str(provider_cfg.get("free_mode") or "") == "local_free"

    def _base_url(self, provider_cfg: dict[str, Any]) -> str:
        return str(provider_cfg.get("base_url") or "").strip().rstrip("/")

    def _has_trusted_free_access_config(self, provider_cfg: dict[str, Any]) -> bool:
        mode = str(provider_cfg.get("free_mode") or "")
        provider_class = str(provider_cfg.get("provider_class") or provider_cfg.get("source_type") or "")
        return mode in TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE and provider_class in TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE[mode]

    def _safe_provider_model_defaults(self, provider_cfg: dict[str, Any]) -> dict[str, Any]:
        raw_defaults = provider_cfg.get("catalog_model_defaults")
        if not isinstance(raw_defaults, dict):
            return {}
        safe_defaults = {key: raw_defaults[key] for key in TRUSTED_PROVIDER_MODEL_FIELDS if key in raw_defaults}
        redacted = redact_sensitive_json(safe_defaults)
        return redacted if isinstance(redacted, dict) else {}

    def _safe_model_id_patterns(self, provider_cfg: dict[str, Any], key: str) -> list[str]:
        raw = provider_cfg.get(key)
        if not isinstance(raw, list):
            return []
        return [str(pattern).strip().lower() for pattern in raw if str(pattern).strip()]

    def _requires_exact_model_allowlist(self, provider_cfg: dict[str, Any]) -> bool:
        provider_class = str(provider_cfg.get("provider_class") or provider_cfg.get("source_type") or "")
        return provider_class == "free_model"

    def _catalog_row_with_normalized_context(self, model: dict[str, Any]) -> dict[str, Any]:
        """Map provider context aliases before defaults can mask sparse rows."""
        if model.get("context_length") not in (None, ""):
            return model
        for alias in CONTEXT_LENGTH_CATALOG_ALIASES:
            value = self._safe_optional_int(model.get(alias))
            if value is not None:
                normalized = dict(model)
                normalized["context_length"] = value
                return normalized
        return model

    def _catalog_row_declares_non_chat(self, model: dict[str, Any]) -> bool:
        """True when the catalog explicitly marks a row as not chat-capable."""
        capabilities = model.get("capabilities")
        if isinstance(capabilities, dict) and "completion_chat" in capabilities:
            return capabilities.get("completion_chat") is not True
        return False

    def _catalog_row_excluded_by_id(self, model: dict[str, Any], exclude_patterns: list[str]) -> bool:
        if not exclude_patterns:
            return False
        model_id = str(model.get("id") or "").lower()
        return any(pattern in model_id for pattern in exclude_patterns)

    def _catalog_row_excluded_by_allowlist(
        self,
        model: dict[str, Any],
        allowlist: list[str],
        *,
        exact: bool = False,
    ) -> bool:
        if not allowlist:
            return False
        return not self._catalog_row_matches_allowlist(model, allowlist, exact=exact)

    def _catalog_row_matches_allowlist(
        self,
        model: dict[str, Any],
        allowlist: list[str],
        *,
        exact: bool = False,
    ) -> bool:
        model_id = str(model.get("id") or "").lower()
        if exact:
            return model_id in allowlist
        return any(pattern in model_id for pattern in allowlist)

    def _safe_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _safe_optional_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except Exception:
            return None

    def fetch_catalog(
        self,
        provider_cfg: dict[str, Any],
        context: CatalogFetchContext,
    ) -> tuple[list[dict[str, Any]], str | None]:
        url = provider_cfg["catalog_url"]
        timeout = context.timeout_seconds
        headers = {"Accept": "application/json", **self.request_headers()}
        if self.source == "mistral":
            key, reason = context.resolve_credentials(self.source, provider_cfg)
            if not key:
                return [], reason
            headers["Authorization"] = f"Bearer {key}"
        elif self.source not in {"openrouter", "nous"}:
            key, _reason = context.resolve_credentials(self.source, provider_cfg)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        # Separate, short connect timeout so an unreachable / black-holing provider
        # endpoint fails fast at connect instead of stalling for the full read timeout.
        connect_timeout = min(5.0, timeout)
        response = context.http_get(url, timeout=(connect_timeout, timeout), headers=headers)
        if response.status_code != 200:
            return [], f"HTTP {response.status_code}"
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return [], "invalid catalog shape"
        return data, None
