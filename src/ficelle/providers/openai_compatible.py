from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

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
    ProviderParameterResolver,
    TRUSTED_FREE_PROVIDER_CLASSES_BY_MODE,
    TRUSTED_PROVIDER_MODEL_FIELDS,
    free_access_payload,
    normalized_free_access_status,
    suppress_implicit_http_auth,
)
from ficelle.redaction import redact_sensitive_json, sanitize_error_detail


def _price_per_token(value: Any) -> float | None:
    # Bools coerce to 1.0 and "Infinity" parses to inf — both would inflate the
    # savings estimate, the one direction it must never err in. Finite and >= 0 only.
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and math.isfinite(parsed) else None


OPENCODE_ZEN_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
CONTEXT_LENGTH_CATALOG_ALIASES = ("context_window", "max_context_length")
CLOUDFLARE_ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_ACCOUNT_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
CLOUDFLARE_MAX_CATALOG_PAGES = 100
CLOUDFLARE_CATALOG_PAGE_SIZE = 100
GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR = "skip_thought_signature_validator"
MINISTRAL_MODEL_PREFIXES = ("ministral-3b", "ministral-8b", "ministral-14b")


def _is_gemini_thought_signature_model(model: dict[str, Any]) -> bool:
    upstream_id = model.get("upstream_id") or model.get("id")
    if not isinstance(upstream_id, str):
        return False
    return any(part.startswith("gemini-") for part in upstream_id.lower().strip("/").split("/"))


def _is_confirmed_ministral_model(model: dict[str, Any]) -> bool:
    upstream_id = model.get("upstream_id") or model.get("id")
    if not isinstance(upstream_id, str):
        return False
    model_name = upstream_id.lower().strip("/").rsplit("/", 1)[-1]
    return any(
        model_name == prefix or model_name.startswith(f"{prefix}-")
        for prefix in MINISTRAL_MODEL_PREFIXES
    )


def _is_standard_user_message(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    return isinstance(content, str) or (isinstance(content, list) and bool(content))


@dataclass(frozen=True)
class OpenAICompatibleCatalogAdapter:
    def adapt_chat_request(
        self,
        payload: dict[str, Any],
        model: dict[str, Any],
        provider_cfg: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply the narrow, live-proven provider request compatibility rewrites.

        Mistral's confirmed Ministral models reject an explicitly disabled ``reasoning`` object;
        that one field is removed from a copied body. Other providers, other Mistral models, Google
        models other than Gemini, and Gemini bodies without an unsigned current-turn tool step keep
        object identity. Gemini requires a thought signature on the first parallel function call,
        but permits its official validator-skip sentinel for tool history from another model.
        Preserve any supplied signature rather than replacing it.
        """
        if self.source == "mistral" and _is_confirmed_ministral_model(model):
            reasoning = payload.get("reasoning")
            if isinstance(reasoning, dict) and reasoning.get("enabled") is False:
                adapted_payload = dict(payload)
                adapted_payload.pop("reasoning", None)
                return adapted_payload

        if self.source != "gemini" or not _is_gemini_thought_signature_model(model):
            return payload

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        latest_user_message_index = max(
            (
                index
                for index, message in enumerate(messages)
                if _is_standard_user_message(message)
            ),
            default=-1,
        )
        if latest_user_message_index < 0:
            return payload

        adapted_payload: dict[str, Any] | None = None
        adapted_messages: list[Any] | None = None
        for message_index in range(latest_user_message_index + 1, len(messages)):
            message = messages[message_index]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            first_function_call_index = next(
                (
                    index
                    for index, tool_call in enumerate(tool_calls)
                    if isinstance(tool_call, dict) and tool_call.get("type") == "function"
                ),
                None,
            )
            if first_function_call_index is None:
                continue
            first_function_call = tool_calls[first_function_call_index]
            assert isinstance(first_function_call, dict)
            extra_content = first_function_call.get("extra_content")
            google_content = extra_content.get("google") if isinstance(extra_content, dict) else None
            thought_signature = (
                google_content.get("thought_signature") if isinstance(google_content, dict) else None
            )
            if isinstance(thought_signature, str) and thought_signature.strip():
                continue

            if adapted_payload is None:
                adapted_payload = dict(payload)
                adapted_messages = list(messages)
                adapted_payload["messages"] = adapted_messages
            assert adapted_messages is not None
            adapted_tool_calls = list(tool_calls)
            adapted_function_call = dict(first_function_call)
            adapted_extra_content = dict(extra_content) if isinstance(extra_content, dict) else {}
            adapted_google_content = dict(google_content) if isinstance(google_content, dict) else {}
            adapted_google_content["thought_signature"] = GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR
            adapted_extra_content["google"] = adapted_google_content
            adapted_function_call["extra_content"] = adapted_extra_content
            adapted_tool_calls[first_function_call_index] = adapted_function_call
            adapted_message = dict(message)
            adapted_message["tool_calls"] = adapted_tool_calls
            adapted_messages[message_index] = adapted_message

        return adapted_payload if adapted_payload is not None else payload

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
        if self.source == "mistral":
            return DEFAULT_FAILURE_MARKERS.with_extra(
                # Mistral's exact 429 verdict means the selected Ministral serving pool is at
                # capacity; it is model-scoped even when the response message is terse.
                upstream_rate_limit_error_codes=(("3505", "backend_out_of_capacity"),),
            )
        if self.source == "cloudflare":
            return DEFAULT_FAILURE_MARKERS.with_extra(
                false_free=(
                    "this model requires a workers paid plan",
                    "not available through standard workers free billing",
                ),
                quota_exhausted=(
                    "used up your daily free allocation of 10,000 neurons",
                ),
            )
        if self.source == "ovhcloud":
            # OVH's anonymous cap is per IP/model. Its terse 429 body does not name the model,
            # so classify the provider-specific wording as quota exhaustion and let
            # free_scope=model select the correct cooldown instead of pausing the provider.
            return DEFAULT_FAILURE_MARKERS.with_extra(
                quota_exhausted=("api rate limit exceeded",),
            )
        if self.source == "ollama":
            # Ollama's cloud catalog can list a model that later requires a subscription. Its live
            # 403 uses account-like status semantics, so these provider-specific words must win
            # before the generic 403 auth classification: quarantine only that model, not Ollama.
            return DEFAULT_FAILURE_MARKERS.with_extra(
                false_free=(
                    "requires a subscription",
                    "subscription required",
                    "upgrade for access",
                )
            )
        if self.source == "openrouter":
            # OpenRouter gates some free endpoints to registered agentic harnesses. A plain API
            # key gets a 403 naming that one model id while every sibling keeps answering, so it
            # must classify as `model_not_found` (quarantine) rather than the generic 403 reading
            # that would cool the whole provider — see incident: one gated free model cooled all
            # of OpenRouter for an hour (2026-09-11).
            return DEFAULT_FAILURE_MARKERS.with_extra(
                model_not_entitled=("only available on agentic harnesses",),
            )
        return DEFAULT_FAILURE_MARKERS

    def reference_prices_for_free_models(self, raw_models: list[Any]) -> dict[str, dict[str, Any]]:
        """Free-model upstream id -> the paid sibling's USD-per-token prices, with provenance.

        OpenRouter dialect: a strict-zero ``<id>:free`` listing shadows a paid ``<id>``
        in the same payload, quoted in USD per token. Both halves of that convention —
        the id suffix and the pricing unit — are this provider's, so they live here;
        other sources return {} until they declare a pairing of their own.
        """
        if self.source != "openrouter":
            return {}
        paid: dict[str, dict[str, float]] = {}
        for model in raw_models:
            if not isinstance(model, dict):
                continue
            upstream_id = str(model.get("id") or "").strip()
            pricing = model.get("pricing")
            if not upstream_id or upstream_id.endswith(":free") or not isinstance(pricing, dict):
                continue
            prompt = _price_per_token(pricing.get("prompt"))
            completion = _price_per_token(pricing.get("completion"))
            if prompt is None or completion is None or (prompt <= 0 and completion <= 0):
                continue
            paid[upstream_id] = {"prompt": prompt, "completion": completion}
        references: dict[str, dict[str, Any]] = {}
        for model in raw_models:
            if not isinstance(model, dict):
                continue
            upstream_id = str(model.get("id") or "").strip()
            if not upstream_id.endswith(":free"):
                continue
            sibling = upstream_id.removesuffix(":free")
            price = paid.get(sibling)
            if price is not None:
                references[upstream_id] = {**price, "basis": f"{self.source}:{sibling}"}
        return references

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
            "catalog_url": provider_cfg.get("catalog_url") or provider_cfg.get("catalog_url_template"),
            "source_type": provider_cfg.get("source_type"),
            "provider_class": provider_cfg.get("provider_class"),
            "free_mode": provider_cfg.get("free_mode"),
            "free_scope": provider_cfg.get("free_scope"),
            "auth_mode": provider_cfg.get("auth_mode"),
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
            model_overrides=self._safe_model_overrides(provider_cfg),
            requires_model_allowlist=bool(provider_cfg.get("require_model_id_allowlist")),
            official_free_ids=self._safe_model_id_patterns(provider_cfg, "official_free_ids"),
            official_free_id_suffixes=self._safe_model_id_patterns(provider_cfg, "official_free_id_suffixes"),
        )

    def normalize_catalog_model(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> ProviderNormalizedCatalogModel:
        if self.source == "requesty":
            normalized_model = self._normalize_requesty_catalog_model(model)
        elif self.source == "cloudflare":
            normalized_model = self._normalize_cloudflare_catalog_model(model)
        elif self.source == "routeway":
            normalized_model = self._normalize_routeway_catalog_model(model)
        else:
            normalized_model = model
        normalized_model = self._catalog_row_with_normalized_limits(normalized_model)
        # A matching family override replaces the provider-wide guess for this row only, so
        # a correction never widens beyond the ids it names.
        defaults = self._model_defaults_for_row(policy, normalized_model)
        trusted_model = deep_merge(defaults, normalized_model)
        default_params = self._safe_string_list(defaults.get("supported_parameters"))
        model_params = self._safe_string_list(normalized_model.get("supported_parameters"))
        if "supported_parameters" in normalized_model and not model_params:
            trusted_model["supported_parameters"] = []
        elif default_params or model_params:
            trusted_model["supported_parameters"] = list(dict.fromkeys([*default_params, *model_params]))
        # Raw provider free_access claims are not trusted; config-derived free_access is added later.
        trusted_model.pop("free_access", None)
        return ProviderNormalizedCatalogModel(normalized_model=normalized_model, trusted_model=trusted_model)

    def _normalize_cloudflare_catalog_model(self, model: dict[str, Any]) -> dict[str, Any]:
        """Translate Workers AI's native catalog without curating model ids.

        Cloudflare's row id is an internal UUID; ``name`` is the invocation model id. A valid
        ``properties`` list is also the provider-owned billing contract: every standard Workers
        model shares the free allocation unless ``require_workers_paid`` is true. Malformed or
        contradictory properties fail closed instead of inheriting that default.
        """
        normalized = dict(model)
        normalized["id"] = str(model.get("name") or "").strip()
        properties, properties_valid = self._cloudflare_properties(model.get("properties"))
        has_paid_marker = "require_workers_paid" in properties
        paid_marker = properties.get("require_workers_paid")
        free_paid_marker = not has_paid_marker or paid_marker is False or paid_marker == "false"
        normalized["workers_free_eligible"] = properties_valid and free_paid_marker

        task = model.get("task")
        task_name = str(task.get("name") or "") if isinstance(task, dict) else ""
        schema = model.get("schema") if isinstance(model.get("schema"), dict) else {}
        input_schema = schema.get("input") if isinstance(schema, dict) else None
        # The live `/ai/models/search` rows carry no `schema` at all (verified 11/09/2026), so
        # the task name is the only chat signal there. A schema, when present, still decides:
        # a row that describes a prompt-only input is not a chat model whatever its task says.
        supports_chat = (
            self._json_schema_declares_property(input_schema, "messages")
            if isinstance(input_schema, dict)
            else task_name == "Text Generation"
        )
        capabilities = model.get("capabilities")
        normalized_capabilities = dict(capabilities) if isinstance(capabilities, dict) else {}
        normalized_capabilities["completion_chat"] = supports_chat
        normalized["capabilities"] = normalized_capabilities

        context_length = self._safe_optional_int(properties.get("context_window"))
        if context_length is not None:
            normalized["context_length"] = context_length

        supported_parameters: list[str] = []
        if properties.get("function_calling") in (True, "true") or self._json_schema_declares_property(
            input_schema,
            "tools",
        ):
            supported_parameters.extend(("tools", "tool_choice"))
        if self._json_schema_declares_property(input_schema, "response_format"):
            supported_parameters.append("response_format")
        if properties.get("reasoning") in (True, "true"):
            supported_parameters.append("reasoning")
        normalized["supported_parameters"] = supported_parameters

        input_modalities = ["text"]
        task_lower = task_name.lower()
        if (
            properties.get("vision") in (True, "true")
            or "image" in task_lower
            or self._json_schema_declares_property(input_schema, "image_url")
        ):
            input_modalities.append("image")
        normalized["architecture"] = {
            "input_modalities": input_modalities,
            "output_modalities": ["text"],
        }
        normalized["pricing"] = self._cloudflare_pricing(properties.get("price"))
        return normalized

    @staticmethod
    def _cloudflare_properties(raw_properties: Any) -> tuple[dict[str, Any], bool]:
        if not isinstance(raw_properties, list) or not raw_properties:
            return {}, False
        properties: dict[str, Any] = {}
        valid = True
        for row in raw_properties:
            if not isinstance(row, dict):
                valid = False
                continue
            key = str(row.get("property_id") or "").strip()
            if not key or "value" not in row:
                valid = False
                continue
            value = row["value"]
            if key in properties and properties[key] != value:
                valid = False
                continue
            properties[key] = value
        return properties, valid

    @staticmethod
    def _cloudflare_pricing(raw_prices: Any) -> dict[str, float]:
        if not isinstance(raw_prices, list):
            return {}
        pricing: dict[str, float] = {}
        for row in raw_prices:
            if not isinstance(row, dict) or str(row.get("currency") or "").upper() != "USD":
                continue
            unit = str(row.get("unit") or "").lower()
            price = _price_per_token(row.get("price"))
            if price is None:
                continue
            if "per m input tokens" in unit and "cached" not in unit:
                pricing["prompt"] = price / 1_000_000
            elif "per m output tokens" in unit:
                pricing["completion"] = price / 1_000_000
            elif "per m cached input tokens" in unit:
                pricing["cached"] = price / 1_000_000
        return pricing

    @staticmethod
    def _json_schema_declares_property(schema: Any, needle: str) -> bool:
        """Find an actual JSON Schema property declaration, not a word in prose or examples."""
        stack = [schema]
        visited = 0
        while stack and visited < 10_000:
            current = stack.pop()
            visited += 1
            if isinstance(current, dict):
                properties = current.get("properties")
                required = current.get("required")
                if isinstance(properties, dict) and needle in properties:
                    return True
                if isinstance(required, list) and needle in required:
                    return True
                if isinstance(properties, dict):
                    stack.extend(properties.values())
                for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
                    nested = current.get(keyword)
                    if isinstance(nested, list):
                        stack.extend(nested)
                for keyword in ("items", "additionalProperties", "not", "if", "then", "else"):
                    nested = current.get(keyword)
                    if isinstance(nested, (dict, list)):
                        stack.append(nested)
                for keyword in ("$defs", "definitions", "patternProperties", "dependentSchemas"):
                    nested = current.get(keyword)
                    if isinstance(nested, dict):
                        stack.extend(nested.values())
            elif isinstance(current, list):
                stack.extend(current)
        return False

    def _normalize_routeway_catalog_model(self, model: dict[str, Any]) -> dict[str, Any]:
        """Translate Routeway's public catalog without turning it into an allowlist.

        Routeway marks free rows twice: an official ``:free`` id suffix and nested
        ``price_per_million_t`` rates. The suffix gate lives in provider config; this
        normalizer preserves the independent pricing proof and the row-level routing facts.
        Missing, contradictory, unavailable, or newly shaped data stays ineligible.
        """
        normalized = dict(model)
        normalized["pricing"] = self._routeway_flat_pricing(model.get("pricing"))

        endpoints = model.get("endpoints")
        capabilities = model.get("capabilities")
        normalized_capabilities = dict(capabilities) if isinstance(capabilities, dict) else {}
        normalized_capabilities["completion_chat"] = (
            model.get("available") is True
            and isinstance(endpoints, list)
            and "/v1/chat/completions" in endpoints
        )
        normalized["capabilities"] = normalized_capabilities

        params = self._safe_string_list(model.get("supported_parameters"))
        if not isinstance(capabilities, dict) or capabilities.get("function_call") is not True:
            params = [param for param in params if param not in {"tools", "tool_choice"}]
        normalized["supported_parameters"] = params

        vision = capabilities.get("vision") if isinstance(capabilities, dict) else None
        if isinstance(vision, bool):
            normalized["architecture"] = {
                "input_modalities": ["text", "image"] if vision else ["text"],
                "output_modalities": ["text"],
            }
        return normalized

    @classmethod
    def _routeway_flat_pricing(cls, raw_pricing: Any) -> dict[str, Any]:
        """Flatten every documented Routeway token rate for the strict-zero guard."""
        if not isinstance(raw_pricing, dict) or not {"input", "output"} <= set(raw_pricing):
            return {}
        if set(raw_pricing) - {"input", "output", "caching"}:
            return {}

        flattened: dict[str, Any] = {}
        for source_field, target_field in (("input", "prompt"), ("output", "completion")):
            rate = cls._routeway_rate_value(raw_pricing.get(source_field))
            if rate is None:
                return {}
            flattened[target_field] = rate

        if "caching" in raw_pricing:
            caching_rates: dict[str, Any] = {}
            if not cls._routeway_collect_rates(raw_pricing.get("caching"), "caching", caching_rates):
                return {}
            flattened.update(caching_rates)
        return flattened

    @staticmethod
    def _routeway_rate_value(value: Any) -> Any | None:
        if not isinstance(value, dict) or "price_per_million_t" not in value:
            return None
        if set(value) - {"unit", "price_per_million_t"}:
            return None
        rate = value.get("price_per_million_t")
        if isinstance(rate, bool) or rate in (None, ""):
            return None
        return rate

    @classmethod
    def _routeway_collect_rates(
        cls,
        value: Any,
        path: str,
        flattened: dict[str, Any],
    ) -> bool:
        if not isinstance(value, dict) or not value:
            return False
        if "price_per_million_t" in value:
            rate = cls._routeway_rate_value(value)
            if rate is None:
                return False
            flattened[path] = rate
            return True
        for key, child in value.items():
            if not cls._routeway_collect_rates(child, f"{path}.{key}", flattened):
                return False
        return True

    def _normalize_requesty_catalog_model(self, model: dict[str, Any]) -> dict[str, Any]:
        """Translate Requesty's public catalog fields into Ficelle's generic contract.

        Every mapped field remains fail-closed: absent prices stay absent, explicit false
        capabilities become an empty parameter list, and a non-chat ``api`` marker is preserved as
        a negative capability. No account response or inferred model-family metadata is involved.
        """
        normalized = dict(model)
        pricing: dict[str, Any] = {}
        for source_field, target_field in (
            ("input_price", "prompt"),
            ("output_price", "completion"),
            ("cached_price", "cached"),
            ("caching_price", "caching"),
        ):
            if source_field in model:
                value = model[source_field]
                pricing[target_field] = None if isinstance(value, bool) else value
        if "pricing" in model and not self._requesty_pricing_tiers_are_strict_zero(model.get("pricing")):
            pricing = {}
        normalized["pricing"] = pricing

        capability_fields = {
            "supports_tool_calling",
            "supports_output_json_object",
            "supports_output_json_schema",
        }
        if capability_fields.intersection(model):
            supported_parameters: list[str] = []
            if model.get("supports_tool_calling") is True:
                supported_parameters.extend(("tools", "tool_choice"))
            if model.get("supports_output_json_object") is True or model.get("supports_output_json_schema") is True:
                supported_parameters.append("response_format")
            normalized["supported_parameters"] = supported_parameters

        capabilities = model.get("capabilities")
        normalized_capabilities = dict(capabilities) if isinstance(capabilities, dict) else {}
        normalized_capabilities["completion_chat"] = model.get("api") == "chat"
        normalized["capabilities"] = normalized_capabilities
        return normalized

    @staticmethod
    def _requesty_pricing_tiers_are_strict_zero(raw_tiers: Any) -> bool:
        if not isinstance(raw_tiers, list) or not raw_tiers:
            return False
        for tier in raw_tiers:
            if not isinstance(tier, dict) or not {"input_price", "output_price"} <= set(tier):
                return False
            exposed_prices = (value for key, value in tier.items() if str(key).endswith("_price"))
            try:
                if any(
                    isinstance(value, bool) or value in (None, "") or float(value) != 0.0
                    for value in exposed_prices
                ):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def excludes_catalog_model(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> bool:
        return (
            self._catalog_row_declares_non_chat(model)
            or self._catalog_row_excluded_by_id(model, policy.model_id_exclude_patterns)
            or (
                self._has_official_free_id_gate(policy)
                and not self._catalog_row_matches_official_free_id(model, policy)
            )
            or self._catalog_row_excluded_by_allowlist(
                model,
                policy.model_id_allowlist,
                exact=policy.requires_exact_model_allowlist,
                required=policy.requires_model_allowlist and not self._has_official_free_id_gate(policy),
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
        if self.source == "cloudflare" and model.get("workers_free_eligible") is not True:
            return free_access_payload(
                False,
                mode,
                sanitize_error_detail(provider_cfg.get("free_access_proof"), 250) or "provider_free_endpoint",
                "provider",
                "unavailable",
                "Cloudflare catalog row requires Workers Paid or has ambiguous billing metadata",
            )
        if (
            policy.requires_model_allowlist
            and not self._has_official_free_id_gate(policy)
            and not self._catalog_row_matches_allowlist(
                model,
                policy.model_id_allowlist,
                exact=True,
            )
        ):
            return None
        proof = sanitize_error_detail(provider_cfg.get("free_access_proof"), 250) or ""
        scope = str(provider_cfg.get("free_scope") or "provider")
        if scope not in FREE_ACCESS_SCOPES:
            scope = "provider"
        verified_flag_field = ""
        if mode == "catalog_free":
            if not proof:
                return None
            if self._has_official_free_id_gate(policy):
                if not self._catalog_row_matches_official_free_id(model, policy):
                    return None
            elif proof == "provider_free_catalog_pricing":
                flag_field = sanitize_error_detail(provider_cfg.get("catalog_free_flag_field"), 120) or ""
                if flag_field and model.get(flag_field) is not True:
                    return free_access_payload(
                        False,
                        mode,
                        proof,
                        scope,
                        "unavailable",
                        f"provider catalog did not mark {flag_field} as true",
                    )
                verified_flag_field = flag_field
            elif not self._catalog_row_matches_allowlist(
                model,
                policy.model_id_allowlist,
                exact=policy.requires_exact_model_allowlist,
            ):
                return None

        access = free_access_payload(
            True,
            mode,
            proof or "provider_free_endpoint",
            scope,
            normalized_free_access_status(provider_cfg.get("free_status"), "available"),
            sanitize_error_detail(provider_cfg.get("free_note"), 250) or f"{mode} provider catalog",
        )
        if verified_flag_field:
            access["catalog_free_flag_verified"] = verified_flag_field
        return access

    def access(
        self,
        provider_cfg: dict[str, Any],
        context: ProviderAccessContext,
        *,
        require_base_url: bool,
    ) -> ProviderAccess:
        if self.source == "nous":
            return self._nous_access(provider_cfg, context)
        base_url = self._base_url(provider_cfg, context.resolve_provider_parameter)
        if self._uses_anonymous_auth(provider_cfg):
            if not base_url:
                return ProviderAccess(None, None, self._missing_base_url_reason())
            # Anonymous remote access is an explicit provider contract, not a fallback after
            # credential resolution. Skipping the resolver is load-bearing: an ambient key must
            # never turn a structurally no-spend endpoint into that provider's paid auth lane.
            return ProviderAccess(None, base_url, "anonymous_remote", auth_status_invokable=True)
        key, reason = context.resolve_credentials(self.source, provider_cfg)
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
                    self._missing_base_url_reason(),
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
            generic_reason,
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

    def _uses_anonymous_auth(self, provider_cfg: dict[str, Any]) -> bool:
        return str(provider_cfg.get("auth_mode") or "") == "anonymous"

    def _base_url(
        self,
        provider_cfg: dict[str, Any],
        resolve_provider_parameter: ProviderParameterResolver | None = None,
    ) -> str:
        if self.source == "cloudflare":
            return self._cloudflare_url(provider_cfg, "base_url_template", resolve_provider_parameter)
        return str(provider_cfg.get("base_url") or "").strip().rstrip("/")

    def _missing_base_url_reason(self) -> str:
        if self.source == "cloudflare":
            return f"missing or invalid {CLOUDFLARE_ACCOUNT_ID_ENV} for provider cloudflare"
        return f"missing base_url for provider {self.source}"

    def _cloudflare_url(
        self,
        provider_cfg: dict[str, Any],
        template_key: str,
        resolve_provider_parameter: ProviderParameterResolver | None,
    ) -> str:
        raw_account_id = str(provider_cfg.get("account_id") or "").strip()
        if not raw_account_id and callable(resolve_provider_parameter):
            raw_account_id = str(resolve_provider_parameter(CLOUDFLARE_ACCOUNT_ID_ENV) or "").strip()
        if not CLOUDFLARE_ACCOUNT_ID_PATTERN.fullmatch(raw_account_id):
            return ""
        template = str(provider_cfg.get(template_key) or "").strip()
        if "{account_id}" not in template:
            return ""
        return template.replace("{account_id}", raw_account_id).rstrip("/")

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

    def _safe_model_overrides(self, provider_cfg: dict[str, Any]) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Per-family corrections, sanitized exactly like the provider-wide defaults.

        Same trusted-field whitelist and redaction: an override can only narrow or correct
        metadata the provider itself is allowed to state, never introduce new keys."""
        raw = provider_cfg.get("catalog_model_overrides")
        if not isinstance(raw, dict):
            return ()
        overrides: list[tuple[str, dict[str, Any]]] = []
        for pattern, values in raw.items():
            key = str(pattern).strip().lower()
            if not key or not isinstance(values, dict):
                continue
            safe = {field: values[field] for field in TRUSTED_PROVIDER_MODEL_FIELDS if field in values}
            redacted = redact_sensitive_json(safe)
            if isinstance(redacted, dict) and redacted:
                overrides.append((key, redacted))
        # Longest pattern first: a specific id wins over the family prefix it belongs to.
        return tuple(sorted(overrides, key=lambda item: (-len(item[0]), item[0])))

    def _model_defaults_for_row(
        self,
        policy: ProviderCatalogPolicy,
        normalized_model: dict[str, Any],
    ) -> dict[str, Any]:
        model_id = str(normalized_model.get("id") or "").strip().lower()
        if not model_id or not policy.model_overrides:
            return policy.model_defaults
        for pattern, values in policy.model_overrides:
            if pattern in model_id:
                # Replace, not merge, for the fields the override names: a corrected
                # `supported_parameters` must be able to REMOVE what the default claimed.
                return {**policy.model_defaults, **values}
        return policy.model_defaults

    def _safe_model_id_patterns(self, provider_cfg: dict[str, Any], key: str) -> list[str]:
        raw = provider_cfg.get(key)
        if not isinstance(raw, list):
            return []
        return [str(pattern).strip().lower() for pattern in raw if str(pattern).strip()]

    def _requires_exact_model_allowlist(self, provider_cfg: dict[str, Any]) -> bool:
        provider_class = str(provider_cfg.get("provider_class") or provider_cfg.get("source_type") or "")
        return provider_class == "free_model" or bool(provider_cfg.get("require_model_id_allowlist"))

    def _catalog_row_with_normalized_limits(self, model: dict[str, Any]) -> dict[str, Any]:
        """Map provider limit aliases before defaults can mask explicit row values."""
        normalized = model
        if model.get("context_length") in (None, ""):
            for alias in CONTEXT_LENGTH_CATALOG_ALIASES:
                value = self._safe_optional_int(model.get(alias))
                if value is not None:
                    normalized = {**model, "context_length": value}
                    break

        top_provider = normalized.get("top_provider")
        nested_limits = dict(top_provider) if isinstance(top_provider, dict) else {}
        output_limit = self._safe_optional_int(normalized.get("max_completion_tokens"))
        if output_limit is not None and nested_limits.get("max_completion_tokens") in (None, ""):
            normalized = {
                **normalized,
                "top_provider": {**nested_limits, "max_completion_tokens": output_limit},
            }
        return normalized

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
        required: bool = False,
    ) -> bool:
        if not allowlist:
            return required
        return not self._catalog_row_matches_allowlist(model, allowlist, exact=exact)

    def _has_official_free_id_gate(self, policy: ProviderCatalogPolicy) -> bool:
        return bool(policy.official_free_ids or policy.official_free_id_suffixes)

    def _catalog_row_matches_official_free_id(
        self,
        model: dict[str, Any],
        policy: ProviderCatalogPolicy,
    ) -> bool:
        model_id = str(model.get("id") or "").strip().lower()
        if not model_id:
            return False
        if model_id in policy.official_free_ids:
            return True
        return any(model_id.endswith(suffix) for suffix in policy.official_free_id_suffixes)

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
        if self.source == "cloudflare":
            return self._fetch_cloudflare_catalog(provider_cfg, context)
        url = provider_cfg["catalog_url"]
        timeout = context.timeout_seconds
        headers = {"Accept": "application/json", **self.request_headers()}
        if not self._uses_anonymous_auth(provider_cfg):
            if self.source == "mistral":
                key, reason = context.resolve_credentials(self.source, provider_cfg)
                if not key:
                    return [], reason
                headers["Authorization"] = f"Bearer {key}"
            elif self.source not in {"openrouter", "nous"}:
                key, _reason = context.resolve_credentials(self.source, provider_cfg)
                if key:
                    headers["Authorization"] = f"Bearer {key}"
        # Anonymous catalog access deliberately skips this whole credential branch: a stray
        # environment variable must never silently add the provider's paid auth posture.
        # Separate, short connect timeout so an unreachable / black-holing provider
        # endpoint fails fast at connect instead of stalling for the full read timeout.
        connect_timeout = min(5.0, timeout)
        request_kwargs: dict[str, Any] = {
            "timeout": (connect_timeout, timeout),
            "headers": headers,
        }
        if self._uses_anonymous_auth(provider_cfg):
            request_kwargs["auth"] = suppress_implicit_http_auth
            # Requests re-applies .netrc credentials while following redirects, even when the
            # original request had an explicit no-op auth handler. Refuse redirects so an
            # anonymous lane cannot cross that transport boundary.
            request_kwargs["allow_redirects"] = False
        response = context.http_get(url, **request_kwargs)
        if response.status_code != 200:
            return [], f"HTTP {response.status_code}"
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return [], "invalid catalog shape"
        return data, None

    def _fetch_cloudflare_catalog(
        self,
        provider_cfg: dict[str, Any],
        context: CatalogFetchContext,
    ) -> tuple[list[dict[str, Any]], str | None]:
        base_catalog_url = self._cloudflare_url(
            provider_cfg,
            "catalog_url_template",
            context.resolve_provider_parameter,
        )
        if not base_catalog_url:
            return [], f"missing or invalid {CLOUDFLARE_ACCOUNT_ID_ENV}"
        key, reason = context.resolve_credentials(self.source, provider_cfg)
        if not key:
            return [], reason

        timeout = context.timeout_seconds
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
            **self.request_headers(),
        }
        models: list[dict[str, Any]] = []
        for page in range(1, CLOUDFLARE_MAX_CATALOG_PAGES + 1):
            query = urlencode(
                {
                    "page": page,
                    "per_page": CLOUDFLARE_CATALOG_PAGE_SIZE,
                    "include_deprecated": "false",
                }
            )
            response = context.http_get(
                f"{base_catalog_url}?{query}",
                timeout=(min(5.0, timeout), timeout),
                headers=headers,
            )
            if response.status_code != 200:
                return [], f"HTTP {response.status_code}"
            payload = response.json()
            result = payload.get("result") if isinstance(payload, dict) else None
            if not isinstance(result, list) or payload.get("success") is not True:
                return [], "invalid Cloudflare catalog shape"
            if not all(isinstance(row, dict) for row in result):
                return [], "invalid Cloudflare catalog row"
            models.extend(result)

            result_info = payload.get("result_info") if isinstance(payload.get("result_info"), dict) else {}
            total_count = self._safe_optional_int(result_info.get("total_count"))
            current_page = self._safe_optional_int(result_info.get("page")) or page
            per_page = self._safe_optional_int(result_info.get("per_page")) or len(result)
            if current_page != page or per_page <= 0:
                return [], "invalid Cloudflare catalog pagination"
            if total_count is not None and len(models) == total_count:
                return models, None
            # `total_count` is not a completeness proof: live, Cloudflare reports 306 while
            # serving 65 rows under `include_deprecated=false` (verified 11/09/2026). A page
            # shorter than `per_page` is the end of what the account can see; an empty first
            # page against a positive count is the one shape that still reads as truncated.
            if not result:
                if models:
                    return models, None
                return [], "incomplete Cloudflare catalog pagination"
            if len(result) < per_page:
                return models, None
        return [], "Cloudflare catalog pagination exceeded safety limit"
