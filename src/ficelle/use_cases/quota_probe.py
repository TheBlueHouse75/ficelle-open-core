from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class QuotaProbePorts:
    fresh_runtime_state: Callable[[], dict[str, Any]]
    safe_float: Callable[[Any, float], float]
    safe_string_list: Callable[[Any], list[str]]
    normalized_virtual_profiles: Callable[[dict[str, Any]], dict[str, dict[str, Any]]]
    canonical_virtual_model_id: Callable[[str], str]
    model_matches_profile_requirements: Callable[[dict[str, Any], dict[str, Any]], bool]
    quota_cooldown_matches_model: Callable[[str, dict[str, Any]], bool]
    find_quota_probe_model: Callable[[dict[str, Any], str, dict[str, Any]], dict[str, Any] | None]
    reserve_quota_probe: Callable[[dict[str, Any], str, dict[str, Any], dict[str, Any]], bool]
    probe_quota_cooldown: Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]
    virtual_models: set[str]
    now_seconds: Callable[[], float] = time.time


@dataclass(frozen=True)
class QuotaProbeStatePorts:
    safe_detail: Callable[[Any], str | None]
    safe_float: Callable[[Any, float], float]
    now_seconds: Callable[[], float]
    now_iso: Callable[[], str]
    update_state: Callable[[Callable[[dict[str, Any]], None], str | None], dict[str, Any]]


@dataclass(frozen=True)
class QuotaProbeModelPorts:
    safe_float: Callable[[Any, float], float]
    now_seconds: Callable[[], float]
    model_is_quarantined: Callable[[dict[str, Any], dict[str, Any]], bool]
    provider_on_cooldown: Callable[[str, dict[str, Any]], tuple[bool, str | None]]
    cooldown_key: Callable[[dict[str, Any]], str]
    normalized_free_access: Callable[[dict[str, Any]], dict[str, Any]]
    quota_cooldown_matches_model: Callable[[str, dict[str, Any]], bool]


def quota_probe_body(model: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    source = str(model.get("source") or "")
    raw_provider = providers.get(source)
    provider_cfg = raw_provider if isinstance(raw_provider, dict) else {}
    configured = provider_cfg.get("quota_probe_body")
    if isinstance(configured, dict):
        body = dict(configured)
    else:
        body = {
            "messages": [{"role": "user", "content": "Reply exactly: ficelle-quota-probe-ok"}],
            "temperature": 0,
            "max_tokens": 8,
        }
    body.pop("stream", None)
    return body


def find_quota_probe_model(
    catalog: dict[str, Any],
    key: str,
    state: dict[str, Any],
    *,
    ports: QuotaProbeModelPorts,
) -> dict[str, Any] | None:
    now_ts = ports.now_seconds()
    for model in catalog.get("models", []) or []:
        if not isinstance(model, dict):
            continue
        if not model.get("invokable"):
            continue
        if ports.model_is_quarantined(model, state):
            continue
        if ports.provider_on_cooldown(str(model.get("source") or ""), state)[0]:
            continue
        cooldowns = state.get("cooldowns") if isinstance(state.get("cooldowns"), dict) else {}
        model_cooldown = cooldowns.get(ports.cooldown_key(model)) if isinstance(cooldowns, dict) else None
        if isinstance(model_cooldown, dict) and ports.safe_float(model_cooldown.get("until"), 0.0) > now_ts:
            continue
        access = ports.normalized_free_access(model)
        if access.get("eligible") is not True or access.get("mode") != "quota_free":
            continue
        if ports.quota_cooldown_matches_model(key, model):
            return model
    return None


def record_quota_probe_result(
    state: dict[str, Any],
    key: str,
    model: dict[str, Any],
    status: str,
    reason: str,
    detail: str | None = None,
    *,
    ports: QuotaProbeStatePorts,
) -> None:
    results = state.setdefault("quota_probe_results", {})
    results[key] = {
        "status": status,
        "reason": reason,
        "detail": ports.safe_detail(detail),
        "probed_at": ports.now_iso(),
        "source": model.get("source"),
        "model_id": model.get("id"),
        "upstream_id": model.get("upstream_id"),
    }


def reserve_quota_probe_in_state(
    state: dict[str, Any],
    key: str,
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    ports: QuotaProbeStatePorts,
) -> bool:
    cooldowns = state.get("quota_cooldowns") if isinstance(state.get("quota_cooldowns"), dict) else {}
    current = cooldowns.get(key)
    if not isinstance(current, dict):
        return False
    lock_seconds = max(30, int(ports.safe_float(config.get("quota_probe_timeout_seconds"), 10.0)) + 5)
    reserved = dict(row)
    now_ts = ports.now_seconds()
    reserved["last_probe_started_at"] = ports.now_iso()
    reserved["next_probe_at"] = now_ts + lock_seconds
    reserved["probe_interval_seconds"] = lock_seconds
    reserved_probe = False

    def mutate(next_state: dict[str, Any]) -> None:
        nonlocal reserved_probe
        next_cooldowns = (
            next_state.get("quota_cooldowns") if isinstance(next_state.get("quota_cooldowns"), dict) else {}
        )
        if key not in next_cooldowns:
            return
        next_cooldowns[key] = reserved
        next_state["quota_cooldowns"] = next_cooldowns
        reserved_probe = True

    ports.update_state(mutate, reason="reserve_quota_probe")
    return reserved_probe


@dataclass(frozen=True)
class QuotaProbeRunner:
    ports: QuotaProbePorts

    def run_due_quota_probes(
        self,
        config: dict[str, Any],
        catalog: dict[str, Any],
        *,
        force: bool = False,
        source: str | None = None,
        keys: set[str] | None = None,
    ) -> dict[str, Any]:
        state = self.ports.fresh_runtime_state()
        cooldowns = state.get("quota_cooldowns") if isinstance(state.get("quota_cooldowns"), dict) else {}
        now_ts = self.ports.now_seconds()
        results: list[dict[str, Any]] = []
        for key, row in list(cooldowns.items()):
            if not isinstance(row, dict):
                continue
            key_text = str(key)
            if keys is not None and key_text not in keys:
                continue
            current_state = self.ports.fresh_runtime_state()
            current_cooldowns = (
                current_state.get("quota_cooldowns")
                if isinstance(current_state.get("quota_cooldowns"), dict)
                else {}
            )
            current_row = current_cooldowns.get(key)
            if not isinstance(current_row, dict):
                continue
            if source and str(current_row.get("source") or "") != source:
                continue
            until = self.ports.safe_float(current_row.get("until"), 0.0)
            next_probe_at = self.ports.safe_float(current_row.get("next_probe_at"), until)
            if not force and next_probe_at > now_ts:
                continue
            model = self.ports.find_quota_probe_model(catalog, key_text, current_state)
            if model is None:
                continue
            if not self.ports.reserve_quota_probe(current_state, key_text, current_row, config):
                continue
            results.append(self.ports.probe_quota_cooldown(key_text, model, config))
        return {"status": "ok", "probed_count": len(results), "results": results}

    def due_quota_probe_keys_for_request(
        self,
        requested_model: str,
        models: list[dict[str, Any]],
        config: dict[str, Any],
        state: dict[str, Any],
    ) -> list[str]:
        relevant_models = self._relevant_models_for_request(requested_model, models, config)
        if not relevant_models:
            return []

        now_ts = self.ports.now_seconds()
        due_keys: list[str] = []
        cooldowns = state.get("quota_cooldowns") if isinstance(state.get("quota_cooldowns"), dict) else {}
        for key, row in cooldowns.items():
            if not isinstance(row, dict):
                continue
            until = self.ports.safe_float(row.get("until"), 0.0)
            next_probe_at = self.ports.safe_float(row.get("next_probe_at"), until)
            if until <= 0 or next_probe_at > now_ts:
                continue
            if any(self.ports.quota_cooldown_matches_model(str(key), model) for model in relevant_models):
                due_keys.append(str(key))
        return due_keys

    def _relevant_models_for_request(
        self,
        requested_model: str,
        models: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        profiles = self.ports.normalized_virtual_profiles(config)
        canonical_requested_model = self.ports.canonical_virtual_model_id(requested_model)
        profile = profiles.get(requested_model) or profiles.get(canonical_requested_model)
        if (
            canonical_requested_model in self.ports.virtual_models
            or requested_model in self.ports.virtual_models
            or requested_model in profiles
        ):
            if profile and profile.get("mode") == "manual_order":
                return self._manual_profile_models(models, profile)
            return [
                model
                for model in models
                if not profile or self.ports.model_matches_profile_requirements(model, profile)
            ]
        return [
            model
            for model in models
            if requested_model in {model.get("id"), model.get("upstream_id")}
        ]

    def _manual_profile_models(
        self,
        models: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        by_id = {str(model.get("id")): model for model in models}
        relevant_models: list[dict[str, Any]] = []
        seen: set[str] = set()
        for model_id in self.ports.safe_string_list(profile.get("models")):
            model = by_id.get(model_id)
            if model is None or not self.ports.model_matches_profile_requirements(model, profile):
                continue
            relevant_models.append(model)
            seen.add(model_id)
        if profile.get("auto_tail", True):
            relevant_models.extend(
                model
                for model in models
                if str(model.get("id")) not in seen
                and self.ports.model_matches_profile_requirements(model, profile)
            )
        return relevant_models
