from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ficelle.domain_models import CatalogModel, SelectionPurpose, SelectionResult
from ficelle.selection import SelectionPolicy, select_models_result_from_typed_rows, typed_catalog_rows


@dataclass(frozen=True)
class ModelSelectionPorts:
    load_config: Callable[[], dict[str, Any]]
    fresh_runtime_state: Callable[[], dict[str, Any]]
    due_quota_probe_keys_for_request: Callable[
        [str, list[dict[str, Any]], dict[str, Any], dict[str, Any]],
        list[str],
    ]
    run_due_quota_probes: Callable[..., dict[str, Any]]
    schedule_quota_probes: Callable[..., None]
    safe_int: Callable[[Any, int], int]
    apply_verified_capability_ttl: Callable[[dict[str, Any]], None]
    apply_route_on_capability_reference: Callable[[dict[str, Any]], None]
    selection_policy: Callable[[], SelectionPolicy]


@dataclass(frozen=True)
class ModelSelectionRunner:
    ports: ModelSelectionPorts

    def select_models_result(
        self,
        requested_model: str,
        catalog: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        purpose: SelectionPurpose = "route",
    ) -> SelectionResult:
        if config is None:
            config = self.ports.load_config()
        typed_rows = typed_catalog_rows(catalog)
        models = [raw for _model, raw in typed_rows if raw.get("invokable")]
        state = self.ports.fresh_runtime_state()
        due_probe_keys = self.ports.due_quota_probe_keys_for_request(requested_model, models, config, state)
        if due_probe_keys:
            # The inline probe is load-bearing, not just eager: clearing the cooldown here is
            # what puts the model back in *this* request's candidate list. Moving it wholesale
            # to the background was tried and reverted — it turns "recovered, answered" into
            # `no_available_model` whenever a profile's only candidates are quota-cooled.
            # So the first key still runs inline, and only the keys the limit already dropped —
            # which used to be skipped entirely, waiting for some later request to re-trigger
            # them — now get probed in the background instead of being forgotten.
            inline_limit = max(0, self.ports.safe_int(config.get("quota_inline_probe_limit"), 1))
            if inline_limit:
                self.ports.run_due_quota_probes(config, catalog, keys=set(due_probe_keys[:inline_limit]))
            deferred = set(due_probe_keys[inline_limit:])
            if deferred:
                self.ports.schedule_quota_probes(config, catalog, keys=deferred)
            state = self.ports.fresh_runtime_state()
        return self.select_models_result_from_state(
            requested_model,
            catalog,
            config,
            state,
            purpose=purpose,
        )

    def select_models_result_from_state(
        self,
        requested_model: str,
        catalog: dict[str, Any],
        config: dict[str, Any],
        state: dict[str, Any],
        *,
        purpose: SelectionPurpose = "route",
    ) -> SelectionResult:
        self.ports.apply_verified_capability_ttl(config)
        self.ports.apply_route_on_capability_reference(config)
        return self.select_models_result_from_typed_rows(
            requested_model,
            typed_catalog_rows(catalog),
            config,
            state,
            purpose=purpose,
        )

    def select_models_result_from_typed_rows(
        self,
        requested_model: str,
        typed_rows: list[tuple[CatalogModel, dict[str, Any]]],
        config: dict[str, Any],
        state: dict[str, Any],
        *,
        purpose: SelectionPurpose = "route",
    ) -> SelectionResult:
        return select_models_result_from_typed_rows(
            requested_model,
            typed_rows,
            config,
            state,
            self.ports.selection_policy(),
            purpose=purpose,
        )
