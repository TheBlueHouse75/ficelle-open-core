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
            inline_limit = max(0, self.ports.safe_int(config.get("quota_inline_probe_limit"), 1))
            if inline_limit:
                self.ports.run_due_quota_probes(config, catalog, keys=set(due_probe_keys[:inline_limit]))
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
