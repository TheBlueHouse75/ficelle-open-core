from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ficelle.json_store import atomic_write_json, load_json


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class ConfigStore:
    config_path: Path
    defaults: dict[str, Any]
    normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def merged(self, existing: Any | None = None) -> dict[str, Any]:
        config = deep_merge(self.defaults, existing if isinstance(existing, dict) else {})
        config["allow_paid_fallback"] = False
        if self.normalize is not None:
            config = self.normalize(config)
        return config

    def load(self) -> dict[str, Any]:
        existing = load_json(self.config_path, {})
        config = self.merged(existing)
        if not self.config_path.exists():
            atomic_write_json(self.config_path, config)
        return config

    def save(self, config: dict[str, Any]) -> dict[str, Any]:
        normalized = self.merged(config)
        atomic_write_json(self.config_path, normalized)
        return normalized

    def update(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        existing = load_json(self.config_path, {})
        config = self.merged(existing)
        mutator(config)
        return self.save(config)
