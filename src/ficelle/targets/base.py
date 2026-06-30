from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


TargetKind = Literal["agent_host", "native_app", "control_plane", "generic_client"]


@dataclass(frozen=True)
class TargetExportContext:
    config: Mapping[str, Any]


@dataclass(frozen=True)
class TargetInstallContext:
    target_home: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class TargetSmokeContext:
    base_url: str
    credentials_expected: bool = False


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    command: Sequence[str] = field(default_factory=tuple)
    description: str = ""
    requires_credentials: bool = False


@dataclass(frozen=True)
class TargetInstallResult:
    target_id: str
    installed_assets: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class TargetExport:
    target_id: str
    base_url: str
    models: Sequence[str]
    config_text: str | None = None
    config: Mapping[str, Any] | None = None
    presets: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    required_assets: Sequence[str] = field(default_factory=tuple)
    verification_commands: Sequence[Sequence[str]] = field(default_factory=tuple)
    redaction_status: str = "no_secrets"

    def legacy_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "base_url": self.base_url,
            "presets": [dict(preset) for preset in self.presets],
        }
        if self.config_text is not None:
            payload["yaml"] = self.config_text
        if self.config is not None:
            payload["config"] = dict(self.config)
        return payload


class TargetAdapter(Protocol):
    target_id: str
    display_name: str
    kind: TargetKind
    default_base_url: str
    supports_plugin_install: bool
    supports_config_export: bool
    supports_health_check: bool

    def export_config(self, context: TargetExportContext) -> TargetExport:
        ...

    def install_assets(self, context: TargetInstallContext) -> TargetInstallResult:
        ...

    def smoke_checks(self, context: TargetSmokeContext) -> Sequence[SmokeCheck]:
        ...


def target_base_url(config: Mapping[str, Any]) -> str:
    host = str(config.get("host") or "127.0.0.1")
    try:
        port = int(config.get("port") or 8646)
    except Exception:
        port = 8646
    return f"http://{host}:{port}/v1"


def visible_model_ids(
    virtual_models: Sequence[str],
    fusion_model_id: str | None,
    fusion_visible_in_model_list: Callable[[Mapping[str, Any]], bool] | None,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Virtual model ids, appending the fusion model id only when it is configured,
    not already listed, and visible in the model list for this config. Shared by the
    generic and OpenClaw target adapters so the rule lives in one place."""
    model_ids = tuple(virtual_models)
    if (
        fusion_model_id
        and fusion_model_id not in model_ids
        and fusion_visible_in_model_list is not None
        and fusion_visible_in_model_list(config)
    ):
        return (*model_ids, fusion_model_id)
    return model_ids
