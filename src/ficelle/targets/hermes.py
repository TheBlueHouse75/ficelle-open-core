from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ficelle.targets.base import (
    SmokeCheck,
    TargetExport,
    TargetExportContext,
    TargetInstallContext,
    TargetInstallResult,
    TargetKind,
    TargetSmokeContext,
)
from ficelle.use_cases.hermes_export import HermesExportBuilder


HermesCredentialResult = tuple[str | None, str | None, str] | None
HermesCredentialResolver = Callable[[str], HermesCredentialResult]


def _insert_hermes_paths(hermes_agent_dir: Path) -> None:
    candidates = (
        hermes_agent_dir,
        hermes_agent_dir / "venv" / "lib" / "python3.11" / "site-packages",
    )
    for candidate in candidates:
        if candidate.exists():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)


def hermes_runtime_resolver(
    provider: str,
    *,
    hermes_agent_dir: Path,
) -> HermesCredentialResult:
    """Resolve Nous credentials through Hermes when its optional runtime is importable."""
    if provider != "nous":
        return None
    _insert_hermes_paths(hermes_agent_dir)
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    try:
        credentials = resolve_nous_runtime_credentials(
            timeout_seconds=15.0,
            force_refresh=False,
        )
        api_key = str(credentials.get("api_key") or "").strip()
        base_url = str(credentials.get("base_url") or "").strip().rstrip("/")
        source = str(credentials.get("source") or "hermes_cli.auth")
        if api_key and base_url:
            return api_key, base_url, source
        return (
            None,
            base_url or None,
            "nous credentials resolved without usable api_key/base_url",
        )
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def build_hermes_runtime_resolver(hermes_agent_dir: Path) -> HermesCredentialResolver:
    """Bind the integration resolver to one Hermes installation without importing it."""
    return partial(hermes_runtime_resolver, hermes_agent_dir=hermes_agent_dir)


@dataclass(frozen=True)
class HermesTargetAdapter:
    export_builder: HermesExportBuilder
    target_id: str = "hermes"
    display_name: str = "Hermes"
    kind: TargetKind = "agent_host"
    default_base_url: str = "http://127.0.0.1:8646/v1"
    supports_plugin_install: bool = True
    supports_config_export: bool = True
    supports_health_check: bool = True

    def export_config(self, context: TargetExportContext) -> TargetExport:
        config = dict(context.config)
        hermes_export = self.export_builder.export(config)
        payload = hermes_export.payload
        return TargetExport(
            target_id=self.target_id,
            base_url=str(payload["base_url"]),
            models=tuple(hermes_export.picker_models),
            config_text=str(payload["yaml"]),
            presets=tuple(dict(preset) for preset in payload["presets"]),
            required_assets=("ficelle", "ficelle-compression"),
            verification_commands=(
                ("ficelle", "health"),
                ("ficelle", "models"),
                ("ficelle", "export"),
            ),
        )

    def install_assets(self, context: TargetInstallContext) -> TargetInstallResult:
        return TargetInstallResult(
            target_id=self.target_id,
            warnings=(
                "Run `ficelle-setup --target hermes` to install or preview Hermes assets.",
            ),
        )

    def smoke_checks(self, context: TargetSmokeContext) -> Sequence[SmokeCheck]:
        checks: list[SmokeCheck] = [
            SmokeCheck("service-health", ("ficelle", "health"), "Core service responds locally."),
            SmokeCheck("model-list", ("ficelle", "models"), "Core service exposes virtual models."),
            SmokeCheck("hermes-export", ("ficelle", "export"), "Hermes export renders without secrets."),
        ]
        if context.credentials_expected:
            checks.append(
                SmokeCheck(
                    "live-route",
                    ("curl", "-s", f"{context.base_url}/chat/completions"),
                    "Optional live route smoke when provider credentials are configured.",
                    requires_credentials=True,
                )
            )
        return tuple(checks)
