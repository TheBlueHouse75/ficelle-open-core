from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

from ficelle.runtime_paths import RuntimePaths
from ficelle.targets import TargetExportContext, TargetSmokeContext
from ficelle.targets.openclaw import OpenClawTargetAdapter, openclaw_model_inputs, openclaw_model_ref


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENCLAW_TARGET_PATH = REPO_ROOT / "src/ficelle/targets/openclaw.py"
CANONICAL_MODELS = (
    "ficelle/auto-orchestrator",
    "ficelle/auto-tools",
    "ficelle/auto-json",
    "ficelle/auto-compression",
    "ficelle/auto-long",
    "ficelle/auto-fast",
    "ficelle/auto-reasoning",
    "ficelle/auto-multimodal",
    "ficelle/auto-vision",
    "ficelle/auto-video",
    "ficelle/auto-audio",
)


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_openclaw_target_adapter_exports_experimental_config_without_rewriting_core_ids():
    adapter = OpenClawTargetAdapter(virtual_models=CANONICAL_MODELS)

    export = adapter.export_config(TargetExportContext(config={"host": "127.0.0.7", "port": "8700"}))

    assert export.target_id == "openclaw"
    assert export.base_url == "http://127.0.0.7:8700/v1"
    assert export.models == CANONICAL_MODELS
    assert export.config is not None
    provider = export.config["models"]["providers"]["ficelle"]
    assert provider["baseUrl"] == "http://127.0.0.7:8700/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "ficelle-local"
    provider_model_ids = [model["id"] for model in provider["models"]]
    assert provider_model_ids == list(CANONICAL_MODELS)
    for provider_model in provider["models"]:
        assert "cost" not in provider_model
    inputs_by_id = {model["id"]: model["input"] for model in provider["models"]}
    assert inputs_by_id["ficelle/auto-tools"] == ["text"]
    assert inputs_by_id["ficelle/auto-multimodal"] == ["text"]
    assert inputs_by_id["ficelle/auto-vision"] == openclaw_model_inputs("ficelle/auto-vision")
    assert inputs_by_id["ficelle/auto-video"] == openclaw_model_inputs("ficelle/auto-video")
    assert inputs_by_id["ficelle/auto-audio"] == openclaw_model_inputs("ficelle/auto-audio")
    defaults = export.config["agents"]["defaults"]
    assert "model" not in defaults
    assert defaults["models"][openclaw_model_ref("ficelle", "ficelle/auto-fast")]["params"] == {
        "ficelleCoreModel": "ficelle/auto-fast"
    }
    assert export.redaction_status == "no_secrets"


def test_openclaw_custom_models_inherit_base_profile_capabilities():
    custom_vision = "ficelle/custom/visual-review"
    custom_reasoning = "ficelle/custom/deep-work"
    adapter = OpenClawTargetAdapter(
        virtual_models=("ficelle/auto-orchestrator", custom_vision, custom_reasoning),
        model_policy_ids={
            custom_vision: "ficelle/auto-vision",
            custom_reasoning: "ficelle/auto-reasoning",
        },
    )

    export = adapter.export_config(TargetExportContext(config={}))

    assert export.config is not None
    models = {
        model["id"]: model
        for model in export.config["models"]["providers"]["ficelle"]["models"]
    }
    assert models[custom_vision]["input"] == ["text", "image"]
    assert models[custom_vision]["reasoning"] is False
    assert models[custom_reasoning]["input"] == ["text"]
    assert models[custom_reasoning]["reasoning"] is True


def test_openclaw_target_adapter_has_no_hermes_imports_or_plugin_assumptions():
    modules = imported_modules(OPENCLAW_TARGET_PATH)
    forbidden_modules = ("ficelle.targets.hermes", "ficelle.use_cases.hermes_export")

    assert not any(module == forbidden or module.startswith(f"{forbidden}.") for module in modules for forbidden in forbidden_modules)

    adapter = OpenClawTargetAdapter(virtual_models=CANONICAL_MODELS)
    export = adapter.export_config(TargetExportContext(config={}))
    serialized = json.dumps(asdict(export), sort_keys=True).lower()

    assert "hermes" not in serialized
    assert "hermes_cli" not in serialized
    assert "ficelle-compression" not in serialized


def test_openclaw_target_export_runs_with_ficelle_home_outside_hermes_home(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    ficelle_home = tmp_path / "portable-ficelle-home"
    paths = RuntimePaths.from_env(
        environ={"HERMES_HOME": str(hermes_home), "FICELLE_HOME": str(ficelle_home)},
    )
    adapter = OpenClawTargetAdapter(virtual_models=CANONICAL_MODELS)

    export = adapter.export_config(TargetExportContext(config={"host": "localhost", "port": "8646"}))
    serialized = json.dumps(asdict(export), sort_keys=True).lower()

    assert paths.ficelle_home == ficelle_home
    assert paths.hermes_home == hermes_home
    assert str(hermes_home).lower() not in serialized
    assert str(ficelle_home).lower() not in serialized
    assert ".hermes" not in serialized


def test_openclaw_smoke_checks_do_not_require_hermes():
    adapter = OpenClawTargetAdapter(virtual_models=CANONICAL_MODELS)

    checks = adapter.smoke_checks(TargetSmokeContext(base_url="http://127.0.0.1:8646/v1"))

    commands = [" ".join(check.command) for check in checks]
    assert any(command == "ficelle health" for command in commands)
    assert any(command == "ficelle models" for command in commands)
    assert any(command == "openclaw models list" for command in commands)
    assert not any("hermes" in command.lower() for command in commands)


def test_openclaw_optional_live_smoke_is_self_contained():
    adapter = OpenClawTargetAdapter(virtual_models=CANONICAL_MODELS)

    checks = adapter.smoke_checks(
        TargetSmokeContext(base_url="http://127.0.0.1:8646/v1", credentials_expected=True)
    )

    live_check = next(check for check in checks if check.name == "openclaw-agent-smoke")
    assert live_check.requires_credentials is True
    assert live_check.command == (
        "openclaw",
        "infer",
        "model",
        "run",
        "--local",
        "--model",
        openclaw_model_ref("ficelle", "ficelle/auto-fast"),
        "--prompt",
        "Reply exactly: ficelle-openclaw-smoke-ok",
        "--json",
    )
