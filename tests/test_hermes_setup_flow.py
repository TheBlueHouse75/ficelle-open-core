"""Plugin-managed Hermes setup flow, exercised with every external effect faked.

No network, no real subprocess, no Hermes install and no home directory is touched: the flow takes
all of them through `Seams`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "src/ficelle/assets/hermes-plugin/ficelle"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sf = _load("ficelle_hermes_setup_flow_under_test", PLUGIN_DIR / "setup_flow.py")
pin_writer = _load("ficelle_write_bootstrap_pin", REPO_ROOT / "scripts/write-bootstrap-pin.py")

MIRROR_RAW = "https://raw.githubusercontent.com/TheBlueHouse75/ficelle-open-core"
BOOTSTRAP_BODY = b"print('bootstrap')\n"
PIN = sf.Pin(
    release_tag="v0.3.13",
    bootstrap_url=f"{MIRROR_RAW}/{'a' * 40}/scripts/bootstrap-ficelle.py",
    bootstrap_sha256=hashlib.sha256(BOOTSTRAP_BODY).hexdigest(),
    core_min=(0, 3, 0),
    core_max_exclusive=(0, 4, 0),
)
HEALTHY = {"status": "ok", "build": {"ficelle_version": "0.3.13"}}
DOCTOR_OK = json.dumps({"status": "ok", "auth": {"groq": {"invokable": True}}})
DOCTOR_NO_KEY = json.dumps({"status": "ok", "auth": {"groq": {"invokable": False}}})
ANSWER = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
# Endpoint and wire protocol are part of the selection: Hermes would otherwise keep a previous
# provider's URL or its `codex_responses` mode.
PERSISTED = ("persist", "ficelle/auto-orchestrator", "ficelle", "http://127.0.0.1:8646/v1", "chat_completions")
INSTALLED_CLI = "/opt/ficelle/bin/ficelle"
# Only a refused connection means the port is free; see `probe`.
REFUSED = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self, limit: int = -1) -> bytes:
        return self._body if limit < 0 else self._body[:limit]


class World:
    """Scripted outside world; `events` is the ordered record of every effect the flow attempted."""

    def __init__(self, tmp_path: Path, **overrides: Any) -> None:
        self.health: Any = HEALTHY
        self.chat: Any = ANSWER
        self.bootstrap_body = BOOTSTRAP_BODY
        self.bootstrap_rc = 0
        self.bootstrap_output = ""
        self.cli_results: dict[tuple[str, ...], tuple[int, str]] = {}
        self.python: str | None = "/opt/python/bin/python3.12"
        self.cli: str | None = INSTALLED_CLI
        self.pin: Any = PIN
        self.secret: str | None = None
        self.interactive = True
        self.consent = True
        self.environ = {"PATH": "/usr/bin", "HOME": str(tmp_path / "home")}
        self.events: list[tuple[Any, ...]] = []
        self.said: list[str] = []
        self.executed_bodies: list[bytes] = []
        self.smoke_authorization: str | None = None
        for key, value in overrides.items():
            setattr(self, key, value)

    def kinds(self, *kinds: str) -> list[tuple[Any, ...]]:
        return [event for event in self.events if event[0] in kinds]

    def seams(self) -> Any:
        def open_url(request: Any, *, timeout: float) -> _Response:
            self.events.append(("http", request.get_method(), request.full_url))
            if request.full_url.endswith("/health"):
                if isinstance(self.health, Exception):
                    raise self.health
                if isinstance(self.health, bytes):
                    return _Response(self.health)
                return _Response(json.dumps(self.health).encode())
            if request.full_url == PIN.bootstrap_url:
                return _Response(self.bootstrap_body)
            if request.full_url.endswith("/v1/chat/completions"):
                self.smoke_authorization = request.get_header("Authorization")
                if isinstance(self.chat, Exception):
                    raise self.chat
                return _Response(json.dumps(self.chat).encode())
            raise AssertionError(f"unexpected URL {request.full_url}")

        def run(argv: list[str], **kwargs: Any) -> Any:
            self.events.append(("run", list(argv), kwargs))
            if argv[0] == self.python and argv[1] == "-I":
                self.executed_bodies.append(Path(argv[2]).read_bytes())
                if self.bootstrap_rc == 0:
                    self.cli = INSTALLED_CLI  # the bootstrap is what puts the command on the machine
                return subprocess.CompletedProcess(argv, self.bootstrap_rc, self.bootstrap_output, "")
            default = (0, DOCTOR_OK if argv[1:] == ["doctor", "--json"] else "{}")
            rc, out = self.cli_results.get(tuple(argv[1:]), default)
            return subprocess.CompletedProcess(argv, rc, out, "")

        def confirm(question: str, description: str) -> bool:
            self.events.append(("confirm", question, description))
            return self.consent

        return sf.Seams(
            router_url="http://127.0.0.1:8646",
            open_url=open_url,
            run=run,
            confirm=confirm,
            is_interactive=lambda: self.interactive,
            find_base_python=lambda: self.python,
            find_cli=lambda: self.cli,
            load_pin=lambda: self.pin,
            get_secret=lambda name: self.secret,
            save_env_value=lambda name, value: self.events.append(("save_env", name, value)),
            persist_model=lambda *selection: self.events.append(("persist", *selection)),
            environ=self.environ,
            say=self.said.append,
        )

    def run_setup(self) -> Any:
        return sf.run_setup(self.seams())


@pytest.fixture()
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def _absent(tmp_path: Path, **overrides: Any) -> World:
    """No listener and no `ficelle` command: Core is not installed."""
    return World(tmp_path, **{"health": REFUSED, "cli": None, **overrides})


# --- reuse of an existing Core -------------------------------------------------------------------

def test_a_healthy_compatible_core_is_reused_without_any_bootstrap(world: World) -> None:
    result = world.run_setup()

    assert result.ready is True
    assert not world.kinds("confirm")
    assert not any(e[2] == PIN.bootstrap_url for e in world.kinds("http"))
    assert not world.executed_bodies
    assert world.kinds("save_env", "persist") == [("save_env", "FICELLE_API_KEY", "ficelle-local"), PERSISTED]


@pytest.mark.parametrize("version", ["0.4.0", "0.2.9"])
def test_a_core_outside_the_pinned_range_stops_before_any_change(world: World, version: str) -> None:
    world.health = {"status": "ok", "build": {"ficelle_version": version}}

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "probe")
    assert not world.kinds("confirm", "run", "save_env", "persist")


def test_a_healthy_core_without_its_command_on_path_still_gets_activated(world: World) -> None:
    world.cli = None

    assert world.run_setup().ready is True
    assert not world.kinds("run", "confirm")


@pytest.mark.parametrize(
    "health",
    [
        urllib.error.HTTPError("http://127.0.0.1:8646/health", 500, "boom", {}, None),
        b"<html>another service</html>",
        urllib.error.URLError(TimeoutError("timed out")),
        {"status": "starting"},
    ],
    ids=["http-error", "foreign-service", "timeout", "not-ok"],
)
def test_an_occupied_port_never_leads_to_an_install(world: World, health: Any) -> None:
    world.health = health
    world.cli = None

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "core-health")
    assert not world.kinds("confirm", "run", "save_env", "persist")
    assert not any(e[2] == PIN.bootstrap_url for e in world.kinds("http"))


# --- consent -------------------------------------------------------------------------------------

def test_declining_consent_changes_nothing(tmp_path: Path) -> None:
    world = _absent(tmp_path, consent=False)

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "consent")
    assert [e[0] for e in world.events if e[0] != "http"] == ["confirm"]
    # Hermes' full-screen prompt hides earlier output, so the disclosure travels with the question.
    assert "keeps running after Hermes exits" in world.kinds("confirm")[0][2]
    assert [e[2] for e in world.kinds("http")] == ["http://127.0.0.1:8646/health"]


def test_a_session_without_a_terminal_fails_without_prompting(tmp_path: Path) -> None:
    world = _absent(tmp_path, interactive=False)

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "consent")
    assert not world.kinds("confirm", "run", "save_env", "persist")


def test_a_copy_without_a_pin_never_offers_the_bootstrap(tmp_path: Path) -> None:
    world = _absent(tmp_path, pin=None)

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "probe")
    assert sf.INSTALL_DOCS_URL in result.resume
    assert not world.kinds("confirm", "run", "save_env", "persist")
    assert [e[2] for e in world.kinds("http")] == ["http://127.0.0.1:8646/health"]


# --- bootstrap integrity and isolation -----------------------------------------------------------

@pytest.mark.parametrize(
    "body",
    [b"print('tampered')\n", BOOTSTRAP_BODY + b"x" * sf.MAX_BOOTSTRAP_BYTES],
    ids=["digest-mismatch", "oversized"],
)
def test_a_bootstrap_that_fails_verification_is_never_executed(tmp_path: Path, body: bytes) -> None:
    world = _absent(tmp_path, bootstrap_body=body)

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "bootstrap")
    assert not world.kinds("run", "save_env", "persist")
    assert not world.executed_bodies


def test_the_bootstrap_runs_on_an_explicit_python_outside_hermes_and_with_no_secret(tmp_path: Path) -> None:
    world = _absent(tmp_path, cli_results={("health",): (1, "")})
    world.environ.update(
        OPENAI_API_KEY="sk-live-123456789",
        HERMES_HOME="/home/u/.hermes",
        FICELLE_API_KEY="ficelle-local",
        FICELLE_LICENSE_KEY="license-123",
        PYTHONPATH="/home/u/.hermes/hermes-agent",
        HTTPS_PROXY="http://proxy:3128",
        LC_ALL="C.UTF-8",
        FICELLE_HOME="/data/ficelle",
    )

    world.run_setup()

    (bootstrap_run,) = [e for e in world.kinds("run") if e[1][0] == world.python]
    argv, kwargs = bootstrap_run[1], bootstrap_run[2]
    assert argv[1] == "-I" and argv[3:] == ["--python", world.python]
    assert sys.executable not in argv
    assert not any(arg.startswith("--connector") for arg in argv)
    assert world.executed_bodies == [BOOTSTRAP_BODY]
    assert kwargs["env"] == {
        "PATH": "/usr/bin",
        "HOME": world.environ["HOME"],
        "HTTPS_PROXY": "http://proxy:3128",
        "LC_ALL": "C.UTF-8",
        "FICELLE_HOME": "/data/ficelle",
    }
    assert kwargs["stdin"] == subprocess.DEVNULL
    # A relative FICELLE_HOME must resolve from the user's directory, not a temporary one.
    assert "cwd" not in kwargs
    for run in world.kinds("run"):
        assert "OPENAI_API_KEY" not in run[2]["env"] and "HERMES_HOME" not in run[2]["env"]


def test_with_no_suitable_python_the_flow_stops_before_downloading(tmp_path: Path) -> None:
    world = _absent(tmp_path, python=None)

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "bootstrap")
    assert "3.11" in result.detail
    assert not any(e[2] == PIN.bootstrap_url for e in world.kinds("http"))
    assert not world.kinds("run", "save_env", "persist")


def test_installer_output_is_redacted_in_the_failure_report(tmp_path: Path) -> None:
    world = _absent(
        tmp_path,
        bootstrap_rc=1,
        bootstrap_output="pip failed\nFICELLE_LICENSE_KEY=license-123\nAuthorization: Bearer abc.def\n",
    )

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "bootstrap")
    assert "pip failed" in result.detail
    assert "license-123" not in result.detail and "abc.def" not in result.detail


# --- ordering: nothing in Hermes before a real request succeeded ---------------------------------

@pytest.mark.parametrize(
    ("overrides", "stage"),
    [
        ({"bootstrap_rc": 1}, "bootstrap"),
        ({"cli_results": {("health",): (1, "")}}, "core-health"),
        (
            {"cli_results": {("doctor", "--json"): (0, DOCTOR_NO_KEY)}, "chat": {"error": {"message": "no route"}}},
            "provider-credential",
        ),
    ],
    ids=["bootstrap-failed", "service-down", "no-usable-provider"],
)
def test_no_hermes_write_happens_before_core_is_verified(
    tmp_path: Path, overrides: dict[str, Any], stage: str
) -> None:
    world = _absent(tmp_path, **overrides)

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, stage)
    assert not world.kinds("save_env", "persist")


def test_the_listener_value_is_never_overwritten(world: World) -> None:
    world.secret = "user-chosen-value"

    assert world.run_setup().ready is True

    assert not world.kinds("save_env")
    assert world.kinds("persist") == [PERSISTED]
    # A Core bound beyond loopback demands its token: the test request sends what Hermes will.
    assert world.smoke_authorization == "Bearer user-chosen-value"


def test_a_core_that_answers_is_not_blocked_by_what_doctor_cannot_see(world: World) -> None:
    # `ficelle doctor` runs without the service's environment and can miss its provider keys.
    world.cli_results = {("doctor", "--json"): (0, DOCTOR_NO_KEY)}

    assert world.run_setup().ready is True


@pytest.mark.parametrize(
    "value",
    ["dummy", "abc", " None ", "xxxx", "sk-xxxxxxxx", "your_token_here"],
    ids=["placeholder", "too-short", "padded", "all-x", "prefixed-x", "your-here"],
)
def test_a_listener_value_hermes_rejects_is_reported_not_declared_ready(world: World, value: str) -> None:
    world.secret = value

    result = world.run_setup()

    assert (result.ready, result.stage) == (False, "hermes-activation")
    assert not world.kinds("save_env", "persist")


@pytest.mark.parametrize(
    "chat",
    [
        urllib.error.HTTPError("u", 503, "unavailable", {}, None),
        {"choices": [{"message": {"role": "assistant", "content": ""}}]},
        {"error": {"message": "no route"}},
    ],
    ids=["http-error", "empty-answer", "error-body"],
)
def test_ready_requires_a_real_answer_to_a_real_request(world: World, chat: Any) -> None:
    world.chat = chat

    result = world.run_setup()

    assert (result.ready, result.stage, result.headline) == (False, "request-smoke", "Not ready")
    # Hermes must not be left defaulting to a provider that cannot answer.
    assert not world.kinds("save_env", "persist")


def test_an_installed_core_that_is_down_is_repaired_not_reinstalled(tmp_path: Path) -> None:
    # Installed (the command exists) but not listening: no consent prompt, no second bootstrap.
    down = World(tmp_path, health=REFUSED, cli_results={("health",): (1, "")})

    result = down.run_setup()

    assert (result.ready, result.stage) == (False, "core-health")
    assert not down.kinds("confirm", "save_env", "persist") and not down.executed_bodies
    assert not any(e[2] == PIN.bootstrap_url for e in down.kinds("http"))


def test_a_fresh_install_runs_every_stage_in_order(tmp_path: Path) -> None:
    world = _absent(tmp_path)

    assert world.run_setup().ready is True

    order = [
        "confirm" if e[0] == "confirm"
        else "bootstrap" if e[0] == "run" and e[1][0] == world.python
        else "cli" if e[0] == "run"
        else "smoke" if e[0] == "http" and e[1] == "POST"
        else e[0]
        for e in world.events
        if not (e[0] == "http" and e[1] == "GET")
    ]
    assert order == ["confirm", "bootstrap", "cli", "cli", "cli", "smoke", "save_env", "persist"]


# --- interpreter selection -----------------------------------------------------------------------

def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_base_python_selection_skips_hermes_venvs_and_old_interpreters(tmp_path: Path) -> None:
    hermes_root = tmp_path / "hermes-agent"
    hermes_python = _executable(hermes_root / "venv/bin/python3.13")
    # The activated venv shadows the system interpreter under the very same name.
    other_venv = _executable(tmp_path / "project/.venv/bin/python3")
    too_old = _executable(tmp_path / "old/bin/python3.11")
    # Stock Debian/Ubuntu: `python -m venv` would fail halfway and leave a runtime with no pip.
    no_ensurepip = _executable(tmp_path / "debian/bin/python3")
    good = _executable(tmp_path / "good/bin/python3")
    reports = {
        str(other_venv): ([3, 12, 1], "/project/.venv", "/usr", True),
        str(too_old): ([3, 10, 9], "/usr", "/usr", True),
        str(no_ensurepip): ([3, 12, 3], "/usr", "/usr", False),
        str(good): ([3, 12, 4], "/usr", "/usr", True),
    }
    probed: list[str] = []

    def run(argv: list[str], **_kwargs: Any) -> Any:
        probed.append(argv[0])
        return subprocess.CompletedProcess(argv, 0, json.dumps(reports[argv[0]]), "")

    path = ":".join(str(p.parent) for p in (hermes_python, other_venv, too_old, no_ensurepip, good))
    found = sf.find_base_python(
        run, {"PATH": path}, [hermes_root.resolve()], hermes_python=str(hermes_python)
    )

    assert found == str(good)
    assert str(hermes_python) not in probed


def test_hermes_own_interpreter_is_never_selected_even_outside_its_install_root(tmp_path: Path) -> None:
    own = _executable(tmp_path / "bin/python3.12")

    found = sf.find_base_python(
        lambda argv, **_k: subprocess.CompletedProcess(argv, 0, json.dumps([[3, 12, 0], "/usr", "/usr", True]), ""),
        {"PATH": str(own.parent)},
        [],
        hermes_python=str(own),
    )

    assert found is None


# --- pin file and release tooling ----------------------------------------------------------------

def _write_pin(path: Path, **changes: Any) -> Path:
    data: dict[str, Any] = {
        "release_tag": PIN.release_tag,
        "bootstrap_url": PIN.bootstrap_url,
        "bootstrap_sha256": PIN.bootstrap_sha256,
        "core_version_range": {"min": "0.3.0", "max_exclusive": "0.4.0"},
    }
    data.update(changes)
    path.write_text(json.dumps(data))
    return path


def test_a_valid_pin_loads(tmp_path: Path) -> None:
    assert sf.load_pin_file(_write_pin(tmp_path / "pin.json")) == PIN


@pytest.mark.parametrize(
    "changes",
    [
        {"bootstrap_sha256": "abc"},
        {"bootstrap_url": PIN.bootstrap_url.replace("https://", "http://")},
        {"bootstrap_url": PIN.bootstrap_url.replace("raw.githubusercontent.com", "example.com")},
        {"bootstrap_url": f"{MIRROR_RAW}/v0.3.13/scripts/bootstrap-ficelle.py"},
        {"bootstrap_url": PIN.bootstrap_url.replace("bootstrap-ficelle.py", "other.py")},
        {"core_version_range": {"min": "latest", "max_exclusive": "0.4.0"}},
    ],
    ids=["short-digest", "plain-http", "foreign-host", "movable-tag", "other-file", "unparsable-range"],
)
def test_a_pin_that_is_not_trustworthy_counts_as_no_pin(tmp_path: Path, changes: dict[str, Any]) -> None:
    assert sf.load_pin_file(_write_pin(tmp_path / "pin.json", **changes)) is None
    assert sf.load_pin_file(tmp_path / "missing.json") is None


def test_the_release_tool_repeats_the_documented_pin_and_detects_drift(tmp_path: Path) -> None:
    commit, digest = "b" * 40, "c" * 64
    bootstrap = tmp_path / "bootstrap-ficelle.py"
    install_doc = tmp_path / "install.md"

    def document(commit: str, digest: str) -> None:
        url = f"{MIRROR_RAW}/{commit}/scripts/bootstrap-ficelle.py"
        # The local script is ahead of the published commit between releases: never hashed.
        bootstrap.write_text(f'"""Bootstrap pinned to public mirror release v1.4.7:\n\n    curl -fsSL {url}\n"""\n')
        install_doc.write_text(f"curl -fsSL {url} -o b.py\nprintf '%s  %s\\n' '{digest}' bootstrap-ficelle.py\n")

    document(commit, digest)
    pin_path = tmp_path / "bootstrap-pin.json"
    args = ["--bootstrap", str(bootstrap), "--install-doc", str(install_doc), "--pin", str(pin_path)]

    assert pin_writer.main([*args, "--check"]) == 1
    assert pin_writer.main(args) == 0
    assert pin_writer.main([*args, "--check"]) == 0

    pin = sf.load_pin_file(pin_path)
    assert pin is not None
    assert (pin.release_tag, pin.bootstrap_sha256) == ("v1.4.7", digest)
    assert pin.bootstrap_url == f"{MIRROR_RAW}/{commit}/scripts/bootstrap-ficelle.py"
    assert (pin.core_min, pin.core_max_exclusive) == ((1, 4, 0), (1, 5, 0))

    # A range widened on purpose at release is not staleness; a re-pinned commit or checksum is.
    assert pin_writer.main([*args, "--max-core", "2.0.0"]) == 0
    assert pin_writer.main([*args, "--check"]) == 0
    document("d" * 40, digest)
    assert pin_writer.main([*args, "--check"]) == 1
    document(commit, "e" * 64)
    assert pin_writer.main([*args, "--check"]) == 1


def test_a_committed_pin_matches_the_documented_bootstrap_pin() -> None:
    if not pin_writer.PIN_PATH.is_file():
        pytest.skip("no pin is committed between releases")
    assert pin_writer.main(["--check"]) == 0


# --- the provider hook ---------------------------------------------------------------------------

def _load_provider(monkeypatch: pytest.MonkeyPatch, *, hook_field: bool) -> tuple[str, Any]:
    registered: list[Any] = []
    fields: list[tuple[str, Any, Any]] = [
        ("name", str, ""), ("aliases", tuple, ()), ("env_vars", tuple, ()), ("display_name", str, ""),
        ("description", str, ""), ("signup_url", str, ""), ("base_url", str, ""), ("models_url", str, ""),
        ("fallback_models", tuple, ()),
    ]
    if hook_field:
        fields.append(("model_setup_flow", object, None))
    profile_cls = dataclasses.make_dataclass(
        "ProviderProfile", [(n, t, dataclasses.field(default=d)) for n, t, d in fields]
    )
    providers = ModuleType("providers")
    providers.register_provider = registered.append  # type: ignore[attr-defined]
    base = ModuleType("providers.base")
    base.ProviderProfile = profile_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "providers", providers)
    monkeypatch.setitem(sys.modules, "providers.base", base)

    def forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("loading the plugin must not reach the network or start a process")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    name = "_ficelle_provider_under_test"
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert len(registered) == 1
    return name, registered[0]


def test_loading_the_provider_has_no_side_effect_and_wires_the_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    name, profile = _load_provider(monkeypatch, hook_field=True)

    assert f"{name}.setup_flow" not in sys.modules
    assert callable(profile.model_setup_flow)


def test_a_hermes_without_the_hook_field_still_loads_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    name, profile = _load_provider(monkeypatch, hook_field=False)

    assert profile.name == "ficelle"
    assert not hasattr(profile, "model_setup_flow")


def test_the_hook_reaches_the_flow_with_the_provider_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name, profile = _load_provider(monkeypatch, hook_field=True)
    flow_module = importlib.import_module(f"{name}.setup_flow")
    world = World(tmp_path)
    urls: list[str] = []
    monkeypatch.setattr(flow_module, "default_seams", lambda url: (urls.append(url), world.seams())[1])

    assert profile.model_setup_flow({}, "(not set)", None) is True
    assert urls == ["http://127.0.0.1:8646"]

    # A failed flow reports why to Hermes' opt-in setup metric.
    failures: list[str] = []
    hermes_model_setup = ModuleType("hermes_cli.model_setup")
    hermes_model_setup.note_setup_failure = failures.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", ModuleType("hermes_cli"))
    monkeypatch.setitem(sys.modules, "hermes_cli.model_setup", hermes_model_setup)
    world.health = {"status": "ok", "build": {"ficelle_version": "9.9.9"}}
    assert profile.model_setup_flow({}, "(not set)", None) is False
    assert failures == ["probe"]
