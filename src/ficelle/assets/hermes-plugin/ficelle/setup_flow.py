"""Explicit `hermes model` setup step for the Ficelle provider.

Hermes calls this only when the user selects Ficelle in `hermes model` or `hermes setup`. Importing
the plugin, listing models and sending requests never reach this module's effects. Every external
effect (HTTP, subprocess, prompt, Hermes helpers) goes through `Seams`, and Hermes is imported
lazily inside `default_seams`, so the flow is testable without Hermes.

The stages run in order and stop at the first failure, naming the stage and one resume action:
probe, consent, bootstrap, core-health, request-smoke, hermes-activation. The real request through
`ficelle/auto-orchestrator` is the only judge of whether Core can serve; when it fails, the
provider-credential check only explains why (no usable provider key means "Setup required").
Hermes is written last, so a failed setup never leaves it pointing at a provider that cannot answer. The flow keeps no
progress file: every run re-derives where it stands from the listener and the `ficelle` command.

Admission rules this file is written against: no write into the plugin directory (3), no `.env`,
`config.yaml`, `auth.json` or Ficelle credential-store access (11), no `shell=True` and no wait for
a person in a non-interactive session (12), and a single setup-time download, the pinned
bootstrap from the official mirror with digest verification (17).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

PROVIDER = "ficelle"
DEFAULT_MODEL = "ficelle/auto-orchestrator"
LISTENER_ENV_KEY = "FICELLE_API_KEY"
# Hermes rejects placeholders (`dummy`, `none`, anything under four characters); the loopback
# listener accepts any non-secret value, so this one only registers the provider.
LISTENER_VALUE = "ficelle-local"
API_MODE = "chat_completions"
# What Hermes treats as "no key configured" (`hermes_cli.auth.has_usable_secret`, same rule in
# 0.22.6 and 0.23.1). That helper lives in a module the catalog flags plugins for importing and
# Hermes exposes no public equivalent, so the rule is repeated in `listener_value_usable`.
_HERMES_PLACEHOLDER_VALUES = frozenset({
    "*", "**", "***", "changeme", "your_api_key", "your_api_key_here", "your-api-key",
    "placeholder", "example", "dummy", "null", "none",
})

PIN_FILENAME = "bootstrap-pin.json"
# A full commit SHA of the public mirror, never a tag: a tag can be moved after review.
OFFICIAL_BOOTSTRAP_RE = re.compile(
    r"https://raw\.githubusercontent\.com/TheBlueHouse75/ficelle-open-core/[0-9a-f]{40}/scripts/bootstrap-ficelle\.py"
)
INSTALL_DOCS_URL = "https://github.com/TheBlueHouse75/ficelle-open-core#quick-start"
MAX_BOOTSTRAP_BYTES = 1_000_000

# Names only: the bootstrap and the Ficelle CLI get no Hermes variable and no secret. FICELLE_HOME
# passes through because an explicitly selected Core home is the user's own choice; HERMES_HOME
# never selects it. The platform entries keep the service managers (systemd --user, Task
# Scheduler) usable from the child.
_ENV_ALLOWLIST = frozenset({
    "PATH", "PATHEXT", "HOME", "USER", "LOGNAME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "SYSTEMROOT", "COMSPEC", "TMPDIR", "TEMP", "TMP", "LANG", "LANGUAGE", "TZ",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "FICELLE_HOME",
})
_PYTHON_CANDIDATES = ("python3.13", "python3.12", "python3.11", "python3", "python")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)\s*[=:]\s*\S+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+\S+")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")

Version = tuple[int, int, int]


@dataclass(frozen=True)
class Pin:
    release_tag: str
    bootstrap_url: str
    bootstrap_sha256: str
    core_min: Version
    core_max_exclusive: Version


@dataclass(frozen=True)
class SetupResult:
    ready: bool
    stage: str
    headline: str
    detail: str = ""
    resume: str = ""


@dataclass
class Seams:
    router_url: str
    open_url: Callable[..., Any]
    run: Callable[..., Any]
    confirm: Callable[[str, str], bool]  # (question, description shown with it)
    is_interactive: Callable[[], bool]
    find_base_python: Callable[[], str | None]
    find_cli: Callable[[], str | None]
    load_pin: Callable[[], Pin | None]
    get_secret: Callable[[str], str | None]
    save_env_value: Callable[[str, str], Any]
    persist_model: Callable[..., Any]
    environ: Mapping[str, str]
    say: Callable[[str], None]


def parse_version(value: Any) -> Version | None:
    match = _VERSION_RE.fullmatch(str(value or "").strip())
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def load_pin_file(path: Path) -> Pin | None:
    """Read the release-written pin; anything malformed counts as no pin (fail closed)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        bounds = data["core_version_range"]
        pin = Pin(
            release_tag=str(data["release_tag"]),
            bootstrap_url=str(data["bootstrap_url"]),
            bootstrap_sha256=str(data["bootstrap_sha256"]).lower(),
            core_min=parse_version(bounds["min"]),  # type: ignore[arg-type]
            core_max_exclusive=parse_version(bounds["max_exclusive"]),  # type: ignore[arg-type]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if (
        pin.core_min is None
        or pin.core_max_exclusive is None
        or not re.fullmatch(r"[0-9a-f]{64}", pin.bootstrap_sha256)
        or not OFFICIAL_BOOTSTRAP_RE.fullmatch(pin.bootstrap_url)
    ):
        return None
    return pin


def listener_value_usable(value: str) -> bool:
    lowered = value.strip().lower()
    if len(lowered) < 4 or lowered in _HERMES_PLACEHOLDER_VALUES:
        return False
    if lowered.startswith("your_") and lowered.endswith("_here"):
        return False
    for prefix in ("sk-", "ghp_", "hf_"):
        tail = lowered[len(prefix):] if lowered.startswith(prefix) else ""
        if tail and set(tail) == {"x"}:
            return False
    letters = lowered.replace(" ", "").replace("-", "").replace("_", "")
    return set(letters) != {"x"}


def child_environment(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in environ.items()
        if key.upper() in _ENV_ALLOWLIST or key.upper().startswith("LC_")
    }


def redact(text: str, *, home: str | None = None, limit: int = 1200) -> str:
    tail = "\n".join(str(text or "").strip().splitlines()[-20:])
    tail = _BEARER_RE.sub("Bearer <redacted>", tail)
    tail = _SECRET_ASSIGNMENT_RE.sub(r"\1=<redacted>", tail)
    tail = _SK_RE.sub("<redacted>", tail)
    if home:
        tail = tail.replace(home, "~")
    return tail[-limit:]


def _fail(stage: str, headline: str, detail: str = "", resume: str = "") -> SetupResult:
    return SetupResult(False, stage, headline, detail, resume)


def _fetch(seams: Seams, request: urllib.request.Request, *, timeout: float, limit: int) -> bytes:
    with seams.open_url(request, timeout=timeout) as response:
        return response.read(limit + 1)


# --- probe ---------------------------------------------------------------------------------------

def probe(seams: Seams, pin: Pin | None) -> tuple[str, str | None]:
    """Classify the loopback listener: absent, unhealthy, incompatible or compatible.

    `GET /health` is the only unauthenticated Core endpoint that reports liveness and build
    identity (`build.ficelle_version`).
    """
    request = urllib.request.Request(seams.router_url.rstrip("/") + "/health")
    try:
        payload = json.loads(_fetch(seams, request, timeout=3.0, limit=65536))
    except (OSError, ValueError) as exc:
        # Only a refused connection proves the port is free. An HTTP error, a timeout or a
        # non-JSON body means something already listens there, and installing Core next to it
        # would fail on the port, so those are never treated as "not installed".
        reason = getattr(exc, "reason", exc)
        return ("absent" if isinstance(reason, ConnectionRefusedError) else "unhealthy"), None
    build = payload.get("build") if isinstance(payload, dict) else None
    version = build.get("ficelle_version") if isinstance(build, dict) else None
    parsed = parse_version(version)
    if not isinstance(payload, dict) or payload.get("status") != "ok" or parsed is None:
        return "unhealthy", None
    if pin is not None and not (pin.core_min <= parsed < pin.core_max_exclusive):
        return "incompatible", str(version)
    return "compatible", str(version)


# --- consent and bootstrap -----------------------------------------------------------------------

def _service_description() -> str:
    if sys.platform == "darwin":
        return "a LaunchAgent (com.ficelle.router) that starts at login"
    if sys.platform.startswith("win"):
        return "a per-user Scheduled Task that starts at logon"
    return "a systemd --user unit that starts at login"


def disclosure(pin: Pin, environ: Mapping[str, str]) -> str:
    home = environ.get("FICELLE_HOME") or "~/.ficelle"
    return "\n".join([
        "Ficelle Core is not running on this machine. Setting it up will:",
        f"  - download Ficelle's official bootstrap ({pin.release_tag}, SHA-256 verified) and run it;",
        "  - install Ficelle Core into its own runtime, ~/.local/share/ficelle/venv, never into Hermes;",
        f"  - keep Ficelle state and provider credentials in {home}, not in Hermes;",
        "  - link the `ficelle` commands into ~/.local/bin and note the state path in ~/.config/ficelle;",
        f"  - create {_service_description()}; it keeps running after Hermes exits;",
        "  - make Hermes call Ficelle at http://127.0.0.1:8646/v1 with a local, non-secret value.",
        "Ficelle is a separate local service you can use without Hermes. Removing this plugin does",
        "not uninstall it or delete its state. Ficelle does not provide unlimited free model access:",
        "routing uses the provider keys you add to Ficelle.",
    ])


def _install_core(seams: Seams, pin: Pin | None) -> SetupResult | None:
    """consent + bootstrap. Returns a failure, or None when Core has been installed."""
    if pin is None:
        return _fail(
            "probe",
            "Ficelle Core is not installed",
            "This copy of the plugin has no verified bootstrap, so it will not install Core itself.",
            f"Install Ficelle Core (see {INSTALL_DOCS_URL}), then select Ficelle again.",
        )
    if not seams.is_interactive():
        return _fail(
            "consent",
            "Not installed",
            "Installing Ficelle Core needs your confirmation and this session has no interactive "
            "terminal. Nothing was changed.",
            "Run `hermes model` from a terminal and select Ficelle.",
        )
    # Printed for the line prompt and passed along for Hermes' full-screen setup menu, which hides
    # everything printed before it: consent must never be asked without the disclosure in view.
    text = disclosure(pin, seams.environ)
    seams.say(text)
    if not seams.confirm("Install Ficelle Core now?", text):
        return _fail("consent", "Not installed", "Nothing was changed.", "Select Ficelle again to restart.")

    python = seams.find_base_python()
    if not python:
        return _fail(
            "bootstrap",
            "No suitable Python found",
            "The Ficelle bootstrap needs a Python 3.11+ outside the Hermes installation that can "
            "create virtual environments.",
            "Install Python 3.11 or newer (on Debian or Ubuntu, also `python3-venv`), then select "
            "Ficelle again.",
        )

    request = urllib.request.Request(pin.bootstrap_url, headers={"User-Agent": "ficelle-hermes-plugin"})
    try:
        body = _fetch(seams, request, timeout=30.0, limit=MAX_BOOTSTRAP_BYTES)
    except OSError as exc:
        return _fail(
            "bootstrap", "Bootstrap download failed", exc.__class__.__name__,
            "Check your network connection, then select Ficelle again.",
        )
    if len(body) > MAX_BOOTSTRAP_BYTES or not hmac.compare_digest(
        hashlib.sha256(body).hexdigest(), pin.bootstrap_sha256
    ):
        return _fail(
            "bootstrap",
            "Bootstrap integrity check failed",
            "The downloaded bootstrap does not match the pinned SHA-256. Nothing was executed.",
            f"Install Ficelle Core manually (see {INSTALL_DOCS_URL}).",
        )

    seams.say("Running the Ficelle bootstrap. This can take a few minutes.")
    with tempfile.TemporaryDirectory(prefix="ficelle-bootstrap-") as scratch:
        script = Path(scratch) / "bootstrap-ficelle.py"
        script.write_bytes(body)
        try:
            # No connector option: Hermes activation stays with this flow. `-I` keeps the script
            # directory, the working directory and PYTHON* variables out of the child's import
            # path, so the caller's working directory is kept: the bootstrap resolves a relative
            # FICELLE_HOME from it, and a temporary one would point Core at a deleted directory.
            completed = seams.run(
                [python, "-I", str(script), "--python", python],
                env=child_environment(seams.environ),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _fail("bootstrap", "Bootstrap failed", exc.__class__.__name__, "Select Ficelle again.")
    if completed.returncode != 0:
        output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        return _fail(
            "bootstrap",
            "Bootstrap failed",
            redact(output, home=seams.environ.get("HOME")),
            "Fix the error above, then select Ficelle again.",
        )
    return None


# --- Core checks ---------------------------------------------------------------------------------

def _cli(seams: Seams, cli: str, *args: str) -> Any:
    return seams.run(
        [cli, *args],
        env=child_environment(seams.environ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def _core_health(seams: Seams) -> SetupResult | None:
    repair = "Run `ficelle restart` (or `ficelle doctor --text`), then select Ficelle again."
    cli = seams.find_cli()
    if not cli:
        return _fail("core-health", "Service repair needed", "The `ficelle` command was not found.", repair)
    for args in (("doctor", "--json"), ("health",), ("models",)):
        try:
            completed = _cli(seams, cli, *args)
        except (OSError, subprocess.SubprocessError) as exc:
            return _fail("core-health", "Service repair needed", f"ficelle {args[0]}: {exc.__class__.__name__}", repair)
        if completed.returncode != 0:
            # `doctor --json` answers with its whole report, which buries the resume action.
            stdout = "" if args[0] == "doctor" else completed.stdout or ""
            output = f"{stdout}\n{completed.stderr or ''}"
            detail = redact(output, home=seams.environ.get("HOME"))
            return _fail("core-health", "Service repair needed", f"`ficelle {args[0]}` failed.\n{detail}".strip(), repair)
    return None


def _provider_credential(seams: Seams) -> SetupResult | None:
    """Explain a failed request: "Setup required" when Ficelle reports no usable provider.

    Only a diagnosis, never a gate. `ficelle doctor` computes provider auth in its own process,
    which runs here without the service's environment, so it can miss keys the running service
    has; the real request is what decides. Anything unknown returns None and keeps that verdict.
    """
    cli = seams.find_cli()
    if not cli:
        return None
    try:
        completed = _cli(seams, cli, "doctor", "--json")
        auth = json.loads(completed.stdout or "").get("auth")
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None
    if not isinstance(auth, dict) or any(
        isinstance(row, dict) and row.get("invokable") for row in auth.values()
    ):
        return None
    return _fail(
        "provider-credential",
        "Setup required",
        "Ficelle is running but no provider can serve a request yet. Provider credentials are "
        "stored by Ficelle itself; this plugin never asks for them.",
        "Add a provider key with `ficelle set-key <provider>` (`ficelle doctor --text` lists them), "
        "then select Ficelle again.",
    )


# --- smoke and Hermes activation -----------------------------------------------------------------

def _activate_hermes(seams: Seams) -> SetupResult | None:
    try:
        existing = seams.get_secret(LISTENER_ENV_KEY)
        if existing and not listener_value_usable(existing):
            # Hermes drops a provider whose key looks like a placeholder, so the request above
            # would have succeeded while Hermes itself could not use Ficelle. The value is the
            # user's, so it is reported, not replaced.
            return _fail(
                "hermes-activation",
                "Activation failed",
                f"{LISTENER_ENV_KEY} is set to a value Hermes rejects as a placeholder.",
                f"Set {LISTENER_ENV_KEY} to any other non-secret value, such as {LISTENER_VALUE}, "
                "then select Ficelle again.",
            )
        if not existing:
            seams.save_env_value(LISTENER_ENV_KEY, LISTENER_VALUE)
        # Endpoint and wire protocol are written explicitly: Hermes otherwise keeps the previous
        # provider's `model.base_url` and `model.api_mode`, and would send Ficelle's model names
        # to that old server or call `/v1/responses`, which Ficelle does not serve.
        seams.persist_model(DEFAULT_MODEL, PROVIDER, seams.router_url.rstrip("/") + "/v1", API_MODE)
    except Exception as exc:  # Hermes helpers raise their own types; none may escape the flow
        return _fail(
            "hermes-activation", "Activation failed", f"{exc.__class__.__name__}: {exc}",
            "Select Ficelle again; Ficelle Core stays installed and healthy.",
        )
    return None


def _router_error(exc: urllib.error.HTTPError) -> str:
    """Ficelle's own explanation of an error response, such as every provider being in cooldown."""
    try:
        message = json.loads(exc.read(100_000))["error"]["message"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return ""
    return f" ({redact(str(message), limit=200)})"


def _request_smoke(seams: Seams) -> SetupResult | None:
    body = json.dumps({
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        listener_value = seams.get_secret(LISTENER_ENV_KEY)
    except Exception:  # an unreadable secret scope only matters if the listener demands a token
        listener_value = None
    if listener_value:
        # A Core bound beyond loopback requires its token even from 127.0.0.1; Hermes would send
        # this value, so the request does too.
        headers["Authorization"] = f"Bearer {listener_value}"
    request = urllib.request.Request(
        seams.router_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    resume = "Run `ficelle doctor --text`, fix what it reports, then select Ficelle again."
    try:
        payload = json.loads(_fetch(seams, request, timeout=180.0, limit=1_000_000))
        message = payload["choices"][0]["message"]
        answered = bool(message.get("content") or message.get("tool_calls"))
    except urllib.error.HTTPError as exc:
        reason = f"HTTP {exc.code}{_router_error(exc)}"
        return _fail("request-smoke", "Not ready", f"The test request failed: {reason}.", resume)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        return _fail("request-smoke", "Not ready", f"The test request failed: {exc.__class__.__name__}.", resume)
    if not answered:
        return _fail("request-smoke", "Not ready", "The test request returned an empty answer.", resume)
    return None


# --- orchestration -------------------------------------------------------------------------------

def run_setup(seams: Seams) -> SetupResult:
    pin = seams.load_pin()
    verdict, version = probe(seams, pin)

    if verdict == "incompatible":
        return _fail(
            "probe",
            "Incompatible Ficelle Core",
            f"Ficelle Core {version} is running, outside the range this plugin supports.",
            "Run `ficelle update --install`, then select Ficelle again.",
        )
    if verdict == "unhealthy":
        return _fail(
            "core-health",
            "Service repair needed",
            "Something answers on the Ficelle port but is not a healthy Ficelle Core.",
            "If Ficelle is installed, run `ficelle restart` (or `ficelle doctor --text`); otherwise "
            "free port 8646. Then select Ficelle again.",
        )

    if verdict == "absent":
        # An installed Core that does not answer is repaired, never replaced by a second runtime.
        # The `ficelle` command is the evidence of an install, so nothing is remembered between runs.
        if seams.find_cli() is None:
            failure = _install_core(seams, pin)
            if failure:
                return failure
        failure = _core_health(seams)
        if failure:
            return failure
    else:
        seams.say(f"Ficelle Core {version} is already running; reusing it.")

    failure = _request_smoke(seams)
    if failure:
        return _provider_credential(seams) or failure
    failure = _activate_hermes(seams)
    if failure:
        return failure
    return SetupResult(
        True, "hermes-activation", "Ready",
        f"A real request succeeded and Hermes now defaults to {DEFAULT_MODEL}.",
        "Optional: `ficelle connectors install hermes` adds the compression plugin and prepares the "
        "recommended Hermes configuration.",
    )


def report(seams: Seams, result: SetupResult) -> None:
    seams.say(result.headline if result.ready else f"{result.headline} (stage: {result.stage})")
    for line in (result.detail, result.resume):
        if line:
            seams.say(line)


# --- real wiring ---------------------------------------------------------------------------------

def _excluded_python_roots(environ: Mapping[str, str]) -> list[Path]:
    roots = []
    if sys.prefix != sys.base_prefix:
        roots.append(Path(sys.prefix))  # Hermes' own virtualenv
    try:
        # The default home has no HERMES_HOME variable, and it holds Hermes' managed Pythons.
        from hermes_constants import get_hermes_home

        roots.append(Path(get_hermes_home()))
    except ImportError:
        roots.append(Path(environ.get("HERMES_HOME") or "~/.hermes").expanduser())
    try:
        import importlib.util

        spec = importlib.util.find_spec("hermes_cli")
        if spec and spec.origin:
            roots.append(Path(spec.origin).resolve().parent.parent)
    except (ImportError, ValueError):
        pass
    return [root.resolve() for root in roots]


def find_base_python(
    run: Callable[..., Any],
    environ: Mapping[str, str],
    excluded_roots: list[Path],
    hermes_python: str,
) -> str | None:
    """A Python 3.11+ that can create virtualenvs, is not one, and is outside the Hermes installation.

    Never Hermes' own interpreter (`sys.executable`). A venv is rejected by asking the candidate
    for `sys.prefix` and `sys.base_prefix`, because a venv's `python3` is often a symlink into a
    system install and its real path alone proves nothing; the same symlink is why the
    interpreter comparison uses the unresolved path.
    """
    # Debian and Ubuntu ship Python without `ensurepip`: `python -m venv` then fails halfway and
    # leaves a runtime with no pip. The bootstrap can still build the runtime with `uv`.
    probe_code = (
        "import importlib.util as u,json,sys;"
        "print(json.dumps([list(sys.version_info[:3]),sys.prefix,sys.base_prefix,"
        "bool(u.find_spec('ensurepip') and u.find_spec('venv'))]))"
    )
    has_uv = shutil.which("uv", path=environ.get("PATH")) is not None
    suffix = ".exe" if os.name == "nt" else ""
    # Every PATH entry is examined: an activated virtualenv usually shadows the system Python
    # under the same names, and stopping at the first hit per name would find only the venv.
    candidates = dict.fromkeys(
        os.path.join(directory, name + suffix)
        for directory in (environ.get("PATH") or "").split(os.pathsep) if directory
        for name in _PYTHON_CANDIDATES
    )
    for found in candidates:
        if not (os.path.isfile(found) and os.access(found, os.X_OK)):
            continue
        absolute = Path(os.path.abspath(found))
        if absolute == Path(os.path.abspath(hermes_python)) or any(
            absolute.is_relative_to(root) or absolute.resolve().is_relative_to(root)
            for root in excluded_roots
        ):
            continue
        try:
            completed = run(
                [found, "-I", "-c", probe_code],
                env=child_environment(environ), stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=15, check=False,
            )
            version, prefix, base_prefix, makes_venvs = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        if completed.returncode != 0 or tuple(version) < (3, 11) or not (makes_venvs or has_uv):
            continue
        if prefix != base_prefix or any(Path(prefix).resolve().is_relative_to(root) for root in excluded_roots):
            continue
        return found
    return None


def find_ficelle_cli(environ: Mapping[str, str]) -> str | None:
    """The `ficelle` command: PATH first, then the two places the bootstrap puts it."""
    on_path = shutil.which("ficelle", path=environ.get("PATH"))
    if on_path:
        return on_path
    home = Path(environ.get("HOME") or Path.home())
    suffix = ".exe" if os.name == "nt" else ""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    for candidate in (
        home / ".local" / "bin" / f"ficelle{suffix}",
        home / ".local" / "share" / "ficelle" / "venv" / bin_dir / f"ficelle{suffix}",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def default_seams(router_url: str) -> Seams:
    """Wire the real effects. Hermes helpers are imported at call time, never at import time."""
    environ = os.environ

    def get_secret(name: str) -> str | None:
        from agent.secret_scope import get_secret as hermes_get_secret

        return hermes_get_secret(name)

    def save_env_value(name: str, value: str) -> Any:
        from hermes_cli.config import save_env_value_secure

        return save_env_value_secure(name, value)

    def persist_model(model: str, provider: str, base_url: str, api_mode: str) -> Any:
        from hermes_cli.model_setup import persist_model_selection

        return persist_model_selection(model, provider, base_url=base_url, api_mode=api_mode)

    def confirm(question: str, description: str) -> bool:
        from hermes_cli.setup import prompt_yes_no

        return prompt_yes_no(question, default=False, description=description)

    return Seams(
        router_url=router_url,
        open_url=urllib.request.urlopen,
        run=subprocess.run,
        confirm=confirm,
        is_interactive=lambda: (
            sys.stdin.isatty()
            and sys.stdout.isatty()
            and environ.get("HERMES_NONINTERACTIVE", "").strip().lower() not in {"1", "true", "yes", "on"}
        ),
        find_base_python=lambda: find_base_python(
            subprocess.run, environ, _excluded_python_roots(environ), sys.executable
        ),
        find_cli=lambda: find_ficelle_cli(environ),
        load_pin=lambda: load_pin_file(Path(__file__).with_name(PIN_FILENAME)),
        get_secret=get_secret,
        save_env_value=save_env_value,
        persist_model=persist_model,
        environ=environ,
        say=print,
    )


def model_setup_flow(config: Any, current_model: str, args: Any, *, router_url: str) -> bool:
    """The `ProviderProfile.model_setup_flow` body; Hermes ignores the return value."""
    seams = default_seams(router_url)
    result = run_setup(seams)
    report(seams, result)
    if not result.ready:
        from hermes_cli.model_setup import note_setup_failure

        note_setup_failure(result.stage.replace("-", "_"))
    return result.ready
