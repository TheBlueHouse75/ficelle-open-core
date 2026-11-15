#!/usr/bin/env python3
"""Install Ficelle Core, and optionally Ficelle Pro, without cloning the repository.

This script is intentionally stdlib-only. Bootstrap pinned to public mirror release v0.3.13:

    curl -fsSL https://raw.githubusercontent.com/TheBlueHouse75/ficelle-open-core/350b724e17d1b7d0d0d95bf41d8938e7a2f171c9/scripts/bootstrap-ficelle.py | python3

Without a license key it installs the versioned open Core from the public GitHub
release. With ``FICELLE_LICENSE_KEY`` (or the legacy ``FICELLE_INSTALL_TOKEN``)
it also downloads the licensed Pro wheel, installs it without consulting PyPI,
activates the entitlement, and verifies both packages. It never prints license keys
or provider secrets.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Iterator, Mapping, Sequence

CORE_VERSION = "0.3.14"
DEFAULT_CORE_WHEEL_URL = (
    "https://github.com/TheBlueHouse75/ficelle-open-core/releases/download/"
    f"v{CORE_VERSION}/ficelle_router-{CORE_VERSION}-py3-none-any.whl"
)
DEFAULT_CORE_SHA256 = (
    "be8838e4cfe327577889d7128b4031df32067e5e4b1cc8f5f80da194df39bba3"
)
DEFAULT_WHEEL_URL = f"https://install.ficelle.ai/api/releases/{CORE_VERSION}/wheel"
PRO_RUNTIME_REQUIREMENTS = ("cryptography>=42",)
CORE_RUNTIME_REQUIREMENTS = ("cryptography>=42", "packaging>=24", "requests>=2.31")
DEFAULT_HERMES_HOME = Path.home() / ".hermes"
CONNECTORS = ("hermes", "openclaw")
FALLBACK_WHEEL_FILENAME = "ficelle_pro-0-py3-none-any.whl"
MAX_ATTESTATION_BYTES = 16 * 1024
# Deliberately accepts only the PEP 440 subset emitted by Ficelle's release service.
# Any valid-but-unrecognized form safely falls back to FALLBACK_WHEEL_FILENAME.
_WHEEL_FILENAME_PATTERN = re.compile(
    r"^[A-Za-z0-9_.]+-"
    r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._][A-Za-z0-9]+)*)?"
    r"(?:-[0-9][A-Za-z0-9_.]*)?"
    r"-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl$"
)
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class BootstrapOptions:
    core_wheel_url: str
    core_sha256: str | None
    wheel_url: str
    license_key: str | None
    sha256: str | None
    python: str | None
    venv: Path
    connectors: tuple[str, ...]
    ficelle_home: Path
    hermes_home: Path
    backup_existing: bool
    skip_service: bool
    skip_smoke: bool
    dry_run: bool
    keep_wheel: bool
    ficelle_home_explicit: bool = True


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


def redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return url
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "[REDACTED]", parsed.fragment))


def command_env(options: BootstrapOptions) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FICELLE_RUNTIME_DIR", None)
    if options.ficelle_home_explicit:
        env["FICELLE_HOME"] = str(options.ficelle_home.absolute())
    else:
        env.pop("FICELLE_HOME", None)
    env.pop("HERMES_HOME", None)
    return env


def run_command(command: Sequence[str], *, dry_run: bool, env: Mapping[str, str] | None = None, cwd: str | None = None) -> CommandResult:
    env_hint = ""
    if env:
        hints = [
            f"{name}={env[name]}"
            for name in ("FICELLE_HOME",)
            if env.get(name)
        ]
        if hints:
            env_hint = " " + " ".join(hints)
    printable = " ".join(command)
    if dry_run:
        print(f"DRY RUN:{env_hint} {printable}")
        return CommandResult(list(command), 0)
    print(f"RUN:{env_hint} {printable}")
    completed = subprocess.run(list(command), text=True, capture_output=True, env=dict(env) if env else None, cwd=cwd)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return CommandResult(list(command), completed.returncode, completed.stdout or "", completed.stderr or "")


@contextmanager
def isolated_command_cwd() -> Iterator[str]:
    # Package names and imports must not bind to files in the caller's directory.
    with tempfile.TemporaryDirectory(prefix="ficelle-pip-") as cwd:
        Path(cwd).chmod(0o700)
        yield cwd


def run_isolated_command(command: Sequence[str], *, dry_run: bool, env: Mapping[str, str] | None = None) -> CommandResult:
    with isolated_command_cwd() as cwd:
        return run_command(command, dry_run=dry_run, env=env, cwd=cwd)


def run_uv_command(
    executable: str, group: Sequence[str], arguments: Sequence[str], *,
    dry_run: bool, env: Mapping[str, str],
) -> CommandResult:
    # uv also discovers config in parents of the empty cwd (including shared /tmp).
    uv_env = dict(env)
    uv_env["UV_NO_CONFIG"] = "1"
    return run_isolated_command(
        [str(Path(executable).absolute()), *group, "--no-config", *arguments],
        dry_run=dry_run, env=uv_env,
    )


def ensure_success(result: CommandResult, *, action: str) -> None:
    if result.returncode != 0:
        raise SystemExit(f"{action} failed with exit code {result.returncode}")


def pip_is_unavailable(result: CommandResult) -> bool:
    combined = f"{result.stdout}\n{result.stderr}"
    return "No module named pip" in combined or "No module named 'pip'" in combined


def run_python_probe(python: Path, code: str) -> CommandResult:
    command = [str(python.absolute()), "-c", code]
    try:
        with isolated_command_cwd() as cwd:
            completed = subprocess.run(command, text=True, capture_output=True, cwd=cwd)
    except OSError as exc:
        return CommandResult(command, 1, stderr=f"{type(exc).__name__}: {exc}")
    return CommandResult(
        command,
        completed.returncode,
        completed.stdout or "",
        completed.stderr or "",
    )


def python_version_result(python: Path) -> CommandResult:
    return run_python_probe(
        python,
        "import sys; "
        "print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); "
        "raise SystemExit(0 if sys.version_info >= (3, 11) else 42)",
    )


def validate_python(python: Path) -> Path:
    result = python_version_result(python)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or str(python)).strip()
        raise SystemExit(f"target Python is not usable or is older than 3.11: {detail}")
    return python


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_ficelle_python(
    options: BootstrapOptions,
    selected_python: Path,
) -> Path:
    """Create or reuse the Ficelle-owned runtime from the selected base Python."""
    python = venv_python(options.venv.absolute())
    if python.exists():
        return validate_python(python)
    venv_existed = options.venv.exists()
    result = run_isolated_command(
        [str(selected_python.absolute()), "-m", "venv", str(options.venv.absolute())],
        dry_run=options.dry_run,
    )
    uv = shutil.which("uv")
    if result.returncode != 0 and uv:
        result = run_uv_command(
            uv, ["venv"], [
                "--python",
                str(selected_python.absolute()),
                str(options.venv.absolute()),
            ],
            dry_run=options.dry_run,
            env=command_env(options),
        )
    if result.returncode != 0 and not options.dry_run and not venv_existed:
        # A failed `python -m venv` (Debian/Ubuntu without python3-venv) leaves an interpreter
        # with no pip behind; the next run would reuse it as a finished runtime and fail later.
        shutil.rmtree(options.venv, ignore_errors=True)
    ensure_success(result, action="isolated Python environment creation")
    if options.dry_run:
        return python
    return validate_python(python)


def resolve_base_python(options: BootstrapOptions) -> Path:
    value = options.python or sys.executable
    selected = Path(value).expanduser()
    # A bare executable name uses PATH; an explicit relative path uses the caller's cwd.
    if not selected.is_absolute() and not os.path.dirname(value):
        executable = shutil.which(value)
        if executable is None:
            raise SystemExit(f"target Python executable was not found on PATH: {value}")
        selected = Path(executable)
    return validate_python(selected.absolute())


def safe_wheel_filename(value: str | None) -> str | None:
    name = str(value or "").strip().replace("\\", "/").split("/")[-1]
    if (
        not name
        or name.startswith(".")
        or not name.isprintable()
        or len(name) > 128
        or not _WHEEL_FILENAME_PATTERN.fullmatch(name)
    ):
        return None
    return name


def wheel_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return safe_wheel_filename(Path(parsed.path).name) or FALLBACK_WHEEL_FILENAME


def wheel_filename_from_response(response: object, url: str) -> str:
    headers = getattr(response, "headers", None)
    get_filename = getattr(headers, "get_filename", None)
    if callable(get_filename):
        try:
            advertised = safe_wheel_filename(get_filename())
        except Exception:
            advertised = None
        if advertised:
            return advertised
    return wheel_filename_from_url(url)


def effective_origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urllib.parse.urlsplit(url)
    default_port = 443 if parsed.scheme == "https" else 80 if parsed.scheme == "http" else None
    return parsed.scheme.lower(), parsed.hostname, parsed.port or default_port


def uses_secure_http_transport(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS


class SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent an authenticated wheel redirect from leaking the license key to another origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        if effective_origin(req.full_url) != effective_origin(newurl):
            raise urllib.error.URLError(
                "refusing a cross-origin redirect for the Ficelle wheel"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_same_origin(request: urllib.request.Request, *, timeout: float) -> object:
    return urllib.request.build_opener(SameOriginRedirectHandler()).open(
        request,
        timeout=timeout,
    )


def open_wheel(
    request: urllib.request.Request,
    *,
    timeout: float,
    authenticated: bool,
) -> object:
    if authenticated:
        return open_same_origin(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        print("SHA256: not provided; skipping checksum verification.")
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise SystemExit(f"SHA256 mismatch for {path.name}: expected {expected}, got {actual}")
    print(f"SHA256 verified: {actual}")


def verify_pro_core_compatibility(wheel: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(
                archive.read(metadata_name).decode("utf-8")
            )
    except (OSError, UnicodeError, zipfile.BadZipFile, StopIteration) as exc:
        raise SystemExit("Pro wheel metadata could not be verified") from exc
    requirements = metadata.get_all("Requires-Dist", [])
    expected = f"ficelle-router=={CORE_VERSION}"
    normalized = {
        requirement.replace(" ", "").lower()
        for requirement in requirements
    }
    if expected not in normalized:
        raise SystemExit(
            f"Pro wheel is incompatible with Ficelle Core {CORE_VERSION}"
        )


def copy_local_wheel(source: Path, destination: Path, *, dry_run: bool) -> Path:
    if dry_run:
        print(f"DRY RUN: copy wheel {source} -> {destination}")
        return destination
    if not source.exists():
        raise SystemExit(f"wheel file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def download_url_wheel(
    url: str,
    download_dir: Path,
    *,
    license_key: str | None,
    dry_run: bool,
) -> Path:
    parsed = urllib.parse.urlsplit(url)
    destination = download_dir / wheel_filename_from_url(url)

    if parsed.scheme == "file":
        return copy_local_wheel(
            Path(urllib.request.url2pathname(parsed.path)),
            destination,
            dry_run=dry_run,
        )
    if parsed.scheme == "":
        return copy_local_wheel(
            Path(url).expanduser(),
            destination,
            dry_run=dry_run,
        )
    if not uses_secure_http_transport(url):
        raise SystemExit(
            "refusing to download a Ficelle wheel over an insecure URL "
            "(use HTTPS, or HTTP on loopback for local development)"
        )

    print(f"Downloading Ficelle wheel: {redact_url(url)}")
    if license_key:
        print("Using license key for authenticated download: [REDACTED]")
    if dry_run:
        print(f"DRY RUN: download wheel -> {destination}")
        return destination

    headers = {"Accept": "application/octet-stream"}
    if license_key:
        headers["Authorization"] = f"Bearer {license_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with open_wheel(
            request,
            timeout=120,
            authenticated=bool(license_key),
        ) as response:
            destination = download_dir / wheel_filename_from_response(response, url)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise SystemExit("wheel download was refused. Check the Ficelle license key or account entitlement.") from exc
        raise SystemExit(f"wheel download failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"wheel download failed: {exc.reason}") from exc
    print(f"Downloaded wheel: {destination}")
    return destination


def download_wheel(options: BootstrapOptions, download_dir: Path) -> Path:
    return download_url_wheel(
        options.wheel_url,
        download_dir,
        license_key=options.license_key,
        dry_run=options.dry_run,
    )


def pro_attestation_url(wheel_url: str) -> str:
    parsed = urllib.parse.urlsplit(wheel_url)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, f"{parsed.path}.attestation", parsed.query, parsed.fragment)
    )


def download_pro_attestation(options: BootstrapOptions, download_dir: Path) -> Path:
    url = pro_attestation_url(options.wheel_url)
    destination = download_dir / "ficelle-pro-wheel.attestation.json"
    print(f"Downloading Ficelle Pro release attestation: {redact_url(url)}")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {options.license_key or ''}",
        },
    )
    try:
        with open_wheel(request, timeout=30, authenticated=True) as response:
            raw = response.read(MAX_ATTESTATION_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"release attestation download failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"release attestation download failed: {exc.reason}") from exc
    if len(raw) > MAX_ATTESTATION_BYTES:
        raise SystemExit("release attestation exceeds the maximum allowed size")
    destination.write_bytes(raw)
    return destination


def pro_wheel_needs_signed_attestation(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme in {"", "file"}:
        return False
    return not (
        parsed.scheme == "http"
        and (parsed.hostname or "").lower().rstrip(".") in _LOOPBACK_HOSTS
    )


def verify_pro_wheel_integrity(
    options: BootstrapOptions,
    python: Path,
    wheel: Path,
    download_dir: Path,
) -> None:
    """Verify a Pro wheel before pip executes it.

    A remote wheel needs an explicit SHA-256 or the signed release attestation rooted
    in Core. A local file or loopback wheel has no attestation sidecar, so a matching
    SHA-256 is mandatory. A missing or mismatched digest stops the bootstrap.
    """
    if options.sha256:
        verify_sha256(wheel, options.sha256)
        return
    if not pro_wheel_needs_signed_attestation(options.wheel_url):
        raise SystemExit(
            "Pro wheel SHA-256 is required for a local or loopback wheel. "
            "Pass --sha256 or set FICELLE_WHEEL_SHA256. "
            "Refusing to install an unverified Pro wheel."
        )
    attestation = download_pro_attestation(options, download_dir)
    result = run_isolated_command(
        [
            str(python.absolute()),
            "-m",
            "ficelle.artifact_integrity",
            "verify",
            str(wheel.absolute()),
            str(attestation.absolute()),
        ],
        dry_run=False,
        env=command_env(options),
    )
    ensure_success(result, action="Pro release attestation verification")


def download_core_wheel(options: BootstrapOptions, download_dir: Path) -> Path:
    return download_url_wheel(
        options.core_wheel_url,
        download_dir,
        license_key=None,
        dry_run=options.dry_run,
    )


def install_wheel(
    options: BootstrapOptions,
    python: Path,
    wheel: Path,
) -> None:
    python = python.absolute()
    wheel = wheel.absolute()
    env = command_env(options)
    pip_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--force-reinstall",
    ]
    pip_command.append("--no-deps")
    pip_command.append(str(wheel))
    result = run_isolated_command(
        pip_command,
        dry_run=options.dry_run,
        env=env,
    )
    if result.returncode == 0:
        return
    uv = shutil.which("uv")
    if pip_is_unavailable(result) and uv:
        print("python -m pip is unavailable; retrying wheel install with uv pip.")
        uv_arguments = [
            "--python",
            str(python),
            "--reinstall",
        ]
        uv_arguments.append("--no-deps")
        uv_arguments.append(str(wheel))
        result = run_uv_command(
            uv, ["pip", "install"], uv_arguments,
            dry_run=options.dry_run,
            env=env,
        )
    ensure_success(result, action="wheel install")


def install_requirements(
    options: BootstrapOptions,
    python: Path,
    requirements: Sequence[str],
) -> None:
    if not requirements:
        return
    # Validate again at the pip boundary so a caller cannot pass flags or URLs through.
    safe_requirements = [validate_runtime_requirement(requirement) for requirement in requirements]
    python = python.absolute()
    env = command_env(options)
    pip_command = [str(python), "-m", "pip", "install", *safe_requirements]
    result = run_isolated_command(
        pip_command,
        dry_run=options.dry_run,
        env=env,
    )
    if result.returncode == 0:
        return
    uv = shutil.which("uv")
    if pip_is_unavailable(result) and uv:
        result = run_uv_command(
            uv, ["pip", "install"],
            [
                "--python",
                str(python),
                *safe_requirements,
            ],
            dry_run=options.dry_run,
            env=env,
        )
    ensure_success(result, action="runtime dependency install")


# Conservative named-index grammar, duplicated in update.py because bootstrap runs
# before Core or packaging is installed. Cross-tests keep both policies equivalent.
_REQUIREMENT_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_REQUIREMENT_EXTRAS = rf"{_REQUIREMENT_NAME}(?:[ ]*,[ ]*{_REQUIREMENT_NAME})*"
_REQUIREMENT_SPECIFIER = r"(?:===|==|!=|<=|>=|~=|<|>)[ ]*[A-Za-z0-9*+!_.-]+"
_REQUIREMENT_SPECIFIERS = rf"{_REQUIREMENT_SPECIFIER}(?:[ ]*,[ ]*{_REQUIREMENT_SPECIFIER})*"
_REQUIREMENT_MARKER = r"[A-Za-z0-9_.\"'<>=! (),]+"
# Pip's ARCHIVE_EXTENSIONS (pip/_internal/utils/filetypes.py) plus defence-in-depth
# extras (.tbz2, .tar.zst, .egg): pip interprets these as local archive filenames.
_RUNTIME_ARCHIVE_SUFFIXES = (
    ".whl", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2",
    ".tar.xz", ".txz", ".tar.lz", ".tlz", ".tar.lzma", ".tar.zst", ".egg",
)
_RUNTIME_REQUIREMENT = re.compile(
    rf"(?P<name>{_REQUIREMENT_NAME})"
    rf"(?:[ ]*\[[ ]*(?:{_REQUIREMENT_EXTRAS})[ ]*\])?"
    rf"(?:[ ]*(?:{_REQUIREMENT_SPECIFIERS}|\([ ]*{_REQUIREMENT_SPECIFIERS}[ ]*\)))?"
    rf"(?:[ ]*;[ ]*{_REQUIREMENT_MARKER})?"
)
_FICELLE_REQUIREMENT_NAMES = ("ficelle-router", "ficelle-pro")


def validate_runtime_requirement(raw: object) -> str:
    """Return a pip-installable name requirement, or stop with a readable error.

    Refuses pip options (entries starting with ``-``), direct URLs, VCS paths, and
    any string that is not a package name with optional extras, specifiers, and marker.
    """
    if not isinstance(raw, str):
        raise SystemExit("refusing a runtime requirement that is not a package specifier")
    # Check the original input before trimming: strip() would hide Unicode/control bytes.
    if any(not (" " <= character <= "~") for character in raw):
        raise SystemExit(
            "refusing a runtime requirement that contains non-ASCII or control characters "
            "(including newline or null)"
        )
    requirement = raw.strip(" ")
    if not requirement:
        raise SystemExit("refusing an empty runtime requirement")
    if requirement.startswith("-"):
        raise SystemExit(
            "refusing a runtime requirement that starts with '-'. "
            "Pip options are not accepted from wheel metadata."
        )
    name_part = requirement.split(";", 1)[0]
    if "://" in requirement or "@" in name_part:
        raise SystemExit(
            "refusing a direct-URL runtime requirement. "
            "Only named packages from the configured index are accepted."
        )
    match = _RUNTIME_REQUIREMENT.fullmatch(requirement)
    if match is None:
        raise SystemExit(
            "refusing a runtime requirement that is not a package specifier"
        )
    if match.group("name").lower().endswith(_RUNTIME_ARCHIVE_SUFFIXES):
        raise SystemExit("refusing a runtime requirement with an archive filename as its package name")
    return requirement


def wheel_runtime_requirements(
    wheel: Path,
    *,
    fallback: Sequence[str] = (),
) -> tuple[str, ...]:
    if not wheel.is_file():
        return tuple(validate_runtime_requirement(requirement) for requirement in fallback)
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    except (OSError, UnicodeError, zipfile.BadZipFile, StopIteration) as exc:
        raise SystemExit(f"wheel metadata could not be read: {wheel}") from exc
    accepted: list[str] = []
    for raw_requirement in metadata.get_all("Requires-Dist", []):
        requirement = validate_runtime_requirement(raw_requirement)
        normalized = requirement.lower().replace("_", "-")
        if normalized.startswith(_FICELLE_REQUIREMENT_NAMES):
            continue
        accepted.append(requirement)
    return tuple(accepted)


def check_runtime(options: BootstrapOptions, python: Path) -> None:
    result = run_isolated_command(
        [str(python.absolute()), "-m", "pip", "check"],
        dry_run=options.dry_run,
        env=command_env(options),
    )
    uv = shutil.which("uv")
    if result.returncode != 0 and pip_is_unavailable(result) and uv:
        result = run_uv_command(
            uv, ["pip", "check"], ["--python", str(python.absolute())],
            dry_run=options.dry_run,
            env=command_env(options),
        )
    ensure_success(result, action="Ficelle runtime dependency check")


def setup_command(
    options: BootstrapOptions,
    python: Path,
    wheel: Path,
    *,
    expose_cli: bool = True,
) -> list[str]:
    command = [
        str(python.absolute()),
        "-m",
        "ficelle.install",
        "--skip-package",
        "--package",
        str(wheel.absolute()),
        "--no-editable",
    ]
    if options.ficelle_home_explicit:
        command.extend(("--ficelle-home", str(options.ficelle_home.absolute())))
    for connector in options.connectors:
        command.extend(("--connector", connector))
    if "hermes" in options.connectors:
        command.extend(("--hermes-home", str(options.hermes_home.absolute())))
    if not options.backup_existing:
        command.append("--no-backup")
    if options.skip_service:
        command.append("--skip-service")
    if options.skip_smoke:
        command.append("--skip-smoke")
    if options.dry_run:
        command.append("--dry-run")
    if expose_cli:
        command.append("--expose-cli")
    return command


def run_packaged_setup(
    options: BootstrapOptions,
    python: Path,
    wheel: Path,
    *,
    expose_cli: bool = True,
) -> None:
    result = run_isolated_command(
        setup_command(options, python, wheel, expose_cli=expose_cli),
        dry_run=options.dry_run,
        env=command_env(options),
    )
    ensure_success(result, action="Ficelle setup")


def verify_install(
    options: BootstrapOptions,
    python: Path,
    *,
    expect_pro: bool,
) -> None:
    """Confirm the requested Core or Core+Pro install imports."""
    if options.dry_run:
        return
    code = (
        "import ficelle, ficelle_pro, ficelle_pro.licensing; "
        "print('ficelle +', getattr(ficelle_pro, '__version__', '?'))"
        if expect_pro
        else "import ficelle; print('ficelle core')"
    )
    result = run_isolated_command(
        [str(python.absolute()), "-c", code],
        dry_run=options.dry_run,
        env=command_env(options),
    )
    if result.returncode != 0:
        raise SystemExit(
            "post-install check failed: the requested Ficelle package set did not import"
        )


def run_license_activation(
    options: BootstrapOptions,
    python: Path,
) -> None:
    """Activate this machine's Pro entitlement after a confirmed install (R7), best-effort.

    The license key is passed through the environment, never on argv — run_command echoes the
    command it runs, so a key on the command line would leak to stdout/logs (the bootstrap must
    never print the license key). A failure here does not fail the bootstrap: the packages are
    installed, so the user can re-run `ficelle license activate` once the key/network are ready.
    """
    if options.dry_run or not options.license_key:
        return
    env = command_env(options)
    env["FICELLE_LICENSE_KEY"] = options.license_key
    result = run_isolated_command([str(python.absolute()), "-m", "ficelle", "license", "activate"], dry_run=options.dry_run, env=env)
    if result.returncode == 0:
        print("Pro license activated on this machine.")
    else:
        print(
            "Note: packages installed, but license activation did not complete. "
            "Set FICELLE_LICENSE_KEY or run `ficelle license activate` interactively "
            "once your key and network are ready."
        )


# Exposing the CLI lives in `ficelle.install` (`--expose-cli`), not here: it is one rule about
# writing outside the install's footprint, and a second copy in this script could drift from the
# one the packaged `ficelle-setup` applies. The bootstrap always owns the isolated runtime, so
# its commands are always safe to expose through that single implementation.


def run_bootstrap(options: BootstrapOptions) -> int:
    selected_python = resolve_base_python(options)
    python = prepare_ficelle_python(options, selected_python)
    print(f"Ficelle runtime: {options.venv}")
    print(f"Ficelle Python: {python}")
    print(f"Ficelle home: {options.ficelle_home}")
    if options.connectors:
        print(f"Connectors: {', '.join(options.connectors)}")

    if options.dry_run:
        download_dir = Path(tempfile.gettempdir()) / "ficelle-bootstrap-dry-run"
        core_wheel = download_core_wheel(options, download_dir)
        print("DRY RUN: skip checksums because no wheel was downloaded.")
        install_requirements(
            options,
            python,
            wheel_runtime_requirements(core_wheel, fallback=CORE_RUNTIME_REQUIREMENTS),
        )
        install_wheel(options, python, core_wheel)
        if options.license_key:
            pro_wheel = download_wheel(options, download_dir)
            verify_pro_core_compatibility(
                pro_wheel,
                dry_run=options.dry_run,
            )
            install_requirements(
                options,
                python,
                wheel_runtime_requirements(pro_wheel, fallback=PRO_RUNTIME_REQUIREMENTS),
            )
            install_wheel(options, python, pro_wheel)
        check_runtime(options, python)
        run_packaged_setup(options, python, core_wheel)
        return 0

    with tempfile.TemporaryDirectory(prefix="ficelle-bootstrap-") as tmp:
        download_dir = Path(tmp)
        core_wheel = download_core_wheel(options, download_dir)
        verify_sha256(core_wheel, options.core_sha256)
        install_requirements(
            options,
            python,
            wheel_runtime_requirements(core_wheel),
        )
        install_wheel(options, python, core_wheel)
        wheels = [core_wheel]
        if options.license_key:
            pro_wheel = download_wheel(options, download_dir)
            verify_pro_wheel_integrity(options, python, pro_wheel, download_dir)
            verify_pro_core_compatibility(pro_wheel, dry_run=False)
            install_requirements(
                options,
                python,
                wheel_runtime_requirements(pro_wheel),
            )
            install_wheel(options, python, pro_wheel)
            wheels.append(pro_wheel)
        check_runtime(options, python)
        run_packaged_setup(options, python, core_wheel)
        verify_install(
            options,
            python,
            expect_pro=bool(options.license_key),
        )
        run_license_activation(options, python)
        if options.keep_wheel:
            keep_dir = options.ficelle_home / "artifacts"
            keep_dir.mkdir(parents=True, exist_ok=True)
            for wheel in wheels:
                kept = keep_dir / wheel.name
                shutil.copy2(wheel, kept)
                print(f"Kept wheel artifact: {kept}")
    print("Ficelle bootstrap complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install Ficelle Core and optionally Ficelle Pro without cloning the repo."
    )
    parser.add_argument(
        "--core-wheel-url",
        default=os.getenv("FICELLE_CORE_WHEEL_URL", DEFAULT_CORE_WHEEL_URL),
        help="Versioned public Core wheel URL or local path.",
    )
    parser.add_argument(
        "--core-sha256",
        default=os.getenv("FICELLE_CORE_WHEEL_SHA256", DEFAULT_CORE_SHA256),
        help="Expected Core wheel SHA256.",
    )
    parser.add_argument("--wheel-url", default=os.getenv("FICELLE_WHEEL_URL", DEFAULT_WHEEL_URL), help="Private wheel URL or local wheel path. Default: Ficelle install service endpoint.")
    parser.add_argument(
        "--sha256",
        default=os.getenv("FICELLE_WHEEL_SHA256"),
        help=(
            "Pro wheel SHA256. Required for a local or loopback wheel; "
            "remote releases may instead use the bundled signing root."
        ),
    )
    parser.add_argument(
        "--python",
        default=os.getenv("FICELLE_PYTHON"),
        help="Base Python 3.11+ used to create Ficelle's isolated runtime.",
    )
    parser.add_argument(
        "--venv",
        default=os.getenv(
            "FICELLE_VENV",
            str(Path.home() / ".local" / "share" / "ficelle" / "venv"),
        ),
        help="Ficelle-owned isolated runtime. It is always used for the installation.",
    )
    parser.add_argument("--ficelle-home", default=None, help="Ficelle state root. Default: FICELLE_HOME or ~/.ficelle")
    parser.add_argument(
        "--connector",
        action="append",
        choices=CONNECTORS,
        default=[],
        help="Install an optional client connector after Core verification. Repeatable.",
    )
    parser.add_argument("--hermes-home", default=str(DEFAULT_HERMES_HOME), help="Hermes home directory. Default: ~/.hermes")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup existing plugin/config files before replacement.")
    parser.add_argument("--skip-service", action="store_true", help="Install package/plugin/config helper but do not start the managed service.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip doctor/health/models smoke checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading, installing, or writing files.")
    parser.add_argument("--keep-wheel", action="store_true", help="Copy the downloaded wheel to FICELLE_HOME/artifacts/ after install.")
    return parser


def options_from_args(args: argparse.Namespace) -> BootstrapOptions:
    configured_ficelle_home = args.ficelle_home or os.getenv("FICELLE_HOME")
    return BootstrapOptions(
        core_wheel_url=args.core_wheel_url,
        core_sha256=args.core_sha256,
        wheel_url=args.wheel_url,
        license_key=os.getenv("FICELLE_LICENSE_KEY") or os.getenv("FICELLE_INSTALL_TOKEN"),
        sha256=args.sha256,
        python=args.python,
        venv=Path(args.venv).expanduser().resolve(),
        connectors=tuple(dict.fromkeys(args.connector)),
        ficelle_home=Path(
            configured_ficelle_home or Path.home() / ".ficelle"
        ).expanduser().resolve(),
        hermes_home=Path(args.hermes_home).expanduser().resolve(),
        backup_existing=not args.no_backup,
        skip_service=args.skip_service,
        skip_smoke=args.skip_smoke,
        dry_run=args.dry_run,
        keep_wheel=args.keep_wheel,
        ficelle_home_explicit=configured_ficelle_home is not None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_bootstrap(options_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
