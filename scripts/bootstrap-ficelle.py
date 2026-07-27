#!/usr/bin/env python3
"""Bootstrap Ficelle Pro from a private wheel without cloning the repository.

This script is intentionally stdlib-only so it can be served as:

    curl -fsSL https://install.ficelle.ai/bootstrap.py | python3 - --license-key ...

It delivers the **Pro** install (open core + closed pack). The wheel it downloads is
the licensed ``ficelle-pro`` pack; installing it pulls the open-core ``ficelle-router``
package from PyPI as its dependency, so both land together. It then runs the core's
packaged setup entrypoint and confirms both packages import. It never prints license
keys or provider secrets.

The free tier needs no bootstrap: ``uv tool install ficelle-router`` (or
``pip install ficelle-router``) installs the open core alone.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_WHEEL_URL = "https://install.ficelle.ai/api/releases/latest/wheel"
DEFAULT_HERMES_HOME = Path.home() / ".hermes"
FALLBACK_WHEEL_FILENAME = "ficelle_pro-0-py3-none-any.whl"
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
    wheel_url: str
    license_key: str | None
    sha256: str | None
    python: str | None
    hermes_home: Path
    configure_hermes: bool
    backup_existing: bool
    skip_service: bool
    skip_smoke: bool
    dry_run: bool
    keep_wheel: bool


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
    env["HERMES_HOME"] = str(options.hermes_home)
    return env


def run_command(command: Sequence[str], *, dry_run: bool, env: Mapping[str, str] | None = None) -> CommandResult:
    env_hint = f" HERMES_HOME={env['HERMES_HOME']}" if env and env.get("HERMES_HOME") else ""
    printable = " ".join(command)
    if dry_run:
        print(f"DRY RUN:{env_hint} {printable}")
        return CommandResult(list(command), 0)
    print(f"RUN:{env_hint} {printable}")
    completed = subprocess.run(list(command), text=True, capture_output=True, env=dict(env) if env else None)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return CommandResult(list(command), completed.returncode, completed.stdout or "", completed.stderr or "")


def ensure_success(result: CommandResult, *, action: str) -> None:
    if result.returncode != 0:
        raise SystemExit(f"{action} failed with exit code {result.returncode}")


def pip_is_unavailable(result: CommandResult) -> bool:
    combined = f"{result.stdout}\n{result.stderr}"
    return "No module named pip" in combined or "No module named 'pip'" in combined


def candidate_hermes_pythons(home: Path) -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("HERMES_PYTHON"):
        candidates.append(Path(os.environ["HERMES_PYTHON"]).expanduser())
    candidates.extend([
        home / "hermes-agent" / "venv" / "bin" / "python",
        home / "venv" / "bin" / "python",
        Path.home() / ".local" / "share" / "hermes" / "venv" / "bin" / "python",
    ])
    candidates.append(Path(sys.executable))
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded not in seen:
            unique.append(expanded)
            seen.add(expanded)
    return unique


def python_version_result(python: Path) -> CommandResult:
    code = (
        "import sys; "
        "print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); "
        "raise SystemExit(0 if sys.version_info >= (3, 11) else 42)"
    )
    completed = subprocess.run([str(python), "-c", code], text=True, capture_output=True)
    return CommandResult([str(python), "-c", code], completed.returncode, completed.stdout or "", completed.stderr or "")


def detect_hermes_python(options: BootstrapOptions) -> Path:
    if options.python:
        candidate = Path(options.python).expanduser()
        result = python_version_result(candidate)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or str(candidate)).strip()
            raise SystemExit(f"target Python is not usable or is older than 3.11: {detail}")
        return candidate

    failures: list[str] = []
    for candidate in candidate_hermes_pythons(options.hermes_home):
        if not candidate.exists():
            failures.append(f"missing: {candidate}")
            continue
        result = python_version_result(candidate)
        if result.returncode == 0:
            return candidate
        failures.append(f"unusable: {candidate} {(result.stderr or result.stdout).strip()}")
    raise SystemExit("No usable Python 3.11+ found. Pass --python /path/to/hermes/venv/bin/python. Tried: " + "; ".join(failures))


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


def copy_local_wheel(source: Path, destination: Path, *, dry_run: bool) -> Path:
    if dry_run:
        print(f"DRY RUN: copy wheel {source} -> {destination}")
        return destination
    if not source.exists():
        raise SystemExit(f"wheel file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def download_wheel(options: BootstrapOptions, download_dir: Path) -> Path:
    url = options.wheel_url
    parsed = urllib.parse.urlsplit(url)
    destination = download_dir / wheel_filename_from_url(url)

    if parsed.scheme == "file":
        return copy_local_wheel(Path(urllib.request.url2pathname(parsed.path)), destination, dry_run=options.dry_run)
    if parsed.scheme == "":
        return copy_local_wheel(Path(url).expanduser(), destination, dry_run=options.dry_run)
    if not uses_secure_http_transport(url):
        raise SystemExit(
            "refusing to download a Ficelle wheel over an insecure URL "
            "(use HTTPS, or HTTP on loopback for local development)"
        )

    print(f"Downloading Ficelle wheel: {redact_url(url)}")
    if options.license_key:
        print("Using license key for authenticated download: [REDACTED]")
    if options.dry_run:
        print(f"DRY RUN: download wheel -> {destination}")
        return destination

    headers = {"Accept": "application/octet-stream"}
    if options.license_key:
        headers["Authorization"] = f"Bearer {options.license_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with open_same_origin(request, timeout=120) as response:
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


def install_wheel(options: BootstrapOptions, python: Path, wheel: Path) -> None:
    env = command_env(options)
    result = run_command([str(python), "-m", "pip", "install", "--force-reinstall", str(wheel)], dry_run=options.dry_run, env=env)
    if result.returncode == 0:
        return
    if pip_is_unavailable(result) and shutil.which("uv"):
        print("python -m pip is unavailable; retrying wheel install with uv pip.")
        result = run_command(["uv", "pip", "install", "--python", str(python), "--reinstall", str(wheel)], dry_run=options.dry_run, env=env)
    ensure_success(result, action="wheel install")


def setup_command(options: BootstrapOptions, python: Path, wheel: Path) -> list[str]:
    command = [
        str(python),
        "-m",
        "ficelle.install",
        "--skip-package",
        "--package",
        str(wheel),
        "--no-editable",
        "--hermes-home",
        str(options.hermes_home),
    ]
    if options.configure_hermes:
        command.append("--configure-hermes")
    if not options.backup_existing:
        command.append("--no-backup")
    if options.skip_service:
        command.append("--skip-service")
    if options.skip_smoke:
        command.append("--skip-smoke")
    if options.dry_run:
        command.append("--dry-run")
    return command


def run_packaged_setup(options: BootstrapOptions, python: Path, wheel: Path) -> None:
    result = run_command(setup_command(options, python, wheel), dry_run=options.dry_run, env=command_env(options))
    ensure_success(result, action="Ficelle setup")


def verify_two_package_install(options: BootstrapOptions, python: Path) -> None:
    """Confirm both halves of a Pro install import: the open core and the closed pack."""
    if options.dry_run:
        return
    code = "import ficelle, ficelle_pro; print('ficelle +', getattr(ficelle_pro, '__version__', '?'))"
    result = run_command([str(python), "-c", code], dry_run=options.dry_run, env=command_env(options))
    if result.returncode != 0:
        raise SystemExit(
            "post-install check failed: the open core (ficelle) and/or the Pro pack (ficelle_pro) "
            "did not import. The core resolves from PyPI as a dependency of the ficelle-pro wheel — "
            "confirm ficelle-router is published and reachable from the target environment."
        )


def run_license_activation(options: BootstrapOptions, python: Path) -> None:
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
    result = run_command([str(python), "-m", "ficelle", "license", "activate"], dry_run=options.dry_run, env=env)
    if result.returncode == 0:
        print("Pro license activated on this machine.")
    else:
        print(
            "Note: packages installed, but license activation did not complete. "
            "Run `ficelle license activate <license-key>` once your key and network are ready."
        )


def run_bootstrap(options: BootstrapOptions) -> int:
    python = detect_hermes_python(options)
    print(f"Target Python: {python}")
    print(f"Hermes home: {options.hermes_home}")

    if options.dry_run:
        download_dir = Path(tempfile.gettempdir()) / "ficelle-bootstrap-dry-run"
        wheel = download_wheel(options, download_dir)
        print("DRY RUN: skip checksum because no wheel was downloaded.")
        install_wheel(options, python, wheel)
        run_packaged_setup(options, python, wheel)
        return 0

    with tempfile.TemporaryDirectory(prefix="ficelle-bootstrap-") as tmp:
        download_dir = Path(tmp)
        wheel = download_wheel(options, download_dir)
        verify_sha256(wheel, options.sha256)
        install_wheel(options, python, wheel)
        run_packaged_setup(options, python, wheel)
        verify_two_package_install(options, python)
        run_license_activation(options, python)
        if options.keep_wheel:
            keep_dir = options.hermes_home / "ficelle" / "artifacts"
            keep_dir.mkdir(parents=True, exist_ok=True)
            kept = keep_dir / wheel.name
            shutil.copy2(wheel, kept)
            print(f"Kept wheel artifact: {kept}")
    print("Ficelle bootstrap complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install Ficelle from a private wheel without cloning the repo.")
    parser.add_argument("--wheel-url", default=os.getenv("FICELLE_WHEEL_URL", DEFAULT_WHEEL_URL), help="Private wheel URL or local wheel path. Default: Ficelle install service endpoint.")
    parser.add_argument("--license-key", default=os.getenv("FICELLE_LICENSE_KEY") or os.getenv("FICELLE_INSTALL_TOKEN"), help="Ficelle license/install token. Also read from FICELLE_LICENSE_KEY or FICELLE_INSTALL_TOKEN.")
    parser.add_argument("--sha256", default=os.getenv("FICELLE_WHEEL_SHA256"), help="Expected wheel SHA256. Strongly recommended for release validation.")
    parser.add_argument("--python", default=os.getenv("HERMES_PYTHON"), help="Target Hermes Python. Auto-detects common Hermes venv paths.")
    parser.add_argument("--hermes-home", default=str(DEFAULT_HERMES_HOME), help="Hermes home directory. Default: ~/.hermes")
    parser.add_argument("--no-configure-hermes", action="store_true", help="Do not ask packaged setup to write the safe Hermes snippet/config helper.")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup existing plugin/config files before replacement.")
    parser.add_argument("--skip-service", action="store_true", help="Install package/plugin/config helper but do not start the managed service.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip doctor/health/models smoke checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading, installing, or writing files.")
    parser.add_argument("--keep-wheel", action="store_true", help="Copy the downloaded wheel to ~/.hermes/ficelle/artifacts/ after install.")
    return parser


def options_from_args(args: argparse.Namespace) -> BootstrapOptions:
    return BootstrapOptions(
        wheel_url=args.wheel_url,
        license_key=args.license_key,
        sha256=args.sha256,
        python=args.python,
        hermes_home=Path(args.hermes_home).expanduser(),
        configure_hermes=not args.no_configure_hermes,
        backup_existing=not args.no_backup,
        skip_service=args.skip_service,
        skip_smoke=args.skip_smoke,
        dry_run=args.dry_run,
        keep_wheel=args.keep_wheel,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_bootstrap(options_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
