from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import zipfile
from email.message import Message
from pathlib import Path

import pytest

from conftest import unreleasable_version
from ficelle import update


BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap-ficelle.py"
spec = importlib.util.spec_from_file_location("ficelle_bootstrap", BOOTSTRAP_PATH)
assert spec is not None and spec.loader is not None
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)


class _WheelResponse(io.BytesIO):
    def __init__(self, filename: str):
        super().__init__(b"wheel-bytes")
        self.headers = Message()
        self.headers.add_header(
            "Content-Disposition",
            "attachment",
            filename=filename,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_options(tmp_path, **overrides):
    target = overrides.pop("target", None)
    configure_hermes = overrides.pop("configure_hermes", False)
    values = {
        "core_wheel_url": str(
            tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
        ),
        "core_sha256": None,
        "wheel_url": str(tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"),
        "license_key": None,
        "sha256": None,
        "python": None,
        "venv": tmp_path / ".local" / "share" / "ficelle" / "venv",
        "connectors": ("hermes",) if target == "hermes" or configure_hermes else (),
        "ficelle_home": tmp_path / ".ficelle",
        "hermes_home": tmp_path / ".hermes",
        "backup_existing": True,
        "skip_service": False,
        "skip_smoke": False,
        "dry_run": False,
        "keep_wheel": False,
        "ficelle_home_explicit": True,
    }
    values.update(overrides)
    return bootstrap.BootstrapOptions(**values)


def write_pro_wheel(path: Path, core_requirement: str) -> None:
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: ficelle-pro\n"
        "Version: 0.1.3\n"
        f"Requires-Dist: {core_requirement}\n"
        "Requires-Dist: cryptography>=42\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "ficelle_pro-0.1.3.dist-info/METADATA",
            metadata,
        )


def test_redact_url_hides_query_tokens():
    assert bootstrap.redact_url("https://example.test/wheel?token=secret") == "https://example.test/wheel?[REDACTED]"
    assert bootstrap.redact_url("https://example.test/wheel") == "https://example.test/wheel"


def test_setup_command_configures_hermes_only_when_requested(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, target="hermes", configure_hermes=True)

    command = bootstrap.setup_command(options, Path("/tmp/ficelle-python"), wheel)

    assert command[:3] == ["/tmp/ficelle-python", "-m", "ficelle.install"]
    assert "--skip-package" in command
    assert command[command.index("--connector") + 1] == "hermes"
    assert command[command.index("--ficelle-home") + 1] == str(tmp_path / ".ficelle")
    assert "--hermes-home" in command
    assert str(tmp_path / ".hermes") in command


def test_setup_command_can_disable_config_and_backups(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, configure_hermes=False, backup_existing=False, skip_service=True, skip_smoke=True, dry_run=True)

    command = bootstrap.setup_command(options, Path("/tmp/ficelle-python"), wheel)

    assert "--configure-hermes" not in command
    assert "--no-backup" in command
    assert "--skip-service" in command
    assert "--skip-smoke" in command
    assert "--dry-run" in command


def test_setup_command_exposes_cli_only_when_requested(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, dry_run=True)
    python = Path("/tmp/isolated/bin/python")

    command = bootstrap.setup_command(options, python, wheel, expose_cli=False)
    exposed_command = bootstrap.setup_command(
        options,
        python,
        wheel,
        expose_cli=True,
    )

    assert "--expose-cli" not in command
    assert "--dry-run" in exposed_command
    assert "--expose-cli" in exposed_command


def test_implicit_home_is_omitted_from_setup_command_and_environment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("FICELLE_HOME", "/inherited/home")
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, ficelle_home_explicit=False)
    captured = {}

    def fake_run(command, *, dry_run, env=None, cwd=None):
        captured["command"] = list(command)
        captured["env"] = dict(env or {})
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)

    bootstrap.run_packaged_setup(
        options,
        Path("/tmp/ficelle-python"),
        wheel,
    )

    assert "--ficelle-home" not in captured["command"]
    assert "FICELLE_HOME" not in captured["env"]


def test_explicit_home_is_passed_to_setup_command_and_environment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("FICELLE_HOME", raising=False)
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, ficelle_home_explicit=True)
    captured = {}

    def fake_run(command, *, dry_run, env=None, cwd=None):
        captured["command"] = list(command)
        captured["env"] = dict(env or {})
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)

    bootstrap.run_packaged_setup(
        options,
        Path("/tmp/ficelle-python"),
        wheel,
    )

    home_index = captured["command"].index("--ficelle-home")
    assert captured["command"][home_index + 1] == str(options.ficelle_home)
    assert captured["env"]["FICELLE_HOME"] == str(options.ficelle_home)


def test_options_resolve_relative_homes_from_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = bootstrap.build_parser().parse_args(
        [
            "--ficelle-home",
            "state/ficelle",
            "--hermes-home",
            "../shared/hermes",
        ]
    )

    options = bootstrap.options_from_args(args)

    assert options.ficelle_home == tmp_path / "state" / "ficelle"
    assert options.hermes_home == tmp_path.parent / "shared" / "hermes"
    assert options.ficelle_home.is_absolute()
    assert options.hermes_home.is_absolute()
    assert options.ficelle_home_explicit is True


def test_options_keep_symbolic_link_homes_for_installer_validation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "real-home"
    target.mkdir()
    ficelle_home = tmp_path / "ficelle-link"
    hermes_home = tmp_path / "hermes-link"
    ficelle_home.symlink_to(target, target_is_directory=True)
    hermes_home.symlink_to(target, target_is_directory=True)
    args = bootstrap.build_parser().parse_args(
        ["--ficelle-home", ficelle_home.name, "--hermes-home", hermes_home.name],
    )

    options = bootstrap.options_from_args(args)

    assert options.ficelle_home == ficelle_home
    assert options.hermes_home == hermes_home


def test_options_mark_environment_home_explicit(monkeypatch, tmp_path):
    configured_home = tmp_path / "environment-home"
    monkeypatch.setenv("FICELLE_HOME", str(configured_home))

    options = bootstrap.options_from_args(bootstrap.build_parser().parse_args([]))

    assert options.ficelle_home == configured_home
    assert options.ficelle_home_explicit is True


def test_options_keep_canonical_default_home_implicit(monkeypatch, tmp_path):
    monkeypatch.delenv("FICELLE_HOME", raising=False)
    monkeypatch.setattr(
        bootstrap.Path,
        "home",
        classmethod(lambda _cls: tmp_path),
    )

    options = bootstrap.options_from_args(bootstrap.build_parser().parse_args([]))

    assert options.ficelle_home == tmp_path / ".ficelle"
    assert options.ficelle_home_explicit is False


def test_options_reuse_the_home_a_previous_setup_persisted(monkeypatch, tmp_path):
    custom_home = tmp_path / "custom-ficelle"
    custom_home.mkdir()
    pointer = tmp_path / ".config" / "ficelle" / "active-home"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"ficelle_home": str(custom_home)}), encoding="utf-8")
    monkeypatch.delenv("FICELLE_HOME", raising=False)
    monkeypatch.setattr(bootstrap.Path, "home", classmethod(lambda _cls: tmp_path))

    options = bootstrap.options_from_args(bootstrap.build_parser().parse_args([]))

    assert options.ficelle_home == custom_home
    assert options.ficelle_home_explicit is True


def test_a_non_utf8_active_home_pointer_reads_as_unreadable(monkeypatch, tmp_path):
    pointer = tmp_path / ".config" / "ficelle" / "active-home"
    pointer.parent.mkdir(parents=True)
    pointer.write_bytes(b"\xff\xfe/not-utf8")
    monkeypatch.delenv("FICELLE_HOME", raising=False)
    monkeypatch.setattr(bootstrap.Path, "home", classmethod(lambda _cls: tmp_path))

    options = bootstrap.options_from_args(bootstrap.build_parser().parse_args([]))

    assert options.ficelle_home == tmp_path / ".ficelle"
    assert options.ficelle_home_explicit is False


def test_install_wheel_falls_back_to_uv_when_pip_is_missing(monkeypatch, tmp_path):
    calls = []
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    wheel.write_text("placeholder")
    options = make_options(tmp_path)

    def fake_run(command, *, dry_run, env=None, cwd=None):
        calls.append(
            (
                command,
                env.get("HERMES_HOME") if env else None,
                env.get("FICELLE_HOME") if env else None,
            )
        )
        if command[:3] == ["/tmp/hermes-python", "-m", "pip"]:
            return bootstrap.CommandResult(command, 1, stderr="/tmp/hermes-python: No module named pip")
        return bootstrap.CommandResult(command, 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/opt/homebrew/bin/uv" if name == "uv" else None)

    bootstrap.install_wheel(options, Path("/tmp/hermes-python"), wheel)

    assert calls == [
        (["/tmp/hermes-python", "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)], None, str(tmp_path / ".ficelle")),
        (["/opt/homebrew/bin/uv", "pip", "install", "--no-config", "--python", "/tmp/hermes-python", "--reinstall", "--no-deps", str(wheel)], None, str(tmp_path / ".ficelle")),
    ]


def test_pro_requirements_fall_back_to_uv_when_pip_is_missing(
    monkeypatch,
    tmp_path,
):
    calls = []
    options = make_options(tmp_path)

    def fake_run(command, *, dry_run, env=None, cwd=None):
        calls.append(command)
        if command[:3] == ["/tmp/ficelle-python", "-m", "pip"]:
            return bootstrap.CommandResult(
                command,
                1,
                stderr="/tmp/ficelle-python: No module named pip",
            )
        return bootstrap.CommandResult(command, 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name: "/opt/homebrew/bin/uv" if name == "uv" else None,
    )

    bootstrap.install_requirements(
        options,
        Path("/tmp/ficelle-python"),
        bootstrap.PRO_RUNTIME_REQUIREMENTS,
    )

    assert calls == [
        [
            "/tmp/ficelle-python",
            "-m",
            "pip",
            "install",
            "cryptography>=42",
        ],
        [
            "/opt/homebrew/bin/uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            "/tmp/ficelle-python",
            "cryptography>=42",
        ],
    ]


def test_runtime_check_falls_back_to_uv_when_pip_is_missing(monkeypatch, tmp_path):
    calls = []
    options = make_options(tmp_path)

    def fake_run(command, *, dry_run, env=None, cwd=None):
        calls.append(command)
        if command[:3] == ["/tmp/ficelle-python", "-m", "pip"]:
            return bootstrap.CommandResult(
                command,
                1,
                stderr="/tmp/ficelle-python: No module named pip",
            )
        return bootstrap.CommandResult(command, 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name: "/opt/homebrew/bin/uv" if name == "uv" else None,
    )

    bootstrap.check_runtime(options, Path("/tmp/ficelle-python"))

    assert calls == [
        ["/tmp/ficelle-python", "-m", "pip", "check"],
        ["/opt/homebrew/bin/uv", "pip", "check", "--no-config", "--python", "/tmp/ficelle-python"],
    ]


def test_download_local_wheel_verifies_sha256_path(tmp_path):
    source = tmp_path / "source" / "ficelle_router-0.1.2-py3-none-any.whl"
    source.parent.mkdir()
    source.write_bytes(b"wheel-bytes")
    options = make_options(tmp_path, wheel_url=str(source))

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.read_bytes() == b"wheel-bytes"
    assert downloaded.name == source.name


def test_download_local_wheel_dry_run_allows_nonexistent_path(tmp_path):
    source = tmp_path / "not-built-yet" / "ficelle_pro-1.0-py3-none-any.whl"
    options = make_options(
        tmp_path,
        wheel_url=str(source),
        dry_run=True,
    )

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded == tmp_path / "download" / source.name


def test_default_wheel_endpoint_uses_valid_fallback_filename():
    assert f"/api/releases/{bootstrap.CORE_VERSION}/wheel" in bootstrap.DEFAULT_WHEEL_URL
    assert bootstrap.wheel_filename_from_url(bootstrap.DEFAULT_WHEEL_URL) == (
        "ficelle_pro-0-py3-none-any.whl"
    )


def test_default_core_wheel_is_versioned_and_pinned():
    # Derived from CORE_VERSION so a release bump does not have to edit this test. What is
    # asserted is the shape: a semantic version pointing at a *tagged* release asset (never
    # `main` or `latest`), with a full-length hash pinned next to it.
    assert re.fullmatch(r"\d+\.\d+\.\d+", bootstrap.CORE_VERSION)
    assert (
        f"/releases/download/v{bootstrap.CORE_VERSION}/"
        in bootstrap.DEFAULT_CORE_WHEEL_URL
    )
    assert bootstrap.DEFAULT_CORE_WHEEL_URL.endswith(
        f"/ficelle_router-{bootstrap.CORE_VERSION}-py3-none-any.whl"
    )
    assert len(bootstrap.DEFAULT_CORE_SHA256) == 64


def test_documented_bootstrap_urls_use_one_immutable_public_commit():
    root = BOOTSTRAP_PATH.parents[1]
    paths = [root / "README.md", BOOTSTRAP_PATH]
    # The public mirror relocates public-readme.md to README.md and omits install.md.
    paths.extend(
        path for path in (root / "docs/install.md", root / "docs/public-readme.md")
        if path.exists()
    )
    refs = []
    for path in paths:
        matches = re.findall(
            r"https://raw\.githubusercontent\.com/TheBlueHouse75/ficelle-open-core/"
            r"([^/\s]+)/scripts/bootstrap-ficelle\.py",
            path.read_text(encoding="utf-8"),
        )
        assert matches, path
        assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in matches), path
        refs.extend(matches)
    assert len(set(refs)) == 1


def test_wheel_filename_rejects_non_version_placeholder():
    assert (
        bootstrap.safe_wheel_filename("ficelle_pro-latest-py3-none-any.whl")
        is None
    )


def test_wheel_filename_rejects_invalid_versions():
    for version in ("1..2", "1!", "1+", "1_2"):
        assert bootstrap.safe_wheel_filename(
            f"ficelle_pro-{version}-py3-none-any.whl"
        ) is None


def test_remote_download_honors_valid_content_disposition(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap,
        "open_wheel",
        lambda request, timeout, authenticated: _WheelResponse(
            "ficelle_pro-1.2.3-py3-none-any.whl"
        ),
    )
    options = make_options(
        tmp_path,
        wheel_url=bootstrap.DEFAULT_WHEEL_URL,
        license_key="FICL-SECRET",
    )

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.name == "ficelle_pro-1.2.3-py3-none-any.whl"
    assert downloaded.read_bytes() == b"wheel-bytes"


def test_remote_download_rejects_unsafe_advertised_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap,
        "open_wheel",
        lambda request, timeout, authenticated: _WheelResponse(
            "../../not-a-wheel.whl"
        ),
    )
    options = make_options(tmp_path, wheel_url=bootstrap.DEFAULT_WHEEL_URL)

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.name == bootstrap.FALLBACK_WHEEL_FILENAME
    assert downloaded.parent == tmp_path / "download"


def test_remote_download_refuses_insecure_non_loopback_url(monkeypatch, tmp_path):
    called = False

    def open_url(request, timeout, authenticated):
        nonlocal called
        called = True
        return _WheelResponse("ficelle_pro-1.0-py3-none-any.whl")

    monkeypatch.setattr(bootstrap, "open_wheel", open_url)
    options = make_options(
        tmp_path,
        wheel_url="http://packages.example.test/ficelle_pro-1.0-py3-none-any.whl",
        license_key="FICL-SECRET",
    )

    try:
        bootstrap.download_wheel(options, tmp_path / "download")
    except SystemExit as exc:
        assert "insecure URL" in str(exc)
    else:
        raise AssertionError("expected an insecure remote URL to be refused")
    assert called is False


def test_remote_download_allows_loopback_http(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap,
        "open_wheel",
        lambda request, timeout, authenticated: _WheelResponse(
            "ficelle_pro-1.0-py3-none-any.whl"
        ),
    )
    options = make_options(
        tmp_path,
        wheel_url="http://127.0.0.1:8799/wheel",
    )

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.name == "ficelle_pro-1.0-py3-none-any.whl"


def test_redirect_handler_refuses_cross_origin_with_license_key():
    handler = bootstrap.SameOriginRedirectHandler()
    request = bootstrap.urllib.request.Request(
        bootstrap.DEFAULT_WHEEL_URL,
        headers={"Authorization": "Bearer FICL-SECRET"},
    )

    try:
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://cdn.example.invalid/ficelle_pro-1.0-py3-none-any.whl",
        )
    except bootstrap.urllib.error.URLError as exc:
        assert "cross-origin" in str(exc.reason)
    else:
        raise AssertionError("expected a cross-origin redirect to be refused")


def test_public_wheel_uses_normal_redirects_but_authenticated_wheel_does_not(
    monkeypatch,
):
    request = bootstrap.urllib.request.Request(
        bootstrap.DEFAULT_CORE_WHEEL_URL
    )
    calls = []
    public_response = object()
    private_response = object()
    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda req, timeout: calls.append(("public", req, timeout))
        or public_response,
    )
    monkeypatch.setattr(
        bootstrap,
        "open_same_origin",
        lambda req, timeout: calls.append(("private", req, timeout))
        or private_response,
    )

    assert bootstrap.open_wheel(
        request,
        timeout=120,
        authenticated=False,
    ) is public_response
    assert bootstrap.open_wheel(
        request,
        timeout=120,
        authenticated=True,
    ) is private_response
    assert [kind for kind, _request, _timeout in calls] == [
        "public",
        "private",
    ]


def test_sha256_verification_rejects_mismatch(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")

    try:
        bootstrap.verify_sha256(wheel, "0" * 64)
    except SystemExit as exc:
        assert "SHA256 mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch")


def test_remote_pro_wheel_is_verified_with_the_installed_core_root(
    monkeypatch,
    tmp_path,
):
    wheel = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    attestation = tmp_path / "release.attestation.json"
    attestation.write_text("{}", encoding="utf-8")
    options = make_options(
        tmp_path,
        wheel_url="https://install.example.test/api/releases/0.1.3/wheel",
        license_key="FICL-SECRET",
    )
    commands = []
    monkeypatch.setattr(
        bootstrap,
        "download_pro_attestation",
        lambda _options, _directory: attestation,
    )

    def run(command, *, dry_run, env=None, cwd=None):
        commands.append(command)
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", run)

    bootstrap.verify_pro_wheel_integrity(
        options,
        Path("/ficelle/bin/python"),
        wheel,
        tmp_path,
    )

    assert commands == [[
        "/ficelle/bin/python",
        "-m",
        "ficelle.artifact_integrity",
        "verify",
        str(wheel),
        str(attestation),
    ]]


def test_remote_pro_wheel_stops_when_signature_verification_fails(
    monkeypatch,
    tmp_path,
):
    wheel = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    attestation = tmp_path / "release.attestation.json"
    attestation.write_text("{}", encoding="utf-8")
    options = make_options(
        tmp_path,
        wheel_url="https://install.example.test/api/releases/0.1.3/wheel",
        license_key="FICL-SECRET",
    )
    monkeypatch.setattr(
        bootstrap,
        "download_pro_attestation",
        lambda _options, _directory: attestation,
    )
    monkeypatch.setattr(
        bootstrap,
        "run_command",
        lambda *args, **kwargs: bootstrap.CommandResult([], 2),
    )

    with pytest.raises(SystemExit, match="attestation verification failed"):
        bootstrap.verify_pro_wheel_integrity(
            options,
            Path("/ficelle/bin/python"),
            wheel,
            tmp_path,
        )


def test_attestation_download_uses_the_license_key_and_bounded_sidecar(
    monkeypatch,
    tmp_path,
):
    captured = {}
    options = make_options(
        tmp_path,
        wheel_url="https://install.example.test/api/releases/0.1.3/wheel?channel=stable",
        license_key="FICL-SECRET",
    )

    def open_attestation(request, *, timeout, authenticated):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["authenticated"] = authenticated
        return io.BytesIO(b'{"schema_version":1}')

    monkeypatch.setattr(bootstrap, "open_wheel", open_attestation)

    path = bootstrap.download_pro_attestation(options, tmp_path)

    assert captured == {
        "url": "https://install.example.test/api/releases/0.1.3/wheel.attestation?channel=stable",
        "authorization": "Bearer FICL-SECRET",
        "authenticated": True,
    }
    assert path.read_bytes() == b'{"schema_version":1}'


def test_pro_wheel_must_require_the_bootstrap_core_version(tmp_path):
    compatible = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    write_pro_wheel(
        compatible,
        f"ficelle-router=={bootstrap.CORE_VERSION}",
    )
    bootstrap.verify_pro_core_compatibility(
        compatible,
        dry_run=False,
    )

    # Derived from the version this guard actually reads, never a literal — see
    # `unreleasable_version`.
    incompatible_version = unreleasable_version(bootstrap.CORE_VERSION)
    incompatible = tmp_path / f"ficelle_pro-{incompatible_version}-py3-none-any.whl"
    write_pro_wheel(incompatible, f"ficelle-router=={incompatible_version}")
    try:
        bootstrap.verify_pro_core_compatibility(
            incompatible,
            dry_run=False,
        )
    except SystemExit as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("expected incompatible Pro wheel to be refused")


def test_explicit_python_is_only_the_base_runtime(monkeypatch, tmp_path):
    explicit_python = Path("/tmp/base-python")
    options = make_options(tmp_path, python=str(explicit_python))
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)

    assert bootstrap.resolve_base_python(options) == explicit_python


def test_python_probe_converts_execution_error_to_unusable(monkeypatch, tmp_path):
    candidate = tmp_path / "python"
    candidate.write_text("not executable", encoding="utf-8")

    def fail_run(*_args, **_kwargs):
        raise PermissionError("blocked")

    monkeypatch.setattr(bootstrap.subprocess, "run", fail_run)

    result = bootstrap.python_version_result(candidate)

    assert result.returncode == 1
    assert "PermissionError" in result.stderr


@pytest.mark.parametrize("selection", ["python3", "./python3"])
def test_base_python_preserves_path_lookup_and_explicit_relative_paths(monkeypatch, tmp_path, selection):
    monkeypatch.chdir(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python3").symlink_to(sys.executable)
    (tmp_path / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(tools))
    options = make_options(tmp_path, python=selection)
    expected = tools / "python3" if selection == "python3" else tmp_path / "python3"

    assert bootstrap.resolve_base_python(options) == expected


def test_base_python_missing_on_path_does_not_execute_a_local_namesake(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tools = tmp_path / "empty-path"
    tools.mkdir()
    (tmp_path / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(tools))

    with pytest.raises(SystemExit, match="not found on PATH"):
        bootstrap.resolve_base_python(make_options(tmp_path, python="python3"))


def test_generic_install_creates_an_isolated_runtime(
    monkeypatch,
    tmp_path,
):
    options = make_options(tmp_path, target="generic")
    selected = Path("/usr/bin/python3")
    created = []
    runtime = bootstrap.venv_python(options.venv)

    def fake_run(command, *, dry_run, env=None, cwd=None):
        created.append(command)
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)

    python = bootstrap.prepare_ficelle_python(options, selected)

    assert python == runtime
    assert created == [
        [str(selected), "-m", "venv", str(options.venv)]
    ]


def test_generic_install_falls_back_to_uv_venv(monkeypatch, tmp_path):
    options = make_options(tmp_path, target="generic")
    selected = Path("/usr/bin/python3")
    runtime = bootstrap.venv_python(options.venv)
    calls = []

    def fake_run(command, *, dry_run, env=None, cwd=None):
        calls.append(command)
        if command[:3] == [str(selected), "-m", "venv"]:
            return bootstrap.CommandResult(command, 1)
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        return bootstrap.CommandResult(command, 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name: "/usr/local/bin/uv" if name == "uv" else None,
    )
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)

    assert bootstrap.prepare_ficelle_python(options, selected) == runtime
    assert calls == [
        [str(selected), "-m", "venv", str(options.venv)],
        [
            "/usr/local/bin/uv",
            "venv",
            "--no-config",
            "--python",
            str(selected),
            str(options.venv),
        ],
    ]


def test_failed_venv_creation_leaves_no_partial_runtime(monkeypatch, tmp_path):
    # Ubuntu without python3-venv: `python -m venv` writes bin/python, then fails on ensurepip.
    # Reusing that interpreter on the next run fails later with "No module named pip".
    options = make_options(tmp_path, target="generic")
    runtime = bootstrap.venv_python(options.venv)

    def fake_run(command, *, dry_run, env=None, cwd=None):
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        return bootstrap.CommandResult(list(command), 1)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)

    with pytest.raises(SystemExit, match="isolated Python environment creation failed"):
        bootstrap.prepare_ficelle_python(options, Path("/usr/bin/python3"))

    assert not options.venv.exists()


def test_explicit_python_still_creates_an_isolated_runtime(monkeypatch, tmp_path):
    selected = Path("/custom/python")
    options = make_options(
        tmp_path,
        target="generic",
        python=str(selected),
    )

    runtime = bootstrap.venv_python(options.venv)
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)

    assert bootstrap.prepare_ficelle_python(options, selected) == runtime


def test_run_license_activation_passes_key_via_env_not_argv(monkeypatch, tmp_path):
    options = make_options(tmp_path, license_key="SK-SECRET-999")
    captured = {}
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", "/stale/inherited/runtime")

    def fake_run(command, *, dry_run, env=None, cwd=None):
        captured["command"] = list(command)
        captured["env"] = env
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    bootstrap.run_license_activation(options, Path("/tmp/ficelle-python"))

    assert captured["command"] == [str(Path("/tmp/ficelle-python")), "-m", "ficelle", "license", "activate"]
    assert "SK-SECRET-999" not in " ".join(captured["command"])  # never on the command line (it is echoed)
    assert captured["env"]["FICELLE_LICENSE_KEY"] == "SK-SECRET-999"  # passed through the environment
    assert "FICELLE_RUNTIME_DIR" not in captured["env"]


def test_bootstrap_accepts_license_key_only_from_protected_environment(monkeypatch):
    monkeypatch.setenv("FICELLE_LICENSE_KEY", "SK-FROM-ENV")
    parser = bootstrap.build_parser()
    options = bootstrap.options_from_args(parser.parse_args([]))

    assert options.license_key == "SK-FROM-ENV"
    assert "--license-key" not in parser.format_help()

    with pytest.raises(SystemExit):
        parser.parse_args(["--license-key", "SK-IN-ARGV"])


def test_run_license_activation_skips_without_key(monkeypatch, tmp_path):
    options = make_options(tmp_path, license_key=None)
    calls = []
    monkeypatch.setattr(bootstrap, "run_command", lambda *a, **k: calls.append(1) or bootstrap.CommandResult([], 0))
    bootstrap.run_license_activation(options, Path("/tmp/ficelle-python"))
    assert calls == []  # no license key → nothing to activate


def test_run_license_activation_is_best_effort_on_failure(monkeypatch, tmp_path, capsys):
    options = make_options(tmp_path, license_key="SK-x")
    monkeypatch.setattr(bootstrap, "run_command", lambda *a, **k: bootstrap.CommandResult([], 1))
    bootstrap.run_license_activation(
        options,
        Path("/tmp/ficelle-python"),
    )  # must not raise
    assert "activation did not complete" in capsys.readouterr().out


def test_core_environment_removes_inherited_client_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", "/inherited/hermes")
    options = make_options(tmp_path, target="generic")

    env = bootstrap.command_env(options)

    assert env["FICELLE_HOME"] == str(tmp_path / ".ficelle")
    assert "HERMES_HOME" not in env


def test_keep_wheel_uses_ficelle_home_artifacts(monkeypatch, tmp_path):
    source = tmp_path / "ficelle_router-1.0-py3-none-any.whl"
    source.write_bytes(b"wheel")
    options = make_options(
        tmp_path,
        target="generic",
        python=sys.executable,
        core_wheel_url=str(source),
        keep_wheel=True,
    )
    monkeypatch.setattr(bootstrap, "install_wheel", lambda *args: None)
    monkeypatch.setattr(bootstrap, "run_packaged_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "verify_install", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "run_license_activation", lambda *args: None)
    monkeypatch.setattr(bootstrap, "check_runtime", lambda *args: None)
    monkeypatch.setattr(bootstrap, "wheel_runtime_requirements", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        bootstrap,
        "verify_pro_core_compatibility",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(bootstrap, "install_requirements", lambda *args: None)

    assert bootstrap.run_bootstrap(options) == 0

    kept = options.ficelle_home / "artifacts" / source.name
    assert kept.read_bytes() == b"wheel"
    assert not (options.hermes_home / "ficelle" / "artifacts").exists()


def test_core_only_bootstrap_never_downloads_or_installs_pro(
    monkeypatch,
    tmp_path,
):
    core = tmp_path / "ficelle_router-0.1.3-py3-none-any.whl"
    core.write_bytes(b"core")
    options = make_options(
        tmp_path,
        python=sys.executable,
        core_wheel_url=str(core),
    )
    installed = []
    monkeypatch.setattr(
        bootstrap,
        "install_wheel",
        lambda _options, _python, wheel, **kwargs: installed.append(
            (wheel, kwargs)
        ),
    )
    monkeypatch.setattr(bootstrap, "run_packaged_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "verify_install", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "run_license_activation", lambda *args: None)
    monkeypatch.setattr(bootstrap, "check_runtime", lambda *args: None)
    monkeypatch.setattr(bootstrap, "wheel_runtime_requirements", lambda *args, **kwargs: ())
    monkeypatch.setattr(bootstrap, "install_requirements", lambda *args: None)
    monkeypatch.setattr(
        bootstrap,
        "download_wheel",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Pro wheel must not be downloaded without a license key")
        ),
    )

    assert bootstrap.run_bootstrap(options) == 0

    assert [(wheel.name, kwargs) for wheel, kwargs in installed] == [
        (core.name, {})
    ]


def test_pro_bootstrap_installs_core_then_pro_without_dependencies(
    monkeypatch,
    tmp_path,
):
    core = tmp_path / "ficelle_router-0.1.3-py3-none-any.whl"
    pro = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    core.write_bytes(b"core")
    pro.write_bytes(b"pro")
    options = make_options(
        tmp_path,
        python=sys.executable,
        core_wheel_url=str(core),
        wheel_url=str(pro),
        license_key="FICL-SECRET",
        sha256=bootstrap.sha256_file(pro),
    )
    installed = []
    requirements = []
    monkeypatch.setattr(
        bootstrap,
        "verify_pro_core_compatibility",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "install_wheel",
        lambda _options, _python, wheel, **kwargs: installed.append(
            (wheel, kwargs)
        ),
    )
    monkeypatch.setattr(bootstrap, "run_packaged_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "verify_install", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "run_license_activation", lambda *args: None)
    monkeypatch.setattr(bootstrap, "check_runtime", lambda *args: None)
    monkeypatch.setattr(
        bootstrap,
        "wheel_runtime_requirements",
        lambda wheel, **kwargs: (
            ("cryptography>=42",) if wheel.name.startswith("ficelle_pro") else ("requests>=2.31",)
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "install_requirements",
        lambda _options, _python, values: requirements.extend(values),
    )

    assert bootstrap.run_bootstrap(options) == 0

    assert [(wheel.name, kwargs) for wheel, kwargs in installed] == [
        (core.name, {}),
        (pro.name, {}),
    ]
    assert requirements == ["requests>=2.31", "cryptography>=42"]


def _wheel_with_requirements(path: Path, requires: list[str]) -> None:
    lines = [
        "Metadata-Version: 2.4",
        "Name: ficelle-pro",
        "Version: 0.1.3",
        *[f"Requires-Dist: {requirement}" for requirement in requires],
        "",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ficelle_pro-0.1.3.dist-info/METADATA", "\n".join(lines))


def test_wheel_runtime_requirements_accept_named_specifiers_only(tmp_path):
    from packaging.requirements import InvalidRequirement, Requirement

    wheel = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    accepted = [
        f"ficelle-router=={bootstrap.CORE_VERSION}",
        "cryptography>=42",
        'packaging>=24; python_version >= "3.11"',
    ]
    _wheel_with_requirements(wheel, accepted)

    assert bootstrap.wheel_runtime_requirements(wheel) == (
        "cryptography>=42",
        'packaging>=24; python_version >= "3.11"',
    )
    for requirement in accepted:
        parsed = Requirement(requirement)
        assert parsed.url is None


@pytest.mark.parametrize(
    "requirement",
    [
        "",
        "   ",
        "-r /tmp/evil.txt",
        "--index-url https://evil.example/simple",
        "evil @ https://evil.example/p.whl",
        "https://evil.example/p.whl",
        "evil @ file:///tmp/p.whl",
    ],
)
def test_wheel_runtime_requirements_reject_options_and_direct_urls(tmp_path, requirement):
    from packaging.requirements import InvalidRequirement, Requirement

    wheel = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    _wheel_with_requirements(
        wheel,
        [f"ficelle-router=={bootstrap.CORE_VERSION}", requirement],
    )

    with pytest.raises(SystemExit, match="refusing"):
        bootstrap.wheel_runtime_requirements(wheel)
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        parsed = None
    if parsed is not None:
        assert parsed.url


def test_runtime_requirement_rejects_embedded_pip_options():
    with pytest.raises(SystemExit, match="newline"):
        bootstrap.validate_runtime_requirement("pkg>=1\n--extra-index-url https://evil.example/simple")


# One table exercises both implementations: the downloaded bootstrap cannot import Core.
@pytest.mark.parametrize("validator,error_type", [
    (bootstrap.validate_runtime_requirement, SystemExit),
    (update._validate_runtime_requirement, update.UpdateError),
], ids=["bootstrap", "updater"])
@pytest.mark.parametrize("requirement,accepted", [
    ("foo", True),
    ("zipp", True), ("tarfile-ng", True), ("wheel", True), ("py.test", True),
    ("pkg.zip-tools>=1", True),
    ("packaging>=24", True),
    ('requests[security]>=2.31; python_version >= "3.11"', True),
    (" foo!=1.0,>=0.9 ", True),
    ("  foo >= 1.0  ", True),
    ("foo[bar,baz]>=1.0, <2", True),
    ("foo (>=1.0)", True),
    ("foo[bar] (>=1.0,<2)", True),
    ('foo (>=1.0) ; python_version >= "3.11"', True),
    ("foo[bar] ( >=1.0 , <2 ) ; python_version >= '3.11'", True),
    ("foo; python_version >= '3.11' and (os_name == 'posix' or os_name == 'nt')", True),
    ("foo===1.0", True),
    ("foo!=1.0.*,~=2.0", True),
    ("", False), ("   ", False), (None, False), (b"foo", False),
    ("foo @ https://private-value@example.test/foo.whl", False),
    ("foo @ file:///private-value.whl", False),
    ("git+https://private-value@example.test/foo.git", False),
    ("--index-url=https://private-value@example.test/simple", False),
    (" -e private-value", False),
    ("foo>=1 --extra-index-url=private-value", False),
    ("private-value ???", False),
    ("foo ()", False), ("foo (>=1", False), ("foo >=1)", False),
    ("foo ((>=1))", False), ("foo (>=1),<2", False),
    ("foo (>=1,)", False),
] + [
    (f"evil{extension}{suffix}", False)
    for extension in (
        ".whl", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2",
        ".tar.xz", ".txz", ".tar.lz", ".tlz", ".tar.lzma", ".tar.zst", ".egg",
    )
    for extension in (extension, extension.upper())
    for suffix in ("", ">=1", "[extra] (>=1)", "; python_version >= '3.11'")
] + [
    ("foo-1.0-py3-none-any.whl", False),
    ("pkg.zip>=1", False),
] + [
    (f"{prefix}{character}{suffix}", False)
    for character in (
        *(chr(code) for code in range(32)), "\x7f",
        "\x85", "\xa0", "\u1680", *(chr(code) for code in range(0x2000, 0x200c)),
        "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\ufeff", "é",
    )
    for prefix, suffix in (("", "foo"), ("foo", ""), ("foo>=", "1"),
                           ("foo; os_name == '", "'"))
])
def test_bootstrap_and_update_runtime_requirement_policy_agrees(validator, error_type, requirement, accepted):
    if accepted:
        assert validator(requirement) == requirement.strip(" ")
    else:
        with pytest.raises(error_type, match="runtime requirement") as error:
            validator(requirement)
        assert "private-value" not in str(error.value)
        if isinstance(requirement, str) and requirement.strip(" "):
            assert requirement.strip(" ") not in str(error.value)


@pytest.mark.parametrize("requirement", [
    "foo @ https://user:private-value@example.test/foo.whl", "", "   ", "--index-url=private-value",
    "\xa0private-value", "private-value\x00",
])
def test_bootstrap_validates_all_requirements_before_running_pip(monkeypatch, tmp_path, requirement):
    calls = []
    monkeypatch.setattr(bootstrap, "run_command", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(SystemExit, match="runtime requirement") as error:
        bootstrap.install_requirements(make_options(tmp_path), Path("python"), ["packaging>=24", requirement])
    assert calls == []
    assert "private-value" not in str(error.value)


def test_install_requirements_does_not_pass_rejected_specifiers_to_pip(monkeypatch, tmp_path):
    options = make_options(tmp_path)
    calls = []
    monkeypatch.setattr(
        bootstrap,
        "run_command",
        lambda command, **kwargs: calls.append(command) or bootstrap.CommandResult(command, 0),
    )

    with pytest.raises(SystemExit, match="starts with '-'"):
        bootstrap.install_requirements(options, Path("/tmp/ficelle-python"), ["--index-url", "https://evil.example/simple"])

    assert calls == []


def test_local_pro_wheel_requires_a_matching_sha256(tmp_path):
    wheel = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    wheel.write_bytes(b"pro-wheel")
    missing = make_options(tmp_path, wheel_url=str(wheel), sha256=None)

    with pytest.raises(SystemExit, match="SHA-256 is required"):
        bootstrap.verify_pro_wheel_integrity(missing, Path("/tmp/ficelle-python"), wheel, tmp_path)

    mismatched = make_options(tmp_path, wheel_url=str(wheel), sha256="0" * 64)
    with pytest.raises(SystemExit, match="SHA256 mismatch"):
        bootstrap.verify_pro_wheel_integrity(mismatched, Path("/tmp/ficelle-python"), wheel, tmp_path)

    matched = make_options(tmp_path, wheel_url=str(wheel), sha256=bootstrap.sha256_file(wheel))
    bootstrap.verify_pro_wheel_integrity(matched, Path("/tmp/ficelle-python"), wheel, tmp_path)


def test_loopback_pro_wheel_requires_sha256_when_no_digest_is_configured(tmp_path):
    wheel = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    wheel.write_bytes(b"pro-wheel")
    options = make_options(tmp_path, wheel_url="http://127.0.0.1:8799/wheel", sha256=None)

    with pytest.raises(SystemExit, match="SHA-256 is required"):
        bootstrap.verify_pro_wheel_integrity(options, Path("/tmp/ficelle-python"), wheel, tmp_path)



def test_runtime_archive_suffix_lists_agree():
    assert bootstrap._RUNTIME_ARCHIVE_SUFFIXES == update._RUNTIME_ARCHIVE_SUFFIXES


@pytest.mark.parametrize("parent_config", ["uv.toml", "tool.uv", "tool.uv.pip"])
@pytest.mark.parametrize("operation", [
    "bootstrap-wheel", "bootstrap-requirements", "bootstrap-check", "bootstrap-venv",
])
def test_uv_fallbacks_ignore_configuration_in_temporary_parent(
    tmp_path, monkeypatch, parent_config, operation,
):
    import stat
    import subprocess

    parent = tmp_path / "shared-temp"
    parent.mkdir()
    filename = "uv.toml" if parent_config == "uv.toml" else "pyproject.toml"
    table = "" if parent_config == "uv.toml" else f"[{parent_config}]\n"
    (parent / filename).write_text(table + 'index-url = "https://attacker.invalid/simple"\n')
    monkeypatch.setattr(bootstrap.tempfile, "tempdir", str(parent))
    monkeypatch.setenv("TMPDIR", str(parent))
    monkeypatch.setenv("UV_NO_CONFIG", "0")
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/tools/uv" if name == "uv" else None)
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)
    calls = []

    def run(command, **kwargs):
        if command[0] != "/tools/uv":
            return subprocess.CompletedProcess(command, 1, "", "No module named pip")
        calls.append(command)
        group_end = 3 if command[1] == "pip" else 2
        assert command[group_end] == "--no-config"
        assert kwargs["env"]["UV_NO_CONFIG"] == "1"
        cwd = Path(kwargs["cwd"])
        assert cwd.parent == parent
        assert list(cwd.iterdir()) == []
        assert stat.S_IMODE(cwd.stat().st_mode) == 0o700
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    options = make_options(tmp_path)
    python = tmp_path / "python"
    wheel = tmp_path / "ficelle.whl"
    if operation == "bootstrap-wheel":
        bootstrap.install_wheel(options, python, wheel)
    elif operation == "bootstrap-requirements":
        bootstrap.install_requirements(options, python, ["packaging"])
    elif operation == "bootstrap-check":
        bootstrap.check_runtime(options, python)
    else:
        bootstrap.prepare_ficelle_python(options, python)
    assert len(calls) == 1


@pytest.mark.parametrize("operation", [
    "bootstrap-check", "bootstrap-venv", "attestation", "setup", "import-core",
    "import-pro", "activation", "update-check", "version", "update-import-core",
    "update-import-pro", "bootstrap-probe",
])
def test_runtime_checks_and_setup_use_fresh_empty_cwd(tmp_path, monkeypatch, operation):
    import stat
    import subprocess

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ficelle.py").write_text("raise RuntimeError('caller shadow')")
    (tmp_path / "pip.py").write_text("raise RuntimeError('caller shadow')")
    monkeypatch.setenv("FICELLE_LICENSE_KEY", "test-only-key")
    options = bootstrap.options_from_args(bootstrap.build_parser().parse_args([
        "--ficelle-home", "state", "--hermes-home", "client", "--venv", "runtime",
        "--connector", "hermes",
    ]))
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/tools/uv" if name == "uv" else None)
    monkeypatch.setattr(update, "_uv_executable", lambda: "/tools/uv")
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)
    monkeypatch.setattr(bootstrap, "download_pro_attestation", lambda *args: Path("signature.json"))
    directories = []

    def run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        assert cwd.is_absolute() and cwd != tmp_path
        assert cwd.is_dir() and not list(cwd.iterdir())
        assert stat.S_IMODE(cwd.stat().st_mode) == 0o700
        assert Path(command[0]).is_absolute()
        for flag in ("--python", "--package", "--ficelle-home", "--hermes-home"):
            if flag in command:
                assert Path(command[command.index(flag) + 1]).is_absolute()
        if operation == "attestation":
            assert command[-2:] == [str(tmp_path / "wheel.whl"), str(tmp_path / "signature.json")]
        if operation == "bootstrap-venv":
            assert command[-1] == str(options.venv)
        directories.append(cwd)
        (cwd / "leftover").touch()
        if command[1:3] == ["-m", "pip"] or command[1:3] == ["-m", "venv"]:
            return subprocess.CompletedProcess(command, 1, "", "No module named pip")
        return subprocess.CompletedProcess(command, 0, update.CORE_VERSION, "")

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    python = Path("bin/python")
    if operation == "bootstrap-check":
        bootstrap.check_runtime(options, python)
    elif operation == "bootstrap-venv":
        bootstrap.prepare_ficelle_python(options, python)
    elif operation == "attestation":
        options = make_options(tmp_path, wheel_url="https://example.invalid/wheel")
        bootstrap.verify_pro_wheel_integrity(options, python, Path("wheel.whl"), tmp_path)
    elif operation == "setup":
        bootstrap.run_packaged_setup(options, python, Path("wheel.whl"))
    elif operation.startswith("import-"):
        bootstrap.verify_install(options, python, expect_pro=operation == "import-pro")
    elif operation == "activation":
        bootstrap.run_license_activation(options, python)
    elif operation == "update-check":
        update._pip_check()
    elif operation == "version":
        assert update._installed_version() == update.CORE_VERSION
    elif operation == "bootstrap-probe":
        bootstrap.run_python_probe(python, "import sys")
    else:
        update._verify_updated_runtime(expect_pro=operation == "update-import-pro")
    assert len(directories) == (2 if operation in {"bootstrap-check", "bootstrap-venv", "update-check"} else 1)
    assert len(set(directories)) == len(directories)
    assert all(not directory.exists() for directory in directories)


@pytest.mark.parametrize("installer", ["bootstrap", "update"])
@pytest.mark.parametrize("kind", ["wheel", "requirements"])
@pytest.mark.parametrize("missing_pip", [False, True])
def test_runtime_installs_use_fresh_empty_cwd(
    tmp_path, monkeypatch, installer, kind, missing_pip,
):
    import subprocess

    monkeypatch.chdir(tmp_path)
    # An innocent-looking requirement must not resolve to this local project.
    (tmp_path / "packaging").mkdir()
    wheel = Path("ficelle_router.whl")
    wheel.write_bytes(b"wheel")
    directories = []
    uv = Path("tools/uv")
    uv.parent.mkdir()
    uv.touch(mode=0o755)
    monkeypatch.setenv("FICELLE_UV_BIN", str(uv))

    def run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        assert cwd.is_absolute() and cwd != tmp_path
        assert cwd.is_dir() and list(cwd.iterdir()) == []
        if kind == "wheel":
            assert command[-1] == str(tmp_path / wheel)
        if installer == "bootstrap":
            python = command[command.index("--python") + 1] if "--python" in command else command[0]
            assert python == str(tmp_path / "venv/bin/python")
        if "--python" in command:
            assert command[0] == str(tmp_path / uv)
        directories.append(cwd)
        # Even an unsuccessful installer can leave files behind.
        (cwd / "leftover").touch()
        if missing_pip and len(directories) == 1:
            return subprocess.CompletedProcess(command, 1, "", "No module named pip")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: str(uv) if name == "uv" else None)
    if installer == "bootstrap":
        options = make_options(tmp_path, dry_run=False)
        python = Path("venv/bin/python")
        if kind == "wheel":
            bootstrap.install_wheel(options, python, wheel)
        else:
            bootstrap.install_requirements(options, python, ["packaging"])
    elif kind == "wheel":
        update._install_wheel(wheel)
    else:
        update._install_requirements(("packaging",))
    assert len(directories) == (2 if missing_pip else 1)
    assert all(not directory.exists() for directory in directories)
