from __future__ import annotations

import importlib.util
import io
import re
import sys
import zipfile
from email.message import Message
from pathlib import Path

import pytest

from conftest import unreleasable_version


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
        "target": "generic",
        "ficelle_home": tmp_path / ".ficelle",
        "hermes_home": tmp_path / ".hermes",
        "configure_hermes": False,
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

    command = bootstrap.setup_command(options, Path("/tmp/hermes-python"), wheel, "hermes")

    assert command[:3] == ["/tmp/hermes-python", "-m", "ficelle.install"]
    assert "--skip-package" in command
    assert "--configure-hermes" in command
    assert command[command.index("--target") + 1] == "hermes"
    assert command[command.index("--ficelle-home") + 1] == str(tmp_path / ".ficelle")
    assert "--hermes-home" in command
    assert str(tmp_path / ".hermes") in command


def test_setup_command_can_disable_config_and_backups(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, configure_hermes=False, backup_existing=False, skip_service=True, skip_smoke=True, dry_run=True)

    command = bootstrap.setup_command(options, Path("/tmp/hermes-python"), wheel, "generic")

    assert "--configure-hermes" not in command
    assert "--no-backup" in command
    assert "--skip-service" in command
    assert "--skip-smoke" in command
    assert "--dry-run" in command


def test_setup_command_exposes_cli_only_when_requested(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, dry_run=True)
    python = Path("/tmp/isolated/bin/python")

    command = bootstrap.setup_command(options, python, wheel, "generic")
    exposed_command = bootstrap.setup_command(
        options,
        python,
        wheel,
        "generic",
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

    def fake_run(command, *, dry_run, env=None):
        captured["command"] = list(command)
        captured["env"] = dict(env or {})
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)

    bootstrap.run_packaged_setup(
        options,
        Path("/tmp/ficelle-python"),
        wheel,
        "generic",
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

    def fake_run(command, *, dry_run, env=None):
        captured["command"] = list(command)
        captured["env"] = dict(env or {})
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)

    bootstrap.run_packaged_setup(
        options,
        Path("/tmp/ficelle-python"),
        wheel,
        "generic",
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


def test_install_wheel_falls_back_to_uv_when_pip_is_missing(monkeypatch, tmp_path):
    calls = []
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    wheel.write_text("placeholder")
    options = make_options(tmp_path)

    def fake_run(command, *, dry_run, env=None):
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

    bootstrap.install_wheel(options, Path("/tmp/hermes-python"), wheel, "generic")

    assert calls == [
        (["/tmp/hermes-python", "-m", "pip", "install", "--force-reinstall", str(wheel)], None, str(tmp_path / ".ficelle")),
        (["uv", "pip", "install", "--python", "/tmp/hermes-python", "--reinstall", str(wheel)], None, str(tmp_path / ".ficelle")),
    ]


def test_pro_requirements_fall_back_to_uv_when_pip_is_missing(
    monkeypatch,
    tmp_path,
):
    calls = []
    options = make_options(tmp_path)

    def fake_run(command, *, dry_run, env=None):
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
        "generic",
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
            "uv",
            "pip",
            "install",
            "--python",
            "/tmp/ficelle-python",
            "cryptography>=42",
        ],
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
    # The docstring's curl line is what users copy; a bump that forgets it would serve an
    # older installer than the wheel it pins.
    assert f"/v{bootstrap.CORE_VERSION}/scripts/bootstrap-ficelle.py" in (
        bootstrap.__doc__ or ""
    )


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


def test_resolve_install_context_probes_explicit_python_once(monkeypatch, tmp_path):
    explicit_python = Path("/tmp/hermes-python")
    options = make_options(tmp_path, target="auto", python=str(explicit_python))
    version_probes = []
    runtime_probes = []
    monkeypatch.setattr(
        bootstrap,
        "python_version_result",
        lambda python: version_probes.append(python)
        or bootstrap.CommandResult([str(python)], 0, "3.11.14\n"),
    )
    monkeypatch.setattr(
        bootstrap,
        "hermes_runtime_result",
        lambda python: runtime_probes.append(python)
        or bootstrap.CommandResult([str(python)], 0),
    )

    target, python = bootstrap.resolve_install_context(options)

    assert target == "hermes"
    assert python == explicit_python
    assert version_probes == [explicit_python]
    assert runtime_probes == [explicit_python]


def test_auto_target_uses_current_python_when_it_contains_hermes(
    monkeypatch,
    tmp_path,
):
    current_python = tmp_path / "current-venv" / "bin" / "python"
    current_python.parent.mkdir(parents=True)
    current_python.touch(mode=0o755)
    options = make_options(tmp_path, target="auto")
    version_probes = []
    runtime_probes = []
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.setattr(bootstrap.sys, "executable", str(current_python))
    monkeypatch.setattr(
        bootstrap.Path,
        "home",
        classmethod(lambda _cls: tmp_path / "isolated-home"),
    )
    monkeypatch.setattr(
        bootstrap,
        "python_version_result",
        lambda python: version_probes.append(python)
        or bootstrap.CommandResult([str(python)], 0, "3.11.14\n"),
    )
    monkeypatch.setattr(
        bootstrap,
        "hermes_runtime_result",
        lambda python: runtime_probes.append(python)
        or bootstrap.CommandResult([str(python)], 0),
    )

    target, python = bootstrap.resolve_install_context(options)

    assert target == "hermes"
    assert python == current_python
    assert version_probes == [current_python]
    assert runtime_probes == [current_python]


def test_detect_hermes_python_skips_unusable_candidates(monkeypatch, tmp_path):
    unusable = tmp_path / "not-an-executable"
    unusable.mkdir()
    options = make_options(tmp_path, target="auto")
    monkeypatch.setattr(
        bootstrap,
        "candidate_hermes_pythons",
        lambda _home: [unusable],
    )

    assert bootstrap.detect_hermes_python(options) is None


def test_python_probe_converts_execution_error_to_unusable(monkeypatch, tmp_path):
    candidate = tmp_path / "python"
    candidate.write_text("not executable", encoding="utf-8")

    def fail_run(*_args, **_kwargs):
        raise PermissionError("blocked")

    monkeypatch.setattr(bootstrap.subprocess, "run", fail_run)

    result = bootstrap.python_version_result(candidate)

    assert result.returncode == 1
    assert "PermissionError" in result.stderr


def test_generic_install_creates_an_isolated_runtime(
    monkeypatch,
    tmp_path,
):
    options = make_options(tmp_path, target="generic")
    selected = Path("/usr/bin/python3")
    created = []
    runtime = bootstrap.venv_python(options.venv)

    def fake_run(command, *, dry_run, env=None):
        created.append(command)
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)

    python, isolated = bootstrap.prepare_target_python(
        options,
        "generic",
        selected,
    )

    assert isolated is True
    assert python == runtime
    assert created == [
        [str(selected), "-m", "venv", str(options.venv)]
    ]


def test_generic_install_falls_back_to_uv_venv(monkeypatch, tmp_path):
    options = make_options(tmp_path, target="generic")
    selected = Path("/usr/bin/python3")
    runtime = bootstrap.venv_python(options.venv)
    calls = []

    def fake_run(command, *, dry_run, env=None):
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

    assert bootstrap.prepare_target_python(
        options,
        "generic",
        selected,
    ) == (runtime, True)
    assert calls == [
        [str(selected), "-m", "venv", str(options.venv)],
        [
            "uv",
            "venv",
            "--python",
            str(selected),
            str(options.venv),
        ],
    ]


def test_explicit_python_is_never_replaced_by_an_isolated_runtime(tmp_path):
    selected = Path("/custom/python")
    options = make_options(
        tmp_path,
        target="generic",
        python=str(selected),
    )

    assert bootstrap.prepare_target_python(
        options,
        "generic",
        selected,
    ) == (selected, False)


def test_run_license_activation_passes_key_via_env_not_argv(monkeypatch, tmp_path):
    options = make_options(tmp_path, license_key="SK-SECRET-999")
    captured = {}
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", "/stale/inherited/runtime")

    def fake_run(command, *, dry_run, env=None):
        captured["command"] = list(command)
        captured["env"] = env
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    bootstrap.run_license_activation(options, Path("/tmp/hermes-python"), "generic")

    assert captured["command"] == [str(Path("/tmp/hermes-python")), "-m", "ficelle", "license", "activate"]
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
    bootstrap.run_license_activation(options, Path("/tmp/hermes-python"), "generic")
    assert calls == []  # no license key → nothing to activate


def test_run_license_activation_is_best_effort_on_failure(monkeypatch, tmp_path, capsys):
    options = make_options(tmp_path, license_key="SK-x")
    monkeypatch.setattr(bootstrap, "run_command", lambda *a, **k: bootstrap.CommandResult([], 1))
    bootstrap.run_license_activation(
        options,
        Path("/tmp/hermes-python"),
        "generic",
    )  # must not raise
    assert "activation did not complete" in capsys.readouterr().out


def test_auto_target_uses_generic_without_hermes_signal(monkeypatch, tmp_path):
    options = make_options(tmp_path, target="auto")
    monkeypatch.setattr(bootstrap, "detect_hermes_python", lambda options: None)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)

    target, python = bootstrap.resolve_install_context(options)

    assert target == "generic"
    assert python == Path(sys.executable)


def test_resolve_install_context_probes_hermes_once(monkeypatch, tmp_path):
    options = make_options(tmp_path, target="auto")
    calls = []
    monkeypatch.setattr(
        bootstrap,
        "detect_hermes_python",
        lambda _options: calls.append("probe") or None,
    )
    monkeypatch.setattr(
        bootstrap,
        "non_python_hermes_installation_signal",
        lambda _options: tmp_path / ".hermes" / "config.yaml",
    )
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)

    target, python = bootstrap.resolve_install_context(options)

    assert target == "hermes"
    assert python == Path(sys.executable)
    assert calls == ["probe"]


def test_auto_target_detects_hermes_from_cli_or_config(monkeypatch, tmp_path):
    options = make_options(tmp_path, target="auto")
    monkeypatch.setattr(bootstrap, "detect_hermes_python", lambda options: None)
    monkeypatch.setattr(bootstrap, "validate_python", lambda python: python)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name: "/usr/local/bin/hermes" if name == "hermes" else None,
    )
    assert bootstrap.resolve_install_context(options)[0] == "hermes"

    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)
    options.hermes_home.mkdir()
    (options.hermes_home / "config.yaml").write_text("model: {}\n")
    assert bootstrap.resolve_install_context(options)[0] == "hermes"


def test_generic_environment_removes_inherited_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", "/inherited/hermes")
    options = make_options(tmp_path, target="generic")

    env = bootstrap.command_env(options, "generic")

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
    monkeypatch.setattr(
        bootstrap,
        "verify_pro_core_compatibility",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "install_requirements",
        lambda *args: (_ for _ in ()).throw(
            AssertionError(
                "Pro dependencies must not be installed without a license key"
            )
        ),
    )

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
        lambda _options, _python, wheel, _target, **kwargs: installed.append(
            (wheel, kwargs)
        ),
    )
    monkeypatch.setattr(bootstrap, "run_packaged_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "verify_install", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "run_license_activation", lambda *args: None)
    monkeypatch.setattr(
        bootstrap,
        "install_requirements",
        lambda *args: (_ for _ in ()).throw(
            AssertionError(
                "Pro dependencies must not be installed without a license key"
            )
        ),
    )
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
        lambda _options, _python, wheel, _target, **kwargs: installed.append(
            (wheel, kwargs)
        ),
    )
    monkeypatch.setattr(bootstrap, "run_packaged_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "verify_install", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "run_license_activation", lambda *args: None)
    monkeypatch.setattr(
        bootstrap,
        "install_requirements",
        lambda _options, _python, _target, values: requirements.extend(values),
    )

    assert bootstrap.run_bootstrap(options) == 0

    assert [(wheel.name, kwargs) for wheel, kwargs in installed] == [
        (core.name, {}),
        (pro.name, {"no_deps": True}),
    ]
    assert requirements == ["cryptography>=42"]
