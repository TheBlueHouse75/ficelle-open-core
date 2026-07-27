from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from ficelle.install import (
    LEGACY_MIGRATION_MARKER,
    MANAGED_CONFIG_BEGIN,
    CommandResult,
    InstallOptions,
    backup_existing_path,
    build_parser,
    collect_preflight_checks,
    configure_hermes,
    copy_plugin_tree,
    dedicated_keychain_path,
    ensure_dedicated_keychain,
    ensure_hermes_compression_plugin_enabled,
    ensure_hermes_toolset_enabled,
    ensure_hermes_plugin_enabled,
    install_plugins,
    legacy_service_listener_status,
    managed_service_status,
    migrate_legacy_ficelle_home,
    options_from_args,
    package_is_local_reference,
    package_install_command,
    probe_target_python,
    resolve_install_target,
    rollback_last_hermes_install,
    run_install,
    uv_package_install_command,
)


def make_options(tmp_path, **overrides):
    values = {
        "package": ".",
        "editable": True,
        "python": "/usr/bin/python3",
        "target": "hermes",
        "ficelle_home": tmp_path / ".ficelle",
        "hermes_home": tmp_path / ".hermes",
        "dry_run": True,
        "skip_package": False,
        "skip_plugin": False,
        "skip_service": False,
        "skip_smoke": False,
        "preflight_only": False,
        "configure_hermes": False,
        "backup_existing": True,
        "rollback": False,
        "ficelle_home_explicit": False,
    }
    values.update(overrides)
    return InstallOptions(**values)


def test_package_install_command_uses_editable_for_local_directory(tmp_path):
    options = make_options(tmp_path, package=str(tmp_path), editable=True)

    assert package_install_command(options) == ["/usr/bin/python3", "-m", "pip", "install", "-e", str(tmp_path)]


@pytest.mark.parametrize(
    "package",
    [
        "https://example.com/ficelle.whl",
        "ficelle @ https://example.com/ficelle.whl",
        "git+https://example.com/ficelle.git",
    ],
)
def test_remote_package_references_are_not_local_paths(package):
    assert package_is_local_reference(package) is False


def test_probe_target_python_normalizes_missing_interpreter(monkeypatch):
    def missing_python(*_args, **_kwargs):
        raise FileNotFoundError("python not found")

    monkeypatch.setattr("ficelle.install.subprocess.run", missing_python)

    result = probe_target_python("/missing/python")

    assert result.returncode == 1
    assert result.stderr.startswith("FileNotFoundError:")


def test_preflight_skips_unused_missing_package(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ficelle.install.probe_target_python",
        lambda python: CommandResult([python], 0, "3.11.14\n"),
    )
    options = make_options(
        tmp_path,
        package=str(tmp_path / "missing.whl"),
        skip_package=True,
        skip_plugin=True,
    )

    checks = collect_preflight_checks(options, "generic")

    package_check = next(check for check in checks if check.name == "package")
    assert package_check.status == "ok"
    assert package_check.detail == "package install skipped"


def test_package_install_command_does_not_edit_wheel(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.0-py3-none-any.whl"
    wheel.write_text("placeholder")
    options = make_options(tmp_path, package=str(wheel), editable=True)

    assert package_install_command(options) == ["/usr/bin/python3", "-m", "pip", "install", str(wheel)]


def test_uv_package_install_command_targets_selected_python(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.0-py3-none-any.whl"
    wheel.write_text("placeholder")
    options = make_options(tmp_path, package=str(wheel), editable=True)

    assert uv_package_install_command(options) == ["uv", "pip", "install", "--python", "/usr/bin/python3", str(wheel)]


def test_uv_package_install_command_uses_editable_for_local_directory(tmp_path):
    options = make_options(tmp_path, package=str(tmp_path), editable=True)

    assert uv_package_install_command(options) == ["uv", "pip", "install", "--python", "/usr/bin/python3", "-e", str(tmp_path)]


def test_backup_existing_path_copies_files(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("provider: old")

    backup = backup_existing_path(source, dry_run=False)

    assert backup is not None
    assert backup.exists()
    assert backup.read_text() == "provider: old"
    assert backup.name.startswith("config.yaml.backup-")


def test_copy_plugin_tree_installs_expected_files_and_backs_up_existing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("plugin")
    (source / "plugin.yaml").write_text("name: ficelle")
    destination = tmp_path / "dest" / "ficelle"
    destination.mkdir(parents=True)
    (destination / "plugin.yaml").write_text("name: old")

    copy_plugin_tree(source, destination, dry_run=False)

    assert (destination / "__init__.py").read_text() == "plugin"
    assert (destination / "plugin.yaml").read_text() == "name: ficelle"
    backups = list(destination.parent.glob("ficelle.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "plugin.yaml").read_text() == "name: old"


def test_run_install_dry_run_orders_package_plugin_service_and_smoke(monkeypatch, tmp_path, capsys):
    copied = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: tmp_path / "plugin-source")
    monkeypatch.setattr("ficelle.install.packaged_compression_plugin_dir", lambda: tmp_path / "compression-plugin-source")
    monkeypatch.setattr("ficelle.install.copy_plugin_tree", lambda source, destination, dry_run, backup_existing=True: copied.append((source, destination, dry_run, backup_existing)))

    options = make_options(tmp_path, dry_run=True)

    assert run_install(options) == 0
    output = capsys.readouterr().out

    assert copied == [
        (tmp_path / "plugin-source", tmp_path / ".hermes" / "plugins" / "model-providers" / "ficelle", True, True),
        (tmp_path / "compression-plugin-source", tmp_path / ".hermes" / "plugins" / "ficelle-compression", True, True),
    ]
    assert "FICELLE_HOME=" in output
    assert "HERMES_HOME=" in output
    assert " /usr/bin/python3 -m pip install" in output
    assert " /usr/bin/python3 -m ficelle.cli install" in output
    assert " /usr/bin/python3 -m ficelle.cli doctor --json" in output
    assert "Ficelle setup complete." in output


def test_run_install_passes_hermes_home_to_service_and_smokes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)
    monkeypatch.setattr("ficelle.install.install_package", lambda options, target=None: None)
    monkeypatch.setattr("ficelle.install.install_plugins", lambda options: None)
    # The macOS-only keychain step is exercised on its own below; keep this test focused
    # on HERMES_HOME propagation and deterministic regardless of the host platform.
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)

    def fake_run(command, *, dry_run, env=None):
        calls.append(
            (
                command,
                env.get("HERMES_HOME") if env else None,
                env.get("FICELLE_HOME") if env else None,
            )
        )
        return CommandResult(command, 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(tmp_path, dry_run=False, skip_package=True, skip_plugin=True)

    assert run_install(options) == 0
    assert calls == [
        (["/usr/bin/python3", "-m", "ficelle.cli", "install"], str(tmp_path / ".hermes"), str(tmp_path / ".ficelle")),
        (["/usr/bin/python3", "-m", "ficelle.cli", "doctor", "--json"], str(tmp_path / ".hermes"), str(tmp_path / ".ficelle")),
        (["/usr/bin/python3", "-m", "ficelle.cli", "health"], str(tmp_path / ".hermes"), str(tmp_path / ".ficelle")),
        (["/usr/bin/python3", "-m", "ficelle.cli", "models"], str(tmp_path / ".hermes"), str(tmp_path / ".ficelle")),
    ]


def test_run_install_stops_on_failed_command(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)

    def fake_run(command, *, dry_run, env=None):
        calls.append(command)
        return CommandResult(command, 17)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(tmp_path, skip_plugin=True, editable=False)

    try:
        run_install(options)
    except SystemExit as exc:
        assert exc.code == 17
    else:
        raise AssertionError("run_install should stop when a command fails")

    assert calls == [["/usr/bin/python3", "-m", "pip", "install", "."]]


def test_run_install_falls_back_to_uv_when_pip_is_missing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)
    # Keep the captured calls focused on the pip->uv fallback; the keychain step is
    # macOS-only and tested separately, so neutralize it here for host independence.
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)

    def fake_run(command, *, dry_run, env=None):
        calls.append(command)
        if command[:3] == ["/usr/bin/python3", "-m", "pip"]:
            return CommandResult(command, 1, stderr="/usr/bin/python3: No module named pip")
        return CommandResult(command, 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: "/opt/homebrew/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        "ficelle.install.active_home_pointer_path",
        lambda: tmp_path / ".config" / "ficelle" / "active-home",
    )
    options = make_options(tmp_path, skip_plugin=True, skip_service=True, skip_smoke=True, editable=False)

    assert run_install(options) == 0
    assert calls == [
        ["/usr/bin/python3", "-m", "pip", "install", "."],
        ["uv", "pip", "install", "--python", "/usr/bin/python3", "."],
    ]


def test_ensure_dedicated_keychain_creates_and_hardens_on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr("ficelle.install.sys.platform", "darwin")
    calls = []

    def fake_run(command, *, dry_run, env=None):
        calls.append(command)
        if command[:2] == ["security", "create-keychain"]:
            # stand in for the real keychain DB the `security` tool would write
            Path(command[-1]).write_text("keychain-db")
        return CommandResult(command, 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path, ficelle_home=tmp_path)

    ensure_dedicated_keychain(options)

    keychain = tmp_path / "ficelle-secrets.keychain-db"
    assert keychain.exists()
    assert calls == [
        ["security", "create-keychain", "-p", "", str(keychain)],
        ["security", "set-keychain-settings", str(keychain)],
    ]
    # never added to the search list — that would risk GUI prompts on unscoped lookups
    assert not any(command[:2] == ["security", "list-keychains"] for command in calls)
    # hardened to owner-only
    assert (keychain.stat().st_mode & 0o777) == 0o600


def test_ensure_dedicated_keychain_is_idempotent_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr("ficelle.install.sys.platform", "darwin")
    keychain = tmp_path / "ficelle-secrets.keychain-db"
    keychain.write_text("existing")
    calls = []
    monkeypatch.setattr("ficelle.install.run_command", lambda command, **_kwargs: calls.append(command) or CommandResult(command, 0))
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path, ficelle_home=tmp_path)

    ensure_dedicated_keychain(options)

    assert calls == []  # no `security` command runs when the keychain already exists
    assert keychain.read_text() == "existing"  # left untouched


def test_ensure_dedicated_keychain_skips_non_macos(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ficelle.install.run_command", lambda command, **_kwargs: calls.append(command) or CommandResult(command, 0))
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path, ficelle_home=tmp_path)

    for platform in ("linux", "win32"):
        monkeypatch.setattr("ficelle.install.sys.platform", platform)
        ensure_dedicated_keychain(options)

    assert calls == []  # Windows/Linux stores need no keychain file to bootstrap
    assert not (tmp_path / "ficelle-secrets.keychain-db").exists()


def test_ensure_dedicated_keychain_warns_and_skips_chmod_on_create_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("ficelle.install.sys.platform", "darwin")
    calls = []

    def fake_run(command, *, dry_run, env=None):
        calls.append(command)
        returncode = 1 if command[:2] == ["security", "create-keychain"] else 0
        return CommandResult(command, returncode)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path, ficelle_home=tmp_path)

    ensure_dedicated_keychain(options)

    # create failed -> no settings call, and resolution still works via the .env fallback
    assert calls == [["security", "create-keychain", "-p", "", str(tmp_path / "ficelle-secrets.keychain-db")]]
    assert "fall back to .env" in capsys.readouterr().err


def test_collect_preflight_reports_dedicated_keychain_on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr("ficelle.install.probe_target_python", lambda python: CommandResult([python], 0, "3.11.14\n"))
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: Path(__file__).parent)
    monkeypatch.setattr("ficelle.install.sys.platform", "darwin")
    options = make_options(tmp_path, skip_plugin=True)

    checks = collect_preflight_checks(options, "generic")

    keychain_checks = [check for check in checks if check.name == "keychain"]
    assert len(keychain_checks) == 1
    assert keychain_checks[0].status == "ok"
    assert "will be created" in keychain_checks[0].detail
    assert str(dedicated_keychain_path(options)) in keychain_checks[0].detail


def test_collect_preflight_omits_keychain_off_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr("ficelle.install.probe_target_python", lambda python: CommandResult([python], 0, "3.11.14\n"))
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: Path(__file__).parent)
    monkeypatch.setattr("ficelle.install.sys.platform", "linux")
    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: "/usr/bin/systemctl")
    options = make_options(tmp_path, skip_plugin=True)

    checks = collect_preflight_checks(options, "generic")

    assert not any(check.name == "keychain" for check in checks)


def test_configure_hermes_creates_config_when_missing(tmp_path):
    options = make_options(tmp_path, dry_run=False, configure_hermes=True)

    configure_hermes(options)

    config = tmp_path / ".hermes" / "config.yaml"
    snippet = tmp_path / ".hermes" / "ficelle" / "hermes-config.snippet.yaml"
    config_text = config.read_text()
    managed_block = config_text.split(MANAGED_CONFIG_BEGIN, 1)[1]
    assert MANAGED_CONFIG_BEGIN in config_text
    assert 'provider: "ficelle"' in config_text
    assert '      - "ficelle/auto-compression"' in config_text
    assert 'model: "ficelle/auto-compression"' in config_text
    assert '    - "ficelle-compression"' in config_text
    assert '  - "ficelle"' in config_text
    assert '    - "ficelle-compression"' not in managed_block
    assert '  - "ficelle"' not in managed_block
    assert 'plugins:' in snippet.read_text()
    assert 'toolsets:' in snippet.read_text()
    assert 'providers:' in snippet.read_text()


def test_configure_hermes_leaves_unmanaged_config_untouched(tmp_path, capsys):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config = hermes_home / "config.yaml"
    config.write_text("model:\n  provider: openrouter\n")
    options = make_options(tmp_path, dry_run=False, configure_hermes=True)

    configure_hermes(options)

    assert config.read_text() == "model:\n  provider: openrouter\n"
    assert (hermes_home / "ficelle" / "hermes-config.snippet.yaml").exists()
    assert "Existing unmanaged Hermes config left untouched" in capsys.readouterr().out


def test_configure_hermes_updates_managed_block_with_backup(tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config = hermes_home / "config.yaml"
    config.write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - \"other-plugin\"\n"
        "toolsets:\n"
        "  - hermes-cli\n"
        f"prefix: true\n{MANAGED_CONFIG_BEGIN}\nold: true\n# END FICELLE MANAGED CONFIG\nsuffix: true\n"
    )
    options = make_options(tmp_path, dry_run=False, configure_hermes=True)

    configure_hermes(options)

    updated = config.read_text()
    assert "old: true" not in updated
    assert 'model: "ficelle/auto-tools"' in updated
    assert "prefix: true" in updated
    assert "suffix: true" in updated
    assert updated.count("plugins:") == 1
    assert '    - "other-plugin"' in updated
    assert '    - "ficelle-compression"' in updated
    assert updated.count("toolsets:") == 1
    assert "  - hermes-cli" in updated
    assert '  - "ficelle"' in updated
    assert list(hermes_home.glob("config.yaml.backup-*"))


def test_ensure_hermes_plugin_enabled_does_not_duplicate_existing_entry():
    text = "plugins:\n  enabled:\n    - \"ficelle-compression\"\n"

    updated = ensure_hermes_plugin_enabled(text)

    assert updated.count("ficelle-compression") == 1


def test_ensure_hermes_plugin_enabled_preserves_pyyaml_list_indentation():
    text = "plugins:\n  enabled:\n  - research-command\nmodel:\n  provider: openrouter\n"

    updated = ensure_hermes_plugin_enabled(text)

    assert "  - research-command\n  - \"ficelle-compression\"" in updated
    assert "    - \"ficelle-compression\"\n  - research-command" not in updated


def test_ensure_hermes_plugin_enabled_ignores_unrelated_mentions():
    text = "disabled_plugins:\n  - ficelle-compression\nplugins:\n  enabled:\n    - \"other-plugin\"\n"

    updated = ensure_hermes_plugin_enabled(text)

    assert '    - "other-plugin"\n    - "ficelle-compression"' in updated


def test_ensure_hermes_plugin_enabled_converts_inline_enabled_list():
    text = "plugins:\n  enabled: [other-plugin]\nmodel:\n  provider: openrouter\n"

    updated = ensure_hermes_plugin_enabled(text)

    assert "  enabled:\n  - other-plugin\n  - \"ficelle-compression\"" in updated
    assert "enabled: [other-plugin]" not in updated


def test_ensure_hermes_plugin_enabled_ignores_nested_enabled_keys():
    text = "plugins:\n  research-command:\n    enabled: true\nmodel:\n  provider: openrouter\n"

    updated = ensure_hermes_plugin_enabled(text)

    assert "  enabled:\n    - \"ficelle-compression\"\n  research-command:" in updated
    assert "  research-command:\n    enabled: true" in updated


def test_ensure_hermes_toolset_enabled_preserves_existing_toolsets():
    text = "toolsets:\n  - hermes-cli\nmodel:\n  provider: openrouter\n"

    updated = ensure_hermes_toolset_enabled(text)

    assert "  - hermes-cli\n  - \"ficelle\"" in updated


def test_ensure_hermes_toolset_enabled_converts_inline_list():
    text = "toolsets: [hermes-cli]\nmodel:\n  provider: openrouter\n"

    updated = ensure_hermes_toolset_enabled(text)

    assert "toolsets:\n  - hermes-cli\n  - \"ficelle\"" in updated


def test_ensure_hermes_compression_plugin_enabled_does_not_duplicate_toolsets():
    text = "toolsets:\n  - hermes-cli\nmodel:\n  provider: openrouter\n"

    updated = ensure_hermes_compression_plugin_enabled(text)

    assert updated.count("plugins:") == 1
    assert updated.count("toolsets:") == 1
    assert "  - hermes-cli\n  - \"ficelle\"" in updated


def test_collect_preflight_warns_for_unmanaged_config(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model:\n  provider: openrouter\n")
    monkeypatch.setattr("ficelle.install.probe_target_python", lambda python: CommandResult([python], 0, "3.11.14\n"))
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: Path(__file__).parent)
    monkeypatch.setattr("ficelle.install.sys.platform", "darwin")
    options = make_options(tmp_path, configure_hermes=True, skip_plugin=True)

    checks = collect_preflight_checks(options, "hermes")

    assert any(check.name == "hermes-config" and check.status == "warn" for check in checks)
    assert not any(check.failed for check in checks)


def test_auto_target_uses_generic_when_hermes_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ficelle.install.candidate_hermes_pythons",
        lambda home, selected_python: (),
    )
    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: None)
    options = make_options(tmp_path, target="auto")

    assert resolve_install_target(options) == "generic"


def test_auto_target_detects_hermes_from_current_interpreter(
    monkeypatch,
    tmp_path,
):
    current_python = tmp_path / "custom-hermes-venv" / "bin" / "python"
    current_python.parent.mkdir(parents=True)
    current_python.write_text("#!/bin/sh\n", encoding="utf-8")
    current_python.chmod(0o700)
    probed = []

    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.setattr("ficelle.install.sys.executable", str(current_python))
    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: None)

    def fake_probe(python):
        probed.append(python)
        return CommandResult(
            [str(python)],
            0 if python == current_python else 1,
        )

    monkeypatch.setattr("ficelle.install.probe_hermes_runtime", fake_probe)

    assert resolve_install_target(make_options(tmp_path, target="auto")) == "hermes"
    assert current_python in probed


def test_auto_target_detects_hermes_from_selected_install_interpreter(
    monkeypatch,
    tmp_path,
):
    selected_python = tmp_path / "selected-hermes-venv" / "bin" / "python"
    selected_python.parent.mkdir(parents=True)
    selected_python.write_text("#!/bin/sh\n", encoding="utf-8")
    selected_python.chmod(0o700)
    probed = []

    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: None)

    def fake_probe(python):
        probed.append(python)
        return CommandResult(
            [str(python)],
            0 if python == selected_python else 1,
        )

    monkeypatch.setattr("ficelle.install.probe_hermes_runtime", fake_probe)
    options = make_options(
        tmp_path,
        target="auto",
        python=str(selected_python),
    )

    assert resolve_install_target(options) == "hermes"
    assert selected_python in probed


def test_auto_target_ignores_directory_and_non_executable_hermes_python(
    monkeypatch,
    tmp_path,
):
    def fail_probe(*_args, **_kwargs):
        raise AssertionError("unusable candidates must not be probed")

    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "ficelle.install.sys.executable",
        str(tmp_path / "missing-current-python"),
    )
    monkeypatch.setattr("ficelle.install.subprocess.run", fail_probe)
    configured = tmp_path / "configured-python"
    configured.mkdir()
    monkeypatch.setenv("HERMES_PYTHON", str(configured))
    options = make_options(
        tmp_path,
        target="auto",
        python=str(tmp_path / "missing-install-python"),
    )

    assert resolve_install_target(options) == "generic"

    configured.rmdir()
    configured.write_text("#!/bin/sh\n", encoding="utf-8")
    configured.chmod(0o600)

    assert resolve_install_target(options) == "generic"


def test_auto_target_continues_after_hermes_python_probe_oserror(
    monkeypatch,
    tmp_path,
):
    def fail_probe(*_args, **_kwargs):
        raise OSError("cannot execute")

    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: None)
    configured = tmp_path / "configured-python"
    configured.write_text("#!/bin/sh\n", encoding="utf-8")
    configured.chmod(0o700)
    monkeypatch.setenv("HERMES_PYTHON", str(configured))
    monkeypatch.setattr("ficelle.install.subprocess.run", fail_probe)
    options = make_options(tmp_path, target="auto")

    assert resolve_install_target(options) == "generic"

    options.hermes_home.mkdir()
    (options.hermes_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    assert resolve_install_target(options) == "hermes"


def test_auto_target_detects_hermes_from_cli_or_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ficelle.install.candidate_hermes_pythons",
        lambda home, selected_python: (),
    )
    monkeypatch.setattr(
        "ficelle.install.shutil.which",
        lambda name: "/usr/local/bin/hermes" if name == "hermes" else None,
    )
    assert resolve_install_target(make_options(tmp_path, target="auto")) == "hermes"

    monkeypatch.setattr("ficelle.install.shutil.which", lambda name: None)
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: {}\n")
    assert resolve_install_target(make_options(tmp_path, target="auto")) == "hermes"


def test_generic_install_passes_only_ficelle_home_and_creates_no_hermes_artifact(
    monkeypatch, tmp_path, capsys
):
    calls = []
    monkeypatch.setenv("HERMES_HOME", "/inherited/hermes")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)
    monkeypatch.setattr("ficelle.install.install_package", lambda options, target=None: None)
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)

    def fake_run(command, *, dry_run, env=None):
        calls.append((command, dict(env or {})))
        return CommandResult(command, 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        skip_package=True,
        skip_plugin=False,
    )

    assert run_install(options) == 0

    assert not options.hermes_home.exists()
    assert calls
    assert all(call_env["FICELLE_HOME"] == str(options.ficelle_home) for _command, call_env in calls)
    assert all("HERMES_HOME" not in call_env for _command, call_env in calls)
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8646/v1" in output
    assert "OpenAI(base_url=" in output


def test_legacy_state_is_copied_when_destination_is_absent(tmp_path):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"kept": true}\n')

    assert migrate_legacy_ficelle_home(options) is True
    assert (options.ficelle_home / "state.json").read_text() == '{"kept": true}\n'
    assert (options.ficelle_home / LEGACY_MIGRATION_MARKER).read_text(
        encoding="utf-8"
    ) == "migrated\n"
    assert (legacy / "state.json").exists()


def test_run_install_quiesces_legacy_service_before_migration(
    monkeypatch,
    tmp_path,
):
    events = []
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        skip_package=True,
        skip_plugin=True,
        skip_service=True,
        skip_smoke=True,
    )
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda *_args: None)
    monkeypatch.setattr(
        "ficelle.install.ensure_dedicated_keychain",
        lambda _options: None,
    )
    monkeypatch.setattr(
        "ficelle.install.persist_active_service_context",
        lambda *_args, **_kwargs: True,
    )

    def fake_run(command, *, dry_run, env=None):
        action = command[-1]
        events.append((action, list(command), dict(env or {})))
        if action == "status":
            return CommandResult(list(command), 1, stderr="not loaded")
        return CommandResult(list(command), 0)

    original_copytree = shutil.copytree

    def tracked_copytree(source, destination, *args, **kwargs):
        events.append(("copy", source, destination))
        return original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    monkeypatch.setattr(
        "ficelle.install.legacy_service_listener_status",
        lambda _legacy_home: "absent",
    )
    monkeypatch.setattr(
        "ficelle.install.shutil.copytree",
        tracked_copytree,
    )

    assert run_install(options) == 0

    assert events[0][0] == "stop"
    assert events[0][1] == [sys.executable, "-m", "ficelle.cli", "stop"]
    assert events[0][2]["FICELLE_RUNTIME_DIR"] == str(legacy)
    assert events[0][2]["PYTHONPATH"].split(os.pathsep)[0] == str(
        Path(__file__).resolve().parents[1] / "src"
    )
    assert events[1][0] == "status"
    assert events[2][0] == "copy"
    assert events[2][1] == legacy


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("active", "active"),
        ("refused", "absent"),
        ("error", "unknown"),
    ],
)
def test_legacy_service_listener_status_is_strict(
    monkeypatch,
    tmp_path,
    outcome,
    expected,
):
    legacy = tmp_path / ".hermes" / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text(
        '{"host": "0.0.0.0", "port": 8765}\n',
        encoding="utf-8",
    )
    endpoints = []

    def connect(endpoint, *, timeout):
        endpoints.append((endpoint, timeout))
        if outcome == "refused":
            raise ConnectionRefusedError
        if outcome == "error":
            raise OSError("indeterminate")
        return _FakeConnection()

    monkeypatch.setattr("ficelle.install.socket.create_connection", connect)

    assert legacy_service_listener_status(legacy) == expected
    assert endpoints == [(("127.0.0.1", 8765), 1)]


class _FakeConnection:
    def close(self):
        return None


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (CommandResult(["status"], 0), "active"),
        (CommandResult(["status"], 1, stderr="not loaded"), "inactive"),
        (
            CommandResult(["status"], 3, stdout="Active: inactive (dead)"),
            "inactive",
        ),
        (CommandResult(["status"], 3, stdout="Active: failed"), "unknown"),
        (CommandResult(["status"], 1, stderr="permission denied"), "unknown"),
    ],
)
def test_managed_service_status_is_conservative(result, expected):
    assert managed_service_status(result) == expected


@pytest.mark.parametrize("stop_returncode", [0, 1])
def test_run_install_aborts_migration_when_legacy_listener_remains_active(
    monkeypatch,
    tmp_path,
    stop_returncode,
):
    copied = []
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        skip_package=True,
        skip_plugin=True,
        skip_service=True,
        skip_smoke=True,
    )
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda *_args: None)

    def fake_run(command, *, dry_run, env=None):
        action = command[-1]
        return CommandResult(
            list(command),
            stop_returncode if action == "stop" else 1,
        )

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    monkeypatch.setattr(
        "ficelle.install.legacy_service_listener_status",
        lambda _legacy_home: "active",
    )
    monkeypatch.setattr(
        "ficelle.install.shutil.copytree",
        lambda *_args, **_kwargs: copied.append(True),
    )

    with pytest.raises(SystemExit, match="inactivity could not be confirmed"):
        run_install(options)

    assert copied == []
    assert not options.ficelle_home.exists()


def test_run_install_aborts_migration_while_service_unit_remains_loaded(
    monkeypatch,
    tmp_path,
):
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        skip_package=True,
        skip_plugin=True,
        skip_service=True,
        skip_smoke=True,
    )
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda *_args: None)
    monkeypatch.setattr(
        "ficelle.install.run_command",
        lambda command, **_kwargs: CommandResult(list(command), 0),
    )
    monkeypatch.setattr(
        "ficelle.install.legacy_service_listener_status",
        lambda _legacy_home: "absent",
    )

    with pytest.raises(SystemExit, match="manager=active"):
        run_install(options)

    assert not options.ficelle_home.exists()


def test_run_install_aborts_migration_when_manager_status_is_unknown(
    monkeypatch,
    tmp_path,
):
    copied = []
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        skip_package=True,
        skip_plugin=True,
        skip_service=True,
        skip_smoke=True,
    )
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda *_args: None)

    def fake_run(command, **_kwargs):
        action = command[-1]
        if action == "status":
            return CommandResult(
                list(command),
                1,
                stderr="permission denied",
            )
        return CommandResult(list(command), 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    monkeypatch.setattr(
        "ficelle.install.legacy_service_listener_status",
        lambda _legacy_home: "absent",
    )
    monkeypatch.setattr(
        "ficelle.install.shutil.copytree",
        lambda *_args, **_kwargs: copied.append(True),
    )

    with pytest.raises(SystemExit, match="manager=unknown"):
        run_install(options)

    assert copied == []
    assert not options.ficelle_home.exists()


def test_legacy_state_merges_into_credential_only_destination(tmp_path):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy_logs = legacy / "logs"
    legacy_logs.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n')
    (legacy / ".env").write_text("OPENROUTER_API_KEY=legacy\n")
    (legacy / "ficelle-secrets.keychain-db").write_text("legacy-keychain")
    (legacy_logs / "routes.jsonl").write_text('{"request_id": "legacy"}\n')
    options.ficelle_home.mkdir(parents=True)
    credential_env = options.ficelle_home / ".env"
    credential_keychain = options.ficelle_home / "ficelle-secrets.keychain-db"
    credential_env.write_text("OPENROUTER_API_KEY=current\n")
    credential_keychain.write_text("current-keychain")

    assert migrate_legacy_ficelle_home(options) is True

    assert credential_env.read_text() == "OPENROUTER_API_KEY=current\n"
    assert credential_keychain.read_text() == "current-keychain"
    assert (options.ficelle_home / "state.json").read_text() == '{"legacy": true}\n'
    assert (options.ficelle_home / "logs" / "routes.jsonl").exists()
    assert (legacy / "state.json").exists()


def test_legacy_state_migration_failed_copy_is_retryable(
    monkeypatch,
    tmp_path,
):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    options.ficelle_home.mkdir(parents=True)
    canonical_env = options.ficelle_home / ".env"
    canonical_env.write_text("OPENROUTER_API_KEY=current\n", encoding="utf-8")
    original_copytree = shutil.copytree

    def fail_after_partial_copy(source, destination, *args, **kwargs):
        destination.mkdir(parents=True)
        (destination / "state.json").write_text(
            '{"partial": true}\n',
            encoding="utf-8",
        )
        raise OSError("simulated copy failure")

    monkeypatch.setattr("ficelle.install.shutil.copytree", fail_after_partial_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        migrate_legacy_ficelle_home(options)

    assert canonical_env.read_text(encoding="utf-8") == (
        "OPENROUTER_API_KEY=current\n"
    )
    assert not (options.ficelle_home / "state.json").exists()
    assert list(options.ficelle_home.parent.glob(".ficelle.migration-*")) == []
    assert (legacy / "state.json").read_text(encoding="utf-8") == (
        '{"legacy": true}\n'
    )

    monkeypatch.setattr("ficelle.install.shutil.copytree", original_copytree)

    assert migrate_legacy_ficelle_home(options) is True
    assert canonical_env.read_text(encoding="utf-8") == (
        "OPENROUTER_API_KEY=current\n"
    )
    assert (options.ficelle_home / "state.json").read_text(encoding="utf-8") == (
        '{"legacy": true}\n'
    )


def test_legacy_state_migration_failed_promotion_restores_and_retries(
    monkeypatch,
    tmp_path,
):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    options.ficelle_home.mkdir(parents=True)
    canonical_env = options.ficelle_home / ".env"
    canonical_env.write_text("OPENROUTER_API_KEY=current\n", encoding="utf-8")
    original_rename = Path.rename

    def fail_staging_promotion(path, target):
        if (
            path.name.startswith(".ficelle.migration-")
            and Path(target) == options.ficelle_home
        ):
            raise OSError("simulated promotion failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_promotion)

    with pytest.raises(OSError, match="simulated promotion failure"):
        migrate_legacy_ficelle_home(options)

    assert canonical_env.read_text(encoding="utf-8") == (
        "OPENROUTER_API_KEY=current\n"
    )
    assert not (options.ficelle_home / "state.json").exists()
    assert list(options.ficelle_home.parent.glob(".ficelle.migration-*")) == []
    assert list(
        options.ficelle_home.parent.glob(".ficelle.pre-migration-*")
    ) == []

    monkeypatch.setattr(Path, "rename", original_rename)

    assert migrate_legacy_ficelle_home(options) is True
    assert (options.ficelle_home / "state.json").read_text(encoding="utf-8") == (
        '{"legacy": true}\n'
    )


def test_legacy_state_migration_recovers_interrupted_rename_window(tmp_path):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    previous = options.ficelle_home.with_name(
        ".ficelle.pre-migration-20260727T000000000000Z"
    )
    previous.mkdir(parents=True)
    (previous / ".env").write_text(
        "OPENROUTER_API_KEY=current\n",
        encoding="utf-8",
    )
    staging = options.ficelle_home.with_name(
        ".ficelle.migration-20260727T000000000000Z"
    )
    staging.mkdir()
    (staging / "state.json").write_text('{"partial": true}\n', encoding="utf-8")

    assert migrate_legacy_ficelle_home(options) is True

    assert (options.ficelle_home / ".env").read_text(encoding="utf-8") == (
        "OPENROUTER_API_KEY=current\n"
    )
    assert (options.ficelle_home / "state.json").read_text(encoding="utf-8") == (
        '{"legacy": true}\n'
    )
    assert list(options.ficelle_home.parent.glob(".ficelle.migration-*")) == []
    assert list(
        options.ficelle_home.parent.glob(".ficelle.pre-migration-*")
    ) == []


def test_legacy_state_migration_dry_run_preserves_interrupted_artifacts(
    tmp_path,
):
    options = make_options(tmp_path, target="generic", dry_run=True)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    previous = options.ficelle_home.with_name(
        ".ficelle.pre-migration-20260727T000000000000Z"
    )
    previous.mkdir(parents=True)
    (previous / ".env").write_text(
        "OPENROUTER_API_KEY=current\n",
        encoding="utf-8",
    )
    staging = options.ficelle_home.with_name(
        ".ficelle.migration-20260727T000000000000Z"
    )
    staging.mkdir()
    (staging / "state.json").write_text('{"partial": true}\n', encoding="utf-8")

    assert migrate_legacy_ficelle_home(options) is True

    assert not options.ficelle_home.exists()
    assert (previous / ".env").read_text(encoding="utf-8") == (
        "OPENROUTER_API_KEY=current\n"
    )
    assert (staging / "state.json").read_text(encoding="utf-8") == (
        '{"partial": true}\n'
    )


@pytest.mark.parametrize("had_previous_destination", [False, True])
def test_legacy_state_migration_recovers_completed_promotion(
    tmp_path,
    had_previous_destination,
    capsys,
):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")
    options.ficelle_home.mkdir(parents=True)
    promoted_state = options.ficelle_home / "state.json"
    promoted_state.write_text('{"promoted": true}\n', encoding="utf-8")
    (options.ficelle_home / LEGACY_MIGRATION_MARKER).write_text(
        "migrated\n",
        encoding="utf-8",
    )
    previous = None
    if had_previous_destination:
        previous = options.ficelle_home.with_name(
            ".ficelle.pre-migration-20260727T000000000000Z"
        )
        previous.mkdir()
        (previous / ".env").write_text(
            "OPENROUTER_API_KEY=current\n",
            encoding="utf-8",
        )

    assert migrate_legacy_ficelle_home(options) is True

    assert promoted_state.read_text(encoding="utf-8") == '{"promoted": true}\n'
    assert (options.ficelle_home / LEGACY_MIGRATION_MARKER).read_text(
        encoding="utf-8"
    ) == "migrated\n"
    assert previous is None or not previous.exists()
    first_output = capsys.readouterr().out
    assert ("Recovered completed Ficelle migration" in first_output) is (
        had_previous_destination
    )

    assert migrate_legacy_ficelle_home(options) is True
    assert promoted_state.read_text(encoding="utf-8") == '{"promoted": true}\n'
    assert "Recovered completed Ficelle migration" not in capsys.readouterr().out


def test_legacy_state_migration_preserves_old_keychain_in_destination(tmp_path):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n')
    (legacy / "hermes-secrets.keychain-db").write_text(
        "legacy-runtime-copy",
        encoding="utf-8",
    )
    options.ficelle_home.mkdir(parents=True)
    old_keychain = options.ficelle_home / "hermes-secrets.keychain-db"
    old_keychain.write_text("current-credential", encoding="utf-8")

    assert migrate_legacy_ficelle_home(options) is True

    assert old_keychain.read_text(encoding="utf-8") == "current-credential"
    assert (options.ficelle_home / "state.json").read_text(encoding="utf-8") == (
        '{"legacy": true}\n'
    )


def test_legacy_state_migration_refuses_existing_runtime_data(tmp_path):
    options = make_options(tmp_path, target="generic", dry_run=False)
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n')
    (legacy / "logs").mkdir()
    options.ficelle_home.mkdir(parents=True)
    current_state = options.ficelle_home / "state.json"
    current_state.write_text('{"current": true}\n')

    assert migrate_legacy_ficelle_home(options) is False

    assert current_state.read_text() == '{"current": true}\n'
    assert not (options.ficelle_home / "logs").exists()
    assert (legacy / "state.json").read_text() == '{"legacy": true}\n'


def test_run_install_uses_legacy_runtime_when_migration_is_refused(
    monkeypatch,
    tmp_path,
):
    command_envs = []
    persisted_contexts = []
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", "/inherited/runtime")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target: None)
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)
    monkeypatch.setattr(
        "ficelle.install.active_home_pointer_path",
        lambda: tmp_path / ".config" / "ficelle" / "active-home",
    )

    def fake_run(command, *, dry_run, env=None):
        command_envs.append(dict(env or {}))
        return CommandResult(
            command,
            1 if command[-1] == "status" else 0,
            stderr="not loaded" if command[-1] == "status" else "",
        )

    def fake_persist(ficelle_home, pointer, *, runtime_dir=None, hermes_home=None):
        persisted_contexts.append((ficelle_home, pointer, runtime_dir, hermes_home))
        return True

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    monkeypatch.setattr(
        "ficelle.install.persist_active_service_context",
        fake_persist,
    )
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        editable=False,
        skip_plugin=True,
        skip_service=True,
    )
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n')
    options.ficelle_home.mkdir(parents=True)
    (options.ficelle_home / "state.json").write_text('{"current": true}\n')

    assert run_install(options) == 0

    assert command_envs
    assert all(
        env["FICELLE_RUNTIME_DIR"] == str(legacy)
        for env in command_envs
    )
    assert persisted_contexts == [
        (
            options.ficelle_home,
            tmp_path / ".config" / "ficelle" / "active-home",
            legacy,
            None,
        )
    ]


def test_run_install_uses_canonical_runtime_after_successful_migration(
    monkeypatch,
    tmp_path,
):
    command_envs = []
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", "/inherited/runtime")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target: None)
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)
    monkeypatch.setattr(
        "ficelle.install.legacy_service_listener_status",
        lambda _legacy_home: "absent",
    )

    def fake_run(command, *, dry_run, env=None):
        command_envs.append(dict(env or {}))
        return CommandResult(
            command,
            1 if command[-1] == "status" else 0,
            stderr="not loaded" if command[-1] == "status" else "",
        )

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        skip_package=True,
        skip_plugin=True,
    )
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n')

    assert run_install(options) == 0

    assert (options.ficelle_home / "state.json").read_text() == '{"legacy": true}\n'
    assert command_envs
    assert all(env["FICELLE_HOME"] == str(options.ficelle_home) for env in command_envs)
    assert all(
        env["FICELLE_RUNTIME_DIR"] == str(legacy)
        for env in command_envs[:2]
    )
    assert all("FICELLE_RUNTIME_DIR" not in env for env in command_envs[2:])


def test_legacy_state_migration_skips_explicit_ficelle_home(tmp_path):
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        ficelle_home=tmp_path / "isolated-home",
        ficelle_home_explicit=True,
    )
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n')

    assert migrate_legacy_ficelle_home(options) is False
    assert not options.ficelle_home.exists()
    assert (legacy / "state.json").read_text() == '{"legacy": true}\n'


def test_run_install_does_not_migrate_into_explicit_ficelle_home(
    monkeypatch,
    tmp_path,
):
    command_envs = []
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", "/inherited/runtime")
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target: None)
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)

    def fake_run(command, *, dry_run, env=None):
        command_envs.append(dict(env or {}))
        return CommandResult(command, 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        ficelle_home=tmp_path / "isolated-home",
        ficelle_home_explicit=True,
        skip_package=True,
        skip_plugin=True,
    )
    options.ficelle_home.mkdir()
    legacy = options.hermes_home / "ficelle"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text('{"legacy": true}\n')

    assert run_install(options) == 0
    assert options.ficelle_home.is_dir()
    assert not (options.ficelle_home / "state.json").exists()
    assert (legacy / "state.json").read_text() == '{"legacy": true}\n'
    assert command_envs
    assert all(env["FICELLE_HOME"] == str(options.ficelle_home) for env in command_envs)
    assert all("FICELLE_RUNTIME_DIR" not in env for env in command_envs)


def test_options_from_args_resolves_relative_home_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        [
            "--ficelle-home",
            "state/ficelle",
            "--hermes-home",
            "../shared/hermes",
        ]
    )

    options = options_from_args(args)

    assert options.ficelle_home == (tmp_path / "state" / "ficelle").resolve()
    assert options.hermes_home == (tmp_path.parent / "shared" / "hermes").resolve()
    assert options.ficelle_home.is_absolute()
    assert options.hermes_home.is_absolute()
    assert options.ficelle_home_explicit is True


def test_options_from_args_marks_environment_home_explicit(monkeypatch, tmp_path):
    configured_home = tmp_path / "environment-home"
    monkeypatch.setenv("FICELLE_HOME", str(configured_home))

    options = options_from_args(build_parser().parse_args([]))

    assert options.ficelle_home == configured_home
    assert options.ficelle_home_explicit is True


def test_options_from_args_marks_canonical_default_home_implicit(monkeypatch, tmp_path):
    monkeypatch.delenv("FICELLE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    options = options_from_args(build_parser().parse_args([]))

    assert options.ficelle_home == tmp_path / ".ficelle"
    assert options.ficelle_home_explicit is False


def test_plugin_reinstall_is_idempotent_and_creates_no_extra_backup(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("plugin")
    (source / "plugin.yaml").write_text("name: ficelle")
    destination = tmp_path / "plugins" / "ficelle"

    assert copy_plugin_tree(source, destination, dry_run=False) is True
    assert copy_plugin_tree(source, destination, dry_run=False) is False
    assert list(destination.parent.glob("ficelle.backup-*")) == []


def test_plugin_reinstall_repairs_directory_in_place_of_packaged_file(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("plugin", encoding="utf-8")
    (source / "plugin.yaml").write_text("name: ficelle", encoding="utf-8")
    destination = tmp_path / "plugins" / "ficelle"
    malformed_entry = destination / "__init__.py"
    malformed_entry.mkdir(parents=True)
    (malformed_entry / "unexpected.txt").write_text("broken", encoding="utf-8")
    (destination / "plugin.yaml").write_text("name: ficelle", encoding="utf-8")

    assert copy_plugin_tree(source, destination, dry_run=False) is True

    assert (destination / "__init__.py").is_file()
    assert (destination / "__init__.py").read_text(encoding="utf-8") == "plugin"
    backups = list(destination.parent.glob("ficelle.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "__init__.py" / "unexpected.txt").read_text(
        encoding="utf-8"
    ) == "broken"


def test_rollback_leaves_fresh_hermes_assets_without_backup_untouched(
    monkeypatch,
    tmp_path,
    capsys,
):
    provider_source = tmp_path / "provider-source"
    compression_source = tmp_path / "compression-source"
    for source in (provider_source, compression_source):
        source.mkdir()
        (source / "__init__.py").write_text(source.name)
        (source / "plugin.yaml").write_text(f"name: {source.name}\n")
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: provider_source)
    monkeypatch.setattr(
        "ficelle.install.packaged_compression_plugin_dir",
        lambda: compression_source,
    )
    options = make_options(tmp_path, target="hermes", dry_run=False, configure_hermes=True)

    install_plugins(options)
    configure_hermes(options)
    assert rollback_last_hermes_install(options) is False

    assert (options.hermes_home / "plugins" / "model-providers" / "ficelle").exists()
    assert (options.hermes_home / "plugins" / "ficelle-compression").exists()
    assert (options.hermes_home / "config.yaml").exists()
    assert (options.hermes_home / "ficelle" / "hermes-config.snippet.yaml").exists()
    output = capsys.readouterr().out
    untouched_paths = (
        options.hermes_home / "plugins" / "model-providers" / "ficelle",
        options.hermes_home / "plugins" / "ficelle-compression",
        options.hermes_home / "config.yaml",
        options.hermes_home / "ficelle" / "hermes-config.snippet.yaml",
    )
    for path in untouched_paths:
        assert output.count(
            f"No backup available for {path}; left current path untouched."
        ) == 1
    assert output.count("No Hermes integration backup was restored.") == 1


def test_plugin_noop_does_not_authorize_rollback_deletion(monkeypatch, tmp_path):
    provider_source = tmp_path / "provider-source"
    compression_source = tmp_path / "compression-source"
    for source in (provider_source, compression_source):
        source.mkdir()
        (source / "__init__.py").write_text(source.name)
        (source / "plugin.yaml").write_text(f"name: {source.name}\n")
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: provider_source)
    monkeypatch.setattr(
        "ficelle.install.packaged_compression_plugin_dir",
        lambda: compression_source,
    )
    options = make_options(tmp_path, target="hermes", dry_run=False)
    install_specs = (
        (
            provider_source,
            options.hermes_home / "plugins" / "model-providers" / "ficelle",
        ),
        (
            compression_source,
            options.hermes_home / "plugins" / "ficelle-compression",
        ),
    )
    for source, destination in install_specs:
        destination.mkdir(parents=True)
        for name in ("__init__.py", "plugin.yaml"):
            (destination / name).write_text((source / name).read_text())

    install_plugins(options)
    assert rollback_last_hermes_install(options) is False

    for source, destination in install_specs:
        assert (destination / "__init__.py").read_text() == (
            source / "__init__.py"
        ).read_text()
        assert (destination / "plugin.yaml").read_text() == (
            source / "plugin.yaml"
        ).read_text()
        assert list(destination.parent.glob(f"{destination.name}.backup-*")) == []


def test_rollback_restores_only_paths_with_backups(monkeypatch, tmp_path):
    provider_source = tmp_path / "provider-source"
    compression_source = tmp_path / "compression-source"
    for source in (provider_source, compression_source):
        source.mkdir()
        (source / "__init__.py").write_text(source.name)
        (source / "plugin.yaml").write_text(f"name: {source.name}\n")
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: provider_source)
    monkeypatch.setattr(
        "ficelle.install.packaged_compression_plugin_dir",
        lambda: compression_source,
    )
    options = make_options(tmp_path, target="hermes", dry_run=False)
    destinations = (
        options.hermes_home / "plugins" / "model-providers" / "ficelle",
        options.hermes_home / "plugins" / "ficelle-compression",
    )
    for destination in destinations:
        destination.mkdir(parents=True)
        (destination / "plugin.yaml").write_text("name: preexisting\n")

    install_plugins(options)
    assert rollback_last_hermes_install(options) is True

    for destination in destinations:
        assert (destination / "plugin.yaml").read_text() == "name: preexisting\n"


def test_rollback_refuses_to_remove_modified_fresh_asset(monkeypatch, tmp_path):
    source = tmp_path / "provider-source"
    source.mkdir()
    (source / "__init__.py").write_text("packaged")
    (source / "plugin.yaml").write_text("name: ficelle\n")
    empty_source = tmp_path / "compression-source"
    empty_source.mkdir()
    (empty_source / "__init__.py").write_text("compression")
    (empty_source / "plugin.yaml").write_text("name: compression\n")
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: source)
    monkeypatch.setattr(
        "ficelle.install.packaged_compression_plugin_dir",
        lambda: empty_source,
    )
    options = make_options(tmp_path, target="hermes", dry_run=False)
    install_plugins(options)
    installed = options.hermes_home / "plugins" / "model-providers" / "ficelle"
    (installed / "__init__.py").write_text("user modification")

    rollback_last_hermes_install(options)

    assert installed.exists()
    assert (installed / "__init__.py").read_text() == "user modification"
