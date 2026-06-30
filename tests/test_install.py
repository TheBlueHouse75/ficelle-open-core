from __future__ import annotations

from pathlib import Path

from ficelle.install import (
    MANAGED_CONFIG_BEGIN,
    CommandResult,
    InstallOptions,
    backup_existing_path,
    collect_preflight_checks,
    configure_hermes,
    copy_plugin_tree,
    dedicated_keychain_path,
    ensure_dedicated_keychain,
    ensure_hermes_compression_plugin_enabled,
    ensure_hermes_toolset_enabled,
    ensure_hermes_plugin_enabled,
    package_install_command,
    run_install,
    uv_package_install_command,
)


def make_options(tmp_path, **overrides):
    values = {
        "package": ".",
        "editable": True,
        "python": "/usr/bin/python3",
        "hermes_home": tmp_path / ".hermes",
        "dry_run": True,
        "skip_package": False,
        "skip_plugin": False,
        "skip_service": False,
        "skip_smoke": False,
        "preflight_only": False,
        "configure_hermes": False,
        "backup_existing": True,
    }
    values.update(overrides)
    return InstallOptions(**values)


def test_package_install_command_uses_editable_for_local_directory(tmp_path):
    options = make_options(tmp_path, package=str(tmp_path), editable=True)

    assert package_install_command(options) == ["/usr/bin/python3", "-m", "pip", "install", "-e", str(tmp_path)]


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
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options: None)
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
    assert "DRY RUN: HERMES_HOME=" in output
    assert " /usr/bin/python3 -m pip install" in output
    assert " /usr/bin/python3 -m ficelle.cli install" in output
    assert " /usr/bin/python3 -m ficelle.cli doctor --json" in output
    assert "Ficelle setup complete." in output


def test_run_install_passes_hermes_home_to_service_and_smokes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options: None)
    monkeypatch.setattr("ficelle.install.install_package", lambda options: None)
    monkeypatch.setattr("ficelle.install.install_plugins", lambda options: None)
    # The macOS-only keychain step is exercised on its own below; keep this test focused
    # on HERMES_HOME propagation and deterministic regardless of the host platform.
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)

    def fake_run(command, *, dry_run, env=None):
        calls.append((command, env.get("HERMES_HOME") if env else None))
        return CommandResult(command, 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(tmp_path, dry_run=False, skip_package=True, skip_plugin=True)

    assert run_install(options) == 0
    assert calls == [
        (["/usr/bin/python3", "-m", "ficelle.cli", "install"], str(tmp_path / ".hermes")),
        (["/usr/bin/python3", "-m", "ficelle.cli", "doctor", "--json"], str(tmp_path / ".hermes")),
        (["/usr/bin/python3", "-m", "ficelle.cli", "health"], str(tmp_path / ".hermes")),
        (["/usr/bin/python3", "-m", "ficelle.cli", "models"], str(tmp_path / ".hermes")),
    ]


def test_run_install_stops_on_failed_command(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options: None)

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
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options: None)
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
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path)

    ensure_dedicated_keychain(options)

    keychain = tmp_path / "hermes-secrets.keychain-db"
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
    keychain = tmp_path / "hermes-secrets.keychain-db"
    keychain.write_text("existing")
    calls = []
    monkeypatch.setattr("ficelle.install.run_command", lambda command, **_kwargs: calls.append(command) or CommandResult(command, 0))
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path)

    ensure_dedicated_keychain(options)

    assert calls == []  # no `security` command runs when the keychain already exists
    assert keychain.read_text() == "existing"  # left untouched


def test_ensure_dedicated_keychain_skips_non_macos(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ficelle.install.run_command", lambda command, **_kwargs: calls.append(command) or CommandResult(command, 0))
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path)

    for platform in ("linux", "win32"):
        monkeypatch.setattr("ficelle.install.sys.platform", platform)
        ensure_dedicated_keychain(options)

    assert calls == []  # Windows/Linux stores need no keychain file to bootstrap
    assert not (tmp_path / "hermes-secrets.keychain-db").exists()


def test_ensure_dedicated_keychain_warns_and_skips_chmod_on_create_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("ficelle.install.sys.platform", "darwin")
    calls = []

    def fake_run(command, *, dry_run, env=None):
        calls.append(command)
        returncode = 1 if command[:2] == ["security", "create-keychain"] else 0
        return CommandResult(command, returncode)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(tmp_path, dry_run=False, hermes_home=tmp_path)

    ensure_dedicated_keychain(options)

    # create failed -> no settings call, and resolution still works via the .env fallback
    assert calls == [["security", "create-keychain", "-p", "", str(tmp_path / "hermes-secrets.keychain-db")]]
    assert "fall back to .env" in capsys.readouterr().err


def test_collect_preflight_reports_dedicated_keychain_on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr("ficelle.install.probe_target_python", lambda python: CommandResult([python], 0, "3.11.14\n"))
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: Path(__file__).parent)
    monkeypatch.setattr("ficelle.install.sys.platform", "darwin")
    options = make_options(tmp_path, skip_plugin=True)

    checks = collect_preflight_checks(options)

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

    checks = collect_preflight_checks(options)

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

    checks = collect_preflight_checks(options)

    assert any(check.name == "hermes-config" and check.status == "warn" for check in checks)
    assert not any(check.failed for check in checks)
