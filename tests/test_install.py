from __future__ import annotations

import io
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from ficelle.install import (
    MANAGED_CONFIG_BEGIN,
    MANAGED_CONFIG_END,
    CommandResult,
    ClientHomeError,
    HermesPluginConflictError,
    InstallOptions,
    _normalized_plugin_payload,
    backup_existing_path,
    build_parser,
    cli_reachability_notice,
    collect_preflight_checks,
    configure_hermes,
    config_outside_managed_block,
    config_has_user_additions,
    copy_plugin_tree,
    doctor_auth_report,
    dedicated_keychain_path,
    ensure_dedicated_keychain,
    ensure_hermes_compression_plugin_enabled,
    ensure_hermes_toolset_enabled,
    ensure_hermes_plugin_enabled,
    expose_cli_scripts,
    file_sha256,
    hermes_plugin_install_specs,
    hermes_connector_metadata,
    install_plugins,
    PROVIDER_PLUGIN_ENV_KEY,
    PROVIDER_PLUGIN_ENV_PLACEHOLDER,
    installed_cli_command,
    offer_first_key_capture,
    options_from_args,
    package_is_local_reference,
    package_install_command,
    probe_target_python,
    refresh_installed_hermes_integration,
    remove_hermes_connector,
    replace_managed_block,
    rollback_last_hermes_install,
    run_install,
    seed_provider_plugin_env_key,
    uv_package_install_command,
    validated_client_home,
)
from ficelle.connector_registry import load_connectors, register_connector, save_connectors


def make_options(tmp_path, **overrides):
    target = overrides.pop("target", "hermes")
    overrides.pop("rollback", None)
    overrides.pop("configure_hermes", None)
    overrides.pop("ficelle_home_explicit", None)
    values = {
        "package": ".",
        "editable": True,
        "python": "/usr/bin/python3",
        "connectors": ("hermes",) if target == "hermes" else (),
        "ficelle_home": tmp_path / ".ficelle",
        "hermes_home": tmp_path / ".hermes",
        "dry_run": True,
        "skip_package": False,
        "skip_plugin": False,
        "skip_service": False,
        "skip_smoke": False,
        "preflight_only": False,
        "backup_existing": True,
        # The CLI implies this for a piped stdin; the test harness is exactly that, so
        # capture tests opt in to interactivity explicitly.
        "non_interactive": True,
    }
    values.update(overrides)
    return InstallOptions(**values)


def _register_test_hermes_connector(options: InstallOptions) -> None:
    from ficelle.router import parse_env_file

    config_existed = (options.hermes_home / "config.yaml").exists()
    env_path = options.hermes_home / ".env"
    env_file_existed = env_path.exists()
    env_key_existed = bool(parse_env_file(env_path).get(PROVIDER_PLUGIN_ENV_KEY))
    plugin_ownership = install_plugins(options)
    configure_hermes(options)
    register_connector(
        options.ficelle_home,
        "hermes",
        client_home=options.hermes_home,
        metadata=hermes_connector_metadata(
            options,
            config_existed=config_existed,
            env_file_existed=env_file_existed,
            env_key_existed=env_key_existed,
            plugin_ownership=plugin_ownership,
        ),
    )


def test_remove_hermes_connector_is_reversible_for_unchanged_artifacts(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert removed is True
    assert messages == ["Hermes connector artifacts removed."]
    assert load_connectors(options.ficelle_home) == {}
    assert not (options.hermes_home / "config.yaml").exists()
    assert not (options.hermes_home / ".env").exists()
    assert not (options.hermes_home / "plugins" / "model-providers" / "ficelle").exists()


def test_remove_hermes_connector_ignores_bytecode_written_by_hermes(tmp_path):
    # Hermes compiles the provider on first import; that must not read as a user modification.
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    plugin_path = options.hermes_home / "plugins" / "model-providers" / "ficelle"
    (plugin_path / "__pycache__").mkdir(exist_ok=True)
    (plugin_path / "__pycache__" / "__init__.cpython-314.pyc").write_bytes(b"\0compiled")

    removed, _messages = remove_hermes_connector(options.ficelle_home)

    assert removed is True
    assert not plugin_path.exists()


def test_remove_hermes_connector_refuses_modified_artifacts(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config_path = options.hermes_home / "config.yaml"
    config_path.write_text(config_path.read_text() + "user_setting: true\n", encoding="utf-8")

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert removed is False
    assert str(config_path) in messages
    assert "hermes" in load_connectors(options.ficelle_home)
    assert (options.hermes_home / "plugins" / "model-providers" / "ficelle").exists()


def test_remove_hermes_connector_refuses_modified_plugin(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    plugin_path = options.hermes_home / "plugins" / "model-providers" / "ficelle"
    (plugin_path / "user-edit.txt").write_text("modified\n", encoding="utf-8")

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert removed is False
    assert str(plugin_path) in messages


def test_remove_hermes_connector_uses_recorded_plugin_hash_after_package_upgrade(
    tmp_path, monkeypatch
):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    installed_specs = tuple(hermes_plugin_install_specs(options.hermes_home))
    upgraded_source = tmp_path / "upgraded-plugin"
    upgraded_source.mkdir()
    (upgraded_source / "__init__.py").write_text("# newer package\n", encoding="utf-8")
    monkeypatch.setattr(
        "ficelle.install.hermes_plugin_install_specs",
        lambda _home: tuple((upgraded_source, destination) for _source, destination in installed_specs),
    )

    removed, _messages = remove_hermes_connector(options.ficelle_home)

    assert removed is True
    assert all(not destination.exists() for _source, destination in installed_specs)
    assert "hermes" not in load_connectors(options.ficelle_home)


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

    checks = collect_preflight_checks(options)

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
    assert "HERMES_HOME=" not in output
    assert " /usr/bin/python3 -P -m pip install" in output
    assert " /usr/bin/python3 -P -m ficelle.cli install" in output
    assert " /usr/bin/python3 -P -m ficelle.cli doctor --json" in output
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
    assert [hermes_home for _command, hermes_home, _ficelle_home in calls] == [None] * 4
    assert all(ficelle_home == str(tmp_path / ".ficelle") for _command, _hermes_home, ficelle_home in calls)


def test_run_install_checks_core_health_before_mutating_hermes(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    connector_mutations: list[str] = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)
    monkeypatch.setattr(
        "ficelle.install.install_plugins",
        lambda options, runtime_dir=None: connector_mutations.append("plugins") or {},
    )
    monkeypatch.setattr(
        "ficelle.install.configure_hermes",
        lambda options, runtime_dir=None: connector_mutations.append("config"),
    )

    def fake_run(command, *, dry_run, env=None):
        calls.append(command)
        if command[-1] == "health":
            return CommandResult(command, 17)
        return CommandResult(command, 0, stdout='{"auth": {}}\n')

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(
        tmp_path,
        dry_run=False,
        skip_package=True,
        skip_service=True,
    )

    with pytest.raises(SystemExit) as error:
        run_install(options)

    assert error.value.code == 17
    assert calls == [
        ["/usr/bin/python3", "-m", "ficelle.cli", "doctor", "--json"],
        ["/usr/bin/python3", "-m", "ficelle.cli", "health"],
    ]
    assert connector_mutations == []
    assert not options.hermes_home.exists()


def test_run_install_rejects_smoke_skip_before_connector_mutation(monkeypatch, tmp_path):
    mutations: list[str] = []
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)
    monkeypatch.setattr(
        "ficelle.install.ensure_dedicated_keychain",
        lambda options: mutations.append("keychain"),
    )
    options = make_options(
        tmp_path,
        dry_run=False,
        skip_package=True,
        skip_service=True,
        skip_smoke=True,
    )

    with pytest.raises(SystemExit, match="Core must pass doctor, health, and models first"):
        run_install(options)

    assert mutations == []
    assert not options.hermes_home.exists()


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

    assert calls == [["/usr/bin/python3", "-m", "pip", "install", str(Path.cwd())]]


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
    options = make_options(
        tmp_path,
        target="generic",
        skip_plugin=True,
        skip_service=True,
        skip_smoke=True,
        editable=False,
    )

    assert run_install(options) == 0
    assert calls == [
        ["/usr/bin/python3", "-m", "pip", "install", str(Path.cwd())],
        ["uv", "pip", "install", "--python", "/usr/bin/python3", str(Path.cwd())],
    ]


def install_with_doctor_auth(monkeypatch, tmp_path, auth, *, pin_cli=True, **option_overrides):
    """Run a generic install whose `doctor --json` smoke check reports `auth`."""
    # Most of these tests are about *what* setup advises, not how the command is spelled — and
    # how it is spelled depends on whether the test runner's venv happens to be on PATH. Pinned
    # to the bare name by default; the spelling has its own tests, which opt out.
    if pin_cli:
        monkeypatch.setattr("ficelle.install.installed_cli_command", lambda name="ficelle", **_kwargs: name)
    monkeypatch.setattr("ficelle.install.run_preflight", lambda options, target=None: None)
    monkeypatch.setattr("ficelle.install.install_package", lambda options, target=None, runtime_dir=None: None)
    monkeypatch.setattr("ficelle.install.ensure_dedicated_keychain", lambda options: None)

    def fake_run(command, *, dry_run, env=None, input_text=None):
        if command[-2:] == ["doctor", "--json"] and auth is not None:
            return CommandResult(command, 0, stdout=json.dumps({"status": "ok", "auth": auth}))
        return CommandResult(command, 0)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    options = make_options(
        tmp_path,
        target="generic",
        dry_run=False,
        skip_package=True,
        skip_plugin=True,
        **option_overrides,
    )
    assert run_install(options) == 0


def test_run_install_tells_an_unkeyed_install_it_cannot_serve_anything(monkeypatch, tmp_path, capsys):
    """The old closing lines sent a keyless user to `ficelle models`, which passes without a key."""
    install_with_doctor_auth(
        monkeypatch,
        tmp_path,
        {
            "openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY"},
            "nous": {"invokable": False, "reason": "missing NOUS_API_KEY"},
        },
    )
    output = capsys.readouterr().out

    assert "Ficelle setup complete." in output
    assert "No provider API key is configured, so Ficelle cannot serve a completion yet." in output
    assert "ficelle set-key openrouter  # create a key at https://openrouter.ai/keys" in output
    assert "ficelle set-key nous" in output
    assert "does NOT mean a request can be served" in output
    # The one instruction that would have taught the user setup worked when it did not.
    assert "Verify: `ficelle health` and `ficelle models`." not in output


def test_run_install_does_not_offer_to_replace_keys_when_the_store_is_unreadable(
    monkeypatch,
    tmp_path,
    capsys,
):
    reason = "unreadable OPENROUTER_API_KEY from wincred: CredReadW error 1312"
    install_with_doctor_auth(
        monkeypatch,
        tmp_path,
        {"openrouter": {"invokable": False, "reason": reason, "key_source": None}},
    )
    output = capsys.readouterr().out

    assert "ficelle set-key openrouter" not in output
    assert "No provider API key is configured" not in output


def test_run_install_does_not_ask_for_a_key_the_install_already_holds(monkeypatch, tmp_path, capsys):
    """First run is where a `base_url` cleared in `config.json` is met, and it used to be
    answered with `ficelle set-key` for a key sitting in the keychain."""
    install_with_doctor_auth(
        monkeypatch,
        tmp_path,
        {
            "openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY", "key_source": None},
            "mistral": {
                "invokable": False,
                "reason": "missing base_url for provider mistral",
                "key_source": "keychain",
            },
        },
    )
    output = capsys.readouterr().out

    assert "No provider is usable yet, so Ficelle cannot serve a completion." in output
    assert "ficelle set-key openrouter" in output
    assert "ficelle set-key mistral" not in output
    assert "These providers already hold a key, and something else is missing:" in output
    assert "  mistral  # missing base_url for provider mistral" in output
    assert "Storing that key again will not change it" in output


def test_run_install_skips_the_set_key_block_when_every_provider_already_has_a_key(monkeypatch, tmp_path, capsys):
    """The block would otherwise print its "store a key" preamble over an empty command list."""
    install_with_doctor_auth(
        monkeypatch,
        tmp_path,
        {
            "mistral": {
                "invokable": False,
                "reason": "missing base_url for provider mistral",
                "key_source": "keychain",
            }
        },
    )
    output = capsys.readouterr().out

    assert "No provider is usable yet, so Ficelle cannot serve a completion." in output
    assert "ficelle set-key" not in output
    assert "Create a key, then store it" not in output
    assert "  mistral  # missing base_url for provider mistral" in output
    assert "does NOT mean a request can be served" in output


def _keyless_auth():
    return {"openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY"}}


def _interactive_capture(monkeypatch, *, pasted=None, run=None):
    """Record run_command calls; patch getpass only when a paste is given."""
    run = run or (lambda command: CommandResult(command, 0))
    calls = []

    def fake_run(command, *, dry_run, env=None, input_text=None):
        calls.append((command, input_text))
        return run(command)

    monkeypatch.setattr("ficelle.install.run_command", fake_run)
    if pasted is not None:
        monkeypatch.setattr("ficelle.install.getpass.getpass", lambda _prompt: pasted)
    return calls


def test_first_key_capture_stores_via_stdin_and_runs_the_demo(monkeypatch, tmp_path, capsys):
    pasted = "sk-or-v1-" + "a" * 40
    calls = _interactive_capture(monkeypatch, pasted=pasted)
    options = make_options(tmp_path, dry_run=False, non_interactive=False)

    assert offer_first_key_capture(options, {}, ["openrouter", "nous"], cli_command="ficelle") is True
    assert calls[0][0] == ["/usr/bin/python3", "-m", "ficelle.cli", "set-key", "openrouter", "--stdin"]
    assert calls[0][1] == pasted + "\n"  # the key travels on stdin, never argv
    assert calls[1] == (["/usr/bin/python3", "-m", "ficelle.cli", "demo"], None)
    out = capsys.readouterr().out
    assert "https://openrouter.ai/keys" in out
    assert pasted not in out  # the secret is never echoed


def test_first_key_capture_offers_the_first_provider_in_configured_order(monkeypatch, tmp_path):
    # Configured order is the codebase's one expression of provider preference; a user
    # who deliberately ordered another provider first gets that provider offered.
    calls = _interactive_capture(monkeypatch)
    prompts = []
    monkeypatch.setattr("ficelle.install.getpass.getpass", lambda prompt: prompts.append(prompt) or "")
    options = make_options(tmp_path, dry_run=False, non_interactive=False)

    assert offer_first_key_capture(options, {}, ["nous", "openrouter"], cli_command="ficelle") is False
    assert prompts == ["Paste the nous API key to finish now (Enter to skip): "]
    assert calls == []


def test_first_key_capture_treats_eof_as_skip(monkeypatch, tmp_path):
    # Ctrl-D at an Enter-to-skip prompt must decline, not crash a finished install.
    calls = _interactive_capture(monkeypatch)

    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("ficelle.install.getpass.getpass", raise_eof)
    options = make_options(tmp_path, dry_run=False, non_interactive=False)

    assert offer_first_key_capture(options, {}, ["openrouter"], cli_command="ficelle") is False
    assert calls == []


def test_first_key_capture_declines_silently_on_empty_paste(monkeypatch, tmp_path):
    calls = _interactive_capture(monkeypatch, pasted="")
    options = make_options(tmp_path, dry_run=False, non_interactive=False)

    assert offer_first_key_capture(options, {}, ["openrouter"], cli_command="ficelle") is False
    assert calls == []


def test_first_key_capture_respects_the_non_interactive_option(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ficelle.install.getpass.getpass",
        lambda _prompt: pytest.fail("--non-interactive must suppress the prompt"),
    )
    options = make_options(tmp_path, dry_run=False, non_interactive=True)

    assert offer_first_key_capture(options, {}, ["openrouter"], cli_command="ficelle") is False


def test_first_key_capture_does_not_prompt_with_no_keyless_provider(monkeypatch, tmp_path):
    # Keys exist but cannot be used: the caller passes an empty keyless list, and
    # pasting another key is not the fix — the advice block explains what is.
    monkeypatch.setattr(
        "ficelle.install.getpass.getpass",
        lambda _prompt: pytest.fail("an unusable key must route to the advice block, not a prompt"),
    )
    options = make_options(tmp_path, dry_run=False, non_interactive=False)

    assert offer_first_key_capture(options, {}, [], cli_command="ficelle") is False


def test_non_interactive_is_implied_without_an_interactive_terminal(monkeypatch):
    args = build_parser().parse_args([])

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("ficelle.install.sys.stdin", Tty())
    monkeypatch.setattr("ficelle.install.sys.stdout", Tty())
    assert options_from_args(args).non_interactive is False
    assert options_from_args(build_parser().parse_args(["--non-interactive"])).non_interactive is True

    # A piped or captured stream on either side suppresses the prompt: the context
    # lines travel on stdout (the bootstrap captures it), the paste arrives on stdin.
    monkeypatch.setattr("ficelle.install.sys.stdin", io.StringIO())
    assert options_from_args(args).non_interactive is True
    monkeypatch.setattr("ficelle.install.sys.stdin", Tty())
    monkeypatch.setattr("ficelle.install.sys.stdout", io.StringIO())
    assert options_from_args(args).non_interactive is True

    # fd 0 closed at interpreter start (pythonw, provisioning managers): stdin is None.
    monkeypatch.setattr("ficelle.install.sys.stdin", None)
    assert options_from_args(args).non_interactive is True


def test_first_key_capture_reports_a_failed_store_and_keeps_the_advice(monkeypatch, tmp_path, capsys):
    def failing_set_key(command):
        return CommandResult(command, 1 if "set-key" in command else 0)

    calls = _interactive_capture(monkeypatch, pasted="not-a-key", run=failing_set_key)
    options = make_options(tmp_path, dry_run=False, non_interactive=False)

    assert offer_first_key_capture(options, {}, ["openrouter"], cli_command="ficelle") is False
    assert len(calls) == 1  # no demo after a failed store
    assert "run `ficelle set-key openrouter` to retry" in capsys.readouterr().out


def test_run_install_drops_the_advice_block_after_an_inline_key_capture(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("ficelle.install.getpass.getpass", lambda _prompt: "sk-or-v1-" + "b" * 40)
    install_with_doctor_auth(monkeypatch, tmp_path, _keyless_auth(), non_interactive=False)
    output = capsys.readouterr().out

    assert "Paste the openrouter API key" not in output  # getpass prompts off-stream
    assert "Create a free openrouter key at https://openrouter.ai/keys" in output
    # The advice block would re-instruct what the capture just did.
    assert "ficelle set-key openrouter" not in output
    assert "does NOT mean a request can be served" not in output


def test_installed_cli_command_names_the_script_a_shell_cannot_reach(tmp_path):
    """Setup closes on `ficelle set-key ...`. A venv the user never activated answers that
    with `command not found`, on the very first instruction the install gives."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    interpreter.touch()
    (venv_bin / "ficelle").touch()

    # Not on PATH at all.
    assert installed_cli_command(interpreter=interpreter, which=lambda _name: None) == str(venv_bin / "ficelle")
    # On PATH, and it is this install: the bare name is the better advice.
    assert installed_cli_command(interpreter=interpreter, which=lambda _name: str(venv_bin / "ficelle")) == "ficelle"
    # Composes with the bootstrap, which exposes the CLI by symlinking it into ~/.local/bin:
    # resolved through the link it is the same file, so setup stays quiet.
    exposed = tmp_path / "local-bin" / "ficelle"
    exposed.parent.mkdir(parents=True)
    exposed.symlink_to(venv_bin / "ficelle")
    assert installed_cli_command(interpreter=interpreter, which=lambda _name: str(exposed)) == "ficelle"
    # On PATH, but it is a different Ficelle — the full path is what runs the one just installed.
    other = tmp_path / "other" / "ficelle"
    other.parent.mkdir(parents=True)
    other.touch()
    assert installed_cli_command(interpreter=interpreter, which=lambda _name: str(other)) == str(venv_bin / "ficelle")
    # Nothing installed beside this interpreter: a path to a file that is not there would be
    # worse advice than none.
    bare = tmp_path / "empty" / "python"
    bare.parent.mkdir(parents=True)
    assert installed_cli_command(interpreter=bare, which=lambda _name: None) == "ficelle"


def test_cli_reachability_notice_is_silent_when_the_bare_name_works():
    assert cli_reachability_notice("ficelle") == []
    assert cli_reachability_notice("ficelle", expose_command="/opt/venv/bin/ficelle-setup") == []

    lines = cli_reachability_notice("/opt/venv/bin/ficelle")

    assert "not reachable by name" in lines[0]
    # "Put it first", not "add it": a different Ficelle earlier on PATH lands here too.
    assert "Put /opt/venv/bin first on your PATH" in lines[1]
    # The permanent fix is offered, never implied: no `--expose-cli` line unless a caller
    # passes the command, because setup does not write outside itself unasked.
    assert len(lines) == 2

    offered = cli_reachability_notice("/opt/venv/bin/ficelle", expose_command="/opt/venv/bin/ficelle-setup")

    assert "`/opt/venv/bin/ficelle-setup --expose-cli`" in offered[2]


def test_expose_cli_scripts_links_both_commands_and_never_overwrites(tmp_path, monkeypatch):
    """The one install step that writes outside the install. It has to be careful twice:
    it must not claim a name someone else owns, and it must not report work it did not do."""
    monkeypatch.setenv("PATH", "/usr/bin")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    for name in ("python", "ficelle", "ficelle-setup"):
        (venv_bin / name).touch()
    destination = tmp_path / "local-bin"

    lines = expose_cli_scripts(interpreter=interpreter, destination_dir=destination)

    assert (destination / "ficelle").resolve() == (venv_bin / "ficelle").resolve()
    assert (destination / "ficelle-setup").is_symlink()
    assert any("linked" in line for line in lines)
    assert any(f"Put {destination} on your PATH" in line for line in lines)

    # Idempotent: the links are left alone, but the still-required PATH step is not lost.
    assert expose_cli_scripts(interpreter=interpreter, destination_dir=destination) == [
        f"Put {destination} on your PATH to use them."
    ]
    monkeypatch.setenv("PATH", str(destination))
    assert expose_cli_scripts(interpreter=interpreter, destination_dir=destination) == []
    monkeypatch.setenv("PATH", "/usr/bin")

    # A command someone else owns is kept, whoever they are — including another Ficelle.
    (destination / "ficelle").unlink()
    (destination / "ficelle").write_text("a different install's script")
    kept = expose_cli_scripts(interpreter=interpreter, destination_dir=destination)

    assert (destination / "ficelle").read_text() == "a different install's script"
    assert any("keeping the existing command" in line for line in kept)

    # A broken symlink is still an owned entry: do not silently reclaim its name.
    broken_target = tmp_path / "removed-install" / "ficelle-setup"
    (destination / "ficelle-setup").unlink()
    (destination / "ficelle-setup").symlink_to(broken_target)
    kept = expose_cli_scripts(interpreter=interpreter, destination_dir=destination)

    assert (destination / "ficelle-setup").is_symlink()
    assert (destination / "ficelle-setup").readlink() == broken_target
    assert any("keeping the existing command" in line for line in kept)


def test_expose_cli_scripts_reports_a_console_script_that_is_not_there(tmp_path):
    """A source-checkout run that reaches this before the package install has nothing to link;
    a symlink to a missing file would look installed and resolve to nothing."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").touch()
    destination = tmp_path / "local-bin"

    lines = expose_cli_scripts(interpreter=venv_bin / "python", destination_dir=destination)

    assert not destination.exists()
    assert all(line.startswith("WARN: no `") for line in lines)


def test_expose_cli_scripts_says_what_a_dry_run_would_do(tmp_path):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").touch()
    destination = tmp_path / "local-bin"

    lines = expose_cli_scripts(interpreter=venv_bin / "python", destination_dir=destination, dry_run=True)

    assert not destination.exists()
    assert len(lines) == 2 and all(line.startswith("DRY RUN: link ") for line in lines)


def test_run_install_only_exposes_the_cli_when_asked(monkeypatch, tmp_path, capsys):
    calls: list[dict] = []
    monkeypatch.setattr(
        "ficelle.install.expose_cli_scripts",
        lambda **kwargs: (calls.append(kwargs), ["linked /home/u/.local/bin/ficelle -> /opt/venv/bin/ficelle"])[1],
    )
    monkeypatch.setattr(
        "ficelle.install.installed_cli_command",
        lambda name="ficelle", **_kwargs: f"/opt/venv/bin/{name}",
    )

    install_with_doctor_auth(monkeypatch, tmp_path, None, pin_cli=False)
    assert calls == []
    assert "linked /home/u/.local/bin/ficelle" not in capsys.readouterr().out

    install_with_doctor_auth(monkeypatch, tmp_path, None, pin_cli=False, expose_cli=True)
    output = capsys.readouterr().out

    assert calls == [{"interpreter": Path("/usr/bin/python3"), "dry_run": False}]
    assert "linked /home/u/.local/bin/ficelle" in output
    # Having just written the link, offering it again would read as the step having failed.
    assert "--expose-cli" not in output


def test_run_install_resolves_closing_commands_from_the_selected_python(monkeypatch, tmp_path):
    calls: list[tuple[str, Path | None]] = []

    def fake_installed_cli_command(name="ficelle", *, interpreter=None, **_kwargs):
        calls.append((name, interpreter))
        return name

    monkeypatch.setattr("ficelle.install.installed_cli_command", fake_installed_cli_command)

    install_with_doctor_auth(monkeypatch, tmp_path, None, pin_cli=False)

    assert calls == [
        ("ficelle", Path("/usr/bin/python3")),
        ("ficelle-setup", Path("/usr/bin/python3")),
    ]


def test_run_install_spells_out_the_commands_a_fresh_shell_cannot_run(monkeypatch, tmp_path, capsys):
    """The whole point of (a): every command setup prints is one the reader is about to type."""
    monkeypatch.setattr(
        "ficelle.install.installed_cli_command",
        lambda name="ficelle", **_kwargs: f"/opt/venv/bin/{name}",
    )
    install_with_doctor_auth(
        monkeypatch,
        tmp_path,
        {"openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY"}},
        pin_cli=False,
    )
    output = capsys.readouterr().out

    assert "`ficelle` is not reachable by name from this shell" in output
    assert "/opt/venv/bin/ficelle set-key openrouter" in output
    assert "`/opt/venv/bin/ficelle doctor --text` reports which providers are actually configured." in output
    # The bare form is what a fresh shell answers with `command not found`.
    assert "  ficelle set-key openrouter" not in output


def test_run_install_leaves_a_configured_install_alone(monkeypatch, tmp_path, capsys):
    install_with_doctor_auth(
        monkeypatch,
        tmp_path,
        {
            "openrouter": {"invokable": True, "reason": "configured", "key_source": "env"},
            "nous": {"invokable": False, "reason": "missing NOUS_API_KEY"},
        },
    )
    output = capsys.readouterr().out

    assert "Verify: `ficelle health` and `ficelle models`." in output
    assert "ficelle set-key" not in output


def test_run_install_says_nothing_about_keys_when_it_could_not_read_them(monkeypatch, tmp_path, capsys):
    """`--skip-smoke` leaves no credential report; a guess here would be a false alarm."""
    install_with_doctor_auth(monkeypatch, tmp_path, None, skip_smoke=True)
    output = capsys.readouterr().out

    assert "Verify: `ficelle health` and `ficelle models`." in output
    assert "ficelle set-key" not in output


def test_doctor_auth_report_reads_the_block_and_refuses_to_guess(tmp_path):
    payload = json.dumps({"status": "ok", "auth": {"nous": {"invokable": True}}})

    assert doctor_auth_report(payload) == {"nous": {"invokable": True}}
    # Setup echoes other lines around the smoke checks; the JSON still has to be found.
    assert doctor_auth_report(f"legacy runtime detected\n{payload}\n") == {"nous": {"invokable": True}}
    assert doctor_auth_report("") is None
    assert doctor_auth_report("not json at all") is None
    assert doctor_auth_report('{"status": "ok"}') is None
    assert doctor_auth_report('{"auth": "not a mapping"}') is None


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
    assert (
        'model:\n  provider: "custom"\n  base_url: "http://127.0.0.1:8646/v1"\n'
        '  key_env: "FICELLE_API_KEY"\n  model: "ficelle/auto-orchestrator"'
    ) in config_text
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


@pytest.fixture(params=[
    pytest.param(f"{MANAGED_CONFIG_END}\nuser: keep\n{MANAGED_CONFIG_BEGIN}\n", id="end-before-begin"),
    pytest.param(f"{MANAGED_CONFIG_BEGIN}\nuser: keep\n", id="missing-end"),
    pytest.param(f"{MANAGED_CONFIG_END}\nuser: keep\n", id="missing-begin"),
    pytest.param(f"{MANAGED_CONFIG_END}\n{MANAGED_CONFIG_BEGIN}\n{MANAGED_CONFIG_END}\n", id="extra-end"),
    pytest.param(f"{MANAGED_CONFIG_BEGIN}\n{MANAGED_CONFIG_BEGIN}\n{MANAGED_CONFIG_END}\n", id="extra-begin"),
    pytest.param(f"note: '{MANAGED_CONFIG_BEGIN}'\nuser: keep\n{MANAGED_CONFIG_END}\n", id="inline-marker"),
])
def malformed_hermes_config(request):
    return f"prefix: keep\n{request.param}\nsuffix: keep\n"


def test_configure_hermes_preserves_malformed_managed_config(tmp_path, capsys, malformed_hermes_config):
    options = make_options(tmp_path, dry_run=False)
    options.hermes_home.mkdir()
    config = options.hermes_home / "config.yaml"
    config.write_text(malformed_hermes_config)
    before = config.read_bytes()

    configure_hermes(options)

    assert config.read_bytes() == before
    assert not list(options.hermes_home.glob("config.yaml.backup-*"))
    assert "malformed" in capsys.readouterr().out.lower()


def test_malformed_managed_config_is_never_treated_as_owned(malformed_hermes_config):
    assert config_outside_managed_block(malformed_hermes_config) == malformed_hermes_config
    assert config_has_user_additions(malformed_hermes_config)
    with pytest.raises(ValueError, match="[Mm]alformed"):
        replace_managed_block(malformed_hermes_config, "replacement")


@pytest.mark.parametrize("failure", ["nul", "unknown-user", "resolve-loop", "home-unavailable"])
@pytest.mark.parametrize("caller", ["validate", "refresh", "remove"])
def test_invalid_client_home_errors_are_contained(tmp_path, monkeypatch, capsys, failure, caller):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config = options.hermes_home / "config.yaml"
    before = config.read_bytes()
    value = {
        "nul": str(options.hermes_home) + "\0invalid",
        "unknown-user": "~nonexistent-user-xyz/hermes",
    }.get(failure, str(options.hermes_home))
    connectors = load_connectors(options.ficelle_home)
    connectors["hermes"]["client_home"] = value
    save_connectors(options.ficelle_home, connectors)
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == options.hermes_home:
            raise RuntimeError("private symlink loop details")
        return original_resolve(path, *args, **kwargs)

    def home():
        raise RuntimeError("private home lookup details")

    with monkeypatch.context() as patch:
        if failure == "resolve-loop":
            patch.setattr(Path, "resolve", resolve)
        elif failure == "home-unavailable":
            patch.setattr(Path, "home", staticmethod(home))
        if caller == "validate":
            with pytest.raises(ClientHomeError, match="could not be validated") as caught:
                validated_client_home(value)
            assert caught.value.__suppress_context__
        elif caller == "refresh":
            refresh_installed_hermes_integration(Path(value), tmp_path, options.ficelle_home)
            assert "could not be validated" in capsys.readouterr().err
        else:
            removed, messages = remove_hermes_connector(options.ficelle_home)
            assert not removed
            assert len(messages) == 1 and "could not be validated" in messages[0]
    assert config.read_bytes() == before
    assert load_connectors(options.ficelle_home) == connectors


@pytest.mark.parametrize("config_created", [False, True])
def test_remove_hermes_refuses_malformed_config_before_any_deletion(
    tmp_path, malformed_hermes_config, config_created,
):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config = options.hermes_home / "config.yaml"
    config.write_text(malformed_hermes_config)
    connectors = load_connectors(options.ficelle_home)
    # A matching digest must not authorize deletion of a malformed block.
    connectors["hermes"]["metadata"]["config_sha256"] = file_sha256(config)
    connectors["hermes"]["metadata"]["config_created"] = config_created
    save_connectors(options.ficelle_home, connectors)
    before = {path: path.read_bytes() for path in options.hermes_home.rglob("*") if path.is_file()}

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert not removed
    assert messages == ["Malformed Ficelle managed config block; config left untouched and nothing was removed."]
    assert {path: path.read_bytes() for path in options.hermes_home.rglob("*") if path.is_file()} == before
    assert load_connectors(options.ficelle_home) == connectors


def test_remove_hermes_preserves_exact_text_outside_valid_block(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config = options.hermes_home / "config.yaml"
    original = "user: keep\n" + config.read_text() + "\n  # keep indentation\n\n"
    config.write_text(original)
    connectors = load_connectors(options.ficelle_home)
    connectors["hermes"]["metadata"]["config_sha256"] = file_sha256(config)
    save_connectors(options.ficelle_home, connectors)
    start = original.index(MANAGED_CONFIG_BEGIN)
    end = original.index(MANAGED_CONFIG_END) + len(MANAGED_CONFIG_END)
    expected = original[:start] + original[end:]

    removed, _messages = remove_hermes_connector(options.ficelle_home)

    assert removed
    assert config.read_text() == expected
    replacement = f"{MANAGED_CONFIG_BEGIN}\nnew: true\n{MANAGED_CONFIG_END}"
    assert replace_managed_block(original, replacement) == original[:start] + replacement + original[end:]


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


@pytest.mark.parametrize(
    "user_model",
    [
        "model:\n  provider: openrouter\n  model: openai/gpt-oss-120b\n",
        "model: openai/gpt-5\n",
        "model: {provider: openrouter, model: openai/gpt-5}\n",
    ],
)
def test_configure_hermes_preserves_user_owned_main_model_without_duplicate(
    tmp_path,
    user_model,
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config = hermes_home / "config.yaml"
    config.write_text(
        f"{MANAGED_CONFIG_BEGIN}\nold: true\n# END FICELLE MANAGED CONFIG\n"
        f"{user_model}"
    )
    options = make_options(tmp_path, dry_run=False, configure_hermes=True)

    configure_hermes(options)

    updated = config.read_text()
    assert sum(line.startswith("model:") for line in updated.splitlines()) == 1
    assert user_model in updated
    assert 'model: "ficelle/auto-orchestrator"' not in updated


def test_configure_hermes_uses_selected_runtime_endpoint(tmp_path):
    runtime_dir = tmp_path / "legacy-runtime"
    runtime_dir.mkdir()
    (runtime_dir / "config.json").write_text(
        json.dumps({"host": "0.0.0.0", "port": 9864}),
        encoding="utf-8",
    )
    options = make_options(tmp_path, dry_run=False, configure_hermes=True)

    configure_hermes(options, runtime_dir=runtime_dir)

    config_text = (tmp_path / ".hermes" / "config.yaml").read_text()
    assert 'api: "http://127.0.0.1:9864/v1"' in config_text
    assert 'base_url: "http://127.0.0.1:9864/v1"' in config_text


def test_configure_hermes_brackets_selected_ipv6_runtime_endpoint(tmp_path):
    runtime_dir = tmp_path / "ipv6-runtime"
    runtime_dir.mkdir()
    (runtime_dir / "config.json").write_text(
        json.dumps({"host": "::1", "port": 9864}),
        encoding="utf-8",
    )
    options = make_options(tmp_path, dry_run=False, configure_hermes=True)

    configure_hermes(options, runtime_dir=runtime_dir)

    config_text = (tmp_path / ".hermes" / "config.yaml").read_text()
    assert 'api: "http://[::1]:9864/v1"' in config_text
    assert 'base_url: "http://[::1]:9864/v1"' in config_text


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
    assert '    - "ficelle"' in updated
    assert '    - "ficelle-compression"' in updated
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


def test_parser_never_auto_selects_a_host_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_PYTHON", str(tmp_path / "hermes-python"))

    options = options_from_args(build_parser().parse_args([]))

    assert options.connectors == ()


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


def test_run_install_ignores_legacy_service_and_state(
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
        "ficelle.install.shutil.copytree",
        tracked_copytree,
    )

    assert run_install(options) == 0

    assert events == []
    assert (legacy / "state.json").read_text(encoding="utf-8") == '{"legacy": true}\n'


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


from ficelle.router import parse_env_file


def _test_plugin_sources(tmp_path, monkeypatch):
    provider_source = tmp_path / "provider-source"
    compression_source = tmp_path / "compression-source"
    for source in (provider_source, compression_source):
        source.mkdir()
        (source / "__init__.py").write_text(source.name, encoding="utf-8")
        (source / "plugin.yaml").write_text(
            f"name: {source.name}\nversion: 2.0.0\n",
            encoding="utf-8",
        )
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: provider_source)
    monkeypatch.setattr(
        "ficelle.install.packaged_compression_plugin_dir",
        lambda: compression_source,
    )
    return provider_source, compression_source


def _install_plugins_into(tmp_path, monkeypatch, **overrides):
    """Run install_plugins() against throwaway plugin sources, return the options."""
    _test_plugin_sources(tmp_path, monkeypatch)
    overrides.setdefault("dry_run", False)
    options = make_options(tmp_path, target="hermes", **overrides)
    install_plugins(options)
    return options


def test_install_plugins_preserves_catalog_owned_assets_and_records_ownership(
    tmp_path,
    monkeypatch,
):
    provider_source, compression_source = _test_plugin_sources(tmp_path, monkeypatch)
    options = make_options(tmp_path, target="hermes", dry_run=False)
    flat_provider = options.hermes_home / "plugins" / "ficelle"
    flat_compression = options.hermes_home / "plugins" / "ficelle-compression"
    for source, destination in (
        (provider_source, flat_provider),
        (compression_source, flat_compression),
    ):
        shutil.copytree(source, destination)
        manifest = destination / "plugin.yaml"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace("2.0.0", "1.0.0"),
            encoding="utf-8",
        )
    (flat_provider / ".hermes-catalog.json").write_text(
        '{"catalog_name":"ficelle","sha":"abc123"}\n',
        encoding="utf-8",
    )
    (options.hermes_home / "plugins" / ".install-metadata.json").write_text(
        '{"ficelle-compression":{"pinned":true,"revision":"abc123"}}\n',
        encoding="utf-8",
    )

    plugin_ownership = install_plugins(options)
    configure_hermes(options)
    metadata = hermes_connector_metadata(
        options,
        config_existed=False,
        env_file_existed=False,
        env_key_existed=False,
        plugin_ownership=plugin_ownership,
    )
    register_connector(
        options.ficelle_home,
        "hermes",
        client_home=options.hermes_home,
        metadata=metadata,
    )

    assert plugin_ownership == {
        "plugins/ficelle": "hermes",
        "plugins/model-providers/ficelle": "ficelle",
        "plugins/ficelle-compression": "hermes",
    }
    assert set(metadata["plugin_sha256"]) == {"plugins/model-providers/ficelle"}
    assert (flat_provider / ".hermes-catalog.json").exists()
    assert "version: 1.0.0" in (flat_compression / "plugin.yaml").read_text(
        encoding="utf-8"
    )

    removed, _messages = remove_hermes_connector(options.ficelle_home)

    assert removed is True
    assert flat_provider.exists()
    assert flat_compression.exists()
    assert not (options.hermes_home / "plugins" / "model-providers" / "ficelle").exists()


@pytest.mark.parametrize(
    ("external_relative", "source_index"),
    [
        (Path("plugins/ficelle"), 0),
        (Path("plugins/ficelle-compression"), 1),
    ],
)
def test_install_plugins_rejects_incompatible_external_payload_before_mutation(
    tmp_path,
    monkeypatch,
    external_relative,
    source_index,
):
    sources = _test_plugin_sources(tmp_path, monkeypatch)
    options = make_options(tmp_path, target="hermes", dry_run=False)
    external_plugin = options.hermes_home / external_relative
    shutil.copytree(sources[source_index], external_plugin)
    (external_plugin / "__init__.py").write_text("stale runtime", encoding="utf-8")

    with pytest.raises(HermesPluginConflictError, match="No Hermes files were changed"):
        install_plugins(options)

    assert (external_plugin / "__init__.py").read_text(encoding="utf-8") == "stale runtime"
    assert not (options.hermes_home / "plugins" / "model-providers" / "ficelle").exists()
    assert not (options.hermes_home / ".env").exists()


def test_external_payload_comparison_ignores_exactly_the_pin_and_the_readme(tmp_path, monkeypatch):
    provider_source, _ = _test_plugin_sources(tmp_path, monkeypatch)
    (provider_source / "setup_flow.py").write_text("flow", encoding="utf-8")
    options = make_options(tmp_path, target="hermes", dry_run=False)
    external = options.hermes_home / "plugins" / "ficelle"
    shutil.copytree(provider_source, external)

    # A catalog copy carries the release pin and its own README; neither is runtime payload.
    (external / "bootstrap-pin.json").write_text("{}", encoding="utf-8")
    (external / "README.md").write_text("catalog readme", encoding="utf-8")
    install_plugins(options)

    # Anything else differing, at any depth, is still a conflict.
    for relative in ("setup_flow.py", "notes.md", "sub/README.md", "sub/bootstrap-pin.json"):
        stale = options.hermes_home / "plugins" / "ficelle-stale"
        shutil.rmtree(stale, ignore_errors=True)
        shutil.copytree(external, stale)
        target = stale / relative
        target.parent.mkdir(exist_ok=True)
        target.write_text("changed", encoding="utf-8")
        assert _normalized_plugin_payload(stale) != _normalized_plugin_payload(provider_source), relative
    assert _normalized_plugin_payload(external) == _normalized_plugin_payload(provider_source)


def test_install_plugins_seeds_the_provider_plugin_env_key(tmp_path, monkeypatch):
    """Installing the plugin without its env var leaves Hermes unable to use it.

    Hermes skips an api_key provider declaring no env var and builds no client
    when the declared one does not resolve, so a seeded value is what makes the
    exported `provider: ficelle` config work at all.
    """
    options = _install_plugins_into(tmp_path, monkeypatch)

    env_values = parse_env_file(options.hermes_home / ".env")
    assert env_values[PROVIDER_PLUGIN_ENV_KEY] == (options.ficelle_home / "api-token").read_text(
        encoding="utf-8"
    )


def test_install_plugins_replaces_a_stale_provider_plugin_env_key(tmp_path, monkeypatch):
    """A reinstall synchronizes a stale key that the hardened router would reject."""
    hermes_home = make_options(tmp_path, target="hermes").hermes_home
    hermes_home.mkdir(parents=True, exist_ok=True)
    env_path = hermes_home / ".env"
    env_path.write_text(f"{PROVIDER_PLUGIN_ENV_KEY}=chosen-by-user\n", encoding="utf-8")

    options = _install_plugins_into(tmp_path, monkeypatch)
    assert options.hermes_home == hermes_home

    assert parse_env_file(env_path)[PROVIDER_PLUGIN_ENV_KEY] == (options.ficelle_home / "api-token").read_text(
        encoding="utf-8"
    )


def test_install_plugins_syncs_real_api_token_for_exposed_listener(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "config.json").write_text('{"host":"0.0.0.0"}', encoding="utf-8")
    (runtime_dir / "api-token").write_text("owner-api-token\n", encoding="utf-8")
    hermes_home = make_options(tmp_path, target="hermes").hermes_home
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / ".env").write_text(
        f"{PROVIDER_PLUGIN_ENV_KEY}={PROVIDER_PLUGIN_ENV_PLACEHOLDER}\n",
        encoding="utf-8",
    )

    options = _install_plugins_into(tmp_path, monkeypatch, ficelle_home=runtime_dir)

    assert parse_env_file(options.hermes_home / ".env")[PROVIDER_PLUGIN_ENV_KEY] == "owner-api-token"


def test_install_plugins_provisions_private_api_token_before_exposed_service_start(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "config.json").write_text('{"host":"::"}', encoding="utf-8")

    options = _install_plugins_into(tmp_path, monkeypatch, ficelle_home=runtime_dir)

    token_path = runtime_dir / "api-token"
    token = token_path.read_text(encoding="utf-8")
    assert len(token) == 32
    assert set(token) <= set("0123456789abcdef")
    assert token_path.stat().st_mode & 0o077 == 0
    assert parse_env_file(options.hermes_home / ".env")[PROVIDER_PLUGIN_ENV_KEY] == token


def test_install_plugins_reads_selected_config_but_uses_canonical_api_token(tmp_path, monkeypatch):
    canonical_home = tmp_path / "canonical"
    legacy_runtime = tmp_path / "legacy"
    canonical_home.mkdir()
    legacy_runtime.mkdir()
    (legacy_runtime / "config.json").write_text('{"host":"0.0.0.0"}', encoding="utf-8")
    (canonical_home / "api-token").write_text("canonical-token", encoding="utf-8")

    options = _install_plugins_into(
        tmp_path,
        monkeypatch,
        ficelle_home=canonical_home,
    )
    seed_provider_plugin_env_key(options, runtime_dir=legacy_runtime)

    assert parse_env_file(options.hermes_home / ".env")[PROVIDER_PLUGIN_ENV_KEY] == "canonical-token"
    assert not (legacy_runtime / "api-token").exists()


def test_refresh_installed_hermes_integration_updates_managed_main_route(tmp_path, monkeypatch):
    from ficelle.install import refresh_installed_hermes_integration

    provider_source = tmp_path / "provider-source"
    compression_source = tmp_path / "compression-source"
    for source in (provider_source, compression_source):
        source.mkdir()
        (source / "__init__.py").write_text(source.name, encoding="utf-8")
        (source / "plugin.yaml").write_text("version: 0.3.4\n", encoding="utf-8")
    monkeypatch.setattr("ficelle.install.packaged_plugin_dir", lambda: provider_source)
    monkeypatch.setattr("ficelle.install.packaged_compression_plugin_dir", lambda: compression_source)
    hermes_home = tmp_path / "hermes"
    runtime_dir = tmp_path / "legacy-runtime"
    ficelle_home = tmp_path / "canonical"
    hermes_home.mkdir()
    runtime_dir.mkdir()
    ficelle_home.mkdir()
    (runtime_dir / "config.json").write_text('{"host":"0.0.0.0"}', encoding="utf-8")
    (ficelle_home / "api-token").write_text("canonical-token", encoding="utf-8")
    config = hermes_home / "config.yaml"
    config.write_text(
        f"{MANAGED_CONFIG_BEGIN}\nmodel:\n  provider: \"custom\"\n"
        "  base_url: \"http://127.0.0.1:8646/v1\"\n"
        "  model: \"ficelle/auto-orchestrator\"\n"
        "# END FICELLE MANAGED CONFIG\n",
        encoding="utf-8",
    )

    refresh_installed_hermes_integration(hermes_home, runtime_dir, ficelle_home)

    assert 'key_env: "FICELLE_API_KEY"' in config.read_text(encoding="utf-8")
    assert parse_env_file(hermes_home / ".env")[PROVIDER_PLUGIN_ENV_KEY] == "canonical-token"
    assert list(hermes_home.glob("config.yaml.backup-*"))
    assert load_connectors(ficelle_home)["hermes"]["metadata"]["config_managed"] is True


def test_refresh_installed_hermes_integration_does_not_block_core_on_catalog_conflict(
    tmp_path,
    monkeypatch,
    capsys,
):
    from ficelle.install import refresh_installed_hermes_integration

    provider_source, _compression_source = _test_plugin_sources(tmp_path, monkeypatch)
    hermes_home = tmp_path / "hermes"
    flat_provider = hermes_home / "plugins" / "ficelle"
    shutil.copytree(provider_source, flat_provider)
    (flat_provider / "__init__.py").write_text("stale runtime", encoding="utf-8")

    refresh_installed_hermes_integration(
        hermes_home,
        tmp_path / "runtime",
        tmp_path / "ficelle-home",
    )

    assert "Hermes connector refresh skipped" in capsys.readouterr().err
    assert (flat_provider / "__init__.py").read_text(encoding="utf-8") == "stale runtime"
    assert not (hermes_home / "plugins" / "model-providers" / "ficelle").exists()
    assert not (hermes_home / ".env").exists()


def test_refresh_keeps_a_user_modified_config_out_of_connector_removal(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config_path = options.hermes_home / "config.yaml"
    original_hash = load_connectors(options.ficelle_home)["hermes"]["metadata"]["config_sha256"]
    config_path.write_text(config_path.read_text(encoding="utf-8") + "user_setting: true\n", encoding="utf-8")

    refresh_installed_hermes_integration(
        options.hermes_home,
        options.ficelle_home,
        options.ficelle_home,
    )

    metadata = load_connectors(options.ficelle_home)["hermes"]["metadata"]
    assert metadata["config_sha256"] == original_hash
    assert metadata["config_managed"] is True
    assert metadata["config_created"] is True
    assert "user_setting: true" in config_path.read_text(encoding="utf-8")
    removed, messages = remove_hermes_connector(options.ficelle_home)
    assert removed is False
    assert str(config_path) in messages
    assert "user_setting: true" in config_path.read_text(encoding="utf-8")
    assert MANAGED_CONFIG_BEGIN in config_path.read_text(encoding="utf-8")


def test_refresh_preserves_malformed_config_and_recorded_ownership(tmp_path, malformed_hermes_config):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config = options.hermes_home / "config.yaml"
    config.write_text(malformed_hermes_config, encoding="utf-8")
    connectors = load_connectors(options.ficelle_home)
    # Legacy metadata may already contain the malformed file's digest. It still cannot
    # authorize a rewrite, and refresh must retain ownership for safe removal checks.
    connectors["hermes"]["metadata"]["config_sha256"] = file_sha256(config)
    save_connectors(options.ficelle_home, connectors)
    previous = connectors["hermes"]["metadata"]
    snippet = options.hermes_home / "ficelle" / "hermes-config.snippet.yaml"
    snippet.write_text("user snippet: keep\n", encoding="utf-8")
    before = config.read_bytes()

    refresh_installed_hermes_integration(options.hermes_home, options.ficelle_home, options.ficelle_home)

    assert config.read_bytes() == before
    assert not list(options.hermes_home.glob("config.yaml.backup-*"))
    assert snippet.read_text(encoding="utf-8") == "user snippet: keep\n"
    metadata = load_connectors(options.ficelle_home)["hermes"]["metadata"]
    for key in ("config_managed", "config_created", "config_sha256"):
        assert metadata[key] == previous[key]
    removed, messages = remove_hermes_connector(options.ficelle_home)
    assert not removed
    assert any("malformed" in message.lower() for message in messages)
    assert config.read_bytes() == before


@pytest.mark.parametrize("config_created", [False, True])
def test_refresh_keeps_ownership_when_user_replaces_managed_config(tmp_path, config_created):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    connectors = load_connectors(options.ficelle_home)
    connectors["hermes"]["metadata"]["config_created"] = config_created
    save_connectors(options.ficelle_home, connectors)
    previous = connectors["hermes"]["metadata"]
    config = options.hermes_home / "config.yaml"
    config.write_text("model: user-owned\n", encoding="utf-8")
    before = config.read_bytes()

    refresh_installed_hermes_integration(options.hermes_home, options.ficelle_home, options.ficelle_home)

    assert config.read_bytes() == before
    metadata = load_connectors(options.ficelle_home)["hermes"]["metadata"]
    for key in ("config_managed", "config_created", "config_sha256"):
        assert metadata[key] == previous[key]
    removed, messages = remove_hermes_connector(options.ficelle_home)
    assert removed
    assert messages[-1] == "config.yaml left untouched because the Ficelle managed config block was absent."
    assert all(not destination.exists() for _, destination in hermes_plugin_install_specs(options.hermes_home))
    assert config.read_bytes() == before


def test_refresh_of_an_unmodified_config_still_allows_removal(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)

    refresh_installed_hermes_integration(
        options.hermes_home,
        options.ficelle_home,
        options.ficelle_home,
    )

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert removed is True
    assert messages == ["Hermes connector artifacts removed."]
    assert not (options.hermes_home / "config.yaml").exists()


def test_remove_keeps_user_config_when_the_recorded_hash_includes_it(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config_path = options.hermes_home / "config.yaml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "user_setting: true\n", encoding="utf-8")
    connectors = load_connectors(options.ficelle_home)
    connectors["hermes"]["metadata"]["config_sha256"] = file_sha256(config_path)
    save_connectors(options.ficelle_home, connectors)

    removed, _messages = remove_hermes_connector(options.ficelle_home)

    text = config_path.read_text(encoding="utf-8")
    assert removed is True
    assert "user_setting: true" in text
    assert MANAGED_CONFIG_BEGIN not in text


def test_remove_refuses_empty_root_home_and_unscoped_client_homes(tmp_path, monkeypatch):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config_path = options.hermes_home / "config.yaml"
    original = config_path.read_text(encoding="utf-8")

    cases = {
        "   ": "no client home",
        "relative/hermes": "absolute path",
        "/": "filesystem root",
        "/usr": "expected scope",
        "/etc/ssh": "expected scope",
        "/private/etc/ssh": "expected scope",
        # macOS links /tmp to /private/tmp: test the real directory for the scope check
        # (it is /tmp itself on Linux) and the link for the symlink refusal.
        os.path.realpath("/tmp"): "expected scope",
    }
    if Path("/tmp").is_symlink():
        cases["/tmp"] = "symbolic link"
    for client_home, expected in cases.items():
        connectors = load_connectors(options.ficelle_home)
        connectors["hermes"]["client_home"] = client_home
        save_connectors(options.ficelle_home, connectors)
        removed, messages = remove_hermes_connector(options.ficelle_home)
        assert removed is False, messages
        assert any(expected in message for message in messages), messages
        assert config_path.read_text(encoding="utf-8") == original

    home = tmp_path / "home"
    home.mkdir()
    sentinel = home / "config.yaml"
    sentinel.write_text("keep-home\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    connectors = load_connectors(options.ficelle_home)
    connectors["hermes"]["client_home"] = str(home)
    save_connectors(options.ficelle_home, connectors)
    removed, messages = remove_hermes_connector(options.ficelle_home)
    assert removed is False
    assert any("user home directory" in message for message in messages)
    assert sentinel.read_text(encoding="utf-8") == "keep-home\n"
    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("path", [
    "/etc", "/private/etc", "/etc/ssh", "/private/etc/ssh",
    "/usr/local/share", "/usr/bin/ficelle", "/bin/x", "/sbin/x",
    "/System/Library/ficelle", "/System/Volumes/Data/etc/ssh",
    "/Library/ficelle", "/Applications/Ficelle.app",
    "/root/hermes", "/var/root/hermes", "/private/var/root/hermes",
    "/var/log/ficelle", "/private/var/log/ficelle",
    *[f"{tree}{suffix}" for tree in (
        "/etc", "/private/etc", "/usr", "/Library", "/System", "/Applications",
        "/opt/homebrew", "/boot", "/dev", "/proc", "/sys", "/run",
        "/var/lib", "/var/db", "/private/var/lib", "/private/var/db",
        "/var/run", "/private/var/run", "/var/spool", "/private/var/spool",
        "/var/cache", "/private/var/cache", "/snap", "/nix",
        "/var/mail", "/private/var/mail", "/var/opt", "/private/var/opt",
        "/var/empty", "/private/var/empty",
        "/var/backups", "/private/var/backups", "/var/www", "/private/var/www",
        "/var/local", "/private/var/local", "/var/crash", "/private/var/crash",
        "/var/vm", "/private/var/vm",
    ) for tree in (tree, tree.upper()) for suffix in ("", "/x", "/x/y")],
    "/PRIVATE/ETC/ssh", "/LIBRARY/x", "/USR/LOCAL/bin", "/PRIVATE/VAR",
    "/tmp", "/private/tmp", "/var", "/private/var", "/Users", "/home", "/opt",
])
def test_validated_client_home_refuses_system_subtrees_and_unscoped_roots(path):
    # Inaccessible system paths must still be refused on non-root Linux runners.
    with pytest.raises(ClientHomeError):
        validated_client_home(path)


@pytest.mark.parametrize("path", [
    "/Users/test-user/hermes", "/home/test-user/hermes",
    "/private/var/folders/zz/test-user/T/hermes", "/var/folders/zz/test-user/T/hermes",
    "/tmp/ficelle-test/hermes", "/private/tmp/ficelle-test/hermes",
    "/opt/ficelle-test/hermes", "/opt/hermes", "/usr-local/hermes",
    "/opt/homebrew-other/hermes", "/var/library/hermes",
    "/srv/hermes", "/mnt/data/hermes", "/media/disk/hermes",
])
def test_validated_client_home_allows_scoped_user_temp_and_application_homes(path):
    assert validated_client_home(path) == Path(path).resolve()


def test_validated_client_home_resolves_parent_links_before_system_scope_check(tmp_path):
    # The leaf is not a symlink: only resolving its parent reveals the protected tree.
    alias = tmp_path / "etc-alias"
    alias.symlink_to(Path("/private/etc") if Path("/private/etc").is_dir() else Path("/etc"))
    with pytest.raises(ClientHomeError, match="expected scope"):
        validated_client_home(alias / "ssh")


def test_validated_client_home_allows_real_temporary_home(tmp_path):
    assert validated_client_home(tmp_path / "hermes") == (tmp_path / "hermes").resolve()


def test_remove_refuses_a_symlinked_client_home_and_plugin_path(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    real = tmp_path / "real-hermes"
    real.mkdir()
    (real / "config.yaml").write_text("keep-real\n", encoding="utf-8")
    link = tmp_path / "link-hermes"
    link.symlink_to(real)
    connectors = load_connectors(options.ficelle_home)
    connectors["hermes"]["client_home"] = str(link)
    save_connectors(options.ficelle_home, connectors)

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert removed is False
    assert any("symbolic link" in message for message in messages)
    assert (real / "config.yaml").read_text(encoding="utf-8") == "keep-real\n"

    connectors["hermes"]["client_home"] = str(options.hermes_home)
    save_connectors(options.ficelle_home, connectors)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep-outside\n", encoding="utf-8")
    plugin = options.hermes_home / "plugins" / "model-providers" / "ficelle"
    shutil.rmtree(plugin)
    plugin.symlink_to(outside)

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert removed is False
    assert any("symbolic link" in message for message in messages)
    assert sentinel.read_text(encoding="utf-8") == "keep-outside\n"
    assert outside.is_dir()


def test_refresh_skips_an_unsafe_client_home(tmp_path, capsys):
    refresh_installed_hermes_integration(Path("/"), tmp_path, tmp_path)

    assert "filesystem root" in capsys.readouterr().err
    assert not (tmp_path / "plugins").exists()


def test_install_plugins_dry_run_writes_no_provider_plugin_env_key(tmp_path, monkeypatch):
    options = _install_plugins_into(tmp_path, monkeypatch, dry_run=True)

    assert not (options.hermes_home / ".env").exists()


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
    register_connector(
        options.ficelle_home,
        "hermes",
        client_home=options.hermes_home,
        metadata={
            "plugin_ownership": {
                str(destination.relative_to(options.hermes_home)): "ficelle"
                for destination in destinations
            }
        },
    )

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


@pytest.mark.parametrize("operation", ["is_symlink", "resolve", "expanduser", "home"])
@pytest.mark.parametrize("caller", ["validate", "refresh", "remove"])
def test_client_home_permission_errors_are_reported_without_traceback(
    tmp_path, monkeypatch, capsys, operation, caller,
):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config = options.hermes_home / "config.yaml"
    before = config.read_bytes()
    original = getattr(Path, operation)

    def denied(path=None, *args, **kwargs):
        if operation == "home" or path == options.hermes_home:
            raise PermissionError("private filesystem details")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, operation, staticmethod(denied) if operation == "home" else denied)
        if caller == "validate":
            with pytest.raises(ClientHomeError, match="filesystem access failed") as caught:
                validated_client_home(options.hermes_home)
            assert caught.value.__suppress_context__
        elif caller == "refresh":
            refresh_installed_hermes_integration(options.hermes_home, tmp_path, options.ficelle_home)
            assert "filesystem access failed" in capsys.readouterr().err
        else:
            removed, messages = remove_hermes_connector(options.ficelle_home)
            assert not removed
            assert messages == ["Hermes client home could not be validated; filesystem access failed."]
    assert config.read_bytes() == before
    assert "hermes" in load_connectors(options.ficelle_home)


@pytest.mark.parametrize("config_created", [False, True])
@pytest.mark.parametrize("text", ["", "model:\n  provider: user-choice\n# keep exactly\n"])
def test_remove_hermes_preserves_config_when_managed_block_is_absent(
    tmp_path, monkeypatch, config_created, text,
):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config = options.hermes_home / "config.yaml"
    config.write_text(text)
    before = config.stat()
    connectors = load_connectors(options.ficelle_home)
    connectors["hermes"]["metadata"]["config_created"] = config_created
    save_connectors(options.ficelle_home, connectors)
    original_sha256 = file_sha256

    def guarded_sha256(path):
        assert path != config, "Absent-block config must not be hash checked"
        return original_sha256(path)

    monkeypatch.setattr("ficelle.install.file_sha256", guarded_sha256)
    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert removed
    assert messages == [
        "Hermes connector artifacts removed.",
        "config.yaml left untouched because the Ficelle managed config block was absent.",
    ]
    assert config.read_bytes() == text.encode()
    assert (config.stat().st_ino, config.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert not (options.hermes_home / "ficelle/hermes-config.snippet.yaml").exists()
    assert not (options.hermes_home / ".env").exists()
    assert all(not destination.exists() for _, destination in hermes_plugin_install_specs(options.hermes_home))
    assert load_connectors(options.ficelle_home) == {}


def test_remove_hermes_absent_block_still_refuses_modified_owned_artifacts(tmp_path):
    options = make_options(tmp_path, dry_run=False)
    _register_test_hermes_connector(options)
    config = options.hermes_home / "config.yaml"
    config.write_text("user: keep\n")
    snippet = options.hermes_home / "ficelle/hermes-config.snippet.yaml"
    snippet.write_text("user: edited\n")
    before = {path: path.read_bytes() for path in options.hermes_home.rglob("*") if path.is_file()}

    removed, messages = remove_hermes_connector(options.ficelle_home)

    assert not removed
    assert messages == ["Connector artifacts were modified; nothing was removed:", str(snippet)]
    assert {path: path.read_bytes() for path in options.hermes_home.rglob("*") if path.is_file()} == before
    assert "hermes" in load_connectors(options.ficelle_home)


@pytest.mark.parametrize("shadow", ["ficelle.py", "ficelle/__init__.py"])
@pytest.mark.parametrize("command_args", [
    ["set-key", "openrouter", "--stdin"], ["demo"], ["install"],
    ["doctor", "--json"], ["health"], ["models"],
])
@pytest.mark.parametrize("existing_home", [True, False])
def test_install_children_ignore_caller_modules_and_preserve_stdin(
    tmp_path, monkeypatch, shadow, command_args, existing_home,
):
    from ficelle import install

    caller = tmp_path / "caller"
    planted = caller / shadow
    planted.parent.mkdir(parents=True)
    planted.write_text("raise RuntimeError('caller shadow')\n")
    home = tmp_path / "state"
    if existing_home:
        home.mkdir(mode=0o755)
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    monkeypatch.chdir(caller)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    captured = []
    def run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        assert cwd.is_absolute() and cwd != caller and cwd.is_dir()
        if not existing_home:
            assert cwd.stat().st_mode & 0o777 == 0o700
        captured.append((command, kwargs))
        return install.subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(install.subprocess, "run", run)
    secret_input = "test-provider-key\n" if "--stdin" in command_args else None
    result = install.run_command(
        ["../venv/bin/python", "-m", "ficelle.cli", *command_args],
        dry_run=False, env={"FICELLE_HOME": "../state"}, input_text=secret_input,
    )
    assert result.returncode == 0
    (command, kwargs), = captured
    assert command == [str(interpreter), "-P", "-m", "ficelle.cli", *command_args]
    assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
    assert kwargs["env"]["FICELLE_HOME"] == str(home)
    assert kwargs["input"] == secret_input
    assert not any("test-provider-key" in arg for arg in command)
    if existing_home:
        assert kwargs["cwd"] == str(home)
    else:
        assert not Path(kwargs["cwd"]).exists()
        assert not home.exists()


def test_install_rejects_symlink_home_before_normalizing_or_spawning(tmp_path, monkeypatch):
    from ficelle import install

    real_home = tmp_path / "real-home"
    real_home.mkdir()
    home = tmp_path / "link"
    home.symlink_to(real_home, target_is_directory=True)
    args = install.build_parser().parse_args(["--ficelle-home", str(home), "--skip-package"])
    monkeypatch.setattr(install, "run_preflight", lambda *a: pytest.fail("unsafe home"))
    with pytest.raises(OSError, match="symbolic link"):
        install.run_install(install.options_from_args(args))


def test_package_commands_make_local_paths_absolute_before_changing_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wheel = tmp_path / "ficelle.whl"
    wheel.touch()
    options = make_options(tmp_path, package="ficelle.whl", python="venv/bin/python", editable=False)
    for command in (package_install_command(options), uv_package_install_command(options)):
        assert command[-1] == str(wheel)
        assert str(tmp_path / "venv/bin/python") in command


@pytest.mark.parametrize("shadow", ["ficelle.py", "ficelle/__init__.py"])
def test_target_python_probe_uses_safe_path_and_empty_private_cwd(tmp_path, monkeypatch, shadow):
    from ficelle import install

    planted = tmp_path / shadow
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("raise RuntimeError('caller shadow')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    calls = []
    def run(command, **kwargs):
        assert Path(kwargs["cwd"]).stat().st_mode & 0o777 == 0o700
        calls.append((command, kwargs))
        return install.subprocess.CompletedProcess(command, 0, "3.11\n", "")
    monkeypatch.setattr(install.subprocess, "run", run)
    assert probe_target_python("venv/bin/python").returncode == 0
    (command, kwargs), = calls
    assert command[:3] == [str(tmp_path / "venv/bin/python"), "-P", "-c"]
    assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
    assert kwargs["cwd"] != str(tmp_path)


@pytest.mark.parametrize("shadow", ["ficelle.py", "ficelle/__init__.py"])
def test_real_python_import_probe_ignores_caller_ficelle(tmp_path, monkeypatch, shadow):
    from ficelle import install

    planted = tmp_path / shadow
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("raise RuntimeError('caller shadow executed')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    result = install.run_python_probe(sys.executable, "import ficelle; print(ficelle.__file__)")
    assert result.returncode == 0, result.stderr
    assert str(tmp_path) not in result.stdout


@pytest.mark.parametrize("kind", ["symlink", "file", "group-writable", "other-writable"])
def test_install_child_refuses_unsafe_home_without_spawning(tmp_path, monkeypatch, kind):
    from ficelle import install

    home = tmp_path / "home"
    home.mkdir()
    if kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(home, target_is_directory=True)
        home = link
    elif kind == "file":
        home = tmp_path / "file"
        home.touch()
    else:
        home.chmod(0o775 if kind == "group-writable" else 0o757)
    monkeypatch.setattr(install.subprocess, "run", lambda *a, **kw: pytest.fail("unsafe home"))
    with pytest.raises(OSError, match="Ficelle home must"):
        install.run_command(
            [sys.executable, "-m", "ficelle.cli", "set-key", "openrouter", "--stdin"],
            dry_run=False, env={"FICELLE_HOME": str(home)}, input_text="test-key\n",
        )



def test_install_child_uses_private_cwd_when_valid_home_is_not_searchable(tmp_path, monkeypatch):
    from ficelle import install

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda path, mode, **kw: (
        False if Path(path) == home else real_access(path, mode, **kw)
    ))
    calls = []
    def run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        assert cwd.is_dir() and cwd != home and cwd.stat().st_mode & 0o777 == 0o700
        calls.append(kwargs)
        return install.subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(install.subprocess, "run", run)
    install.run_command(
        [sys.executable, "-m", "ficelle.cli", "health"],
        dry_run=False, env={"FICELLE_HOME": str(home)},
    )
    assert calls[0]["env"]["FICELLE_HOME"] == str(home)
    assert not Path(calls[0]["cwd"]).exists()
