from __future__ import annotations

import io
import json
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
    cli_reachability_notice,
    collect_preflight_checks,
    configure_hermes,
    copy_plugin_tree,
    doctor_auth_report,
    dedicated_keychain_path,
    ensure_dedicated_keychain,
    ensure_hermes_compression_plugin_enabled,
    ensure_hermes_toolset_enabled,
    ensure_hermes_plugin_enabled,
    expose_cli_scripts,
    install_plugins,
    PROVIDER_PLUGIN_ENV_KEY,
    PROVIDER_PLUGIN_ENV_PLACEHOLDER,
    installed_cli_command,
    legacy_service_listener_status,
    managed_service_status,
    migrate_legacy_ficelle_home,
    offer_first_key_capture,
    options_from_args,
    package_is_local_reference,
    package_install_command,
    probe_target_python,
    resolve_install_target,
    rollback_last_hermes_install,
    run_install,
    seed_provider_plugin_env_key,
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
        # The CLI implies this for a piped stdin; the test harness is exactly that, so
        # capture tests opt in to interactivity explicitly.
        "non_interactive": True,
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
    args = build_parser().parse_args(["--target", "generic"])

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


from ficelle.router import parse_env_file


def _install_plugins_into(tmp_path, monkeypatch, **overrides):
    """Run install_plugins() against throwaway plugin sources, return the options."""
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
    overrides.setdefault("dry_run", False)
    options = make_options(tmp_path, target="hermes", **overrides)
    install_plugins(options)
    return options


def test_install_plugins_seeds_the_provider_plugin_env_key(tmp_path, monkeypatch):
    """Installing the plugin without its env var leaves Hermes unable to use it.

    Hermes skips an api_key provider declaring no env var and builds no client
    when the declared one does not resolve, so a seeded value is what makes the
    exported `provider: ficelle` config work at all.
    """
    options = _install_plugins_into(tmp_path, monkeypatch)

    env_values = parse_env_file(options.hermes_home / ".env")
    assert env_values[PROVIDER_PLUGIN_ENV_KEY] == PROVIDER_PLUGIN_ENV_PLACEHOLDER


def test_install_plugins_keeps_an_existing_provider_plugin_env_key(tmp_path, monkeypatch):
    """A value the user already chose must survive a reinstall."""
    hermes_home = make_options(tmp_path, target="hermes").hermes_home
    hermes_home.mkdir(parents=True, exist_ok=True)
    env_path = hermes_home / ".env"
    env_path.write_text(f"{PROVIDER_PLUGIN_ENV_KEY}=chosen-by-user\n", encoding="utf-8")

    options = _install_plugins_into(tmp_path, monkeypatch)
    assert options.hermes_home == hermes_home

    assert parse_env_file(env_path)[PROVIDER_PLUGIN_ENV_KEY] == "chosen-by-user"


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


def test_install_plugins_reads_legacy_config_but_uses_canonical_api_token(tmp_path, monkeypatch):
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
