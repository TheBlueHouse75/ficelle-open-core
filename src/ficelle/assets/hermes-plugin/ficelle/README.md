# Ficelle provider for Hermes

Registers the `ficelle` model provider so Hermes can call
[Ficelle](https://github.com/TheBlueHouse75/ficelle-open-core), an independent local
OpenAI-compatible model router, at `http://127.0.0.1:8646/v1`. Ficelle routes your requests to the
upstream providers you configure in Ficelle; it does not provide unlimited free model access.

Requires Hermes 0.22.6 or newer, the first release with the provider setup step.

## What the plugin does

- **On load:** registers one provider profile (`ficelle`) and the virtual `ficelle/auto-*` model
  names. Loading the plugin, listing models and sending requests never download, install, start or
  write anything. Model listing asks `http://127.0.0.1:8646/v1/models` and falls back to a built-in
  list when Ficelle is not running.
- **Network:** only the loopback address above, plus the setup step below when you confirm it.
  No telemetry, no analytics, no other host.

## Setup step (`hermes model` or `hermes setup`, then select Ficelle)

The plugin runs a setup step only when you select Ficelle in `hermes model` or `hermes setup`. It
runs these stages in order and stops at the first failure, naming the stage and one action to
resume. Selecting Ficelle again resumes.

1. **probe:** checks `http://127.0.0.1:8646/health`. A healthy, compatible Ficelle Core is reused as
   it is; nothing is downloaded or installed.
2. **consent:** if Ficelle Core is not running, shows what will happen and asks for confirmation
   (default: no). A session without an interactive terminal stops here and changes nothing.
3. **bootstrap:** downloads Ficelle's official bootstrap script
   (`scripts/bootstrap-ficelle.py` from an immutable commit of `TheBlueHouse75/ficelle-open-core`,
   over HTTPS), checks it
   against the SHA-256 recorded in this plugin's `bootstrap-pin.json`, and refuses to run it on any
   mismatch. This is the only download the plugin performs. The bootstrap then verifies Ficelle
   Core's own wheel against a pinned SHA-256 before installing it. A copy of this plugin installed
   by Ficelle itself carries no pin and never offers this step.
4. **core-health:** after an install, runs `ficelle doctor`, `ficelle health` and
   `ficelle models`.
5. **request-smoke:** sends one real request through `ficelle/auto-orchestrator` to the local
   listener, with the `FICELLE_API_KEY` value Hermes already holds if there is one. It uses one
   request of whichever provider Ficelle routes it to. Nothing is written in Hermes unless it
   succeeds. If it fails and `ficelle doctor` reports no usable provider (**provider-credential**),
   setup says to add a provider key in Ficelle.
6. **hermes-activation:** writes to Hermes (see below) and prints **Ready**.

### What the bootstrap installs

- Ficelle Core into its own Python environment, `~/.local/share/ficelle/venv`, never into the
  Hermes environment, using a Python 3.11+ found outside the Hermes installation that can create
  virtual environments (on Debian or Ubuntu, install `python3-venv` first). The bootstrap runs
  with a minimal environment (path, home, temp, locale, proxy and certificate variables) and no
  Hermes variable or secret.
- Ficelle's state in `~/.ficelle` (or the `FICELLE_HOME` you already use).
- The `ficelle` and `ficelle-setup` commands, linked into `~/.local/bin`, and
  `~/.config/ficelle/active-home`, which records where that state lives.
- A **long-running background service** owned by Ficelle: a LaunchAgent on macOS, a `systemd --user`
  unit on Linux, a per-user Scheduled Task on Windows. It keeps running after Hermes exits and
  serves any client of yours that points at it. Ficelle updates itself through its own updater,
  outside this plugin.

### What the plugin writes in Hermes

Only through Hermes' own helpers, and only these two things:

- `FICELLE_API_KEY`: a non-secret local listener value (`ficelle-local`), set only when it is
  unset; an existing value is never overwritten. If the existing value is a placeholder Hermes
  rejects, setup stops and says so instead of replacing it.
- The default model, `ficelle/auto-orchestrator` on provider `ficelle`, with its endpoint
  `http://127.0.0.1:8646/v1` and the chat-completions protocol.

The plugin never opens `.env`, `config.yaml`, `auth.json` or any Ficelle credential file, never
writes into its own directory, and never runs `ficelle connectors install hermes`. That command
remains available: it adds Ficelle's compression plugin, replaces the listener value with
Ficelle's own listener token, and writes the recommended Hermes configuration (auxiliary models,
fallback), or a snippet to merge when you already have a `config.yaml`. The setup step stores no
progress file; each run checks the machine again.

## Credentials

Provider keys stay in Ficelle (`ficelle set-key <provider>`, or Ficelle's own setup). The plugin
never asks for, reads or stores them. Hermes holds only the local listener value above.

## Removal

`hermes plugins remove ficelle` removes the plugin and Hermes' record of it. Hermes runs no plugin
code at removal, so afterwards:

1. pick another model in `hermes model`;
2. if you ever ran `ficelle connectors install hermes` for this profile, run
   `ficelle connectors remove hermes`, which removes only unchanged Ficelle-owned artifacts.

Removing the plugin never stops or uninstalls Ficelle Core, because other clients may use it. To
uninstall Core and its state, use Ficelle's own commands and read their data warning first.
