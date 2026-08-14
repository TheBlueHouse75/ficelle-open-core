# Codex CLI on Ficelle

Route Codex CLI's completions through Ficelle: same agent, strict-zero free
models, full request observability on the local dashboard.

Config format from the Codex CLI documentation as of 2026-08; not yet validated
against a live install — please report deltas.

## Configure

Add a provider and a profile to `~/.codex/config.toml`:

```toml
[model_providers.ficelle]
name = "Ficelle"
base_url = "http://127.0.0.1:8646/v1"
env_key = "FICELLE_API_KEY"
# Ficelle serves the OpenAI chat-completions API, not the Responses API.
wire_api = "chat"

[profiles.ficelle]
model_provider = "ficelle"
model = "ficelle/auto-tools"
```

The router ignores the key's value but Codex requires the variable to be set:

```bash
export FICELLE_API_KEY=ficelle-local
codex --profile ficelle
```

To make it the default, set `profile = "ficelle"` at the top of the same file.

## Notes

- `ficelle/auto-tools` is the right default profile for Codex's tool-heavy loop;
  switch a session to `ficelle/auto-orchestrator` for heavier multi-step work.
- Expect free-tier pacing: Ficelle absorbs upstream rate limits by failing over
  between free models, but long agent bursts can still see slower turns than a
  paid endpoint. The Requests page on `http://127.0.0.1:8646/admin` shows every
  fallback as it happens.
