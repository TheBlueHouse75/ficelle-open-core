# Codex CLI and Ficelle

**Not supported today.** No Codex CLI configuration can route through Ficelle at the
moment. This page explains why, so you do not spend time on a config that cannot work.

Checked on 11/11/2026 against the Codex documentation and repository.

## Why it does not work

- **Codex only speaks the OpenAI Responses API.** Codex announced the deprecation of
  Chat Completions support in
  [discussion #7782](https://github.com/openai/codex/discussions/7782), and the
  `wire_api = "chat"` code path was removed by
  [PR #10157](https://github.com/openai/codex/pull/10157), merged on 03/02/2026. The
  [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
  now lists `responses` as the only supported `wire_api` value.
- **Ficelle does not serve `/v1/responses`.** It serves the OpenAI Chat Completions API
  (`/v1/chat/completions`) and the Anthropic Messages API (`/v1/messages`). A custom
  Codex provider pointed at `http://127.0.0.1:8646/v1` therefore sends requests to a
  route Ficelle does not implement.
- **The old profile recipe is also obsolete.** Since Codex 0.134.0, `--profile` no
  longer reads `[profiles.<name>]` tables from `config.toml`, and the top-level
  `profile = "<name>"` selector is gone; profiles now live in
  `~/.codex/<name>.config.toml`
  ([advanced configuration](https://learn.chatgpt.com/docs/config-file/config-advanced)).
  Earlier versions of this page used both, together with `wire_api = "chat"`.

If you followed that earlier recipe, remove the `[model_providers.ficelle]` and
`[profiles.ficelle]` tables and any `profile = "ficelle"` line from
`~/.codex/config.toml`.

## What works instead

- Agent clients that speak Chat Completions, such as [Continue](continue.md),
  [Open WebUI](open-webui.md) or [your own scripts](openai-sdk.md).
- [Claude Code](claude-code.md), through Ficelle's Anthropic Messages edge.

Serving the Responses API is not part of Ficelle's current scope (see the
[Claude Code page](claude-code.md#scope)). This page will change if that does.

The one-request check in [the clients index](README.md#verify-any-client-in-one-request)
still tells you whether the router itself answers.
