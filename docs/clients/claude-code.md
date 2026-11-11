# Claude Code and Ficelle

Claude Code can use Ficelle directly through the native Anthropic Messages
surface. Ficelle translates the request at the edge and sends it through the
same `ChatCompletionRouter` as `/v1/chat/completions`.

## Setup

Claude Code needs a base URL and a credential even though Ficelle's loopback
listener does not authenticate the value. The URL is the local Ficelle root; the
client appends `/v1/messages` itself:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8646
export ANTHROPIC_AUTH_TOKEN=ficelle-local
```

`ANTHROPIC_AUTH_TOKEN` is the variable Anthropic recommends when you are not told
which one a gateway expects
([connect Claude Code to an LLM gateway](https://code.claude.com/docs/en/llm-gateway-connect.md)):
Claude Code sends it as `Authorization: Bearer`, and it takes precedence over a saved
claude.ai login without the one-time approval prompt that `ANTHROPIC_API_KEY` triggers.
`ANTHROPIC_API_KEY` (sent as `x-api-key`) works too; Ficelle accepts either header.
Run `/status` in Claude Code to confirm the base URL and credential source in use.

For a non-loopback bind, replace that value with `ficelle access-token api`.
Clients that build the request themselves send it as `Authorization: Bearer <token>`
or `x-api-key: <token>`. Requests must carry `Anthropic-Version: 2023-06-01`.

## What Anthropic supports

Anthropic documents gateways for Claude Code but states that it does not support
routing Claude Code to non-Claude models through any gateway
([LLM gateways](https://code.claude.com/docs/en/llm-gateway)). Ficelle does exactly
that: it serves free non-Claude models. Treat this setup as supported by Ficelle, not
by Anthropic; report problems here, not to Anthropic. Features built for Claude models,
such as prompt caching and extended thinking, do not carry over (see below).

## Model mapping and honesty

Claude-family request IDs are not upstream identity claims. Configure the
mapping explicitly in `~/.ficelle/config.json`; this is the recommended tier
convention:

```json
{
  "anthropic_model_mapping": {
    "claude-code/haiku*": "ficelle/auto-fast",
    "claude-code/sonnet*": "ficelle/auto-coding",
    "claude-code*": "ficelle/auto-coding",
    "*haiku*": "ficelle/auto-fast",
    "*sonnet*": "ficelle/auto-coding",
    "*opus*": "ficelle/auto-orchestrator",
    "*fable*": "ficelle/auto-orchestrator"
  }
}
```

Haiku maps to `auto-fast`; Sonnet and the generic Claude Code route map to
`auto-coding`; Opus and Fable map to `auto-orchestrator`. The response's
`model` is the Ficelle profile, and `X-Ficelle-Model` identifies the concrete
model that actually served the request. Ficelle never presents a free upstream
model as Claude.

`cache_control` and thinking blocks are accepted, ignored, and announced in
`X-Ficelle-Dropped-Params`; no prompt caching or Anthropic thinking semantics
are claimed.

## Verification

The R9 smoke uses Claude Code 2.1.259 against a local fake upstream, so it has
no provider cost. It proves a streamed multi-turn session, a real Bash tool
execution, and tool-call ID round-trip:

```bash
PYTHONPATH=src python scripts/smoke-claude-code.py
```

The recorded evidence is dated 08/09/2026. A missing `claude` executable is a
local prerequisite failure; the script does not substitute a mocked client.

## Scope

This surface implements `POST /v1/messages` only. The OpenAI Responses API,
MCP, and `count_tokens` are non-goals; use the OpenAI-compatible client recipes
for clients that speak `/v1/chat/completions`.

Anthropic lists `/v1/messages/count_tokens` as optional for a gateway
([gateway compatibility guide](https://code.claude.com/docs/en/llm-gateway-protocol.md)):
without it, Claude Code falls back to a character-based estimate, so `/context`
shows approximate token counts.
