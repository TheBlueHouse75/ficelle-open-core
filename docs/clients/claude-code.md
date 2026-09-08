# Claude Code and Ficelle

Claude Code can use Ficelle directly through the native Anthropic Messages
surface. Ficelle translates the request at the edge and sends it through the
same `ChatCompletionRouter` as `/v1/chat/completions`.

## Setup

Claude Code requires both settings even though Ficelle's loopback listener does
not authenticate the value. The URL is the local Ficelle root; the client
appends `/v1/messages` itself:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8646
export ANTHROPIC_API_KEY=ficelle-local
```

For a non-loopback bind, replace that value with `ficelle access-token api`.
The endpoint accepts it as `Authorization: Bearer <token>` or `x-api-key: <token>`
for clients that build the request themselves. Requests must carry
`Anthropic-Version: 2023-06-01`.

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
