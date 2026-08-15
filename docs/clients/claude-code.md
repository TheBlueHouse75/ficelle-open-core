# Claude Code and Ficelle — where that stands

Short version: **Claude Code cannot point at Ficelle directly today**, and this
page exists so nobody burns an evening discovering that the hard way.

## Why

Claude Code speaks the Anthropic Messages protocol (`/v1/messages`). Ficelle
exposes the OpenAI chat-completions protocol (`/v1/chat/completions`). Setting
Claude Code's base URL to Ficelle produces protocol errors, not routed requests
— the two APIs differ in request shape, streaming framing, and tool-call
encoding.

## What works today

- **Everything Claude Code shells out to.** Scripts, hooks, and subprocesses
  that Claude Code runs can use Ficelle through the
  [OpenAI SDK recipe](openai-sdk.md) — a common pattern is routing bulk or
  background LLM calls (classification, extraction, summarization in helper
  scripts) through free models while Claude Code itself stays on Anthropic.
- **Protocol translation proxies.** Community projects exist that translate the
  Anthropic Messages API to OpenAI chat completions and can sit between Claude
  Code and Ficelle. None is validated or endorsed here: a translator owns the
  fidelity of tool-call and streaming semantics, which is exactly where agent
  clients break subtly. If you try one, verify tool calls end-to-end before
  trusting a session.

## What is tracked

A native Anthropic-compatible `/v1/messages` endpoint on Ficelle — so Claude
Code, and every other Anthropic-protocol client, points at the router with one
environment variable — now has an owning specification:
[`../prds/anthropic-messages-surface-prd.md`](../prds/anthropic-messages-surface-prd.md)
(15/08/2026). It is specified, **not implemented**. Until it ships, treat any
"Claude Code on Ficelle" claim as aspirational, and this page stays the honest
answer.
