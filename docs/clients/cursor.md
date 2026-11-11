# Cursor and Ficelle

**Not supported today.** Cursor cannot use a Ficelle router running on your machine.
This is a property of how Cursor is built, not a Ficelle bug.

Checked on 11/11/2026 against
[Cursor's API keys documentation](https://cursor.com/docs/settings/api-keys).

## Why it does not work

- **Every request goes through Cursor's servers.** Cursor's documentation states that
  all requests are routed through its servers for final prompt building, and that a
  custom key is sent to its backend with each request. The call to the model provider
  therefore starts from Cursor's infrastructure, not from your laptop.
- **Those servers cannot reach `127.0.0.1`.** Ficelle listens on loopback by default.
  From Cursor's backend, `http://127.0.0.1:8646/v1` is its own machine, not yours, so no
  base-URL override can point it at your router.
- **Custom keys only cover chat.** Even for providers Cursor supports, a custom key only
  applies to chat models; Tab completion keeps using Cursor's built-in models.
- The same page now documents keys for a fixed list of providers (OpenAI, Anthropic,
  Google, Azure OpenAI, AWS Bedrock) and no longer documents a custom base-URL override.

## Why there is no tunnel recipe

The only way around this would be exposing the router on a public address that
Cursor's servers can reach (a tunnel or a public bind). That trades away Ficelle's
local-only posture, sends your prompts through Cursor's backend to your exposed router,
and puts the admin surfaces, designed for loopback, within reach of the internet.
Ficelle does not document or support that setup.

## What works instead

- Editor and agent clients that call the endpoint from your machine, such as
  [Continue](continue.md) or [Claude Code](claude-code.md).
- [Open WebUI](open-webui.md) or [your own scripts](openai-sdk.md) for chat.

The one-request check in [the clients index](README.md#verify-any-client-in-one-request)
tells you whether the router itself answers.
