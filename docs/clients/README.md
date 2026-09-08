# Point your agent client at Ficelle

Every recipe below ends at the same place: an OpenAI-compatible client talking to

```text
http://127.0.0.1:8646/v1
```

without a Ficelle token on the default loopback listener. A deliberate non-loopback
bind requires the owner API token returned locally by `ficelle access-token api`.
Install and start Ficelle first, then store a
provider key (`ficelle set-key openrouter`): the model
list is served from public catalogs without credentials, so **a populated model
list does not mean a completion can be served** — `ficelle doctor --text` tells
you which providers actually can.

These clients do not host Ficelle. Core always runs from its Ficelle-owned runtime. Most
OpenAI-compatible clients need only this recipe; Hermes and OpenClaw use optional add-on connectors
because they also need host-specific assets/configuration.

Some clients want the base URL and append routes themselves; others want the full
chat endpoint pasted verbatim. Both are valid:

```text
base URL       http://127.0.0.1:8646/v1
chat endpoint  http://127.0.0.1:8646/v1/chat/completions
```

`ficelle client-config` prints both, plus the model list, as JSON.

## Which model id to use

| Model | Use it for |
|---|---|
| `ficelle/auto-coding` | coding assistants; qualified models on locally available provider deployments |
| `ficelle/auto-tools` | the general agent profile: tool calling without coding-quality certification |
| `ficelle/auto-fast` | quick, low-stakes calls: titles, summaries, one-liners |
| `ficelle/auto-json` | structured extraction, JSON output |
| `ficelle/auto-orchestrator` | heavier multi-step reasoning |
| `ficelle/auto-long` | large-context requests |
| `ficelle/auto-compression` | conversation-history compaction |

`ficelle models` lists everything currently routable, including the
capability-specific profiles (reasoning, vision, audio, video).

`auto-coding` may correctly return `503 no_certified_coding_model` even while other profiles
work. That means the bundled pool has no qualification matching this install's live
free provider deployments carrying a qualified model; Ficelle will not silently substitute a
merely tool-capable model.

## Recipes

- [Codex CLI](codex.md)
- [Continue](continue.md)
- [Cursor](cursor.md)
- [Open WebUI](open-webui.md)
- [OpenAI SDK and custom scripts](openai-sdk.md)
- [Claude Code](claude-code.md) — protocol status and what works today
- Hermes uses a packaged optional connector instead of a paste-in recipe. Run Ficelle standalone
  first, then `ficelle connectors install hermes`. `ficelle connectors export hermes` prints the
  recommended YAML.

## Verify any client in one request

```bash
curl -s http://127.0.0.1:8646/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $FICELLE_API_KEY" \
  -d '{"model":"ficelle/auto-fast","messages":[{"role":"user","content":"Reply exactly: ficelle-ok"}],"temperature":0}'
```

If this answers and your client does not, the problem is the client config, not
the router. If this fails with `credentials unavailable`, store a provider key
(`ficelle set-key openrouter`). The admin dashboard at
`http://127.0.0.1:8646/admin` shows every routed request, what it cost ($0.00
under strict-zero), and the estimated savings at the same models' paid rates.
