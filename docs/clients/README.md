# Point your agent client at Ficelle

Every recipe below ends at the same place: an OpenAI-compatible client talking to

```text
http://127.0.0.1:8646/v1
```

with any non-empty API key (the conventional placeholder is `ficelle-local` — the
router serves loopback callers and ignores the key's value). Install and start
Ficelle first, and store a provider key (`ficelle set-key openrouter`): the model
list is served from public catalogs without credentials, so **a populated model
list does not mean a completion can be served** — `ficelle doctor --text` tells
you which providers actually can.

Some clients want the base URL and append routes themselves; others want the full
chat endpoint pasted verbatim. Both are valid:

```text
base URL       http://127.0.0.1:8646/v1
chat endpoint  http://127.0.0.1:8646/v1/chat/completions
```

`ficelle export --target generic` prints both, plus the model list, as JSON.

## Which model id to use

| Model | Use it for |
|---|---|
| `ficelle/auto-tools` | the default agent profile: tool calling, coding assistants |
| `ficelle/auto-fast` | quick, low-stakes calls: titles, summaries, one-liners |
| `ficelle/auto-json` | structured extraction, JSON output |
| `ficelle/auto-orchestrator` | heavier multi-step reasoning |
| `ficelle/auto-long` | large-context requests |
| `ficelle/auto-compression` | conversation-history compaction |

`ficelle models` lists everything currently routable, including the
capability-specific profiles (reasoning, vision, audio, video).

## Recipes

- [Codex CLI](codex.md)
- [Continue](continue.md)
- [Cursor](cursor.md)
- [Open WebUI](open-webui.md)
- [OpenAI SDK and custom scripts](openai-sdk.md)
- [Claude Code](claude-code.md) — protocol status and what works today
- Hermes ships as a packaged first-class integration instead of a paste-in
  recipe: `ficelle-setup --target hermes` installs the provider and compression
  plugins, and `ficelle export --target hermes` prints the recommended YAML.

## Verify any client in one request

```bash
curl -s http://127.0.0.1:8646/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ficelle/auto-fast","messages":[{"role":"user","content":"Reply exactly: ficelle-ok"}],"temperature":0}'
```

If this answers and your client does not, the problem is the client config, not
the router. If this fails with `credentials unavailable`, store a provider key
(`ficelle set-key openrouter`). The admin dashboard at
`http://127.0.0.1:8646/admin` shows every routed request, what it cost ($0.00
under strict-zero), and the estimated savings at the same models' paid rates.
