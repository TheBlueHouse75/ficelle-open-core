# Continue on Ficelle

Point the Continue extension (VS Code / JetBrains) at Ficelle for chat, edit,
and autocomplete-adjacent flows on strict-zero free models.

Config format from the Continue documentation as of 2026-08; not yet validated
against a live install — please report deltas.

## Configure

Add Ficelle models to `~/.continue/config.yaml`:

```yaml
models:
  - name: Ficelle agent
    provider: openai
    model: ficelle/auto-tools
    apiBase: http://127.0.0.1:8646/v1
    apiKey: ficelle-local
    roles:
      - chat
      - edit
  - name: Ficelle fast
    provider: openai
    model: ficelle/auto-fast
    apiBase: http://127.0.0.1:8646/v1
    apiKey: ficelle-local
```

Older installs using `config.json` take the same fields on a `models` entry
(`"provider": "openai"`, `"apiBase"`, `"apiKey"`).

## Notes

- Keep autocomplete on a local or dedicated fast model if you use it heavily:
  free-tier rate limits make per-keystroke traffic the quickest way to spend a
  quota. `ficelle/auto-fast` fits chat-length quick calls, not keystrokes.
- `ficelle/auto-json` is the profile to reference from prompts/blocks that need
  structured output.
