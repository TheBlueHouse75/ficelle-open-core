# OpenAI SDK and custom scripts on Ficelle

Anything that speaks the OpenAI chat-completions API is a Ficelle client with a
two-line change: the base URL and Ficelle's owner API token. Load it without
copying it into source code:

```bash
export FICELLE_API_KEY="$(ficelle access-token api)"
```

## Python

```python
import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8646/v1", api_key=os.environ["FICELLE_API_KEY"])

response = client.chat.completions.create(
    model="ficelle/auto-tools",
    messages=[{"role": "user", "content": "Say hello from a free model."}],
)
print(response.choices[0].message.content)
```

## JavaScript / TypeScript

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8646/v1",
  apiKey: process.env.FICELLE_API_KEY,
});

const response = await client.chat.completions.create({
  model: "ficelle/auto-tools",
  messages: [{ role: "user", content: "Say hello from a free model." }],
});
console.log(response.choices[0].message.content);
```

## curl

```bash
curl -s http://127.0.0.1:8646/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $FICELLE_API_KEY" \
  -d '{"model":"ficelle/auto-fast","messages":[{"role":"user","content":"hello"}]}'
```

## Notes

- Streaming, tool calls, and structured output ride the same API surface as any
  OpenAI-compatible endpoint; pick the profile for the job from
  [the clients index](README.md).
- Token accounting: non-streamed responses are counted on the admin Requests
  page automatically; for streamed calls, pass
  `stream_options: {"include_usage": true}` if you want your tokens counted —
  Ficelle records usage only when the upstream reports it and never alters your
  request to force it.
- Environment-variable form, for tools that read the standard names:
  `export OPENAI_BASE_URL=http://127.0.0.1:8646/v1` and
  `export OPENAI_API_KEY="$(ficelle access-token api)"`.
