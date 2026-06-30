# Ficelle

**A local, OpenAI-compatible router that keeps your AI agents running — and never lets
them overspend.** Point any OpenAI-compatible client at `http://127.0.0.1:8646/v1`
(nothing to import, no account) and your agent fails over across providers and can never
run up a surprise bill.

Ficelle sells reliability, not "free AI": it routes to free LLM capacity, fails over when
a provider rate-limits or breaks, and enforces a **strict-zero** wall so it never makes a
paid call.

## Why Ficelle

- **Reliability** — multi-provider auto-fallback, per-reason cooldowns, and quota-recovery
  probes, so one provider's outage doesn't stop your agent.
- **Strict-zero billing safety** — paid fallback is hardcoded off; a model that tries to
  bill is auto-quarantined. No surprise invoices.
- **Local & private** — traffic and API keys stay on your machine. Ficelle is not a hosted
  proxy.
- **OpenAI-compatible** — drop-in `/v1/models` and `/v1/chat/completions`, with stable
  virtual models (`ficelle/auto-tools`, `ficelle/auto-json`, `ficelle/auto-reasoning`,
  `ficelle/auto-long`, …).

## Quick start

```bash
uv tool install ficelle-router      # or: pip install ficelle-router
ficelle install                     # local service (macOS LaunchAgent / Linux systemd --user)
ficelle models                      # list the virtual models
```

Point your client at `http://127.0.0.1:8646/v1`, configure a provider key (e.g.
`OPENROUTER_API_KEY`), and check the router:

```bash
ficelle doctor --json
ficelle health
curl -s http://127.0.0.1:8646/admin/status.json | python -m json.tool
```

## How it works

```text
OpenAI-compatible client  →  127.0.0.1:8646/v1  →  Ficelle router
                                                     ├─ strict-zero catalog filtering
                                                     ├─ provider credential resolution
                                                     ├─ model scoring + fallback
                                                     ├─ cooldowns / quarantine / failure classification
                                                     └─ admin API + dashboard + logs
                                                            ↓
                                                    free LLM providers
```

Runtime state lives under `~/.hermes/ficelle/`. Provider secrets resolve from the
environment / OS keychain and are never written to the repository.

## Open core & Ficelle Pro

This repository is the **open core** (Business Source License 1.1): the routing engine,
the strict-zero safety model, the provider-adapter framework, and reference providers
(OpenRouter, Nous). It runs standalone as the free tier.

**Ficelle Pro** is a separate, licensed package that adds the maintained value:

- the full curated provider pool, kept working as providers change their terms;
- compound-model fusion routing;
- native request compression;
- continuous provider-integration updates and support.

The paywall never sits on the core loop — routing, strict-zero, and free-provider access
stay free and open. The Pro pack is not part of this repository; the core runs fully
without it.

## Hermes

Ficelle ships a Hermes provider plugin (provider name `ficelle`). Start with low-risk
auxiliary slots before any main-model experiment:

```yaml
auxiliary:
  title_generation: { provider: "ficelle", model: "ficelle/auto-fast" }
  compression:      { provider: "ficelle", model: "ficelle/auto-compression" }
  web_extract:      { provider: "ficelle", model: "ficelle/auto-json" }
```

Export the recommended YAML with `ficelle export`.

## License

The open core is licensed under the **Business Source License 1.1** (see the `LICENSE` file):
you may use, modify, redistribute, and self-host it; you may not offer it to third parties
as a competing hosted service. Each released version converts to the Apache License 2.0
four years after its publication.
