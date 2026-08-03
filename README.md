# Ficelle

<!--
  Site links point at the Netlify deployment while the ficelle.ai domain is being
  registered. Once it resolves, swap the three occurrences of
  https://ficelle-website.netlify.app back to https://ficelle.ai (badge, benchmark
  link, pricing link) and restore the "ficelle.ai" badge label. The release_url in
  the update-manifest example further down is an illustration, not a live link, and
  already uses the final domain.
-->

[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-f26a1b)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Website](https://img.shields.io/badge/website-live-1c1410)](https://ficelle-website.netlify.app)

**A local, OpenAI-compatible router that keeps your AI agents running, and never lets
them overspend.** Point any OpenAI-compatible client at `http://127.0.0.1:8646/v1`
(nothing to import, no account) and your agent fails over across providers and can never
run up a surprise bill.

Ficelle sells reliability, not "free AI": it routes to free LLM capacity, fails over when
a provider rate-limits or breaks, and enforces a **strict-zero** wall so it never makes a
paid call.

## Why Ficelle

- **Reliability:** multi-provider auto-fallback, per-reason cooldowns, and quota-recovery
  probes, so one provider's outage doesn't stop your agent.
- **Strict-zero billing safety:** paid fallback is hardcoded off; a model that tries to
  bill is auto-quarantined. No surprise invoices.
- **Local control plane:** the router, configuration, and stored API keys stay on your
  machine. Ficelle is not a hosted proxy; prompts are sent directly to the upstream
  provider you choose.
- **OpenAI-compatible:** drop-in `/v1/models` and `/v1/chat/completions`, with stable
  virtual models (`ficelle/auto-tools`, `ficelle/auto-json`, `ficelle/auto-reasoning`,
  `ficelle/auto-long`, …).

## What to route to free models

Free models are not a drop-in replacement for a frontier model on every task, and this
project does not pretend otherwise. Across a 76-task benchmark against `gpt-5`, free
routing was *quality-safe* (tied or better) on:

| Workload | Quality-safe | Virtual model |
| --- | --- | --- |
| Classification, labelling, triage | 100% | `ficelle/auto-fast` |
| Structured extraction to JSON | 95% | `ficelle/auto-json` |
| Reasoning and code | 94% | `ficelle/auto-reasoning` |
| Polished long-form writing | 50% | keep your paid model |

Route the first three through Ficelle and keep the budget for the writing. Method and
raw numbers: [the benchmark write-up](https://ficelle-website.netlify.app/blog/free-vs-paid-llm-benchmark/).

## Quick start

Install the versioned open Core from its GitHub Release:

```bash
curl -fsSL https://raw.githubusercontent.com/TheBlueHouse75/ficelle-open-core/v0.1.6/scripts/bootstrap-ficelle.py | python3 -
~/.local/bin/ficelle models
```

The installer uses an isolated runtime and auto-detects Hermes; Ficelle remains fully
standalone when Hermes is absent. To install Pro after purchase without putting the
key in shell history, enter it silently before running the same command:

```bash
(
  read -s FICELLE_LICENSE_KEY
  export FICELLE_LICENSE_KEY
  curl -fsSL https://raw.githubusercontent.com/TheBlueHouse75/ficelle-open-core/v0.1.6/scripts/bootstrap-ficelle.py | python3 -
)
```

Point your client at `http://127.0.0.1:8646/v1`, configure a provider key (e.g.
`OPENROUTER_API_KEY`), and check the router:

```bash
ficelle doctor --json
ficelle health
curl -s http://127.0.0.1:8646/admin/status.json | python3 -m json.tool
```

The local endpoint works with the OpenAI client without a hosted Ficelle account:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8646/v1",
    api_key="ficelle-local",
)
```

`ficelle-setup --target auto` selects Hermes only when it detects a reliable local
Hermes signal; otherwise it selects the same standalone generic target. The explicit
launch targets are `generic` and `hermes`.

## Updates

Ficelle checks for a newer verified Core release in the background after startup. The
local Admin Control Center displays the release notes and offers a one-click install;
the equivalent CLI commands are:

```bash
ficelle update --check
ficelle update --install
```

The updater downloads the release wheel, verifies its SHA-256, keeps a backup of the
installed package, runs import/service smoke checks, and restarts the managed user
service. A failed update restores the previous package. It never stores a Pro license
key. A paid release can advertise a compatible authenticated Pro artifact in Ficelle's
compact release manifest. For production, `authorization: "entitlement"` lets the
license service authorize the already-cached signed entitlement token; Core never sends
the user's license key or persists a new update secret. `authorization: "bearer"` is
available for managed deployments through the short-lived `FICELLE_UPDATE_PRO_TOKEN`.

The default check source is the latest GitHub Release. A deployment can point the Core at
its own HTTPS manifest with `FICELLE_UPDATE_MANIFEST_URL`. The compact manifest shape is:

```json
{
  "version": "0.1.6",
  "release_url": "https://ficelle.ai/releases/0.1.6",
  "core": {
    "wheel_url": "https://downloads.example/ficelle_router-0.1.6-py3-none-any.whl",
    "sha256": "<64 hexadecimal characters>"
  },
  "pro": {
    "wheel_url": "https://downloads.example/ficelle_pro-0.1.6-py3-none-any.whl",
    "sha256": "<64 hexadecimal characters>",
    "authorization": "bearer"
  }
}
```

Update checks are non-blocking and can be disabled for a managed environment with
`FICELLE_DISABLE_UPDATE_CHECK=1`.

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

Runtime state lives under `~/.ficelle/`. Provider secrets resolve from the environment /
Ficelle keychain (`~/.ficelle/ficelle-secrets.keychain-db` on macOS) and are never
written to the repository. On first setup after an upgrade, legacy
`~/.hermes/ficelle/` state is copied only when no Ficelle home was explicitly selected
and `~/.ficelle/` has no runtime data (absent, empty, or credential-only). Existing
credentials and the legacy source are preserved.

## Open core & Ficelle Pro

This repository is the **open core** (Business Source License 1.1): the routing engine,
the strict-zero safety model, the provider-adapter framework, and reference providers
(OpenRouter, Nous). It runs standalone as the free tier.

**Ficelle Pro** is a separate, licensed package that adds the maintained value:

- the full curated provider pool, kept working as providers change their terms;
- compound-model fusion routing;
- native request compression;
- continuous provider-integration updates and support.

The paywall never sits on the core loop: routing, strict-zero, and free-provider access
stay free and open. The Pro pack is not part of this repository; the core runs fully
without it. [Pricing and purchase](https://ficelle-website.netlify.app/#pricing) live on
the website.

## Optional Hermes integration

Hermes is not required. If you want the integration and Hermes is not installed yet, use
its official installer, which launches the setup wizard:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Then `ficelle-setup --target hermes` installs the Ficelle provider and compression
plugins with backups. Hermes config is still opt-in:
`ficelle-setup --target hermes --configure-hermes`. To restore the latest available
plugin/config backups, run `ficelle-setup --target hermes --rollback`; paths without
backups are left untouched.

The provider name is `ficelle`. Start with low-risk auxiliary slots before any main-model
experiment:

```yaml
auxiliary:
  title_generation: { provider: "ficelle", model: "ficelle/auto-fast" }
  compression:      { provider: "ficelle", model: "ficelle/auto-compression" }
  web_extract:      { provider: "ficelle", model: "ficelle/auto-json" }
```

Export the recommended YAML with `ficelle export --target hermes`.

## License

The open core is licensed under the **Business Source License 1.1** (see the `LICENSE` file):
you may use, modify, redistribute, and self-host it; you may not offer it to third parties
as a competing hosted service. Each released version converts to the Apache License 2.0
four years after its publication.

## Contributing and support

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow. Report security
issues privately using [`SECURITY.md`](SECURITY.md). For installation or billing support,
email `support@weesperneonflow.ai`.
