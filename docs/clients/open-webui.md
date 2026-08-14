# Open WebUI on Ficelle

Give Open WebUI a strict-zero model menu: every Ficelle profile appears in the
model picker, routed and observable locally.

Config format from the Open WebUI documentation as of 2026-08; not yet validated
against a live install — please report deltas.

## Configure

In **Admin Panel → Settings → Connections → OpenAI API**, add a connection:

- API Base URL: `http://127.0.0.1:8646/v1`
- API Key: `ficelle-local` (any non-empty value)

Save, then refresh the model list: the `ficelle/auto-*` profiles appear in the
picker.

## Docker

`127.0.0.1` inside the Open WebUI container is the container, not your machine,
and Ficelle binds to the host's loopback only. Two working setups:

- **macOS / Windows (Docker Desktop):** use `http://host.docker.internal:8646/v1`.
  Docker Desktop's VM forwards that alias to the host's loopback, which reaches
  the router.
- **Linux (native Docker Engine):** run the container with `--network host` and
  keep `http://127.0.0.1:8646/v1` — the container then shares the host's network
  namespace, so loopback is the host's loopback. The often-suggested
  `--add-host=host.docker.internal:host-gateway` does **not** work here: it maps
  the alias to the bridge gateway IP, an address a loopback-only bind never
  accepts. Binding Ficelle to a non-loopback address instead is possible but
  changes its security posture (non-loopback listeners require the access-token
  flow) — prefer `--network host`.

## Notes

- Pick `ficelle/auto-tools` for tool-using chats, `ficelle/auto-fast` for the
  default conversational model, `ficelle/auto-long` for big-context threads.
