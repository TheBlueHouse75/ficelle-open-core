# Ficelle compression retrieval tool for Hermes

Registers one tool, `ficelle_retrieve_compressed`, which fetches the original content behind a
`<<ficelle:compressed:<hash>>>` marker from a local
[Ficelle](https://github.com/TheBlueHouse75/ficelle-open-core) router. Ficelle emits those markers
only when its native live-zone compression is active; that compression is part of the paid Ficelle
Pro edition and is off by default. With Ficelle Core alone, the tool reports that Ficelle Pro is not
installed. The plugin is optional and useful only with Ficelle.

Requires Hermes 0.22.6 or newer and a running Ficelle Core.

## What the plugin does

- **On load:** registers the tool. Nothing is downloaded, installed, started or written.
- **Network:** when the model calls the tool, one request to the local Ficelle router
  (`http://127.0.0.1:8646` by default; `FICELLE_ROUTER_URL` or the `providers.ficelle` address in
  your Hermes configuration override it, read through Hermes' own configuration API). No other
  host, no telemetry.
- **Credentials:** none. The plugin stores and sends no credential.
- **Background processes:** none. It does not install Ficelle; the `ficelle` provider plugin's
  setup step does that, after explicit confirmation. Without a running Ficelle the tool reports
  that the router is unreachable.

## Removal

`hermes plugins remove ficelle-compression` removes it. It leaves Ficelle Core and the `ficelle`
provider untouched.
