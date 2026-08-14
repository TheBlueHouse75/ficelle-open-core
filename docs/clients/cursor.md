# Cursor on Ficelle

Use Ficelle as Cursor's OpenAI-compatible endpoint for chat.

Config format from Cursor's settings as of 2026-08; not yet validated against a
live install — and this client carries a real caveat, read it before filing a
router bug.

## Configure

In **Cursor Settings → Models**:

1. Add the model id `ficelle/auto-tools` as a custom model name.
2. Enable the **OpenAI API key** override and paste any non-empty key
   (`ficelle-local`).
3. Enable **Override OpenAI Base URL** and set `http://127.0.0.1:8646/v1`.
4. Select the `ficelle/auto-tools` model in chat.

## The caveat

Parts of Cursor run from Cursor's servers, not your machine — including, in some
versions, the "verify" step on a custom base URL and some agent features. A URL
on `127.0.0.1` is unreachable from there: verification can fail or server-side
features can silently not use your endpoint even though local chat does. That is
a Cursor architecture property, not a Ficelle failure. If verification refuses
the loopback URL on your version, the only workaround is exposing the router on
an address Cursor's servers can reach (a tunnel) — which trades away the
local-only posture and is **not** recommended by default; Ficelle's admin
surfaces are designed for loopback.

The one-request check in [the clients index](README.md) tells you definitively
whether the router side works.
