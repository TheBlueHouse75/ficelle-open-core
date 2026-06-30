# Bundled dashboard fonts

These fonts are served locally by the `/admin/fonts/<file>.woff2` route so the
Ficelle admin dashboard renders with a consistent, distinctive type system and
never reaches an external network (CDN-free, matching the local-only posture).

All fonts are licensed under the **SIL Open Font License 1.1 (OFL)**. The license
text is included alongside the binaries.

| File | Family | Role | Source | License |
|---|---|---|---|---|
| `geist-variable.woff2` | Geist (variable, wght) | UI / body | The Geist Project (Vercel) | `OFL-Geist.txt` |
| `geist-mono-variable.woff2` | Geist Mono (variable, wght) | technical data, ids | The Geist Project (Vercel) | `OFL-Geist.txt` |
| `bricolage-grotesque-variable.woff2` | Bricolage Grotesque (variable, wght) | display / headings | Atelier Triay | `OFL-BricolageGrotesque.txt` |

`OFL-Geist.txt` covers both Geist and Geist Mono (same project).

Files are the `latin` subset only, fetched from the Fontsource distribution of
each upstream project. Replace them by dropping in a same-named `.woff2` (the
route allowlists `.woff2` files in this directory).
