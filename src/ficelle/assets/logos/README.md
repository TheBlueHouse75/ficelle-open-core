# Bundled provider logos

These provider marks are served locally by the `/admin/logos/<file>.svg` route so
the admin dashboard shows each provider's logo without any external/network call
(CDN-free, matching the local-only posture). The route sends `Cache-Control: no-cache`
(revalidate each load, like the admin JS) rather than an immutable long cache: filenames
are stable but a mark can be re-traced or replaced in place, so an updated logo must be
able to supersede the browser's copy. (A logo that was already cached under the old
immutable header still needs one hard refresh to flush that stale entry.)

They are rendered **monochrome and theme-adaptive** in the dashboard (CSS mask +
`currentColor`), so the source colour does not matter — only the shape is used.

| File | Provider | Source |
|---|---|---|
| `mistral.svg` | Mistral AI | Simple Icons (`mistralai`), CC0 |
| `openrouter.svg` | OpenRouter | Simple Icons (`openrouter`), CC0 |
| `nous.svg` | Nous Research | `nousresearch.com/safari-pinned-tab.svg` (monochrome mask) |
| `nvidia.svg` | NVIDIA | Simple Icons (`nvidia`), CC0 |
| `groq.svg` | Groq | Simple Icons (`groq`), CC0 |
| `gemini.svg` | Google Gemini | Simple Icons (`googlegemini`), CC0 |
| `cerebras.svg` | Cerebras | Official brand mark (via svgl.app) |
| `siliconflow.svg` | SiliconFlow | LobeHub `@lobehub/icons` (SiliconCloud mark), MIT |
| `opencode.svg` | OpenCode Zen (`opencode_zen`) | Official OpenCode square-"O" mark (the `o` glyph of the `opencode.ai` wordmark / favicon): square ring with the inner block in the lower counter. Re-traced on a square `0 0 24 24` viewBox so it fills the chip like the other marks (the earlier `0 0 24 30` portrait box rendered narrow/misaligned). Monochrome-optimised — the inner block is detached by a negative gap so the frame/block nuance survives the single-colour mask. |
| `ollama.svg` | Ollama Cloud (`ollama`) | Simple Icons (`ollama`), CC0 |
| `cohere.svg` | Cohere (`cohere`) | Official brand mark (via svgl.app); flattened to a single-colour silhouette of the three mark shapes. |
| `kilo.svg` | Kilo Code (`kilo`) | Official Kilo Code mark (via svgl.app); the framed pixel mark flattened to a single colour (frame hole preserved). |
| `zai.svg` | Z.ai (`zai`) | LobeHub `@lobehub/icons-static-svg` 1.95.1 (`zai`), MIT |
| `sambanova.svg` | SambaNova Cloud (`sambanova`) | LobeHub `@lobehub/icons-static-svg` 1.95.1 (`sambanova`), MIT |
| `cloudflare.svg` | Cloudflare Workers AI (`cloudflare`) | Simple Icons (`cloudflare`), CC0 |
| `hetzner.svg` | Hetzner Inference (`hetzner`) | Simple Icons (`hetzner`), CC0 |
| `ovhcloud.svg` | OVHcloud AI Endpoints (`ovhcloud`) | Simple Icons (`ovh`), CC0 |
| `requesty.svg` | Requesty (`requesty`) | Official `requesty.ai` app icon, traced to a single-colour silhouette with potrace (the `>_` prompt kept as a cut-out). |
| `orcarouter.svg` | OrcaRouter (`orcarouter`) | Official `orcarouter.ai` orca logo, traced to a single-colour silhouette with potrace (white markings kept as cut-outs). |
| `routeway.svg` | Routeway (`routeway`) | Official `routeway.ai` favicon: the white arrow mark traced with potrace, without its gradient disc. |

Marks for the closed pack's relay providers ship with the Pro pack, under
`ficelle_pro/assets/logos/`, not here. The pack maps their sources in
`PROVIDER_PACK_LOGOS` (`ficelle_pro/provider_pack.py`); `/admin/state` exposes that map as
`provider_logos` and the dashboard overlays it on `PROVIDER_LOGO_FILES`, so this open-core
tree never names those providers.

Marks taken from LobeHub's MIT-licensed icon set (`siliconflow.svg`, `zai.svg`, `sambanova.svg`)
ship with its copyright and permission notice in `LICENSE-lobehub.txt`, as the MIT licence requires.

The logos are the respective companies' trademarks, used nominatively here only to
identify each upstream provider in the local control plane. To add a provider logo,
drop an `.svg` in this directory and map its source slug to that file name in the
`PROVIDER_LOGO_FILES` table in `app.js`. Several sources may map to one file.
Unknown providers fall back to a letter avatar, but every registered provider must have a mark:
`test_every_registered_provider_has_a_servable_dashboard_logo` in `tests/test_router.py` fails
when one is missing, so adding a provider includes adding its logo here or in the pack.
