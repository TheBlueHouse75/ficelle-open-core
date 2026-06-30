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

Marks for the closed pack's relay providers ship with the Pro pack, under
`ficelle_pro/assets/logos/`, not here.

The logos are the respective companies' trademarks, used nominatively here only to
identify each upstream provider in the local control plane. To add a provider logo,
drop an `.svg` in this directory and map its source slug to that file name in the
`PROVIDER_LOGO_FILES` table in `app.js`. Several sources may map to one file.
Unknown providers fall back to a letter avatar.
