/* ===================== Ficelle admin — pure helpers + static config ===================== */

export const profileOrder = [
  "ficelle/auto-orchestrator", "ficelle/auto-tools", "ficelle/auto-json", "ficelle/auto-compression",
  "ficelle/auto-long", "ficelle/auto-fast", "ficelle/auto-reasoning", "ficelle/auto-multimodal",
  "ficelle/auto-vision", "ficelle/auto-video", "ficelle/auto-audio"
];
export const fusionModelId = "ficelle/auto-fusion";
export const profileLabels = {
  "ficelle/auto-orchestrator": ["Main orchestrator", "Your primary agent brain — the model most requests go to."],
  "ficelle/auto-tools": ["Tool fallback", "A backup that takes over when the main model fails a tool call."],
  "ficelle/auto-json": ["Structured JSON", "Extraction and parsing jobs that need clean JSON back."],
  "ficelle/auto-compression": ["Compression", "Short, bounded summaries for chat compaction."],
  "ficelle/auto-long": ["Long context", "Very large prompts where the context window is the limit."],
  "ficelle/auto-fast": ["Fast side tasks", "Titles, small drafts, quick low-risk work."],
  "ficelle/auto-reasoning": ["Reasoning", "Models that expose explicit reasoning controls."],
  "ficelle/auto-multimodal": ["Media input", "Image, video, or audio understanding."],
  "ficelle/auto-vision": ["Vision", "Reading images and screenshots."],
  "ficelle/auto-video": ["Video", "Video understanding experiments."],
  "ficelle/auto-audio": ["Audio", "Audio understanding experiments."]
};
export const profilePresets = {
  "ficelle/auto-orchestrator": "A free, Ficelle-only candidate to replace your primary model. Keep a stronger paid fallback ready until it proves itself in real use.",
  "ficelle/auto-tools": "Catches main-model provider outages without making free models your default. Wire it as a fallback, not the primary.",
  "ficelle/auto-json": "Cheaper structured extraction with JSON support. Good for web extract, parsing, and enrichment jobs.",
  "ficelle/auto-compression": "Free, bounded summarization for chat compaction. Run a quick canary on this machine before relying on it.",
  "ficelle/auto-long": "Routes very large prompts when the context window is the constraint. Not your default summarizer.",
  "ficelle/auto-fast": "Quick, cheap side jobs that never touch the main orchestrator — titles, small drafts, summaries.",
  "ficelle/auto-reasoning": "Tries free models with explicit reasoning knobs. A reasoning knob alone does not make a model the best agent.",
  "ficelle/auto-multimodal": "One virtual model for media-understanding experiments. Only route payloads that match the declared modality.",
  "ficelle/auto-vision": "Free screenshot and image reading, gated behind a real canary image test.",
  "ficelle/auto-video": "Tests free video understanding when a catalog exposes it. Keep it gated behind benchmarks.",
  "ficelle/auto-audio": "Tests free audio understanding when a catalog exposes it. Do not replace speech-to-text blindly."
};

// Capability -> the specialized profile whose benchmark proves it (for provenance badges).
// Keep in sync with GATE_CAPABILITY_BY_PROFILE in router.py (same pairs, inverse direction).
export const CAP_PROFILE = {
  tools: "ficelle/auto-tools",
  structured: "ficelle/auto-json",
  reasoning: "ficelle/auto-reasoning",
  image: "ficelle/auto-vision",
  video: "ficelle/auto-video",
  audio: "ficelle/auto-audio",
};
// How each provenance state renders: badge color class, glyph marker, and tooltip wording.
// Confidence order: verified > catalog (provider-declared) > reference (cross-provider prior) > default (blind guess).
export const CAP_STATE_META = {
  verified:  { cls: "ok",     mark: " ✓", word: "Verified by benchmark" },
  refuted:   { cls: "danger", mark: " ✗", word: "Refuted by benchmark" },
  refuted_reference: { cls: "danger", mark: " ⚠", word: "Cross-provider reference says the base model lacks this; the provider default over-claims" },
  reference: { cls: "info",   mark: " ~",      word: "Assumed from cross-provider reference" },
  catalog:   { cls: "",       mark: "",        word: "Declared by the provider catalog, not benchmarked" },
  default:   { cls: "warn",   mark: " ?",      word: "Assumed from provider defaults, not confirmed" },
};

export const $ = (id) => document.getElementById(id);
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const cloneJson = (v) => JSON.parse(JSON.stringify(v || {}));
// Shared by every "wait for the daemon" loop in app.js, which each used to inline it.
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Provider display labels are overlaid from the backend (/admin/state provider_labels) so
// the closed pack's providers — incl. the grey-market relays, which are NOT named in this
// open-core file — get proper labels when ficelle_pro is installed. The literal map below
// is the fallback for the open providers and for the first render before the state loads.
let _runtimeProviderLabels = {};
export function setProviderLabels(m) { _runtimeProviderLabels = m || {}; }
export function providerLabel(s) { return _runtimeProviderLabels[s] || ({ openrouter: "OpenRouter", nous: "Nous Research", mistral: "Mistral", nvidia: "NVIDIA", groq: "Groq", gemini: "Google Gemini", cerebras: "Cerebras", siliconflow: "SiliconFlow", opencode_zen: "OpenCode Zen", kilo: "Kilo Code", ollama: "Ollama Cloud", cohere: "Cohere" })[s] || s || "unknown"; }
export function formatContext(v) { const n = Number(v || 0); if (n >= 1e6) return (n / 1e6).toFixed(n % 1e6 ? 1 : 0) + "M"; if (n >= 1e3) return Math.round(n / 1e3) + "k"; return String(n); }
export function formatTokenLimit(v) { const n = Number(v || 0); if (!n) return null; if (n >= 1e6) return (n / 1e6).toFixed(n % 1e6 ? 1 : 0) + "M"; if (n >= 1e3) return Math.round(n / 1e3) + "k"; return String(n); }
export function formatSeconds(v) { const n = Number(v || 0); if (!n) return "0s"; if (n >= 3600) return Math.round(n / 3600) + "h"; if (n >= 60) return Math.round(n / 60) + " min"; return Math.round(n) + "s"; }
// Floored at every magnitude: an inflated savings claim is the one direction this
// figure must not err in.
export function formatUsd(v) {
  const n = Math.max(0, Number(v || 0));
  if (!n) return "$0.00";
  if (n >= 100) return "$" + Math.floor(n);
  if (n >= 0.01) return "$" + (Math.floor(n * 100) / 100).toFixed(2);
  return "<$0.01";
}
export function formatDuration(v) {
  const n = Math.max(0, Number(v || 0));
  if (!n) return "0.00s";
  const minutes = Math.floor(n / 60);
  const seconds = n - (minutes * 60);
  if (minutes) return minutes + "m " + seconds.toFixed(2).padStart(5, "0") + "s";
  return seconds.toFixed(2) + "s";
}
export function pct(a, b) { const t = Number(b || 0); if (!t) return 0; return Math.round((Number(a || 0) / t) * 100); }

export function timeAgo(value) {
  if (!value) return { label: "never", title: "never" };
  const d = new Date(value);
  if (isNaN(d.getTime())) return { label: String(value), title: String(value) };
  const diff = (Date.now() - d.getTime()) / 1000;
  const title = d.toLocaleString();
  let label;
  if (diff < -86400) label = "in " + Math.round(Math.abs(diff) / 86400) + "d";
  else if (diff < -3600) label = "in " + Math.round(Math.abs(diff) / 3600) + "h";
  else if (diff < -45) label = "in " + Math.round(Math.abs(diff) / 60) + " min";
  else if (diff < 0) label = "just now";
  else if (diff < 45) label = "just now";
  else if (diff < 3600) label = Math.round(diff / 60) + " min ago";
  else if (diff < 86400) label = Math.round(diff / 3600) + "h ago";
  else if (diff < 7 * 86400) label = Math.round(diff / 86400) + "d ago";
  else label = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return { label, title };
}
export const timeChip = (v) => { const t = timeAgo(v); return '<span title="' + esc(t.title) + '">' + esc(t.label) + "</span>"; };

export const hasIn = (m, x) => (m.input_modalities || []).map(String).includes(x);
export const hasOut = (m, x) => (m.output_modalities || []).map(String).includes(x);
export const hasParam = (m, x) => (m.supported_parameters || []).map(String).includes(x);
export const runtimeKey = (m) => m.source + "::" + m.upstream_id;

export const FREE_ACCESS_LABELS = { catalog_free: "Catalog free", quota_free: "Free quota", local_free: "Local free", paid: "Paid", unknown: "Unknown cost" };
export function freeAccess(m) {
  if (m?.free_access) return m.free_access;
  if (m?.pricing_safety?.status === "strict_zero") return { eligible: true, mode: "catalog_free", proof: "provider_pricing" };
  return { eligible: false, mode: "unknown" };
}
export function freeModeLabel(mode) { return FREE_ACCESS_LABELS[mode] || "Unknown cost"; }
export function providerFreeModeLabel(row) {
  if (row?.free_mode) return freeModeLabel(row.free_mode);
  if (row?.source_type === "zero_price_catalog") return "Catalog free";
  return "Free status unknown";
}

export const REASON_LABELS = {
  empty_assistant_message: "Empty model response",
  truncated_before_content: "Token budget too small",
  bad_upstream_request: "Invalid request body",
  bad_upstream_contract: "Provider/model request contract",
  malformed_tool_call: "Tool call without a name",
  client_disconnected: "Client hung up",
  invalid_success_json: "Invalid provider response",
  invalid_json: "Invalid JSON",
  invalid_schema: "Invalid JSON schema",
  judge_invalid_json: "Invalid judge JSON",
  judge_invalid_schema: "Invalid judge schema",
  rate_limited: "Rate limited",
  rate_limited_upstream: "Model pool saturated",
  request_too_large: "Tokens-per-minute limit",
  server_error: "Provider server error",
  timeout: "Timed out",
  unavailable: "Unavailable",
  auth_or_credit: "Auth or credit issue",
  billing_or_paid: "Billing guard",
  quota_exhausted: "Free quota exhausted",
  no_free_quota: "No free-tier allocation",
  model_not_found: "Model not serveable to us",
  benchmark_failed: "Benchmark failed",
  failed_capability_check: "Capability failed",
  failed_benchmark: "Virtual model benchmark failed",
  upstream_failure: "Upstream failure",
  no_available_model: "No available model",
  mid_stream_failure: "Mid-stream failure",
};
export const REASON_DESCRIPTIONS = {
  empty_assistant_message: "The provider returned success, but no usable assistant text was found.",
  truncated_before_content: "The request's max_tokens ran out before the model emitted any content, typically because a reasoning model spent the budget on reasoning tokens. Request-side limit: the model is not cooled.",
  bad_upstream_request: "The provider rejected the request body itself (HTTP 400/422), typically a malformed tool_call or an unsupported field. Request-side problem: no other candidate is tried and the model is not cooled.",
  bad_upstream_contract: "One model rejected an otherwise valid request option or required a private provider field. Ficelle keeps the request unchanged, tries the next candidate, and cools neither the model nor the provider.",
  malformed_tool_call: "Ficelle refused the request before calling any provider: a tool call replayed in the conversation history carries no function name, which every provider rejects. The error names the offending message index — fix the client that built the history, or start a new conversation. No model was picked, called, or cooled.",
  client_disconnected: "The client closed the connection while the answer was streaming. Client-side abort: the model is not cooled.",
  invalid_success_json: "The provider returned success, but the JSON response could not be parsed.",
  invalid_json: "The judge response could not be parsed as JSON.",
  invalid_schema: "The judge returned JSON, but not in the schema Fusion expects.",
  judge_invalid_json: "The judge response could not be parsed as JSON.",
  judge_invalid_schema: "The judge returned JSON, but not in the schema Fusion expects.",
  rate_limited: "The provider asked Ficelle to slow down.",
  rate_limited_upstream: "The shared pool behind this one model id is saturated upstream, which says nothing about your account. Only this model is cooled (briefly); every other model of the provider keeps routing.",
  request_too_large: "The provider rejected this request as too large for its tokens-per-minute budget (HTTP 413). A transient, model-scoped limit that recharges on its own; the model is cooled briefly and recovers without action.",
  server_error: "The provider returned or reported a server-side error.",
  timeout: "The provider did not answer before the configured timeout.",
  unavailable: "The provider or model was unavailable.",
  auth_or_credit: "The provider rejected the request because of credentials, credit, or account state.",
  billing_or_paid: "Ficelle blocked a model that appeared to require paid access.",
  quota_exhausted: "A free quota appears to be exhausted until a later probe succeeds.",
  no_free_quota: "The provider grants this model zero free-tier allocation (limit: 0); it is quarantined and will not be proposed.",
  model_not_found: "The provider will not serve this model id to this account — a 404/410 (listed in its catalog but not deployed), or a 403 naming it as a tier an admin has to switch on; it is quarantined and will stop being benchmarked until you clear it.",
  benchmark_failed: "The model failed a capability check for this route.",
  failed_capability_check: "The model failed the latest capability check for this virtual model.",
  failed_benchmark: "The model failed the latest benchmark for this virtual model.",
};
export function reasonLabel(reason) { return REASON_LABELS[String(reason || "")] || String(reason || "cooldown"); }
export function reasonDescription(reason) { return REASON_DESCRIPTIONS[String(reason || "")] || reasonLabel(reason); }
export function reasonWithCode(reason) {
  const code = String(reason || "cooldown");
  const label = reasonLabel(code);
  return label === code ? label : label + " (" + code + ")";
}

// `level` is "ok" | "warn" | "error" (empty for neutral). `action` is an optional
// {label, onClick}: a message that asks the user to do something needs a way to do it, and
// needs to outlive the 2.8s a plain acknowledgement gets.
export function showToast(msg, level = "", action = null) {
  const toast = $("toast");
  $("toastMsg").textContent = msg;
  toast.classList.remove("is-ok", "is-warn", "is-error");
  if (level) toast.classList.add("is-" + level);
  const button = $("toastAction");
  button.hidden = !action;
  button.textContent = action?.label || "";
  // `onclick` holds one handler by construction, so the previous toast's action cannot
  // still be attached here.
  button.onclick = action
    ? () => { toast.classList.remove("show"); clearTimeout(showToast.t); action.onClick(); }
    : null;
  toast.classList.add("show");
  clearTimeout(showToast.t);
  const ms = action ? 14000 : (level === "warn" || level === "error" ? 6000 : 2800);
  showToast.t = setTimeout(() => toast.classList.remove("show"), ms);
}

// The `auth.key_source` families a removal cannot reach, mapped to the next move — same shape
// as REASON_LABELS/reasonLabel above, so a family added later falls back to its own name and is
// still stated rather than silently dropped. `env` and `external` are the two nothing inside
// Ficelle can clear, and the two `remaining_legacy_sources` never enumerated.
const KEY_SOURCE_REMEDIES = {
  env: "a process environment variable — unset it where the service starts, then restart",
  external: "an external credential resolver — clear the key in that integration",
};
export function keySourceRemedy(keySource) {
  const family = String(keySource || "");
  return KEY_SOURCE_REMEDIES[family] || family;
}

// Keep the removal verdict pure so it can be tested without a DOM, and so the destructive
// legacy purge is offered exactly once: after a normal removal reveals a source the purge
// can reach, never after an explicit purge and never on a locked keychain it must skip.
//
// `keySource` is `auth.key_source` recomputed after the removal — the store the resolver
// would read a key from *now*, or null when none does. It is what makes the verdict honest
// for every family: `legacySources` enumerates legacy `.env` files and keychains only, so a
// key held in a process environment variable or served by an external resolver (R4) used to
// come back as a green "key removed" with the provider still configured.
export function providerKeyRemovalNotice(label, removed, legacySources, purgeLegacy, keySource = null) {
  const cleared = removed.length
    ? (label + " key removed from " + removed.length + " location" + (removed.length > 1 ? "s" : "") + ".")
    : ("No " + label + " key was stored by Ficelle.");
  const stillConfigured = keySource
    ? (" " + label + " is still configured: a key still resolves from " + keySourceRemedy(keySource) + ".")
    : "";
  // Each case below states its own message, level and button in one place. Only the rule that
  // cuts across them lives here: a key that still resolves makes the toast red, unless a button
  // is still offered that could finish the job.
  const notice = (tail, level, offerLegacyPurge) => ({
    message: cleared + stillConfigured + tail,
    level: (keySource && !offerLegacyPurge) ? "error" : level,
    offerLegacyPurge,
  });
  if (!legacySources.length) return notice("", removed.length ? "ok" : "warn", false);
  if (purgeLegacy) {
    return notice(" Could not clear " + legacySources.join(", ") + " — remove it manually.", "error", false);
  }
  // A locked keychain is reported but cannot be purged — the purge skips it so it never
  // opens the GUI prompt that would hang the daemon. Offering the button there would spend
  // a "other tools may lose access" confirmation on a guaranteed no-op.
  if (!legacySources.some((source) => !source.endsWith(":locked"))) {
    return notice(" " + legacySources.join(", ") + " is locked, so it cannot be read or cleared — unlock it, or remove the key manually.", "warn", false);
  }
  return notice(" Legacy credential sources remain at " + legacySources.join(", ") + ".", "warn", true);
}

// The clipboard rejects for reasons the user can act on (permission denied, document not
// focused), so every caller needs the fallback message — not a raw DOMException in a toast.
export async function copyToClipboard(text, okMessage, failMessage = "Copy failed.") {
  try { await navigator.clipboard.writeText(text); showToast(okMessage); }
  catch { showToast(failMessage); }
}
