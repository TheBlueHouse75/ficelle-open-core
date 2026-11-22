/* ===================== Ficelle admin — pure helpers + static config ===================== */

export const profileOrder = [
  "ficelle/auto-orchestrator", "ficelle/auto-tools", "ficelle/auto-json", "ficelle/auto-compression",
  "ficelle/auto-long", "ficelle/auto-fast", "ficelle/auto-reasoning", "ficelle/auto-multimodal",
  "ficelle/auto-vision", "ficelle/auto-video", "ficelle/auto-audio", "ficelle/auto-coding"
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
  "ficelle/auto-audio": ["Audio", "Audio understanding experiments."],
  "ficelle/auto-coding": ["Coding", "Certified coding-assistant models."]
};
export const profilePresets = {
  "ficelle/auto-orchestrator": "The default Ficelle main-model route: strict-zero orchestration through the best eligible provider. Verify a fresh request after setup.",
  "ficelle/auto-tools": "Catches main-model provider outages through Ficelle's tool-capable fallback pool.",
  "ficelle/auto-json": "Cheaper structured extraction with JSON support. Good for web extract, parsing, and enrichment jobs.",
  "ficelle/auto-compression": "Free, bounded summarization for chat compaction. Run a quick canary on this machine before relying on it.",
  "ficelle/auto-long": "Routes very large prompts when the context window is the constraint. Not your default summarizer.",
  "ficelle/auto-fast": "Quick, cheap side jobs that never touch the main orchestrator — titles, small drafts, summaries.",
  "ficelle/auto-reasoning": "Tries free models with explicit reasoning knobs. A reasoning knob alone does not make a model the best agent.",
  "ficelle/auto-multimodal": "One virtual model for media-understanding experiments. Only route payloads that match the declared modality.",
  "ficelle/auto-vision": "Free screenshot and image reading, gated behind a real canary image test.",
  "ficelle/auto-video": "Tests free video understanding when a catalog exposes it. Keep it gated behind benchmarks.",
  "ficelle/auto-audio": "Tests free audio understanding when a catalog exposes it. Do not replace speech-to-text blindly.",
  "ficelle/auto-coding": "Routes only to qualified coding models on locally compatible provider deployments."
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
// What a client puts in its API key field, for the rail's endpoint card. `/v1` follows the bind,
// not the address the dashboard was opened on: a wildcard bind is still reached on 127.0.0.1 but
// requires the token from every client. Anything but an explicit loopback answer names the token,
// so an unknown bind never tells a remote client it can go without one.
export function endpointKeyHint(bindIsLoopback) {
  return bindIsLoopback === true
    ? "API key: not needed on this machine (any value works)"
    : "API key: <code>ficelle access-token api</code>";
}
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
// Sub-second timings read in milliseconds (a 4 ms router overhead would round to "0.00s");
// null stays a dash so an unmeasured value never reads as instant.
export function formatLatency(v) {
  if (v == null || v === "" || !Number.isFinite(Number(v)) || Number(v) < 0) return "—";
  const n = Number(v);
  if (n < 0.001) return "<1 ms";
  if (n < 1) return Math.round(n * 1000) + " ms";
  return n.toFixed(2) + "s";
}
// Request-row facts about the context-window filter. Plain strings (the caller escapes them) and
// only what the row recorded: a row from before these fields existed yields no claim at all.
export function requestContextMeta(e) {
  const meta = [];
  const isNum = (v) => v != null && Number.isFinite(Number(v));
  // `prompt_tokens_estimate` is the upper prompt estimate PLUS the completion allowance: it is the
  // figure a window had to hold, not a prompt size, so it must not read as a token count.
  if (isNum(e.prompt_tokens_estimate)) meta.push("est. " + formatContext(e.prompt_tokens_estimate) + " tokens prompt + completion");
  if (isNum(e.prompt_estimate_tokens) && isNum(e.prompt_tokens)) meta.push("prompt est. " + formatContext(e.prompt_estimate_tokens) + " vs " + formatContext(e.prompt_tokens) + " reported");
  const filtered = Number(e.excluded_for_context || 0) > 0 || Number(e.demoted_for_context || 0) > 0;
  if (filtered && isNum(e.context_required_tokens) && Number(e.context_required_tokens) > 0) {
    meta.push("needs " + formatContext(e.context_required_tokens) + " window" + (isNum(e.context_largest_in_pool) ? ", largest in pool " + formatContext(e.context_largest_in_pool) : ""));
  }
  if (Number(e.excluded_for_context || 0) > 0) meta.push(Number(e.excluded_for_context) + " too small for prompt");
  if (Number(e.demoted_for_context || 0) > 0) meta.push(Number(e.demoted_for_context) + " demoted for tight window");
  return meta;
}
export const CONTEXT_CHANGED_NOTE = "Context filter changed the first-choice model";

// The Requests card for the pre-call prompt estimator. Nothing is claimed without evidence:
// no requests with both figures is an empty state, not a perfect ratio.
export function contextEstimateCard(c) {
  const n = Number(c?.n || 0);
  if (!n) return { v: "n/a", note: "needs requests that reported prompt tokens (1k or more)", cls: "info" };
  const ratio = (v) => (v == null ? "n/a" : Number(v).toFixed(2) + "×");
  return {
    v: ratio(c.ratio_p50),
    note: "estimate / reported prompt tokens &middot; p1 " + ratio(c.ratio_p1) + " &middot; p90 " + ratio(c.ratio_p90) + " &middot; " + n + " requests &middot; " + Math.round(Number(c.underestimate_rate || 0) * 100) + "% under-estimated",
    cls: "info",
  };
}
export function pct(a, b) { const t = Number(b || 0); if (!t) return 0; return Math.round((Number(a || 0) / t) * 100); }

// Remaining-quota gauge. Turns `provider_stats[source].admission` (ProviderAdmissionLedger.snapshot)
// into plain presentation data. A numeric gauge exists only for a fresh, provider-reported
// observation with a known denominator; every other state is a non-numeric statement with its
// reason, so a number is never invented for a limit nobody reported. Every figure is an estimate:
// it describes the provider's last response and cannot see other clients using the same account.
const QUOTA_LOW_FRACTION = 0.1;
export const QUOTA_ESTIMATE_NOTE = "Estimate from the provider's last response; other clients on the same account are not visible.";
function quotaCount(n) { return Number(n).toLocaleString("en-US"); }
// Not `formatSeconds`: an age reads as whole minutes elapsed ("under a minute" below one, never
// hours), floored so an observation is never reported older than it is.
function quotaAge(seconds) {
  const minutes = Math.floor(Math.max(0, Number(seconds) || 0) / 60);
  return minutes < 1 ? "under a minute" : minutes + " min";
}
// The snapshot describes the moment /admin/state was built, and the page can stay open long after.
// Ageing each bucket by the time since that load keeps a passed reset, or a report older than the
// ledger's stale horizon, from still reading as live (the same rule as `is_fresh` in
// ProviderAdmissionLedger._snapshot_row: an exhausted request bucket stays live until its reset).
function agedQuotaMetric(m, metric, elapsed, staleAfter) {
  if (!m || m.source !== "provider_headers" || !(elapsed > 0)) return m;
  const age = (Number(m.age_seconds) || 0) + elapsed;
  const resetIn = m.reset_in_seconds != null ? Number(m.reset_in_seconds) - elapsed : null;
  const gating = metric === "requests" && Number(m.remaining) <= 0;
  const fresh = Boolean(m.fresh) && resetIn != null && resetIn > 0 && (gating || age <= staleAfter);
  return { ...m, age_seconds: Math.floor(age), reset_in_seconds: resetIn != null && resetIn > 0 ? Math.ceil(resetIn) : null, fresh };
}
// Seconds until an aged, still-fresh bucket stops being shown as live, or null when none is.
function quotaMetricExpiry(m, metric, staleAfter) {
  if (!m || !m.fresh || m.reset_in_seconds == null) return null;
  const gating = metric === "requests" && Number(m.remaining) <= 0;
  return gating ? m.reset_in_seconds : Math.min(m.reset_in_seconds, Math.max(0, staleAfter - Number(m.age_seconds || 0)));
}
export function agedQuotaAdmission(admission, elapsedSeconds = 0) {
  const a = admission && typeof admission === "object" ? admission : null;
  if (!a || !(elapsedSeconds > 0) || a.state !== "observed") return a;
  const staleAfter = Number(a.stale_after_seconds) || 900;
  const requests = agedQuotaMetric(a.requests, "requests", elapsedSeconds, staleAfter);
  const tokens = agedQuotaMetric(a.tokens, "tokens", elapsedSeconds, staleAfter);
  const live = [requests, tokens].some((m) => m && m.source === "provider_headers" && m.fresh);
  return { ...a, requests, tokens, state: live ? "observed" : "stale" };
}
// When the gauge drawn from this admission next changes on its own (a reset or the stale horizon).
export function quotaExpiresInSeconds(admission) {
  const a = admission && typeof admission === "object" ? admission : null;
  if (!a || a.state !== "observed") return null;
  const staleAfter = Number(a.stale_after_seconds) || 900;
  const due = [quotaMetricExpiry(a.requests, "requests", staleAfter), quotaMetricExpiry(a.tokens, "tokens", staleAfter)].filter((v) => v != null);
  return due.length ? Math.min(...due) : null;
}
function quotaMetricLine(name, m) {
  if (!m || m.source !== "provider_headers") return null;
  const limit = Number(m.limit), remaining = Number(m.remaining);
  if (!(limit > 0) || !(remaining >= 0)) return null;
  const window = m.window === "day" ? "per day" : "per minute";
  if (!m.fresh) return { text: quotaCount(remaining) + " / " + quotaCount(limit) + " " + name + " left " + window + ", last reported " + quotaAge(m.age_seconds) + " ago", fresh: false, tone: "muted", fraction: null };
  const reset = m.reset_in_seconds != null ? ", resets in " + formatSeconds(m.reset_in_seconds) : "";
  const fraction = Math.min(1, remaining / limit);
  return {
    text: quotaCount(remaining) + " / " + quotaCount(limit) + " " + name + " left " + window + reset,
    fresh: true,
    tone: remaining <= 0 ? "danger" : fraction <= QUOTA_LOW_FRACTION ? "warn" : "ok",
    fraction, remaining, limit,
  };
}
function quotaDeclaredText(declared) {
  if (!declared || !declared.rpm) return null;
  return "Declared limit " + declared.rpm + "/min" + (declared.headroom_percent ? " — Ficelle keeps " + declared.headroom_percent + "% back" : "")
    + (declared.scope === "model" ? ", counted per model" : "");
}
export function quotaGauge(admission, elapsedSeconds = 0) {
  const a = agedQuotaAdmission(admission && typeof admission === "object" ? admission : null, elapsedSeconds);
  const none = { state: "unknown", tone: "muted", headline: "No observation yet", detail: "Provider has not reported a limit yet. Observations are kept in memory and start over when Ficelle restarts.", bar: null, lines: [] };
  if (!a || !["observed", "declared_only", "stale"].includes(a.state)) return none;
  const declared = quotaDeclaredText(a.declared);
  const req = quotaMetricLine("requests", a.requests);
  const tok = quotaMetricLine("tokens", a.tokens);
  const scope = a.requests && a.requests.scope === "model" && a.requests.model ? "Tightest model: " + a.requests.model : "";
  if (a.state === "declared_only") {
    return { state: "declared_only", tone: "muted", headline: declared || "Declared limit", detail: "Set in Ficelle's configuration and not yet confirmed by the provider. Provider-reported numbers appear after its first response with rate-limit headers.", bar: null, lines: [] };
  }
  if (a.state === "stale") {
    const age = Math.max(req ? Number(a.requests.age_seconds) || 0 : 0, tok ? Number(a.tokens.age_seconds) || 0 : 0);
    return { state: "stale", tone: "warn", headline: "Observation is " + quotaAge(age) + " old", detail: "Not shown as a live gauge. " + (declared ? declared + "." : "No limit is declared for this provider."), bar: null, lines: [req, tok].filter(Boolean).map((l) => ({ text: "Last report: " + l.text, tone: "muted" })) };
  }
  // observed: at least one metric is fresh.
  const extra = [];
  if (tok && tok.fresh) extra.push({ text: tok.text, tone: tok.tone });
  if (req && !req.fresh) extra.push({ text: "Requests: " + req.text, tone: "muted" });
  if (tok && !tok.fresh) extra.push({ text: "Tokens: " + tok.text, tone: "muted" });
  if (req && req.fresh) {
    if (req.remaining <= 0) extra.unshift({ text: "Ficelle skips it until the reset and tries other candidates.", tone: "muted" });
    if (scope) extra.push({ text: scope, tone: "muted" });
    return {
      state: "observed", tone: req.tone,
      headline: req.remaining <= 0 ? "Exhausted" + (a.requests.reset_in_seconds != null ? " — resets in " + formatSeconds(a.requests.reset_in_seconds) : "") : quotaCount(req.remaining) + " / " + quotaCount(req.limit) + " requests left",
      detail: QUOTA_ESTIMATE_NOTE, bar: { fraction: req.fraction, remaining: req.remaining, limit: req.limit }, lines: extra,
    };
  }
  // Only the token window is fresh: the request side stays an honest "not reported".
  return { state: "observed", tone: "muted", headline: "Request limit not reported", detail: (declared ? declared + ". " : "") + QUOTA_ESTIMATE_NOTE, bar: null, lines: extra };
}
// Rollup over every provider: how many are provider-reported, declared-only, stale or unknown, and
// which ones are exhausted or nearly so. Empty when no provider has anything to say.
export function quotaRollup(admissionBySource, elapsedSeconds = 0) {
  const out = { observed: 0, declared_only: 0, stale: 0, unknown: 0, exhausted: [], low: [] };
  Object.entries(admissionBySource || {}).forEach(([source, admission]) => {
    const g = quotaGauge(admission, elapsedSeconds);
    out[g.state] += 1;
    if (g.state === "observed" && g.bar) {
      if (g.bar.remaining <= 0) out.exhausted.push(source);
      else if (g.tone === "warn") out.low.push(source);
    }
  });
  return out;
}

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
  upstream_finish_error: "Model stopped on an error",
  truncated_before_content: "Token budget too small",
  bad_upstream_request: "Invalid request body",
  bad_upstream_contract: "Provider/model request contract",
  context_length_exceeded: "Context too small",
  malformed_tool_call: "Tool call without a name",
  client_disconnected: "Client hung up",
  service_restarting: "Ficelle was restarting",
  service_stopping: "Cut by Ficelle shutdown",
  request_deadline_exceeded: "Request budget spent",
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
  tls_error: "TLS handshake failed",
  network_unreachable: "Network unreachable",
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
  model_unavailable: "Model not selectable now",
  no_certified_coding_model: "No certified coding model",
  pre_stream_failure: "Failed before streaming",
  internal_error: "Ficelle internal error",
  bad_request: "Malformed request",
  server_busy: "Response too large",
  mid_stream_failure: "Mid-stream failure",
  runaway_output: "Runaway output",
};
export const REASON_DESCRIPTIONS = {
  empty_assistant_message: "The provider returned success, but no usable assistant text was found.",
  upstream_finish_error: "The provider returned success, but the model stopped on its own error rather than finishing the answer (finish_reason: error). The answer is unusable, so the attempt fails over to the next candidate.",
  truncated_before_content: "The request's max_tokens ran out before the model emitted any content, typically because a reasoning model spent the budget on reasoning tokens. Request-side limit: the model is not cooled.",
  bad_upstream_request: "The provider rejected the request body itself (HTTP 400/422), typically a malformed tool_call or an unsupported field. Request-side problem: no other candidate is tried and the model is not cooled.",
  bad_upstream_contract: "One model rejected an otherwise valid request option or required a private provider field. Ficelle keeps the request unchanged, tries the next candidate, and cools neither the model nor the provider.",
  context_length_exceeded: "The request does not fit this model's context window. Ficelle retries on a candidate whose context is at least the size the provider named; the model is not cooled. If every attempt ends this way, no eligible model is large enough for this request.",
  malformed_tool_call: "Ficelle refused the request before calling any provider: a tool call replayed in the conversation history carries no function name, which every provider rejects. The error names the offending message index — fix the client that built the history, or start a new conversation. No model was picked, called, or cooled.",
  client_disconnected: "The client closed the connection before Ficelle could answer — mid-stream, or while a fallback attempt was still being chosen. Client-side abort: the model is not cooled.",
  service_restarting: "Ficelle was stopping (service restart or update) and did not start further fallback attempts for this request. Nothing upstream failed and no model is cooled.",
  service_stopping: "Ficelle was stopping (service restart or update) and its shutdown drain ran out while this attempt was still in flight, so Ficelle ended it: a stream already started got a terminal error event, a request not yet answered got a 503. Nothing upstream failed and no model is cooled.",
  request_deadline_exceeded: "Ficelle's own wall-clock budget for the request ran out before the next fallback could start. The earlier attempts are what spent it, so no model is cooled and none is blamed for the ending.",
  invalid_success_json: "The provider returned success, but the JSON response could not be parsed.",
  invalid_json: "The judge response could not be parsed as JSON.",
  invalid_schema: "The judge returned JSON, but not in the schema Fusion expects.",
  judge_invalid_json: "The judge response could not be parsed as JSON.",
  judge_invalid_schema: "The judge returned JSON, but not in the schema Fusion expects.",
  rate_limited: "The provider asked Ficelle to slow down.",
  rate_limited_upstream: "The shared pool behind this one model id is saturated upstream, which says nothing about your account. Only this model is cooled (briefly); every other model of the provider keeps routing.",
  request_too_large: "The provider rejected this request as too large for its tokens-per-minute budget (HTTP 413). A transient, model-scoped limit that recharges on its own; the model is cooled briefly and recovers without action. When the request alone exceeds the limit the provider names, Ficelle remembers it for a week and skips this model for requests that large.",
  server_error: "The provider returned or reported a server-side error.",
  timeout: "The provider did not answer before the configured timeout.",
  tls_error: "The TLS handshake with the provider failed, so the request never reached it. Usually a transient network or certificate problem; the model is cooled briefly.",
  network_unreachable: "The provider's host could not be resolved or routed to, so the request never left this machine. Usually the local network or ISP is down: the provider is paused for a minute and no model's score is affected.",
  unavailable: "The provider or model was unavailable.",
  auth_or_credit: "The provider rejected the request because of credentials, credit, or account state.",
  billing_or_paid: "Ficelle blocked a model that appeared to require paid access.",
  quota_exhausted: "A free quota appears to be exhausted until a later probe succeeds.",
  no_free_quota: "The provider grants this model zero free-tier allocation (limit: 0); it is quarantined and will not be proposed.",
  model_not_found: "Either the provider will not serve this model id to this account — a 404/410 (listed in its catalog but not deployed), or a 403 naming it as a tier an admin has to switch on; it is then quarantined and stops being benchmarked until you clear it. Or, on a Requests row with no attempts, the client named a model id that is not in Ficelle's catalog: Ficelle answered 404 itself, so check the spelling against GET /v1/models. No provider was called.",
  model_unavailable: "The model id is in Ficelle's catalog but cannot be selected right now (cooldown, quarantine, quota or missing credentials). Ficelle answered 503 itself without calling a provider; the request row names the blocking reason and Retry-After gives the earliest retry.",
  no_certified_coding_model: "The coding profile only routes to models holding a valid Ficelle coding certification, and none is selectable at the moment. Ficelle answered 503 itself without calling a provider; the excluded counts show what blocked the rest of the pool.",
  pre_stream_failure: "The attempt failed before any byte reached the client (connection error, upstream error event or reset), so Ficelle could still fall back to the next candidate.",
  internal_error: "Ficelle's own request handler failed (HTTP 500) after accepting the request. The error type names the exception; the request id finds the stderr line to read.",
  bad_request: "Ficelle rejected the request as malformed (HTTP 400) before routing it, typically an invalid body or an unsupported header value.",
  server_busy: "A provider response grew past Ficelle's size safety limit, so Ficelle stopped reading it and answered 503 instead of buffering it.",
  benchmark_failed: "The model failed a capability check for this route.",
  failed_capability_check: "The model failed the latest capability check for this virtual model.",
  failed_benchmark: "The model failed the latest benchmark for this virtual model.",
  runaway_output: "The model generated past the profile's completion character budget. Ficelle stopped the answer, cooled the model, and used fallback when the response had not yet been committed.",
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
    ? () => { hideToast(); action.onClick(); }
    : null;
  // A failure has to interrupt a screen reader; a plain acknowledgement must not.
  toast.setAttribute("role", level === "error" ? "alert" : "status");
  toast.setAttribute("aria-live", level === "error" ? "assertive" : "polite");
  bindToastPause(toast);
  toast.classList.add("show");
  toastMs = action ? 14000 : (level === "error" ? errorToastMs(msg) : (level === "warn" ? 6000 : 2800));
  armToastTimer();
}

// An error often carries the way out ("run `ficelle set-key … --allow-plaintext`"), so its
// time on screen scales with its length instead of vanishing after the 2.8 s of a plain "Saved.".
const errorToastMs = (msg) => Math.min(20000, Math.max(6000, String(msg).length * 70));

// The single entry for a caught failure: the error styling is what makes it read as a failure,
// and a call site that forgets the level used to look like a success.
export function showError(e) { showToast(e?.message || String(e), "error"); }

let toastTimer = 0;
let toastMs = 0;
let toastBound = false;

function hideToast() {
  clearTimeout(toastTimer);
  $("toast").classList.remove("show");
}

function armToastTimer() {
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, toastMs);
}

// A toast that sits over controls for 14s must be dismissable, and must not vanish while
// the user is reading it or about to press its action: hover/focus pauses the countdown.
// Bound on first use rather than at import, because tests evaluate this module without a DOM.
function bindToastPause(toast) {
  if (toastBound) return;
  toastBound = true;
  const pause = () => clearTimeout(toastTimer);
  const resume = () => { if (toast.classList.contains("show")) armToastTimer(); };
  toast.addEventListener("mouseenter", pause);
  toast.addEventListener("mouseleave", () => { if (!toast.matches(":focus-within")) resume(); });
  toast.addEventListener("focusin", pause);
  toast.addEventListener("focusout", (e) => { if (!toast.contains(e.relatedTarget) && !toast.matches(":hover")) resume(); });
  $("toastClose").addEventListener("click", hideToast);
}

// The `auth.key_source` families a removal cannot reach, mapped to the next move — same shape
// as REASON_LABELS/reasonLabel above, so a family added later falls back to its own name and is
// still stated rather than silently dropped. `env` and `external` are the two nothing inside
// Ficelle can clear.
const KEY_SOURCE_REMEDIES = {
  env: "a process environment variable — unset it where the service starts, then restart",
  external: "an external credential resolver — clear the key in that integration",
};
export function keySourceRemedy(keySource) {
  const family = String(keySource || "");
  return KEY_SOURCE_REMEDIES[family] || family;
}

// The key form saved a key that resolution still ranks below an older one. The way out is to
// replace or clear only the entry that wins: `ficelle remove-key` clears every Ficelle store,
// the key just saved included, and could leave the provider with no key at all.
export function providerKeyShadowNotice(label, source, shadowedBy) {
  const family = String(shadowedBy || "");
  const where = KEY_SOURCE_REMEDIES[family] ? keySourceRemedy(family)
    : family === ".env" ? "the Ficelle .env file — replace or delete that line there"
    : "the " + family + " store — run `ficelle set-key " + source + "` on this host to replace it there";
  return label + " key saved, but an older key still takes precedence: " + where + ".";
}

// `keySource` is `auth.key_source` recomputed after the removal — the store the resolver
// would read a key from *now*, or null when none does. It is what makes the verdict honest
// when a key held in a process environment variable remains configured.
// `unverified` is the server's list of places that could not confirm the delete — an OS store
// that refused. It has to override the "no key was stored" reading: a store that would not
// answer the delete does not answer the re-resolution either, so `keySource` comes back null
// and every other signal here agrees on a removal that may never have happened.
export function providerKeyRemovalNotice(label, removed, keySource = null, unverified = []) {
  const cleared = removed.length
    ? (label + " key removed from " + removed.length + " location" + (removed.length > 1 ? "s" : "") + ".")
    // Silent when the removal was not confirmed: "no key was stored" is a claim about a store
    // that declined to answer, and the sentence below replaces it rather than qualifying it.
    : (unverified.length ? "" : ("No " + label + " key was stored by Ficelle."));
  const unconfirmed = unverified.length
    ? ((removed.length ? " " : "") + "Could not confirm the " + label + " key removal: "
      + unverified.join("; ") + ". It may still be stored.")
    : "";
  const stillConfigured = keySource
    ? (" " + label + " is still configured: a key still resolves from " + keySourceRemedy(keySource) + ".")
    : "";
  // Each case below states its own message, level and button in one place. Only the rules that
  // cut across them live here: a key that still resolves makes the toast red, unless a button
  // is still offered that could finish the job — and an unconfirmed removal is red whatever the
  // rest says, since no button on this toast can reach the store that refused.
  return {
    message: cleared + unconfirmed + stillConfigured,
    level: (unverified.length || keySource) ? "error" : (removed.length ? "ok" : "warn"),
  };
}

// The clipboard rejects for reasons the user can act on (permission denied, document not
// focused), so every caller needs the fallback message — not a raw DOMException in a toast.
export async function copyToClipboard(text, okMessage, failMessage = "Copy failed.") {
  try { await navigator.clipboard.writeText(text); showToast(okMessage); }
  catch { showToast(failMessage); }
}
