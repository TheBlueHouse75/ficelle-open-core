/* ===================== Ficelle admin — shared mutable state =====================
 * ES module live bindings: importers read `state`, `activeProfile`, … and always
 * see the current value. Importers cannot reassign an imported binding, so every
 * reassignment goes through a setter here. Object-property mutation (e.g.
 * `state.virtual_profiles = …`) stays inline since it mutates the shared object.
 */

export let state = null;
export let auditEntries = [];
export let draftProfiles = {};
export let draftFusion = {};
export let draftSettings = {};
export let compressionPayload = { observability: {}, events: [] };
export let activeProfile = "ficelle/auto-orchestrator";
export let dragged = null;
export let currentView = "routing";
export let selectedProvider = null;
// Provider detail model-list filter. `q` survives the detail re-render loadState() triggers
// after a key save / cooldown action; `src` records which provider the query belongs to so
// renderProviders can reset it when the selection moves. Session-only, not persisted.
export const providerModelFilter = { src: null, q: "" };
export const expandedModels = new Set();
// Available-models filter/sort state. Object-property mutation only (Sets/scalars),
// so importers always read the live value without a setter. Session-only, not persisted.
export const modelFilters = { providers: new Set(), caps: new Set(), minContext: 0, sortKey: "score", sortDir: "desc", newOnly: false };
// Requests view — derived request-log payloads + session-only filters (not persisted).
export let requestsPayload = { entries: [] };
export let requestsSummary = {};
// Live tail of the Requests list: SSE when the browser can hold an event stream, a 5s poll otherwise.
export let requestsLive = false;
export const expandedRequests = new Set();
export const requestFilters = { profile: "", source: "", reason: "", status: "", q: "", window: "24h" };

export function setState(v) { state = v; }
export function setAuditEntries(v) { auditEntries = v; }
export function setDraftProfiles(v) { draftProfiles = v; }
export function setDraftFusion(v) { draftFusion = v; }
export function setDraftSettings(v) { draftSettings = v; }
export function setCompressionPayload(v) { compressionPayload = v; }
export function setActiveProfile(v) { activeProfile = v; }
export function setDragged(v) { dragged = v; }
export function setCurrentView(v) { currentView = v; }
export function setSelectedProvider(v) { selectedProvider = v; }
export function setRequestsPayload(v) { requestsPayload = v; }
export function setRequestsSummary(v) { requestsSummary = v; }
export function setRequestsLive(v) { requestsLive = v; }
