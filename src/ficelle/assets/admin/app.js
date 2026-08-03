import { profileOrder, fusionModelId, profileLabels, profilePresets, CAP_PROFILE, CAP_STATE_META, $, esc, cloneJson, sleep,
  providerLabel, setProviderLabels, formatContext, formatTokenLimit, formatSeconds, formatDuration, pct, timeAgo, timeChip,
  hasIn, hasOut, hasParam, runtimeKey, freeAccess, freeModeLabel,
  providerFreeModeLabel, reasonDescription, reasonLabel, reasonWithCode, showToast, copyToClipboard } from "./lib.js";
import { state, auditEntries, draftProfiles, draftFusion, draftSettings, activeProfile, dragged, modelFilters, currentView, expandedModels,
  compressionPayload, setState, setAuditEntries, setDraftProfiles, setDraftFusion, setDraftSettings, setCompressionPayload, setActiveProfile, setDragged, setCurrentView,
  selectedProvider, setSelectedProvider,
  requestsPayload, requestsSummary, requestsLive, expandedRequests, requestFilters,
  setRequestsPayload, setRequestsSummary, setRequestsLive } from "./store.js";

/* ===================== Ficelle admin app ===================== */

    // Optional closed-pack views (Fusion/Compression). Populated by /admin/static/pro/pro-views.js
    // when ficelle_pro is installed; absent on a core-only install (placeholder shown instead).
    const proViews = (window.__ficelleProViews = window.__ficelleProViews || {});
    // A license key deep-linked from the purchase success page (#/license?key=…),
    // captured once at load before setView() rewrites the hash to a clean #/license. Pre-fills the
    // License key field so activation is one click — pre-fill only, never auto-submitted. Query
    // parameters are intentionally unsupported because the HTTP request line may be logged.
    const deepLinkedKey = deepLinkKey();
    // Scrub legacy ?key= links without extracting or using the value. The first request may
    // already have reached server logs, but this prevents persistence in history and referrers.
    if (location.search) {
      try {
        const cleaned = new URL(location.href);
        if (cleaned.searchParams.has("key")) {
          cleaned.searchParams.delete("key");
          history.replaceState(null, "", cleaned.pathname + cleaned.search + cleaned.hash);
        }
      } catch (e) { /* malformed URL — ignore */ }
    }

    /* ---------- formatting helpers ---------- */

    let benchmarkRunInFlight = false;
    // Per-card benchmarks in flight, keyed by model id. Tracked at module scope (not just on the
    // clicked button) so the spinner survives re-renders: navigating away and back rebuilds the
    // cards, and bindModelCard re-applies the spinner to whichever models are still being tested.
    const benchmarkingModels = new Set();
    // compressionRetrieveResult and FUSION_DRAFT_DEFAULTS moved to the closed-pack module (pro-views.js)
    // together with the Fusion/Compression view code that was their only consumer.

    function setButtonLoading(button, loading) {
      if (!button) return;
      button.disabled = Boolean(loading);
      button.classList.toggle("is-loading", Boolean(loading));
      if (loading) button.setAttribute("aria-busy", "true");
      else button.removeAttribute("aria-busy");
    }

    function serverBenchmarkRunning() {
      return state?.state?.admin_jobs?.benchmark?.status === "running";
    }

    function syncLongRunningActions() {
      setButtonLoading($("benchmarkBtn"), benchmarkRunInFlight || serverBenchmarkRunning());
    }

    const modelStats = (m) => state?.state?.stats?.[runtimeKey(m)] || null;
    function modelCooldown(m) { const c = state?.state?.cooldowns?.[runtimeKey(m)] || null; return c?.active ? c : null; }
    function providerCooldown(s) { const c = state?.state?.provider_cooldowns?.[s] || null; return c?.active ? c : null; }
    function providerManuallyDisabled(s) { return state?.state?.provider_disabled?.[s] || null; }
    const activeCount = (rows) => Object.values(rows || {}).filter((r) => r?.active).length;
    function quotaCooldownKey(m, scope) {
      if (scope === "model") return "model:" + m.source + "::" + m.upstream_id;
      if (scope === "account") return "account:" + m.source + "::" + (m.provider_account_id || m.source);
      // shared_account pools quota across an account's keys: the backend keys it by
      // account id only, with no source (router.py:quota_cooldown_key_for_scope).
      if (scope === "shared_account") return "shared_account:" + (m.provider_account_id || m.source);
      return "provider:" + m.source;
    }
    function modelQuotaCooldown(m) {
      const rows = state?.state?.quota_cooldowns || {};
      for (const scope of ["shared_account", "provider", "account", "model"]) {
        const c = rows[quotaCooldownKey(m, scope)] || null;
        if (c?.active) return c;
      }
      return null;
    }
    function providerQuotaCooldown(s, p) {
      const rows = state?.state?.quota_cooldowns || {};
      const direct = rows["provider:" + s] || null;
      if (direct?.active) return direct;
      // A shared-account pool is keyed by account id, not source, so resolve it via
      // the provider's account id when the summary carries one.
      const account = p?.provider_account_id;
      if (account) {
        const shared = rows["shared_account:" + account] || null;
        if (shared?.active) return shared;
      }
      return null;
    }
    const providerStats = (s) => state?.state?.provider_stats?.[s] || null;
    const providerError = (s) => state?.state?.provider_errors?.[s] || null;
    const modelBenchmark = (m) => state?.state?.benchmark_results?.[runtimeKey(m)]?.[activeProfile] || null;
    const modelVerified = (m, p = activeProfile) => state?.state?.verified_capabilities?.[runtimeKey(m)]?.[p] || null;
    const modelQuarantine = (m) => state?.state?.quarantine?.[runtimeKey(m)] || null;
    function modelFailedProfileEvidence(m, profileId = activeProfile) {
      const rows = state?.profiles?.[profileId]?.failed_profile_candidates || [];
      const found = rows.find((row) => row?.model_id === m.id || (row?.source === m.source && row?.upstream_id === m.upstream_id));
      if (found) return found;
      const verified = modelVerified(m, profileId);
      if (verified?.status === "failed") return { reason: "failed_capability_check", message: verified.message, failed_at: verified.failed_at };
      const benchmark = state?.state?.benchmark_results?.[runtimeKey(m)]?.[profileId] || null;
      if (benchmark?.status === "fail") return { reason: "failed_benchmark", message: benchmark.message, failed_at: benchmark.ran_at };
      return null;
    }


    /* ---------- profile logic ---------- */
    const modelMap = () => new Map((state?.catalog?.models || []).map((m) => [m.id, m]));
    const incAll = (items, req) => { const s = new Set((items || []).map(String)); return (req || []).every((x) => s.has(String(x))); };
    const incAny = (items, req) => { if (!(req || []).length) return true; const s = new Set((items || []).map(String)); return req.some((x) => s.has(String(x))); };
    function matchesReq(m, req) {
      req = req || {};
      if (Number(m.context_length || 0) < Number(req.min_context || state?.config?.min_context_length || 131072)) return false;
      if (req.free !== false && !freeAccess(m).eligible) return false;
      if (req.tools !== false && !m.supports_tools) return false;
      if (req.structured === true && !m.supports_structured_outputs) return false;
      if (req.structured === false && m.supports_structured_outputs) return false;
      if (!incAll(m.input_modalities, req.input_modalities)) return false;
      if (!incAny(m.input_modalities, req.input_modalities_any)) return false;
      if (!incAll(m.output_modalities, req.output_modalities)) return false;
      if (!incAny(m.output_modalities, req.output_modalities_any)) return false;
      if (!incAll(m.supported_parameters, req.supported_parameters)) return false;
      if (!incAny(m.supported_parameters, req.supported_parameters_any)) return false;
      return true;
    }
    function sortByScore(profileId, models) {
      return [...models].sort((a, b) => {
        const as = Number(a.auto_scores?.[profileId] ?? -1), bs = Number(b.auto_scores?.[profileId] ?? -1);
        if (as !== bs) return bs - as;
        const af = Number(modelStats(a)?.consecutive_failures || 0), bf = Number(modelStats(b)?.consecutive_failures || 0);
        if (af !== bf) return af - bf;
        const cd = Number(b.context_length || 0) - Number(a.context_length || 0);
        if (cd !== 0) return cd;
        return String(a.id || "").localeCompare(String(b.id || ""));
      });
    }
    function candidatePreview(profileId) {
      const profile = draftProfiles[profileId] || {};
      const excluded = new Set(excludedModelIds(profile));
      const eligible = (state?.catalog?.models || []).filter((m) => m.invokable && !excluded.has(String(m.id)) && !modelQuarantine(m) && !modelCooldown(m) && !providerCooldown(m.source) && !modelQuotaCooldown(m) && matchesReq(m, profile.requirements || {}));
      if (profile.mode !== "manual_order") return sortByScore(profileId, eligible);
      const byId = new Map(eligible.map((m) => [m.id, m])); const head = []; const seen = new Set();
      for (const id of profile.models || []) { const m = byId.get(id); if (!m || seen.has(m.id)) continue; head.push(m); seen.add(m.id); }
      const tail = profile.auto_tail === false ? [] : sortByScore(profileId, eligible.filter((m) => !seen.has(m.id)));
      return [...head, ...tail];
    }
    const candidateOrigin = (profile, m) => profile.mode !== "manual_order" ? "auto" : ((profile.models || []).includes(m.id) ? "manual" : "auto-fill");
    const eligibleCount = (profileId) => candidatePreview(profileId).length;
    function defaultReq() { return { free: true, tools: true, min_context: state?.config?.min_context_length || 131072, structured: null, input_modalities: [], input_modalities_any: [], output_modalities: [], output_modalities_any: [], supported_parameters: [], supported_parameters_any: [] }; }
    const currentProfile = () => draftProfiles[activeProfile] || { mode: "auto", models: [], excluded_models: [], auto_tail: true, requirements: defaultReq() };
    const uniqueStrings = (items) => Array.from(new Set((items || []).map(String).filter(Boolean)));
    const excludedModelIds = (profile = currentProfile()) => uniqueStrings(profile.excluded_models || []);
    const isExcludedFromProfile = (id, profile = currentProfile()) => excludedModelIds(profile).includes(String(id));
    function setProfile(next) {
      const cur = currentProfile();
      const req = { ...defaultReq(), ...(cur.requirements || {}), ...(next?.requirements || {}) };
      req.free = true; req.tools = true;
      req.min_context = Math.max(Number(state?.config?.min_context_length || 131072), Number(req.min_context || 131072));
      const models = uniqueStrings(next?.models || cur.models);
      const excluded = uniqueStrings(next?.excluded_models || cur.excluded_models);
      draftProfiles[activeProfile] = { ...cur, ...next, models, excluded_models: excluded, requirements: req };
      render();
    }

    /* ---------- unsaved-draft tracking ----------
       Mode, auto-fill, order, add/remove and Reset only touch draftProfiles; a reload
       used to drop them without a word, and Save looked identical either way. Compare
       the draft to what the server last returned so the button can say so.

       Both sides go through the same canonical form first: setProfile() normalizes
       requirements on every edit, so a raw compare would call a profile dirty after a
       round trip that changed nothing (Manual order then back to Automatic). */
    function canonicalProfile(profile) {
      const req = { ...defaultReq(), ...(profile?.requirements || {}) };
      req.free = true; req.tools = true;
      req.min_context = Math.max(Number(state?.config?.min_context_length || 131072), Number(req.min_context || 131072));
      return JSON.stringify({
        // Anything that is not manual_order routes as auto — treat it as auto here too.
        mode: profile?.mode === "manual_order" ? "manual_order" : "auto",
        models: uniqueStrings(profile?.models || []),
        excluded_models: uniqueStrings(profile?.excluded_models || []),
        // Absent means on, matching the auto_tail === false test in candidate ranking.
        auto_tail: profile?.auto_tail !== false,
        requirements: Object.keys(req).sort().reduce((out, key) => { out[key] = req[key]; return out; }, {}),
      });
    }
    function unsavedProfileIds() {
      const saved = state?.virtual_profiles || {};
      const drafts = draftProfiles || {};
      const ids = new Set([...Object.keys(saved), ...Object.keys(drafts)]);
      return [...ids].filter((id) => canonicalProfile(drafts[id]) !== canonicalProfile(saved[id]));
    }
    // Save posts every virtual model at once, so the count is what makes edits parked on
    // another card visible from the one you are looking at.
    function syncSaveButton() {
      const btn = $("saveBtn");
      if (!btn) return;
      // A save in flight owns the button; setButtonLoading re-enables it when it resolves.
      if (btn.classList.contains("is-loading")) return;
      const count = unsavedProfileIds().length;
      btn.disabled = count === 0;
      btn.title = count === 0
        ? "No unsaved changes."
        : "Save " + count + " edited virtual model" + (count > 1 ? "s" : "") + ". Exclude/restore already saved themselves.";
      const badge = $("saveCount");
      // No leading space: .btn's flex gap already separates the badge from the label.
      if (badge) badge.textContent = count ? "· " + count : "";
    }
    function addModel(id) {
      const p = currentProfile();
      const models = p.models || [];
      if (models.includes(id)) {
        showToast("Already in this virtual model.");
        return;
      }
      setProfile({
        mode: "manual_order",
        models: [...models, id],
        excluded_models: excludedModelIds(p).filter((x) => x !== id),
      });
    }
    function removeModel(id) { setProfile({ models: currentProfile().models.filter((x) => x !== id) }); }
    function excludeModel(id) {
      const p = currentProfile();
      const excludedModels = excludedModelIds(p);
      if (excludedModels.includes(id)) {
        showToast("Already excluded from this virtual model.");
        return;
      }
      setProfile({
        models: (p.models || []).filter((x) => x !== id),
        excluded_models: [...excludedModels, id],
      });
      saveProfiles("Model excluded from this virtual model.");
    }
    function restoreModel(id) {
      const p = currentProfile();
      const excludedModels = excludedModelIds(p);
      if (!excludedModels.includes(id)) {
        showToast("Model is not excluded from this virtual model.");
        return;
      }
      setProfile({ excluded_models: excludedModels.filter((x) => x !== id) });
      saveProfiles("Model restored to this virtual model.");
    }
    function moveModel(id, off) { const p = currentProfile(); const a = [...p.models]; const i = a.indexOf(id); const j = i + off; if (i < 0 || j < 0 || j >= a.length) return; a.splice(i, 1); a.splice(j, 0, id); setProfile({ models: a }); }
    function reorderModel(id, beforeId) { const p = currentProfile(); const a = p.models.filter((x) => x !== id); const bi = beforeId ? a.indexOf(beforeId) : -1; if (bi >= 0) a.splice(bi, 0, id); else a.push(id); setProfile({ mode: "manual_order", models: a }); }

    /* ---------- capability chips ---------- */
    function statusDot(m) {
      const q = modelQuarantine(m);
      if (q) return '<span class="sdot danger" title="Disabled &middot; ' + esc(q.note || q.reason) + '"></span>';
      const cd = modelCooldown(m);
      if (cd) return '<span class="sdot warn" title="Paused ' + esc(formatSeconds(cd.seconds_remaining)) + " &middot; " + esc(reasonDescription(cd.reason)) + '"></span>';
      if (providerCooldown(m.source)) return '<span class="sdot warn" title="Provider paused"></span>';
      const qc = modelQuotaCooldown(m);
      if (qc) return '<span class="sdot warn" title="Free quota paused &middot; next probe ' + esc(timeAgo(qc.next_probe_at_iso || qc.next_probe_at).label) + '"></span>';
      const failed = modelFailedProfileEvidence(m);
      if (failed) return '<span class="sdot warn" title="' + esc(reasonDescription(failed.reason)) + '"></span>';
      if (m.invokable) return '<span class="sdot ok" title="Ready"></span>';
      return '<span class="sdot warn" title="No API key &middot; ' + esc(m.auth_reason || "missing credentials") + '"></span>';
    }
    /* The dot carries the reason a model is held back only in a hover title —
       unreachable on touch, invisible at a glance — and the tinted card background
       says "will not route" without distinguishing the five causes behind it. Name
       the state in the card header instead.

       Reports the model's runtime state in every column, in the same order statusDot()
       resolves it, so the dot and the badge can never disagree. It is deliberately not
       tied to the card tint: `blocked` only counts cooldowns in the available list, so
       a paused model sitting in Your order keeps a plain card — it stays in the ranking
       and is skipped for now — but still says "Paused" rather than showing a bare amber
       dot. Ready models get nothing: a badge on every row would be noise, and the green
       dot already says it. Per-virtual-model exclusion is not here — capChips() already
       emits "Excluded here" for that, and it is a per-lane choice, not model state. */
    function statusLabel(m) {
      const q = modelQuarantine(m);
      if (q) return { cls: "danger", text: "Disabled", title: "Disabled &middot; " + esc(q.note || q.reason) };
      const cd = modelCooldown(m);
      if (cd) return { cls: "warn", text: "Paused", title: "Paused " + esc(formatSeconds(cd.seconds_remaining)) + " &middot; " + esc(reasonDescription(cd.reason)) };
      if (providerCooldown(m.source)) return { cls: "warn", text: "Provider paused", title: "Every model from this provider is paused" };
      const qc = modelQuotaCooldown(m);
      if (qc) return { cls: "warn", text: "Quota paused", title: "Free quota paused &middot; next probe " + esc(timeAgo(qc.next_probe_at_iso || qc.next_probe_at).label) };
      if (!m.invokable) return { cls: "warn", text: "No API key", title: "No API key &middot; " + esc(m.auth_reason || "missing credentials") };
      return null;
    }
    function statusLabelHtml(m) {
      const label = statusLabel(m);
      if (!label) return "";
      return '<span class="badge ' + label.cls + ' mcard-status" title="' + label.title + '">' + esc(label.text) + "</span>";
    }
    function scoreDetails(m, profileId = activeProfile) {
      return m?.auto_score_details?.[profileId] || null;
    }
    function scoreTitle(details) {
      if (!details) return "Ficelle ranking score for this virtual model";
      const bits = [
        "Base " + Number(details.score_base || 0).toFixed(1),
        "evidence " + esc(details.evidence_status || "missing"),
        "verified " + Number(details.verified_bonus || 0).toFixed(1),
        "benchmark " + Number(details.benchmark_bonus || 0).toFixed(1),
      ];
      if (Number(details.failure_penalty || 0) > 0) bits.push("failure penalty -" + Number(details.failure_penalty || 0).toFixed(1));
      if (details.evidence_status === "stale") bits.push("decay " + Math.round(Number(details.score_decay_factor || 0) * 100) + "%");
      if (details.evidence_reason) bits.push("reason " + esc(String(details.evidence_reason).replaceAll("_", " ")));
      if (details.last_evidence_at) bits.push("last evidence " + esc(timeAgo(details.last_evidence_at).label));
      return bits.join(" &middot; ");
    }
    // Provenance verdict for one capability of one model, aggregated across the profile that proves it.
    // Priority: benchmark (verified/refuted) > reference-refuted (oracle contradicts the default) >
    // reference-confirmed > provider default > real catalog declaration.
    function capabilityVerdict(m, capability) {
      const v = modelVerified(m, CAP_PROFILE[capability]);
      if (v?.status === "verified") return { state: "verified", at: v.verified_at };
      if (v?.status === "failed") return { state: "refuted", at: v.failed_at, msg: v.message };
      if ((m.capabilities_refuted_by_reference || []).includes(capability)) return { state: "refuted_reference" };
      if ((m.capabilities_from_reference || []).includes(capability)) return { state: "reference", confidence: m.reference_confidence || "" };
      if ((m.capabilities_from_defaults || []).includes(capability)) return { state: "default" };
      return { state: "catalog" };
    }
    function capBadge(m, label, capability) {
      const verdict = capabilityVerdict(m, capability);
      const meta = CAP_STATE_META[verdict.state] || CAP_STATE_META.catalog;
      let tip = label + " — " + meta.word;
      if (verdict.state === "verified" && verdict.at) tip += " · " + timeAgo(verdict.at).label;
      else if (verdict.state === "refuted") tip += (verdict.at ? " · " + timeAgo(verdict.at).label : "") + (verdict.msg ? " · " + verdict.msg : "") + " · Retest candidates to refresh.";
      else if (verdict.state === "reference" && verdict.confidence) tip += " (" + verdict.confidence + " confidence)";
      return '<span class="badge' + (meta.cls ? " " + meta.cls : "") + '" title="' + esc(tip) + '">' + esc(label + meta.mark) + "</span>";
    }
    // Verdict for a profile whose benchmark probes a composite skill with no single catalog flag
    // (e.g. auto-orchestrator's orchestration probe). Shows only once a benchmark has run — there is
    // no catalog claim to surface as an unconfirmed "?" beforehand, unlike Tools/JSON.
    function profileVerdictBadge(m, profileId, label) {
      const v = modelVerified(m, profileId);
      if (v?.status !== "verified" && v?.status !== "failed") return "";
      const ok = v.status === "verified";
      const meta = ok ? CAP_STATE_META.verified : CAP_STATE_META.refuted;
      const at = ok ? v.verified_at : v.failed_at;
      let tip = label + " — " + meta.word + (at ? " · " + timeAgo(at).label : "");
      if (!ok && v.message) tip += " · " + v.message;
      return '<span class="badge ' + meta.cls + '" title="' + esc(tip) + '">' + esc(label + meta.mark) + "</span>";
    }
    function capChips(m) {
      const out = [];
      if (isExcludedFromProfile(m.id)) out.push('<span class="badge warn">Excluded here</span>');
      const orch = profileVerdictBadge(m, "ficelle/auto-orchestrator", "Orchestration");
      if (orch) out.push(orch);
      if (m.supports_tools) out.push(capBadge(m, "Tools", "tools"));
      else out.push('<span class="badge warn">No tools</span>');
      if (m.supports_structured_outputs) out.push(capBadge(m, "JSON", "structured"));
      if (hasParam(m, "reasoning") || hasParam(m, "include_reasoning")) out.push(capBadge(m, "Reasoning", "reasoning"));
      if (hasIn(m, "image")) out.push(capBadge(m, "Image", "image"));
      if (hasIn(m, "video")) out.push(capBadge(m, "Video", "video"));
      if (hasIn(m, "audio")) out.push(capBadge(m, "Audio", "audio"));
      const sc = m.auto_scores?.[activeProfile];
      if (sc !== undefined && sc !== null && Number(sc) > 0) out.push('<span class="badge accent" title="' + scoreTitle(scoreDetails(m)) + '">Score ' + Number(sc).toFixed(0) + "</span>");
      const failed = modelFailedProfileEvidence(m);
      if (failed) out.push('<span class="badge warn" title="' + esc((failed.message || reasonDescription(failed.reason)) + " Retest candidates to refresh this status.") + '">' + esc(reasonLabel(failed.reason)) + "</span>");
      return out.join("");
    }
    function detailRows(m) {
      const rows = [];
      const push = (k, v, mono) => { if (v === null || v === undefined || v === "") return; rows.push('<div class="detail"><div class="d-k">' + esc(k) + '</div><div class="d-v' + (mono ? " mono" : "") + '">' + v + "</div></div>"); };
      push("Upstream id", esc(m.upstream_id), true);
      const ctxSrc = m.context_length_source === "default"
        ? "provider default, unconfirmed" + (m.context_length_estimate ? " &middot; reference estimates ~" + formatContext(m.context_length_estimate) : "")
        : "provider-declared";
      push("Context window", formatContext(m.context_length) + " &middot; " + ctxSrc);
      const mo = formatTokenLimit(m.max_completion_tokens); push("Max output", mo ? esc(mo) + " tokens" : null);
      push("Knowledge cutoff", m.knowledge_cutoff ? esc(m.knowledge_cutoff) : null);
      push("Tokenizer", m.tokenizer ? esc(m.tokenizer) : null, true);
      push("Instruct format", m.instruct_type ? esc(m.instruct_type) : null, true);
      if (m.is_moderated !== null && m.is_moderated !== undefined) push("Moderation", m.is_moderated ? "Moderated upstream" : "Not moderated");
      const ps = m.pricing_safety || {};
      const fa = freeAccess(m);
      const freeLabel = esc(freeModeLabel(fa.mode));
      push("Free access", fa.eligible ? "Free &middot; " + freeLabel : freeLabel);
      push("Free proof", fa.proof ? esc(fa.proof) : null, true);
      push("Free scope", fa.scope ? esc(fa.scope) : null);
      push("Free status", fa.status ? esc(fa.status) : null);
      push("Pricing", ps.status === "strict_zero" ? "zero-price verified" : esc(ps.status || "not verified"));
      const s = modelStats(m);
      if (s) {
        push("Success rate", Math.round(pct(s.successes, s.requests)) + "% &middot; " + Number(s.successes || 0) + "/" + Number(s.requests || 0) + " calls");
        if (s.latency_ewma != null) push("Avg latency", Number(s.latency_ewma).toFixed(2) + "s");
        // Kept apart: a caller-caused failure (a too-small max_tokens) updates the last reason
        // without extending the streak, so pairing them would blame the streak on the wrong event.
        // Splitting them also surfaces those failures, which carry no streak at all.
        if (s.consecutive_failures) push("Recent failures", '<span style="color:var(--danger)">' + Number(s.consecutive_failures) + " in a row</span>");
        if (s.last_failure_reason) push("Last failure", esc(reasonLabel(s.last_failure_reason)));
        if (s.last_success_at) push("Last success", timeChip(s.last_success_at));
      }
      const b = modelBenchmark(m);
      if (b) { const lat = b.latency_seconds != null ? " &middot; " + Number(b.latency_seconds).toFixed(2) + "s" : ""; push("Last benchmark", esc(b.status) + lat + " &middot; " + esc(b.message || b.test_type || "")); }
      const v = modelVerified(m);
      if (v) push("Verified capability", esc(v.status) + " &middot; " + esc(v.capability || v.test_type || "unknown") + " &middot; " + timeChip(v.verified_at || v.failed_at));
      const refCaps = m.capabilities_from_reference || [];
      if (refCaps.length) push("Reference prior", esc(refCaps.join(", ")) + (m.reference_confidence ? " &middot; " + esc(m.reference_confidence) + " confidence" : ""));
      const failed = modelFailedProfileEvidence(m);
      if (failed) push("Virtual model guard", '<span style="color:var(--warn)">' + esc(reasonWithCode(failed.reason)) + "</span> &middot; " + esc(failed.message || reasonDescription(failed.reason)) + " &middot; Retest candidates to refresh.");
      const cd = modelCooldown(m);
      if (cd) push("Cooling down", esc(formatSeconds(cd.seconds_remaining)) + " left &middot; " + esc(reasonWithCode(cd.reason)));
      const qc = modelQuotaCooldown(m);
      if (qc) push("Free quota", "Paused &middot; next probe " + timeChip(qc.next_probe_at_iso || qc.next_probe_at));
      const q = modelQuarantine(m);
      if (q) push("Disabled", esc(q.note || q.reason) + " &middot; " + timeChip(q.set_at));
      return rows.join("");
    }

    /* ---------- model card ---------- */
    // Single source of truth for every SVG icon used across the admin app. Each value is
    // the inner SVG content (no <svg> wrapper); callers (iconBtn, ic, health cards,
    // auditIcon) supply the wrapping <svg viewBox="0 0 24 24" ...>. Keys are concept-based.
    const ICONS = {
      exclude: '<circle cx="12" cy="12" r="9"/><path d="M8 8l8 8"/>',
      restore: '<path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/>',
      disable: '<path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><path d="M12 2v10"/>',
      resume: '<path d="M3 12a9 9 0 1 0 9-9"/><path d="M3 4v5h5"/>',
      refresh: '<path d="M3 12a9 9 0 0 1 9-9 9 9 0 0 1 6.7 3M21 12a9 9 0 0 1-9 9 9 9 0 0 1-6.7-3"/><path d="M21 3v5h-5M3 21v-5h5"/>',
      save: '<path d="M5 3h11l3 3v15H5z"/><path d="M9 3v5h6"/>',
      canary: '<path d="M3 12h4l2 5 4-12 2 7h6"/>',
      benchmark: '<path d="M4 19V5M4 19h16M9 16v-6M14 16V9M19 16v-4"/>',
      compression: '<path d="M4 7h16"/><path d="M7 12h10"/><path d="M10 17h4"/>',
      arrowUp: '<path d="M12 5v14M6 11l6-6 6 6"/>',
      arrowDown: '<path d="M12 5v14M6 13l6 6 6-6"/>',
      paused: '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>',
      watcher: '<circle cx="12" cy="12" r="3"/><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/>',
      search: '<path d="M21 21l-4.3-4.3"/><circle cx="11" cy="11" r="7"/>',
      retest: '<path d="M5 3l14 9-14 9z"/>',
      export: '<path d="M12 3v12m0 0 4-4m-4 4-4-4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
      external: '<path d="M15 3h6v6"/><path d="M21 3 10 14"/><path d="M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/>',
      fusion: '<path d="M4 7h5l3 5-3 5H4"/><path d="M20 7h-5l-3 5 3 5h5"/><path d="M9 12h6"/>',
      profiles: '<path d="M5 3h14v18l-7-4-7 4Z"/>',
      settings: '<circle cx="12" cy="12" r="3"/><path d="M3 12h3M18 12h3M12 3v3M12 18v3"/>',
      cooldownClock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
      auditDefault: '<circle cx="12" cy="12" r="9"/>',
      providers: '<rect x="4" y="4" width="16" height="6" rx="1"/><rect x="4" y="14" width="16" height="6" rx="1"/><path d="M8 7h.01M8 17h.01"/>',
      models: '<path d="M12 3l8 4-8 4-8-4 8-4z"/><path d="M4 12l8 4 8-4M4 17l8 4 8-4"/>',
      latency: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8z"/>',
    };

    function modelCard(m, ctx, index) {
      const id = m.id;
      const expanded = expandedModels.has(id);
      const blocked = !!(modelQuarantine(m) || (ctx === "available" && (modelCooldown(m) || providerCooldown(m.source) || modelQuotaCooldown(m) || !m.invokable)));
      const profileExcluded = isExcludedFromProfile(id);
      const grip = ctx === "selected"
        ? '<div class="mcard-grip" title="Drag to reorder"><svg viewBox="0 0 24 24" fill="currentColor"><circle cx="9" cy="6" r="1.6"/><circle cx="15" cy="6" r="1.6"/><circle cx="9" cy="12" r="1.6"/><circle cx="15" cy="12" r="1.6"/><circle cx="9" cy="18" r="1.6"/><circle cx="15" cy="18" r="1.6"/></svg></div>'
        : '<div class="mcard-grip"><span class="mcard-rank">' + (index != null && index >= 0 ? Number(index) + 1 : "") + "</span></div>";
      const rankPrefix = ctx === "selected" ? '<span class="mcard-rank">' + (Number(index) + 1) + "</span>" : "";
      const fa = freeAccess(m);
      const displayName = m.name || m.upstream_id;
      // Several provider snapshots can share one display name (e.g. Mistral labels every
      // mistral-medium-* version "mistral-medium-latest"); surface the upstream id so they're distinguishable.
      const versionSub = m.upstream_id && m.upstream_id !== displayName ? [m.upstream_id] : [];
      const ctxLabel = formatContext(m.context_length) + " context" + (m.context_length_source === "default" ? " (unconfirmed)" : "");
      const subParts = [esc(providerLabel(m.source)), ...versionSub, ctxLabel, freeModeLabel(fa.mode)];
      let actions = "";
      if (ctx === "selected") {
        actions =
          iconBtn("up", id, "Move up", ICONS.arrowUp) +
          iconBtn("down", id, "Move down", ICONS.arrowDown) +
          iconBtn("benchmark", id, "Benchmark this model", ICONS.retest) +
          (modelCooldown(m) ? iconBtn("clearcd", id, "Resume now", ICONS.resume) : "") +
          (modelQuarantine(m) ? iconBtn("unquar", id, "Re-enable", ICONS.restore) : iconBtn("quar", id, "Disable", ICONS.disable)) +
          iconBtn("exclude", id, "Exclude from this virtual model", ICONS.exclude) +
          '<button class="iconbtn danger" type="button" data-act="remove" data-id="' + esc(id) + '" title="Remove"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12h14"/></svg></button>';
      } else if (ctx === "excluded") {
        actions = iconBtn("restore", id, "Restore to this virtual model", ICONS.restore);
      } else {
        const disabled = blocked || currentProfile().models.includes(id);
        actions =
          iconBtn("benchmark", id, "Benchmark this model", ICONS.retest) +
          (modelCooldown(m) ? iconBtn("clearcd", id, "Resume now", ICONS.resume) : "") +
          (modelQuarantine(m) ? iconBtn("unquar", id, "Re-enable", ICONS.restore) : iconBtn("quar", id, "Disable", ICONS.disable)) +
          (profileExcluded
            ? iconBtn("restore", id, "Restore to this virtual model", ICONS.restore)
            : iconBtn("exclude", id, "Exclude from this virtual model", ICONS.exclude) +
              '<button class="iconbtn add" type="button" data-act="add" data-id="' + esc(id) + '" title="Add to virtual model" ' + (disabled ? "disabled" : "") + '><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg></button>');
      }
      return (
        '<article class="mcard ' + (ctx === "available" || ctx === "excluded" ? "no-grip " : "") + (blocked ? "is-blocked " : "") + (profileExcluded ? "is-excluded " : "") + (expanded ? "is-expanded" : "") + '"' + (ctx === "selected" || ctx === "available" ? ' draggable="true"' : "") + ' data-model="' + esc(id) + '" data-ctx="' + ctx + '">' +
          '<div class="mcard-top">' +
            grip +
            '<button class="mcard-id" type="button" data-act="toggle" data-id="' + esc(id) + '" style="border:0;background:none;text-align:left;cursor:pointer;padding:0">' +
              '<div class="mcard-name">' + statusDot(m) + (ctx === "selected" ? rankPrefix + " " : "") + esc(displayName) +
                '<svg class="chev" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--text-muted)"><path d="m6 9 6 6 6-6"/></svg></div>' +
              '<div class="mcard-sub">' + subParts.map((p, i) => (i ? '<span class="sep">&middot;</span>' : "") + "<span>" + esc(p) + "</span>").join("") + "</div>" +
            "</button>" +
            '<div class="mcard-right">' + statusLabelHtml(m) + '<div class="mcard-actions">' + actions + "</div></div>" +
          "</div>" +
          '<div class="mcard-chips">' + metricChip(m, ctx) + capChips(m) + "</div>" +
          '<div class="mcard-details"><div class="detail-grid">' + detailRows(m) + "</div></div>" +
        "</article>"
      );
    }
    const iconBtn = (act, id, title, path) => '<button class="iconbtn" type="button" data-act="' + act + '" data-id="' + esc(id) + '" title="' + esc(title) + '"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + path + "</svg></button>";

    function bindModelCard(root) {
      root.querySelectorAll("[data-act]").forEach((b) => {
        const id = b.dataset.id, act = b.dataset.act;
        if (act === "benchmark" && benchmarkingModels.has(id)) setButtonLoading(b, true);
        b.addEventListener("click", (e) => {
          e.stopPropagation();
          if (act === "toggle") { expandedModels.has(id) ? expandedModels.delete(id) : expandedModels.add(id); const card = b.closest(".mcard"); if (card) card.classList.toggle("is-expanded"); return; }
          if (act === "add") addModel(id);
          else if (act === "remove") removeModel(id);
          else if (act === "exclude") excludeModel(id);
          else if (act === "restore") restoreModel(id);
          else if (act === "up") moveModel(id, -1);
          else if (act === "down") moveModel(id, 1);
          else if (act === "clearcd") clearModelCooldown(id);
          else if (act === "quar") updateQuarantine(id, "add");
          else if (act === "unquar") updateQuarantine(id, "remove");
          else if (act === "benchmark") runModelBenchmark(id, b);
        });
      });
    }
    function bindDrag(root, ctx) {
      root.querySelectorAll('[data-model]').forEach((row) => {
        row.addEventListener("dragstart", (e) => { setDragged({ source: ctx, id: row.dataset.model }); row.classList.add("dragging"); e.dataTransfer.effectAllowed = ctx === "selected" ? "move" : "copy"; e.dataTransfer.setData("text/plain", JSON.stringify(dragged)); });
        row.addEventListener("dragend", () => row.classList.remove("dragging"));
        if (ctx === "selected") {
          row.addEventListener("dragover", (e) => e.preventDefault());
          row.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); const p = readDrag(e); if (!p?.id) return; if (p.source === "available") addModel(p.id); else reorderModel(p.id, row.dataset.model); });
        }
      });
    }
    const readDrag = (e) => { try { return JSON.parse(e.dataTransfer.getData("text/plain")) || dragged; } catch { return dragged; } };

    /* ---------- renderers ---------- */

    function renderProfileList() {
      const profileCards = profileOrder.map((pid) => {
        const lab = profileLabels[pid] || [pid, ""];
        const elig = eligibleCount(pid);
        const manual = (draftProfiles[pid] || {}).mode === "manual_order";
        const modeTag = manual ? '<span class="badge info">Manual</span>' : '<span class="badge">Auto</span>';
        return '<button class="profile-tab ' + (pid === activeProfile ? "active" : "") + '" type="button" data-profile="' + esc(pid) + '">' +
          '<span class="pt-name">' + esc(lab[0]) + '</span>' +
          '<span class="pt-counts">' + modeTag + '<span class="badge accent">' + elig + " usable</span></span>" +
          '<span class="pt-id">' + esc(pid) + '</span>' +
          '<span class="pt-meta">' + esc(lab[1]) + "</span>" +
          "</button>";
      });
      if (draftFusion?.enabled && draftFusion?.visible_in_models) {
        const obs = state?.fusion?.observability || {};
        const readiness = obs.readiness || "ready";
        const statusClass = readiness === "failed" ? "danger" : (readiness === "degraded" ? "warn" : "accent");
        profileCards.push(
          '<button class="profile-tab" type="button" data-fusion-profile="' + fusionModelId + '">' +
            '<span class="pt-name">Fusion</span>' +
            '<span class="pt-counts"><span class="badge info">Compound</span><span class="badge ' + statusClass + '">' + esc(readiness) + '</span></span>' +
            '<span class="pt-id">' + fusionModelId + '</span>' +
            '<span class="pt-meta">Free-model panel, judge, and synthesis. Configure it in the Fusion tab.</span>' +
          "</button>"
        );
      }
      $("profileList").innerHTML = profileCards.join("");
      $("profileList").querySelectorAll("[data-profile]").forEach((b) => b.addEventListener("click", () => { setActiveProfile(b.dataset.profile); render(); }));
      $("profileList").querySelectorAll("[data-fusion-profile]").forEach((b) => b.addEventListener("click", () => setView("fusion")));
    }

    function renderActiveProfile() {
      const lab = profileLabels[activeProfile] || [activeProfile, ""];
      const p = currentProfile();
      $("activeProfileName").textContent = lab[0];
      $("activeProfileId").textContent = activeProfile;
      $("eligibleBadge").textContent = eligibleCount(activeProfile) + " usable";
      const manual = p.mode === "manual_order";
      const full = candidatePreview(activeProfile);
      const preview = full.slice(0, 4);
      const more = full.length - preview.length;
      const previewHtml = preview.length
        ? preview.map((m, i) => { const sc = m.auto_scores?.[activeProfile]; const v = modelVerified(m); return '<div class="hp"><span class="n">' + (i + 1) + ".</span><span>" + esc(m.name || m.upstream_id) + " &middot; " + esc(candidateOrigin(p, m)) + (sc ? ' &middot; <span title="' + scoreTitle(scoreDetails(m)) + '">score ' + Number(sc).toFixed(0) + "</span>" : "") + (v ? " &middot; " + esc(v.status) : "") + "</span></div>"; }).join("")
          + (more > 0 ? '<div class="hp" style="color:var(--text-muted)"><span class="n">+</span><span>' + more + " more usable model" + (more > 1 ? "s" : "") + (manual ? ", further down this list" : ", ranked by score") + "</span></div>" : "")
        : '<div class="hp"><span>No usable models match this virtual model yet.</span></div>';
      const heading = !full.length
        ? "Routing preview:"
        : manual
          ? "Your order decides, not the score — Ficelle tries these " + full.length + " usable models top-first until one responds (paused models are already skipped):"
          : "Ficelle ranks all " + full.length + " usable models by score and tries them best-first until one responds:";
      $("profileHint").innerHTML = "<b>" + esc(lab[0]) + ".</b> " + esc(profilePresets[activeProfile] || "") +
        '<div class="hint-preview"><span style="color:var(--text-soft);font-weight:600">' + esc(heading) + "</span>" + previewHtml + "</div>";
      $("autoModeBtn").classList.toggle("active", p.mode !== "manual_order");
      $("manualModeBtn").classList.toggle("active", p.mode === "manual_order");
      $("autoTailToggle").checked = Boolean(p.auto_tail);
      $("orderBlock").style.display = manual ? "flex" : "none";
      $("autoNote").style.display = manual ? "none" : "flex";
      $("autoTailWrap").style.display = manual ? "" : "none";
      $("clearProfileBtn").style.display = manual ? "" : "none";
    }

    function renderSelected() {
      const models = modelMap(); const p = currentProfile(); const list = $("selectedList");
      const selectedIds = (p.models || []).filter((id) => !isExcludedFromProfile(id, p));
      if (!selectedIds.length) { list.innerHTML = '<div class="empty">No models picked. This virtual model runs fully automatic — Ficelle ranks every usable model for you.</div>'; }
      else {
        list.innerHTML = selectedIds.map((id, i) => {
          const m = models.get(id);
          if (!m) return '<article class="mcard no-grip is-blocked" data-model="' + esc(id) + '" data-ctx="selected"><div class="mcard-top"><div class="mcard-grip"><span class="mcard-rank">' + (i + 1) + '</span></div><div class="mcard-id"><div class="mcard-name">Unknown model</div><div class="mcard-sub mono">' + esc(id) + '</div></div><div class="mcard-right"><span class="badge danger mcard-status" title="This id is no longer in the catalog. Ficelle keeps it in case the model comes back; remove it to drop it for good.">Missing</span><div class="mcard-actions"><button class="iconbtn danger" data-act="remove" data-id="' + esc(id) + '" title="Remove"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14"/></svg></button></div></div></div></article>';
          return modelCard(m, "selected", i);
        }).join("");
      }
      bindModelCard(list); bindDrag(list, "selected");
      list.ondragover = (e) => { e.preventDefault(); list.classList.add("drop-active"); };
      list.ondragleave = () => list.classList.remove("drop-active");
      list.ondrop = (e) => { e.preventDefault(); list.classList.remove("drop-active"); const pl = readDrag(e); if (!pl?.id) return; if (pl.source === "available") addModel(pl.id); else reorderModel(pl.id, null); };
    }

    function renderExcluded() {
      const models = modelMap();
      const excluded = excludedModelIds();
      const block = $("excludedBlock");
      const list = $("excludedList");
      block.style.display = excluded.length ? "block" : "none";
      if (!excluded.length) {
        list.innerHTML = "";
        return;
      }
      list.innerHTML = excluded.map((id, i) => {
        const m = models.get(id);
        if (!m) return '<article class="mcard no-grip is-excluded" data-model="' + esc(id) + '" data-ctx="excluded"><div class="mcard-top"><div class="mcard-grip"><span class="mcard-rank">' + (i + 1) + '</span></div><div class="mcard-id"><div class="mcard-name">Unknown model</div><div class="mcard-sub mono">' + esc(id) + '</div></div><div class="mcard-right"><span class="badge warn mcard-status" title="Excluded from this virtual model, and no longer in the catalog either.">Excluded</span><div class="mcard-actions">' + iconBtn("restore", id, "Restore to this virtual model", ICONS.restore) + "</div></div></div></article>";
        return modelCard(m, "excluded", i);
      }).join("");
      bindModelCard(list);
    }

    function renderAvailable() {
      const q = $("searchInput").value.trim().toLowerCase();
      const f = modelFilters;
      let models = [...(state.catalog?.models || [])];
      if (f.providers.size) models = models.filter((m) => f.providers.has(m.source));
      models = models.filter((m) => matchesReq(m, currentProfile().requirements || {}));
      if (f.caps.size) models = models.filter((m) => [...f.caps].every((c) => modelHasCap(m, c)));
      if (f.minContext) models = models.filter((m) => Number(m.context_length || 0) >= f.minContext);
      models = f.sortKey === "score" ? sortByScore(activeProfile, models) : sortModels(f.sortKey, f.sortDir, models);
      if (f.sortKey === "score" && f.sortDir === "asc") models.reverse();
      if (q) models = models.filter((m) => [m.name, m.id, m.source, m.upstream_id, m.modality, ...(m.input_modalities || []), ...(m.supported_parameters || [])].join(" ").toLowerCase().includes(q));
      $("modelCountBadge").textContent = models.length + " models";
      const list = $("availableList");
      list.innerHTML = models.length ? models.map((m, i) => modelCard(m, "available", i)).join("") : '<div class="empty">No models match these filters.</div>';
      bindModelCard(list); bindDrag(list, "available");
    }

    /* ---------- available-models capability filter + sort helpers ---------- */
    function modelHasCap(m, cap) {
      if (cap === "json") return !!m.supports_structured_outputs;
      if (cap === "reasoning") return hasParam(m, "reasoning") || hasParam(m, "include_reasoning");
      if (cap === "image") return hasIn(m, "image");
      if (cap === "audio") return hasIn(m, "audio");
      if (cap === "video") return hasIn(m, "video");
      return true;
    }
    // Sort value for a non-score key. Returns null when the model has no runtime
    // data (latency/success) so it can always be pushed to the end of the list.
    function sortMetric(m, key) {
      if (key === "context") return Number(m.context_length || 0);
      if (key === "name") return String(m.name || m.upstream_id || "").toLowerCase();
      const s = modelStats(m);
      if (key === "latency") return s && s.latency_ewma != null ? Number(s.latency_ewma) : null;
      if (key === "success") return s && Number(s.requests) ? pct(s.successes, s.requests) : null;
      return null;
    }
    function sortModels(key, dir, models) {
      const mul = dir === "asc" ? 1 : -1;
      return [...models].sort((a, b) => {
        const av = sortMetric(a, key), bv = sortMetric(b, key);
        const an = av === null || av === undefined, bn = bv === null || bv === undefined;
        if (an && bn) return String(a.id || "").localeCompare(String(b.id || ""));
        if (an) return 1;   // no-data sorts last, regardless of direction
        if (bn) return -1;
        const c = key === "name" ? String(av).localeCompare(String(bv)) : av - bv;
        return c !== 0 ? c * mul : String(a.id || "").localeCompare(String(b.id || ""));
      });
    }
    function metricChip(m, ctx) {
      if (ctx !== "available") return "";
      const key = modelFilters.sortKey;
      if (key !== "latency" && key !== "success") return "";
      const s = modelStats(m);
      if (key === "latency") {
        if (!s || s.latency_ewma == null) return '<span class="badge" title="No latency recorded yet">No latency data</span>';
        return '<span class="badge mono mcard-metric" title="Average latency (EWMA)">' + Number(s.latency_ewma).toFixed(2) + "s</span>";
      }
      if (!s || !Number(s.requests)) return '<span class="badge" title="No calls recorded yet">No success data</span>';
      return '<span class="badge mono mcard-metric" title="' + Number(s.successes || 0) + "/" + Number(s.requests) + ' calls">' + Math.round(pct(s.successes, s.requests)) + "% success</span>";
    }

    /* ---------- available-models filter + sort bar ---------- */
    const CAP_OPTIONS = [["json", "JSON"], ["reasoning", "Reasoning"], ["image", "Vision / image"], ["audio", "Audio"], ["video", "Video"]];
    const SORT_OPTIONS = [["score", "Score (this virtual model)"], ["latency", "Latency"], ["success", "Success rate"], ["context", "Context"], ["name", "Name"]];
    const CHEV = '<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>';
    const DIR_DESC = '<path d="M12 5v14M6 13l6 6 6-6"/>';
    const DIR_ASC = '<path d="M12 19V5M6 11l6-6 6 6"/>';
    let openFilter = null;

    function contextTiers() {
      const floor = Number(state?.config?.min_context_length || 131072);
      const tiers = [[0, "Any context"]];
      for (const t of [256000, 512000, 1000000]) if (t > floor) tiers.push([t, "≥ " + formatContext(t)]);
      return tiers;
    }
    function fltTriggerLabel(id) {
      const f = modelFilters;
      if (id === "prov") return "Providers" + (f.providers.size ? '<span class="flt-count">' + f.providers.size + "</span>" : "");
      if (id === "cap") return "Capabilities" + (f.caps.size ? '<span class="flt-count">' + f.caps.size + "</span>" : "");
      if (id === "ctx") { const t = contextTiers().find(([v]) => v === f.minContext); return f.minContext && t ? esc(t[1]) : "Context"; }
      if (id === "sort") { const s = SORT_OPTIONS.find(([v]) => v === f.sortKey); return "Sort: " + esc((s ? s[1] : "Score").replace(" (this virtual model)", "")); }
      return "";
    }
    function fltTriggerOn(id) {
      const f = modelFilters;
      if (id === "prov") return !!f.providers.size;
      if (id === "cap") return !!f.caps.size;
      if (id === "ctx") return !!f.minContext;
      if (id === "sort") return f.sortKey !== "score";
      return false;
    }
    const dirTitle = () => modelFilters.sortDir === "asc" ? "Ascending — click for descending" : "Descending — click for ascending";
    const fltAnyActive = () => { const f = modelFilters; return !!(f.providers.size || f.caps.size || f.minContext || f.sortKey !== "score" || f.sortDir !== "desc"); };
    const dirSvg = () => '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + (modelFilters.sortDir === "asc" ? DIR_ASC : DIR_DESC) + "</svg>";

    function renderFilterBar() {
      openFilter = null;
      const f = modelFilters;
      const provs = [...new Set((state.catalog?.models || []).map((m) => m.source))].sort();
      for (const p of [...f.providers]) if (!provs.includes(p)) f.providers.delete(p);
      const tiers = contextTiers();
      if (f.minContext && !tiers.some(([v]) => v === f.minContext)) f.minContext = 0;

      const checkbox = (checked, attr, label) => '<label class="flt-opt"><input type="checkbox" ' + (checked ? "checked " : "") + attr + "><span>" + esc(label) + "</span></label>";
      const radio = (checked, name, attr, label) => '<label class="flt-opt"><input type="radio" name="' + name + '" ' + (checked ? "checked " : "") + attr + "><span>" + esc(label) + "</span></label>";
      const provPop = '<div class="flt-pop-head">Filter by provider</div>' + provs.map((s) => checkbox(f.providers.has(s), 'data-prov="' + esc(s) + '"', providerLabel(s))).join("");
      const capPop = '<div class="flt-pop-head">Require capability</div>' + CAP_OPTIONS.map(([v, l]) => checkbox(f.caps.has(v), 'data-cap="' + esc(v) + '"', l)).join("");
      const ctxPop = '<div class="flt-pop-head">Minimum context</div>' + tiers.map(([v, l]) => radio(f.minContext === v, "fltctx", 'data-ctx="' + v + '"', l)).join("");
      const sortPop = '<div class="flt-pop-head">Sort by</div>' + SORT_OPTIONS.map(([v, l]) => radio(f.sortKey === v, "fltsort", 'data-sort="' + esc(v) + '"', l)).join("");
      const trig = (id, popClass, pop) => '<div class="flt" data-flt="' + id + '">' +
        '<button type="button" class="flt-trigger' + (fltTriggerOn(id) ? " on" : "") + '" data-flt-trigger="' + id + '" aria-haspopup="true" aria-expanded="false">' + fltTriggerLabel(id) + CHEV + "</button>" +
        '<div class="flt-pop' + (popClass ? " " + popClass : "") + '">' + pop + "</div></div>";

      $("filterBar").innerHTML =
        '<div class="flt-row">' + trig("prov", "", provPop) + trig("cap", "", capPop) + trig("ctx", "align-right", ctxPop) + "</div>" +
        '<div class="flt-row">' + trig("sort", "", sortPop) +
        '<button type="button" class="flt-dir" id="fltDir" title="' + dirTitle() + '" aria-label="' + dirTitle() + '">' + dirSvg() + "</button>" +
        '<button type="button" class="flt-reset" id="fltReset" title="Reset filters and sort" aria-label="Reset filters and sort"' + (fltAnyActive() ? "" : ' style="display:none"') + '><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg></button></div>';
      bindFilterBar($("filterBar"));
    }
    function syncFilterBar() {
      const bar = $("filterBar"); if (!bar) return;
      ["prov", "cap", "ctx", "sort"].forEach((id) => {
        const t = bar.querySelector('[data-flt="' + id + '"] .flt-trigger'); if (!t) return;
        t.classList.toggle("on", fltTriggerOn(id));
        t.innerHTML = fltTriggerLabel(id) + CHEV;
      });
      const dir = bar.querySelector("#fltDir"); if (dir) { dir.innerHTML = dirSvg(); dir.title = dirTitle(); dir.setAttribute("aria-label", dirTitle()); }
      const reset = bar.querySelector("#fltReset"); if (reset) reset.style.display = fltAnyActive() ? "" : "none";
    }
    function closeFilterPops() {
      openFilter = null;
      document.querySelectorAll("#filterBar .flt.open").forEach((el) => { el.classList.remove("open"); const t = el.querySelector(".flt-trigger"); if (t) t.setAttribute("aria-expanded", "false"); });
    }
    function toggleFilterPop(fltEl) {
      if (!fltEl) return;
      const id = fltEl.getAttribute("data-flt");
      const willOpen = openFilter !== id;
      closeFilterPops();
      if (willOpen) { openFilter = id; fltEl.classList.add("open"); const t = fltEl.querySelector(".flt-trigger"); if (t) t.setAttribute("aria-expanded", "true"); }
    }
    function bindFilterBar(bar) {
      const f = modelFilters;
      const refresh = () => { renderAvailable(); syncFilterBar(); };
      bar.querySelectorAll("[data-flt-trigger]").forEach((b) => b.addEventListener("click", (e) => { e.stopPropagation(); toggleFilterPop(b.closest(".flt")); }));
      bar.querySelectorAll(".flt-pop").forEach((p) => p.addEventListener("click", (e) => e.stopPropagation()));
      bar.querySelectorAll("[data-prov]").forEach((cb) => cb.addEventListener("change", () => { cb.checked ? f.providers.add(cb.dataset.prov) : f.providers.delete(cb.dataset.prov); refresh(); }));
      bar.querySelectorAll("[data-cap]").forEach((cb) => cb.addEventListener("change", () => { cb.checked ? f.caps.add(cb.dataset.cap) : f.caps.delete(cb.dataset.cap); refresh(); }));
      bar.querySelectorAll("[data-ctx]").forEach((rb) => rb.addEventListener("change", () => { f.minContext = Number(rb.dataset.ctx); refresh(); }));
      bar.querySelectorAll("[data-sort]").forEach((rb) => rb.addEventListener("change", () => { f.sortKey = rb.dataset.sort; f.sortDir = (f.sortKey === "latency" || f.sortKey === "name") ? "asc" : "desc"; refresh(); }));
      const dir = bar.querySelector("#fltDir"); if (dir) dir.addEventListener("click", () => { f.sortDir = f.sortDir === "asc" ? "desc" : "asc"; refresh(); });
      const reset = bar.querySelector("#fltReset"); if (reset) reset.addEventListener("click", () => { f.providers.clear(); f.caps.clear(); f.minContext = 0; f.sortKey = "score"; f.sortDir = "desc"; renderFilterBar(); renderAvailable(); });
    }

    // Map each provider source to its bundled logo file. Several sources may share one
    // mark; sources without an entry fall back to a letter avatar.
    const PROVIDER_LOGO_FILES = { openrouter: "openrouter", nous: "nous", mistral: "mistral", nvidia: "nvidia", groq: "groq", gemini: "gemini", cerebras: "cerebras", siliconflow: "siliconflow", opencode_zen: "opencode", ollama: "ollama", cohere: "cohere", kilo: "kilo" };
    // Cache-bust token for bundled logo URLs. The file name is stable but a mark can be
    // re-traced/replaced in place; a browser that cached a logo under the old immutable
    // header won't revalidate it (a hard refresh doesn't dislodge a mask-image fetched by
    // JS), so the URL itself must change. Bump this when a logo's pixels change.
    const LOGO_ASSET_VERSION = "2";
    function providerLogo(src) {
      const file = PROVIDER_LOGO_FILES[src];
      if (file) {
        const url = "/admin/logos/" + esc(file) + ".svg?v=" + LOGO_ASSET_VERSION;
        return '<span class="provider-logo" title="' + esc(providerLabel(src)) + '"><span class="mark" style="-webkit-mask-image:url(' + url + ');mask-image:url(' + url + ')"></span></span>';
      }
      return '<span class="provider-logo"><span class="letter">' + esc((providerLabel(src) || "?").charAt(0).toUpperCase()) + "</span></span>";
    }
    // Catalogue composition for the provider bar: ordered buckets over raw_count, so the dominant
    // rejection reason (usually "not free" or "not chat" — neither was surfaced by the old reject
    // chips) is visible. Each bucket's color is shared by its bar segment and its legend swatch.
    // `not free` merges the free-quota (not_eligible) and free-model (unsafe_pricing) variants;
    // `paid` is a subset of unsafe_pricing so it is not added again. These buckets cover every
    // reject reason emitted by build_catalog in router.py (not_chat/not_eligible/unsafe_pricing/
    // no_tools/small_context/invalid); a new backend reason would fall into `other` rather than
    // be dropped. Any remainder (e.g. de-duped duplicate ids) also lands in `other` so the
    // segments always sum to raw_count.
    function providerComposition(p) {
      const rej = p.rejected || {};
      const n = (k) => Number(rej[k] || 0);
      const raw = Number(p.raw_count || 0);
      const buckets = [
        { label: "usable", count: Number(p.accepted_count || 0), color: "var(--accent)" },
        { label: "not free", count: n("not_eligible") + n("unsafe_pricing"), color: "var(--text-muted)" },
        { label: "not chat", count: n("not_chat"), color: "var(--violet)" },
        { label: "no tools", count: n("no_tools"), color: "var(--info)" },
        { label: "too small", count: n("small_context"), color: "var(--warn)" },
        { label: "invalid", count: n("invalid"), color: "var(--danger)" },
      ];
      const known = buckets.reduce((sum, b) => sum + b.count, 0);
      const other = Math.max(0, raw - known);
      if (other > 0) buckets.push({ label: "other", count: other, color: "color-mix(in srgb, var(--text-muted) 32%, transparent)" });
      return { raw, buckets };
    }
    function providerModelReasonLabel(reason) {
      return ({
        accepted: "accepted",
        not_eligible: "not eligible",
        not_free: "not eligible",
        no_tools: "no tools",
        small_context: "too small",
        invalid: "invalid",
      })[reason] || reason || "unknown";
    }
    function compactNumber(value) {
      const n = Number(value || 0);
      if (!Number.isFinite(n)) return "0";
      return n >= 1000 ? Math.round(n / 1000) + "k" : String(n);
    }
    function providerModelList(p, openByDefault) {
      const rows = Array.isArray(p.models) ? p.models : [];
      if (!rows.length) return "";
      return '<details class="provider-models"' + (openByDefault ? " open" : "") + '><summary>' + rows.length + " models</summary>" +
        '<div class="provider-model-list">' +
        rows.map((m) => {
          const ok = m.status === "accepted";
          const reason = providerModelReasonLabel(m.reason);
          const context = Number(m.context_length || 0);
          return '<div class="provider-model-row ' + (ok ? "ok" : "muted") + '">' +
            '<div class="provider-model-main"><span class="provider-model-id">' + esc(m.id || m.model_id || "unknown") + "</span>" +
            (m.name && m.name !== m.id ? '<span class="provider-model-name">' + esc(m.name) + "</span>" : "") + "</div>" +
            '<div class="provider-model-meta">' +
              '<span class="badge ' + (ok ? "ok" : "warn") + '">' + esc(reason) + "</span>" +
              (m.free_mode ? '<span class="badge">' + esc(freeModeLabel(m.free_mode)) + "</span>" : "") +
              (context ? '<span class="badge mono">' + esc(compactNumber(context)) + " ctx</span>" : "") +
            "</div>" +
          "</div>";
        }).join("") +
        "</div></details>";
    }
    // Providers view — master/detail split. The left list (#providerList) is a navigation
    // index; the right panel (#providerDetail) carries the full per-provider surface (config
    // zone + model listing). render() rebuilds both, so selection lives in the store and
    // survives re-renders triggered by loadState() after a key save / cooldown action.
    function renderProviders() {
      const provs = state.catalog?.providers || {};
      const keys = Object.keys(provs).sort();
      const listEl = $("providerList"), detailEl = $("providerDetail");
      if (!listEl || !detailEl) return;
      renderProviderStats(keys, provs);
      if (!keys.length) {
        listEl.innerHTML = "";
        detailEl.innerHTML = '<div class="empty">No providers configured.</div>';
        return;
      }
      if (!selectedProvider || !keys.includes(selectedProvider)) setSelectedProvider(keys[0]);
      listEl.innerHTML = keys.map((src) => providerListItemHTML(src, provs[src] || {})).join("");
      listEl.querySelectorAll("[data-prov]").forEach((b) => b.addEventListener("click", () => {
        if (b.dataset.prov === selectedProvider) return;
        setSelectedProvider(b.dataset.prov);
        renderProviders();
      }));
      detailEl.innerHTML = providerDetailHTML(selectedProvider, provs[selectedProvider] || {});
      bindProviderDetailActions(detailEl);
    }

    // Median of a numeric list (null when empty). Returns the average of the two middle values
    // for an even count. Used for the providers latency KPI so a few slow providers don't drag
    // the figure up the way a mean does.
    function median(nums) {
      if (!nums.length) return null;
      const s = [...nums].sort((a, b) => a - b);
      const mid = Math.floor(s.length / 2);
      return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    }

    // Shared KPI card markup for the Health and Providers stat grids. `k`/`v` are escaped here;
    // `ic` (inline SVG) and `note` (may carry &middot; entities or a button) are raw HTML.
    function hcardHTML(c) {
      return '<article class="card hcard"><div class="hc-k"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--' + c.cls + ')">' + c.ic + '</svg>' + esc(c.k) + '</div><div class="hc-v">' + esc(String(c.v)) + '</div><div class="hc-note">' + c.note + "</div></article>";
    }

    // Top-of-page KPI row for the Providers view. Reuses the Health .hcard markup so there is no
    // new component. Every value maps to a field already exposed by /admin/status.json: connectivity
    // (invokable), aggregate usable models (accepted_count vs full catalogue), cumulative call
    // success, and the median of per-provider EWMA latencies — all since-start counters from
    // provider_stats, not a rolling window (the Requests page carries true windowed p50/p95).
    function renderProviderStats(keys, provs) {
      const el = $("providerStats");
      if (!el) return;
      if (!keys.length) { el.innerHTML = ""; return; }
      let connected = 0, usableNow = 0, totalRaw = 0;
      let reqSum = 0, okSum = 0, failSum = 0, fastest = null;
      const latencies = [];
      keys.forEach((src) => {
        const p = provs[src] || {};
        const { accepted, raw } = providerCounts(p);
        totalRaw += raw;
        if (p.invokable) { connected += 1; usableNow += accepted; }
        const st = providerStats(src);
        if (!st) return;
        const reqN = Number(st.requests || 0);
        reqSum += reqN; okSum += Number(st.successes || 0); failSum += Number(st.failures || 0);
        if (st.latency_ewma != null && reqN > 0) {
          const l = Number(st.latency_ewma);
          latencies.push(l);
          if (!fastest || l < fastest.l) fastest = { src, l };
        }
      });
      const missing = keys.length - connected;
      const rate = reqSum ? pct(okSum, reqSum) : null;
      const medLat = median(latencies);
      const cards = [
        { k: "Providers", ic: ICONS.providers, v: connected + " / " + keys.length,
          note: "connected &middot; " + (missing ? missing + " missing credentials" : "all keyed"),
          cls: missing ? "warn" : "ok" },
        { k: "Usable models", ic: ICONS.models, v: String(usableNow),
          note: "routable now &middot; of " + totalRaw + " returned", cls: "accent" },
        // Same ok/warn/danger thresholds as the per-provider detail badge (95%+ healthy, 80%+ warns).
        { k: "Success rate", ic: ICONS.restore, v: rate == null ? "—" : rate + "%",
          note: rate == null ? "no traffic yet" : (reqSum + " calls &middot; " + failSum + " failures"),
          cls: rate == null ? "accent" : (rate >= 95 ? "ok" : rate >= 80 ? "warn" : "danger") },
        { k: "Median latency", ic: ICONS.latency, v: medLat == null ? "—" : medLat.toFixed(2) + "s",
          note: medLat == null ? "no latency recorded yet" : (latencies.length + " of " + keys.length + " providers measured" + (fastest ? " &middot; fastest " + esc(providerLabel(fastest.src)) : "")),
          cls: "info" },
      ];
      el.innerHTML = cards.map(hcardHTML).join("");
    }

    // Single source of truth for provider status precedence (disabled → cooldown → quota →
    // connected → no key). Returns a neutral state key + dynamic label; each surface maps the
    // state to its own presentation — the left-list dot tone (STATUS_DOT_TONE) and the detail
    // badge (providerDetailHTML) — so the precedence chain is never duplicated.
    function providerStatus(src, p) {
      const cd = providerCooldown(src), qc = providerQuotaCooldown(src, p);
      // Disabled by static config OR a manual operator toggle — both hold the provider out.
      if (p.enabled === false || providerManuallyDisabled(src)) return { state: "disabled", label: "Disabled" };
      if (cd) return { state: "cooldown", label: "Paused " + formatSeconds(cd.seconds_remaining) };
      if (qc) return { state: "quota", label: "Free quota paused" };
      if (p.invokable) return { state: "connected", label: "Connected" };
      return { state: "no_key", label: "No API key" };
    }
    const STATUS_DOT_TONE = { connected: "ok", no_key: "muted" }; // default warn
    // accepted = models routable here; raw = everything the provider's catalog returned (the
    // composition bar, the list count, and the Usable-models KPI all use raw as the denominator
    // so "usable of total" reads the same everywhere — never the inventory length, which omits
    // paid models for free-model providers and so differs from raw).
    function providerCounts(p) {
      return { accepted: Number(p.accepted_count || 0), raw: Number(p.raw_count || 0) };
    }

    function providerListItemHTML(src, p) {
      const { accepted, raw } = providerCounts(p);
      const s = providerStatus(src, p);
      return '<button type="button" class="prov-list-item' + (src === selectedProvider ? " active" : "") + '" data-prov="' + esc(src) + '">' +
        providerLogo(src) +
        '<span class="pli-main"><span class="pli-name">' + esc(providerLabel(src)) + "</span>" +
          '<span class="pli-sub"><span class="pli-dot ' + (STATUS_DOT_TONE[s.state] || "warn") + '"></span>' + esc(s.label) + "</span></span>" +
        '<span class="pli-count mono" title="usable / total returned">' + accepted + '<span class="pli-sep">/</span>' + raw + "</span>" +
      "</button>";
    }

    // Full per-provider detail: the exact card body that used to live in the tile grid, now
    // shown one at a time. Config article (header, bars, chips, note, auth, key form) above the
    // model listing, which is forced open here since it is the panel's dedicated zone.
    function providerDetailHTML(src, p) {
      p = p || {};
      const cd = providerCooldown(src), qc = providerQuotaCooldown(src, p), st = providerStats(src), err = providerError(src);
      const probe = state?.state?.quota_probe_results?.["provider:" + src] || null;
      const isFreeQuota = p.provider_class === "free_quota" || p.free_mode === "quota_free" || p.source_type === "free_quota";
      const quotaBadge = isFreeQuota
        ? (qc ? '<span class="badge warn">Quota held out</span>' : "")
        : "";
      const probeBadge = probe ? '<span class="badge" title="' + esc(timeAgo(probe.probed_at).title + (probe.detail || probe.reason ? " — " + (probe.detail || probe.reason) : "")) + '">last probe: ' + esc(probe.status || "unknown") + " &middot; " + esc(timeAgo(probe.probed_at).label) + "</span>" : "";
      // A provider only counts as currently in error while a cooldown is active; once it expires the
      // last error is historical, so show it muted ("recovered") instead of red to avoid implying a live failure.
      // provider_errors only clears on the next successful call, so drop a stale recovered note after 6h to keep healthy providers clean.
      const errActive = !!(cd || qc);
      const errAgeMs = err && err.seen_at ? Date.now() - new Date(err.seen_at).getTime() : Infinity;
      const errBadge = err && (errActive || errAgeMs <= 6 * 3600 * 1000)
        ? '<span class="badge' + (errActive ? " danger" : "") + '" title="' + esc([err.status, err.detail].filter(Boolean).join(" · ")) + '">'
          + (errActive
              ? "last error: " + esc(err.reason || "unknown")
              : "recovered &middot; " + esc(err.reason || "unknown") + " " + esc(timeAgo(err.seen_at).label))
          + "</span>"
        : "";
      const s = providerStatus(src, p);
      const stateBadge = '<span class="badge ' + (s.state === "connected" ? "ok" : "warn") + '">'
        + (s.state === "disabled" ? "" : '<span class="dot"></span>') + esc(s.label) + "</span>";
      const { accepted } = providerCounts(p);
      // Chip families: Class (static classification), Runtime (usage metrics), Health (incidents).
      const callsOk = st ? pct(st.successes, st.requests) : null;
      const runtimePills = st
        ? '<span class="badge ' + (callsOk >= 95 ? "ok" : callsOk >= 80 ? "warn" : "danger") + '">' + callsOk + "% calls ok</span>"
          + (st.latency_ewma != null ? '<span class="badge" title="Average latency (EWMA)">' + Number(st.latency_ewma).toFixed(2) + "s latency</span>" : "")
        : "";
      const nextProbeBadge = qc ? '<span class="badge warn">next quota probe ' + esc(timeAgo(qc.next_probe_at_iso || qc.next_probe_at).label) + "</span>" : "";
      const healthPills = [quotaBadge, errBadge, nextProbeBadge, probeBadge].filter(Boolean).join("");
      // Manual on/off toggle (mirrors the per-model Disable/Re-enable): holds the whole provider
      // out of routing until re-enabled, independent of automatic cooldowns and the config flag.
      const manuallyDisabled = !!providerManuallyDisabled(src);
      const toggleBtn = manuallyDisabled
        ? '<button class="btn ghost sm" data-enable-provider="' + esc(src) + '">' + ic(ICONS.restore) + "Enable provider</button>"
        : '<button class="btn ghost sm" data-disable-provider="' + esc(src) + '">' + ic(ICONS.disable) + "Disable provider</button>";
      const actions = (cd ? '<button class="btn ghost sm" data-clear-provider="' + esc(src) + '">Resume now</button>' : "")
        + (isFreeQuota ? '<button class="btn ghost sm" data-probe-provider="' + esc(src) + '">Probe quota</button>' : "")
        + toggleBtn;
      // Catalogue composition bar over the provider's true raw_count (segments + colour-keyed legend).
      const comp = providerComposition(p);
      const shownBuckets = comp.buckets.filter((b) => b.count > 0);
      const compBlock = comp.raw > 0
        ? '<div class="provider-composition"><div class="comp-head"><span class="comp-title">Catalogue composition</span>'
          + '<span class="comp-sub">' + accepted + " usable &middot; " + comp.raw + " returned</span></div>"
          + '<div class="comp-bar">' + shownBuckets.map((b) => '<span style="width:' + pct(b.count, comp.raw) + "%;background:" + b.color + '" title="' + b.count + " " + b.label + '"></span>').join("") + "</div>"
          + '<div class="comp-legend">' + shownBuckets.map((b) => '<span class="comp-legend-item"><span class="sw" style="background:' + b.color + '"></span>' + b.count + " " + esc(b.label) + "</span>").join("") + "</div></div>"
        : '<div class="provider-note">No catalogue loaded' + (p.invokable ? "" : " — add a key to fetch this provider's models") + ".</div>";
      // A stored key (key_source set) shows a redacted preview (first/last chars only, never the
      // full value) plus where it lives; Replace reveals the hidden paste box to rotate it. Gate on
      // key_source, not p.invokable: a keyless_local provider is invokable with no stored key
      // and must keep the plain paste box rather than a false mask.
      const storeChip = p.key_source ? '<span class="badge mono" title="Where Ficelle keeps this key">' + esc(p.key_source) + "</span>" : "";
      const keyMask = p.key_preview || "•".repeat(16);
      const keyForm = p.key_source
        ? '<div class="provider-chips key-form">' +
            '<span class="key-mask mono" title="Stored key — only the first and last characters are shown">' + esc(keyMask) + "</span>" +
            storeChip +
            '<button class="btn ghost sm" data-replace-key="' + esc(src) + '">Replace</button>' +
            '<button class="btn ghost sm" data-remove-key="' + esc(src) + '">Remove key</button>' +
            '<input type="password" class="key-input is-hidden" data-key-input="' + esc(src) + '" placeholder="Paste new ' + esc(providerLabel(src)) + ' key" autocomplete="off" spellcheck="false" />' +
            '<button class="btn ghost sm is-hidden" data-set-key="' + esc(src) + '">Save</button>' +
          "</div>"
        : '<div class="provider-chips key-form">' +
            '<input type="password" class="key-input" data-key-input="' + esc(src) + '" placeholder="Paste ' + esc(providerLabel(src)) + ' API key" autocomplete="off" spellcheck="false" />' +
            '<button class="btn ghost sm" data-set-key="' + esc(src) + '">Save key</button>' +
          "</div>";
      return '<div class="provider-detail-stack">' +
        '<article class="card provider-card">' +
          '<div class="provider-top"><div class="provider-head">' + providerLogo(src) + '<div><div class="provider-name">' + esc(providerLabel(src)) + '</div><div class="provider-url">' + esc(p.catalog_url || "catalog unavailable") + '</div></div></div>' + stateBadge + "</div>" +
          compBlock +
          '<div class="pill-rows">' +
            '<div class="pill-row"><span class="pill-row-label">Class</span><span class="badge outline">' + esc(providerFreeModeLabel(p)) + "</span></div>" +
            (runtimePills ? '<div class="pill-row"><span class="pill-row-label">Runtime</span>' + runtimePills + "</div>" : "") +
            (healthPills ? '<div class="pill-row"><span class="pill-row-label">Health</span>' + healthPills + "</div>" : "") +
          "</div>" +
          (p.key_url ? '<div class="provider-keyurl"><a href="' + esc(p.key_url) + '" target="_blank" rel="noopener noreferrer">' + ic(ICONS.external) + "Get your " + esc(providerLabel(src)) + " key</a></div>" : "") +
          (p.invokable ? "" : '<div class="provider-chips"><span class="badge" style="color:var(--text-muted)">' + esc(p.auth_reason || "Add this provider key to use its models") + "</span></div>") +
          (actions ? '<div class="provider-chips">' + actions + "</div>" : "") +
          keyForm +
        "</article>" +
        providerModelList(p, true) +
      "</div>";
    }

    function bindProviderDetailActions(root) {
      root.querySelectorAll("[data-clear-provider]").forEach((b) => b.addEventListener("click", () => clearProviderCooldown(b.dataset.clearProvider)));
      root.querySelectorAll("[data-probe-provider]").forEach((b) => b.addEventListener("click", () => probeProviderQuota(b.dataset.probeProvider, b)));
      root.querySelectorAll("[data-disable-provider]").forEach((b) => b.addEventListener("click", () => toggleProvider(b.dataset.disableProvider, false, b)));
      root.querySelectorAll("[data-enable-provider]").forEach((b) => b.addEventListener("click", () => toggleProvider(b.dataset.enableProvider, true, b)));
      root.querySelectorAll("[data-set-key]").forEach((b) => b.addEventListener("click", () => setProviderKey(b.dataset.setKey)));
      root.querySelectorAll("[data-remove-key]").forEach((b) => b.addEventListener("click", () => removeProviderKey(b.dataset.removeKey)));
      root.querySelectorAll("[data-replace-key]").forEach((b) => b.addEventListener("click", () => {
        const form = b.closest(".key-form");
        if (!form) return;
        form.querySelectorAll(".is-hidden").forEach((el) => el.classList.remove("is-hidden"));
        form.querySelectorAll(".key-mask, [data-replace-key]").forEach((el) => el.classList.add("is-hidden"));
        const input = form.querySelector(".key-input");
        if (input) input.focus();
      }));
    }

	    // knownProfileIds, fusionProfileOptions, fusionCallMultiplierDraft, and fusionMetric moved to
	    // the closed-pack module (ficelle_pro/assets/admin/pro-views.js). fusionSetting and
	    // fusionSettingsGroup stay here because the core Settings view also uses them as layout primitives.
	    function fusionSetting(id, label, control, help, wide, className) {
	      return '<div class="setting' + (wide ? " setting-wide" : "") + (className ? " " + className : "") + '">' +
	        '<label for="' + esc(id) + '">' + esc(label) + "</label>" + control +
	        (help ? '<div class="setting-help">' + esc(help) + "</div>" : "") +
	        "</div>";
	    }
	    function fusionSettingsGroup(title, help, body) {
	      return '<section class="fusion-setting-group"><h3>' + esc(title) + "</h3>" +
	        (help ? '<div class="group-help">' + esc(help) + "</div>" : "") +
	        body + "</section>";
	    }
	    // Fusion run-detail/panel helpers moved to the closed-pack module (pro-views.js).
	    function renderFusion() { proViews.renderFusion?.(proApi); }
	    // License view: the full activate/refresh/deactivate page lives in the closed-pack module
	    // (pro-views.js) when the pack is present; core-only shows the in-app unlock form below.
	    function renderLicense() {
	      if (proViews.renderLicense) { proViews.renderLicense(proApi); return; }
	      renderCoreLicenseInstall();
	    }
	    // Core-only License view: an in-app Pro unlock. Enter the key → the daemon installs the Pro
	    // pack (POST /admin/license/install spawns a detached helper) and restarts → this page reloads
	    // into the full License page. Lives in the open core (not pro-views.js) so it runs without the
	    // pack. Idempotent: it never wipes an unlock already in progress.
	    function renderCoreLicenseInstall() {
	      const view = $("view-license");
	      if (!view || $("unlockProBtn")) return;
	      view.innerHTML =
	        '<div class="card"><div class="card-head"><div>' +
	          "<h2>Unlock Ficelle Pro</h2>" +
	          '<div class="ch-sub">Enter your license key — Ficelle installs and activates Pro on this machine, no terminal needed.</div>' +
	        "</div></div><div class=\"card-body\">" +
	          '<div class="page-actions">' +
	            '<input type="text" class="key-input" id="proKeyInput" placeholder="Enter your license key" autocomplete="off" spellcheck="false" />' +
	            '<button class="btn primary" type="button" id="unlockProBtn">Unlock Pro</button>' +
	          "</div>" +
	          '<div class="ch-sub" id="proInstallProgress" style="margin-top:12px"></div>' +
	          '<p class="ch-sub" style="margin-top:14px">No key yet? <a href="https://ficelle.ai" target="_blank" rel="noopener">Get Ficelle Pro</a>.</p>' +
	        "</div></div>";
	      $("unlockProBtn").addEventListener("click", unlockPro);
	      if (deepLinkedKey) $("proKeyInput").value = deepLinkedKey;
	    }
	    let proInstalledForActivation = false;
	    async function unlockPro() {
	      const key = ($("proKeyInput") ? $("proKeyInput").value : "").trim();
	      const progress = $("proInstallProgress");
	      const btn = $("unlockProBtn");
	      const resetUnlock = function (message, label) {
	        if (progress) progress.textContent = message;
	        if (btn) { btn.disabled = false; btn.textContent = label || "Unlock Pro"; }
	      };
	      if (!key) { if (progress) progress.textContent = "Enter your license key."; return; }
	      if (!proInstalledForActivation) {
	        if (btn) { btn.disabled = true; btn.textContent = "Installing…"; }
	        if (progress) progress.textContent = "Starting the Ficelle Pro install…";
	        try {
	          const r = await fetch("/admin/license/install", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ license_key: key }) });
	          if (!r.ok) {
	            const p = await r.json().catch(() => ({}));
	            resetUnlock((p && p.error && p.error.message) || "The install could not start.");
	            return;
	          }
	        } catch (e) {
	          resetUnlock("The install could not start.");
	          return;
	        }
	        if (progress) progress.textContent = "Installing Ficelle Pro and restarting the service — this can take a minute…";
	        const install = await pollUntilProReady(150000);
	        if (install.error) {
	          resetUnlock(install.error);
	          return;
	        }
	        if (!install.ready) {
	          resetUnlock("Taking longer than expected. Reload the page, or press Unlock to retry.");
	          return;
	        }
	        proInstalledForActivation = true;
	      }
	      // The pack is loaded; activate with the key we still hold, then reload into the full page.
	      if (btn) { btn.disabled = true; btn.textContent = "Activating…"; }
	      if (progress) progress.textContent = "Activating your Ficelle Pro license…";
	      try {
	        const activation = await fetch("/admin/license/activate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ license_key: key }) });
	        if (!activation.ok) {
	          const payload = await activation.json().catch(() => ({}));
	          resetUnlock((payload && payload.error && payload.error.message) || "Ficelle Pro installed, but license activation failed. Try again.", "Retry activation");
	          return;
	        }
	      } catch (e) {
	        resetUnlock("Ficelle Pro installed, but license activation failed. Check the service and try again.", "Retry activation");
	        return;
	      }
	      if (progress) progress.textContent = "Ficelle Pro is ready. Reloading…";
	      setTimeout(function () { window.location.reload(); }, 900);
	    }
	    async function pollUntilProReady(timeoutMs) {
	      const deadline = Date.now() + timeoutMs;
	      while (Date.now() < deadline) {
	        await sleep(2500);
	        try {
	          const installResponse = await fetch("/admin/license/install", { cache: "no-store" });
	          if (installResponse.ok) {
	            const install = await installResponse.json();
	            if (install && install.status === "failed") {
	              return { ready: false, error: install.message || "The Ficelle Pro install failed." };
	            }
	          }
	          // /admin/state is always 200 (no 404 console noise while polling) and reports
	          // pro_installed, which flips true once the restarted daemon has loaded the pack.
	          const r = await fetch("/admin/state", { cache: "no-store" });
	          if (r.ok) { const s = await r.json(); if (s && s.pro_installed) return { ready: true }; }
	        } catch (e) { /* daemon restarting — keep polling */ }
	      }
	      return { ready: false };
	    }
	    function deepLinkKey() {
	      try {
	        const hash = location.hash || "";
	        const qi = hash.indexOf("?");
	        if (qi >= 0) {
	          const fromHash = new URLSearchParams(hash.slice(qi + 1)).get("key");
	          if (fromHash) return fromHash.trim();
	        }
	      } catch (e) { /* malformed URL — ignore */ }
	      return "";
	    }

	    function settingsNumber(id, field, value, min, max, step = "1") {
	      return '<input id="' + esc(id) + '" type="number" min="' + min + '" max="' + max + '" step="' + esc(step) + '" data-settings-field="' + esc(field) + '" value="' + esc(value) + '">';
	    }
	    function settingsToggle(id, field, label, checked, help) {
	      return '<label class="switch fusion-toggle">' +
	        '<input id="' + esc(id) + '" type="checkbox" data-settings-field="' + esc(field) + '"' + (checked ? " checked" : "") + ">" +
	        '<span class="track"></span><span class="switch-copy"><span class="switch-title">' + esc(label) + "</span>" +
	        (help ? '<span class="setting-help">' + esc(help) + "</span>" : "") +
	        "</span></label>";
	    }
	    function settingsCooldownField(id, reason, label, value, help) {
	      return fusionSetting(id, label, settingsNumber(id, "cooldown_seconds." + reason, value, 5, 604800), help);
	    }
	    // Single source of truth for the cooldown rows: [inputId, reason, label, help].
	    // Drives both the rendered fields and readSettingsDraft so the list is edited in one place.
	    const SETTINGS_COOLDOWN_FIELDS = [
	      ["setCdRateLimited", "rate_limited", "Rate limited (HTTP 429)", "Bench time after the upstream returns an account-wide rate-limit. Default 900s (15 min). Provider-scoped: every model of that provider waits."],
	      ["setCdRateLimitedUpstream", "rate_limited_upstream", "Model pool saturated (HTTP 429 upstream)", "Bench time after a 429 that names the model's shared upstream pool rather than your account. Short by design (default 300s) and model-scoped — the provider's other models keep routing."],
	      ["setCdReqTooLarge", "request_too_large", "Request too large (HTTP 413 TPM)", "Bench time after the upstream rejects a request as too large for its tokens-per-minute budget. Short by design (default 120s) — a transient, model-scoped throughput limit that recharges on its own; not a billing block."],
	      ["setCdQuota", "quota_exhausted", "Free quota exhausted", "Bench time after a provider reports its free quota is used up. Default 3600s (1 hour)."],
	      ["setCdAuth", "auth_or_credit", "Auth or credit error", "Bench time after an authentication or credit failure. Default 3600s (1 hour). Usually a key needs fixing."],
	      ["setCdBilling", "billing_or_paid", "Flagged as paid/billed", "Bench time after the anti-false-free guard flags a model as billed. Long by design (default 86400s = 24h) so paid models stop being proposed."],
	      ["setCdBenchmark", "benchmark_failed", "Benchmark failed", "Bench time after a model fails its capability benchmark. Default 3600s (1 hour)."],
	      ["setCdServer", "server_error", "Upstream server error (5xx)", "Bench time after an upstream 5xx. Short by design (default 180s) since these are often transient."],
	      ["setCdTimeout", "timeout", "Timeout", "Bench time after a model times out. Default 300s (5 min)."],
	      ["setCdUnavailable", "unavailable", "Unavailable / unusable reply", "Bench time after a connection error or an unusable success (empty or invalid reply). Default 600s (10 min). Also the fallback when a failure has no specific value."],
	    ];
	    const SETTINGS_COMPRESSION_STRATEGIES = [
	      ["setCompressionJsonArray", "json_array", "JSON arrays", "Summarize repeated JSON array rows while preserving counts and anomalies."],
	      ["setCompressionLog", "log", "Logs", "Keep failures, warnings, stack traces, and summary lines while omitting repeated progress noise."],
	      ["setCompressionDiff", "diff", "Diffs", "Keep file/hunk structure and high-signal additions or removals."],
	      ["setCompressionSearch", "search_results", "Search results", "Group path:line matches and keep representative/high-signal matches."],
	      ["setCompressionTextExcerpt", "text_excerpt", "Text excerpts", "Disabled by default in V1 until deterministic extraction is proven."],
	      ["setCompressionCode", "code", "Code", "Disabled by default in V1 because code compression can break semantics."],
	    ];
	    function settingsCompressionMode(value) {
	      const options = [["off", "Off"], ["dry_run", "Dry run"], ["live_zone", "Live zone"]];
	      return '<select id="setCompressionMode" data-settings-field="compression.mode">' + options.map(([v, label]) => '<option value="' + esc(v) + '"' + (value === v ? " selected" : "") + ">" + esc(label) + "</option>").join("") + "</select>";
	    }
	    function renderSettings() {
	      const s = draftSettings || {};
	      const cd = s.cooldown_seconds || {};
	      const retryControls =
	        '<div class="settings-grid">' +
	          fusionSetting("setMaxAttempts", "Max attempts per request",
	            settingsNumber("setMaxAttempts", "max_attempts_per_request", s.max_attempts_per_request, 0, 12),
	            "How many upstream models Ficelle tries for one request to a virtual model. If the first fails, it moves to the next eligible model in the pool, up to this many. 0 means try every eligible candidate. The effective number is capped by how many models the pool has. Direct (non-virtual) model calls ignore this and never fall back. Default 4.") +
	        "</div>";
	      const cooldownControls =
	        '<div class="settings-grid">' +
	          SETTINGS_COOLDOWN_FIELDS.map(([id, reason, label, help]) => settingsCooldownField(id, reason, label, cd[reason], help)).join("") +
	        "</div>";
	      const backoffValue = Array.isArray(s.quota_probe_backoff_seconds) ? s.quota_probe_backoff_seconds.join(", ") : "";
	      const timeoutControls =
	        '<div class="settings-grid">' +
	          fusionSetting("setCatalogTimeout", "Catalog fetch timeout",
	            settingsNumber("setCatalogTimeout", "catalog_timeout_seconds", s.catalog_timeout_seconds, 5, 120),
	            "Time limit for fetching a provider's model catalog during a refresh. Default 30s.") +
	          fusionSetting("setProbeTimeout", "Quota probe timeout",
	            settingsNumber("setProbeTimeout", "quota_probe_timeout_seconds", s.quota_probe_timeout_seconds, 1, 60),
	            "Time limit for the tiny test request Ficelle sends to check whether a free quota is back. Default 10s.") +
	          fusionSetting("setBackoff", "Quota re-check backoff (seconds)",
	            '<input id="setBackoff" type="text" inputmode="numeric" data-settings-field="quota_probe_backoff_seconds" value="' + esc(backoffValue) + '">',
	            "Escalating wait between quota re-checks after repeated failures, as a comma-separated list of seconds. Ficelle waits the 1st value after the first failed probe, the 2nd after the next, then holds at the last. Default 3600, 21600, 86400 (1h, 6h, 24h).", true) +
	        "</div>";
	      const verificationControls =
	        '<div class="settings-grid">' +
	          fusionSetting("setVerifiedTtl", "Capability re-check interval (seconds)",
	            settingsNumber("setVerifiedTtl", "verified_capability_ttl_seconds", s.verified_capability_ttl_seconds, 86400, 2592000),
	            "How long a model's verified or failed capability verdict is trusted before Ficelle re-benchmarks it for a specialized virtual model (auto-tools, auto-json, auto-reasoning, vision/video/audio). Minimum 86400s (24h) so benchmarks don't run too often; default 604800s (7 days); maximum 2592000s (30 days).") +
	        "</div>";
	      const benchmarkBudgetControls =
	        '<div class="settings-grid">' +
	          fusionSetting("setLongCtxTokens", "Long-context probe size (tokens)",
	            settingsNumber("setLongCtxTokens", "benchmark_long_context_tokens", s.benchmark_long_context_tokens, 1000, 256000),
	            "Token budget of the long-context needle-in-haystack probe. Bigger probes test deeper recall but cost more free-quota tokens per benchmarked model. Default 64000.") +
	          fusionSetting("setCompressionChars", "Compression probe size (characters)",
	            settingsNumber("setCompressionChars", "benchmark_compression_chars", s.benchmark_compression_chars, 2000, 200000),
	            "Character volume of the compression probe. Larger volumes stress summarization harder but cost more free-quota tokens per benchmarked model. Default 18000.") +
	        "</div>";
	      const referenceRoutingControls =
	        '<div class="settings-grid">' +
	          settingsToggle("setRouteOnRef", "route_on_capability_reference", "Route on capability reference (Phase B)",
	            !!s.route_on_capability_reference,
	            "When on, a model whose specialized capability is confirmed by the high-confidence curated capability reference can route to that virtual model (auto-tools, auto-json, auto-reasoning, vision/video/audio) before it has been benchmarked. A failed benchmark always overrides, and verdicts are still re-checked on the interval above. Off by default — turn it on only once the capability-discrepancy log shows the reference agreeing with your providers' benchmarks.") +
	        "</div>";
	      const autoBenchmarkControls =
	        '<div class="settings-grid">' +
	          settingsToggle("setAutoBench", "auto_benchmark_enabled", "Discover capabilities automatically",
	            s.auto_benchmark_enabled !== false,
	            "When on, Ficelle probes new/unproven models in the background and maps each to every virtual model it actually supports — so you don't click Canary. Each cycle takes a few not-yet-fully-discovered models and tests them across all capabilities (tools, json, reasoning, vision...). Failing one capability only marks that one as failed; it never benches the model elsewhere. Verdicts re-test themselves after the re-check interval above.") +
	          fusionSetting("setAutoBenchInterval", "Discovery cycle interval (seconds)",
	            settingsNumber("setAutoBenchInterval", "auto_benchmark_interval_seconds", s.auto_benchmark_interval_seconds, 3600, 604800),
	            "How often the background loop runs a discovery cycle. Minimum 3600s (1h); default 10800s (3h); maximum 604800s (7 days).") +
	          fusionSetting("setAutoBenchModels", "Models probed per cycle",
	            settingsNumber("setAutoBenchModels", "auto_benchmark_models_per_cycle", s.auto_benchmark_models_per_cycle, 1, 50),
	            "How many not-yet-fully-discovered models each cycle probes (across all their capabilities). Higher covers a new provider's catalog faster but uses more free-quota per cycle. Default 5.") +
	        "</div>";
	      $("settingsControls").innerHTML =
	        '<div class="fusion-setting-groups">' +
	          fusionSettingsGroup("Retry budget", "How hard Ficelle tries within a virtual model's pool before returning an error to the client.", retryControls) +
	          fusionSettingsGroup("Cooldown durations", "When an upstream fails, Ficelle benches it so the next request skips it. Each row is the bench time (in seconds) for one kind of failure. Higher means more cautious; lower retries a flaky model sooner.", cooldownControls) +
	          fusionSettingsGroup("Timeouts and quota probes", "Network time limits and how often Ficelle re-checks a provider whose free quota ran out.", timeoutControls) +
	          fusionSettingsGroup("Capability re-verification", "How long a benchmark verdict (verified or failed) is trusted before a model is re-tested for a specialized virtual model. Longer means fewer benchmarks but slower recovery when a provider changes a model.", verificationControls) +
	          fusionSettingsGroup("Benchmark probe budgets", "Input size of the heavy capability probes. Bigger long-context/compression probes test deeper but cost more free-quota tokens per benchmarked model. Lower them if quotas are tight.", benchmarkBudgetControls) +
	          fusionSettingsGroup("Capability reference routing", "Opt-in (Phase B): trust the curated capability prior for routing before a model is benchmarked. Safe defaults keep this off; a failed benchmark always overrides it.", referenceRoutingControls) +
	          fusionSettingsGroup("Automatic capability discovery", "Ficelle probes new models in the background and maps each to every virtual model it supports, so you don't click Canary. On by default; bounded per cycle and quota-aware.", autoBenchmarkControls) +
	        "</div>";
	      $("settingsControls").querySelectorAll("[data-settings-field]").forEach((el) => el.addEventListener("change", readSettingsDraft));
	    }
	    function readSettingsDraft() {
	      // A blanked/invalid field falls back to the value currently loaded in the draft, so
	      // clearing a box and saving keeps the prior value instead of silently resetting it.
	      const prev = draftSettings || {};
	      const num = (id, fallback) => { const v = parseInt($(id)?.value, 10); return Number.isFinite(v) ? v : fallback; };
	      const decimal = (id, fallback) => {
	        const raw = $(id)?.value;
	        if (raw == null || raw.trim() === "") return fallback;
	        const v = Number(raw);
	        return Number.isFinite(v) ? v : fallback;
	      };
	      const checked = (id, fallback) => $(id) ? !!$(id).checked : !!fallback;
	      const cd = prev.cooldown_seconds || {};
	      const compression = prev.compression || {};
	      const compressionStrategies = compression.strategies || {};
	      // Number() (not parseInt) so a typo like "21abc" is dropped rather than coerced to 21.
	      const backoff = ($("setBackoff")?.value || "").split(",").map((x) => Number(x.trim())).filter((n) => Number.isFinite(n) && n > 0);
	      setDraftSettings({
	        max_attempts_per_request: num("setMaxAttempts", prev.max_attempts_per_request),
	        catalog_timeout_seconds: num("setCatalogTimeout", prev.catalog_timeout_seconds),
	        quota_probe_timeout_seconds: num("setProbeTimeout", prev.quota_probe_timeout_seconds),
	        verified_capability_ttl_seconds: num("setVerifiedTtl", prev.verified_capability_ttl_seconds),
	        benchmark_long_context_tokens: num("setLongCtxTokens", prev.benchmark_long_context_tokens),
	        benchmark_compression_chars: num("setCompressionChars", prev.benchmark_compression_chars),
	        route_on_capability_reference: $("setRouteOnRef") ? $("setRouteOnRef").checked : prev.route_on_capability_reference,
	        auto_benchmark_enabled: $("setAutoBench") ? $("setAutoBench").checked : prev.auto_benchmark_enabled,
	        auto_benchmark_interval_seconds: num("setAutoBenchInterval", prev.auto_benchmark_interval_seconds),
	        auto_benchmark_models_per_cycle: num("setAutoBenchModels", prev.auto_benchmark_models_per_cycle),
	        cooldown_seconds: Object.fromEntries(SETTINGS_COOLDOWN_FIELDS.map(([id, reason]) => [reason, num(id, cd[reason])])),
	        quota_probe_backoff_seconds: backoff.length ? backoff : (prev.quota_probe_backoff_seconds || []),
	        compression: {
	          mode: $("setCompressionMode")?.value || compression.mode || "off",
	          min_chars: num("setCompressionMinChars", compression.min_chars),
	          min_savings_ratio: decimal("setCompressionSavings", compression.min_savings_ratio),
	          max_compressed_blocks: num("setCompressionMaxBlocks", compression.max_compressed_blocks),
	          store_ttl_seconds: num("setCompressionTtl", compression.store_ttl_seconds),
	          store_max_entries: num("setCompressionMaxEntries", compression.store_max_entries),
	          allow_streaming: checked("setCompressionStreaming", compression.allow_streaming),
	          strategies: Object.fromEntries(SETTINGS_COMPRESSION_STRATEGIES.map(([id, key]) => [key, checked(id, compressionStrategies[key])])),
	        },
	      });
	      return draftSettings;
	    }

	    // Compression views (renderCompression + its dashboard/retrieval/metric helpers) live in the
	    // closed-pack module (pro-views.js); only renderCompression is called from the core render() cycle.
	    // currentCompressionObservability stays here because the Health view also reads it.
	    function renderCompression() { proViews.renderCompression?.(proApi); }
	    function currentCompressionObservability() {
	      const obs = compressionPayload?.observability;
	      return obs && Object.keys(obs).length ? obs : (state?.compression || {});
	    }

	    function renderHealth() {
      const rt = state.state || {}, canary = rt.canary || {}, sum = canary.summary || {}, ab = rt.auto_benchmark || {};
      const compression = currentCompressionObservability();
      const cdM = activeCount(rt.cooldowns);
      const cdP = activeCount(rt.provider_cooldowns);
      const cdQ = activeCount(rt.quota_cooldowns);
      const q = Object.keys(rt.quarantine || {}).length;
      const healthy = canary.status === "pass" && !cdM && !cdP && !cdQ && !q;
      const compressionMode = compression.configured_mode || "off";
      const compressionStatus = compression.last_status || (compressionMode === "off" ? "off" : "waiting");
      const compressionSaved = Number(compression.estimated_saved_tokens || 0);
      const compressionEntries = Number(compression.store?.entry_count || 0);
      const cards = [
        { k: "Canary check", ic: ICONS.canary, v: (canary.status === "pass" ? "Passing" : (canary.status || "Not run")), note: (sum.passed ?? 0) + " of " + (sum.total ?? 0) + " virtual models ok &middot; last run " + timeAgo(canary.last_run_at).label, cls: canary.status === "pass" ? "ok" : "warn" },
        { k: "Paused", ic: ICONS.paused, v: (cdM + cdP + cdQ), note: cdM + " models &middot; " + cdP + " providers &middot; " + cdQ + " free quotas", cls: (cdM + cdP + cdQ) ? "warn" : "ok" },
        { k: "Disabled", ic: ICONS.disable, v: q, note: q ? "manually held out of routing" : "nothing quarantined", cls: q ? "warn" : "ok" },
        // blocked means a probe cooled/stopped its model; skipped also counts still-due probes held
        // behind a freshly observed provider/quota cooldown. Both drive the warning colour: a cycle
        // reporting only "0 verified, 0 failed" used to look green while routing was paused.
        { k: "Capability discovery", ic: ICONS.benchmark, v: (ab.last_run_at ? "Last run " + timeAgo(ab.last_run_at).label : "Not run yet"), note: ab.last_run_at ? ((ab.models ?? 0) + " models &middot; " + (ab.verified ?? 0) + " verified &middot; " + (ab.failed ?? 0) + " failed &middot; " + (ab.skipped ?? 0) + " paused &middot; " + (ab.blocked ?? 0) + " blocked") : "background discovery &middot; configure in Settings", cls: ((ab.skipped ?? 0) + (ab.blocked ?? 0)) ? "warn" : "ok" },
        { k: "Context compression", ic: ICONS.compression, v: compressionStatus, note: "mode " + compressionMode + " &middot; ~" + compressionSaved + " tokens saved &middot; " + compressionEntries + " CCR entries <button class=\"btn ghost sm\" data-open-compression>Dashboard</button>", cls: compression.health_digest === "compression_errors_repeated" ? "warn" : "ok" },
        { k: "Watcher", ic: ICONS.watcher, v: "Silent", note: "alerts only when status is not ok &middot; catalog refresh hourly, background auto-benchmark (see Settings)", cls: "ok" }
      ];
      $("healthGrid").innerHTML = cards.map(hcardHTML).join("");
      $("healthGrid").querySelectorAll("[data-open-compression]").forEach((b) => b.addEventListener("click", () => setView("compression")));

      const guards = [];
      Object.entries(rt.cooldowns || {}).filter(([, r]) => r?.active).forEach(([key, r]) => guards.push({ title: key.split("::")[1] || key, meta: "Paused " + formatSeconds(r.seconds_remaining) + " - " + reasonLabel(r.reason), detail: reasonDescription(r.reason), kind: "warn", model: key }));
      Object.entries(rt.provider_cooldowns || {}).filter(([, r]) => r?.active).forEach(([key, r]) => guards.push({ title: providerLabel(key) + " (provider)", meta: "Paused " + formatSeconds(r.seconds_remaining) + " - " + reasonLabel(r.reason), detail: reasonDescription(r.reason), kind: "warn", provider: key }));
      Object.entries(rt.quota_cooldowns || {}).filter(([, r]) => r?.active).forEach(([key, r]) => guards.push({ title: providerLabel(r.source || key) + " (free quota)", meta: "Paused " + formatSeconds(r.seconds_remaining) + " - next probe " + timeAgo(r.next_probe_at_iso || r.next_probe_at).label, kind: "warn", quotaProvider: r.source }));
      Object.entries(rt.quarantine || {}).forEach(([key, r]) => guards.push({ title: r.upstream_id || key.split("::")[1] || key, meta: (r.note || r.reason || "disabled") + " - " + timeAgo(r.set_at).label, kind: "danger" }));
      Object.entries(rt.provider_disabled || {}).forEach(([key, r]) => guards.push({ title: providerLabel(key) + " (provider)", meta: "Manually disabled - " + timeAgo(r.set_at).label, detail: "Held out of routing by an operator until re-enabled.", kind: "danger", enableProvider: key }));
      Object.entries(state?.profiles || {}).forEach(([profileId, profile]) => {
        const label = profileLabels[profileId]?.[0] || profileId;
        (profile?.failed_profile_candidates || []).forEach((row) => {
          const reason = row.reason || "failed_capability_check";
          guards.push({
            title: row.upstream_id || row.model_id || "unknown model",
            meta: label + " - " + reasonLabel(reason),
            detail: (row.message || reasonDescription(reason)) + " Retest candidates to refresh this status.",
            kind: "warn",
            badge: "Guarded",
          });
        });
      });
      $("guardList").innerHTML = guards.length ? guards.map((g) =>
        '<div class="guard-row"><div><div class="g-title">' + esc(g.title) + ' <span class="badge ' + g.kind + '">' + esc(g.badge || (g.kind === "danger" ? "Disabled" : "Paused")) + '</span></div><div class="g-meta" title="' + esc(g.detail || g.meta) + '">' + esc(g.meta) + "</div></div>" +
        (g.provider ? '<button class="btn ghost sm" data-clear-provider="' + esc(g.provider) + '">Resume now</button>' : "") +
        (g.quotaProvider ? '<button class="btn ghost sm" data-probe-provider="' + esc(g.quotaProvider) + '">Probe quota</button>' : "") +
        (g.enableProvider ? '<button class="btn ghost sm" data-enable-provider="' + esc(g.enableProvider) + '">Enable provider</button>' : "") + "</div>"
      ).join("") : '<div class="empty">Nothing held back. Every usable model is in rotation.</div>';
      $("guardList").querySelectorAll("[data-clear-provider]").forEach((b) => b.addEventListener("click", () => clearProviderCooldown(b.dataset.clearProvider)));
      $("guardList").querySelectorAll("[data-probe-provider]").forEach((b) => b.addEventListener("click", () => probeProviderQuota(b.dataset.probeProvider, b)));
      $("guardList").querySelectorAll("[data-enable-provider]").forEach((b) => b.addEventListener("click", () => toggleProvider(b.dataset.enableProvider, true, b)));
    }

    function statusPill(s) { const v = s || "unknown"; const cls = ["ok", "pass", "verified"].includes(v) ? "ok" : (["fail", "failed"].includes(v) ? "danger" : ""); return '<span class="badge ' + cls + '">' + esc(v) + "</span>"; }
    function compressionPerfMeta(row) {
      if (!row.compression_status) return "Compression: none";
      const strategies = Object.entries(row.compression_strategy_mix || {}).filter(([, v]) => Number(v) > 0).map(([k, v]) => k + " x" + v).join(", ") || "no strategy";
      return "Compression: " + statusPill(row.compression_status) + " " + esc(row.compression_mode || "unknown") + " &middot; ~" + Number(row.compression_estimated_saved_tokens || 0) + " tokens saved &middot; " + esc(strategies);
    }
    function renderPerformance() {
      const hist = state.performance_history || {};
      const rows = profileOrder.map((p) => hist[p]).filter(Boolean);
      if (!rows.length) { $("perfList").innerHTML = '<div class="empty">No traffic yet. Route a request or run a benchmark to see history here.</div>'; return; }
      $("perfList").innerHTML = rows.map((row) => {
        const lab = profileLabels[row.profile_id] || [row.profile_id, ""];
        const routeMeta = row.last_route_seen_at ? esc(row.last_route_upstream || row.last_route_reason || "unknown") + " &middot; " + Number(row.last_route_latency_seconds || 0).toFixed(2) + "s &middot; " + timeAgo(row.last_route_seen_at).label : "no request routed yet";
        const fb = (row.fallback_reasons || []).length ? row.fallback_reasons.map((x) => esc(x.upstream || x.model || "unknown") + ": " + esc(x.reason || x.status || "failed")).join(", ") : "none";
        const models = (row.models || []).map((m) => {
          const lat = m.latency_ewma != null ? Number(m.latency_ewma).toFixed(2) + "s avg" : "no latency";
          const last = m.last_success_at || m.last_failure_at;
          return '<div class="perf-model"><span class="badge accent">#' + esc(m.rank || "?") + '</span><span class="badge">' + esc(m.origin || "auto") + '</span><span class="pm-name">' + esc(m.upstream_id || m.model_id || "unknown") + '</span><span class="badge">' + pct(m.successes, m.requests) + "% ok</span><span class=\"badge\">" + esc(lat) + "</span>" + statusPill(m.benchmark_status) + statusPill(m.verified_status) + "<span style=\"color:var(--text-muted)\">" + (last ? timeChip(last) : "never") + (m.last_failure_reason ? " &middot; " + esc(m.last_failure_reason) : "") + "</span></div>";
        }).join("") || '<div class="perf-model"><span>No eligible candidates.</span></div>';
        return '<article class="card perf-row"><div><div class="perf-title">' + esc(lab[0]) + '</div><div class="perf-id">' + esc(row.profile_id) + '</div><div class="perf-meta">Winner: <b style="color:var(--text)">' + esc(row.selected_upstream || "none") + "</b> &middot; " + Number(row.candidate_count || 0) + " candidates<br>Last request: " + statusPill(row.last_route_status || "never") + " " + routeMeta + "<br>" + compressionPerfMeta(row) + "<br>Fallbacks: " + fb + "</div></div><div class=\"perf-models\">" + models + "</div></article>";
      }).join("");
    }

	    function auditIcon(action) {
	      if ((action || "").includes("fusion")) return ICONS.fusion;
	      if ((action || "").includes("compression")) return ICONS.compression;
	      if ((action || "").includes("profiles")) return ICONS.profiles;
	      if ((action || "").includes("settings")) return ICONS.settings;
	      if ((action || "").includes("canary")) return ICONS.canary;
	      if ((action || "").includes("benchmark")) return ICONS.benchmark;
      if ((action || "").includes("quarantine")) return ICONS.disable;
      if ((action || "").includes("cooldown")) return ICONS.cooldownClock;
      if ((action || "").includes("refresh")) return ICONS.refresh;
      return ICONS.auditDefault;
	    }
	    function auditTitle(action) {
	      return ({ "admin.profiles.save": "Saved virtual models", "admin.profiles.prune": "Removed models their provider dropped","admin.fusion.save": "Saved fusion settings", "admin.settings.save": "Saved routing settings", "admin.compression.clear": "Cleared compression store", "admin.canary": "Ran a canary health check", "admin.benchmark": "Ran a benchmark", "admin.quarantine.add": "Disabled a model", "admin.quarantine.remove": "Re-enabled a model", "admin.cooldowns.clear_model": "Resumed a model", "admin.cooldowns.clear_provider": "Resumed a provider", "admin.providers.probe": "Probed provider free quota", "admin.refresh": "Refreshed the model catalog" })[action] || action || "Admin action";
	    }
	    function auditMeta(e) {
	      const m = e.metadata || {}, parts = [];
	      if (m.profile_count !== undefined) parts.push(m.profile_count + " virtual models");
	      if (m.max_attempts_per_request !== undefined) parts.push(m.max_attempts_per_request + " max attempts");
	      if (m.enabled !== undefined) parts.push(m.enabled ? "enabled" : "disabled");
	      if (m.visible_in_models !== undefined) parts.push(m.visible_in_models ? "visible in models" : "hidden from models");
	      if (m.model_id) parts.push(m.model_id);
	      if (m.source) parts.push(providerLabel(m.source));
	      if (m.scope) parts.push("scope " + m.scope);
	      if (m.resource) parts.push(m.resource);
	      if (m.compression_mode) parts.push("compression " + m.compression_mode);
	      // A prune reports {virtualModelId: [modelIds]}; every other action reports a count.
	      if (m.removed && typeof m.removed === "object") {
	        const dropped = new Set(Object.values(m.removed).flat());
	        const lanes = Object.keys(m.removed).map((x) => profileLabels[x]?.[0] || x);
	        parts.push(esc(dropped.size + " model" + (dropped.size > 1 ? "s" : "") + " dropped from " + lanes.join(", ")));
	      } else if (m.removed !== undefined) parts.push(m.removed + " removed");
	      if ((m.profiles || []).length) parts.push((m.profiles || []).map((x) => (profileLabels[x]?.[0] || x)).join(", "));
	      return parts.join(" &middot; ") || e.id;
	    }
    function renderAudit() {
      const list = $("auditList");
	      if (!auditEntries.length) { list.innerHTML = '<div class="empty">No admin actions recorded yet.</div>'; return; }
	      list.innerHTML = auditEntries.map((e) => {
	        // A prune is the one virtual-model change the user did not ask for, so Undo
	        // matters there at least as much as on a manual save.
	        const profileRoll = (e.action === "admin.profiles.save" || e.action === "admin.profiles.prune") && e.before?.virtual_profiles;
	        const fusionRoll = e.action === "admin.fusion.save" && e.before?.fusion;
		        const settingsRoll = e.action === "admin.settings.save" && e.before?.settings;
	        const t = timeAgo(e.logged_at);
	        const undo = profileRoll ? '<button class="btn ghost sm" data-rollback-profiles="' + esc(e.id) + '">Undo</button>' : (fusionRoll ? '<button class="btn ghost sm" data-rollback-fusion="' + esc(e.id) + '">Undo</button>' : (settingsRoll ? '<button class="btn ghost sm" data-rollback-settings="' + esc(e.id) + '">Undo</button>' : ""));
	        return '<div class="audit-row"><div class="audit-ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + auditIcon(e.action) + '</svg></div>' +
	          '<div><div class="audit-action">' + esc(auditTitle(e.action)) + '</div><div class="audit-meta">' + auditMeta(e) + '</div></div>' +
	          '<div style="display:flex;gap:10px;align-items:center"><span class="audit-time" title="' + esc(t.title) + '">' + esc(t.label) + "</span>" + undo + "</div></div>";
	      }).join("");
	      list.querySelectorAll("[data-rollback-profiles]").forEach((b) => b.addEventListener("click", () => rollbackProfiles(b.dataset.rollbackProfiles)));
	      list.querySelectorAll("[data-rollback-fusion]").forEach((b) => b.addEventListener("click", () => rollbackFusion(b.dataset.rollbackFusion)));
	      list.querySelectorAll("[data-rollback-settings]").forEach((b) => b.addEventListener("click", () => rollbackSettings(b.dataset.rollbackSettings)));
	    }

    /* ---------- requests (derived route-log index) ---------- */
    let requestsTimer = null;          // polling fallback ticker
    let requestsStream = null;         // EventSource live tail
    let requestsFlushTimer = null;
    let requestsPending = [];          // streamed rows waiting for the next paint
    let requestsSummaryAt = 0;
    let requestsCursor = null;         // index position the rendered list was read at
    let requestsDegraded = false;
    const REQ_WINDOWS = ["1h", "24h", "7d", "30d"];
    const REQ_LIST_CAP = 200;          // rows kept in the payload and in the DOM
    const REQ_FLUSH_MS = 400;          // coalesce a burst of streamed rows into one paint
    const REQ_SUMMARY_MS = 15000;      // the aggregates are the heavy read; refresh them sparingly
    const liveLabel = () => ic(ICONS.watcher) + "Live: " + (requestsLive ? "on" : "off");

    function requestQuery() {
      const f = requestFilters;
      const params = new URLSearchParams({ limit: String(REQ_LIST_CAP) });
      if (f.profile) params.set("profile", f.profile);
      if (f.source) params.set("source", f.source);
      if (f.reason) params.set("reason", f.reason);
      if (f.status) params.set("status", f.status);
      if (f.q) params.set("q", f.q);
      return params.toString();
    }
    const summaryUrl = () => "/admin/requests/summary?window=" + encodeURIComponent(requestFilters.window || "24h");
    // Toast only on the transition into a degraded state, so a live tail doesn't spam it every tick.
    function noteRequestsDegraded(errorType) {
      if (errorType && !requestsDegraded) showToast("Request index unavailable: " + errorType);
      requestsDegraded = Boolean(errorType);
    }
    async function loadRequests(after = true) {
      const [lp, sp] = await Promise.all([loadRequestsList(false), fetchRequestsSummary()]);
      setRequestsSummary(sp);
      requestsSummaryAt = Date.now();
      noteRequestsDegraded(lp?.error_type || sp?.error_type);
      if (after) {
        renderRequests();
        // Refresh the filter controls (newly seen providers/outcomes) unless the user is
        // interacting with one (typing in search, or holding a dropdown open) — rebuilding
        // would steal focus or collapse an open <select>.
        if (!$("requestsFilters").contains(document.activeElement)) renderRequestsFilters();
      }
    }
    // List only. A resync takes this path rather than loadRequests(): the aggregates are the
    // expensive read, and they have their own cadence.
    async function loadRequestsList(after = true) {
      const r = await fetch("/admin/requests?" + requestQuery(), { cache: "no-store" });
      const p = await r.json();
      if (!r.ok) throw new Error(p?.error?.message || "requests failed");
      // A full list read supersedes anything the stream queued for the old list.
      requestsPending = [];
      requestsCursor = typeof p?.cursor === "number" ? p.cursor : null;
      setRequestsPayload(p || { entries: [] });
      if (after) { noteRequestsDegraded(p?.error_type); renderRequestsList(); }
      return p;
    }
    async function fetchRequestsSummary() {
      const r = await fetch(summaryUrl(), { cache: "no-store" });
      const p = await r.json();
      if (!r.ok) throw new Error(p?.error?.message || "summary failed");
      return p || {};
    }
    function enterRequests() {
      renderRequestsFilters();
      renderRequests();
      loadRequests().catch((e) => showToast(e.message || String(e)));
      if (requestsLive) startRequestsLive();
    }

    /* Live tail. The server pushes only the rows a subscriber has not seen yet, so the page
     * stays current without re-fetching the whole list; the 5s poll is the fallback for a
     * browser or proxy that cannot hold an event stream. Both stop when the view is left or
     * the tab is hidden — a background tab must not keep a connection or a timer alive. */
    function stopRequestsLive() { closeRequestsStream(); stopRequestsPolling(); }
    function closeRequestsStream() {
      if (requestsStream) { requestsStream.close(); requestsStream = null; }
      if (requestsFlushTimer) { clearTimeout(requestsFlushTimer); requestsFlushTimer = null; }
      requestsPending = [];
    }
    function stopRequestsPolling() { if (requestsTimer) { clearInterval(requestsTimer); requestsTimer = null; } }
    function startRequestsLive() {
      stopRequestsLive();
      if (document.hidden) return;  // resumed by the visibilitychange handler
      if (typeof EventSource !== "function") { startRequestsPolling(); return; }
      // Resume from the snapshot the list was built on, so nothing routed between that read
      // and this connection is lost. EventSource cannot set headers, hence the query param.
      const at = requestsCursor === null ? "" : "&cursor=" + encodeURIComponent(requestsCursor);
      const stream = new EventSource("/admin/requests/stream?" + requestQuery() + at);
      requestsStream = stream;
      stream.addEventListener("requests", (ev) => {
        if (stream !== requestsStream) return;
        let batch;
        try { batch = JSON.parse(ev.data); } catch { return; }
        if (typeof batch?.cursor === "number") requestsCursor = batch.cursor;
        // The server asks for a resync when a resuming tail fell too far behind to replay.
        // Reload the list, not the aggregates: the stream is still live and a resync happens
        // exactly when the router is busiest, which is the worst moment to pay for them.
        if (batch?.resync) { loadRequestsList().catch(() => {}); return; }
        queueStreamedRequests(batch?.entries || []);
      });
      stream.addEventListener("error", () => {
        if (stream !== requestsStream) return;
        // EventSource retries on its own while it is CONNECTING (a server restart, or the
        // stream's own lifetime cap). CLOSED means it gave up — that is the only failure
        // worth downgrading the transport for.
        if (stream.readyState !== EventSource.CLOSED) return;
        closeRequestsStream();
        startRequestsPolling();
        showToast("Live stream unavailable; falling back to polling.");
      });
    }
    function startRequestsPolling() {
      stopRequestsPolling();
      requestsTimer = setInterval(() => {
        if (currentView !== "requests" || !requestsLive) { stopRequestsPolling(); return; }
        if (document.hidden) return;
        loadRequests().catch(() => {});
      }, 5000);
    }
    function toggleRequestsLive() {
      setRequestsLive(!requestsLive);
      const b = $("liveRequestsBtn");
      if (b) { b.innerHTML = liveLabel(); b.classList.toggle("primary", requestsLive); b.setAttribute("aria-pressed", requestsLive ? "true" : "false"); }
      if (requestsLive) startRequestsLive(); else stopRequestsLive();
    }
    function queueStreamedRequests(entries) {
      if (!entries.length) return;
      requestsPending.push(...entries);
      if (!requestsFlushTimer) requestsFlushTimer = setTimeout(flushStreamedRequests, REQ_FLUSH_MS);
    }
    function flushStreamedRequests() {
      requestsFlushTimer = null;
      const pending = requestsPending;
      requestsPending = [];
      const el = $("requestsList");
      if (!pending.length || !el || currentView !== "requests") return;
      const known = new Set((requestsPayload.entries || []).map((e) => e.request_id));
      const fresh = [];
      for (const e of pending) {
        if (!e?.request_id || known.has(e.request_id)) continue;
        known.add(e.request_id);
        fresh.push(e);
      }
      if (!fresh.length) return;
      fresh.reverse();  // the tail arrives oldest first; the list reads newest first
      setRequestsPayload({ ...requestsPayload, entries: fresh.concat(requestsPayload.entries || []).slice(0, REQ_LIST_CAP) });
      if (el.querySelector(".empty")) {
        renderRequestsList();  // the list was empty: swap the placeholder out
      } else {
        // Prepend just the new rows instead of rebuilding: the rows already on screen keep
        // their expanded state, and a long list costs nothing to leave alone.
        el.insertAdjacentHTML("afterbegin", fresh.map((e) => reqRowHTML(e, true)).join(""));
        while (el.children.length > REQ_LIST_CAP) el.lastElementChild.remove();
      }
      if (Date.now() - requestsSummaryAt >= REQ_SUMMARY_MS) {
        requestsSummaryAt = Date.now();  // claim the slot before awaiting so a burst cannot stack fetches
        fetchRequestsSummary().then((sp) => { setRequestsSummary(sp); renderRequestsSummary(); renderRequestsTimeline(); }).catch(() => {});
      }
    }

    function renderRequests() { renderRequestsSummary(); renderRequestsWindow(); renderRequestsTimeline(); renderRequestsList(); }

    function renderRequestsSummary() {
      const el = $("requestsSummary"); if (!el) return;
      const s = requestsSummary || {};
      const total = Number(s.total || 0);
      const rate = s.success_rate == null ? "n/a" : Math.round(Number(s.success_rate) * 100) + "%";
      const p95 = s.latency_p95 == null ? "n/a" : Number(s.latency_p95).toFixed(2) + "s";
      const p50 = s.latency_p50 == null ? "n/a" : Number(s.latency_p50).toFixed(2) + "s";
      const errs = Number(s.errors || 0);
      const fb = Number(s.fallback_count || 0);
      const top = (s.by_reason || []).find((r) => r.reason !== "ok");
      const windowLabel = formatSeconds(s.window_seconds || 86400);
      const rateCls = s.success_rate == null ? "ok" : (Number(s.success_rate) >= 0.95 ? "ok" : (Number(s.success_rate) >= 0.8 ? "warn" : "danger"));
      const cards = [
        { k: "Requests", ic: ICONS.models, v: total, note: "in the last " + windowLabel, cls: "info" },
        { k: "Success rate", ic: ICONS.canary, v: rate, note: Number(s.ok || 0) + " ok &middot; " + errs + " errors", cls: rateCls },
        { k: "Latency p95", ic: ICONS.latency, v: p95, note: "median " + p50, cls: "info" },
        { k: "Fallbacks", ic: ICONS.retest, v: fb, note: fb ? "needed a backup model" : "no fallback needed", cls: fb ? "warn" : "ok" },
        { k: "Top error", ic: ICONS.disable, v: top ? reasonLabel(top.reason) : "none", note: top ? (top.total + " in window") : "no errors in window", cls: top ? "danger" : "ok" },
      ];
      el.innerHTML = cards.map(hcardHTML).join("");
    }

    function renderRequestsWindow() {
      const wEl = $("requestsWindow"); if (!wEl) return;
      wEl.innerHTML = '<div class="segmented" role="group" aria-label="Summary window">' +
        REQ_WINDOWS.map((w) => '<button type="button" data-req-window="' + w + '"' + (requestFilters.window === w ? ' class="active"' : "") + ">" + w + "</button>").join("") + "</div>";
      wEl.querySelectorAll("[data-req-window]").forEach((b) => b.addEventListener("click", () => { requestFilters.window = b.dataset.reqWindow; loadRequests().catch((e) => showToast(e.message)); }));
    }

    function renderRequestsTimeline() {
      const el = $("requestsTimeline"); if (!el) return;
      const buckets = (requestsSummary?.timeline) || [];
      // Leave the chart alone while the pointer is on it: the 5s poll would otherwise
      // rebuild the DOM under the cursor and drop the tooltip mid-read. This also comes
      // before the empty state so an expiring last bucket waits for pointer leave.
      if ($("reqChart")?.matches(":hover")) return;
      if (!buckets.some((b) => Number(b.total || 0) > 0)) {
        stopChartResizeWatch();
        el.innerHTML = '<div class="empty">No requests in this window yet.</div>';
        return;
      }
      // Built once and then kept: the host has to outlive the poll for its hover
      // listeners and its resize observer to be bound once rather than every 5s.
      if (!$("reqChart")) {
        el.innerHTML = '<div class="req-chart" id="reqChart"></div>' +
          '<div class="req-spark-legend"><span><i class="dot ok"></i>success</span><span><i class="dot danger"></i>error</span>' +
          '<span class="req-spark-span" id="reqChartSpan"></span></div>';
        startChartResizeWatch();
      }
      $("reqChartSpan").textContent = new Date(Number(buckets[0].bucket) * 1000).toLocaleString() + " → now";
      drawRequestsChart();
    }

    /* ---------- request volume chart ----------
       Drawn in real pixels measured off the card rather than in a stretched viewBox:
       the plot is ~12x wider than tall, so a non-uniform scale would flatten the bars
       into slabs and turn every corner radius into an ellipse. Redrawn on resize. */
    const CHART_PAD = { top: 12, right: 12, bottom: 24, left: 38 };
    const CHART_PLOT_H = 132;
    let chartResizeObserver = null;

    // One reading of a bucket, so the bars and the tooltip can never disagree about it.
    // `Number(x) || 0` rather than `Number(x || 0)`: the latter keeps NaN for a
    // non-numeric field and paints it as "NaN" instead of an empty bucket.
    function bucketCounts(b) {
      const total = Number(b?.total) || 0;
      const ok = Math.max(0, Math.min(total, Number(b?.ok) || 0));
      return { total, ok, err: total - ok };
    }

    // A y-axis top that lands on a whole number, aiming for about three intervals.
    function niceAxis(maxValue) {
      const rough = Math.max(1, maxValue) / 3;
      const mag = Math.pow(10, Math.floor(Math.log10(rough)));
      const unit = rough / mag;
      const step = Math.max(1, Math.round((unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * mag));
      return { step, max: Math.ceil(Math.max(1, maxValue) / step) * step };
    }

    // Shared by the axis ticks and the hover tooltip so both always read the same way.
    const chartClock = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const chartDay = (d) => d.toLocaleDateString([], { day: "2-digit", month: "2-digit" });

    // Bucket width decides how much of the date a tick label needs to stay unambiguous.
    function chartTickFormat(bucketSeconds) {
      if (bucketSeconds >= 86400) return chartDay;
      if (bucketSeconds >= 21600) return (d) => chartDay(d) + " " + chartClock(d);
      return chartClock;
    }

    function drawRequestsChart() {
      const host = $("reqChart"); if (!host) return;
      const buckets = (requestsSummary?.timeline) || [];
      const width = Math.round(host.clientWidth || 0);
      if (!buckets.length || width < 80) return;
      // Same width, same numbers, same picture: skip the poll's redundant repaint, and
      // make the resize observer a no-op unless the width actually moved. Parked on the
      // host so the cache dies with the drawing it describes.
      const key = width + "|" + JSON.stringify(buckets);
      if (host.dataset.key === key) return;
      host.dataset.key = key;

      const P = CHART_PAD, H = CHART_PLOT_H;
      const plotW = width - P.left - P.right;
      const height = H + P.top + P.bottom;
      const n = buckets.length;
      const axis = niceAxis(Math.max(...buckets.map((b) => bucketCounts(b).total)));
      const yOf = (v) => P.top + H - (v / axis.max) * H;
      const slot = plotW / n;
      const bw = Math.max(2, Math.min(slot - 3, 26));
      const bucketSeconds = Number(requestsSummary?.bucket_seconds || 300);
      const fmt = chartTickFormat(bucketSeconds);

      let grid = "";
      for (let v = 0; v <= axis.max; v += axis.step) {
        const y = Math.round(yOf(v)) + 0.5; // half-pixel offset keeps the rule at 1 crisp px
        grid += '<line class="rc-grid' + (v === 0 ? " rc-base" : "") + '" x1="' + P.left + '" x2="' + (width - P.right) + '" y1="' + y + '" y2="' + y + '"/>' +
          '<text class="rc-ylab" x="' + (P.left - 8) + '" y="' + (y + 4) + '">' + v + "</text>";
      }

      const bars = buckets.map((b, i) => {
        const { total, ok: okc, err: errc } = bucketCounts(b);
        if (!total) return "";
        // Floor the drawn height so a lone request in a tall window stays visible, then
        // clamp the error segment so neither side of a mixed bucket vanishes.
        const hTotal = Math.max(okc && errc ? 3 : 1.5, (total / axis.max) * H);
        const hErr = !errc ? 0 : !okc ? hTotal
          : Math.min(Math.max(hTotal - (okc / total) * hTotal, 1.5), hTotal - 1.5);
        const hOk = hTotal - hErr;
        const x = P.left + i * slot + (slot - bw) / 2;
        const top = P.top + H - hTotal;
        return '<g class="rc-bar" data-i="' + i + '">' +
          (hErr ? '<rect class="rc-err" x="' + x.toFixed(1) + '" y="' + top.toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + hErr.toFixed(1) + '" rx="2"/>' : "") +
          // Only the top segment gets rounded corners, so the stack reads as one bar.
          (hOk ? '<rect class="rc-ok" x="' + x.toFixed(1) + '" y="' + (P.top + H - hOk).toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + hOk.toFixed(1) + '" rx="' + (hErr ? 0 : 2) + '"/>' : "") +
          "</g>";
      }).join("");

      const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(plotW / 110))));
      let ticks = "";
      for (let i = 0; i < n; i += every) {
        const cx = Math.min(width - P.right - 18, Math.max(P.left + 18, P.left + i * slot + slot / 2));
        ticks += '<text class="rc-xlab" x="' + cx.toFixed(1) + '" y="' + (P.top + H + 16) + '">' + esc(fmt(new Date(Number(buckets[i].bucket) * 1000))) + "</text>";
      }

      const hits = buckets.map((_, i) =>
        '<rect class="rc-hit" data-i="' + i + '" x="' + (P.left + i * slot).toFixed(1) + '" y="' + P.top + '" width="' + slot.toFixed(1) + '" height="' + H + '"/>'
      ).join("");

      host.innerHTML = '<svg class="rc-svg" width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + " " + height + '" role="img" aria-label="Request volume over time, success and error stacked per time bucket">' +
        grid +
        '<rect class="rc-band" id="rcBand" y="' + P.top + '" height="' + H + '" width="' + slot.toFixed(1) + '" rx="3" visibility="hidden"/>' +
        bars + ticks + hits + "</svg>" +
        '<div class="req-tip" id="reqTip" hidden></div>';
      bindChartHover(host, buckets, { width, bucketSeconds });
    }

    function bindChartHover(host, buckets, geo) {
      const band = $("rcBand"), tip = $("reqTip");
      host.querySelectorAll(".rc-hit").forEach((hit) => {
        hit.addEventListener("mouseenter", () => {
          const i = Number(hit.dataset.i);
          const b = buckets[i]; if (!b || !band || !tip) return;
          const { total, ok: okc, err: errc } = bucketCounts(b);
          const start = new Date(Number(b.bucket) * 1000);
          const end = new Date((Number(b.bucket) + geo.bucketSeconds) * 1000);
          // Take the slot geometry off the hit rect itself rather than recomputing it,
          // so the band and the tooltip cannot drift from the bars they point at.
          const slotX = Number(hit.getAttribute("x")), slotW = Number(hit.getAttribute("width"));
          band.setAttribute("x", slotX.toFixed(1));
          band.setAttribute("visibility", "visible");
          // The hit rects sit above the bars and swallow the pointer, so :hover never
          // reaches the bar itself — carry the highlight explicitly.
          host.querySelector(".rc-bar.is-active")?.classList.remove("is-active");
          host.querySelector('.rc-bar[data-i="' + i + '"]')?.classList.add("is-active");
          tip.innerHTML = '<div class="rt-when">' + esc(chartDay(start)) + " " + esc(chartClock(start)) + " – " + esc(chartClock(end)) + "</div>" +
            (total
              ? '<div class="rt-counts"><span><i class="dot ok"></i>' + okc + " ok</span><span><i class=\"dot danger\"></i>" + errc + " error" + (errc === 1 ? "" : "s") + "</span></div>" +
                '<div class="rt-total">' + total + " request" + (total === 1 ? "" : "s") + "</div>"
              : '<div class="rt-total">no traffic</div>');
          tip.hidden = false;
          // Sit beside the slot, on whichever side has room — never on top of the bar
          // being read — then clamp so the card stays inside the plot.
          const w = tip.offsetWidth;
          const beside = slotX < geo.width / 2 ? slotX + slotW + 8 : slotX - w - 8;
          tip.style.left = Math.max(4, Math.min(geo.width - w - 4, beside)) + "px";
        });
      });
      // The host outlives the poll, so this binds once for its whole lifetime.
      if (host.dataset.hoverBound) return;
      host.dataset.hoverBound = "1";
      host.addEventListener("mouseleave", () => {
        $("rcBand")?.setAttribute("visibility", "hidden");
        const t = $("reqTip"); if (t) t.hidden = true;
        host.querySelector(".rc-bar.is-active")?.classList.remove("is-active");
        renderRequestsTimeline(); // catch up on any poll skipped while hovering
      });
    }

    function startChartResizeWatch() {
      const host = $("reqChart"); if (!host || typeof ResizeObserver === "undefined") return;
      stopChartResizeWatch();
      chartResizeObserver = new ResizeObserver(() => drawRequestsChart());
      chartResizeObserver.observe(host);
    }
    function stopChartResizeWatch() {
      if (chartResizeObserver) { chartResizeObserver.disconnect(); chartResizeObserver = null; }
    }

    function reqOption(value, label, current) {
      return '<option value="' + esc(value) + '"' + (String(current) === String(value) ? " selected" : "") + ">" + esc(label) + "</option>";
    }
    function renderRequestsFilters() {
      const el = $("requestsFilters"); if (!el) return;
      const f = requestFilters;
      const sources = (requestsSummary?.by_source || []).map((r) => r.source).filter(Boolean);
      if (f.source && !sources.includes(f.source)) sources.push(f.source);
      const profileOpts = '<option value="">All virtual models</option>' + [...profileOrder, fusionModelId].map((p) => reqOption(p, (profileLabels[p]?.[0] || p), f.profile)).join("");
      const sourceOpts = '<option value="">All providers</option>' + sources.map((s) => reqOption(s, providerLabel(s), f.source)).join("");
      const reasons = (requestsSummary?.by_reason || []).map((r) => r.reason).filter(Boolean);
      if (f.reason && !reasons.includes(f.reason)) reasons.push(f.reason);
      const reasonOpts = '<option value="">All outcomes</option>' + reasons.map((rk) => reqOption(rk, rk === "ok" ? "Success (ok)" : reasonLabel(rk), f.reason)).join("");
      const statuses = (requestsSummary?.by_status || []).map((r) => r.status).filter((s) => s != null);
      if (f.status && !statuses.map(String).includes(String(f.status))) statuses.push(Number(f.status));
      const statusOpts = '<option value="">Any status</option>' + statuses.map((c) => reqOption(c, String(c), f.status)).join("");
      const hasFilter = f.profile || f.source || f.reason || f.status || f.q;
      el.innerHTML =
        '<label class="field req-search">' + ic(ICONS.search) + '<input id="reqSearch" type="search" placeholder="Search id, model, upstream" value="' + esc(f.q || "") + '"></label>' +
        '<select id="reqProfile" class="req-select" aria-label="Virtual model">' + profileOpts + "</select>" +
        '<select id="reqSource" class="req-select" aria-label="Provider">' + sourceOpts + "</select>" +
        '<select id="reqReason" class="req-select" aria-label="Outcome">' + reasonOpts + "</select>" +
        '<select id="reqStatus" class="req-select" aria-label="HTTP status">' + statusOpts + "</select>" +
        (hasFilter ? '<button class="btn ghost sm" type="button" id="reqReset">Reset</button>' : "");
      // A live tail is filtered server-side, so a filter change has to reopen it.
      const apply = () => loadRequests().then(() => { if (requestsLive) startRequestsLive(); }).catch((e) => showToast(e.message));
      $("reqProfile").addEventListener("change", (e) => { f.profile = e.target.value; apply(); });
      $("reqSource").addEventListener("change", (e) => { f.source = e.target.value; apply(); });
      $("reqReason").addEventListener("change", (e) => { f.reason = e.target.value; apply(); });
      $("reqStatus").addEventListener("change", (e) => { f.status = e.target.value; apply(); });
      let qt;
      $("reqSearch").addEventListener("input", (e) => { f.q = e.target.value.trim(); clearTimeout(qt); qt = setTimeout(apply, 320); });
      const reset = $("reqReset");
      if (reset) reset.addEventListener("click", () => { f.profile = ""; f.source = ""; f.reason = ""; f.status = ""; f.q = ""; renderRequestsFilters(); apply(); });
    }

    function renderRequestsList() {
      const el = $("requestsList"); if (!el) return;
      const rows = requestsPayload?.entries || [];
      if (!rows.length) { el.innerHTML = '<div class="empty">No matching requests yet. Route a request — or relax the filters — to see traffic here.</div>'; return; }
      el.innerHTML = rows.map((e) => reqRowHTML(e)).join("");
      // Delegated, and assigned rather than added: the live tail prepends rows into this same
      // container, and a per-row listener would have to be re-attached on every batch.
      el.onclick = onRequestRowActivate;
      el.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); onRequestRowActivate(ev); } };
    }
    function onRequestRowActivate(ev) {
      const row = ev.target.closest("[data-req-id]");
      if (!row) return;
      const id = row.dataset.reqId;
      if (expandedRequests.has(id)) expandedRequests.delete(id); else expandedRequests.add(id);
      renderRequestsList();
    }
    function reqRowHTML(e, isNew = false) {
      const ok = e.reason === "ok";
      const open = expandedRequests.has(e.request_id);
      const statusBadge = '<span class="badge ' + (ok ? "ok" : "danger") + '">' + esc(String(e.status ?? "—")) + "</span>";
      const reasonText = ok ? "ok" : reasonWithCode(e.reason);
      const errType = (!ok && e.error_type) ? ' <span class="req-errtype">' + esc(e.error_type) + "</span>" : "";
      const lat = e.latency_seconds == null ? "—" : Number(e.latency_seconds).toFixed(2) + "s";
      const t = timeAgo(e.logged_at);
      const fallback = e.fallback ? '<span class="badge warn" title="' + Number(e.attempt_count || 0) + ' attempts">fallback</span>' : "";
      const stream = e.stream ? '<span class="req-stream" title="streamed response">≋</span>' : "";
      const head = '<div class="req-row' + (open ? " is-open" : "") + '" data-req-id="' + esc(e.request_id) + '" role="button" tabindex="0">' +
        '<span class="req-time" title="' + esc(t.title) + '">' + esc(t.label) + "</span>" +
        '<span class="req-profile">' + esc(profileLabels[e.profile]?.[0] || e.profile || "—") + "</span>" +
        '<span class="req-prov">' + esc(e.source ? providerLabel(e.source) : "—") + "</span>" +
        '<span class="req-up mono">' + esc(e.upstream_id || e.model_id || "—") + "</span>" +
        statusBadge +
        '<span class="req-reason">' + esc(reasonText) + errType + "</span>" +
        '<span class="req-lat">' + esc(lat) + "</span>" +
        '<span class="req-flags">' + fallback + stream + "</span>" +
        "</div>";
      return '<div class="req-item' + (isNew ? " is-new" : "") + '">' + head + (open ? reqDetailHTML(e) : "") + "</div>";
    }
    function reqDetailHTML(e) {
      const attempts = e.attempts || [];
      const chain = attempts.length ? attempts.map((a, i) => {
        const aok = a.reason === "ok";
        const acls = aok ? "ok" : (Number(a.status) >= 200 && Number(a.status) < 300 ? "warn" : "danger");
        const al = a.latency_seconds != null ? Number(a.latency_seconds).toFixed(2) + "s" : "";
        return '<div class="req-attempt"><span class="badge">#' + (i + 1) + "</span>" +
          '<span class="mono req-attempt-up">' + esc(a.upstream || a.model || "—") + "</span>" +
          '<span class="req-attempt-src">' + esc(a.source ? providerLabel(a.source) : "—") + "</span>" +
          '<span class="badge ' + acls + '">' + esc(String(a.status ?? a.reason ?? "—")) + "</span>" +
          '<span class="req-attempt-reason">' + esc(reasonWithCode(a.reason)) + (a.error_type ? " · " + esc(a.error_type) : "") + "</span>" +
          (al ? '<span class="req-attempt-lat">' + esc(al) + "</span>" : "") +
          "</div>";
      }).join("") : '<div class="req-attempt"><span class="req-attempt-reason">No per-attempt detail recorded.</span></div>';
      const meta = ["request " + esc(e.request_id)];
      if (e.competence) meta.push("competence " + esc(e.competence));
      if (e.candidate_count != null) meta.push(Number(e.candidate_count) + " candidates");
      if (e.compression_status) meta.push("compression " + esc(e.compression_status));
      return '<div class="req-detail"><div class="req-detail-meta mono">' + meta.join(" · ") + "</div>" +
        '<div class="req-chain">' + chain + "</div></div>";
    }

    function renderNavCounts() {
      const provs = state?.catalog?.providers || {};
      const rt = state?.state || {};
      const cd = activeCount(rt.cooldowns) + activeCount(rt.provider_cooldowns) + activeCount(rt.quota_cooldowns);
      const q = Object.keys(rt.quarantine || {}).length;
      const profileGuardCount = Object.values(state?.profiles || {}).reduce((total, profile) => total + Number((profile?.failed_profile_candidates || []).length), 0);
      const guardCount = cd + q + profileGuardCount;
      const set = (k, v) => { const el = document.querySelector('[data-count="' + k + '"]'); if (el) { el.textContent = v; el.style.display = v ? "" : "none"; } };
	      set("providers", Object.keys(provs).length);
	      set("compression", compressionPayload?.event_count || (compressionPayload?.events || []).length || 0);
	      set("health", guardCount);
	      set("audit", auditEntries.length);
      // rail health
      const canary = rt.canary || {};
      const healthy = canary.status === "pass" && !guardCount;
      $("railDot").className = "health-dot " + (healthy ? "ok" : "watch");
      $("railHealthMain").textContent = healthy ? "Healthy" : "Needs a look";
      $("railHealthSub").textContent = healthy ? "canary green" : guardCount + " guardrails active";
    }

	    function render() {
	      if (!state) return;
	      setProviderLabels(state.provider_labels);
	      renderUpdateBanner();
	      renderProfileList(); renderActiveProfile(); renderFilterBar(); renderSelected(); renderExcluded(); renderAvailable();
	      renderProviders(); renderFusion(); renderSettings(); renderCompression(); renderLicense(); renderHealth(); renderPerformance(); renderAudit(); renderNavCounts();
      syncLongRunningActions();
      syncSaveButton();
	    }

    let updateInstallInFlight = false;
    let updateStatusTimer = null;
    let updateStatusRefreshInFlight = false;
    function renderUpdateBanner() {
      const banner = $("updateBanner");
      if (!banner) return;
      const update = state?.update || {};
      const status = String(update.status || "not_checked");
      const available = Boolean(update.update_available);
      banner.className = "update-banner";
      banner.hidden = true;
      if (available && status === "available") {
        banner.hidden = false;
        const version = esc(update.latest_version || "the latest release");
        const message = update.pro_update_required
          ? "A compatible Pro artifact is required for this installed Pro build."
          : "Install the verified release and restart the local service.";
        banner.innerHTML = '<div class="update-copy"><div class="update-title">Ficelle ' + version + ' is available</div><div class="update-detail">' + esc(message) + (update.release_url ? ' <a href="' + esc(update.release_url) + '" target="_blank" rel="noopener">Release notes</a>.' : "") + '</div></div>' +
          '<div class="update-actions">' + (update.pro_update_required ? '<span class="badge warn">Pro update needed</span>' : '<button class="btn primary sm" type="button" id="installUpdateBtn"' + (updateInstallInFlight ? " disabled" : "") + '>' + (updateInstallInFlight ? "Installing…" : "Install update") + '</button>') + '</div>';
        const button = $("installUpdateBtn");
        if (button) button.addEventListener("click", installUpdate);
        return;
      }
      if (["queued", "installing", "restarting"].includes(status) || updateInstallInFlight) {
        banner.hidden = false;
        banner.classList.add("is-progress");
        const detail = status === "restarting" ? "The service is restarting; this page may briefly lose connection." : "The verified package is being installed in the background.";
        banner.innerHTML = '<div class="update-copy"><div class="update-title">Updating Ficelle…</div><div class="update-detail">' + esc(detail) + '</div></div><div class="update-actions"><span class="badge info">In progress</span></div>';
        return;
      }
      if (status === "failed" && update.message) {
        banner.hidden = false;
        banner.classList.add("is-error");
        banner.innerHTML = '<div class="update-copy"><div class="update-title">Ficelle update failed</div><div class="update-detail">' + esc(update.message) + '</div></div><div class="update-actions"><button class="btn sm" type="button" id="checkUpdateBtn">Check again</button></div>';
        const button = $("checkUpdateBtn");
        if (button) button.addEventListener("click", checkUpdate);
      }
    }

    async function refreshUpdateStatus() {
      if (!state || updateStatusRefreshInFlight) return;
      updateStatusRefreshInFlight = true;
      try {
        const r = await fetch("/admin/update", { cache: "no-store" });
        if (!r.ok) return;
        state.update = await r.json();
        renderUpdateBanner();
      } catch (_) {
        // The update check is optional; a transient local request failure must not affect the UI.
      } finally {
        updateStatusRefreshInFlight = false;
      }
    }

    function startUpdateStatusPolling() {
      void refreshUpdateStatus();
      if (updateStatusTimer) return;
      updateStatusTimer = setInterval(() => { void refreshUpdateStatus(); }, 15000);
    }

    async function checkUpdate() {
      try {
        const r = await fetch("/admin/update", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "check" }) });
        const p = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(p?.error?.message || "Update check failed.");
        state.update = p;
        renderUpdateBanner();
      } catch (e) { showToast(e.message || "Update check failed."); }
    }

    async function installUpdate() {
      if (updateInstallInFlight) return;
      updateInstallInFlight = true;
      renderUpdateBanner();
      try {
        const r = await fetch("/admin/update", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "install" }) });
        const p = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(p?.error?.message || "The update could not start.");
        showToast("Ficelle update started. The service will restart shortly.");
        await pollUntilUpdateReady(180000);
      } catch (e) {
        updateInstallInFlight = false;
        showToast(e.message || "The update could not start.");
        try { await loadState(); } catch (_) { renderUpdateBanner(); }
      }
    }

    async function pollUntilUpdateReady(timeoutMs) {
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        await sleep(2500);
        try {
          const r = await fetch("/admin/update", { cache: "no-store" });
          if (r.ok) {
            const update = await r.json();
            state.update = update;
            if (update.status === "failed") {
              updateInstallInFlight = false;
              renderUpdateBanner();
              showToast(update.message || "The update failed.");
              return;
            }
            if (["installed", "up_to_date"].includes(update.status)) {
              updateInstallInFlight = false;
              await loadState();
              showToast("Ficelle is up to date.");
              return;
            }
            renderUpdateBanner();
          }
        } catch (e) {
          // The daemon is expected to disappear briefly while the helper restarts it.
        }
      }
      updateInstallInFlight = false;
      renderUpdateBanner();
      showToast("The update is taking longer than expected. Check the banner again shortly.");
    }

    /* ---------- view router ---------- */
    const viewMeta = {
	      routing: { eyebrow: "Models local control", title: "Routing", desc: "Map your free provider models onto the virtual models your apps call. The top working model wins each request." },
	      providers: { eyebrow: "Catalog", title: "Providers", desc: "Where Ficelle pulls free, zero-price models from. Connect a key to unlock a provider's catalog." },
		      fusion: { eyebrow: "Compound model", title: "Fusion", desc: "Configure auto-fusion and inspect its metadata-only observability in one place." },
      settings: { eyebrow: "Configure", title: "Settings", desc: "Global routing controls: how many upstream models Ficelle tries per request, how long failures are benched, and network time limits. These apply across every virtual model." },
      compression: { eyebrow: "Configure", title: "Compression", desc: "Tune native request compression and inspect metadata-only savings, CCR storage, retrievals, errors, and recent route events." },
      license: { eyebrow: "Configure", title: "License", desc: "Activate, refresh, or release this machine’s Ficelle Pro license, and check its current status." },
	      health: { eyebrow: "Observe", title: "Health", desc: "Canary status and the guardrails currently holding models or providers out of rotation." },
	      performance: { eyebrow: "Observe", title: "Performance", desc: "What actually happened: winning models, fallbacks, success rates, and latency per virtual model." },
	      requests: { eyebrow: "Observe", title: "Requests", desc: "Every routed request, newest first: virtual model, provider, upstream model, outcome, error type, and latency — with health aggregates over a window." },
	      audit: { eyebrow: "Observe", title: "Audit", desc: "Every local change, newest first. Virtual model saves can be undone." },
      export: { eyebrow: "Ship it", title: "Export", desc: "Generate a generic OpenAI-compatible client configuration. No provider secrets included." }
    };
    function setView(name) {
      name = String(name).split("?")[0];  // tolerate a deep-link query, e.g. #/license?key=…
      if (!viewMeta[name]) name = "routing";
      setCurrentView(name);
      document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === name));
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.nav === name));
      document.body.classList.toggle("routing-fixed", name === "routing");
      const m = viewMeta[name];
      $("viewEyebrow").textContent = m.eyebrow; $("viewTitle").textContent = m.title; $("viewDesc").textContent = m.desc;
      $("viewActions").innerHTML = viewActions(name);
      bindViewActions(name);
      syncLongRunningActions();
      if (name === "requests") enterRequests(); else stopRequestsLive();
      if (location.hash !== "#/" + name) history.replaceState(null, "", "#/" + name);
      window.scrollTo({ top: 0, behavior: "instant" in document.documentElement.style ? "instant" : "auto" });
    }
    function viewActions(name) {
	      const btn = (id, label, primary, icon, title) => '<button class="btn ' + (primary ? "primary" : "") + '" id="' + id + '"' + (title ? ' title="' + esc(title) + '"' : "") + ">" + (icon || "") + label + "</button>";
	      if (name === "routing") return btn("refreshBtn", "Refresh catalog", false, ic(ICONS.refresh), "Re-fetch each provider's model list and rebuild the free-only catalog. Metadata only — no test requests, no tokens spent.") + btn("benchmarkBtn", "Retest candidates", false, ic(ICONS.retest), "Send a real test request to every usable model in this pool, one by one, to refresh pass/fail evidence. Uses a little quota per model. Tests of one provider are spaced ~4s apart so this cannot trip its rate limit, so a large pool takes several minutes.") + btn("saveBtn", 'Save<span id="saveCount" class="save-count"></span>', true, ic(ICONS.save));
	      if (name === "fusion") return state?.pro_installed ? btn("saveFusionBtn", "Save settings", true, ic(ICONS.save)) : "";
	      if (name === "settings") return btn("saveSettingsBtn", "Save settings", true, ic(ICONS.save));
	      if (name === "compression") return state?.pro_installed ? btn("reloadCompressionBtn", "Reload", false, ic(ICONS.refresh)) + btn("saveCompressionBtn", "Save settings", true, ic(ICONS.save)) : "";
	      if (name === "health") return btn("canaryBtn", "Run canary", true, ic(ICONS.canary));
      if (name === "audit") return btn("reloadAuditBtn", "Reload", false, ic(ICONS.refresh));
      if (name === "requests") return btn("reloadRequestsBtn", "Reload", false, ic(ICONS.refresh)) + '<button class="btn' + (requestsLive ? " primary" : "") + '" id="liveRequestsBtn" type="button" title="Stream new requests as they are routed, without reloading the page." aria-pressed="' + (requestsLive ? "true" : "false") + '">' + liveLabel() + "</button>";
      if (name === "export") return btn("buildExportBtn", "Build snippet", true, ic(ICONS.export));
      return "";
    }
    const ic = (p) => '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + p + "</svg>";
    function bindViewActions(name) {
	      if (name === "routing") {
	        $("refreshBtn").addEventListener("click", refreshCatalog);
	        $("benchmarkBtn").addEventListener("click", runBenchmark);
	        $("saveBtn").addEventListener("click", saveProfiles);
	        // The header is rebuilt on every view switch, so the fresh button needs the count.
	        syncSaveButton();
	      } else if (name === "fusion") { if (state?.pro_installed) $("saveFusionBtn").addEventListener("click", saveFusion); }
		      else if (name === "settings") $("saveSettingsBtn").addEventListener("click", saveSettings);
	      else if (name === "compression") {
	        if (state?.pro_installed) {
	          $("reloadCompressionBtn").addEventListener("click", () => loadCompression().then(() => showToast("Compression dashboard reloaded.")).catch((e) => showToast(e.message)));
	          $("saveCompressionBtn").addEventListener("click", saveCompression);
	        }
	      }
	      else if (name === "health") $("canaryBtn").addEventListener("click", runCanary);
      else if (name === "audit") $("reloadAuditBtn").addEventListener("click", () => loadAudit().then(() => showToast("Audit reloaded.")).catch((e) => showToast(e.message)));
      else if (name === "requests") {
        $("reloadRequestsBtn").addEventListener("click", () => loadRequests().then(() => showToast("Requests reloaded.")).catch((e) => showToast(e.message)));
        $("liveRequestsBtn").addEventListener("click", toggleRequestsLive);
      }
      else if (name === "export") $("buildExportBtn").addEventListener("click", () => exportConfig().catch((e) => showToast(e.message)));
    }

    /* ---------- data + actions ---------- */
	    async function loadState() {
	      // Only this fetch can say "the daemon is not up yet"; everything below runs after it
	      // answered. bootstrapState reads `bootRetryable` rather than sniffing the error type,
	      // because a TypeError thrown by loadAudit or render() would otherwise be mistaken for
	      // an unreachable service and replay the whole pipeline for 20s.
	      let r;
	      try { r = await fetch("/admin/state", { cache: "no-store" }); }
	      catch (e) { e.bootRetryable = true; throw e; }
	      if (!r.ok) {
	        const err = new Error("state failed: " + r.status);
	        err.status = r.status; err.bootRetryable = r.status >= 500;
	        throw err;
	      }
	      setState(await r.json());
	      setDraftProfiles(cloneJson(state.virtual_profiles || {}));
	      setDraftFusion(cloneJson(state.fusion?.config || {}));
		      setDraftSettings(cloneJson(state.settings?.config || {}));
	      for (const pid of profileOrder) { draftProfiles[pid] ||= { mode: "auto", models: [], excluded_models: [], auto_tail: true, requirements: defaultReq() }; draftProfiles[pid].excluded_models ||= []; draftProfiles[pid].requirements = { ...defaultReq(), ...(draftProfiles[pid].requirements || {}) }; }
      await Promise.all([loadAudit(false), loadOptionalCompression()]); render();
	      // Refresh the action toolbar now that state (incl. pro_installed) is loaded — it was first
	      // built by setView() at bootstrap when state was null, so Pro-gated Fusion/Compression
	      // buttons would otherwise never appear on a direct page load.
      setView(currentView);
      startUpdateStatusPolling();
    }
    async function loadAudit(after = true) {
      const r = await fetch("/admin/audit?limit=12", { cache: "no-store" }); const p = await r.json();
      if (!r.ok) throw new Error(p?.error?.message || "audit failed"); setAuditEntries(p.entries || []);
      if (after) { renderAudit(); renderNavCounts(); }
    }
    // Delegates to the closed pack. No-op (resolves undefined) on a core-only install so callers
    // (loadState, loadOptionalCompression, the Reload button) degrade gracefully without the Pro module.
    async function loadCompression(after = true) {
      return proViews.loadCompression ? proViews.loadCompression(proApi, after) : undefined;
    }
	    async function loadOptionalCompression() {
	      try {
	        await loadCompression(false);
	      } catch {
	        setCompressionPayload({ observability: {}, events: [], event_count: 0 });
	      }
    }
    async function saveProfiles(successMessage) {
      const toastMessage = typeof successMessage === "string" ? successMessage : "Virtual models saved.";
      const btn = $("saveBtn"); setButtonLoading(btn, true);
      try {
        const r = await fetch("/admin/profiles", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ virtual_profiles: draftProfiles }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "save failed");
        state.virtual_profiles = p.virtual_profiles; setDraftProfiles(cloneJson(p.virtual_profiles));
        showToast(toastMessage); await loadAudit(false); render();
        // render() ran while the button still carried is-loading, which syncSaveButton
        // refuses to touch — so re-sync once the spinner is off, or the count would
        // survive its own save.
      } catch (e) { showToast(e.message || String(e)); } finally { setButtonLoading(btn, false); syncSaveButton(); }
    }
	    async function rollbackProfiles(id) {
	      if (!id || !window.confirm("Undo to this saved snapshot?")) return;
	      try { const r = await fetch("/admin/profiles/rollback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audit_id: id }) });
	        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "undo failed");
	        state.virtual_profiles = p.virtual_profiles; setDraftProfiles(cloneJson(p.virtual_profiles)); showToast("Virtual models restored."); await loadAudit(false); render();
	      } catch (e) { showToast(e.message || String(e)); }
	    }
	    // saveFusion delegates to the closed pack; it no-ops (resolves undefined) when the
	    // Pro module is absent (no fusion form rendered).
	    async function saveFusion() { return proViews.saveFusion ? proViews.saveFusion(proApi) : undefined; }
	    async function rollbackFusion(id) {
	      if (!id || !window.confirm("Undo to this saved fusion snapshot?")) return;
	      try {
	        const r = await fetch("/admin/fusion/rollback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audit_id: id }) });
	        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "fusion undo failed");
	        state.fusion = { config: p.fusion, observability: p.observability }; setDraftFusion(cloneJson(p.fusion));
	        showToast("Fusion settings restored."); await loadAudit(false); render();
	      } catch (e) { showToast(e.message || String(e)); }
	    }
	    async function saveRouterSettings(buttonId, successMessage, failureMessage, options = {}) {
	      const btn = $(buttonId); setButtonLoading(btn, true);
	      try {
	        const settings = readSettingsDraft();
	        const r = await fetch("/admin/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ settings }) });
	        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || failureMessage);
	        state.settings = p.settings; setDraftSettings(cloneJson(p.settings.config));
	        showToast(successMessage);
		        if (options.refreshCompression) await Promise.all([loadAudit(false), loadOptionalCompression()]);
	        else await loadAudit(false);
	        render();
	      } catch (e) { showToast(e.message || String(e)); } finally { setButtonLoading(btn, false); }
	    }
	    async function saveSettings() {
	      await saveRouterSettings("saveSettingsBtn", "Routing settings saved.", "settings save failed");
	    }
	    async function rollbackSettings(id) {
	      if (!id || !window.confirm("Undo to this saved settings snapshot?")) return;
	      try {
	        const r = await fetch("/admin/settings/rollback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audit_id: id }) });
	        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "settings undo failed");
	        state.settings = p.settings; setDraftSettings(cloneJson(p.settings.config));
	        showToast("Routing settings restored."); await loadAudit(false); render();
	      } catch (e) { showToast(e.message || String(e)); }
	    }
	    // saveCompression delegates to the closed pack (the Save button calls it); no-op on a
	    // core-only install. Retrieve/clear are wired entirely inside the Pro retrieval UI (pro-views.js).
	    async function saveCompression() { return proViews.saveCompression ? proViews.saveCompression(proApi) : undefined; }
		    async function refreshCatalog() {
      const b = $("refreshBtn"); setButtonLoading(b, true);
      try { const r = await fetch("/admin/refresh", { method: "POST" }); const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "refresh failed");
        showToast("Catalog refreshed: " + (p.summary?.model_count || 0) + " models."); await loadState();
      } catch (e) { showToast(e.message || String(e)); } finally { setButtonLoading(b, false); }
    }
    async function runBenchmark() {
      if (benchmarkRunInFlight) return;
      benchmarkRunInFlight = true;
      syncLongRunningActions();
      try { const r = await fetch("/admin/benchmark", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profiles: [activeProfile], all_candidates: true }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "benchmark failed");
        const candidates = p.summary?.candidates;
        showToast(candidates ? ("Benchmark candidates " + (candidates.passed || 0) + "/" + (candidates.tested || 0) + " passed.") : ("Benchmark " + (p.summary?.passed || 0) + "/" + (p.summary?.total || 0) + " passed.")); await loadState();
      } catch (e) { showToast(e.message || String(e)); } finally { benchmarkRunInFlight = false; if (state?.state?.admin_jobs?.benchmark) delete state.state.admin_jobs.benchmark; syncLongRunningActions(); }
    }
    async function runModelBenchmark(id, btn) {
      if (benchmarkingModels.has(id)) return; // already testing this model — ignore repeat clicks
      benchmarkingModels.add(id);
      setButtonLoading(btn, true);
      try {
        const r = await fetch("/admin/benchmark", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profiles: [activeProfile], all_candidates: true, model_id: id }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "benchmark failed");
        const res = (p.results || [])[0] || {};
        const cr = (res.candidate_results || [])[0] || null;
        if (cr) showToast("Benchmark " + (cr.status === "pass" ? "passed" : "failed") + (cr.latency_seconds != null ? " in " + Number(cr.latency_seconds).toFixed(2) + "s" : "") + (cr.message ? ": " + cr.message : ""));
        else showToast(res.message || "Benchmark: no result");
      } catch (e) { showToast(e.message || String(e)); }
      finally {
        // Clear the id before the refresh so loadState()'s re-render drops the spinner. The
        // setButtonLoading here is the fallback that unsticks the button if loadState() — and thus
        // the re-render — fails; delete() runs on every path so a thrown fetch never strands the id.
        benchmarkingModels.delete(id);
        setButtonLoading(btn, false);
        loadState().catch((e) => showToast("State refresh failed: " + (e.message || String(e))));
      }
    }
    async function runCanary() {
      const b = $("canaryBtn"); setButtonLoading(b, true);
      try { const r = await fetch("/admin/canary", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "canary failed");
        showToast("Canary " + p.canary_status + ": " + (p.summary?.passed || 0) + "/" + (p.summary?.total || 0) + " passed."); await loadState();
      } catch (e) { showToast(e.message || String(e)); } finally { setButtonLoading(b, false); }
    }
    async function clearModelCooldown(id) {
      try { const r = await fetch("/admin/cooldowns", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "clear", scope: "model", model_id: id }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "failed"); showToast(p.changed ? "Model resumed." : "No active pause."); await loadState();
      } catch (e) { showToast(e.message || String(e)); }
    }
    async function clearProviderCooldown(src) {
      try { const r = await fetch("/admin/cooldowns", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "clear", scope: "provider", source: src }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "failed"); showToast(p.changed ? "Provider resumed." : "No active pause."); await loadState();
      } catch (e) { showToast(e.message || String(e)); }
    }
    async function probeProviderQuota(src, btn) {
      setButtonLoading(btn, true);
      try { const r = await fetch("/admin/providers/probe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: src }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "failed"); showToast("Quota probe checked " + Number((p.results || []).length) + " cooldowns."); await loadState();
      } catch (e) { showToast(e.message || String(e)); }
      finally { setButtonLoading(btn, false); }
    }
    async function toggleProvider(src, enable, btn) {
      setButtonLoading(btn, true);
      try { const r = await fetch("/admin/providers/toggle", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: src, action: enable ? "enable" : "disable" }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "failed"); showToast(providerLabel(src) + (enable ? " enabled." : " disabled.")); await loadState();
      } catch (e) { showToast(e.message || String(e)); }
      finally { setButtonLoading(btn, false); }
    }
    function adminToken() { return document.querySelector('meta[name="ficelle-admin-token"]')?.content || ""; }
    async function setProviderKey(src) {
      const input = $("providerGrid").querySelector('[data-key-input="' + src + '"]');
      const key = (input?.value || "").trim();
      if (!key) { showToast("Paste a key first."); return; }
      try { const r = await fetch("/admin/providers/" + encodeURIComponent(src) + "/key", { method: "POST", headers: { "Content-Type": "application/json", "X-Ficelle-Admin-Token": adminToken() }, body: JSON.stringify({ key }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "failed"); if (input) input.value = ""; showToast(providerLabel(src) + " key saved."); await loadState();
      } catch (e) { showToast(e.message || String(e)); }
    }
    async function removeProviderKey(src) {
      try { const r = await fetch("/admin/providers/" + encodeURIComponent(src) + "/key", { method: "POST", headers: { "Content-Type": "application/json", "X-Ficelle-Admin-Token": adminToken() }, body: JSON.stringify({ remove: true }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "failed");
        const legacySources = Array.isArray(p.remaining_legacy_sources) ? p.remaining_legacy_sources : [];
        if (legacySources.length) {
          showToast("Legacy credential fallback sources still apply. Remove them manually from: " + legacySources.join(", ") + ". The provider may remain configured until cleanup is complete.");
        } else {
          showToast((p.removed && p.removed.length) ? (providerLabel(src) + " stored key removed.") : "No Ficelle-managed key found.");
        }
        await loadState();
      } catch (e) { showToast(e.message || String(e)); }
    }
    async function updateQuarantine(id, action) {
      try { const r = await fetch("/admin/quarantine", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: id, action, reason: "manual", note: "manual admin change" }) });
        const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "failed"); showToast(action === "remove" ? "Model re-enabled." : "Model disabled."); await loadState();
      } catch (e) { showToast(e.message || String(e)); }
    }
    async function exportConfig() {
      const r = await fetch("/admin/export/generic"); const p = await r.json(); if (!r.ok) throw new Error(p?.error?.message || "export failed");
      $("configExport").value = JSON.stringify(p.config || {}, null, 2); showToast("Configuration ready.");
      // Spell out the same values the JSON below carries, from the same payload: the
      // server derives base_url from config.host/port, so deriving them here from
      // location.origin instead would print different URLs in one card.
      $("exportEndpointUrl").textContent = p.config?.base_url || "";
      $("exportEndpointChatUrl").textContent = p.config?.chat_completions_url || "";
      $("exportEndpointKey").textContent = p.config?.api_key || "";
      $("exportEndpoint").hidden = false;
    }
    async function copyExport() { const t = $("configExport").value; if (!t) { showToast("Build the snippet first."); return; } await copyToClipboard(t, "JSON copied."); }

    /* ---------- endpoint card ---------- */
    function initEndpointCard() {
      // location.origin, not the config: this card paints before any fetch, and the address
      // the dashboard was reached on is by definition one that resolves. The Export view
      // shows the server's own base_url instead — see exportConfig.
      const base = location.origin + "/v1";
      // Both forms, because clients disagree on what an "endpoint" field means. Offering only
      // the base leaves the verbatim kind to assemble a URL by hand and land on POST /v1.
      bindEndpointCopy("endpointCopy", "endpointUrl", base);
      bindEndpointCopy("endpointChatCopy", "endpointChatUrl", base + "/chat/completions");
    }
    function bindEndpointCopy(buttonId, valueId, url) {
      $(valueId).textContent = url;
      $(buttonId).addEventListener("click", () => copyToClipboard(url, "Endpoint copied: " + url, "Copy failed. The endpoint is " + url));
    }

    /* ---------- theme ---------- */
    const THEME_KEY = "ficelle.theme";
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    function applyTheme() {
      const pref = localStorage.getItem(THEME_KEY) || "system";
      const eff = pref === "system" ? (mq.matches ? "dark" : "light") : pref;
      document.documentElement.setAttribute("data-theme", eff);
      document.querySelectorAll("[data-theme-pref]").forEach((b) => b.classList.toggle("active", b.dataset.themePref === pref));
    }
    function initTheme() {
      applyTheme();
      document.querySelectorAll("[data-theme-pref]").forEach((b) => b.addEventListener("click", () => { localStorage.setItem(THEME_KEY, b.dataset.themePref); applyTheme(); }));
      mq.addEventListener("change", () => { if ((localStorage.getItem(THEME_KEY) || "system") === "system") applyTheme(); });
    }

    /* ---------- boot ---------- */
    function initNav() {
      document.querySelectorAll(".nav-item").forEach((b) => b.addEventListener("click", () => setView(b.dataset.nav)));
      $("railHealth").addEventListener("click", () => setView("health"));
      window.addEventListener("hashchange", () => setView((location.hash || "#/routing").replace("#/", "")));
    }
    function initStaticControls() {
      $("autoModeBtn").addEventListener("click", () => setProfile({ mode: "auto" }));
      $("manualModeBtn").addEventListener("click", () => setProfile({ mode: "manual_order" }));
      $("autoTailToggle").addEventListener("change", (e) => setProfile({ auto_tail: e.target.checked }));
      $("clearProfileBtn").addEventListener("click", () => setProfile({ mode: "auto", models: [], excluded_models: [] }));
      $("searchInput").addEventListener("input", renderAvailable);
      $("copyExportBtn").addEventListener("click", () => copyExport().catch((e) => showToast(e.message)));
      initEndpointCard();
      // Closing or reloading with a draft in memory silently discards it. The browser
      // shows its own generic wording here; the text is ignored by every modern engine.
      window.addEventListener("beforeunload", (e) => {
        if (!unsavedProfileIds().length) return;
        e.preventDefault();
        e.returnValue = "";
      });
      // Close any open filter popover when clicking outside the bar. Capture phase so it
      // still fires even though model-card buttons stopPropagation on bubble.
      document.addEventListener("click", (e) => { if (!e.target.closest("#filterBar .flt")) closeFilterPops(); }, true);
      // A hidden tab holds no live tail: drop it on the way out, and resync on the way back
      // rather than replaying whatever piled up while nobody was looking.
      document.addEventListener("visibilitychange", () => {
        if (currentView !== "requests" || !requestsLive) return;
        if (document.hidden) { stopRequestsLive(); return; }
        loadRequests().then(() => startRequestsLive()).catch(() => startRequestsLive());
      });
    }
    // Bridge for the optional closed-pack views (pro-views.js). The moved Fusion/Compression render
    // and save/load functions read every helper, state accessor, and core callback they need from this
    // single object — store live bindings are exposed as getters (importers must see current values),
    // setters are passed through directly. Defined here, after every function/const above is in scope.
    const proApi = {
      // lib.js helpers
      $, esc, cloneJson, showToast, formatDuration, timeChip, timeAgo, reasonWithCode, providerLabel,
      profileLabels, profileOrder,
      // shared core layout primitives + icons (also used by the core Settings view)
      fusionSetting, fusionSettingsGroup, settingsNumber, settingsToggle, settingsCompressionMode,
      SETTINGS_COMPRESSION_STRATEGIES, ic, ICONS,
      // license key deep-linked from the purchase success page (pre-fills the License field)
      deepLinkedKey,
      // store live bindings (getters) + setters
      getState: () => state,
      getDraftFusion: () => draftFusion, setDraftFusion,
      getDraftSettings: () => draftSettings,
      getDraftProfiles: () => draftProfiles,
      getCompressionPayload: () => compressionPayload, setCompressionPayload,
      // core functions the moved bodies call back into
      readSettingsDraft, saveRouterSettings, loadAudit, renderNavCounts,
      currentCompressionObservability,
      rerender: render,
    };
    // Lets pro-views.js trigger one render after it registers its implementations (any load order).
    proViews.__boot = () => render();

    initTheme(); initNav(); initStaticControls();
    // A deep-linked key means the user just bought Pro — land on the License view to unlock.
    setView(deepLinkedKey ? "license" : (location.hash || "#/routing").replace("#/", ""));

    /* First load is a race against the daemon: `bootstrap-ficelle.py` starts the service
       and prints the dashboard URL in the same breath, and launchd may still be restarting
       it. A single failed load used to leave the page on an ambiguous "0 usable / 0 models"
       with nothing but a toast that faded away, and no retry — the user had to guess to
       reload. The notice is armed on a timer rather than shown upfront: the first
       /admin/state can also simply be *slow* (it builds the catalog), and a pending fetch
       never reaches a catch — so an error-only notice would never paint in that case,
       while showing it upfront would flash on every healthy load. */
    const BOOT_DEADLINE_MS = 20000; // matches wait_for_admin_status() in cli.py
    const BOOT_NOTICE_DELAY_MS = 600;
    function bootNotice(text) {
      const el = $("bootNotice");
      el.textContent = text;
      el.hidden = false;
    }
    async function bootstrapState() {
      const slow = setTimeout(() => bootNotice("Waiting for the Ficelle service to come up…"), BOOT_NOTICE_DELAY_MS);
      const deadline = Date.now() + BOOT_DEADLINE_MS;
      let lastError = null;
      let attempt = 0;
      try {
        while (Date.now() < deadline) {
          try {
            await loadState();
            $("bootNotice").hidden = true;
            return;
          } catch (e) {
            lastError = e;
            // Only "nobody answered yet" is worth retrying — loadState tags that case
            // itself. A 4xx, or any failure after /admin/state already returned 200, would
            // replay the whole state+audit+render pipeline a dozen times for one outcome.
            if (!e?.bootRetryable) break;
            await sleep(Math.min(2000, 300 * ++attempt));
          }
        }
      } finally {
        clearTimeout(slow);
      }
      bootNotice("Can't reach the Ficelle service. Check it with `ficelle status`, then reload this page.");
      showToast(lastError ? (lastError.message || String(lastError)) : "Loading the router state failed.");
    }
    bootstrapState();
