"""Behaviour of the admin dashboard that an operator relies on but no server test can see.

Each test evaluates the real functions of ``app.js`` / ``lib.js`` in Node with the DOM and the
network replaced by stubs, so a regression in the shipped asset fails here rather than in a
browser nobody is watching.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ADMIN_ASSETS = Path(__file__).parents[1] / "src" / "ficelle" / "assets" / "admin"


def asset(name: str) -> str:
    return (ADMIN_ASSETS / name).read_text(encoding="utf-8")


def js_function(source: str, name: str) -> str:
    """Extract one simple dashboard function (sync or async) by brace matching."""
    for prefix in (f"async function {name}(", f"function {name}("):
        start = source.find(prefix)
        if start >= 0:
            break
    else:
        raise AssertionError(f"missing JavaScript function {name}")
    brace = source.index("{", start)
    depth = 0
    for offset, char in enumerate(source[brace:], start=brace):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated JavaScript function {name}")


def run_node(*lines: str) -> str:
    script = "\n".join(lines)
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", f"(async () => {{\n{script}\n}})()"],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout


def test_exclude_saves_only_the_exclusion_and_keeps_other_unsaved_edits_as_drafts():
    app_js = asset("app.js")
    out = run_node(
        "const clone = (v) => JSON.parse(JSON.stringify(v || {}));",
        "const cloneJson = clone;",
        "const state = {virtual_profiles:{A:{mode:'auto',models:[],excluded_models:[]},B:{mode:'manual_order',models:['m1'],excluded_models:[]}}};",
        "let draftProfiles = clone(state.virtual_profiles);",
        # Unreviewed edits: lane B has a new model in its order, lane A was switched to manual order.
        "draftProfiles.B.models = ['m1','m2']; draftProfiles.A.mode = 'manual_order'; draftProfiles.A.models = ['m3'];",
        "let activeProfile = 'A';",
        "const uniqueStrings = (items) => Array.from(new Set((items || []).map(String).filter(Boolean)));",
        "const currentProfile = () => draftProfiles[activeProfile];",
        "const excludedModelIds = (profile = currentProfile()) => uniqueStrings(profile.excluded_models || []);",
        "function setProfile(next) { draftProfiles[activeProfile] = {...currentProfile(), ...next}; return true; }",
        "function setDraftProfiles(value) { draftProfiles = value; }",
        "const button = {classList:{contains:()=>false}}; const $ = () => button;",
        "let posted = null; let toast = '';",
        "async function fetch(_url, options) { posted = JSON.parse(options.body).virtual_profiles; return {ok:true,json:async()=>({virtual_profiles:posted})}; }",
        "async function loadState() {} function render() {} function showError() {} function setButtonLoading() {} function syncSaveButton() {}",
        "function showToast(message) { toast = message; }",
        js_function(app_js, "saveProfiles"),
        js_function(app_js, "exclusionOnlyProfiles"),
        "let exclusionSaveQueue = Promise.resolve();",
        js_function(app_js, "queueExclusionSave"),
        js_function(app_js, "excludeModel"),
        "excludeModel('m9'); await exclusionSaveQueue;",
        "console.log(JSON.stringify({posted, draftA: draftProfiles.A, draftB: draftProfiles.B}));",
    )
    result = json.loads(out)
    posted = result["posted"]
    assert posted["A"]["excluded_models"] == ["m9"]
    # The saved order of lane A and lane B go out as they were saved, not as they were edited.
    assert posted["A"]["mode"] == "auto" and posted["A"]["models"] == []
    assert posted["B"]["models"] == ["m1"]
    # The edits stay parked as drafts, so Save still offers them.
    assert result["draftA"]["mode"] == "manual_order" and result["draftA"]["excluded_models"] == ["m9"]
    assert result["draftB"]["models"] == ["m1", "m2"]


def test_two_quick_exclusions_both_stay_saved():
    # Both POSTs used to start from the same saved state, so the second save dropped the first
    # exclusion while both confirmations were shown.
    app_js = asset("app.js")
    out = run_node(
        "const clone = (v) => JSON.parse(JSON.stringify(v || {})); const cloneJson = clone;",
        "const state = {virtual_profiles:{A:{mode:'auto',models:[],excluded_models:[]}}};",
        "let draftProfiles = clone(state.virtual_profiles); let activeProfile = 'A';",
        "const uniqueStrings = (items) => Array.from(new Set((items || []).map(String).filter(Boolean)));",
        "const currentProfile = () => draftProfiles[activeProfile];",
        "const excludedModelIds = (profile = currentProfile()) => uniqueStrings(profile.excluded_models || []);",
        "function setProfile(next) { draftProfiles[activeProfile] = {...currentProfile(), ...next}; return true; }",
        "function setDraftProfiles(value) { draftProfiles = value; }",
        "const button = {classList:{contains:()=>false}}; const $ = () => button;",
        "const saved = []; const pending = [];",
        # The server answers each POST later, with exactly what it stored.
        "async function fetch(_url, options) { const body = JSON.parse(options.body).virtual_profiles; saved.push(body.A.excluded_models);"
        " await new Promise((resolve) => pending.push(resolve)); return {ok:true,json:async()=>({virtual_profiles:body})}; }",
        "async function loadState() {} function render() {} function showError(e) { throw e; } function setButtonLoading() {} function syncSaveButton() {} function showToast() {}",
        js_function(app_js, "saveProfiles"),
        js_function(app_js, "exclusionOnlyProfiles"),
        "let exclusionSaveQueue = Promise.resolve();",
        js_function(app_js, "queueExclusionSave"),
        js_function(app_js, "excludeModel"),
        "excludeModel('m1'); excludeModel('m2');",
        "const settle = async () => { while (pending.length || saved.length < 2) { await new Promise((r) => setTimeout(r, 0)); pending.shift()?.(); } };",
        "await settle(); await exclusionSaveQueue;",
        "console.log(JSON.stringify({saved, final: state.virtual_profiles.A.excluded_models}));",
    )
    result = json.loads(out)
    assert result["saved"] == [["m1"], ["m1", "m2"]]
    assert result["final"] == ["m1", "m2"]


def test_restore_saves_only_the_restoration():
    app_js = asset("app.js")
    out = run_node(
        "const clone = (v) => JSON.parse(JSON.stringify(v || {})); const cloneJson = clone;",
        "const state = {virtual_profiles:{A:{mode:'auto',models:[],excluded_models:['m9','m8']}}};",
        "let draftProfiles = clone(state.virtual_profiles); draftProfiles.A.mode = 'manual_order';",
        "let activeProfile = 'A';",
        "const uniqueStrings = (items) => Array.from(new Set((items || []).map(String).filter(Boolean)));",
        "const currentProfile = () => draftProfiles[activeProfile];",
        "const excludedModelIds = (profile = currentProfile()) => uniqueStrings(profile.excluded_models || []);",
        "function setProfile(next) { draftProfiles[activeProfile] = {...currentProfile(), ...next}; return true; }",
        "let posted = null; function saveProfiles(_message, profiles) { posted = profiles; } function showToast() {}",
        js_function(app_js, "exclusionOnlyProfiles"),
        "let exclusionSaveQueue = Promise.resolve();",
        js_function(app_js, "queueExclusionSave"),
        js_function(app_js, "restoreModel"),
        "restoreModel('m9'); await exclusionSaveQueue; console.log(JSON.stringify(posted));",
    )
    posted = json.loads(out)
    assert posted["A"]["excluded_models"] == ["m8"]
    assert posted["A"]["mode"] == "auto"


def test_state_reload_from_an_action_keeps_unsaved_drafts():
    app_js = asset("app.js")
    run_node(
        "let state = {virtual_profiles:{p:{label:'Saved'}},catalog:{}};",
        "let draftProfiles = {p:{label:'Edited'}}; let draftFusion = {panel_size:4}; let draftSettings = {max_attempts:9}; let currentView = 'routing';",
        "async function fetch() { return {ok:true,json:async()=>({virtual_profiles:{p:{label:'Saved'}},fusion:{config:{panel_size:2}},settings:{config:{max_attempts:2}},catalog:{},notices:{}})}; }",
        "const cloneJson = (value) => JSON.parse(JSON.stringify(value)); const unsavedProfileIds = () => ['p'];",
        "function setState(value) { state = value; } function seedProfileDrafts(ids) { const previous = draftProfiles; draftProfiles = cloneJson(state.virtual_profiles); ids.forEach((id) => { draftProfiles[id] = previous[id]; }); }",
        "function setDraftFusion(value) { draftFusion = value; } function setDraftSettings(value) { draftSettings = value; }",
        "async function loadAudit() {} async function loadOptionalCompression() {} async function refreshNotices() {} function render() {} function catalogMarker() { return null; } async function resyncIfCatalogRepublished() {} function setView() {} function startUpdateStatusPolling() {} let renderedCatalogMarker;",
        js_function(app_js, "draftMatchesSaved"),
        js_function(app_js, "loadState"),
        # Refresh catalog, Retest, Disable, a key save... all end in a bare loadState().
        "await loadState();",
        "if (draftProfiles.p.label !== 'Edited' || draftFusion.panel_size !== 4 || draftSettings.max_attempts !== 9) throw new Error('an action refresh discarded unsaved drafts');",
        # Only the first load of the page seeds every draft from the server.
        "draftProfiles = {}; draftFusion = {}; draftSettings = {}; await loadState(true);",
        "if (draftProfiles.p.label !== 'Saved' || draftFusion.panel_size !== 2 || draftSettings.max_attempts !== 2) throw new Error('boot did not seed the drafts from the server');",
    )


def test_explicit_reload_takes_new_values_for_drafts_left_untouched():
    # Another tab saved new Fusion settings. An untouched draft kept the old values, which then
    # read as pending edits and a later Save wrote them back over the other tab's work.
    app_js = asset("app.js")
    out = run_node(
        "let state = {virtual_profiles:{},fusion:{config:{panel_size:2}},settings:{config:{max_attempts:2}},catalog:{}};",
        "let draftProfiles = {}; let draftFusion = {panel_size:2}; let draftSettings = {max_attempts:9}; let currentView = 'routing';",
        "async function fetch() { return {ok:true,json:async()=>({virtual_profiles:{},fusion:{config:{panel_size:5}},settings:{config:{max_attempts:4}},catalog:{},notices:{}})}; }",
        "const cloneJson = (value) => JSON.parse(JSON.stringify(value)); const unsavedProfileIds = () => [];",
        "function setState(value) { state = value; } function seedProfileDrafts() {}",
        "function setDraftFusion(value) { draftFusion = value; } function setDraftSettings(value) { draftSettings = value; }",
        "async function loadAudit() {} async function loadOptionalCompression() {} async function refreshNotices() {} function render() {} function catalogMarker() { return null; } async function resyncIfCatalogRepublished() {} function setView() {} function startUpdateStatusPolling() {} let renderedCatalogMarker;",
        js_function(app_js, "draftMatchesSaved"),
        js_function(app_js, "loadState"),
        "await loadState(); console.log(JSON.stringify({draftFusion, draftSettings}));",
    )
    result = json.loads(out)
    assert result["draftFusion"] == {"panel_size": 5}
    # The edited settings draft is still the operator's.
    assert result["draftSettings"] == {"max_attempts": 9}


def test_a_shadowed_key_save_never_advises_removing_every_stored_key():
    # `ficelle remove-key` clears every Ficelle store, the key just saved included.
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    out = run_node(
        f"const {{ providerKeyShadowNotice }} = await import({json.dumps(lib_uri)});",
        "console.log(JSON.stringify(['keychain', '.env', 'env'].map((family) => providerKeyShadowNotice('Groq', 'groq', family))));",
    )
    keychain, env_file, env = json.loads(out)
    assert all("remove-key" not in notice for notice in (keychain, env_file, env))
    assert "keychain" in keychain and "ficelle set-key groq" in keychain
    assert ".env" in env_file
    assert "environment variable" in env


def _toast_dom() -> str:
    return "\n".join(
        [
            "const attrs = {}; const classes = new Set(); let timeout = null;",
            "const toast = {classList:{add:(...c)=>c.forEach((x)=>classes.add(x)),remove:(...c)=>c.forEach((x)=>classes.delete(x)),contains:(c)=>classes.has(c)},setAttribute:(k,v)=>{attrs[k]=v;},addEventListener(){},matches:()=>false,contains:()=>false};",
            "const fields = {toast, toastMsg:{}, toastAction:{}, toastClose:{addEventListener(){}}};",
            "globalThis.document = {getElementById:(id)=>fields[id]};",
            "globalThis.setTimeout = (fn, ms) => { timeout = ms; return 1; }; globalThis.clearTimeout = () => {};",
        ]
    )


def test_failures_are_announced_as_errors_and_stay_long_enough_to_read():
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    hint = "The OS secret store refused the key. Run `ficelle set-key groq --allow-plaintext` to keep it in the Ficelle home instead."
    out = run_node(
        _toast_dom(),
        f"const {{ showError, showToast }} = await import({json.dumps(lib_uri)});",
        f"showError(new Error({json.dumps(hint)}));",
        "const failure = {error: classes.has('is-error'), role: attrs.role, live: attrs['aria-live'], ms: timeout, text: fields.toastMsg.textContent};",
        "showToast('Saved.');",
        "const after = {error: classes.has('is-error'), role: attrs.role, live: attrs['aria-live'], ms: timeout};",
        "console.log(JSON.stringify({failure, after}));",
    )
    result = json.loads(out)
    failure, after = result["failure"], result["after"]
    assert failure["error"] and failure["role"] == "alert" and failure["live"] == "assertive"
    assert failure["text"] == hint
    # A two-sentence remedy cannot be read in the 2.8 s a plain acknowledgement gets.
    assert failure["ms"] >= 8000
    assert not after["error"] and after["role"] == "status" and after["live"] == "polite" and after["ms"] == 2800


def test_removing_a_provider_key_needs_confirmation():
    app_js = asset("app.js")
    out = run_node(
        "let confirmed = false; const prompts = []; globalThis.window = {confirm: (message) => { prompts.push(message); return confirmed; }};",
        "let posts = 0; async function fetch() { posts += 1; return {ok:true,json:async()=>({removed:['keychain:GROQ_API_KEY'],auth:{}})}; }",
        "const providerLabel = () => 'Groq'; const adminToken = () => 't'; const providerKeyRemovalNotice = () => ({message:'removed',level:'ok'});",
        "function showToast() {} function showError() {} async function loadState() {}",
        js_function(app_js, "removeProviderKey"),
        "await removeProviderKey('groq'); const declined = posts;",
        "confirmed = true; await removeProviderKey('groq');",
        "console.log(JSON.stringify({declined, accepted: posts, prompt: prompts[0]}));",
    )
    result = json.loads(out)
    assert result["declined"] == 0
    assert result["accepted"] == 1
    # The prompt must say what the click costs: routing for that provider stops and Ficelle cannot bring the key back.
    assert "Groq" in result["prompt"] and "cannot recover" in result["prompt"]


def _first_run_state(invokable: bool) -> str:
    return (
        "let state = {catalog:{providers:{groq:{invokable:%s},nous:{invokable:false}},models:[]},state:{},profiles:{}};"
        % ("true" if invokable else "false")
    )


def _provider_status_stubs(app_js: str) -> list[str]:
    return [
        "const providerManuallyDisabled = (src) => state?.state?.provider_disabled?.[src] || null;",
        js_function(app_js, "credentialsUnreadable"),
        js_function(app_js, "noProviderConnected"),
    ]


def test_first_run_hint_says_enable_when_every_keyed_provider_is_disabled():
    # "No provider key yet" sent the operator to paste a key the provider already had; only
    # re-enabling it makes models usable again.
    app_js = asset("app.js")
    out = run_node(
        "let state = {catalog:{providers:{groq:{invokable:false,enabled:false,key_source:'keychain'},nous:{invokable:false}},models:[]},state:{}};",
        "const modelFilters = {newOnly:false};",
        *_provider_status_stubs(app_js),
        js_function(app_js, "emptyAvailableNotice"),
        "const disabled = emptyAvailableNotice();",
        "state.catalog.providers.groq = {invokable:false, key_source:'keychain'}; state.state.provider_disabled = {groq:{at:'x'}};",
        "const toggledOff = emptyAvailableNotice();",
        "console.log(JSON.stringify({disabled, toggledOff}));",
    )
    result = json.loads(out)
    for notice in (result["disabled"], result["toggledOff"]):
        assert "No provider key yet" not in notice
        assert "disabled" in notice and "Enable one in Providers" in notice


def test_one_disabled_keyed_provider_does_not_hide_another_enabled_one():
    # Only when every provider holding a key is disabled is "enable one" the whole answer;
    # an enabled keyed provider that still cannot serve needs its own fix.
    app_js = asset("app.js")
    out = run_node(
        "let state = {catalog:{providers:{groq:{invokable:false,enabled:false,key_source:'keychain'},cloudflare:{invokable:false,key_source:'keychain'}},models:[]},state:{}};",
        *_provider_status_stubs(app_js),
        "console.log(JSON.stringify(noProviderConnected()));",
    )
    assert json.loads(out) == ""


def test_fresh_install_points_to_providers_instead_of_reporting_guardrails():
    app_js = asset("app.js")
    script = lambda invokable: [  # noqa: E731 - one stub set, two states
        _first_run_state(invokable),
        "const rail = {}; const els = {railDot:{}, railHealthMain:rail, railHealthSub:{}, licenseNavText:null};",
        "const $ = (id) => els[id]; const document = {querySelector: () => null};",
        "const activeCount = () => 0; let compressionPayload = {}; let auditEntries = [];",
        "const modelFilters = {newOnly:false};",
        *_provider_status_stubs(app_js),
        js_function(app_js, "renderNavCounts"),
        "const emptyAvailableNotice = () => '';",
        "renderNavCounts(); console.log(JSON.stringify({main: els.railHealthMain.textContent, sub: els.railHealthSub.textContent}));",
    ]
    fresh = json.loads(run_node(*script(False)))
    assert fresh["main"] == "Needs a look"
    assert "No provider key yet" in fresh["sub"]
    assert "guardrails" not in fresh["sub"]
    # With a provider connected and no canary run, the sub-line says the canary is the missing piece.
    connected = json.loads(run_node(*script(True)))
    assert connected["sub"] == "canary not run yet"


def test_empty_available_list_says_a_key_is_missing_before_blaming_the_filters():
    app_js = asset("app.js")
    out = run_node(
        _first_run_state(False),
        "const modelFilters = {newOnly:false};",
        *_provider_status_stubs(app_js),
        js_function(app_js, "emptyAvailableNotice"),
        "const first = emptyAvailableNotice(); state.catalog.providers.groq.invokable = true; const later = emptyAvailableNotice();",
        "console.log(JSON.stringify({first, later}));",
    )
    result = json.loads(out)
    assert "No provider key yet" in result["first"] and "Providers" in result["first"]
    assert result["later"] == "No models match these filters."


def _requests_stubs(app_js: str) -> list[str]:
    return [
        "const requestFilters = {profile:'',source:'',reason:'',status:'',q:'',window:'24h'}; const REQ_LIST_CAP = 200;",
        "let requestsPayload = {entries:[]}; let requestsSummary = {}; let requestsPending = []; let requestsCursor = null;",
        "let requestsDegraded = false; let requestsSummaryAt = 0; let requestsLoad = {state:'idle', detail:''};",
        "const setRequestsPayload = (v) => { requestsPayload = v; }; const setRequestsSummary = (v) => { requestsSummary = v; };",
        "const $ = () => ({contains:()=>false}); const document = {activeElement:null};",
        "const toasts = []; const showToast = (m) => { toasts.push(m); };",
        "function renderRequests() {} function renderRequestsList() {} function renderRequestsFilters() {}",
        js_function(app_js, "requestQuery"),
        "const summaryUrl = () => '/admin/requests/summary?window=' + requestFilters.window;",
        js_function(app_js, "noteRequestsDegraded"),
        js_function(app_js, "requestsStatusNotice"),
        js_function(app_js, "loadRequests"),
        js_function(app_js, "loadRequestsList"),
        js_function(app_js, "fetchRequestsSummary"),
    ]


def test_requests_view_names_a_loading_or_failed_read_instead_of_claiming_no_traffic():
    app_js = asset("app.js")
    out = run_node(
        *_requests_stubs(app_js),
        "requestsLoad = {state:'loading', detail:''}; const loading = requestsStatusNotice();",
        # A degraded index answers HTTP 200 with no rows and an error type.
        "globalThis.fetch = async (url) => ({ok:true,json:async()=>(url.includes('summary') ? {total:0,error_type:'OperationalError'} : {entries:[],error_type:'OperationalError'})});",
        "await loadRequests(false); const degraded = requestsStatusNotice();",
        "globalThis.fetch = async () => ({ok:true,json:async()=>({entries:[],total:0})}); await loadRequests(false); const healthyEmpty = requestsStatusNotice();",
        "globalThis.fetch = async () => ({ok:false,json:async()=>({error:{message:'requests failed'}})});",
        "let failure = ''; try { await loadRequests(false); } catch (e) { failure = e.message; } const failed = requestsStatusNotice();",
        "requestsPayload = {entries:[{request_id:'r1'}]}; const staleButUseful = requestsStatusNotice();",
        "console.log(JSON.stringify({loading, degraded, healthyEmpty, failed, failure, staleButUseful}));",
    )
    result = json.loads(out)
    assert "Loading requests" in result["loading"]
    assert "Request index unavailable" in result["degraded"] and "OperationalError" in result["degraded"]
    assert result["healthyEmpty"] == ""
    assert "requests failed" in result["failed"] and result["failure"] == "requests failed"
    # Rows already on screen are not hidden by a later failed refresh.
    assert result["staleButUseful"] == ""


def test_slow_response_for_an_old_filter_cannot_overwrite_the_current_one():
    app_js = asset("app.js")
    out = run_node(
        *_requests_stubs(app_js),
        "const pending = {};",
        # Only the list read is held back, so the test controls which filter's answer lands last.
        "globalThis.fetch = (url) => url.includes('/summary') ? Promise.resolve({ok:true,json:async()=>({total:1})}) : new Promise((resolve) => { pending[url] = (body) => resolve({ok:true,json:async()=>body}); });",
        "requestFilters.q = 'a'; const oldQuery = requestQuery(); const older = loadRequests(false);",
        "requestFilters.q = 'abc'; const newQuery = requestQuery(); const newer = loadRequests(false);",
        "pending['/admin/requests?' + newQuery]({entries:[{request_id:'new'}],cursor:2}); await newer;",
        "pending['/admin/requests?' + oldQuery]({entries:[{request_id:'old'}],cursor:1}); await older;",
        "console.log(JSON.stringify({ids: requestsPayload.entries.map((e) => e.request_id), cursor: requestsCursor}));",
    )
    result = json.loads(out)
    assert result["ids"] == ["new"]
    assert result["cursor"] == 2


def test_leaving_requests_cannot_reopen_the_live_stream():
    app_js = asset("app.js")
    out = run_node(
        "let requestsLive = true; let currentView = 'routing'; let requestsCursor = null; let requestsStream = null; let requestsTimer = null; let requestsFlushTimer = null; let requestsPending = [];",
        "const document = {hidden:false}; let opened = 0; class EventSource { constructor() { opened += 1; this.addEventListener = () => {}; } close() {} }; globalThis.EventSource = EventSource;",
        "const transports = []; const setRequestsTransport = (s) => { transports.push(s); }; const requestQuery = () => 'limit=200';",
        "const stopRequestsTransport = () => {}; function startRequestsPolling() { opened += 100; }",
        js_function(app_js, "startRequestsLive"),
        # A filter change started on Requests resolves after the operator moved on.
        "startRequestsLive(); const left = {opened, last: transports.at(-1)};",
        "currentView = 'requests'; startRequestsLive();",
        "console.log(JSON.stringify({left, entered: opened}));",
    )
    result = json.loads(out)
    assert result["left"] == {"opened": 0, "last": "paused"}
    assert result["entered"] == 1


def test_filter_popover_closes_on_escape_and_when_focus_leaves():
    app_js = asset("app.js")
    out = run_node(
        "let openFilter = 'prov'; const listeners = {}; const trigger = {focused:false, focus() { this.focused = true; }};",
        "const flt = {classList:{contains:()=>openFilter === 'prov'}, contains:(node) => node === inside, querySelector:()=>trigger, addEventListener:(type, fn) => { listeners[type] = fn; }};",
        "const inside = {}; const outside = {};",
        "let closed = 0; function closeFilterPops() { closed += 1; openFilter = null; }",
        js_function(app_js, "bindFilterPopDismissal"),
        "bindFilterPopDismissal(flt);",
        # Tabbing between the popover's own controls keeps it open. So does a pointer press on an option's
        # label text: focus lands on a focusable ancestor before the click, and closing then would swallow it.
        "listeners.focusout({relatedTarget: inside}); listeners.pointerdown(); listeners.focusout({relatedTarget: outside}); const stayed = closed;",
        "listeners.keydown({key:'Tab'}); listeners.focusout({relatedTarget: outside}); const afterTab = closed;",
        "openFilter = 'prov'; listeners.keydown({key:'a'}); const otherKey = closed; listeners.keydown({key:'Escape'});",
        "console.log(JSON.stringify({stayed, afterTab, otherKey, escaped: closed, refocused: trigger.focused}));",
    )
    result = json.loads(out)
    assert result == {"stayed": 0, "afterTab": 1, "otherKey": 1, "escaped": 2, "refocused": True}


def test_audit_names_every_action_and_warns_that_undo_reverts_later_saves():
    app_js = asset("app.js")
    out = run_node(
        js_function(app_js, "auditTitle"),
        "console.log(JSON.stringify({known: auditTitle('admin.providers.remove_key'), unknown: auditTitle('admin.widgets.do_thing'), empty: auditTitle('')}));",
    )
    result = json.loads(out)
    assert result["known"] == "Removed a provider key"
    # A future server action is still readable, and never shows the dotted identifier.
    assert result["unknown"] == "widgets do thing"
    assert result["empty"] == "Admin action"

    prompts: list[str] = []
    out = run_node(
        "const prompts = []; globalThis.window = {confirm: (m) => { prompts.push(m); return false; }};",
        js_function(app_js, "rollbackProfiles"),
        "await rollbackProfiles('audit-1'); console.log(JSON.stringify(prompts));",
    )
    prompts = json.loads(out)
    assert len(prompts) == 1
    assert "every virtual model" in prompts[0] and "overwritten" in prompts[0]


def _stale_state_stubs(app_js: str, *, age_ms: int) -> list[str]:
    return [
        f"let stateLoadedAt = Date.now() - {age_ms}; let resyncs = 0; const STATE_FRESH_MS = 30000;",
        "async function resyncState() { resyncs += 1; }",
        js_function(app_js, "refreshStaleState"),
    ]


def test_entering_health_rereads_runtime_state_only_when_it_is_old():
    app_js = asset("app.js")
    stale = run_node(*_stale_state_stubs(app_js, age_ms=10 * 60 * 1000), "await refreshStaleState(); console.log(resyncs);")
    fresh = run_node(*_stale_state_stubs(app_js, age_ms=1000), "await refreshStaleState(); console.log(resyncs);")
    assert stale.strip() == "1"
    assert fresh.strip() == "0"


def test_a_tab_that_booted_during_a_benchmark_learns_when_it_ends():
    app_js = asset("app.js")
    out = run_node(
        "let running = true; const serverBenchmarkRunning = () => running; let benchmarkRunInFlight = false;",
        "let benchmarkJobCheckedAt = 0; const BENCHMARK_JOB_POLL_MS = 60000; let resyncs = 0; async function resyncState() { resyncs += 1; }",
        js_function(app_js, "refreshRunningBenchmark"),
        "await refreshRunningBenchmark(); const first = resyncs; await refreshRunningBenchmark(); const throttled = resyncs;",
        "running = false; benchmarkJobCheckedAt = 0; await refreshRunningBenchmark(); const idle = resyncs;",
        "running = true; benchmarkRunInFlight = true; benchmarkJobCheckedAt = 0; await refreshRunningBenchmark(); const local = resyncs;",
        "console.log(JSON.stringify({first, throttled, idle, local}));",
    )
    # Polled once per minute while a job is flagged, never when none is or when this tab runs it itself.
    assert json.loads(out) == {"first": 1, "throttled": 1, "idle": 1, "local": 1}

    # The marker that keeps Retest spinning must not also freeze the resync that clears it.
    out = run_node(
        "let openFilter = null; let benchmarkRunInFlight = false; const serverBenchmarkRunning = () => true;",
        "const document = {activeElement:null}; const $ = () => null;",
        js_function(app_js, "resyncWouldDisturb"),
        "console.log(resyncWouldDisturb());",
    )
    assert out.strip() == "false"


def test_background_resync_keeps_the_notice_banners_on_screen():
    # /admin/state does not carry the notices: a resync that swaps the whole state would render
    # the auth-cooldown, paid-model and license banners as hidden until the next 15 s poll.
    app_js = asset("app.js")
    out = run_node(
        "let state = {catalog:{generated_at:'A'}, notices:{auth_cooldowns:[{id:'groq@1',provider:'groq',until:9e9}]}};",
        "let resyncInFlight = false; let renderedCatalogMarker = 'A'; let seen = null;",
        "const document = {hidden:false}; const resyncWouldDisturb = () => false; const unsavedProfileIds = () => [];",
        "const setState = (next) => { state = next; }; const seedProfileDrafts = () => {};",
        "const captureScroll = () => ({y:0,els:[]}); const restoreScroll = () => {}; const render = () => { seen = state.notices; };",
        "async function fetch() { return {ok:true,json:async()=>({catalog:{generated_at:'B'}})}; }",
        js_function(app_js, "catalogMarker"),
        js_function(app_js, "resyncState"),
        "await resyncState('B'); console.log(JSON.stringify({seen, marker: renderedCatalogMarker}));",
    )
    result = json.loads(out)
    assert result["seen"]["auth_cooldowns"][0]["id"] == "groq@1"
    assert result["marker"] == "B"


def test_auth_cooldown_banner_can_be_cleared_for_every_provider_and_hides_once_expired():
    app_js = asset("app.js")
    lines = app_js.splitlines()
    declarations = [
        next(line.strip() for line in lines if line.strip().startswith(f"const {name} ="))
        for name in ("readDismissed", "writeDismissed", "AUTH_COOLDOWN_NOTICE_KEY")
    ]
    out = run_node(
        "const store = {}; globalThis.localStorage = {getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = String(v); }};",
        "const handlers = {}; const banner = {hidden: true, innerHTML: ''};",
        "const $ = (id) => id === 'authCooldownBanner' ? banner : {addEventListener: (_type, fn) => { handlers[id] = fn; }};",
        "const esc = (v) => String(v); const providerLabel = (p) => p; const formatSeconds = (s) => Math.round(s) + 's';",
        "const setSelectedProvider = () => {}; const setView = () => {}; const render = () => {};",
        *declarations,
        js_function(app_js, "renderAuthCooldownBanner"),
        "const now = Date.now() / 1000;",
        "let state = {notices:{auth_cooldowns:[{id:'groq@1',provider:'groq',until:now+600},{id:'cerebras@1',provider:'cerebras',until:now+600}]}};",
        "const shown = () => banner.hidden ? null : banner.innerHTML.split(' rejected')[0].split('>').pop();",
        "renderAuthCooldownBanner(); const first = shown();",
        "handlers.authCooldownDismissBtn(); const second = shown();",
        "handlers.authCooldownDismissBtn(); const cleared = shown(); renderAuthCooldownBanner(); const afterRender = shown();",
        "state = {notices:{auth_cooldowns:[{id:'groq@2',provider:'groq',until:now-12}]}}; renderAuthCooldownBanner(); const expired = shown();",
        "console.log(JSON.stringify({first, second, cleared, afterRender, expired}));",
    )
    assert json.loads(out) == {"first": "groq", "second": "cerebras", "cleared": None, "afterRender": None, "expired": None}


def _quota_gauges(admissions: dict) -> dict:
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    out = run_node(
        f"const {{ quotaGauge, quotaRollup }} = await import({json.dumps(lib_uri)});",
        f"const rows = {json.dumps(admissions)};",
        "const gauges = Object.fromEntries(Object.entries(rows).map(([k, v]) => [k, quotaGauge(v)]));",
        "console.log(JSON.stringify({gauges, rollup: quotaRollup(rows)}));",
    )
    return json.loads(out)


def _headers_metric(limit, remaining, **extra):
    return {"limit": limit, "remaining": remaining, "reset_in_seconds": 41, "source": "provider_headers",
            "observed_at": "2026-11-17T10:00:00+00:00", "age_seconds": 12, "scope": "provider", "fresh": True,
            "window": "minute", **extra}


def test_quota_gauge_is_numeric_only_for_a_fresh_provider_report_with_a_denominator():
    declared = {"rpm": 30, "admission_limit": 27, "headroom_percent": 10, "scope": "provider"}
    declared_requests = {"limit": 30, "remaining": None, "reset_in_seconds": None, "source": "declared",
                         "observed_at": None, "scope": "provider", "fresh": False, "window": "minute"}
    result = _quota_gauges({
        "seen": {"state": "observed", "requests": _headers_metric(30, 7), "tokens": None, "declared": declared},
        "low": {"state": "observed", "requests": _headers_metric(30, 2), "tokens": None, "declared": None},
        "out": {"state": "observed", "requests": _headers_metric(30, 0), "tokens": None, "declared": None},
        "declared": {"state": "declared_only", "requests": declared_requests, "tokens": None, "declared": declared},
        "old": {"state": "stale", "requests": _headers_metric(30, 7, fresh=False, age_seconds=1100, reset_in_seconds=None),
                "tokens": None, "declared": declared},
        "tokens_only": {"state": "observed", "requests": declared_requests,
                        "tokens": _headers_metric(6000, 5000), "declared": declared},
        "quiet": {"state": "unknown", "requests": None, "tokens": None, "declared": None},
        "absent": None,
    })
    gauges = result["gauges"]

    assert gauges["seen"]["headline"] == "7 / 30 requests left" and gauges["seen"]["bar"]["remaining"] == 7
    assert gauges["seen"]["tone"] == "ok" and gauges["low"]["tone"] == "warn"
    assert gauges["out"]["headline"] == "Exhausted — resets in 41s" and gauges["out"]["tone"] == "danger"
    # Everything below is a statement with its reason, never a number or a bar.
    assert gauges["declared"]["headline"] == "Declared limit 30/min — Ficelle keeps 10% back"
    assert gauges["declared"]["bar"] is None
    assert gauges["old"]["headline"] == "Observation is 18 min old" and gauges["old"]["bar"] is None
    assert gauges["tokens_only"]["headline"] == "Request limit not reported" and gauges["tokens_only"]["bar"] is None
    assert any("5,000 / 6,000 tokens left" in line["text"] for line in gauges["tokens_only"]["lines"])
    for name in ("quiet", "absent"):
        assert gauges[name]["headline"] == "No observation yet" and gauges[name]["bar"] is None
    assert result["rollup"] == {"observed": 4, "declared_only": 1, "stale": 1, "unknown": 2, "exhausted": ["out"], "low": ["low"]}




def test_a_quota_gauge_left_open_past_its_reset_stops_reading_as_live():
    # The page never re-reads the snapshot, so "Exhausted — resets in 41s" stayed up long after
    # the bucket had reopened.
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    exhausted = {"state": "observed", "requests": _headers_metric(30, 0), "tokens": None, "declared": None, "stale_after_seconds": 900}
    healthy = {"state": "observed", "requests": _headers_metric(30, 7, reset_in_seconds=3600), "tokens": None, "declared": None, "stale_after_seconds": 900}
    out = run_node(
        f"const {{ quotaGauge, quotaRollup }} = await import({json.dumps(lib_uri)});",
        f"const exhausted = {json.dumps(exhausted)}; const healthy = {json.dumps(healthy)};",
        "console.log(JSON.stringify({soon: quotaGauge(exhausted, 30), after: quotaGauge(exhausted, 60),"
        " aged: quotaGauge(healthy, 1000), rollup: quotaRollup({exhausted}, 60)}));",
    )
    result = json.loads(out)
    assert result["soon"]["headline"] == "Exhausted — resets in 11s"
    assert result["after"]["state"] == "stale" and result["after"]["bar"] is None
    # Past the stale horizon a bucket that still has a reset ahead is not live either.
    assert result["aged"]["state"] == "stale" and result["aged"]["bar"] is None
    assert result["rollup"]["exhausted"] == [] and result["rollup"]["stale"] == 1


def test_an_expiring_quota_gauge_is_redrawn_in_place_without_replacing_the_state():
    # The expiry timer used to call resyncState(): that replaced the saved state under untouched
    # drafts (they then read as edits) and, when it deferred to the operator, never came back.
    app_js = asset("app.js")
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    exhausted = {"state": "observed", "requests": _headers_metric(30, 0, reset_in_seconds=5), "tokens": None,
                 "declared": None, "stale_after_seconds": 900}
    out = run_node(
        f"const {{ quotaGauge, quotaRollup, agedQuotaAdmission, quotaExpiresInSeconds }} = await import({json.dumps(lib_uri)});",
        "const esc = (v) => String(v); let stateLoadedAt = 0; let currentView = 'providers';",
        f"const state = {{catalog:{{providers:{{groq:{{invokable:true}}}}}}, state:{{provider_stats:{{groq:{{admission:{json.dumps(exhausted)}}}}}}}}};",
        "const providerStats = (s) => state.state.provider_stats[s] || null; const providerLabel = (s) => s;",
        "let resyncs = 0; async function resyncState() { resyncs += 1; }",
        "const timers = []; globalThis.setTimeout = (fn, ms) => { timers.push({fn, ms}); return timers.length; }; globalThis.clearTimeout = () => {};",
        "const gauge = {dataset:{quotaSource:'groq'}, outerHTML:''}; const rollup = {outerHTML:''};",
        "globalThis.document = {querySelectorAll: (sel) => sel.startsWith('.quota-gauge') ? [gauge] : [], querySelector: (sel) => sel === '.quota-rollup' ? rollup : null};",
        "const quotaElapsedSeconds = () => (stateLoadedAt ? (Date.now() - stateLoadedAt) / 1000 : 0);",
        js_function(app_js, "quotaGaugeHTML"),
        js_function(app_js, "quotaRollupHTML"),
        "let quotaExpiryTimer = null;",
        js_function(app_js, "armQuotaExpiry"),
        js_function(app_js, "redrawQuotaGauges"),
        "stateLoadedAt = Date.now(); armQuotaExpiry(['groq']); const firstDelay = timers[0].ms;",
        "stateLoadedAt = Date.now() - 10000; timers[0].fn();",
        "console.log(JSON.stringify({firstDelay, resyncs, gauge: gauge.outerHTML, rollup: rollup.outerHTML}));",
    )
    result = json.loads(out)
    assert result["firstDelay"] == 6000
    assert result["resyncs"] == 0
    assert 'data-quota-state="stale"' in result["gauge"] and 'data-quota-source="groq"' in result["gauge"]
    assert "Exhausted" not in result["rollup"]


def test_request_row_names_the_estimate_as_prompt_plus_completion_and_makes_no_claim_for_old_rows():
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    out = run_node(
        f"const {{ requestContextMeta }} = await import({json.dumps(lib_uri)});",
        "const filtered = requestContextMeta({prompt_tokens_estimate: 200009, prompt_estimate_tokens: 195000, prompt_tokens: 190000,"
        " excluded_for_context: 1, demoted_for_context: 2, context_required_tokens: 224250, context_largest_in_pool: 1000000});",
        "const legacy = requestContextMeta({request_id: 'old'});",
        "const quiet = requestContextMeta({prompt_tokens_estimate: 40, excluded_for_context: 0, demoted_for_context: 0, context_required_tokens: 46});",
        "console.log(JSON.stringify({filtered, legacy, quiet}));",
    )
    result = json.loads(out)
    assert result["filtered"] == [
        "est. 200k tokens prompt + completion",
        "prompt est. 195k vs 190k reported",
        "needs 224k window, largest in pool 1M",
        "1 too small for prompt",
        "2 demoted for tight window",
    ]
    assert result["legacy"] == []
    # The threshold is only worth a line when the filter acted; a quiet row stays quiet.
    assert result["quiet"] == ["est. 40 tokens prompt + completion"]


def test_prompt_estimate_card_shows_an_honest_empty_state_and_the_low_tail():
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    out = run_node(
        f"const {{ contextEstimateCard }} = await import({json.dumps(lib_uri)});",
        "const empty = contextEstimateCard({n: 0, ratio_p50: null});",
        "const missing = contextEstimateCard(undefined);",
        "const full = contextEstimateCard({n: 120, ratio_p1: 0.84, ratio_p50: 1.02, ratio_p90: 1.11, underestimate_rate: 0.18});",
        "console.log(JSON.stringify({empty, missing, full}));",
    )
    result = json.loads(out)
    assert result["empty"]["v"] == "n/a" and result["missing"]["v"] == "n/a"
    assert result["full"]["v"] == "1.02×"
    assert "p1 0.84×" in result["full"]["note"] and "120 requests" in result["full"]["note"]
    assert "18% under-estimated" in result["full"]["note"]


def test_endpoint_card_names_the_api_token_only_when_the_bind_requires_it():
    # Loopback `/v1` takes any key; a wildcard or LAN bind takes only the owner token, even
    # from a client on the same machine. An unknown answer must not read as "no key needed".
    lib_uri = (ADMIN_ASSETS / "lib.js").as_uri()
    out = run_node(
        f"const {{ endpointKeyHint }} = await import({json.dumps(lib_uri)});",
        "console.log(JSON.stringify([true, false, undefined].map(endpointKeyHint)));",
    )
    loopback, exposed, unknown = json.loads(out)
    assert "access-token api" not in loopback and "any value works" in loopback
    assert "ficelle access-token api" in exposed
    assert unknown == exposed
