from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ficelle.failures import REQUEST_REJECTION_STATUSES as FAILURE_REQUEST_REJECTION_STATUSES


class ProbeResponseValidator(Protocol):
    def validate_response(
        self,
        test_type: str,
        expected: str,
        text: str,
        payload: Any | None = None,
    ) -> tuple[bool, str]:
        ...


def record_background_job_error_in_state(
    state: dict[str, Any],
    job_name: str,
    exc: BaseException,
    *,
    state_key: str,
    safe_detail: Callable[[Any], str | None],
    now_iso: Callable[[], str],
) -> None:
    section = state.get(state_key)
    if not isinstance(section, dict):
        section = {}
    section["last_error"] = {
        "job": safe_detail(job_name),
        "type": safe_detail(type(exc).__name__),
        "message": safe_detail(str(exc)),
        "trace": [safe_detail(f"{type(exc).__name__}: {exc}")],
        "seen_at": now_iso(),
    }
    state[state_key] = section


def clear_background_job_error_in_state(state: dict[str, Any], *, state_key: str) -> bool:
    section = state.get(state_key)
    if not isinstance(section, dict) or "last_error" not in section:
        return False
    section.pop("last_error", None)
    state[state_key] = section
    return True


def probeable_capability_profiles(
    config: dict[str, Any],
    *,
    profile_ids: list[str],
    benchmark_body: Callable[[str, dict[str, Any] | None], tuple[dict[str, Any], str, str]],
) -> list[str]:
    """Return discovery profiles that have an available probe body."""
    out: list[str] = []
    for profile_id in profile_ids:
        try:
            benchmark_body(profile_id, config)
        except Exception:
            continue
        out.append(profile_id)
    return out


def model_due_capabilities(
    model: dict[str, Any],
    profile_ids: list[str],
    state: dict[str, Any],
    *,
    verified_capability_row: Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]],
    catalog_denies_capability: Callable[[str, dict[str, Any]], bool],
) -> list[str]:
    """Return probeable capabilities without a fresh verified/failed verdict.

    A capability the catalog has already ruled out is not due at all: probing it can only return the
    404 the catalog predicted. Dropping it here rather than at the probe means the model converges
    and leaves the discovery queue, instead of coming back every cycle for an answer it has — which
    is why the port is required: a caller that forgot it would still queue the model with nothing
    left to ask it.
    """
    return [
        profile_id
        for profile_id in profile_ids
        if not catalog_denies_capability(profile_id, model)
        and not verified_capability_row(profile_id, model, state)
    ]


def discovery_eligible_models(
    catalog: dict[str, Any],
    state: dict[str, Any],
    *,
    model_on_cooldown: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str | None]],
    model_is_quarantined: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Return invokable catalog models that discovery can safely probe."""
    out: list[dict[str, Any]] = []
    for model in catalog.get("models", []) if isinstance(catalog, dict) else []:
        if not isinstance(model, dict) or not model.get("invokable"):
            continue
        on_cooldown, _ = model_on_cooldown(model, state)
        if on_cooldown or model_is_quarantined(model, state):
            continue
        out.append(model)
    return out


def models_needing_discovery(
    catalog: dict[str, Any],
    state: dict[str, Any],
    profile_ids: list[str],
    *,
    model_on_cooldown: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str | None]],
    model_is_quarantined: Callable[[dict[str, Any], dict[str, Any]], bool],
    verified_capability_row: Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]],
    catalog_denies_capability: Callable[[str, dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Return eligible models with at least one due discovery capability."""
    return [
        model
        for model in discovery_eligible_models(
            catalog,
            state,
            model_on_cooldown=model_on_cooldown,
            model_is_quarantined=model_is_quarantined,
        )
        if model_due_capabilities(
            model,
            profile_ids,
            state,
            verified_capability_row=verified_capability_row,
            catalog_denies_capability=catalog_denies_capability,
        )
    ]


# The provider rejected the REQUEST we sent, not the account. A 400/422 is by definition about this
# body, so it stays a per-profile verdict unconditionally — matching the behaviour before route
# rejections were added below. Defined in `failures.py`, where the chat path reads the same statuses
# as `bad_upstream_request`; the consequences differ by path on purpose, the statuses do not.
REQUEST_REJECTION_STATUSES = FAILURE_REQUEST_REJECTION_STATUSES

# The provider has no route for this request shape. Discovery deliberately bypasses the
# per-capability shape filter, so it probes audio against text-only models and gets 404 back — a
# capability verdict, not an outage. These defer to a provider-wide block, because a 404 can also
# carry a genuine payment demand or mean the model id itself is gone (`model_not_found`); a bare
# "credit"/"billing" substring inside a URL no longer triggers that, since `classify_failure`
# strips URLs before matching the false-free markers.
ROUTE_REJECTION_STATUSES = frozenset({404, 410})

# `skip` sent nothing and cooled nothing; `verified`/`failed` are real capability verdicts and let
# the cycle continue to the next profile. `blocked` is the only sent/cooldown path and stops the
# remaining probes for that model.
ProbeVerdict = Literal["verified", "failed", "blocked", "skip"]

# Spacing between two probes of the SAME provider when it declares no `rate_limit_rpm` — a safety
# floor, not a preference: an operator raising `auto_benchmark_models_per_cycle` would otherwise
# multiply the burst without noticing. The clock starts when a probe is SENT, so a slow probe costs
# no extra wait and only a burst of fast ones is paced.
PROBE_MIN_SECONDS_BETWEEN_CALLS = 4.0

# Probing never spends a provider's whole published budget: live routing and any other client share
# it, and providers measure over sliding windows that punish pacing right at the edge. 0.75 of the
# published rate is the working margin — at OpenRouter's 20 RPM it yields exactly the 4s floor above.
PROBE_RATE_LIMIT_HEADROOM = 0.75

# A misdeclared `rate_limit_rpm` must not be able to stall discovery for good, nor to disable pacing.
PROBE_INTERVAL_BOUNDS_SECONDS = (0.5, 60.0)


def provider_probe_interval_seconds(
    config: dict[str, Any],
    source: str,
    *,
    default_seconds: float = PROBE_MIN_SECONDS_BETWEEN_CALLS,
) -> float:
    """Seconds to leave between two probes of ``source``, from its declared requests-per-minute.

    Read per call rather than captured once, so an admin editing the setting takes effect on the
    next probe. A provider that declares nothing keeps the conservative default: the floor is sized
    for the tightest free tier, which is the wrong price for a local runtime but the safe one.
    """
    providers = config.get("providers") if isinstance(config, dict) else None
    provider_cfg = (providers or {}).get(str(source or "")) if isinstance(providers, dict) else None
    raw_rpm = provider_cfg.get("rate_limit_rpm") if isinstance(provider_cfg, dict) else None
    try:
        rpm = float(raw_rpm)
    except (TypeError, ValueError):
        return default_seconds
    if rpm <= 0:
        return default_seconds
    low, high = PROBE_INTERVAL_BOUNDS_SECONDS
    return min(max(60.0 / (rpm * PROBE_RATE_LIMIT_HEADROOM), low), high)


@dataclass
class ProviderProbePacer:
    """Spaces PROBE traffic per provider, across every prober in the process.

    One pacer is shared by capability discovery and the benchmark path because they hit the same
    upstream budget: pacing them separately would let their sum exceed what either respects on its
    own — a background discovery cycle and an admin "Test candidates" click can overlap. Live
    routing is deliberately NOT paced; it would add latency to a user's request for no safety gain.
    The sharing ends at the process boundary: `ficelle canary` runs in its own process and gets its
    own budget, which would take cross-process coordination to close.

    Slots are reserved rather than merely timestamped, so two threads probing one provider queue
    behind each other instead of both waking at the same instant. Reserving happens under the lock;
    the wait itself does not, so probes of other providers are never blocked.
    """

    pause: Callable[[float], None]
    min_seconds_between_calls: float = PROBE_MIN_SECONDS_BETWEEN_CALLS
    monotonic: Callable[[], float] = time.monotonic
    _next_slot_at: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def wait_turn(self, source: str, min_seconds: float | None = None) -> None:
        key = str(source or "")
        interval = self.min_seconds_between_calls if min_seconds is None else max(0.0, min_seconds)
        with self._lock:
            now = self.monotonic()
            # `max(now, ...)` is what makes a slow probe free: once its own slot is in the past, the
            # next call proceeds immediately instead of paying a wait it already served in latency.
            slot = max(now, self._next_slot_at.get(key, now))
            self._next_slot_at[key] = slot + interval
        wait = slot - self.monotonic()
        if wait > 0:
            self.pause(wait)

# Stamped on a verdict the probe reached by route rejection rather than by reading an answer. Both
# are `failed`, but they age differently: "this model answered and got the capability wrong" is a
# property of the model, while "the route said no" can just as well be an account-level condition
# (OpenRouter's "No endpoints found matching your data policy") or a provider hiccup. The reader is
# `benchmark_result_is_aged`, which caps these at a short TTL so a fixed setting recovers in one
# discovery cycle instead of staying gated for the full verified-capability TTL.
VERDICT_BASIS_KEY = "verdict_basis"
ROUTE_REJECTION_VERDICT = "route_rejection"

# How many probes in a row have reached the verdict the same way. The basis alone cannot answer
# "have we already retried this?": each probe rewrites the row, so an attempt that declined to
# re-stamp the marker would read as "never rejected" on the next pass and hand back the short TTL
# forever — the very loop the streak exists to end.
VERDICT_STREAK_KEY = "verdict_basis_streak"
MAX_VERDICT_STREAK = 2**31 - 1


def consecutive_verdict_count(previous_row: Any, row: dict[str, Any]) -> int:
    """Continue the previous streak when the new verdict repeats it, otherwise start at 1.

    A different basis, or the same basis reached on a different `test_type`, is a different claim:
    re-probing a redesigned audio check is a fresh question, not a repeat of the old answer.
    """
    if not isinstance(previous_row, dict):
        return 1
    if previous_row.get(VERDICT_BASIS_KEY) != row.get(VERDICT_BASIS_KEY):
        return 1
    if previous_row.get("test_type") != row.get("test_type"):
        return 1
    previous_streak = previous_row.get(VERDICT_STREAK_KEY)
    if type(previous_streak) is not int or not 1 <= previous_streak <= MAX_VERDICT_STREAK:
        # Missing on pre-streak rows, or corrupt in hand-edited state. The matching basis and test
        # still prove this is at least the second consecutive verdict, so converge immediately.
        return 2
    return min(previous_streak + 1, MAX_VERDICT_STREAK)


@dataclass(frozen=True)
class CapabilityDiscoveryJob:
    benchmark_body: Callable[[str, dict[str, Any] | None], tuple[dict[str, Any], str, str]]
    invoke_model: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any]
    classify_failure: Callable[[int, str, dict[str, Any]], str]
    set_cooldown: Callable[..., None]
    record_benchmark_result: Callable[[str, dict[str, Any], dict[str, Any]], None]
    record_verified_capability: Callable[[str, dict[str, Any], dict[str, Any]], None]
    record_success: Callable[[dict[str, Any], float], None]
    extract_message_text: Callable[[Any], str]
    safe_detail: Callable[..., str]
    now_iso: Callable[[], str]
    route_blocking_reasons: frozenset[str]
    pacer: ProviderProbePacer
    registry: ProbeResponseValidator

    def _cool_and_stop(
        self,
        model: dict[str, Any],
        reason: str,
        config: dict[str, Any],
        *,
        detail: str,
        profile_id: str,
    ) -> ProbeVerdict:
        """Bench the model and stop probing it — one call so the two cannot drift apart.

        A benched model cannot answer the capabilities queued behind the one that just failed, so
        every probe fired at it is a wasted call into a limit already recorded.
        """
        self.set_cooldown(model, reason, config, detail=detail, profile_id=profile_id)
        return "blocked"

    def probe_model_capability(self, model: dict[str, Any], profile_id: str, config: dict[str, Any]) -> ProbeVerdict:
        try:
            body, test_type, expected = self.benchmark_body(profile_id, config)
        except Exception:
            return "skip"
        source = str(model.get("source") or "")
        self.pacer.wait_turn(source, provider_probe_interval_seconds(config, source))
        started = time.time()
        try:
            response = self.invoke_model(model, body, config)
        except Exception as exc:
            return self._cool_and_stop(
                model,
                "unavailable",
                config,
                detail=f"discovery {type(exc).__name__}",
                profile_id=profile_id,
            )
        result: dict[str, Any] = {
            "profile_id": profile_id,
            "model_id": model.get("id"),
            "upstream_id": model.get("upstream_id"),
            "source": model.get("source"),
            "test_type": test_type,
            "ran_at": self.now_iso(),
        }
        if not (200 <= response.status_code < 300):
            return self._record_http_capability_result(model, profile_id, config, response, result)
        try:
            payload = response.json()
            text = self.extract_message_text(payload)
            passed, message = self.registry.validate_response(test_type, expected, text, payload)
        except Exception as exc:
            return self._cool_and_stop(
                model,
                "unavailable",
                config,
                detail=f"discovery parse {type(exc).__name__}",
                profile_id=profile_id,
            )
        result.update({
            "status": "pass" if passed else "fail",
            "message": self.safe_detail(message),
            "text_preview": self.safe_detail(text, 160),
            "latency_seconds": round(time.time() - started, 3),
        })
        self.record_benchmark_result(profile_id, model, result)
        self.record_verified_capability(profile_id, model, result)
        if passed:
            self.record_success(model, round(time.time() - started, 3))
        return "verified" if passed else "failed"

    def _record_http_capability_result(
        self,
        model: dict[str, Any],
        profile_id: str,
        config: dict[str, Any],
        response: Any,
        result: dict[str, Any],
    ) -> ProbeVerdict:
        reason = self.classify_failure(response.status_code, response.text, model)
        blocking = reason in self.route_blocking_reasons
        # A rejected probe is a verdict on the capability, not on the model, so it must NOT bench the
        # model globally. Route rejections still defer to a provider-wide block: strict-zero means a
        # genuine payment demand has to quarantine even when it arrives on a probe.
        rejected_request = response.status_code in REQUEST_REJECTION_STATUSES
        rejected_route = response.status_code in ROUTE_REJECTION_STATUSES and not blocking
        if rejected_request or rejected_route:
            result.update({"status": "fail", "message": f"HTTP {response.status_code}: {reason}"})
            if rejected_route:
                # The probe never reached the model, so this verdict ages fast (see the constant).
                # Only the first one does: the writer counts repeats, the reader stops shortening
                # once the route has said no twice.
                result[VERDICT_BASIS_KEY] = ROUTE_REJECTION_VERDICT
            self.record_benchmark_result(profile_id, model, result)
            self.record_verified_capability(profile_id, model, result)
            return "failed"
        return self._cool_and_stop(
            model,
            reason,
            config,
            detail=f"discovery HTTP {response.status_code}: {reason}",
            profile_id=profile_id,
        )

    def run_auto_benchmark_cycle(
        self,
        config: dict[str, Any],
        *,
        auto_benchmark_enabled: Callable[[dict[str, Any]], bool],
        load_or_refresh_catalog: Callable[[dict[str, Any]], dict[str, Any]],
        load_state: Callable[[], dict[str, Any]],
        write_runtime_state: Callable[[dict[str, Any]], None],
        probeable_capability_profiles: Callable[[dict[str, Any]], list[str]],
        models_needing_discovery: Callable[
            [dict[str, Any], dict[str, Any], list[str]],
            list[dict[str, Any]],
        ],
        model_due_capabilities: Callable[[dict[str, Any], list[str], dict[str, Any]], list[str]],
        auto_benchmark_models_per_cycle: Callable[[dict[str, Any]], int],
        probing_is_within_budget: Callable[[dict[str, Any], dict[str, Any]], bool],
        probe_model_capability: Callable[[dict[str, Any], str, dict[str, Any]], str] | None = None,
    ) -> dict[str, Any]:
        if not auto_benchmark_enabled(config):
            return {"status": "disabled"}

        def fresh_state() -> dict[str, Any]:
            state = load_state()
            return state if isinstance(state, dict) else {}

        catalog = load_or_refresh_catalog(config)
        profile_ids = probeable_capability_profiles(config)
        model_limit = auto_benchmark_models_per_cycle(config)
        models = models_needing_discovery(catalog, fresh_state(), profile_ids)[:model_limit]
        if not models:
            return {"status": "idle", "models": 0}
        probe_one = probe_model_capability or self.probe_model_capability
        # Counted even when the probe produced no capability answer: `blocked` stopped a model,
        # `skipped` never ran. Reporting only verified/failed made a cycle that benched several
        # models look clean.
        counts = {"verified": 0, "failed": 0, "skipped": 0, "blocked": 0, "budget_capped": 0}
        for model in models:
            state = fresh_state()
            due = model_due_capabilities(model, profile_ids, state)
            # Re-asked against fresh state, not the snapshot this cycle opened with: a probe of an
            # earlier model may have cooled this one's whole provider — or its quota scope — and
            # probing it would only deepen a limit already recorded. `models_needing_discovery` is
            # the same eligibility predicate that picked these models, so there is one authority on
            # what is probeable rather than a second, narrower copy of it here.
            still_probeable = models_needing_discovery(catalog, state, profile_ids)
            if not any(other.get("id") == model.get("id") for other in still_probeable):
                counts["skipped"] += len(due)
                continue
            for index, profile_id in enumerate(due):
                # Discovery converges over days; a user's request fails now. When the share of the
                # pool's budget reserved for probing is spent, the rest of the window belongs to
                # routing. Checked per probe, on freshly loaded state, so the probes of one model
                # cannot overrun the line — a model with N due capabilities used to send all N
                # after a single test.
                if not probing_is_within_budget(fresh_state(), model):
                    # Counted apart from `skipped`: "discovery has nothing left to ask" and
                    # "discovery ran out of allowance" look identical in a single counter, and they
                    # call for opposite reactions.
                    counts["budget_capped"] += len(due) - index
                    break
                # Spacing lives in `probe_model_capability`, next to the call it protects, so a probe
                # reached any other way is paced too.
                verdict = probe_one(model, profile_id, config)
                if verdict == "verified":
                    counts["verified"] += 1
                elif verdict == "failed":
                    counts["failed"] += 1
                elif verdict == "blocked":
                    counts["blocked"] += 1
                    break
                else:
                    counts["skipped"] += 1
        state = fresh_state()
        state["auto_benchmark"] = {
            "last_run_at": self.now_iso(),
            "models": len(models),
            **counts,
        }
        write_runtime_state(state)
        return {"status": "ran", "models": len(models), **counts}
