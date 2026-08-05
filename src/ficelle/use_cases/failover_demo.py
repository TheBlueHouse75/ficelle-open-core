"""Forced-failover demonstration: make the reroute visible on a real request.

The activation problem this solves: a provider outage is absorbed *silently*.
Cooled-down models leave the candidate pool before selection (``selection.py``), so
after the one request that detects the breakage every following request routes
around the dead provider in a single attempt and shows nothing at all. The
protection is least visible exactly when it is working hardest, and a user can run
Ficelle for two weeks, be covered the whole time, and never once see it act.

So the demo knocks a candidate out on purpose and shows what happens next.

**What is simulated, and what is not.** Exactly one thing is fabricated: the HTTP
response of the candidate that would have answered. Everything else is the
production path — the real catalog, the real candidate order, the real failure
classifier, the real failover decision, and a real completion from whichever model
actually answers. The point is defeated by faking the recovery, so it is not faked.

Two invariants this module exists to hold:

- **The demo changes nothing.** Cooldowns, success stats and route logs stay
  untouched. Benching a healthy provider right after install would be the opposite
  of a demonstration, and a simulated outage written into the real route log would
  corrupt the operator's own observability, which reads that log to report which
  providers failed.
- **The demo says what it simulated.** Every rendering marks the injected attempt,
  and the synthetic payload carries its own ``ficelle_demo_simulated_outage`` error
  code, so the label survives a screenshot, a pasted log, or a raw JSON dump.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ficelle.provider_credentials import ProviderCredentialsUnavailable
from ficelle.use_cases.provider_auth import (
    invokable_provider_sources,
    provider_key_setup_commands,
    unconfigured_provider_sources,
)


# The class name is what survives into an attempt record: `evaluate_invocation_exception`
# stores `type(exc).__name__` and drops the message, so keying on the name is the only
# way to recognise a missing key without string-matching prose. Taken from the class
# itself rather than spelled out, so a rename cannot leave this reading a dead value.
CREDENTIALS_UNAVAILABLE_ERROR_TYPE = ProviderCredentialsUnavailable.__name__

# A rate limit is the outage this product is most often hired to survive, and 429 is
# the one status every provider agrees on. `classify_failure` maps a 429 whose body
# names no upstream pool to `rate_limited`: provider-scoped and retryable, which is
# precisely the branch the demo needs to exercise.
SIMULATED_OUTAGE_STATUS = 429
SIMULATED_OUTAGE_CODE = "ficelle_demo_simulated_outage"

ModelInvoker = Callable[..., Any]
ResponseBuilder = Callable[[int, dict[str, Any]], Any]


def simulated_outage_payload(model_id: str) -> dict[str, Any]:
    """The body the knocked-out candidate answers with.

    Worded so that it stays self-describing out of context: this text is what the
    failure classifier reads, what a `--json` dump carries, and what would show up
    in a screenshot of the trace. It must never be mistakable for a real outage.

    It deliberately avoids the quota markers (`quota exceeded`, `usage limit
    exceeded`…) and the upstream-pool markers (`rate limited upstream`) that would
    steer `classify_failure` onto a different branch, so the demo classifies the
    same way every run.
    """
    return {
        "error": {
            "message": (
                f"Simulated by `ficelle demo`: rate limit reached for {model_id}. "
                "No request was sent to this provider."
            ),
            "type": "rate_limit_error",
            "code": SIMULATED_OUTAGE_CODE,
        }
    }


def forced_failure_invoker(
    invoke: ModelInvoker,
    *,
    knocked_out: Iterable[str],
    build_response: ResponseBuilder,
) -> ModelInvoker:
    """Wrap a model invoker so the named candidates answer a synthetic outage.

    Wrapping the invoker — rather than adding a fault-injection switch to the served
    request path — is what keeps this out of production: the router has no way to be
    told to fail a request, so no caller, header or config can ask it to.

    Every other candidate reaches the real invoker untouched, which is what makes
    the answer at the end of the trace a real one.
    """
    knocked_out_ids = {str(model_id) for model_id in knocked_out if str(model_id)}

    def invoker(model: dict[str, Any], body: dict[str, Any], config: dict[str, Any], **kwargs: Any) -> Any:
        model_id = str(model.get("id") or "")
        if model_id in knocked_out_ids:
            return build_response(SIMULATED_OUTAGE_STATUS, simulated_outage_payload(model_id))
        return invoke(model, body, config, **kwargs)

    return invoker


def knocked_out_candidate_ids(candidates: Sequence[dict[str, Any]], count: int) -> list[str]:
    """The first ``count`` candidate ids — the models the router would really have picked.

    Knocking out the head of the list is the honest scenario ("the model you were
    about to use just broke"); knocking out a tail candidate would demonstrate
    nothing, since it was never going to be called.

    At least one candidate is always left alive: a demo where everything fails
    proves the failure path, not the failover, and the caller can see an exhausted
    pool without simulating one.
    """
    ids = [str(model.get("id") or "") for model in candidates if str(model.get("id") or "")]
    if len(ids) <= 1:
        return []
    return ids[: max(0, min(int(count), len(ids) - 1))]


@dataclass(frozen=True)
class FailoverDemoAttempt:
    model: str
    source: str
    upstream: str
    status: Any
    reason: str
    latency_seconds: float | None
    simulated: bool
    error_type: str = ""

    @property
    def succeeded(self) -> bool:
        return self.reason == "ok"

    @property
    def credentials_unavailable(self) -> bool:
        """This candidate was never called: no key was configured for its provider.

        ``reason`` is only ``unavailable`` here — the same value a dead socket
        produces — so the failure has to be told apart by the exception type.
        """
        return self.error_type == CREDENTIALS_UNAVAILABLE_ERROR_TYPE


@dataclass(frozen=True)
class FailoverDemoResult:
    profile: str
    prompt: str
    candidate_count: int
    knocked_out: list[str]
    attempts: list[FailoverDemoAttempt]
    answer: str | None
    duration_seconds: float
    all_candidates_free: bool
    paid_fallback_allowed: bool

    @property
    def answered_by(self) -> FailoverDemoAttempt | None:
        return next((attempt for attempt in self.attempts if attempt.succeeded), None)

    @property
    def rerouted(self) -> bool:
        """A real failover happened: something was knocked out, and a later attempt answered."""
        return bool(self.knocked_out) and self.answered_by is not None

    @property
    def unkeyed_sources(self) -> list[str]:
        """Providers that turned the run down for want of a key, in attempt order.

        Empty unless *every* candidate the demo really tried was refused that way:
        one configured provider that genuinely failed makes "the pool is offline" the
        honest reading, and the demo must not answer a real outage with a lecture
        about API keys. The knocked-out candidate is excluded — its outage is
        fabricated, so it says nothing about how the install is configured.
        """
        attempted = [attempt for attempt in self.attempts if not attempt.simulated]
        if not attempted or not all(attempt.credentials_unavailable for attempt in attempted):
            return []
        return list(dict.fromkeys(attempt.source for attempt in attempted if attempt.source))

    @property
    def upstream_cost(self) -> str:
        """What the demonstrated request cost.

        Only ever `0.00` while the router refuses paid fallback and every candidate
        is a free model, and the caller is told which of those two facts it rests
        on rather than being handed a bare zero.
        """
        return "0.00" if self.all_candidates_free and not self.paid_fallback_allowed else "unknown"


def demo_attempts_from_route(
    raw_attempts: Iterable[dict[str, Any]],
    knocked_out: Iterable[str],
) -> list[FailoverDemoAttempt]:
    """Read the route's own attempt records, marking the ones the demo fabricated."""
    knocked_out_ids = {str(model_id) for model_id in knocked_out}
    attempts: list[FailoverDemoAttempt] = []
    for raw in raw_attempts:
        if not isinstance(raw, dict):
            continue
        model = str(raw.get("model") or "")
        latency = raw.get("latency_seconds")
        attempts.append(
            FailoverDemoAttempt(
                model=model,
                source=str(raw.get("source") or ""),
                upstream=str(raw.get("upstream") or ""),
                status=raw.get("status"),
                reason=str(raw.get("reason") or ""),
                latency_seconds=float(latency) if isinstance(latency, (int, float)) else None,
                simulated=model in knocked_out_ids,
                error_type=str(raw.get("error_type") or ""),
            )
        )
    return attempts


def answer_text_from_payload(payload: Any) -> str | None:
    """The assistant message of an OpenAI-shaped completion, or ``None``.

    Tolerant on purpose: the demo prints whatever the provider actually returned,
    and a body it cannot read is a reason to show no answer, never to fail the run
    after a completion has already been paid for in latency.
    """
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    # Some providers answer with the content-parts array instead of a plain string.
    if isinstance(content, list):
        parts = [part.get("text") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        joined = "".join(parts).strip()
        return joined or None
    return None


def _format_latency(latency_seconds: float | None) -> str:
    if latency_seconds is None:
        return ""
    return f" · {latency_seconds * 1000:.0f} ms"


def _truncated(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


def _key_setup_block(sources: Sequence[str], key_urls: Mapping[str, str]) -> list[str]:
    return [
        "Ficelle routes through your own provider accounts and ships no keys of its own,",
        "so it cannot call a model until at least one key is stored:",
        "",
        *(f"  {line}" for line in provider_key_setup_commands(sources, key_urls)),
        "",
    ]


def unroutable_pool_message(profile: str, auth: Mapping[str, Any], key_urls: Mapping[str, str]) -> str:
    """Why the demo has no pool to work with, said in terms the user can act on.

    A keyless install lands here rather than in the trace: catalog entries are stamped
    with their provider's ``invokable`` flag at refresh time and selection drops the
    ones that are not, so with no key configured *every* profile selects nothing. The
    generic "nothing is routable" reading — a bad profile, a stale catalog, providers
    genuinely down — is only honest once something can actually be called.
    """
    if invokable_provider_sources(auth):
        return (
            f"No model is currently routable for {profile}. Run `ficelle doctor --text` "
            "to see which providers are configured and reachable."
        )
    return "\n".join(
        [
            f"Nothing is routable for {profile}: no provider API key is configured.",
            *_key_setup_block(unconfigured_provider_sources(auth), key_urls),
            "Then run `ficelle refresh` and try again.",
        ]
    )


def _unkeyed_install_lines(sources: Sequence[str], key_urls: Mapping[str, str]) -> list[str]:
    """What to print when the run failed because the install has no provider key.

    The pool was never the problem here, and saying it was sends the one user who
    most needs a key off hunting an outage that does not exist. The catalog loads
    without credentials, so this is the first moment a fresh install is told.
    """
    return [
        "No request was sent: no provider API key is configured.",
        *_key_setup_block(sources, key_urls),
        "`ficelle doctor --text` reports which providers are configured.",
    ]


def render_failover_demo(
    result: FailoverDemoResult,
    *,
    answer_limit: int = 220,
    key_urls: Mapping[str, str] | None = None,
) -> str:
    """The human-readable trace. This is the demo's actual product.

    Reads top to bottom as a story with one line per attempt, because it is meant to
    be watched in a terminal recording, and the simulated attempt is labelled on its
    own line rather than in a footnote nobody reads.

    ``key_urls`` is the caller's key-acquisition registry (``PROVIDER_KEY_URLS``),
    used only to point an unconfigured install at the page where a key is created.
    """
    lines = [
        f"Ficelle failover demo · profile {result.profile}",
        f'Prompt: "{_truncated(result.prompt, 72)}"',
        f"Candidates in the pool: {result.candidate_count}",
        "",
    ]
    for index, attempt in enumerate(result.attempts, start=1):
        label = f"  {index}. {attempt.model} ({attempt.source})"
        if attempt.simulated:
            lines.append(f"{label} — SIMULATED OUTAGE, HTTP {attempt.status} → {attempt.reason}")
            lines.append("     (fabricated by the demo; no request was sent to this provider)")
        elif attempt.succeeded:
            lines.append(f"{label} — answered, HTTP {attempt.status}{_format_latency(attempt.latency_seconds)}")
        elif attempt.credentials_unavailable:
            # "failed, HTTP exception → unavailable" reads as a broken provider. Nothing
            # left the machine: the candidate was skipped for want of a key.
            lines.append(f"{label} — not called: no API key configured for {attempt.source}")
        else:
            lines.append(
                f"{label} — failed, HTTP {attempt.status} → {attempt.reason}"
                f"{_format_latency(attempt.latency_seconds)} (real)"
            )
    lines.append("")

    answered_by = result.answered_by
    if answered_by is not None:
        lines.append(f"Answered by {answered_by.model} in {result.duration_seconds:.1f}s.")
        if result.answer:
            lines.append(f'  "{_truncated(result.answer, answer_limit)}"')
        lines.append(f"Cost of this request: ${result.upstream_cost}")
        if result.rerouted:
            lines.append("")
            lines.append(
                "That is the whole product: the model you were about to use went down, "
                "and the request still came back."
            )
    elif result.unkeyed_sources:
        lines.extend(_unkeyed_install_lines(result.unkeyed_sources, key_urls or {}))
    else:
        lines.append(
            f"No candidate answered after {len(result.attempts)} attempt(s). "
            "The pool is exhausted or offline, not spending your money."
        )
    lines.append("")
    lines.append("Nothing was changed: no cooldown was applied, no route log was written.")
    return "\n".join(lines)


def failover_demo_payload(result: FailoverDemoResult) -> dict[str, Any]:
    """Machine-readable form, for `--json` and for the tests that assert the contract."""
    answered_by = result.answered_by
    return {
        "profile": result.profile,
        "prompt": result.prompt,
        "candidate_count": result.candidate_count,
        "knocked_out": list(result.knocked_out),
        "rerouted": result.rerouted,
        "answered_by": answered_by.model if answered_by is not None else None,
        "answer": result.answer,
        "duration_seconds": round(result.duration_seconds, 3),
        # A `--json` consumer diagnosing a silent install needs the same distinction the
        # rendered trace makes: nothing answered because nothing was configured, versus
        # nothing answered because the pool is down.
        "unkeyed_sources": result.unkeyed_sources,
        "upstream_cost": result.upstream_cost,
        "paid_fallback_allowed": result.paid_fallback_allowed,
        "all_candidates_free": result.all_candidates_free,
        "state_mutated": False,
        "attempts": [
            {
                "model": attempt.model,
                "source": attempt.source,
                "upstream": attempt.upstream,
                "status": attempt.status,
                "reason": attempt.reason,
                "error_type": attempt.error_type,
                "latency_seconds": attempt.latency_seconds,
                "simulated": attempt.simulated,
            }
            for attempt in result.attempts
        ],
    }
