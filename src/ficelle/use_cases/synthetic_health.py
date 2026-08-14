from __future__ import annotations

import hashlib
import re
import json
from dataclasses import dataclass, field
from typing import Any, Literal


# Corpus 2 (L3): the portable base tool fixture uses the conservative common JSON Schema
# subset (advanced dialect features moved to explicit T8 cases), the continuation validator
# splits protocol/grounding/exact-instruction, and length-exhausted tool cases become
# inconclusive instead of failed capability.
CORPUS_VERSION = "2"
# Schema 2 (L3): outcomes carry staged verdicts and a primary failure layer; summaries add
# the availability block and dual tool denominators. Readers stay compatible with schema 1
# artifacts by treating the new fields as absent, never by rewriting old runs.
SCHEMA_VERSION = 2
VERDICT_AXES = ("transport", "protocol", "semantic", "routing", "policy", "telemetry", "safety")
PASSING_AXIS_STATUSES = frozenset({"pass", "not_applicable"})

# Validator identities for the corpus hash: bump a validator's version whenever its
# assertion semantics change, so two runs disagreeing on what "pass" means never compare.
VALIDATOR_VERSIONS = {
    "tool-choice": 2,
    "tool-continuation": 2,
}

# The staged verdict rungs (L3-R1), outermost first. Each rung is meaningful only when the
# previous one held; a report must never read a missing rung as a failure.
STAGED_VERDICT_FIELDS = (
    "selection_reachable",
    "upstream_reached",
    "protocol_valid",
    "tool_semantic_valid",
    "end_user_success",
)

# Exactly one primary layer per failure row (L3-R1 / artifact contract): secondary
# dimensions may coexist, but one candidate-free 503 is never double-counted as catalog,
# HTTP, tool, and model failure simultaneously.
PRIMARY_FAILURE_LAYERS = (
    "preflight",
    "catalog",
    "selection",
    "downstream_transport",
    "router_transport",
    "upstream_transport",
    "request_compatibility",
    "provider_capacity",
    "protocol",
    "tool_semantics",
    "model_semantics",
    "safety",
    "harness",
)

CaseLane = Literal["deterministic", "live"]
CaseStatus = Literal[
    "planned",
    "pass",
    "pass_with_fallback",
    "degraded",
    "fail",
    "skipped",
    "harness_error",
    "safety_breach",
]


@dataclass(frozen=True)
class SyntheticHealthCase:
    id: str
    phase: int
    lane: CaseLane
    target_kind: str
    target_id: str
    validator: str
    timeout_class: str
    body: dict[str, Any] | None = field(default=None, repr=False)
    expected: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    model_id: str | None = None
    upstream_id: str | None = None
    tags: tuple[str, ...] = ()
    skip_reason: str | None = None

    @property
    def stream(self) -> bool:
        return bool((self.body or {}).get("stream"))

    def public_plan(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": 1,
            "phase": self.phase,
            "lane": self.lane,
            "target": {"kind": self.target_kind, "id": self.target_id},
            "source": self.source,
            "model_id": self.model_id,
            "upstream_id": self.upstream_id,
            "validator": self.validator,
            "timeout_class": self.timeout_class,
            "tags": list(self.tags),
            "skip_reason": self.skip_reason,
            "request_template_hash": request_template_hash(self.body or {}),
            "request_shape": request_shape(self.body or {}),
            "expected": self.expected,
        }


@dataclass(frozen=True)
class DeterministicFailureScenario:
    id: str
    status_code: int | None
    body: str
    expected_reason: str
    expected_retry: bool
    expected_scope: str
    upstream_model_id: str = "synthetic/model-a"
    free_access: dict[str, Any] = field(default_factory=lambda: {"eligible": True, "mode": "catalog_free"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def request_template_hash(body: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in body.items() if not str(key).startswith("__")})


def _content_shape(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"type": "text", "chars": len(content)}
    if isinstance(content, list):
        return {
            "type": "parts",
            "count": len(content),
            "part_types": [str(part.get("type") or "unknown") for part in content if isinstance(part, dict)],
        }
    if content is None:
        return {"type": "none", "chars": 0}
    return {"type": type(content).__name__}


def request_shape(body: dict[str, Any]) -> dict[str, Any]:
    """Return bounded request metadata without persisting synthetic prompt text."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    tool_names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            tool_names.append(str(name)[:80])
    core_keys = {"model", "messages", "tools", "stream"}
    optional = sorted(str(key) for key in body if key not in core_keys and not str(key).startswith("__"))
    total_chars = sum(
        len(message.get("content"))
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )
    return {
        "model": str(body.get("model") or "")[:250],
        "stream": bool(body.get("stream")),
        "message_count": len(messages),
        "message_roles": [str(message.get("role") or "")[:30] for message in messages if isinstance(message, dict)],
        "message_content": [_content_shape(message.get("content")) for message in messages if isinstance(message, dict)],
        "character_count": total_chars,
        "estimated_tokens": max(1, total_chars // 4) if total_chars else 0,
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tool_choice": body.get("tool_choice") if isinstance(body.get("tool_choice"), (str, dict)) else None,
        "response_format": body.get("response_format") if isinstance(body.get("response_format"), dict) else None,
        "optional_parameters": optional,
    }


_RUN_MARKER_PATTERN = re.compile(r"ficelle-health-[0-9a-f]{16}")


def _normalized_case_contract(value: Any, marker: str | None = None) -> str:
    """JSON dump with run-derived markers replaced, so the hash is stable across runs.

    Correlation markers are deterministic per (run_id, case_id); hashing them raw would
    make every run's corpus hash unique and break resume/baseline comparison. Both the
    case's own declared marker and the generated-marker pattern are normalized, so the
    invariant holds whatever shape a marker takes."""
    dumped = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if marker:
        dumped = dumped.replace(json.dumps(marker, ensure_ascii=False)[1:-1], "<MARKER>")
    return _RUN_MARKER_PATTERN.sub("<MARKER>", dumped)


def corpus_hash(cases: list[SyntheticHealthCase]) -> str:
    signatures: list[dict[str, Any]] = []
    for case in cases:
        plan = case.public_plan()
        signatures.append(
            {
                "id": plan["id"],
                "version": plan["version"],
                "phase": plan["phase"],
                "lane": plan["lane"],
                "target": plan["target"],
                "validator": plan["validator"],
                # The full normalized case contract (L3 / artifact contract): the request
                # template, the expected values, and the validator identity/version all
                # participate, so any fixture or assertion change changes the identity.
                "validator_version": VALIDATOR_VERSIONS.get(plan["validator"], 1),
                "request_template_hash": stable_hash(
                    _normalized_case_contract(case.body or {}, str(case.expected.get("marker") or "") or None)
                ),
                "expected": _normalized_case_contract(
                    case.expected, str(case.expected.get("marker") or "") or None
                ),
                "timeout_class": plan["timeout_class"],
                "tags": plan["tags"],
                "request_shape": plan["request_shape"],
            }
        )
    return stable_hash(signatures)


def deterministic_failure_scenarios() -> tuple[DeterministicFailureScenario, ...]:
    return (
        DeterministicFailureScenario("http-400-caller", 400, '{"error":{"message":"invalid messages shape"}}', "bad_upstream_request", False, "none"),
        DeterministicFailureScenario("http-400-model-contract", 400, '{"error":{"message":"reasoning_content must be passed back to the API"}}', "bad_upstream_contract", True, "none"),
        DeterministicFailureScenario("http-401", 401, '{"error":{"message":"invalid api key"}}', "auth_or_credit", True, "provider"),
        DeterministicFailureScenario("http-402", 402, '{"error":{"message":"payment required"}}', "billing_or_paid", True, "quarantine"),
        DeterministicFailureScenario("http-403", 403, '{"error":{"message":"access denied"}}', "auth_or_credit", True, "provider"),
        DeterministicFailureScenario("http-403-model-entitlement", 403, '{"error":{"message":"synthetic/model-a must enable access"}}', "model_not_found", True, "quarantine"),
        DeterministicFailureScenario("http-404-model", 404, '{"error":{"message":"synthetic/model-a not found"}}', "model_not_found", True, "quarantine"),
        DeterministicFailureScenario("http-410-model", 410, '{"error":{"message":"synthetic/model-a is gone"}}', "model_not_found", True, "quarantine"),
        DeterministicFailureScenario("http-413", 413, '{"error":{"message":"request too large"}}', "request_too_large", True, "model"),
        DeterministicFailureScenario("http-422-caller", 422, '{"error":{"message":"invalid tool history"}}', "bad_upstream_request", False, "none"),
        DeterministicFailureScenario("http-422-model-contract", 422, '{"error":{"message":"reasoning_content must be passed back to the API"}}', "bad_upstream_contract", True, "none"),
        DeterministicFailureScenario("http-429-provider", 429, '{"error":{"message":"account rate limit exceeded"}}', "rate_limited", True, "provider"),
        DeterministicFailureScenario("http-429-upstream", 429, '{"error":{"message":"synthetic/model-a rate-limited upstream"}}', "rate_limited_upstream", True, "model"),
        DeterministicFailureScenario("http-500", 500, '{"error":{"message":"internal error"}}', "server_error", True, "model"),
        DeterministicFailureScenario("http-502", 502, '{"error":{"message":"bad gateway"}}', "server_error", True, "model"),
        DeterministicFailureScenario("deceptive-200-auth", None, '{"error":{"type":"authentication_error","message":"invalid api key"}}', "auth_or_credit", True, "provider"),
        DeterministicFailureScenario(
            "quota-free-exhausted",
            429,
            '{"error":{"message":"free tier quota exhausted"}}',
            "quota_exhausted",
            True,
            "quota",
            free_access={"eligible": True, "mode": "quota_free"},
        ),
        DeterministicFailureScenario(
            "quota-free-zero",
            429,
            '{"error":{"message":"free tier quota exhausted, limit: 0"}}',
            "no_free_quota",
            True,
            "quarantine",
            free_access={"eligible": True, "mode": "quota_free"},
        ),
    )


def expected_policy_sections(policy: Any) -> set[str]:
    """Every state section this policy writes, not only its primary scope.

    A policy may write several sections at once (billing_or_paid: quarantine AND a model
    cooldown, from CooldownPolicy's model_cooldown default) — a checker expecting only the
    primary scope misreads its own contract as a scope mismatch."""
    sections: set[str] = set()
    if getattr(policy, "quarantine", None) is not None:
        sections.add("quarantine")
    if bool(getattr(policy, "quota_cooldown", False)):
        sections.add("quota_cooldowns")
    if bool(getattr(policy, "provider_cooldown", False)):
        sections.add("provider_cooldowns")
    if bool(getattr(policy, "model_cooldown", False)):
        sections.add("cooldowns")
    return sections


def expected_policy_scope(policy: Any) -> str:
    if getattr(policy, "quarantine", None) is not None:
        return "quarantine"
    if bool(getattr(policy, "quota_cooldown", False)):
        return "quota"
    if bool(getattr(policy, "provider_cooldown", False)):
        return "provider"
    if bool(getattr(policy, "model_cooldown", False)):
        return "model"
    return "none"


def aggregate_case_status(
    axes: dict[str, dict[str, Any]],
    *,
    fallback_observed: bool = False,
    skipped: bool = False,
    harness_error: bool = False,
) -> CaseStatus:
    if skipped:
        return "skipped"
    if axes.get("safety", {}).get("status") == "fail":
        return "safety_breach"
    if harness_error:
        return "harness_error"
    required = [axis for name, axis in axes.items() if name in VERDICT_AXES]
    if any(axis.get("status") == "fail" for axis in required):
        user_axes = ("transport", "protocol", "semantic")
        if all(axes.get(name, {}).get("status") in PASSING_AXIS_STATUSES for name in user_axes):
            return "degraded"
        return "fail"
    if fallback_observed:
        return "pass_with_fallback"
    return "pass"


def describe_build(identity: dict[str, Any]) -> str:
    version = str(identity.get("ficelle_version") or "unknown version")
    package_dir = str(identity.get("package_dir") or "unknown path")
    content_hash = identity.get("runtime_content_hash")
    short_hash = str(content_hash)[:12] if content_hash else "no content hash"
    return f"{version} at {package_dir} ({short_hash})"


def unverified_build_verdict(reason: str) -> dict[str, Any]:
    """The third verdict state: nothing answered, so nothing was compared.
    ``match: None`` is neither a pass nor the fail-closed mismatch below."""
    return {"match": None, "diagnostic": reason}


def build_identity_match(harness: dict[str, Any], service: Any) -> dict[str, Any]:
    """Compare the harness build to the build the live service reports (fail closed).

    The harness identity proves nothing about the service on the loopback port, and a
    shared ``ficelle_version`` proves nothing either (see ``build_identity.py``): only an
    equal runtime content hash counts as a match, and a missing identity on either side
    is a mismatch, never benefit of the doubt.
    """
    if not isinstance(service, dict):
        return {
            "match": False,
            "diagnostic": (
                "build mismatch: the service did not report a build identity "
                "(a Ficelle build predating the build guard, or a foreign process) — "
                "restart the service on the current build"
            ),
        }
    harness_hash = harness.get("runtime_content_hash")
    service_hash = service.get("runtime_content_hash")
    if not harness_hash or not service_hash:
        missing = "harness" if not harness_hash else "service"
        return {
            "match": False,
            "diagnostic": f"build mismatch: the {missing} build identity has no runtime content hash",
        }
    if harness_hash != service_hash:
        return {
            "match": False,
            "diagnostic": (
                f"build mismatch: harness runs {describe_build(harness)}, "
                f"service runs {describe_build(service)}"
            ),
        }
    return {"match": True, "diagnostic": None}


def validate_unique_case_ids(cases: list[SyntheticHealthCase]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.id in seen:
            duplicates.add(case.id)
        seen.add(case.id)
    if duplicates:
        raise ValueError(f"duplicate synthetic-health case ids: {', '.join(sorted(duplicates))}")
