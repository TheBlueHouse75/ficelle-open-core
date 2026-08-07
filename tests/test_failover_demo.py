"""Tests for the forced-failover demo.

Two things are worth more than the rest here and are asserted directly:

- the reroute the demo shows is **real** — only the knocked-out candidate is
  fabricated, and the model that answers is genuinely invoked;
- the demo **writes nothing** — no cooldown, no success stat, no route log. A demo
  that benched a healthy provider, or filed a fabricated outage in the history the
  admin dashboard reads back, would corrupt the product it is advertising.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

import ficelle.router as router
from ficelle.failures import classify_failure as classify_failure_result
from ficelle.use_cases.chat_completion import (
    ChatCompletionAttemptPlan,
    ChatCompletionAttemptPorts,
    ChatCompletionRouter,
    CompressionRoutePlan,
)
from ficelle.provider_credentials import ProviderCredentialsUnavailable
from ficelle.use_cases.failover_demo import (
    CREDENTIALS_UNAVAILABLE_ERROR_TYPE,
    SIMULATED_OUTAGE_CODE,
    SIMULATED_OUTAGE_STATUS,
    FailoverDemoAttempt,
    FailoverDemoResult,
    answer_text_from_payload,
    demo_attempts_from_route,
    failover_demo_payload,
    forced_failure_invoker,
    knocked_out_candidate_ids,
    render_failover_demo,
    simulated_outage_payload,
    unroutable_pool_message,
)


def free_model(model_id: str, source: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "upstream_id": model_id.split("/", 1)[-1],
        "source": source,
        "invokable": True,
        "pricing": {"prompt": "0", "completion": "0"},
    }


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")
        self.text = json.dumps(payload)
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def answer_response(text: str) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


# --- the synthetic outage -------------------------------------------------------


def test_simulated_outage_payload_labels_itself_as_fabricated():
    payload = simulated_outage_payload("openrouter/some-model")

    assert payload["error"]["code"] == SIMULATED_OUTAGE_CODE
    message = payload["error"]["message"]
    assert "Simulated by `ficelle demo`" in message
    assert "openrouter/some-model" in message
    # The label has to survive being read out of context, which is the whole reason
    # it sits in the body rather than only in the rendered trace.
    assert "No request was sent" in message


def test_simulated_outage_classifies_as_a_retryable_provider_rate_limit():
    """The demo rests on this: a terminal reason would stop the loop instead of failing over."""
    payload = simulated_outage_payload("openrouter/some-model")

    reason = classify_failure_result(
        SIMULATED_OUTAGE_STATUS,
        json.dumps(payload),
        upstream_model_id="some-model",
    )

    assert reason == "rate_limited"


# --- the invoker wrapper --------------------------------------------------------


def test_forced_failure_invoker_fabricates_only_the_knocked_out_candidate():
    real_calls: list[str] = []

    def real_invoke(model, body, config, **kwargs):
        real_calls.append(str(model["id"]))
        return answer_response("real answer")

    invoke = forced_failure_invoker(
        real_invoke,
        knocked_out=["a/one"],
        build_response=lambda status, payload: FakeResponse(status, payload),
    )

    knocked = invoke(free_model("a/one", "a"), {}, {})
    healthy = invoke(free_model("b/two", "b"), {}, {})

    assert knocked.status_code == SIMULATED_OUTAGE_STATUS
    assert knocked.json()["error"]["code"] == SIMULATED_OUTAGE_CODE
    assert healthy.json()["choices"][0]["message"]["content"] == "real answer"
    # The knocked-out provider is never contacted; the healthy one really is.
    assert real_calls == ["b/two"]


def test_forced_failure_invoker_passes_the_real_arguments_through():
    seen: dict[str, Any] = {}

    def real_invoke(model, body, config, **kwargs):
        seen.update({"model": model, "body": body, "config": config, "kwargs": kwargs})
        return answer_response("ok")

    invoke = forced_failure_invoker(real_invoke, knocked_out=[], build_response=lambda *_: None)
    model = free_model("b/two", "b")
    invoke(model, {"messages": []}, {"cfg": True}, timeout_seconds=3)

    assert seen["model"] is model
    assert seen["body"] == {"messages": []}
    assert seen["config"] == {"cfg": True}
    assert seen["kwargs"] == {"timeout_seconds": 3}


# --- choosing what to knock out -------------------------------------------------


def test_knocked_out_candidates_take_the_head_and_always_leave_one_alive():
    candidates = [free_model("a/one", "a"), free_model("b/two", "b"), free_model("c/three", "c")]

    assert knocked_out_candidate_ids(candidates, 1) == ["a/one"]
    assert knocked_out_candidate_ids(candidates, 2) == ["a/one", "b/two"]
    # Asking for everything still leaves a candidate: a demo where nothing answers
    # would prove the failure path, not the failover.
    assert knocked_out_candidate_ids(candidates, 99) == ["a/one", "b/two"]


def test_knocked_out_candidates_refuses_a_pool_too_small_to_fail_over():
    assert knocked_out_candidate_ids([free_model("a/one", "a")], 1) == []
    assert knocked_out_candidate_ids([], 1) == []


# --- reading the answer ---------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"choices": [{"message": {"content": "  hello  "}}]}, "hello"),
        ({"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}, "ab"),
        ({"choices": [{"message": {"content": ""}}]}, None),
        ({"choices": []}, None),
        ({"choices": [{}]}, None),
        ({}, None),
        ("not a payload", None),
    ],
)
def test_answer_text_reads_what_providers_actually_return(payload, expected):
    assert answer_text_from_payload(payload) == expected


# --- rendering ------------------------------------------------------------------


def demo_result(**overrides: Any) -> FailoverDemoResult:
    defaults: dict[str, Any] = {
        "profile": "ficelle/auto-tools",
        "prompt": "In one sentence, what is a rate limit?",
        "candidate_count": 4,
        "knocked_out": ["a/one"],
        "attempts": [
            FailoverDemoAttempt("a/one", "a", "one", 429, "rate_limited", 0.002, True),
            FailoverDemoAttempt("b/two", "b", "two", 200, "ok", 0.85, False),
        ],
        "answer": "A cap on how many requests you may send in a window.",
        "duration_seconds": 1.2,
        "all_candidates_free": True,
        "paid_fallback_allowed": False,
    }
    defaults.update(overrides)
    return FailoverDemoResult(**defaults)


def test_render_marks_the_simulated_attempt_and_credits_the_real_one():
    text = render_failover_demo(demo_result())

    assert "SIMULATED OUTAGE" in text
    assert "no request was sent to this provider" in text
    assert "Answered by b/two" in text
    assert "A cap on how many requests" in text
    assert "Cost of this request: $0.00" in text
    # The reader must be told the demo left their routing state alone.
    assert "no cooldown was applied" in text


def test_render_reports_an_exhausted_pool_without_claiming_a_reroute():
    text = render_failover_demo(
        demo_result(
            attempts=[FailoverDemoAttempt("a/one", "a", "one", 429, "rate_limited", 0.002, True)],
            answer=None,
        )
    )

    assert "No candidate answered" in text
    assert "not spending your money" in text
    assert "Answered by" not in text


def unkeyed_attempt(model: str, source: str) -> FailoverDemoAttempt:
    """What the router records for a candidate whose provider has no key configured."""
    return FailoverDemoAttempt(
        model,
        source,
        model.split("/", 1)[-1],
        "exception",
        "unavailable",
        0.0,
        False,
        CREDENTIALS_UNAVAILABLE_ERROR_TYPE,
    )


def test_render_blames_the_missing_key_not_the_pool_on_a_keyless_install():
    """The old text sent a user with no API key hunting a provider outage."""
    text = render_failover_demo(
        demo_result(
            attempts=[
                FailoverDemoAttempt("a/one", "a", "one", 429, "rate_limited", 0.002, True),
                unkeyed_attempt("b/two", "b"),
            ],
            answer=None,
        ),
        key_urls={"b": "https://b.example/keys"},
    )

    assert "no provider API key is configured" in text
    assert "ficelle set-key b  # create a key at https://b.example/keys" in text
    # The candidate was skipped, not tried: calling that a failure is the same lie.
    assert "not called: no API key configured for b" in text
    assert "The pool is exhausted or offline" not in text


def test_render_keeps_the_exhausted_pool_message_when_a_provider_really_answered_badly():
    """One configured provider that genuinely failed makes the outage reading the honest one."""
    text = render_failover_demo(
        demo_result(
            attempts=[
                FailoverDemoAttempt("a/one", "a", "one", 429, "rate_limited", 0.002, True),
                unkeyed_attempt("b/two", "b"),
                FailoverDemoAttempt("c/three", "c", "three", 503, "server_error", 0.4, False),
            ],
            answer=None,
        ),
        key_urls={"b": "https://b.example/keys"},
    )

    assert "The pool is exhausted or offline" in text
    assert "no provider API key is configured" not in text


def test_unkeyed_sources_ignores_the_fabricated_outage():
    """The knocked-out candidate is never contacted, so it proves nothing about the keys."""
    result = demo_result(
        attempts=[
            FailoverDemoAttempt("a/one", "a", "one", 429, "rate_limited", 0.002, True),
            unkeyed_attempt("b/two", "b"),
            unkeyed_attempt("b/three", "b"),
        ],
        answer=None,
    )

    assert result.unkeyed_sources == ["b"]
    assert failover_demo_payload(result)["unkeyed_sources"] == ["b"]


def test_a_run_that_answered_is_never_reported_as_unkeyed():
    assert demo_result().unkeyed_sources == []


def test_an_empty_pool_is_explained_by_the_missing_key_when_there_is_one():
    message = unroutable_pool_message(
        "ficelle/auto-tools",
        {"openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY"}},
        {"openrouter": "https://openrouter.example/keys"},
    )

    assert "Nothing is routable for ficelle/auto-tools: no provider API key is configured." in message
    assert "ficelle set-key openrouter  # create a key at https://openrouter.example/keys" in message


def test_an_empty_pool_names_the_key_that_is_stored_and_still_unusable():
    """Selection drops these models too, so the demo has to account for them — but not
    under "no API key is configured", and not behind a `set-key` line for a key it has."""
    message = unroutable_pool_message(
        "ficelle/auto-tools",
        {
            "openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY", "key_source": None},
            "mistral": {
                "invokable": False,
                "reason": "missing base_url for provider mistral",
                "key_source": "keychain",
            },
        },
        {"openrouter": "https://openrouter.example/keys", "mistral": "https://mistral.example/keys"},
    )

    assert "Nothing is routable for ficelle/auto-tools: no configured provider is usable." in message
    assert "ficelle set-key openrouter  # create a key at https://openrouter.example/keys" in message
    assert "ficelle set-key mistral" not in message
    assert "  mistral  # missing base_url for provider mistral" in message
    assert "Then run `ficelle refresh` and try again." in message


def test_an_empty_pool_drops_the_set_key_block_when_there_is_nothing_to_store():
    message = unroutable_pool_message(
        "ficelle/auto-tools",
        {
            "mistral": {
                "invokable": False,
                "reason": "missing base_url for provider mistral",
                "key_source": "keychain",
            }
        },
        {"mistral": "https://mistral.example/keys"},
    )

    assert "until at least one key is stored" not in message
    assert "  mistral  # missing base_url for provider mistral" in message


def test_an_empty_pool_stays_a_routing_problem_once_a_provider_is_configured():
    message = unroutable_pool_message(
        "ficelle/auto-tools",
        {"openrouter": {"invokable": True, "reason": "configured"}, "nous": {"invokable": False}},
        {"openrouter": "https://openrouter.example/keys"},
    )

    assert message == (
        "No model is currently routable for ficelle/auto-tools. Run `ficelle doctor --text` "
        "to see which providers are configured and reachable."
    )


def test_cost_is_only_claimed_when_the_router_could_not_have_paid():
    assert demo_result().upstream_cost == "0.00"
    assert demo_result(paid_fallback_allowed=True).upstream_cost == "unknown"
    assert demo_result(all_candidates_free=False).upstream_cost == "unknown"


def test_payload_carries_the_machine_readable_contract():
    payload = failover_demo_payload(demo_result())

    assert payload["rerouted"] is True
    assert payload["answered_by"] == "b/two"
    assert payload["knocked_out"] == ["a/one"]
    assert payload["state_mutated"] is False
    assert [attempt["simulated"] for attempt in payload["attempts"]] == [True, False]


def test_attempts_from_route_marks_only_the_knocked_out_models():
    attempts = demo_attempts_from_route(
        [
            {"model": "a/one", "source": "a", "upstream": "one", "status": 429, "reason": "rate_limited", "latency_seconds": 0.1},
            {"model": "b/two", "source": "b", "upstream": "two", "status": 200, "reason": "ok", "latency_seconds": 0.5},
            "not an attempt",
        ],
        ["a/one"],
    )

    assert [(attempt.model, attempt.simulated) for attempt in attempts] == [("a/one", True), ("b/two", False)]
    assert attempts[1].succeeded is True


# --- the reroute, through the real routing path ---------------------------------


def test_the_production_router_really_fails_over_off_the_simulated_outage():
    """End to end on the real `ChatCompletionRouter`: knock one out, the next answers."""
    candidates = [free_model("a/one", "a"), free_model("b/two", "b")]
    invoked: list[str] = []
    telemetry: list[Any] = []
    cooldowns: list[Any] = []

    def real_invoke(model, body, config, **kwargs):
        invoked.append(str(model["id"]))
        return answer_response("Free models break; this one did not.")

    demo_router = ChatCompletionRouter(
        config={},
        load_catalog=lambda _config: {"models": candidates},
        select_candidates=lambda *_args, **_kwargs: candidates,
        is_fusion_profile=lambda _model: False,
        is_virtual_model=lambda _model: True,
        response_headers=lambda request_id, requested_model: {},
        record_last_route=lambda *_args, **_kwargs: None,
        write_route_log=lambda *_args, **_kwargs: None,
        max_attempts_for_request=lambda _model, _config, count: count,
        prepare_compression_route_body=lambda body, _config: CompressionRoutePlan(body=body, metadata=None),
        now=lambda: 100.0,
    )

    def ports(_plan: ChatCompletionAttemptPlan) -> ChatCompletionAttemptPorts:
        return ChatCompletionAttemptPorts(
            invoke_model=forced_failure_invoker(
                real_invoke,
                knocked_out=["a/one"],
                build_response=lambda status, payload: FakeResponse(status, payload),
            ),
            apply_cooldown=cooldowns.append,
            record_success=lambda _model, _latency: None,
            resolve_competence=lambda _profile, _model: "verified",
            record_telemetry=telemetry.append,
            record_success_with_telemetry=lambda _model, _latency, row: telemetry.append(row),
            stream_response=lambda *_args: pytest.fail("the demo never streams"),
            detect_success_error=lambda *_args: None,
            has_deliverable=lambda _payload: True,
            classify_failure=lambda status, text, model=None, **_kwargs: classify_failure_result(
                status, text, upstream_model_id=str((model or {}).get("upstream_id") or "")
            ),
            is_timeout_exception=lambda _exc: False,
            build_success_headers=lambda *_args, **_kwargs: {},
            build_failure_error=lambda *_args, **_kwargs: {},
            build_failure_headers=lambda *_args, **_kwargs: {},
        )

    result = demo_router.handle(
        {"model": "ficelle/auto-tools", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        request_id="demo-req",
        request_started=99.0,
        ports_factory=ports,
    )

    # The knocked-out provider was never contacted; the one that answered really was.
    assert invoked == ["b/two"]
    assert result.attempt_result is not None
    assert result.attempt_result.outcome == "non_streaming_success"

    attempts = demo_attempts_from_route(telemetry[0].last_route.attempts, ["a/one"])
    assert [(a.model, a.reason, a.simulated) for a in attempts] == [
        ("a/one", "rate_limited", True),
        ("b/two", "ok", False),
    ]
    # The real router asked for a cooldown on the fabricated failure — which is exactly
    # why `run_failover_demo` refuses to carry that call through to the state store.
    assert [request.model["id"] for request in cooldowns] == ["a/one"]


def test_a_missing_key_reaches_the_demo_as_a_recognisable_attempt():
    """The signal the renderer keys on has to survive the real attempt recorder.

    `evaluate_invocation_exception` keeps `error_type` and drops the message, so this
    is the test that would catch the day the router goes back to raising a bare
    RuntimeError and the demo silently reverts to blaming the pool.
    """
    candidates = [free_model("a/one", "a"), free_model("b/two", "b")]
    telemetry: list[Any] = []

    def unkeyed_invoke(model, body, config, **kwargs):
        raise ProviderCredentialsUnavailable("credentials unavailable: missing B_API_KEY")

    demo_router = ChatCompletionRouter(
        config={},
        load_catalog=lambda _config: {"models": candidates},
        select_candidates=lambda *_args, **_kwargs: candidates,
        is_fusion_profile=lambda _model: False,
        is_virtual_model=lambda _model: True,
        response_headers=lambda request_id, requested_model: {},
        record_last_route=lambda *_args, **_kwargs: None,
        write_route_log=lambda *_args, **_kwargs: None,
        max_attempts_for_request=lambda _model, _config, count: count,
        prepare_compression_route_body=lambda body, _config: CompressionRoutePlan(body=body, metadata=None),
        now=lambda: 100.0,
    )

    def ports(_plan: ChatCompletionAttemptPlan) -> ChatCompletionAttemptPorts:
        return ChatCompletionAttemptPorts(
            invoke_model=forced_failure_invoker(
                unkeyed_invoke,
                knocked_out=["a/one"],
                build_response=lambda status, payload: FakeResponse(status, payload),
            ),
            apply_cooldown=lambda _cooldown: None,
            record_success=lambda _model, _latency: None,
            resolve_competence=lambda _profile, _model: "verified",
            record_telemetry=telemetry.append,
            record_success_with_telemetry=lambda _model, _latency, row: telemetry.append(row),
            stream_response=lambda *_args: pytest.fail("the demo never streams"),
            detect_success_error=lambda *_args: None,
            has_deliverable=lambda _payload: True,
            classify_failure=lambda status, text, model=None, **_kwargs: classify_failure_result(
                status, text, upstream_model_id=str((model or {}).get("upstream_id") or "")
            ),
            is_timeout_exception=lambda _exc: False,
            build_success_headers=lambda *_args, **_kwargs: {},
            build_failure_error=lambda *_args, **_kwargs: {},
            build_failure_headers=lambda *_args, **_kwargs: {},
        )

    demo_router.handle(
        {"model": "ficelle/auto-tools", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        request_id="demo-req",
        request_started=99.0,
        ports_factory=ports,
    )

    attempts = demo_attempts_from_route(telemetry[0].last_route.attempts, ["a/one"])
    assert [(a.model, a.credentials_unavailable) for a in attempts] == [("a/one", False), ("b/two", True)]
    result = demo_result(attempts=attempts, answer=None)
    assert result.unkeyed_sources == ["b"]


# --- the wiring in router.py ----------------------------------------------------


@pytest.fixture
def demo_wiring(monkeypatch):
    """Point `run_failover_demo` at a two-model pool with no network and no state."""
    candidates = [free_model("a/one", "a"), free_model("b/two", "b")]
    invoked: list[str] = []

    monkeypatch.setattr(router.license_ops, "is_entitled", lambda: False)
    monkeypatch.setattr(router, "effective_runtime_config", lambda config, entitled=None: dict(config))
    monkeypatch.setattr(router, "load_or_refresh_catalog", lambda config, *a, **k: {"models": candidates})
    monkeypatch.setattr(router, "select_models", lambda *a, **k: list(candidates))

    def real_invoke(model, body, config, **kwargs):
        invoked.append(str(model["id"]))
        return answer_response("A cap on requests per window.")

    monkeypatch.setattr(router, "invoke_model", real_invoke)
    # Any of these firing means the demo mutated the operator's routing state.
    monkeypatch.setattr(router, "set_cooldown", lambda *a, **k: pytest.fail("the demo must not apply cooldowns"))
    monkeypatch.setattr(router, "record_success", lambda *a, **k: pytest.fail("the demo must not record success"))
    monkeypatch.setattr(router, "record_last_route", lambda *a, **k: pytest.fail("the demo must not record the route"))
    monkeypatch.setattr(router, "write_route_log", lambda *a, **k: pytest.fail("the demo must not write the route log"))
    return invoked


def test_run_failover_demo_reroutes_without_touching_state(demo_wiring):
    result = router.run_failover_demo({"allow_paid_fallback": False}, prompt="what is a rate limit?")

    assert result.rerouted is True
    assert result.knocked_out == ["a/one"]
    assert demo_wiring == ["b/two"], "only the surviving candidate may be invoked for real"
    answered_by = result.answered_by
    assert answered_by is not None and answered_by.model == "b/two"
    assert result.answer == "A cap on requests per window."
    assert result.upstream_cost == "0.00"
    assert [(a.model, a.simulated) for a in result.attempts] == [("a/one", True), ("b/two", False)]


def test_run_failover_demo_refuses_a_model_with_no_pool_behind_it(demo_wiring):
    with pytest.raises(ValueError, match="not a routing profile"):
        router.run_failover_demo({}, profile="openrouter/some-model")


def test_run_failover_demo_refuses_a_pool_it_cannot_fail_over_in(monkeypatch, demo_wiring):
    monkeypatch.setattr(router, "select_models", lambda *a, **k: [free_model("a/one", "a")])

    with pytest.raises(RuntimeError, match="nothing to fail over to"):
        router.run_failover_demo({})


def test_run_failover_demo_reports_an_empty_pool_instead_of_faking_one(monkeypatch, demo_wiring):
    monkeypatch.setattr(router, "select_models", lambda *a, **k: [])
    monkeypatch.setattr(router, "auth_status", lambda _config: {"a": {"invokable": True, "reason": "configured"}})

    with pytest.raises(RuntimeError, match="No model is currently routable"):
        router.run_failover_demo({})


def test_run_failover_demo_names_the_missing_key_when_nothing_can_be_called(monkeypatch, demo_wiring):
    """This is what a keyless install actually hits, and it used to describe an empty pool.

    Catalog entries carry their provider's `invokable` flag and selection drops the
    ones that are not, so with no key every profile selects nothing and the demo
    refuses before it can trace anything.
    """
    monkeypatch.setattr(router, "select_models", lambda *a, **k: [])
    monkeypatch.setattr(
        router,
        "auth_status",
        lambda _config: {"openrouter": {"invokable": False, "reason": "missing OPENROUTER_API_KEY"}},
    )

    with pytest.raises(RuntimeError) as refusal:
        router.run_failover_demo({})

    message = str(refusal.value)
    assert "no provider API key is configured" in message
    assert f"ficelle set-key openrouter  # create a key at {router.PROVIDER_KEY_URLS['openrouter']}" in message
    assert "Then run `ficelle refresh` and try again." in message


def test_simulated_outage_response_is_a_real_transport_object():
    response = router.simulated_outage_response(SIMULATED_OUTAGE_STATUS, simulated_outage_payload("a/one"))

    assert response.status_code == SIMULATED_OUTAGE_STATUS
    assert response.json()["error"]["code"] == SIMULATED_OUTAGE_CODE
    assert "Simulated by" in response.text


# --- the command ----------------------------------------------------------------


def test_demo_command_prints_the_trace_and_exits_clean(monkeypatch, capsys):
    import ficelle.cli as cli

    monkeypatch.setattr(cli.router, "run_failover_demo", lambda *a, **k: demo_result())

    assert cli.main(["demo"]) == 0
    printed = capsys.readouterr().out
    assert "SIMULATED OUTAGE" in printed
    assert "Answered by b/two" in printed


def test_demo_command_json_is_parseable(monkeypatch, capsys):
    import ficelle.cli as cli

    monkeypatch.setattr(cli.router, "run_failover_demo", lambda *a, **k: demo_result())

    assert cli.main(["demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rerouted"] is True
    assert payload["state_mutated"] is False


def test_demo_command_forwards_its_options(monkeypatch):
    import ficelle.cli as cli

    seen: dict[str, Any] = {}

    def capture(config, *, profile=None, prompt=None, knock_out=1):
        seen.update({"profile": profile, "prompt": prompt, "knock_out": knock_out})
        return demo_result()

    monkeypatch.setattr(cli.router, "run_failover_demo", capture)
    cli.main(["demo", "--profile", "ficelle/auto-json", "--prompt", "hi", "--knock-out", "2"])

    assert seen == {"profile": "ficelle/auto-json", "prompt": "hi", "knock_out": 2}


def test_demo_command_reports_a_refusal_without_a_traceback(monkeypatch, capsys):
    import ficelle.cli as cli

    def refuse(*_args, **_kwargs):
        raise RuntimeError("No model is currently routable for ficelle/auto-tools.")

    monkeypatch.setattr(cli.router, "run_failover_demo", refuse)

    assert cli.main(["demo"]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "No model is currently routable for ficelle/auto-tools."
    assert "Traceback" not in captured.err


def test_demo_command_exits_non_zero_when_nothing_answered(monkeypatch, capsys):
    import ficelle.cli as cli

    monkeypatch.setattr(
        cli.router,
        "run_failover_demo",
        lambda *a, **k: demo_result(
            attempts=[FailoverDemoAttempt("a/one", "a", "one", 429, "rate_limited", 0.002, True)],
            answer=None,
        ),
    )

    assert cli.main(["demo"]) == 1
    assert "No candidate answered" in capsys.readouterr().out


def test_demo_command_points_an_unkeyed_install_at_the_real_key_page(monkeypatch, capsys):
    """The URL comes from the shared registry, so the command names no link of its own."""
    import ficelle.cli as cli

    monkeypatch.setattr(
        cli.router,
        "run_failover_demo",
        lambda *a, **k: demo_result(
            attempts=[
                FailoverDemoAttempt("or/one", "openrouter", "one", 429, "rate_limited", 0.002, True),
                unkeyed_attempt("or/two", "openrouter"),
            ],
            answer=None,
        ),
    )

    assert cli.main(["demo"]) == 1
    printed = capsys.readouterr().out
    assert "no provider API key is configured" in printed
    assert f"ficelle set-key openrouter  # create a key at {cli.router.PROVIDER_KEY_URLS['openrouter']}" in printed
