"""Upstream transport hygiene: idle pool expiry, stale-socket replay, timeout attribution.

A NAT or load balancer between Ficelle and a provider drops an idle keep-alive flow without
telling either end. The next send writes into a socket nobody is listening on, the kernel
retransmits, and the call fails as a read timeout long before its budget. These tests pin the
defences where they now live — inside the transport: a pooled connection is aged individually,
a send that reused one and died before any response object exists is replayed exactly once, and
neither costs the model an attempt row or a cooldown.
"""

from __future__ import annotations

import errno
import importlib
import time

import pytest
import requests
import urllib3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_router(monkeypatch: pytest.MonkeyPatch, tmp_path):
    ficelle_home = tmp_path / ".ficelle"
    ficelle_home.mkdir()
    monkeypatch.setenv("FICELLE_HOME", str(ficelle_home))
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", str(ficelle_home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import ficelle.router as router

    return importlib.reload(router)


@pytest.fixture
def router(monkeypatch, tmp_path):
    """The router module bound to a throwaway Ficelle home."""
    return load_router(monkeypatch, tmp_path)


def stale_failure() -> requests.exceptions.ReadTimeout:
    """The shape a keep-alive socket dropped by the network arrives in."""
    failure = requests.exceptions.ReadTimeout("Read timed out. (read timeout=600.0)")
    failure.__cause__ = OSError(errno.ETIMEDOUT, "Operation timed out")
    return failure


def wrapped_stale_failure() -> requests.exceptions.ConnectionError:
    """The same cause after urllib3 reclassified it and `requests` re-wrapped it."""
    return requests.exceptions.ConnectionError(
        urllib3.exceptions.ReadTimeoutError(None, "https://provider.example", "Read timed out.")
    )


class FakeConnection:
    """Enough of a urllib3 connection for the pool to hand it out and take it back."""

    def __init__(self) -> None:
        self.sock: object | None = object()
        self.closes = 0

    @property
    def is_connected(self) -> bool:
        return self.sock is not None

    def close(self) -> None:
        self.closes += 1
        self.sock = None


class FakeResponse:
    status_code = 200
    text = "{}"
    content = b"{}"
    headers = {"Content-Type": "application/json"}


class FakeSession:
    """A scripted session that also reports connection provenance the way a real pool does."""

    def __init__(self, router, outcomes, *, reused: bool = True) -> None:
        self.posts: list[str] = []
        self._router = router
        self._outcomes = list(outcomes)
        self._reused = reused

    def post(self, url, **_kwargs):
        self.posts.append(url)
        self._router._UPSTREAM_CONNECTION_STATE.reused = self._reused
        outcome = self._outcomes.pop(0) if self._outcomes else FakeResponse()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def install_fake_session(monkeypatch, router, outcomes, *, reused: bool = True) -> FakeSession:
    session = FakeSession(router, outcomes, reused=reused)
    monkeypatch.setattr(router, "_UPSTREAM_SESSION", session)
    return session


def allow_provider(monkeypatch, router) -> None:
    monkeypatch.setattr(
        router,
        "provider_access_for",
        lambda _source, _config: router.ProviderAccess("real-key", "https://provider.example/v1", "ok"),
    )


def invokable_model(model_id: str = "ficelle/openrouter/agent-ready") -> dict:
    return {
        "id": model_id,
        "source": "openrouter",
        "upstream_id": model_id.rsplit("/", 1)[-1],
    }


def chat_body() -> dict:
    return {"model": "ficelle/auto-fast", "messages": [{"role": "user", "content": "hi"}]}


# --------------------------------------------------------------------------- #
# Per-connection idle expiry
# --------------------------------------------------------------------------- #
def test_a_recently_pooled_connection_is_handed_back_still_open(router):
    pool = router.IdleExpiringHTTPSConnectionPool("provider.example", maxsize=2)
    connection = FakeConnection()
    pool._get_conn()  # drain the empty slot the pool starts with

    pool._put_conn(connection)
    assert isinstance(getattr(connection, router._POOLED_AT_ATTRIBUTE), float)

    assert pool._get_conn() is connection
    assert connection.closes == 0
    assert router._UPSTREAM_CONNECTION_STATE.reused is True


def test_a_connection_idle_past_the_window_is_closed_on_its_way_out(router):
    pool = router.IdleExpiringHTTPSConnectionPool("provider.example", maxsize=2)
    connection = FakeConnection()
    pool._get_conn()
    pool._put_conn(connection)
    setattr(
        connection,
        router._POOLED_AT_ATTRIBUTE,
        time.monotonic() - router.UPSTREAM_POOL_IDLE_MAX_SECONDS - 1.0,
    )

    # The object is kept, not discarded: urllib3 reconnects a connection whose socket is gone.
    assert pool._get_conn() is connection
    assert connection.closes == 1
    assert router._UPSTREAM_CONNECTION_STATE.reused is False


def test_a_fresh_connection_is_never_reported_as_reused(router):
    pool = router.IdleExpiringHTTPSConnectionPool("provider.example", maxsize=2)

    connection = pool._get_conn()

    assert getattr(connection, "sock", None) is None
    assert router._UPSTREAM_CONNECTION_STATE.reused is False


def test_the_pooled_session_builds_idle_expiring_pools(router):
    session = router.upstream_session()

    adapter = session.get_adapter("https://provider.example/v1/chat/completions")
    assert isinstance(adapter, router.IdleExpiringHTTPAdapter)
    assert adapter.poolmanager.pool_classes_by_scheme == router._IDLE_EXPIRING_POOL_CLASSES
    # The override is an instance attribute, so no other `requests` user is affected.
    assert urllib3.poolmanager.pool_classes_by_scheme["https"] is urllib3.HTTPSConnectionPool


# --------------------------------------------------------------------------- #
# Stale-connection detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        pytest.param(stale_failure(), True, id="read-timeout-over-etimedout"),
        pytest.param(wrapped_stale_failure(), True, id="urllib3-timeout-inside-connection-error"),
        pytest.param(
            requests.exceptions.ConnectionError(OSError(errno.ECONNRESET, "Connection reset by peer")),
            True,
            id="econnreset",
        ),
        pytest.param(
            requests.exceptions.ConnectTimeout(OSError(errno.ETIMEDOUT, "Operation timed out")),
            False,
            id="connect-timeout-keeps-its-fast-failover-attribution",
        ),
        pytest.param(
            requests.exceptions.ConnectionError("connection refused"),
            False,
            id="provider-refusal",
        ),
    ],
)
def test_stale_connection_signatures(router, failure, expected):
    # Early enough that the errno-less arm cannot be what decides these: each case is judged on
    # its shape.
    assert router.stale_upstream_connection(failure, elapsed_seconds=1.0) is expected


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        pytest.param(18.0, True, id="early-timeout"),
        pytest.param(29.9, True, id="just-inside-the-cap"),
        pytest.param(45.0, False, id="timeout-past-the-cap"),
        pytest.param(599.5, False, id="timeout-that-spent-a-whole-long-budget"),
    ],
)
def test_errno_less_timeout_is_stale_only_when_it_fired_early(router, elapsed, expected):
    failure = requests.exceptions.ReadTimeout("Read timed out. (read timeout=600.0)")
    assert router.stale_upstream_connection(failure, elapsed_seconds=elapsed) is expected


def test_errno_stays_conclusive_however_late_it_arrives(router):
    assert router.stale_upstream_connection(stale_failure(), elapsed_seconds=599.0)


def test_slow_provider_on_a_reused_socket_is_not_replayed(monkeypatch, router):
    silent = requests.exceptions.ReadTimeout("Read timed out. (read timeout=600.0)")
    session = install_fake_session(monkeypatch, router, [silent, FakeResponse()])
    clock = iter([0.0, 599.0])
    monkeypatch.setattr(router.time, "monotonic", lambda: next(clock))

    with pytest.raises(requests.exceptions.ReadTimeout):
        router.upstream_post("https://provider.example/v1/chat/completions", timeout=(5.0, 600.0))

    assert len(session.posts) == 1


# --------------------------------------------------------------------------- #
# One replay, inside the transport
# --------------------------------------------------------------------------- #
def test_upstream_post_replays_once_on_a_reused_connection(monkeypatch, router):
    session = install_fake_session(monkeypatch, router, [stale_failure(), FakeResponse()])

    response = router.upstream_post("https://provider.example/v1/chat/completions")

    assert len(session.posts) == 2
    assert response._ficelle_stale_latency_seconds >= 0


def test_upstream_post_replays_a_timeout_wrapped_in_a_connection_error(monkeypatch, router):
    session = install_fake_session(monkeypatch, router, [wrapped_stale_failure(), FakeResponse()])

    router.upstream_post("https://provider.example/v1/chat/completions")

    assert len(session.posts) == 2


def test_upstream_post_does_not_replay_a_fresh_connection(monkeypatch, router):
    session = install_fake_session(monkeypatch, router, [stale_failure()], reused=False)

    with pytest.raises(requests.exceptions.ReadTimeout):
        router.upstream_post("https://provider.example/v1/chat/completions")

    assert len(session.posts) == 1


def test_upstream_post_does_not_replay_a_provider_refusal(monkeypatch, router):
    session = install_fake_session(
        monkeypatch, router, [requests.exceptions.ConnectionError("connection refused")]
    )

    with pytest.raises(requests.exceptions.ConnectionError):
        router.upstream_post("https://provider.example/v1/chat/completions")

    assert len(session.posts) == 1


def test_upstream_post_reports_a_second_failure_as_it_came(monkeypatch, router):
    second = stale_failure()
    session = install_fake_session(monkeypatch, router, [stale_failure(), second])

    with pytest.raises(requests.exceptions.ReadTimeout) as raised:
        router.upstream_post("https://provider.example/v1/chat/completions")

    assert len(session.posts) == 2
    assert raised.value is second
    assert raised.value._ficelle_stale_latency_seconds >= 0


def test_upstream_post_skips_the_replay_the_caller_cannot_afford(monkeypatch, router):
    session = install_fake_session(monkeypatch, router, [stale_failure(), FakeResponse()])

    with pytest.raises(requests.exceptions.ReadTimeout):
        router.upstream_post(
            "https://provider.example/v1/chat/completions", replay_allowed=lambda: False
        )

    assert len(session.posts) == 1


# --------------------------------------------------------------------------- #
# What the route log is told
# --------------------------------------------------------------------------- #
def test_invoke_model_reports_the_replay_on_the_attempt(monkeypatch, router):
    from ficelle.use_cases.chat_completion import add_transport_timeout_telemetry

    session = install_fake_session(monkeypatch, router, [stale_failure(), FakeResponse()])
    allow_provider(monkeypatch, router)

    response = router.invoke_model(invokable_model(), chat_body(), router.load_config())

    assert len(session.posts) == 2
    assert response.status_code == 200
    attempt_update: dict = {}
    add_transport_timeout_telemetry(attempt_update, response)
    assert attempt_update["transport_retry"] == "stale_connection"
    assert attempt_update["stale_latency_seconds"] >= 0
    # The model's own latency starts at the send that actually reached it, so the discarded
    # wait is added back to the recorded start instead of being charged to the model.
    assert response._ficelle_request_started_monotonic >= response._ficelle_stale_latency_seconds


def test_invoke_model_keeps_the_replay_marker_on_a_second_failure(monkeypatch, router):
    session = install_fake_session(monkeypatch, router, [stale_failure(), stale_failure()])
    allow_provider(monkeypatch, router)

    with pytest.raises(requests.exceptions.ReadTimeout) as raised:
        router.invoke_model(invokable_model(), chat_body(), router.load_config())

    assert len(session.posts) == 2
    # A second failure is classified and cooled down as the ordinary timeout it is, with the
    # replay still on the record so the route log states what was tried.
    assert raised.value._ficelle_timeout_phase == "response_headers"
    assert raised.value._ficelle_stale_latency_seconds >= 0


def test_invoke_model_does_not_replay_past_the_request_deadline(monkeypatch, router):
    session = install_fake_session(monkeypatch, router, [stale_failure(), FakeResponse()])
    allow_provider(monkeypatch, router)

    with pytest.raises(requests.exceptions.ReadTimeout):
        router.invoke_model(
            invokable_model(),
            chat_body(),
            router.load_config(),
            deadline_monotonic=time.monotonic(),
        )

    assert len(session.posts) == 1


def test_invoke_model_does_not_replay_once_upstream_bytes_arrived(monkeypatch, router):
    class DrippingResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}

        def iter_content(self, chunk_size=None):
            del chunk_size
            yield b'{"partial":'
            raise stale_failure()

        def close(self):
            pass

    session = install_fake_session(monkeypatch, router, [DrippingResponse()])
    allow_provider(monkeypatch, router)

    with pytest.raises(requests.exceptions.ReadTimeout) as raised:
        router.invoke_model(invokable_model(), chat_body(), router.load_config())

    # The replay lives below the response object: once one exists, it is out of reach.
    assert len(session.posts) == 1
    assert not hasattr(raised.value, "_ficelle_stale_latency_seconds")
    assert raised.value._ficelle_timeout_phase == "response_body"


def test_bare_timeout_before_any_response_is_reported_as_a_timeout(monkeypatch, router):
    install_fake_session(monkeypatch, router, [TimeoutError("timed out")], reused=False)
    allow_provider(monkeypatch, router)

    with pytest.raises(requests.exceptions.ReadTimeout) as raised:
        router.invoke_model(invokable_model(), chat_body(), router.load_config())

    # `unavailable` would cool the model down for 600s; a timeout costs 300s and blames nobody
    # for a socket that never answered.
    assert raised.value._ficelle_timeout_phase == "response_headers"


# --------------------------------------------------------------------------- #
# Walking the cause chain
# --------------------------------------------------------------------------- #
def test_walk_exception_chain_stops_on_a_cycle():
    from ficelle.failures import walk_exception_chain

    outer = requests.exceptions.ConnectionError("outer")
    inner = OSError(errno.ECONNRESET, "Connection reset by peer")
    outer.__cause__ = inner
    inner.__cause__ = outer

    assert list(walk_exception_chain(outer)) == [outer, inner]


# --------------------------------------------------------------------------- #
# Attempt telemetry
# --------------------------------------------------------------------------- #
def test_exception_attempt_records_the_transport_detail_and_errno():
    from ficelle.use_cases.chat_completion import evaluate_invocation_exception

    decision = evaluate_invocation_exception(
        stale_failure(),
        {"id": "ficelle/openrouter/slow", "upstream_id": "slow", "source": "openrouter"},
        latency_seconds=18.02,
        timeout=True,
    )

    assert decision.attempt_update["error_detail"] == "ReadTimeout: Read timed out. (read timeout=600.0)"
    assert decision.attempt_update["error_errno"] == errno.ETIMEDOUT
    assert decision.attempt_update["reason"] == "timeout"


def test_the_replay_name_is_written_where_the_discarded_latency_is():
    from ficelle.use_cases.chat_completion import add_transport_timeout_telemetry

    class Subject:
        _ficelle_stale_latency_seconds = 18.01234
        _ficelle_timeout_phase = "response_headers"

    attempt_update: dict = {}
    add_transport_timeout_telemetry(attempt_update, Subject())

    assert attempt_update["transport_retry"] == "stale_connection"
    assert attempt_update["stale_latency_seconds"] == 18.0123
    assert attempt_update["timeout_phase"] == "response_headers"


def test_an_ordinary_timeout_carries_no_transport_retry():
    from ficelle.use_cases.chat_completion import add_transport_timeout_telemetry

    class Subject:
        _ficelle_timeout_phase = "response_body"

    attempt_update: dict = {}
    add_transport_timeout_telemetry(attempt_update, Subject())

    assert "transport_retry" not in attempt_update


def test_request_log_keeps_the_new_attempt_fields():
    from ficelle import request_log

    assert {
        "error_detail",
        "error_errno",
        "transport_retry",
        "stale_latency_seconds",
    } <= set(request_log._ATTEMPT_KEYS)
