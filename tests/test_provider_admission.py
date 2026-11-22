from __future__ import annotations

from ficelle.provider_admission import (
    OBSERVATION_STALE_SECONDS,
    ProviderAdmissionLedger,
    admission_scope_keys,
)


def test_unknown_limits_fail_open_without_creating_a_budget() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)

    assert ledger.reserve("unknown", {}).allowed is True
    assert ledger.reserve("unknown", {"rate_limit_rpm": "invalid"}).allowed is True


def test_declared_rpm_is_reserved_atomically_with_headroom() -> None:
    now = [0.0]
    ledger = ProviderAdmissionLedger(monotonic=lambda: now[0])
    config = {"rate_limit_rpm": 10}

    decisions = [ledger.reserve("provider", config) for _ in range(10)]

    assert all(decision.allowed for decision in decisions[:9])
    assert decisions[8].remaining == 0
    assert decisions[9].allowed is False
    assert decisions[9].retry_after_seconds == 60

    now[0] = 60.0
    recovered = ledger.reserve("provider", config)
    assert recovered.allowed is True
    assert recovered.remaining == 8


def test_provider_budgets_are_independent() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    config = {"rate_limit_rpm": 1}

    assert ledger.reserve("a", config).allowed is True
    assert ledger.reserve("a", config).allowed is False
    assert ledger.reserve("b", config).allowed is True


def test_model_scoped_budgets_and_observed_headers_are_independent() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    config = {"rate_limit_rpm": 1}

    assert ledger.reserve("provider", config, scope_key="model-a").allowed is True
    assert ledger.reserve("provider", config, scope_key="model-a").allowed is False
    assert ledger.reserve("provider", config, scope_key="model-b").allowed is True

    ledger.observe(
        "provider",
        {
            "X-RateLimit-Limit-Requests": "20",
            "X-RateLimit-Remaining-Requests": "0",
            "X-RateLimit-Reset-Requests": "60s",
        },
        scope_key="model-a",
    )

    assert ledger.reserve("provider", config, scope_key="model-a").allowed is False
    assert ledger.reserve("provider", config, scope_key="model-b").allowed is False
    assert ledger.reserve("provider", config, scope_key="model-c").allowed is True


def test_provider_headers_override_a_stale_or_unknown_declaration() -> None:
    now = [10.0]
    ledger = ProviderAdmissionLedger(monotonic=lambda: now[0])
    ledger.observe(
        "groq",
        {
            "X-RateLimit-Limit-Requests": "30",
            "X-RateLimit-Remaining-Requests": "1",
            "X-RateLimit-Reset-Requests": "2m30s",
        },
    )

    allowed = ledger.reserve("groq", {})
    blocked = ledger.reserve("groq", {})

    assert allowed.allowed is True
    assert allowed.source == "provider_headers"
    assert allowed.remaining == 0
    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 150

    now[0] = 160.0
    assert ledger.reserve("groq", {}).allowed is True


def test_absolute_reset_timestamp_does_not_create_a_day_long_local_block() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    ledger.observe(
        "provider",
        {
            "X-RateLimit-Limit-Requests": "10",
            "X-RateLimit-Remaining-Requests": "0",
            "X-RateLimit-Reset-Requests": "1787500000",
        },
    )

    decision = ledger.reserve("provider", {"rate_limit_rpm": 10})

    assert decision.allowed is True
    assert decision.source == "declared_rpm"


GROQ_LIKE = {"rate_limit_rpm": 30}


def _headers(limit: int, remaining: int, reset: str, *, kind: str = "requests") -> dict[str, str]:
    return {
        f"x-ratelimit-limit-{kind}": str(limit),
        f"x-ratelimit-remaining-{kind}": str(remaining),
        f"x-ratelimit-reset-{kind}": reset,
    }


def _reserve_for_model(ledger: ProviderAdmissionLedger, cfg: dict, model: str):
    declared, observed = admission_scope_keys(cfg, model)
    return ledger.reserve("groq", cfg, scope_key=declared, observed_scope_key=observed)


def _observe_for_model(ledger: ProviderAdmissionLedger, cfg: dict, model: str, headers: dict) -> None:
    _, observed = admission_scope_keys(cfg, model)
    ledger.observe("groq", headers, scope_key=observed)


def test_a_header_from_one_model_does_not_block_a_sibling_model_of_the_same_provider() -> None:
    # Groq reports its buckets per model; a provider without `rate_limit_scope` still declares
    # its RPM provider-wide, but what a response header says only describes the model that answered.
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    _observe_for_model(ledger, GROQ_LIKE, "model-a", _headers(14400, 0, "45s"))

    assert _reserve_for_model(ledger, GROQ_LIKE, "model-a").allowed is False
    assert _reserve_for_model(ledger, GROQ_LIKE, "model-b").allowed is True


def test_an_explicit_provider_scope_keeps_observed_headers_provider_wide() -> None:
    cfg = {"rate_limit_rpm": 30, "rate_limit_scope": "provider"}
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    _observe_for_model(ledger, cfg, "model-a", _headers(100, 0, "45s"))

    assert _reserve_for_model(ledger, cfg, "model-b").allowed is False


def test_a_daily_header_never_lifts_the_declared_per_minute_guard() -> None:
    # Groq's requests headers describe the per-day bucket (a 14.4k limit): honouring them in place
    # of the declared 30 RPM used to leave the per-minute guard switched off.
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    ledger.observe("groq", _headers(14400, 14000, "3m20s"))

    decisions = [ledger.reserve("groq", GROQ_LIKE) for _ in range(28)]

    assert [d.allowed for d in decisions] == [True] * 27 + [False]
    assert decisions[-1].source == "declared_rpm"


def test_minute_and_day_observations_are_separate_buckets() -> None:
    now = [0.0]
    ledger = ProviderAdmissionLedger(monotonic=lambda: now[0])
    ledger.observe("p", _headers(14400, 14000, "3m20s"))
    ledger.observe("p", _headers(30, 0, "20s"))

    refused = ledger.reserve("p", {})
    assert refused.allowed is False
    assert refused.retry_after_seconds == 20

    # The minute window rolled over; the daily one is still tracked and still decrements.
    now[0] = 21.0
    allowed = ledger.reserve("p", {})
    assert allowed.allowed is True
    assert allowed.limit == 14400
    assert allowed.remaining == 13999


def test_a_bucket_whose_reset_drifts_under_two_minutes_is_not_tracked_twice() -> None:
    # One long window draining through the minute/day boundary used to leave its older copy
    # behind; that copy kept being decremented and refused a model the provider still admitted.
    now = [0.0]
    ledger = ProviderAdmissionLedger(monotonic=lambda: now[0])
    ledger.observe("p", _headers(100, 1, "2m10s"))
    now[0] = 20.0
    ledger.observe("p", _headers(100, 50, "1m50s"))

    assert ledger.reserve("p", {}).allowed is True
    second = ledger.reserve("p", {})
    assert second.allowed is True
    assert second.remaining == 48


def test_a_day_bucket_draining_under_two_minutes_leaves_a_live_minute_bucket_alone() -> None:
    # Re-keying the drained day bucket into `minute` overwrote the exhausted per-minute bucket, so
    # the router admitted requests the provider was about to refuse with a 429.
    now = [0.0]
    ledger = ProviderAdmissionLedger(monotonic=lambda: now[0])
    ledger.observe("p", _headers(30, 0, "60s"))
    ledger.observe("p", _headers(100, 50, "2m10s"))
    now[0] = 20.0
    ledger.observe("p", _headers(100, 50, "1m50s"))

    refused = ledger.reserve("p", {})
    assert refused.allowed is False
    assert refused.retry_after_seconds == 40


def test_a_refusal_by_the_observed_bucket_does_not_spend_the_declared_allowance() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    ledger.observe("p", _headers(30, 0, "20s"))
    for _ in range(5):
        assert ledger.reserve("p", {"rate_limit_rpm": 2}).allowed is False

    ledger.observe("p", _headers(30, 5, "20s"))
    assert ledger.reserve("p", {"rate_limit_rpm": 2}).allowed is True


def test_token_headers_are_observed_but_never_gate_admission() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    ledger.observe("p", _headers(6000, 0, "7.66s", kind="tokens"))

    assert ledger.reserve("p", {}).allowed is True
    tokens = ledger.snapshot({"p": {}})["p"]["tokens"]
    assert (tokens["limit"], tokens["remaining"], tokens["source"]) == (6000, 0, "provider_headers")


def test_non_finite_header_values_are_ignored_instead_of_raising() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    ledger.observe("p", {**_headers(10, 5, "10s"), "x-ratelimit-limit-requests": "inf"})
    ledger.observe("p", {**_headers(10, 5, "10s"), "x-ratelimit-remaining-requests": "nan"})

    assert ledger.snapshot({"p": {}})["p"]["state"] == "unknown"


def test_snapshot_distinguishes_observed_declared_stale_and_unknown() -> None:
    now = [100.0]
    ledger = ProviderAdmissionLedger(monotonic=lambda: now[0], wall=lambda: 1_800_000_000.0 + now[0])
    providers = {
        "seen": {},
        "declared": {"rate_limit_rpm": 30},
        "silent": {},
        "late": {"rate_limit_rpm": 20},
    }
    ledger.observe("seen", _headers(30, 7, "41s"), namespace="home")
    ledger.observe("late", _headers(30, 7, "41s"), namespace="home")
    ledger.observe("other-home", _headers(30, 1, "41s"), namespace="elsewhere")

    now[0] += OBSERVATION_STALE_SECONDS + 1
    ledger.observe("seen", _headers(30, 7, "41s"), namespace="home")
    rows = ledger.snapshot(providers, namespace="home")

    seen = rows["seen"]
    assert seen["state"] == "observed"
    assert seen["requests"]["source"] == "provider_headers"
    assert (seen["requests"]["limit"], seen["requests"]["remaining"]) == (30, 7)
    assert seen["requests"]["reset_in_seconds"] == 41
    assert seen["requests"]["observed_at"].endswith("+00:00")
    assert seen["tokens"] is None

    declared = rows["declared"]
    assert declared["state"] == "declared_only"
    assert declared["requests"]["source"] == "declared"
    assert declared["requests"]["limit"] == 30
    assert declared["requests"]["remaining"] is None
    assert declared["declared"]["admission_limit"] == 27

    late = rows["late"]
    assert late["state"] == "stale"
    assert late["requests"]["fresh"] is False
    assert late["declared"]["rpm"] == 20

    assert rows["silent"]["state"] == "unknown"
    assert rows["silent"]["requests"] is None
    # Another FICELLE_HOME's observations never leak into this one's snapshot.
    assert "other-home" not in rows


def test_snapshot_reports_the_most_constrained_model_and_exposes_no_header_dump() -> None:
    ledger = ProviderAdmissionLedger(monotonic=lambda: 0.0)
    cfg = {"rate_limit_rpm": 30}
    _observe_for_model(ledger, cfg, "roomy", {**_headers(100, 90, "30s"), "authorization": "Bearer sk-secret"})
    _observe_for_model(ledger, cfg, "tight", _headers(100, 3, "30s"))

    row = ledger.snapshot({"groq": cfg})["groq"]

    assert row["requests"]["scope"] == "model"
    assert row["requests"]["model"] == "tight"
    assert row["requests"]["remaining"] == 3
    assert "sk-secret" not in repr(row)


def test_an_exhausted_bucket_stays_a_live_gauge_for_as_long_as_it_refuses() -> None:
    # A refused model is never dispatched, so its exhausted bucket is never re-observed: aging it
    # into "stale" hid the reset countdown while admission kept skipping the model.
    now = [0.0]
    ledger = ProviderAdmissionLedger(monotonic=lambda: now[0])
    ledger.observe("spent", _headers(1000, 0, "5h"))
    ledger.observe("draining", _headers(1000, 400, "5h"))

    now[0] = OBSERVATION_STALE_SECONDS + 100
    assert ledger.reserve("spent", {}).allowed is False
    rows = ledger.snapshot({"spent": {}, "draining": {}})

    assert rows["spent"]["state"] == "observed"
    assert rows["spent"]["requests"]["fresh"] is True
    assert rows["spent"]["requests"]["remaining"] == 0
    assert rows["spent"]["requests"]["reset_in_seconds"] == 18000 - OBSERVATION_STALE_SECONDS - 100
    # A bucket that still had capacity is only a guess after the same age.
    assert rows["draining"]["state"] == "stale"
