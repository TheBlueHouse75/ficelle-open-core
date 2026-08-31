from __future__ import annotations

from ficelle.provider_admission import ProviderAdmissionLedger


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
