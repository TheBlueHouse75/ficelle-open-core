from __future__ import annotations

from datetime import datetime, timezone

from ficelle.domain_models import ProviderBudget
from ficelle.use_cases.provider_budget import (
    PROBE_BUDGET_SHARE,
    budget_from_state,
    budget_pool_scope,
    budget_window,
    drop_provider_budget_in_state,
    probing_is_within_budget,
    provider_budget_rows,
    provider_spend,
    record_provider_spend,
    store_provider_budgets,
    stored_provider_budgets,
)


NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
NEXT_DAY = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
NEXT_MONTH = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def spend(state, meter_key="provider:openrouter", source="openrouter", amount=1.0, *, now=NOW, budget=None):
    record_provider_spend(state, meter_key, source, amount, now=now, budget=budget)


def test_counting_is_per_pool_and_per_window():
    state: dict = {}
    for _ in range(3):
        spend(state)
    spend(state, "provider:nous", "nous")

    assert provider_spend(state, "provider:openrouter", now=NOW) == 3
    assert provider_spend(state, "provider:nous", now=NOW) == 1
    assert provider_spend(state, "provider:groq", now=NOW) == 0  # never called

    # A new window starts from zero without anyone having to reset it.
    assert provider_spend(state, "provider:openrouter", now=NEXT_DAY) == 0
    spend(state, now=NEXT_DAY)
    assert provider_spend(state, "provider:openrouter", now=NEXT_DAY) == 1
    assert state["provider_budget_spend"]["provider:openrouter"] == {
        "window": "2026-08-05",
        "unit": "requests",
        "spent": 1.0,
        "sources": ["openrouter"],
    }


def test_a_shared_pool_is_one_meter_however_many_sources_draw_on_it():
    """Keyed by source, each tier of a shared account would be allowed the full probe share of the same pool."""
    state: dict = {}
    spend(state, "shared_account:acct", "tier_basic", 1.0)
    spend(state, "shared_account:acct", "tier_premium", 5.0)

    assert provider_spend(state, "shared_account:acct", now=NOW) == 6.0
    # Both contributors are remembered, which is the only way back from the pool key to its members.
    assert state["provider_budget_spend"]["shared_account:acct"]["sources"] == [
        "tier_basic",
        "tier_premium",
    ]


def test_a_request_spends_its_burn_weight_not_one():
    state: dict = {}
    spend(state, amount=5.0)
    spend(state, amount=0.0)  # a corrupt weight cannot make a request free — the caller floors it
    assert provider_spend(state, "provider:openrouter", now=NOW) == 5.0


def test_a_monthly_budget_is_a_different_window_not_a_different_schema():
    state: dict = {}
    monthly = ProviderBudget(1000.0, period="month")
    spend(state, budget=monthly)
    spend(state, now=datetime(2026, 8, 30, tzinfo=timezone.utc), budget=monthly)

    assert provider_spend(state, "provider:openrouter", now=NOW, budget=monthly) == 2
    assert state["provider_budget_spend"]["provider:openrouter"]["window"] == "2026-08"
    # September starts from zero.
    assert provider_spend(state, "provider:openrouter", now=NEXT_MONTH, budget=monthly) == 0


def test_an_unknown_budget_never_holds_probing_back():
    """Most providers publish nothing. Throttling them on a guess would be worse than not knowing."""
    state: dict = {}
    spend(state, "provider:mystery", "mystery")

    assert probing_is_within_budget(state, "provider:mystery", None, now=NOW) is True


def test_a_budget_in_a_unit_the_meter_does_not_record_is_not_enforced():
    """Comparing dollars against a count of requests would cap probing on nonsense."""
    state: dict = {}
    for _ in range(100):
        spend(state)

    dollars = ProviderBudget(5.0, unit="dollars")
    assert probing_is_within_budget(state, "provider:openrouter", dollars, now=NOW) is True


def test_probing_stops_at_its_share_and_leaves_the_rest_for_the_user():
    """The share exists so a spent probe budget never costs a user their request."""
    state: dict = {}
    budget = ProviderBudget(50.0)  # the tightest OpenRouter tier
    allowance = int(budget.amount * PROBE_BUDGET_SHARE)

    for _ in range(allowance - 1):
        spend(state)
    assert probing_is_within_budget(state, "provider:openrouter", budget, now=NOW) is True

    spend(state)
    assert probing_is_within_budget(state, "provider:openrouter", budget, now=NOW) is False

    # A generous tier is not throttled by a share sized for a tight one: the same code, read from
    # the account rather than assumed, gives an account with purchased credits 20x the room.
    assert probing_is_within_budget(state, "provider:openrouter", ProviderBudget(1000.0), now=NOW) is True


def test_a_pool_too_small_to_share_still_allows_one_probe():
    """`2 * 0.4` leaves room for less than one probe, and zero probes never converge."""
    state: dict = {}
    budget = ProviderBudget(2.0)
    assert probing_is_within_budget(state, "provider:tiny", budget, now=NOW) is True

    spend(state, "provider:tiny", "tiny")
    assert probing_is_within_budget(state, "provider:tiny", budget, now=NOW) is False


def test_a_corrupt_meter_cannot_silently_disable_the_budget():
    """Hand-edited or half-written state must not read as "nothing spent yet"."""
    for broken in ("many", None, -5, float("nan"), float("inf"), {"nested": 1}):
        state = {
            "provider_budget_spend": {
                "provider:openrouter": {"window": "2026-08-04", "unit": "requests", "spent": broken}
            }
        }
        assert provider_spend(state, "provider:openrouter", now=NOW) == 0
        # Counting again repairs it rather than crashing the probe that noticed.
        spend(state)
        assert provider_spend(state, "provider:openrouter", now=NOW) == 1

    # A meter map of the wrong shape entirely is replaced, not crashed on.
    state = {"provider_budget_spend": ["not", "a", "map"]}
    assert provider_spend(state, "provider:openrouter", now=NOW) == 0
    spend(state)
    assert provider_spend(state, "provider:openrouter", now=NOW) == 1


def test_pre_release_budget_keys_are_dropped_on_sight():
    """A dogfood install that ran unreleased builds must not carry the abandoned schema forever.

    No release ever wrote `provider_daily_usage` / `provider_daily_limits`, so they are not read —
    the cost is at most one window's probe share re-granted on those installs, once.
    """
    state = {
        "provider_daily_usage": {"day": "2026-08-04", "requests": {"openrouter": 19}},
        "provider_daily_limits": {"openrouter": 1000},
    }
    assert stored_provider_budgets(state) == {}
    assert provider_spend(state, "provider:openrouter", now=NOW) == 0

    spend(state)
    assert "provider_daily_usage" not in state
    assert "provider_daily_limits" not in state
    assert provider_spend(state, "provider:openrouter", now=NOW) == 1


def test_budgets_are_stored_and_reread_with_unit_and_period():
    state: dict = {}
    budgets = {
        "openrouter": ProviderBudget(1000.0),
        "cohere": ProviderBudget(1000.0, unit="requests", period="month"),
    }
    store_provider_budgets(state, budgets)
    assert stored_provider_budgets(state) == budgets

    drop_provider_budget_in_state(state, "openrouter")
    assert stored_provider_budgets(state) == {"cohere": budgets["cohere"]}


def test_budget_from_state_refuses_garbage():
    for broken in (None, "many", -5, 0, 1000, float("nan"), {"amount": "many"}, {"amount": -1}, {}):
        assert budget_from_state(broken) is None
    assert budget_from_state({"amount": 5, "unit": "dollars", "period": "nonsense"}) == ProviderBudget(
        5.0, unit="dollars", period="day"
    )


def test_budget_windows_and_pool_scopes():
    assert budget_window("day", NOW) == "2026-08-04"
    assert budget_window("month", NOW) == "2026-08"
    # A budget is an account-level stock: only a shared account changes the pool.
    assert budget_pool_scope("shared_account") == "shared_account"
    for scope in ("model", "provider", "account", "anything"):
        assert budget_pool_scope(scope) == "provider"


def test_admin_rows_surface_spend_limits_and_shared_pools():
    state: dict = {}
    store_provider_budgets(state, {"openrouter": ProviderBudget(1000.0), "tier_basic": ProviderBudget(5.0, unit="dollars")})
    spend(state)
    spend(state, "shared_account:acct", "tier_basic", 3.0)
    spend(state, "shared_account:acct", "tier_plus", 2.0)

    rows = provider_budget_rows(state, NOW)
    assert rows["openrouter"] == {"spent": 1, "limit": 1000.0, "unit": "requests", "period": "day", "shared": False}
    # Each tier of a shared pool reads the pool's whole spend — that is what holds it out.
    assert rows["tier_plus"]["spent"] == 5.0
    assert rows["tier_plus"]["shared"] is True
    # A budget in another unit is surfaced but its spend is not comparable to the meter's count.
    assert rows["tier_basic"]["limit"] == 5.0
    assert rows["tier_basic"]["unit"] == "dollars"
    assert rows["tier_basic"]["spent"] is None
    # A provider with neither spend nor budget has no row at all.
    assert "groq" not in rows

    # Yesterday's meters read as nothing rather than as today's spend.
    assert provider_budget_rows(state, NEXT_DAY).get("tier_plus") is None
