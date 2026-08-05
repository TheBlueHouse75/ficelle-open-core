"""Per-provider spend accounting against each account's own budget.

Counting only. What a provider's budget is comes from the provider adapter, and what to do with a
spent one is the caller's call — today only background probing is held back, never a user's request.
Only what Ficelle sends is counted: a key shared with another tool undercounts, and the answer to
that is a dedicated key.

The meter is deliberately not "requests per source per UTC day":

- **Pools, not sources.** Spend is keyed by the pool the budget refills — ``provider:{source}`` for
  an allowance that covers the whole account, ``shared_account:{account_id}`` where several
  configured sources draw on one pool. Keyed by source, each tier of a shared pool would be allowed
  the full probe share of the same allowance.
- **Amounts, not increments.** A request spends its model's ``burn_weight`` (default 1), so a tier
  that drains a shared pool five times faster is metered five times heavier.
- **Windows, not days.** A budget carries the period it refills on (``day`` or ``month``); the
  meter stamps each record with the concrete window it counts ("2026-08-04", "2026-08"), and a
  record from another window is replaced, never rolled over — yesterday's spend can never hold back
  today's probing.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, TypeGuard

from ficelle.domain_models import ProviderBudget

# Share of a pool's budget background probing may spend per window. The rest is kept for the user's
# own traffic: discovery converges over days, a blocked request fails now.
PROBE_BUDGET_SHARE = 0.4

# What Ficelle's meter actually records: requests it sent, weighted by burn_weight. A budget in any
# other unit (tokens, dollars, Neurons) is stored and surfaced but never enforced against this
# meter — capping one unit on a count of another would hold probing back on nonsense.
METER_UNIT = "requests"

BUDGET_PERIODS = ("day", "month")

_SPEND_KEY = "provider_budget_spend"
_BUDGETS_KEY = "provider_budgets"
# A pre-release iteration of the budget wrote requests-per-source-per-UTC-day under these keys.
# No release ever shipped them, so they are not read — only removed on sight, so a dogfood install
# that ran unreleased builds does not carry dead keys forever.
_RETIRED_KEYS = ("provider_daily_usage", "provider_daily_limits")


def budget_window(period: str, now: datetime) -> str:
    """The window id ``now`` falls in — "2026-08-04" for a daily budget, "2026-08" for a monthly.

    ``now`` must be timezone-aware UTC: a provider's own reset is not in the operator's timezone,
    and a local-time window would shift the boundary twice a year.
    """
    return now.strftime("%Y-%m") if period == "month" else now.strftime("%Y-%m-%d")


def _current_window(budget: ProviderBudget | None, now: datetime) -> str:
    """A pool with no known budget is metered on the day window, the schedule most budgets use."""
    return budget_window(budget.period if budget else "day", now)


def budget_pool_scope(free_access_scope: str) -> str:
    """The pool a provider's budget refills, from the model's free-access scope.

    A budget is an account-level stock: OpenRouter's daily allowance spans every model on the key
    even though its quota exhaustion is tracked per model, so everything collapses to the
    provider's own pool — except ``shared_account``, where several configured sources draw on one
    account. A provider whose allowance genuinely refills per model would need its budget to carry
    that scope; the meter key format already accommodates it.
    """
    return "shared_account" if free_access_scope == "shared_account" else "provider"


def _finite_positive(value: Any) -> float:
    """Read a stored amount the one way, so writers and readers cannot disagree about it.

    Anything unusable — absent, text, boolean, negative or non-finite from a hand edit — reads
    as 0. The writer normalizes through here too: incrementing a corrupt value in place would leave
    the meter stuck, and a meter that never fills is a budget that never applies.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) and value > 0 else 0.0


def budget_from_state(value: Any) -> ProviderBudget | None:
    """Rebuild a stored budget, or None for anything unusable."""
    if not isinstance(value, dict):
        return None
    amount = _finite_positive(value.get("amount"))
    if not amount:
        return None
    unit = value.get("unit")
    period = value.get("period")
    return ProviderBudget(
        amount,
        unit if isinstance(unit, str) and unit else METER_UNIT,
        period if period in BUDGET_PERIODS else "day",
    )


def stored_provider_budgets(state: dict[str, Any] | None) -> dict[str, ProviderBudget]:
    """Every budget state knows, by source."""
    budgets: dict[str, ProviderBudget] = {}
    raw = state.get(_BUDGETS_KEY) if isinstance(state, dict) else None
    if isinstance(raw, dict):
        for source, value in raw.items():
            budget = budget_from_state(value)
            if budget:
                budgets[str(source)] = budget
    return budgets


def store_provider_budgets(state: dict[str, Any], budgets: dict[str, ProviderBudget]) -> None:
    state[_BUDGETS_KEY] = {
        str(source): {"amount": budget.amount, "unit": budget.unit, "period": budget.period}
        for source, budget in budgets.items()
    }
    for key in _RETIRED_KEYS:
        state.pop(key, None)


def retired_budget_state_present(state: dict[str, Any] | None) -> bool:
    """Whether a write is still owed just to drop the pre-release keys."""
    return isinstance(state, dict) and any(key in state for key in _RETIRED_KEYS)


def drop_provider_budget_in_state(state: dict[str, Any], source: str) -> None:
    budgets = state.get(_BUDGETS_KEY)
    if isinstance(budgets, dict):
        budgets.pop(source, None)


def _record_is_current(record: Any, window: str) -> TypeGuard[dict[str, Any]]:
    """The one rule for whether a meter record counts: right shape, this window, the meter's unit.

    Everything else — another window, a unit from some future schema change, hand-edited garbage —
    is stale: read as empty, replaced on the next write, never rolled over.
    """
    return isinstance(record, dict) and record.get("window") == window and record.get("unit") == METER_UNIT


def record_provider_spend(
    state: dict[str, Any],
    meter_key: str,
    source: str,
    amount: float,
    *,
    now: datetime,
    budget: ProviderBudget | None = None,
) -> None:
    """Spend ``amount`` against ``meter_key``'s current window.

    ``source`` is remembered on the record so a per-provider surface can find the pools a provider
    draws on without re-deriving scope from config — a shared pool's key does not name its members.
    ``budget`` supplies the window schedule only; the recorded unit is always what the meter
    measures, never what a budget happens to be denominated in.
    """
    amount = _finite_positive(amount)
    if not meter_key or not source or not amount:
        return
    window = _current_window(budget, now)
    meters = state.setdefault(_SPEND_KEY, {})
    if not isinstance(meters, dict):
        meters = {}
        state[_SPEND_KEY] = meters
    record = meters.get(meter_key)
    if not _record_is_current(record, window):
        record = {"window": window, "unit": METER_UNIT, "spent": 0.0, "sources": []}
        meters[meter_key] = record
    record["spent"] = _finite_positive(record.get("spent")) + amount
    sources = record.get("sources")
    if not isinstance(sources, list):
        sources = []
        record["sources"] = sources
    if source not in sources:
        sources.append(source)
    for key in _RETIRED_KEYS:
        state.pop(key, None)


def provider_spend(
    state: dict[str, Any],
    meter_key: str,
    *,
    now: datetime,
    budget: ProviderBudget | None = None,
) -> float:
    """What ``meter_key``'s pool has spent in its current window. Zero for unknown or stale."""
    meters = state.get(_SPEND_KEY) if isinstance(state, dict) else None
    record = meters.get(meter_key) if isinstance(meters, dict) else None
    if not _record_is_current(record, _current_window(budget, now)):
        return 0.0
    return _finite_positive(record.get("spent"))


def probing_is_within_budget(
    state: dict[str, Any],
    meter_key: str,
    budget: ProviderBudget | None,
    *,
    now: datetime,
) -> bool:
    """Whether background probing may still spend on ``meter_key``'s pool this window.

    ``budget`` is None when nothing reliable is known — then probing is never held back, since the
    alternative is throttling on a number nobody verified. A budget in a unit the meter does not
    record is likewise never enforced: the comparison would be meaningless. The spend that crosses
    the share line is allowed and the next one is not, so a pool too small to share still gets one
    probe — refusing outright would freeze the catalog on an account whose tier is merely tiny.
    """
    if budget is None or budget.unit != METER_UNIT or not (budget.amount > 0):
        return True
    spent = provider_spend(state, meter_key, now=now, budget=budget)
    return spent < budget.amount * PROBE_BUDGET_SHARE


def provider_budget_rows(state: dict[str, Any], now: datetime) -> dict[str, dict[str, Any]]:
    """Per-source budget surface for the admin: ``{source: {spent, limit, unit, period, shared}}``.

    Sources come from the meters themselves — each record remembers its contributors, which is the
    only way back from a shared pool's key to the providers that drew on it — and from the stored
    budgets, so a provider that only ever spent, or only ever resolved an allowance, still gets a
    row. ``spent`` is None when the budget's unit is not what the meter records: surfacing the
    allowance still explains the provider's tier, but the two numbers must not be read as a ratio.
    """
    budgets = stored_provider_budgets(state)
    meters = state.get(_SPEND_KEY) if isinstance(state, dict) else None
    meter_by_source: dict[str, tuple[str, dict[str, Any]]] = {}
    if isinstance(meters, dict):
        for key, record in meters.items():
            if not isinstance(record, dict):
                continue
            sources = record.get("sources")
            for member in sources if isinstance(sources, list) else []:
                meter_by_source[str(member)] = (str(key), record)

    rows: dict[str, dict[str, Any]] = {}
    for source in {*budgets, *meter_by_source}:
        budget = budgets.get(source)
        meter = meter_by_source.get(source)
        shared = False
        spent = 0.0
        if meter is not None and _record_is_current(meter[1], _current_window(budget, now)):
            spent = _finite_positive(meter[1].get("spent"))
            shared = meter[0].startswith("shared_account:")
        if budget is None and not spent:
            continue
        comparable = budget is None or budget.unit == METER_UNIT
        rows[source] = {
            # Rounded so a float sum of weighted amounts cannot surface as 12.299999999999999.
            "spent": round(spent, 3) if comparable else None,
            "limit": budget.amount if budget else None,
            "unit": budget.unit if budget else METER_UNIT,
            "period": budget.period if budget else "day",
            "shared": shared,
        }
    return rows
