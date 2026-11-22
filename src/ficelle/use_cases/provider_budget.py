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
- **Windows, not days.** A budget carries the period it refills on (``day`` or ``month``); each
  meter counts both schedules, stamped with the concrete window ("2026-08-04", "2026-08"), and a
  window that has passed is replaced, never rolled over — yesterday's spend can never hold back
  today's probing.

The meters live in their own small file (``state_store.SpendMeterStore``), not in runtime state:
every upstream call writes one before it leaves, and that write must cost neither a whole-state
rewrite nor a state read. Counting both windows is what lets the writer skip the budget lookup.
Budgets stay in runtime state; they change on a catalog refresh, not per request.
"""

from __future__ import annotations

import math
import operator
from datetime import datetime
from typing import Any, Callable

from ficelle.domain_models import ProviderBudget

# Share of a pool's budget background probing may spend per window. The rest is kept for the user's
# own traffic: discovery converges over days, a blocked request fails now.
PROBE_BUDGET_SHARE = 0.4

# What Ficelle's meter actually records: requests it sent, weighted by burn_weight. A budget in any
# other unit (tokens, dollars, Neurons) is stored and surfaced but never enforced against this
# meter — capping one unit on a count of another would hold probing back on nonsense.
METER_UNIT = "requests"

BUDGET_PERIODS = ("day", "month")

# Where runtime state held the meters before they moved to their own file. Read by the migration
# that carries them over (`legacy_spend_meters`) and left there, frozen, until every window in it
# has passed (`legacy_spend_meters_expired`): a rolled-back update restores a version that reads
# only this key, and it must not see the window's quota as untouched.
LEGACY_SPEND_STATE_KEY = "provider_budget_spend"
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


def _window_slot(record: Any, period: str, window: str) -> dict[str, Any] | None:
    """The one rule for whether a meter counts: right shape, the meter's unit, this window.

    Everything else — another window, a unit from some future schema change, hand-edited garbage —
    is stale: read as empty, replaced on the next write, never rolled over.
    """
    if not isinstance(record, dict) or record.get("unit") != METER_UNIT:
        return None
    windows = record.get("windows")
    slot = windows.get(period) if isinstance(windows, dict) else None
    return slot if isinstance(slot, dict) and slot.get("window") == window else None


def _window_members(record: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    """``slot``'s contributors (source: last spend time).

    An unreleased build kept them on the record instead of the window, and its spend files are on
    real machines: a slot it wrote has no ``sources`` of its own, so the record's list stands in,
    stamped at 0 like a legacy migration. Every slot written since carries its own, so that list
    never reaches a later window.
    """
    members = slot.get("sources")
    if isinstance(members, dict):
        return members
    previous = record.get("sources")
    return {str(member): 0.0 for member in previous if member} if isinstance(previous, list) else {}


def _writable_record(meters: dict[str, Any], meter_key: str) -> dict[str, Any]:
    record = meters.get(meter_key)
    if not isinstance(record, dict) or record.get("unit") != METER_UNIT:
        record = {"unit": METER_UNIT, "windows": {}}
        meters[meter_key] = record
    if not isinstance(record.get("windows"), dict):
        record["windows"] = {}
    return record


def _add_to_window(
    record: dict[str, Any],
    period: str,
    window: str,
    amount: float,
    *,
    combine: Callable[[float, float], float],
    members: dict[str, float],
) -> None:
    """Add ``amount`` to ``period``'s slot and stamp ``members`` (source: last spend time) on it.

    Members belong to the window, not the record, so they reset with it: a provider whose config
    moved it to another shared account stops reading the old pool once it has not spent there in
    the current window. Each keeps the time it last spent here, so a move within one window resolves
    to the pool it spent on last.
    """
    windows = record["windows"]
    slot = windows.get(period)
    if not isinstance(slot, dict) or slot.get("window") != window:
        if isinstance(slot, dict) and str(slot.get("window") or "") > window:
            return  # an older window never replaces a newer one
        slot = {"window": window, "spent": 0.0, "sources": {}}
        windows[period] = slot
    slot["spent"] = combine(_finite_positive(slot.get("spent")), amount)
    slot["sources"] = _window_members(record, slot)
    for source, spent_at in members.items():
        slot["sources"][source] = max(_finite_positive(slot["sources"].get(source)), spent_at)


def record_provider_spend(
    meters: dict[str, Any],
    meter_key: str,
    source: str,
    amount: float,
    *,
    now: datetime,
) -> None:
    """Spend ``amount`` against ``meter_key``'s current day and month windows.

    Both schedules are counted, so the writer — on every request's path — needs no budget and so no
    read of runtime state; a reader picks the window its budget refills on. ``source`` is
    remembered in each window so a per-provider surface can find the pool a provider draws on now
    without re-deriving scope from config — a shared pool's key does not name its members.
    """
    amount = _finite_positive(amount)
    if not meter_key or not source or not amount:
        return
    record = _writable_record(meters, meter_key)
    for period in BUDGET_PERIODS:
        _add_to_window(
            record, period, budget_window(period, now), amount,
            combine=operator.add, members={source: now.timestamp()},
        )


def legacy_spend_meters(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """The meters an earlier version kept in runtime state, or None when there is nothing to move."""
    if not isinstance(state, dict) or LEGACY_SPEND_STATE_KEY not in state:
        return None
    meters = state.get(LEGACY_SPEND_STATE_KEY)
    return meters if isinstance(meters, dict) else {}


def _legacy_window(legacy_record: Any) -> tuple[str, str, float] | None:
    """A legacy record's ``(period, window, spent)``, or None for one that never counted.

    A legacy record counted one window, whose id says its period ("2026-08" a month, "2026-08-04"
    a day).
    """
    window = legacy_record.get("window") if isinstance(legacy_record, dict) else None
    if not isinstance(window, str) or legacy_record.get("unit") != METER_UNIT:
        return None
    spent = _finite_positive(legacy_record.get("spent"))
    if not spent:
        return None
    return ("month" if len(window) == len("2026-08") else "day"), window, spent


def legacy_spend_meters_expired(legacy: dict[str, Any], *, now: datetime) -> bool:
    """Whether no legacy record still counts a current window — the only time it may be dropped.

    Until then the state key is the only meter a rolled-back (or downgraded) version reads; after,
    that version would read every record as stale anyway, so removing it loses nothing.
    """
    for legacy_record in legacy.values():
        parsed = _legacy_window(legacy_record)
        if parsed and parsed[1] >= budget_window(parsed[0], now):
            return False
    return True


def merge_legacy_spend_meters(meters: dict[str, Any], legacy: dict[str, Any], *, now: datetime) -> None:
    """Carry meters out of runtime state into ``meters`` without counting anything twice.

    Only windows still current at ``now`` are copied: a day record from earlier this month no
    longer counts its day but still counts its month, which must not restart at zero. The state key
    stays in place, frozen, while any record's own window is current, so every start brings the
    same records over again (as do two processes migrating at once): within one window the larger
    count wins, which is idempotent, and an older window never replaces a newer one.
    """
    for meter_key, legacy_record in legacy.items():
        parsed = _legacy_window(legacy_record)
        if not parsed:
            continue
        period, window, spent = parsed
        raw_sources = legacy_record.get("sources")
        # Stamped at 0: a legacy record carries no time, so any spend this version recorded since
        # is the more recent evidence of which pool a provider draws on now.
        members = {str(member): 0.0 for member in raw_sources if member} if isinstance(raw_sources, list) else {}
        # The new meter counts both windows; a day's requests also belong to its month, the only
        # month figure this record can still provide (a lower bound for the month).
        windows = {period: window, "month": window[: len("2026-08")]}
        current = {target: wid for target, wid in windows.items() if wid >= budget_window(target, now)}
        if current:
            record = _writable_record(meters, str(meter_key))
            for target, target_window in current.items():
                _add_to_window(record, target, target_window, spent, combine=max, members=members)


def provider_spend(
    meters: dict[str, Any],
    meter_key: str,
    *,
    now: datetime,
    budget: ProviderBudget | None = None,
) -> float:
    """What ``meter_key``'s pool has spent in its current window. Zero for unknown or stale.

    A pool with no known budget reads the day window, the schedule most budgets use.
    """
    period = budget.period if budget else "day"
    record = meters.get(meter_key) if isinstance(meters, dict) else None
    slot = _window_slot(record, period, budget_window(period, now))
    return _finite_positive(slot.get("spent")) if slot else 0.0


def probing_is_within_budget(
    meters: dict[str, Any],
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
    spent = provider_spend(meters, meter_key, now=now, budget=budget)
    return spent < budget.amount * PROBE_BUDGET_SHARE


def provider_budget_rows(
    state: dict[str, Any], meters: dict[str, Any], now: datetime
) -> dict[str, dict[str, Any]]:
    """Per-source budget surface for the admin: ``{source: {spent, limit, unit, period, shared}}``.

    Budgets come from runtime state, spend from the meters. Sources come from the meters
    themselves — each window remembers its contributors, which is the only way back from a shared
    pool's key to the providers that drew on it — and from the stored budgets, so a provider that
    only ever spent, or only ever resolved an allowance, still gets a row. A provider reads the pool
    it spent on last in the window its budget refills on: one its config moved off reads neither
    the old pool's later spend nor a pool it left in an earlier window. ``spent`` is None when the
    budget's unit is not what the meter records: surfacing the allowance still explains the
    provider's tier, but the two numbers must not be read as a ratio.
    """
    budgets = stored_provider_budgets(state)
    # (period, source) -> (last spend time, meter key), over the current window of each period.
    latest: dict[tuple[str, str], tuple[float, str]] = {}
    for period in BUDGET_PERIODS:
        window = budget_window(period, now)
        for key, record in (meters.items() if isinstance(meters, dict) else ()):
            slot = _window_slot(record, period, window)
            for member, spent_at in (_window_members(record, slot).items() if slot else ()):
                candidate = (_finite_positive(spent_at), str(key))
                latest[(period, str(member))] = max(latest.get((period, str(member)), candidate), candidate)

    rows: dict[str, dict[str, Any]] = {}
    for source in {*budgets, *(member for _period, member in latest)}:
        budget = budgets.get(source)
        found = latest.get((budget.period if budget else "day", source))
        meter_key = found[1] if found else None
        spent = provider_spend(meters, meter_key, now=now, budget=budget) if meter_key else 0.0
        shared = bool(spent) and str(meter_key).startswith("shared_account:")
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
