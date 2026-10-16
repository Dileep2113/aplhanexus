import dataclasses

import pytest

from intraday.models import Direction, Setup
from intraday.risk import DayLedger, RiskLimits, size


def _setup(entry=1000.0, stop=990.0) -> Setup:
    return Setup(symbol="X", name="X", direction=Direction.LONG, setup_type="t",
                 trigger=entry, entry_low=entry, entry_high=entry, stop_loss=stop,
                 target1=entry + 20, target2=entry + 40, invalidate=stop - 5, atr=5.0)


def test_spec_worked_example_without_exposure_cap(limits):
    # Spec section 17: 5,00,000 * 0.5% = 2,500 risk; risk/share 10 -> 250 shares.
    wide = dataclasses.replace(limits, max_exposure_pct=100.0)
    r = size(_setup(), wide)
    assert r.shares == 250
    assert r.risk_amount == pytest.approx(2500)
    assert r.notional == pytest.approx(250_000)
    assert r.allowed and r.reasons == ()


def test_exposure_cap_bites(limits):
    # 40% of 5L = 2L exposure -> 200 shares at 1000, below the 250 risk allows.
    r = size(_setup(), limits)
    assert r.shares == 200
    assert r.risk_amount == pytest.approx(2000)
    assert any("exposure" in x for x in r.reasons)


def test_open_exposure_reduces_room(limits):
    ledger = DayLedger(open_exposure=150_000, open_positions=1, trades_taken=1)
    r = size(_setup(), limits, ledger)
    assert r.shares == 50   # 200k - 150k = 50k room


def test_daily_loss_limit_blocks(limits):
    ledger = DayLedger(realised_pnl=-7_500)   # 1.5% of 5L
    r = size(_setup(), limits, ledger)
    assert r.shares == 0 and any("daily loss" in x for x in r.reasons)


def test_remaining_loss_budget_shrinks_risk(limits):
    # Already down 6,000 of a 7,500 budget -> only 1,500 risk left -> 150 sh.
    ledger = DayLedger(realised_pnl=-6_000)
    r = size(_setup(), limits, ledger)
    assert r.shares == 150


def test_trade_count_and_position_caps(limits):
    assert size(_setup(), limits, DayLedger(trades_taken=5)).shares == 0
    assert size(_setup(), limits, DayLedger(open_positions=3)).shares == 0


def test_too_wide_stop_yields_zero_not_one(limits):
    # Risk/share 5,000 > 2,500 allowed: a single share already breaches.
    r = size(_setup(entry=10_000, stop=5_000), limits)
    assert r.shares == 0
    assert any("even one lot" in x for x in r.reasons)


def test_lot_rounding(limits):
    lots = dataclasses.replace(limits, lot_rounding=75, max_exposure_pct=100.0)
    assert size(_setup(), lots).shares == 225   # 250 rounded down to a 75 multiple
