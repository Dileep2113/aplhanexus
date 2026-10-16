from datetime import date, datetime, timedelta

import pytest

from conftest import DAY, bar, flat_bars, t
from intraday.indicators import atr, ema, rvol, vwap, vwap_snapshot
from intraday.models import MarketState
from intraday.session import IST_TZ, SessionClock, SessionTimes


def test_vwap_is_volume_weighted_typical_price():
    bars = [bar(t(9, 15), 100, 102, 98, 100, 100),    # typical 100
            bar(t(9, 20), 100, 110, 110, 110, 300)]   # typical 110 -- wait, incoherent; fix below
    bars[1] = bar(t(9, 20), 105, 112, 108, 110, 300)  # typical 110
    assert vwap(bars) == pytest.approx((100 * 100 + 110 * 300) / 400)


def test_vwap_none_without_volume():
    assert vwap([bar(t(9, 15), 1, 1, 1, 1, 0)]) is None
    assert vwap([]) is None


def test_vwap_snapshot_slope_sign():
    rising = [bar(t(9, 15) + timedelta(minutes=5 * i), p, p, p, p, 100)
              for i, p in enumerate([100, 101, 102, 103, 104, 105])]
    snap = vwap_snapshot(rising)
    assert snap is not None and snap.rising and snap.bars_used == 6


def test_atr_needs_period_plus_one_and_is_positive():
    bars = flat_bars(t(9, 15), 14)
    assert atr(bars, 14) is None
    bars = flat_bars(t(9, 15), 15)
    assert atr(bars, 14) == pytest.approx(0.4)   # range 0.4 every bar, no gaps


def test_rvol_relative_to_previous_bars():
    bars = flat_bars(t(9, 15), 6, vol=1000)
    bars.append(bar(t(9, 45), 100, 100.2, 99.8, 100, 2500))
    assert rvol(bars, lookback=5) == pytest.approx(2.5)
    assert rvol(bars[:5], lookback=5) is None


def test_ema_basic():
    assert ema([1, 2, 3, 4, 5], 5) == 3.0
    assert ema([1, 2], 5) is None


# ---- session clock ---------------------------------------------------------

def test_weekend_and_hours():
    c = SessionClock()
    sat = datetime(2026, 9, 5, 11, 0, tzinfo=IST_TZ)
    assert c.state_at(sat) is MarketState.WEEKEND
    assert c.state_at(t(9, 5)) is MarketState.PRE_OPEN
    assert c.state_at(t(9, 15)) is MarketState.OPEN
    assert c.state_at(t(15, 29)) is MarketState.OPEN
    assert c.state_at(t(15, 30)) is MarketState.CLOSED
    assert c.state_at(t(8, 0)) is MarketState.CLOSED


def test_known_holiday_and_exchange_override():
    c = SessionClock()
    gandhi = datetime(2026, 10, 2, 10, 0, tzinfo=IST_TZ)
    assert c.state_at(gandhi) is MarketState.HOLIDAY
    c.exchange_says_closed = True
    assert c.state_at(t(10, 0)) is MarketState.HOLIDAY


def test_entry_window_and_square_off():
    c = SessionClock(SessionTimes())
    assert not c.is_entry_window(t(9, 20))     # opening noise excluded
    assert c.is_entry_window(t(9, 30))
    assert c.is_entry_window(t(14, 44))
    assert not c.is_entry_window(t(14, 45))
    assert not c.is_past_square_off(t(15, 14))
    assert c.is_past_square_off(t(15, 15))


def test_last_trading_day_skips_weekend():
    c = SessionClock()
    sun = datetime(2026, 9, 6, 12, 0, tzinfo=IST_TZ)
    assert c.last_trading_day(sun) == date(2026, 9, 4)
