"""State-machine tests.  Every scenario feeds closed bars into SetupMonitor
with a REPLAY meta (so every bar counts as closed) and asserts the alerts."""

from datetime import timedelta

import pytest

from conftest import DAY, bar, flat_bars, meta, t
from intraday.models import AlertKind, Direction, Freshness, MarketState, Setup, SetupState
from intraday.monitor import MonitorBook, MonitorConfig, SetupMonitor
from intraday.risk import DayLedger


def _mon(setup, clock, limits, mcfg) -> SetupMonitor:
    return SetupMonitor(setup, clock, limits, DayLedger(), mcfg)


def _feed(mon, bars, freshness=Freshness.REPLAY, state=MarketState.OPEN):
    """Push bars one at a time as a live loop would; collect alerts."""
    alerts = []
    for i in range(2, len(bars) + 1):   # a 1-bar window is (correctly) a DATA_WARNING
        window = bars[:i]
        alerts += mon.update(window, meta(window[-1].ts, freshness, state), window[-1].ts)
    return alerts


def _background(n=8, px=100.0, start=None):
    # Background bars at 09:30+ so we're inside the entry window.
    return flat_bars(start or t(9, 30), n, px=px, vol=1000)


# --------------------------------------------------------------------------

def test_long_full_lifecycle_to_target2(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [
        bar(nxt, 100.6, 100.9, 100.5, 100.8, 1000),                  # approaching (0.2 ATR away)
        bar(nxt + timedelta(minutes=5), 100.9, 101.4, 100.8, 101.3, 2000),   # closes > trigger, RVOL 2
        bar(nxt + timedelta(minutes=10), 101.3, 104.2, 101.2, 104.0, 1500),  # tags T1
        bar(nxt + timedelta(minutes=15), 104.0, 106.5, 103.9, 106.2, 1500),  # tags T2
    ]
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds == [AlertKind.PRE_ENTRY, AlertKind.ENTRY, AlertKind.TARGET1, AlertKind.TARGET2]
    assert mon.setup.state is SetupState.CLOSED_TARGET
    assert mon.position.exit_price == long_setup.target2
    assert mon.ledger.realised_pnl > 0 and mon.ledger.open_positions == 0
    assert mon.position.mfe >= 106.5 - 101.3 - 1e-9


def test_entry_is_not_actionable_on_delayed_data(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [bar(nxt, 100.9, 101.4, 100.8, 101.3, 2000),
             bar(nxt + timedelta(minutes=5), 101.3, 101.5, 101.2, 101.4, 1000)]  # forming bar
    alerts = _feed(mon, bars, freshness=Freshness.DELAYED)
    entry = [a for a in alerts if a.kind is AlertKind.ENTRY]
    assert entry and entry[0].actionable is False
    assert "observed" in entry[0].message and "INFORMATIONAL" in entry[0].render()


def test_entry_is_actionable_only_on_realtime(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [bar(nxt, 100.9, 101.4, 100.8, 101.3, 2000),
             bar(nxt + timedelta(minutes=5), 101.3, 101.5, 101.2, 101.4, 1000)]
    alerts = _feed(mon, bars, freshness=Freshness.REALTIME)
    entry = [a for a in alerts if a.kind is AlertKind.ENTRY]
    assert entry and entry[0].actionable is True
    assert "confirmed" in entry[0].message


def test_live_mode_ignores_forming_bar(long_setup, clock, limits, mcfg):
    """Market OPEN + DELAYED: the last bar is forming and must not trigger."""
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background() + [bar(t(10, 10), 100.9, 101.4, 100.8, 101.3, 2000)]
    alerts = mon.update(bars, meta(bars[-1].ts, Freshness.DELAYED), bars[-1].ts)
    assert not any(a.kind is AlertKind.ENTRY for a in alerts)
    assert mon.setup.state is SetupState.WATCHING


def test_stop_loss_closes_position(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [bar(nxt, 100.9, 101.4, 100.8, 101.3, 2000),
             bar(nxt + timedelta(minutes=5), 101.3, 101.5, 98.8, 99.0, 3000)]
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds == [AlertKind.ENTRY, AlertKind.STOP_LOSS]
    assert mon.setup.state is SetupState.CLOSED_STOP
    assert mon.position.exit_price == long_setup.stop_loss
    assert mon.ledger.realised_pnl < 0


def test_bar_spanning_stop_and_target_assumes_stop(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [bar(nxt, 100.9, 101.4, 100.8, 101.3, 2000),
             bar(nxt + timedelta(minutes=5), 101.3, 107.0, 98.5, 105.0, 5000)]
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds[-1] is AlertKind.STOP_LOSS


def test_no_entry_without_volume(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [bar(nxt, 100.9, 101.4, 100.8, 101.3, 1000)]   # RVOL 1.0 < 1.2
    alerts = _feed(mon, bars)
    assert mon.setup.state in (SetupState.WATCHING, SetupState.PRE_ENTRY)
    assert alerts and alerts[-1].kind is AlertKind.PRE_ENTRY and "RVOL" in alerts[-1].message


def test_chasing_beyond_entry_band_is_refused(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [bar(nxt, 100.9, 102.5, 100.8, 102.4, 3000)]   # close 102.4 > entry_high 101.5
    alerts = _feed(mon, bars)
    assert mon.setup.state is not SetupState.ACTIVE
    assert "chasing" in alerts[-1].message


def test_invalidation_before_entry(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background() + [bar(t(10, 10), 99.0, 99.2, 97.5, 97.8, 3000)]
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds == [AlertKind.INVALIDATION]
    assert mon.setup.state is SetupState.INVALIDATED
    # Terminal: further bars produce nothing.
    assert mon.update(bars + [bar(t(10, 15), 101, 105, 101, 105, 9000)],
                      meta(t(10, 15), Freshness.REPLAY), t(10, 15)) == []


def test_expiry_without_trigger(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background(start=t(14, 0), n=8) + [bar(t(14, 45), 100, 100.2, 99.8, 100, 1000)]
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds == [AlertKind.EXPIRY]
    assert mon.setup.state is SetupState.EXPIRED


def test_no_entry_in_opening_noise_window(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = flat_bars(t(9, 15), 2, vol=1000) + [bar(t(9, 25), 100.9, 101.4, 100.8, 101.3, 9000)]
    # Not enough bars for RVOL anyway; make lookback tiny so the time rule is what fails.
    mcfg.rvol_lookback = 2
    alerts = _feed(mon, bars)
    assert mon.setup.state is not SetupState.ACTIVE
    assert any("entry window" in a.message for a in alerts)


def test_square_off_forces_exit(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background(start=t(14, 0))
    nxt = bars[-1].ts + timedelta(minutes=5)          # 14:40
    bars += [bar(nxt, 100.9, 101.4, 100.8, 101.3, 2000),
             bar(t(15, 15), 101.3, 101.6, 101.1, 101.5, 1000)]
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds == [AlertKind.ENTRY, AlertKind.SQUARE_OFF]
    assert mon.setup.state.is_terminal


def test_stale_feed_freezes_and_warns_once(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    bars = _background() + [bar(t(10, 10), 100.9, 101.4, 100.8, 101.3, 2000)]
    a1 = mon.update(bars, meta(t(10, 40), Freshness.STALE), t(10, 40))
    a2 = mon.update(bars, meta(t(10, 41), Freshness.STALE), t(10, 41))
    assert [a.kind for a in a1] == [AlertKind.DATA_WARNING]
    assert a2 == []
    assert mon.setup.state is SetupState.WATCHING


def test_short_lifecycle(short_setup, clock, limits, mcfg):
    mon = _mon(short_setup, clock, limits, mcfg)
    bars = _background()
    nxt = bars[-1].ts + timedelta(minutes=5)
    bars += [bar(nxt, 99.3, 99.4, 98.6, 98.7, 2000),                    # closes < 99 trigger, RVOL 2
             bar(nxt + timedelta(minutes=5), 98.7, 98.8, 95.8, 96.0, 1500),   # T1 96
             bar(nxt + timedelta(minutes=10), 96.0, 96.1, 93.5, 93.8, 1500)]  # T2 94
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds == [AlertKind.ENTRY, AlertKind.TARGET1, AlertKind.TARGET2]
    assert mon.ledger.realised_pnl > 0


def test_risk_limits_block_entry(long_setup, clock, limits, mcfg):
    mon = _mon(long_setup, clock, limits, mcfg)
    mon.ledger.trades_taken = limits.max_trades_per_day
    bars = _background() + [bar(t(10, 10), 100.9, 101.4, 100.8, 101.3, 2000)]
    kinds = [a.kind for a in _feed(mon, bars)]
    assert kinds == [AlertKind.INVALIDATION]
    assert "risk limits forbid" in mon.history[-1].message


def test_book_rejects_bad_geometry(clock, limits, mcfg, long_setup):
    bad = Setup(symbol="BAD", name="Bad", direction=Direction.LONG, setup_type="t",
                trigger=100, entry_low=100, entry_high=100, stop_loss=100,
                target1=101, target2=102, invalidate=99, atr=1)
    book = MonitorBook([long_setup, bad], clock, limits, mcfg)
    assert "TEST" in book.monitors and "BAD" in book.rejected
    assert "REJECTED" in book.summary()
