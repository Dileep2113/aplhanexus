from datetime import datetime

import pytest

from conftest import DAY, bar, t
from intraday.models import DataMeta, Direction, Freshness, MarketState, Setup


def _long(**over) -> Setup:
    base = dict(symbol="X", name="X", direction=Direction.LONG, setup_type="t",
                trigger=100, entry_low=100, entry_high=100.5, stop_loss=98,
                target1=103, target2=105, invalidate=97, atr=1.0)
    base.update(over)
    return Setup(**base)


def test_valid_long_geometry():
    assert _long().validate() == []


def test_rr_uses_worst_case_entry():
    s = _long()
    # entry_reference is entry_high=100.5 for longs; risk 2.5, reward to T2 4.5
    assert s.entry_reference == 100.5
    assert s.rr() == pytest.approx(4.5 / 2.5)
    assert s.rr(s.target1) == pytest.approx(2.5 / 2.5)


def test_short_geometry_and_rr():
    s = Setup(symbol="X", name="X", direction=Direction.SHORT, setup_type="t",
              trigger=100, entry_low=99.5, entry_high=100, stop_loss=102,
              target1=97, target2=95, invalidate=103, atr=1.0)
    assert s.validate() == []
    assert s.entry_reference == 99.5
    assert s.rr() == pytest.approx(4.5 / 2.5)


@pytest.mark.parametrize("bad, fragment", [
    (dict(stop_loss=100.5), "zero risk"),
    (dict(stop_loss=101), "wrong side"),
    (dict(target1=100), "target1 is behind"),
    (dict(target2=103), "target2 does not extend"),
    (dict(target3=104), "target3 does not extend"),
    (dict(invalidate=100.5), "invalidation level is not behind"),
    (dict(atr=0), "atr must be positive"),
    (dict(entry_low=101, entry_high=100.5), "entry_low above"),
])
def test_invalid_geometry_is_rejected(bad, fragment):
    errs = _long(**bad).validate()
    assert any(fragment in e for e in errs), errs


def test_bar_coherence():
    assert bar(DAY, 10, 11, 9, 10).is_coherent()
    assert not bar(DAY, 10, 9.5, 9, 10).is_coherent()      # high below open
    assert not bar(DAY, 10, 11, 10.5, 10).is_coherent()    # low above close
    assert not bar(DAY, 10, 11, 9, 10, -1).is_coherent()   # negative volume


def test_freshness_only_realtime_is_actionable():
    assert Freshness.REALTIME.is_actionable
    for f in (Freshness.DELAYED, Freshness.HISTORICAL, Freshness.REPLAY, Freshness.STALE):
        assert not f.is_actionable


def test_datameta_never_upgrades_freshness():
    m = DataMeta("s", t(10, 0), t(9, 55), MarketState.OPEN, Freshness.DELAYED)
    assert m.degraded_to(Freshness.REALTIME, "nope").freshness is Freshness.DELAYED
    d = m.degraded_to(Freshness.STALE, "old")
    assert d.freshness is Freshness.STALE and "old" in d.warnings
    assert m.lag_seconds == 300
    assert "DELAYED" in m.label() and "lag 5.0m" in m.label()
