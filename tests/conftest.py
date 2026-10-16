import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intraday.models import Bar, DataMeta, Direction, Freshness, MarketState, Setup  # noqa: E402
from intraday.monitor import MonitorConfig  # noqa: E402
from intraday.risk import RiskLimits  # noqa: E402
from intraday.session import IST_TZ, SessionClock  # noqa: E402

# A Wednesday, well inside 2026 with no fixed-date holiday.
DAY = datetime(2026, 9, 9, tzinfo=IST_TZ)


def t(hh: int, mm: int, day: datetime = DAY) -> datetime:
    return day.replace(hour=hh, minute=mm, second=0, microsecond=0)


def bar(ts: datetime, o: float, h: float, l: float, c: float, v: float = 1000.0) -> Bar:
    return Bar(ts, o, h, l, c, v)


def flat_bars(start: datetime, n: int, px: float = 100.0, vol: float = 1000.0,
              step_min: int = 5) -> list[Bar]:
    """n identical, coherent bars -- background for RVOL and VWAP."""
    return [bar(start + timedelta(minutes=step_min * i), px, px + 0.2, px - 0.2, px, vol)
            for i in range(n)]


def meta(ts: datetime, freshness: Freshness = Freshness.DELAYED,
         state: MarketState = MarketState.OPEN) -> DataMeta:
    return DataMeta("test", ts, ts, state, freshness)


@pytest.fixture
def clock() -> SessionClock:
    return SessionClock()


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(account_capital=500_000, risk_per_trade_pct=0.5, max_daily_loss_pct=1.5,
                      max_open_positions=3, max_trades_per_day=5, max_exposure_pct=40.0)


@pytest.fixture
def mcfg() -> MonitorConfig:
    return MonitorConfig(pre_entry_atr_fraction=0.35, confirm_bars=1,
                         require_volume_confirmation=True, volume_confirm_rvol=1.2,
                         alert_cooldown_seconds=0, rvol_lookback=5)


@pytest.fixture
def long_setup() -> Setup:
    return Setup(symbol="TEST", name="Test", direction=Direction.LONG, setup_type="ORB",
                 trigger=101.0, entry_low=101.0, entry_high=101.5, stop_loss=99.0,
                 target1=104.0, target2=106.0, invalidate=98.0, atr=1.0)


@pytest.fixture
def short_setup() -> Setup:
    return Setup(symbol="TEST", name="Test", direction=Direction.SHORT, setup_type="ORB",
                 trigger=99.0, entry_low=98.5, entry_high=99.0, stop_loss=101.0,
                 target1=96.0, target2=94.0, invalidate=102.0, atr=1.0)
