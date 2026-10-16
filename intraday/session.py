"""NSE session clock.

Decides what the market is doing *right now* from the wall clock, so the
monitor can refuse to arm setups on weekends, before 09:15, or after the
configured square-off.  Holidays come from an editable list; when the list
is silent we fall back to whatever nseindia.com's marketStatus says, and
if that is unavailable too we assume a weekday is a trading day and flag it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from .models import IST, MarketState

IST_TZ = ZoneInfo(IST)

# NSE trading holidays.  Extend per the exchange circular each December;
# anything missing here is caught at runtime by the marketStatus fallback.
# Only 2026 dates that are unambiguous fixed-date holidays are hardcoded;
# lunar-calendar holidays vary and should come from the exchange feed.
KNOWN_HOLIDAYS: frozenset[date] = frozenset({
    date(2026, 1, 26),   # Republic Day
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
})


def now_ist() -> datetime:
    return datetime.now(IST_TZ)


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class SessionTimes:
    pre_open: time = time(9, 0)
    open: time = time(9, 15)
    close: time = time(15, 30)
    no_trade_until: time = time(9, 30)
    entry_expiry: time = time(14, 45)
    square_off: time = time(15, 15)

    @classmethod
    def from_config(cls, cfg: dict) -> "SessionTimes":
        s = cfg.get("session", {})
        m = cfg.get("monitor", {})
        return cls(
            pre_open=parse_hhmm(s.get("pre_open_ist", "09:00")),
            open=parse_hhmm(s.get("open_ist", "09:15")),
            close=parse_hhmm(s.get("close_ist", "15:30")),
            no_trade_until=parse_hhmm(s.get("no_trade_until_ist", "09:30")),
            entry_expiry=parse_hhmm(m.get("entry_expiry_ist", "14:45")),
            square_off=parse_hhmm(m.get("square_off_ist", "15:15")),
        )


class SessionClock:
    """Answers "is the market open?" for a given instant."""

    def __init__(self, times: SessionTimes = SessionTimes(),
                 holidays: Iterable[date] = KNOWN_HOLIDAYS,
                 exchange_says_closed: Optional[bool] = None):
        self.times = times
        self.holidays = frozenset(holidays)
        # Set by the NSE context provider when it has a fresh marketStatus.
        self.exchange_says_closed = exchange_says_closed

    def state_at(self, ts: Optional[datetime] = None) -> MarketState:
        ts = (ts or now_ist()).astimezone(IST_TZ)
        d, t = ts.date(), ts.time()
        if d.weekday() >= 5:
            return MarketState.WEEKEND
        if d in self.holidays:
            return MarketState.HOLIDAY
        if self.times.pre_open <= t < self.times.open:
            return MarketState.PRE_OPEN
        if self.times.open <= t < self.times.close:
            # A holiday we didn't know about: trust the exchange over our list.
            if self.exchange_says_closed:
                return MarketState.HOLIDAY
            return MarketState.OPEN
        return MarketState.CLOSED

    def is_entry_window(self, ts: Optional[datetime] = None) -> bool:
        """Between the opening-noise cutoff and the entry expiry."""
        ts = (ts or now_ist()).astimezone(IST_TZ)
        if self.state_at(ts) is not MarketState.OPEN:
            return False
        return self.times.no_trade_until <= ts.time() < self.times.entry_expiry

    def is_past_square_off(self, ts: Optional[datetime] = None) -> bool:
        ts = (ts or now_ist()).astimezone(IST_TZ)
        return ts.time() >= self.times.square_off

    def session_open_dt(self, ts: Optional[datetime] = None) -> datetime:
        ts = (ts or now_ist()).astimezone(IST_TZ)
        return ts.replace(hour=self.times.open.hour, minute=self.times.open.minute,
                          second=0, microsecond=0)

    def last_trading_day(self, ts: Optional[datetime] = None) -> date:
        """Most recent date that was (or is) a trading day."""
        d = (ts or now_ist()).astimezone(IST_TZ).date()
        while d.weekday() >= 5 or d in self.holidays:
            d -= timedelta(days=1)
        return d
