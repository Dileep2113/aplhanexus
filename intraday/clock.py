"""MarketClock -- thin extension of SessionClock with helpers used by run.py.

Adds:
* ``to_ist`` -- convert any datetime (or naive, assumed IST) to IST.
* ``MarketClock.from_config`` -- create from config.yaml dict + optional
  holiday list fetched from the NSE context provider.
* ``MarketClock.phase_at`` -- richer 'phase' string for the banner.
* ``MarketClock.state_warnings`` -- non-fatal warnings for the banner.
* ``MarketClock.trading_holidays`` -- stub; real holidays come from NSE.

The symbols ``IST_TZ`` and ``to_ist`` are imported by ``run.py`` from this
module; ``SessionClock`` itself lives in ``session.py``.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Iterable, Optional

from .models import MarketState
from .session import IST_TZ, SessionClock, SessionTimes, now_ist  # re-export IST_TZ


def to_ist(dt: datetime) -> datetime:
    """Return *dt* expressed in IST.

    * Tz-aware inputs are converted.
    * Naive inputs are assumed to already be in IST (localised, not converted).
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST_TZ)
    return dt.astimezone(IST_TZ)


class Phase(str, Enum):
    """Finer-grained market phase used in the run.py banner."""
    PRE_OPEN = "PRE_OPEN"
    NO_TRADE_WINDOW = "NO_TRADE_WINDOW"   # 09:15-09:30 opening noise
    OPEN = "OPEN"
    ENTRY_CLOSED = "ENTRY_CLOSED"          # 14:45-15:15 -- setups no longer arm
    SQUARE_OFF = "SQUARE_OFF"              # 15:15-15:30 -- forced exits
    CLOSED = "CLOSED"
    WEEKEND = "WEEKEND"
    HOLIDAY = "HOLIDAY"


class MarketClock(SessionClock):
    """SessionClock with additional helpers expected by *run.py*."""

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict,
                    holidays: Optional[Iterable[date]] = None) -> "MarketClock":
        """Factory that reads ``session:`` and ``monitor:`` sections."""
        times = SessionTimes.from_config(cfg)
        from .session import KNOWN_HOLIDAYS
        hol = frozenset(holidays) if holidays is not None else KNOWN_HOLIDAYS
        obj = cls(times=times, holidays=hol)
        return obj

    # ------------------------------------------------------------------
    def phase_at(self, now: datetime) -> Phase:
        """Granular phase for the banner."""
        now = to_ist(now)
        state = self.state_at(now)
        if state is MarketState.WEEKEND:
            return Phase.WEEKEND
        if state is MarketState.HOLIDAY:
            return Phase.HOLIDAY
        if state is MarketState.PRE_OPEN:
            return Phase.PRE_OPEN
        if state is MarketState.CLOSED:
            return Phase.CLOSED
        # OPEN window -- further subdivide by intraday rules
        t = now.time()
        if t < self.times.no_trade_until:
            return Phase.NO_TRADE_WINDOW
        if t < self.times.entry_expiry:
            return Phase.OPEN
        if t < self.times.square_off:
            return Phase.ENTRY_CLOSED
        return Phase.SQUARE_OFF

    # ------------------------------------------------------------------
    def state_warnings(self, now: datetime) -> list[str]:
        """Non-fatal informational strings for the banner."""
        warns: list[str] = []
        now = to_ist(now)
        state = self.state_at(now)
        if state is MarketState.OPEN and self.exchange_says_closed is None:
            warns.append("exchange holiday calendar not loaded; weekday holidays may be missed")
        if state is MarketState.OPEN and now.time() < self.times.no_trade_until:
            warns.append("inside opening 15-minute noise window -- no entries until 09:30")
        if state is MarketState.OPEN and now.time() >= self.times.entry_expiry:
            warns.append("entry window closed -- setups in WATCHING/PRE_ENTRY will not arm")
        return warns

    # ------------------------------------------------------------------
    def trading_holidays(self) -> Optional[list[date]]:
        """Return the known holiday list, or None if the live feed is used.

        This stub always returns the static set.  When the NSE context
        provider is available (``run.py`` passes ``nse.trading_holidays()``),
        that result is passed directly to ``from_config`` instead of calling
        this method.
        """
        return sorted(self.holidays)
