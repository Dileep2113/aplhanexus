"""Core value objects for the intraday monitoring loop.

Everything crossing a module boundary is one of these types.  The important
one is :class:`DataMeta`: no price travels through this system without a
companion ``DataMeta`` recording where it came from and how far behind live
it is (spec section 27).  Alerts carry it through to the console so a
delayed number can never be read as a live one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional

IST = "Asia/Kolkata"


class MarketState(str, Enum):
    PRE_OPEN = "PRE_OPEN"       # 09:00-09:15 IST call auction
    OPEN = "OPEN"
    CLOSED = "CLOSED"           # weekday, outside session hours
    WEEKEND = "WEEKEND"
    HOLIDAY = "HOLIDAY"

    @property
    def is_tradable(self) -> bool:
        return self is MarketState.OPEN


class Freshness(str, Enum):
    """How far behind live a price is.  Never promote a value up this list."""

    REALTIME = "REALTIME"       # broker tick feed; needs credentials
    DELAYED = "DELAYED"         # free feed lagging by minutes
    HISTORICAL = "HISTORICAL"   # completed session, market shut
    REPLAY = "REPLAY"           # simulated from stored bars
    STALE = "STALE"             # older than the configured tolerance

    @property
    def is_actionable(self) -> bool:
        """True only when fresh enough to act on a live entry."""
        return self is Freshness.REALTIME


@dataclass(frozen=True)
class DataMeta:
    """Provenance for a price or bar series (spec section 27)."""

    source: str                      # "yahoo:RELIANCE.NS", "nse:allIndices", ...
    fetched_at: datetime             # tz-aware, when we made the call
    data_timestamp: Optional[datetime]  # tz-aware, the stamp on the data itself
    market_state: MarketState
    freshness: Freshness
    warnings: tuple[str, ...] = ()

    @property
    def lag_seconds(self) -> Optional[float]:
        if self.data_timestamp is None:
            return None
        return (self.fetched_at - self.data_timestamp).total_seconds()

    def label(self) -> str:
        """One-line provenance string; goes on every alert and report."""
        lag = self.lag_seconds
        lag_txt = "lag unknown" if lag is None else f"lag {lag / 60:.1f}m"
        stamp = "no timestamp"
        if self.data_timestamp is not None:
            stamp = self.data_timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")
        parts = [self.source, self.freshness.value, self.market_state.value,
                 stamp, lag_txt]
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return " | ".join(parts)

    def with_warning(self, *msgs: str) -> "DataMeta":
        return DataMeta(self.source, self.fetched_at, self.data_timestamp,
                        self.market_state, self.freshness,
                        self.warnings + tuple(msgs))

    def degraded_to(self, freshness: Freshness, why: str) -> "DataMeta":
        """Downgrade freshness. Refuses to upgrade -- that would be a lie."""
        order = [Freshness.REALTIME, Freshness.DELAYED, Freshness.HISTORICAL,
                 Freshness.REPLAY, Freshness.STALE]
        if order.index(freshness) < order.index(self.freshness):
            return self
        return DataMeta(self.source, self.fetched_at, self.data_timestamp,
                        self.market_state, freshness, self.warnings + (why,))


@dataclass(frozen=True)
class Bar:
    """One OHLCV candle.  ``ts`` is the bar's opening time, tz-aware IST."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def typical(self) -> float:
        """HLC/3 -- the price VWAP accumulates against."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low

    def is_coherent(self) -> bool:
        """OHLC self-consistency. Free feeds do serve broken bars."""
        return (self.high >= max(self.open, self.close)
                and self.low <= min(self.open, self.close)
                and self.high >= self.low
                and self.volume >= 0)


@dataclass(frozen=True)
class Quote:
    """Latest known price for a symbol, plus the session aggregates we need."""

    symbol: str
    ltp: float
    meta: DataMeta
    previous_close: Optional[float] = None
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[float] = None
    vwap: Optional[float] = None

    @property
    def change_pct(self) -> Optional[float]:
        if not self.previous_close:
            return None
        return (self.ltp - self.previous_close) / self.previous_close * 100.0


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short.  Lets one formula serve both sides."""
        return 1 if self is Direction.LONG else -1


class SetupState(str, Enum):
    """Lifecycle of a monitored setup (spec section 21)."""

    WATCHING = "WATCHING"          # armed, price away from the trigger
    PRE_ENTRY = "PRE_ENTRY"        # price approaching the trigger
    ACTIVE = "ACTIVE"              # entry confirmed, position notionally open
    T1_HIT = "T1_HIT"              # first target tagged, still running
    CLOSED_TARGET = "CLOSED_TARGET"
    CLOSED_STOP = "CLOSED_STOP"
    INVALIDATED = "INVALIDATED"    # structure broke before entry
    EXPIRED = "EXPIRED"            # never triggered inside its window

    @property
    def is_terminal(self) -> bool:
        return self in (SetupState.CLOSED_TARGET, SetupState.CLOSED_STOP,
                        SetupState.INVALIDATED, SetupState.EXPIRED)

    @property
    def is_open(self) -> bool:
        return self in (SetupState.ACTIVE, SetupState.T1_HIT)


class AlertKind(str, Enum):
    PRE_ENTRY = "PRE_ENTRY"
    ENTRY = "ENTRY"
    TARGET1 = "TARGET1"
    TARGET2 = "TARGET2"
    STOP_LOSS = "STOP_LOSS"
    INVALIDATION = "INVALIDATION"
    EXPIRY = "EXPIRY"
    SQUARE_OFF = "SQUARE_OFF"
    DATA_WARNING = "DATA_WARNING"


@dataclass
class Setup:
    """A trade plan handed to the monitor.

    The monitor never invents levels; it only watches the ones defined here
    and reports what price does to them.  ``trigger`` is the price that must
    be taken out on a closed bar, ``invalidate`` cancels the idea outright.
    """

    symbol: str
    name: str
    direction: Direction
    setup_type: str
    trigger: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target1: float
    target2: float
    invalidate: float
    atr: float
    target3: Optional[float] = None
    score: Optional[float] = None
    reasons: tuple[str, ...] = ()
    expiry_ist: Optional[time] = None
    state: SetupState = SetupState.WATCHING

    # --- geometry -------------------------------------------------------
    @property
    def risk_per_share(self) -> float:
        """Always positive: distance from entry to stop."""
        return abs(self.entry_reference - self.stop_loss)

    @property
    def entry_reference(self) -> float:
        """Price used for R:R maths -- the far edge of the entry band, so the
        published ratio is the worst case a filled order would see."""
        return self.entry_high if self.direction is Direction.LONG else self.entry_low

    def reward_per_share(self, target: float) -> float:
        return abs(target - self.entry_reference)

    def rr(self, target: Optional[float] = None) -> Optional[float]:
        tgt = self.target2 if target is None else target
        risk = self.risk_per_share
        if risk <= 0:
            return None
        return self.reward_per_share(tgt) / risk

    def expected_move_pct(self, target: Optional[float] = None) -> float:
        tgt = self.target2 if target is None else target
        return abs(tgt - self.entry_reference) / self.entry_reference * 100.0

    def validate(self) -> list[str]:
        """Structural checks. A non-empty result means do not monitor this."""
        errs: list[str] = []
        s = self.direction.sign
        if self.entry_low > self.entry_high:
            errs.append("entry_low above entry_high")
        if self.atr <= 0:
            errs.append("atr must be positive")
        if self.risk_per_share <= 0:
            errs.append("stop-loss coincides with entry: zero risk is not a trade")
        # Stop must sit against the direction of the trade.
        if s * (self.entry_reference - self.stop_loss) <= 0:
            errs.append(f"{self.direction.value} stop-loss is on the wrong side of entry")
        # Targets must advance with the trade.
        if s * (self.target1 - self.entry_reference) <= 0:
            errs.append("target1 is behind entry")
        if s * (self.target2 - self.target1) <= 0:
            errs.append("target2 does not extend past target1")
        if self.target3 is not None and s * (self.target3 - self.target2) <= 0:
            errs.append("target3 does not extend past target2")
        # Trigger must be reachable in the direction of the trade.
        if s * (self.entry_high if s > 0 else self.entry_low) < s * self.trigger:
            errs.append("entry band does not contain or follow the trigger")
        if s * (self.trigger - self.invalidate) <= 0:
            errs.append("invalidation level is not behind the trigger")
        return errs


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    symbol: str
    message: str
    price: Optional[float]
    at: datetime
    meta: DataMeta
    actionable: bool = False

    def render(self) -> str:
        px = "" if self.price is None else f" @ {self.price:,.2f}"
        flag = "" if self.actionable else "  [INFORMATIONAL - data not real-time]"
        return (f"[{self.at.strftime('%H:%M:%S')}] {self.kind.value:<13} "
                f"{self.symbol:<14}{px}  {self.message}{flag}\n"
                f"    data: {self.meta.label()}")
