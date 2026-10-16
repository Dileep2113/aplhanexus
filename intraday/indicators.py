"""Indicators the monitor needs, computed from :class:`Bar` sequences.

Kept dependency-free on purpose (no pandas-ta): these are few, they must be
exactly reproducible in tests, and the monitor calls them every poll.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from .models import Bar


def session_bars(bars: Sequence[Bar], day: date) -> list[Bar]:
    """Bars belonging to one trading date (IST)."""
    return [b for b in bars if b.ts.date() == day]


def vwap(bars: Sequence[Bar]) -> Optional[float]:
    """Volume-weighted average of typical price over the given bars.

    Callers pass the *current session's* bars only -- VWAP resets at 09:15.
    Returns None when there is no volume to weight against.
    """
    pv = 0.0
    vol = 0.0
    for b in bars:
        pv += b.typical * b.volume
        vol += b.volume
    if vol <= 0:
        return None
    return pv / vol


@dataclass(frozen=True)
class VwapSnapshot:
    value: float
    slope: float          # per-bar change over the last few bars
    bars_used: int

    @property
    def rising(self) -> bool:
        return self.slope > 0

    @property
    def falling(self) -> bool:
        return self.slope < 0


def vwap_snapshot(bars: Sequence[Bar], slope_lookback: int = 3) -> Optional[VwapSnapshot]:
    """VWAP now and its recent slope (VWAP a few bars ago vs now)."""
    if not bars:
        return None
    now = vwap(bars)
    if now is None:
        return None
    if len(bars) <= slope_lookback:
        return VwapSnapshot(now, 0.0, len(bars))
    earlier = vwap(bars[:-slope_lookback])
    slope = 0.0 if earlier is None else (now - earlier) / slope_lookback
    return VwapSnapshot(now, slope, len(bars))


def true_range(prev_close: Optional[float], bar: Bar) -> float:
    if prev_close is None:
        return bar.range
    return max(bar.high - bar.low,
               abs(bar.high - prev_close),
               abs(bar.low - prev_close))


def atr(bars: Sequence[Bar], period: int = 14) -> Optional[float]:
    """Wilder ATR.  Needs at least ``period + 1`` bars."""
    if len(bars) < period + 1:
        return None
    trs = [true_range(bars[i - 1].close, bars[i]) for i in range(1, len(bars))]
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def ema(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def rvol(bars: Sequence[Bar], lookback: int = 20) -> Optional[float]:
    """Last bar's volume relative to the mean of the preceding ``lookback``.

    This is bar-level relative volume, used to confirm a trigger bar.  It
    is deliberately not the day-level RVOL a scanner would use.
    """
    if len(bars) < lookback + 1:
        return None
    ref = bars[-lookback - 1:-1]
    avg = sum(b.volume for b in ref) / lookback
    if avg <= 0:
        return None
    return bars[-1].volume / avg


def coherent_only(bars: Sequence[Bar]) -> tuple[list[Bar], int]:
    """Drop OHLC-inconsistent bars. Returns (kept, dropped_count)."""
    kept = [b for b in bars if b.is_coherent()]
    return kept, len(bars) - len(kept)
