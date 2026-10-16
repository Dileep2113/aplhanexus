"""Ready-made setup presets.

Each function takes bars already fetched by the caller and returns a
:class:`~intraday.models.Setup` derived purely from price structure.  The
monitor decides whether the setup actually triggers; these functions only
compute the levels.

Currently implemented
---------------------
``previous_day_range_breakout``
    The prior session's High (for LONG) or Low (for SHORT) is the trigger.
    Stop sits one ATR below/above the prior day's range.  Targets at 1×, 2×
    and optionally 3× the prior-day range.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

from .indicators import atr as _calc_atr, session_bars
from .models import Bar, Direction, Setup


# ---------------------------------------------------------------------------

def previous_day_range_breakout(
    symbol: str,
    name: str,
    bars: Sequence[Bar],
    session: date,
    direction: Direction,
    atr_period: int = 14,
    t1_r: float = 1.0,
    t2_r: float = 2.0,
    t3_r: Optional[float] = None,
    entry_buffer_atr: float = 0.05,
) -> Setup:
    """Build a previous-day range-breakout setup.

    Parameters
    ----------
    symbol:
        NSE symbol (e.g. ``"RELIANCE"``).
    name:
        Display name (may be the same as symbol).
    bars:
        All intraday bars available for this symbol (sorted ascending).
        Must include at least the prior session.
    session:
        The *current* trading date.  Bars from the day *before* this date
        are used to compute the prior-day High/Low/ATR.
    direction:
        LONG (breakout above prior High) or SHORT (breakdown below prior Low).
    atr_period:
        Lookback for the Wilder ATR computed across *all* provided bars.
    t1_r / t2_r / t3_r:
        Target multiples as a fraction of the prior-day range.
    entry_buffer_atr:
        Fraction of ATR added to the trigger to form the entry band.

    Raises
    ------
    ValueError
        When there is not enough prior-day data to compute a setup.
    """
    # --- prior-day bars ---------------------------------------------------
    prior_bars = [b for b in bars if b.ts.date() < session]
    if not prior_bars:
        raise ValueError(f"no prior-day bars before {session}")

    last_prior_date = max(b.ts.date() for b in prior_bars)
    pd_bars = session_bars(prior_bars, last_prior_date)
    if not pd_bars:
        raise ValueError(f"no bars on prior day {last_prior_date}")

    pd_high = max(b.high for b in pd_bars)
    pd_low = min(b.low for b in pd_bars)
    pd_range = pd_high - pd_low
    if pd_range <= 0:
        raise ValueError(f"prior-day range is zero ({pd_high:.2f})")

    # --- ATR across all supplied bars ------------------------------------
    atr_val = _calc_atr(list(bars), atr_period)
    if atr_val is None or atr_val <= 0:
        # Fall back to a fraction of the prior-day range.
        atr_val = pd_range / 2.0

    buf = atr_val * entry_buffer_atr

    # --- LONG: breakout above prior High ---------------------------------
    if direction is Direction.LONG:
        trigger = pd_high
        entry_low = trigger
        entry_high = trigger + buf
        stop_loss = pd_low - atr_val * 0.5
        invalidate = pd_low - atr_val          # structure collapses
        target1 = trigger + pd_range * t1_r
        target2 = trigger + pd_range * t2_r
        target3 = (trigger + pd_range * t3_r) if t3_r is not None else None

    # --- SHORT: breakdown below prior Low --------------------------------
    else:
        trigger = pd_low
        entry_high = trigger
        entry_low = trigger - buf
        stop_loss = pd_high + atr_val * 0.5
        invalidate = pd_high + atr_val
        target1 = trigger - pd_range * t1_r
        target2 = trigger - pd_range * t2_r
        target3 = (trigger - pd_range * t3_r) if t3_r is not None else None

    return Setup(
        symbol=symbol.upper(),
        name=name,
        direction=direction,
        setup_type="prev_day_range_breakout",
        trigger=round(trigger, 2),
        entry_low=round(entry_low, 2),
        entry_high=round(entry_high, 2),
        stop_loss=round(stop_loss, 2),
        target1=round(target1, 2),
        target2=round(target2, 2),
        target3=round(target3, 2) if target3 is not None else None,
        invalidate=round(invalidate, 2),
        atr=round(atr_val, 2),
        reasons=(f"prior-day range {pd_range:.2f}  "
                 f"H={pd_high:.2f} L={pd_low:.2f}  "
                 f"ATR={atr_val:.2f}",),
    )
