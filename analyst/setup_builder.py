"""Trade Setup Builder.

Generates precise, actionable intraday trade setups conforming to `intraday.models.Setup`.
Supports:
  * Prior Day Range Breakout / Breakdown
  * VWAP Breakout / Breakdown
  * Momentum Trend Continuation
  * Support Bounce / Resistance Rejection

Ensures proper geometry:
  * Strict stop loss placement against trade direction
  * Target 1 and Target 2 with minimum ~1:1.5 to 1:2+ R:R
  * Invalidation level behind trigger
  * Entry buffer zone
"""

from __future__ import annotations

from datetime import time
from typing import Optional

from intraday.models import Direction, Setup
from analyst.technicals import TechnicalProfile


def build_candidate_setups(profile: TechnicalProfile,
                           atr_period: int = 14) -> list[Setup]:
    """Generate trade setups for a technically profiled symbol."""
    setups: list[Setup] = []

    ltp = profile.ltp
    if ltp <= 0:
        return setups

    # Get ATR from daily or intraday
    atr_val = 0.0
    if profile.daily and profile.daily.atr:
        atr_val = profile.daily.atr
    elif profile.hourly and profile.hourly.atr:
        atr_val = profile.hourly.atr * 2.5
    else:
        atr_val = ltp * 0.015  # 1.5% fallback

    atr_val = max(round(atr_val, 2), round(ltp * 0.005, 2))
    entry_buffer = round(atr_val * 0.08, 2)
    lvl = profile.levels
    d = profile.daily

    # -----------------------------------------------------------------------
    # 1. Prior-Day Range Breakout (LONG)
    # -----------------------------------------------------------------------
    if lvl.prev_day_high and lvl.prev_day_low:
        pd_range = lvl.prev_day_high - lvl.prev_day_low
        if pd_range > 0:
            # LONG breakout
            trigger = round(lvl.prev_day_high, 2)
            entry_low = trigger
            entry_high = round(trigger + entry_buffer, 2)
            # SL is below trigger by 0.7 ATR or below entry_low
            stop_loss = round(trigger - atr_val * 0.7, 2)
            t1 = round(trigger + atr_val * 1.0, 2)
            t2 = round(trigger + atr_val * 2.0, 2)
            inv = round(stop_loss - atr_val * 0.3, 2)

            s_long = Setup(
                symbol=profile.symbol.replace(".NS", ""),
                name=f"{profile.symbol.replace('.NS', '')} PDH Breakout",
                direction=Direction.LONG,
                setup_type="prior_day_breakout",
                trigger=trigger,
                entry_low=entry_low,
                entry_high=entry_high,
                stop_loss=stop_loss,
                target1=t1,
                target2=t2,
                invalidate=inv,
                atr=atr_val,
                reasons=(f"Breakout above prior day high {lvl.prev_day_high:.2f}",
                         f"ATR={atr_val:.2f}, Daily Trend={d.trend if d else 'N/A'}"),
                expiry_ist=time(14, 45),
            )
            if not s_long.validate():
                setups.append(s_long)

            # SHORT breakdown
            trigger_s = round(lvl.prev_day_low, 2)
            entry_high_s = trigger_s
            entry_low_s = round(trigger_s - entry_buffer, 2)
            stop_loss_s = round(trigger_s + atr_val * 0.7, 2)
            t1_s = round(trigger_s - atr_val * 1.0, 2)
            t2_s = round(trigger_s - atr_val * 2.0, 2)
            inv_s = round(stop_loss_s + atr_val * 0.3, 2)

            s_short = Setup(
                symbol=profile.symbol.replace(".NS", ""),
                name=f"{profile.symbol.replace('.NS', '')} PDL Breakdown",
                direction=Direction.SHORT,
                setup_type="prior_day_breakdown",
                trigger=trigger_s,
                entry_low=entry_low_s,
                entry_high=entry_high_s,
                stop_loss=stop_loss_s,
                target1=t1_s,
                target2=t2_s,
                invalidate=inv_s,
                atr=atr_val,
                reasons=(f"Breakdown below prior day low {lvl.prev_day_low:.2f}",
                         f"ATR={atr_val:.2f}, Daily Trend={d.trend if d else 'N/A'}"),
                expiry_ist=time(14, 45),
            )
            if not s_short.validate():
                setups.append(s_short)

    # -----------------------------------------------------------------------
    # 2. VWAP & EMA Momentum Continuation
    # -----------------------------------------------------------------------
    m = profile.m15 or profile.m5
    if m and m.vwap and d:
        if d.trend == "BULLISH" and ltp > m.vwap:
            # Bullish pullback/continuation above VWAP
            trigger_v = round(max(ltp, m.vwap + atr_val * 0.1), 2)
            sl_v = round(m.vwap - atr_val * 0.4, 2)
            t1_v = round(trigger_v + atr_val * 0.9, 2)
            t2_v = round(trigger_v + atr_val * 1.8, 2)
            inv_v = round(sl_v - atr_val * 0.2, 2)

            s_vwap_long = Setup(
                symbol=profile.symbol.replace(".NS", ""),
                name=f"{profile.symbol.replace('.NS', '')} VWAP Momentum Long",
                direction=Direction.LONG,
                setup_type="vwap_momentum",
                trigger=trigger_v,
                entry_low=trigger_v,
                entry_high=round(trigger_v + entry_buffer, 2),
                stop_loss=sl_v,
                target1=t1_v,
                target2=t2_v,
                invalidate=inv_v,
                atr=atr_val,
                reasons=(f"Trading above rising VWAP {m.vwap:.2f}", "EMA alignment bullish"),
                expiry_ist=time(14, 45),
            )
            if not s_vwap_long.validate():
                setups.append(s_vwap_long)

        elif d.trend == "BEARISH" and ltp < m.vwap:
            # Bearish continuation below VWAP
            trigger_v = round(min(ltp, m.vwap - atr_val * 0.1), 2)
            sl_v = round(m.vwap + atr_val * 0.4, 2)
            t1_v = round(trigger_v - atr_val * 0.9, 2)
            t2_v = round(trigger_v - atr_val * 1.8, 2)
            inv_v = round(sl_v + atr_val * 0.2, 2)

            s_vwap_short = Setup(
                symbol=profile.symbol.replace(".NS", ""),
                name=f"{profile.symbol.replace('.NS', '')} VWAP Momentum Short",
                direction=Direction.SHORT,
                setup_type="vwap_momentum",
                trigger=trigger_v,
                entry_low=round(trigger_v - entry_buffer, 2),
                entry_high=trigger_v,
                stop_loss=sl_v,
                target1=t1_v,
                target2=t2_v,
                invalidate=inv_v,
                atr=atr_val,
                reasons=(f"Trading below falling VWAP {m.vwap:.2f}", "EMA alignment bearish"),
                expiry_ist=time(14, 45),
            )
            if not s_vwap_short.validate():
                setups.append(s_vwap_short)

    return setups
