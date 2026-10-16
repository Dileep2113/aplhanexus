"""Multi-timeframe technical analysis engine.

For every candidate symbol, downloads bars across 5 timeframes and computes:
  * Trend: EMA 9/20/50/200 + slope
  * Momentum: RSI(14), MACD(12,26,9)
  * Volatility: ATR(14), Bollinger Bands(20,2)
  * Volume: Relative Volume (RVOL vs 20-bar avg), OBV trend
  * Level: VWAP (intraday), Prior Day High/Low, Weekly High/Low
  * Strength: ADX(14)
  * Multi-timeframe alignment score (0–10)

All computation is done from OHLCV bar lists — no new API calls once bars
are fetched. Uses the existing `intraday.indicators` functions where possible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Sequence
import math


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class LevelSet:
    """Key price levels for a symbol."""
    prev_day_high: Optional[float] = None
    prev_day_low:  Optional[float] = None
    week_high:     Optional[float] = None
    week_low:      Optional[float] = None
    month_high:    Optional[float] = None
    month_low:     Optional[float] = None
    support:       Optional[float] = None   # nearest notable support
    resistance:    Optional[float] = None   # nearest notable resistance


@dataclass
class TimeframeView:
    name: str          # "Daily" | "Weekly" | "1H" | "15M" | "5M"
    interval: str      # yfinance interval string
    bars_count: int = 0
    trend: str = "NEUTRAL"          # BULLISH | BEARISH | NEUTRAL
    ema9:   Optional[float] = None
    ema20:  Optional[float] = None
    ema50:  Optional[float] = None
    ema200: Optional[float] = None
    rsi:    Optional[float] = None
    macd:   Optional[float] = None  # MACD line
    macd_signal: Optional[float] = None
    macd_hist:   Optional[float] = None
    atr:    Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_mid:   Optional[float] = None
    adx:    Optional[float] = None
    rvol:   Optional[float] = None  # relative volume (last bar)
    last_close: Optional[float] = None
    vwap:   Optional[float] = None  # intraday only
    above_vwap: Optional[bool] = None

    def trend_score(self) -> int:
        """0=bearish, 1=neutral, 2=bullish."""
        if self.trend == "BULLISH":
            return 2
        if self.trend == "BEARISH":
            return 0
        return 1


@dataclass
class TechnicalProfile:
    symbol: str
    ltp: float                            # last traded price
    levels: LevelSet = field(default_factory=LevelSet)
    daily:  Optional[TimeframeView] = None
    weekly: Optional[TimeframeView] = None
    hourly: Optional[TimeframeView] = None
    m15:    Optional[TimeframeView] = None
    m5:     Optional[TimeframeView] = None
    mtf_score: int = 0          # multi-timeframe alignment: 0–10
    setup_signals: list[str] = field(default_factory=list)  # detected patterns
    data_freshness: str = "DELAYED"

    # -----------------------------------------------------------------------
    def compute_mtf_score(self) -> None:
        """Score 0-10 based on how many timeframes agree on direction."""
        views = [v for v in (self.daily, self.weekly, self.hourly, self.m15, self.m5)
                 if v is not None]
        if not views:
            self.mtf_score = 0
            return
        scores = [v.trend_score() for v in views]
        bull = scores.count(2)
        bear = scores.count(0)
        n = len(scores)
        if bull > bear:
            self.mtf_score = round(bull / n * 10)
        elif bear > bull:
            self.mtf_score = round((n - bear) / n * 10)
        else:
            self.mtf_score = 5

    def dominant_trend(self) -> str:
        views = [v for v in (self.daily, self.weekly) if v is not None]
        bulls = sum(1 for v in views if v.trend == "BULLISH")
        bears = sum(1 for v in views if v.trend == "BEARISH")
        if bulls > bears:
            return "BULLISH"
        if bears > bulls:
            return "BEARISH"
        return "NEUTRAL"

    def render_summary(self) -> str:
        lines = [f"  {self.symbol}  LTP={self.ltp:,.2f}  MTF={self.mtf_score}/10"
                 f"  Trend={self.dominant_trend()}"]
        for v in (self.daily, self.m15, self.m5):
            if v:
                rsi = f"RSI={v.rsi:.1f}" if v.rsi else ""
                rvol = f"RVOL={v.rvol:.2f}" if v.rvol else ""
                adx = f"ADX={v.adx:.1f}" if v.adx else ""
                lines.append(f"    [{v.name}] trend={v.trend:8}  {rsi}  {rvol}  {adx}")
        if self.setup_signals:
            lines.append(f"    signals: {', '.join(self.setup_signals)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure indicator implementations (numpy-free for compatibility)
# ---------------------------------------------------------------------------

def _ema_series(closes: list[float], period: int) -> list[Optional[float]]:
    """Return EMA series, None until period bars are available."""
    if len(closes) < period:
        return [None] * len(closes)
    k = 2.0 / (period + 1)
    result: list[Optional[float]] = [None] * (period - 1)
    e = sum(closes[:period]) / period
    result.append(e)
    for c in closes[period:]:
        e = c * k + e * (1 - k)
        result.append(e)
    return result


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    # Wilder smoothing
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - 100 / (1 + rs)


def _macd(closes: list[float],
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (macd_line, signal_line, histogram) or (None, None, None)."""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)
    macd_vals: list[float] = []
    for f, s in zip(ema_fast, ema_slow):
        if f is not None and s is not None:
            macd_vals.append(f - s)
    if len(macd_vals) < signal:
        return None, None, None
    sig_series = _ema_series(macd_vals, signal)
    sig = sig_series[-1]
    m = macd_vals[-1]
    if sig is None:
        return m, None, None
    return m, sig, m - sig


def _bollinger(closes: list[float], period: int = 20,
               std_mult: float = 2.0) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (upper, mid, lower) or (None, None, None)."""
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((c - mid) ** 2 for c in window) / period
    std = math.sqrt(variance)
    return mid + std_mult * std, mid, mid - std_mult * std


def _atr(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def _adx(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> Optional[float]:
    """Wilder's ADX."""
    if len(closes) < period * 2:
        return None
    plus_dm_list, minus_dm_list, tr_list = [], [], []
    for i in range(1, len(closes)):
        up   = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm_list.append(up   if up > down and up > 0 else 0)
        minus_dm_list.append(down if down > up and down > 0 else 0)
        tr_list.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))

    def wilder_smooth(lst: list[float], p: int) -> list[float]:
        out = [sum(lst[:p])]
        for v in lst[p:]:
            out.append(out[-1] - out[-1] / p + v)
        return out

    tr_s   = wilder_smooth(tr_list, period)
    pdm_s  = wilder_smooth(plus_dm_list, period)
    mdm_s  = wilder_smooth(minus_dm_list, period)

    dx_list = []
    for tr, pdm, mdm in zip(tr_s, pdm_s, mdm_s):
        if tr == 0:
            continue
        pdi = 100 * pdm / tr
        mdi = 100 * mdm / tr
        denom = pdi + mdi
        dx_list.append(100 * abs(pdi - mdi) / denom if denom else 0)

    if len(dx_list) < period:
        return None
    adx = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


def _rvol(volumes: list[float], lookback: int = 20) -> Optional[float]:
    if len(volumes) < lookback + 1:
        return None
    avg = sum(volumes[-lookback - 1:-1]) / lookback
    if avg <= 0:
        return None
    return volumes[-1] / avg


def _vwap(opens, highs, lows, closes, volumes) -> Optional[float]:
    pv = 0.0
    vol = 0.0
    for o, h, l, c, v in zip(opens, highs, lows, closes, volumes):
        typical = (h + l + c) / 3
        pv += typical * v
        vol += v
    return pv / vol if vol > 0 else None


def _determine_trend(close: float, ema20: Optional[float],
                     ema50: Optional[float], ema200: Optional[float],
                     rsi: Optional[float]) -> str:
    scores = 0
    checks = 0
    if ema20 is not None:
        checks += 1
        if close > ema20:
            scores += 1
    if ema50 is not None:
        checks += 1
        if close > ema50:
            scores += 1
    if ema200 is not None:
        checks += 1
        if close > ema200:
            scores += 1
    if rsi is not None:
        checks += 1
        if rsi > 55:
            scores += 1
        elif rsi < 45:
            scores += 0
        else:
            scores += 0.5
    if checks == 0:
        return "NEUTRAL"
    ratio = scores / checks
    if ratio >= 0.65:
        return "BULLISH"
    if ratio <= 0.35:
        return "BEARISH"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def _analyse_bars(name: str, interval: str,
                  bars_df) -> Optional[TimeframeView]:
    """Compute all indicators for one timeframe from a pandas DataFrame."""
    if bars_df is None or len(bars_df) < 5:
        return None
    try:
        closes  = list(bars_df["Close"].astype(float))
        highs   = list(bars_df["High"].astype(float))
        lows    = list(bars_df["Low"].astype(float))
        opens   = list(bars_df["Open"].astype(float))
        volumes = list(bars_df["Volume"].astype(float))
    except Exception:
        return None

    view = TimeframeView(name=name, interval=interval, bars_count=len(closes))
    view.last_close = closes[-1]

    # EMAs
    e9  = _ema_series(closes, 9)
    e20 = _ema_series(closes, 20)
    e50 = _ema_series(closes, 50)
    e200= _ema_series(closes, 200)
    view.ema9   = e9[-1]
    view.ema20  = e20[-1]
    view.ema50  = e50[-1]
    view.ema200 = e200[-1]

    # RSI
    view.rsi = _rsi(closes)

    # MACD
    view.macd, view.macd_signal, view.macd_hist = _macd(closes)

    # Bollinger
    view.bb_upper, view.bb_mid, view.bb_lower = _bollinger(closes)

    # ATR
    view.atr = _atr(highs, lows, closes)

    # ADX
    view.adx = _adx(highs, lows, closes)

    # RVOL
    view.rvol = _rvol(volumes)

    # VWAP (for intraday timeframes use session bars; for daily just skip)
    if interval in ("5m", "15m", "1h"):
        # Use last session bars only
        import pandas as pd
        if hasattr(bars_df.index, 'date'):
            last_date = bars_df.index[-1].date()
            session = bars_df[bars_df.index.map(lambda x: x.date()) == last_date]
        else:
            session = bars_df
        if len(session) > 0:
            view.vwap = _vwap(
                list(session["Open"].astype(float)),
                list(session["High"].astype(float)),
                list(session["Low"].astype(float)),
                list(session["Close"].astype(float)),
                list(session["Volume"].astype(float)),
            )
            if view.vwap:
                view.above_vwap = closes[-1] > view.vwap

    # Trend
    view.trend = _determine_trend(
        closes[-1], view.ema20, view.ema50, view.ema200, view.rsi)

    return view


def analyse_symbol(yahoo_symbol: str,
                   fast_mode: bool = False) -> Optional[TechnicalProfile]:
    """Download bars and compute full technical profile for one symbol.

    Parameters
    ----------
    fast_mode : bool
        If True, only download daily bars (much faster for the initial universe scan).
    """
    import yfinance as yf

    profile = TechnicalProfile(symbol=yahoo_symbol, ltp=0.0)

    try:
        ticker = yf.Ticker(yahoo_symbol)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Daily — always fetch
            daily_df = ticker.history(period="200d", interval="1d")
            # Fallback to BSE (.BO) if .NS returned no data (e.g. symbol discrepancy)
            if (daily_df is None or len(daily_df) < 5) and yahoo_symbol.endswith(".NS"):
                bse_sym = yahoo_symbol.replace(".NS", ".BO")
                ticker = yf.Ticker(bse_sym)
                daily_df = ticker.history(period="200d", interval="1d")

            if daily_df is not None and len(daily_df) >= 5:
                # Normalize MultiIndex if needed
                if hasattr(daily_df.columns, 'levels'):
                    daily_df.columns = daily_df.columns.droplevel(1)
                profile.daily = _analyse_bars("Daily", "1d", daily_df)
                if profile.daily:
                    profile.ltp = profile.daily.last_close or 0.0

                    # Prior day levels
                    if len(daily_df) >= 2:
                        profile.levels.prev_day_high = float(daily_df["High"].iloc[-2])
                        profile.levels.prev_day_low  = float(daily_df["Low"].iloc[-2])

                    # Weekly high/low from last 5 daily bars
                    w = daily_df.iloc[-5:]
                    profile.levels.week_high = float(w["High"].max())
                    profile.levels.week_low  = float(w["Low"].min())

                    # Monthly
                    m = daily_df.iloc[-22:]
                    profile.levels.month_high = float(m["High"].max())
                    profile.levels.month_low  = float(m["Low"].min())

            if fast_mode:
                profile.compute_mtf_score()
                _detect_signals(profile)
                return profile

            # Weekly
            weekly_df = ticker.history(period="2y", interval="1wk")
            if weekly_df is not None and len(weekly_df) >= 5:
                if hasattr(weekly_df.columns, 'levels'):
                    weekly_df.columns = weekly_df.columns.droplevel(1)
                profile.weekly = _analyse_bars("Weekly", "1wk", weekly_df)

            # 1-hour
            hourly_df = ticker.history(period="60d", interval="1h")
            if hourly_df is not None and len(hourly_df) >= 10:
                if hasattr(hourly_df.columns, 'levels'):
                    hourly_df.columns = hourly_df.columns.droplevel(1)
                profile.hourly = _analyse_bars("1H", "1h", hourly_df)

            # 15-minute
            m15_df = ticker.history(period="5d", interval="15m")
            if m15_df is not None and len(m15_df) >= 10:
                if hasattr(m15_df.columns, 'levels'):
                    m15_df.columns = m15_df.columns.droplevel(1)
                profile.m15 = _analyse_bars("15M", "15m", m15_df)

            # 5-minute
            m5_df = ticker.history(period="5d", interval="5m")
            if m5_df is not None and len(m5_df) >= 10:
                if hasattr(m5_df.columns, 'levels'):
                    m5_df.columns = m5_df.columns.droplevel(1)
                profile.m5 = _analyse_bars("5M", "5m", m5_df)

    except Exception as e:
        return None

    profile.compute_mtf_score()
    _detect_signals(profile)
    return profile


def _detect_signals(profile: TechnicalProfile) -> None:
    """Detect notable technical patterns and append to profile.setup_signals."""
    signals = []
    d = profile.daily
    if d is None:
        return

    close = profile.ltp

    # --- Trend signals ---
    if d.ema9 and d.ema20 and d.ema9 > d.ema20:
        signals.append("EMA9>EMA20 (bullish)")
    if d.ema9 and d.ema20 and d.ema9 < d.ema20:
        signals.append("EMA9<EMA20 (bearish)")
    if d.ema200 and close > d.ema200:
        signals.append("Above EMA200")

    # --- Prior day range breakout potential ---
    lvl = profile.levels
    if lvl.prev_day_high and close > lvl.prev_day_high * 0.998:
        signals.append("Near Prev-Day High (LONG candidate)")
    if lvl.prev_day_low and close < lvl.prev_day_low * 1.002:
        signals.append("Near Prev-Day Low (SHORT candidate)")

    # --- RSI ---
    if d.rsi:
        if d.rsi > 60:
            signals.append(f"RSI strong ({d.rsi:.0f})")
        elif d.rsi < 40:
            signals.append(f"RSI weak ({d.rsi:.0f})")
        elif 50 < d.rsi <= 60:
            signals.append(f"RSI neutral-bullish ({d.rsi:.0f})")

    # --- MACD ---
    if d.macd_hist is not None and d.macd_hist > 0 and d.macd is not None and d.macd > 0:
        signals.append("MACD bullish")
    if d.macd_hist is not None and d.macd_hist < 0 and d.macd is not None and d.macd < 0:
        signals.append("MACD bearish")

    # --- Volume ---
    if d.rvol and d.rvol > 1.5:
        signals.append(f"High RVOL ({d.rvol:.1f}x)")

    # --- ADX ---
    if d.adx and d.adx > 25:
        signals.append(f"Strong trend ADX={d.adx:.0f}")

    # --- Intraday VWAP ---
    m = profile.m5 or profile.m15
    if m and m.above_vwap is not None:
        signals.append("Above VWAP" if m.above_vwap else "Below VWAP")

    profile.setup_signals = signals


def batch_analyse_symbols(yahoo_symbols: list[str], chunk_size: int = 80) -> dict[str, TechnicalProfile]:
    """Efficiently download and compute technical profiles for thousands of stocks in batch chunks.

    Gracefully skips symbols that raise NoneType/subscript errors (suspended,
    SME-migrated, or recently relisted tickers in the full NSE universe).
    """
    import yfinance as yf
    import pandas as pd
    profiles: dict[str, TechnicalProfile] = {}
    total = len(yahoo_symbols)

    def _safe_sym_df(raw, sym: str):
        """Safely extract per-symbol DataFrame — returns None on any error."""
        try:
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique()
                if sym not in lvl0:
                    return None
                df_s = raw[sym]
                if df_s is None or (hasattr(df_s, "empty") and df_s.empty):
                    return None
                return df_s.dropna(how="all")
            else:
                return raw.dropna(how="all")
        except Exception:
            return None

    for i in range(0, total, chunk_size):
        chunk = yahoo_symbols[i:i + chunk_size]
        print(f"    [technicals] batch processing {min(i + chunk_size, total)}/{total} symbols...", flush=True)
        raw = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    chunk, period="60d", interval="1d",
                    progress=False, auto_adjust=False,
                    group_by="ticker", threads=False,
                )
        except Exception:
            pass  # raw stays None; all syms in chunk will be skipped below

        for sym in chunk:
            try:
                df_s = _safe_sym_df(raw, sym)
                if df_s is None or len(df_s) < 5:
                    continue
                close_col = df_s["Close"].dropna()
                if len(close_col) == 0:
                    continue
                p = TechnicalProfile(symbol=sym, ltp=float(close_col.iloc[-1]))
                p.daily = _analyse_bars("Daily", "1d", df_s)
                if p.daily:
                    high_col = df_s["High"].dropna()
                    low_col  = df_s["Low"].dropna()
                    if len(high_col) >= 2:
                        p.levels.prev_day_high = float(high_col.iloc[-2])
                        p.levels.prev_day_low  = float(low_col.iloc[-2])
                    w = df_s.iloc[-5:]
                    p.levels.week_high  = float(w["High"].max())
                    p.levels.week_low   = float(w["Low"].min())
                    m = df_s.iloc[-22:]
                    p.levels.month_high = float(m["High"].max())
                    p.levels.month_low  = float(m["Low"].min())
                    p.compute_mtf_score()
                    _detect_signals(p)
                    profiles[sym] = p
            except Exception:
                continue  # silently skip bad symbols

    return profiles


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = analyse_symbol("RELIANCE.NS")
    if p:
        print(p.render_summary())
