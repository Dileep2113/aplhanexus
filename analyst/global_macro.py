"""Global macro + Indian market context snapshot.

Fetches:
  * US equity indices: S&P 500, Dow Jones, Nasdaq 100
  * US VIX (fear gauge)
  * US 10-year Treasury yield
  * Dollar Index (DXY)
  * Commodities: Crude Oil (WTI), Gold, Silver, Copper
  * Currency: USD/INR, EUR/INR
  * Asian indices: Nikkei 225, Hang Seng, Shanghai Composite
  * India: Nifty 50, Bank Nifty, India VIX, FII/DII net flows, A/D ratio

All data is clearly labelled DELAYED / HISTORICAL.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PricePoint:
    symbol: str
    label: str
    value: float
    change_pct: Optional[float]
    date: Optional[str] = None
    freshness: str = "DELAYED"

    def as_line(self, width: int = 22) -> str:
        chg = f"{self.change_pct:+.2f}%" if self.change_pct is not None else "   N/A"
        arrow = "↑" if (self.change_pct or 0) > 0 else ("↓" if (self.change_pct or 0) < 0 else "→")
        return f"  {self.label:<{width}} {self.value:>12,.2f}  {chg}  {arrow}"


@dataclass
class NseContext:
    exchange_status: Optional[str] = None
    trade_date: Optional[str] = None
    nifty50: Optional[PricePoint] = None
    bank_nifty: Optional[PricePoint] = None
    india_vix: Optional[PricePoint] = None
    advance_decline: Optional[str] = None    # "26/24 (1.08)"
    fii_net_cr: Optional[float] = None
    dii_net_cr: Optional[float] = None
    fii_dii_date: Optional[str] = None
    sector_snapshots: dict[str, PricePoint] = field(default_factory=dict)


@dataclass
class GlobalSnapshot:
    fetched_at: datetime
    us: list[PricePoint] = field(default_factory=list)
    us_futures: list[PricePoint] = field(default_factory=list)
    europe: list[PricePoint] = field(default_factory=list)
    commodities: list[PricePoint] = field(default_factory=list)
    currency: list[PricePoint] = field(default_factory=list)
    asian: list[PricePoint] = field(default_factory=list)
    nse: NseContext = field(default_factory=NseContext)
    regime: str = "NEUTRAL"           # RISK_ON / RISK_OFF / MIXED / NEUTRAL
    regime_reasons: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    def compute_regime(self) -> None:
        """Determine market regime from available data points."""
        bullish = 0
        bearish = 0
        reasons = []

        def _check(pts: list[PricePoint], label: str):
            nonlocal bullish, bearish
            for p in pts:
                if p.change_pct is None:
                    continue
                if p.change_pct > 0.3:
                    bullish += 1
                    reasons.append(f"{label} {p.label} +{p.change_pct:.1f}%")
                elif p.change_pct < -0.3:
                    bearish += 1
                    reasons.append(f"{label} {p.label} {p.change_pct:.1f}%")

        _check(self.us, "US")
        _check(self.europe, "Europe")
        _check(self.asian, "Asia")

        # VIX special: rising VIX = bearish
        vix = self.nse.india_vix
        if vix and vix.change_pct is not None:
            if vix.change_pct > 5:
                bearish += 2
                reasons.append(f"India VIX spiking +{vix.change_pct:.1f}%")
            elif vix.change_pct < -3:
                bullish += 1
                reasons.append(f"India VIX falling {vix.change_pct:.1f}%")

        # FII flows
        if self.nse.fii_net_cr is not None:
            if self.nse.fii_net_cr > 500:
                bullish += 1
                reasons.append(f"FII net buyers +{self.nse.fii_net_cr:,.0f} cr")
            elif self.nse.fii_net_cr < -500:
                bearish += 1
                reasons.append(f"FII net sellers {self.nse.fii_net_cr:,.0f} cr")

        # Crude: rising crude generally = slight headwind for India
        for p in self.commodities:
            if "Crude" in p.label and p.change_pct is not None:
                if p.change_pct > 2:
                    bearish += 1
                    reasons.append(f"Crude {p.change_pct:+.1f}% (inflation risk)")
                elif p.change_pct < -2:
                    bullish += 1
                    reasons.append(f"Crude {p.change_pct:+.1f}% (cost relief)")

        self.regime_reasons = reasons[:6]
        if bullish > bearish + 1:
            self.regime = "RISK_ON"
        elif bearish > bullish + 1:
            self.regime = "RISK_OFF"
        elif bullish > 0 or bearish > 0:
            self.regime = "MIXED"
        else:
            self.regime = "NEUTRAL"

    # -----------------------------------------------------------------------
    def render(self) -> str:
        lines: list[str] = []
        add = lines.append

        add("─" * 72)
        add(f" GLOBAL MARKET SNAPSHOT  |  {self.fetched_at.strftime('%Y-%m-%d %H:%M IST')}")
        add("─" * 72)

        if self.us:
            add(" US MARKETS")
            for p in self.us:
                add(p.as_line())

        if self.us_futures:
            add(" US EQUITY FUTURES")
            for p in self.us_futures:
                add(p.as_line())

        if self.europe:
            add(" EUROPEAN MARKETS")
            for p in self.europe:
                add(p.as_line())

        if self.asian:
            add(" ASIAN MARKETS")
            for p in self.asian:
                add(p.as_line())

        if self.commodities:
            add(" COMMODITIES")
            for p in self.commodities:
                add(p.as_line())

        if self.currency:
            add(" CURRENCY")
            for p in self.currency:
                add(p.as_line())

        n = self.nse
        add(" INDIA")
        if n.nifty50:
            add(n.nifty50.as_line())
        if n.bank_nifty:
            add(n.bank_nifty.as_line())
        if n.india_vix:
            add(n.india_vix.as_line())
        if n.advance_decline:
            add(f"  {'A/D Ratio':<22} {n.advance_decline}")
        if n.fii_net_cr is not None:
            add(f"  {'FII Net (cr)':<22} {n.fii_net_cr:>+12,.0f}")
        if n.dii_net_cr is not None:
            add(f"  {'DII Net (cr)':<22} {n.dii_net_cr:>+12,.0f}")

        if n.sector_snapshots:
            add(" SECTOR INDICES")
            for p in n.sector_snapshots.values():
                add(p.as_line())

        add("─" * 72)
        regime_color = {"RISK_ON": "✅", "RISK_OFF": "🔴", "MIXED": "⚠️", "NEUTRAL": "⬜"}
        add(f" REGIME: {regime_color.get(self.regime, '')} {self.regime}")
        for r in self.regime_reasons:
            add(f"   • {r}")
        add("─" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _yf_last(ticker_sym: str, label: str) -> Optional[PricePoint]:
    """Fetch last close + 1-day change from Yahoo Finance."""
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker_sym)
            h = t.history(period="5d", interval="1d")
        if h is None or len(h) < 2:
            return None
        prev, last = float(h["Close"].iloc[-2]), float(h["Close"].iloc[-1])
        chg = (last - prev) / prev * 100
        date_str = str(h.index[-1].date())
        return PricePoint(ticker_sym, label, last, chg, date_str)
    except Exception:
        return None


def _batch_yf(sym_label_pairs: list[tuple[str, str]]) -> list[PricePoint]:
    """Batch-fetch a list of (yahoo_symbol, label) pairs."""
    results: list[PricePoint] = []
    import yfinance as yf
    syms = [s for s, _ in sym_label_pairs]
    raw = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(syms, period="5d", interval="1d",
                              progress=False, auto_adjust=False)
    except Exception as e:
        print(f"  [macro] batch fetch note ({e}), using per-ticker fallback", flush=True)
        raw = None

    if raw is None or raw.empty:
        # fallback to per-ticker
        for sym, lbl in sym_label_pairs:
            p = _yf_last(sym, lbl)
            if p:
                results.append(p)
        return results

    import pandas as pd
    for sym, lbl in sym_label_pairs:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"][sym].dropna()
            else:
                closes = raw["Close"].dropna()
            if len(closes) < 2:
                continue
            prev, last = float(closes.iloc[-2]), float(closes.iloc[-1])
            chg = (last - prev) / prev * 100
            date_str = str(closes.index[-1].date())
            results.append(PricePoint(sym, lbl, last, chg, date_str))
        except Exception:
            p = _yf_last(sym, lbl)
            if p:
                results.append(p)
    return results


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch_global_snapshot(nse_provider=None) -> GlobalSnapshot:
    """Fetch the complete global + Indian market snapshot.

    Parameters
    ----------
    nse_provider : NseContextProvider (optional)
        If supplied, uses it for authoritative NSE index/FII/DII data.
        Otherwise, all Indian data comes from Yahoo Finance.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    snap = GlobalSnapshot(fetched_at=now)

    # ---- US markets ------------------------------------------------------
    print("  [macro] fetching US markets...", flush=True)
    us_pairs = [
        ("^GSPC",    "S&P 500"),
        ("^DJI",     "Dow Jones"),
        ("^IXIC",    "NASDAQ"),
        ("^RUT",     "Russell 2000"),
        ("^VIX",     "US VIX"),
        ("^TNX",     "US 10Y Yield"),
        ("DX-Y.NYB", "Dollar Index (DXY)"),
    ]
    snap.us = _batch_yf(us_pairs)

    # ---- US Futures ------------------------------------------------------
    us_futures_pairs = [
        ("ES=F", "S&P 500 Fut"),
        ("NQ=F", "Nasdaq Fut"),
    ]
    snap.us_futures = _batch_yf(us_futures_pairs)

    # ---- European markets ------------------------------------------------
    print("  [macro] fetching European markets...", flush=True)
    europe_pairs = [
        ("^FTSE",    "FTSE 100"),
        ("^GDAXI",   "DAX 40"),
        ("^FCHI",    "CAC 40"),
        ("^STOXX50E","STOXX 50"),
    ]
    snap.europe = _batch_yf(europe_pairs)

    # ---- Asian markets ---------------------------------------------------
    print("  [macro] fetching Asian markets...", flush=True)
    asian_pairs = [
        ("^N225",     "Nikkei 225"),
        ("^HSI",      "Hang Seng"),
        ("000001.SS", "Shanghai Comp"),
        ("^KS11",     "KOSPI"),
        ("^STI",      "Straits Times"),
    ]
    snap.asian = _batch_yf(asian_pairs)

    # ---- Commodities -----------------------------------------------------
    print("  [macro] fetching commodities...", flush=True)
    comm_pairs = [
        ("CL=F",  "Crude Oil (WTI)"),
        ("BZ=F",  "Brent Crude"),
        ("GC=F",  "Gold"),
        ("SI=F",  "Silver"),
        ("HG=F",  "Copper"),
        ("NG=F",  "Natural Gas"),
    ]
    snap.commodities = _batch_yf(comm_pairs)

    # ---- Currency --------------------------------------------------------
    print("  [macro] fetching currency...", flush=True)
    fx_pairs = [
        ("USDINR=X", "USD/INR"),
        ("EURINR=X", "EUR/INR"),
    ]
    snap.currency = _batch_yf(fx_pairs)

    # ---- Indian market context ------------------------------------------
    print("  [macro] fetching Indian context...", flush=True)
    nse_ctx = snap.nse

    if nse_provider is not None:
        # Use authoritative NSE provider (already has breadth, FII/DII)
        try:
            ctx = nse_provider.fetch()
            nse_ctx.exchange_status = ctx.exchange_status
            nse_ctx.trade_date = ctx.trade_date

            def _snap(name: str, label: str) -> Optional[PricePoint]:
                ix = ctx.index(name)
                if ix is None:
                    return None
                return PricePoint(name, label, ix.last, ix.pct_change,
                                  freshness="DELAYED")

            nse_ctx.nifty50    = _snap("NIFTY 50",     "Nifty 50")
            nse_ctx.bank_nifty = _snap("NIFTY BANK",   "Bank Nifty")
            nse_ctx.india_vix  = _snap("INDIA VIX",    "India VIX")

            # A/D ratio from Nifty 500 or best available
            for name in ("NIFTY 500", "NIFTY 50", "NIFTY 100"):
                ix = ctx.index(name)
                if ix and ix.advances and ix.declines:
                    ratio = ix.advances / ix.declines if ix.declines else 0
                    nse_ctx.advance_decline = (
                        f"{ix.advances}/{ix.declines} ({ratio:.2f})")
                    break

            nse_ctx.fii_net_cr  = ctx.flows.fii_net_cr  if ctx.flows else None
            nse_ctx.dii_net_cr  = ctx.flows.dii_net_cr  if ctx.flows else None
            nse_ctx.fii_dii_date = ctx.flows.date        if ctx.flows else None

            # Sector indices
            for sname, slabel in [
                ("NIFTY IT",      "Nifty IT"),
                ("NIFTY AUTO",    "Nifty Auto"),
                ("NIFTY BANK",    "Nifty Bank"),
                ("NIFTY PHARMA",  "Nifty Pharma"),
                ("NIFTY FMCG",    "Nifty FMCG"),
                ("NIFTY METAL",   "Nifty Metal"),
                ("NIFTY MIDCAP 100", "Nifty MidCap"),
            ]:
                p = _snap(sname, slabel)
                if p:
                    nse_ctx.sector_snapshots[slabel] = p
        except Exception as e:
            print(f"  [macro] NSE provider error: {e}")

    # Fall back to Yahoo for any missing Indian data
    if nse_ctx.nifty50 is None:
        p = _yf_last("^NSEI", "Nifty 50")
        if p:
            nse_ctx.nifty50 = p
    if nse_ctx.bank_nifty is None:
        p = _yf_last("^NSEBANK", "Bank Nifty")
        if p:
            nse_ctx.bank_nifty = p
    if nse_ctx.india_vix is None:
        p = _yf_last("^INDIAVIX", "India VIX")
        if p:
            nse_ctx.india_vix = p

    # ---- Compute regime --------------------------------------------------
    snap.compute_regime()
    return snap


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    snap = fetch_global_snapshot()
    print(snap.render())
