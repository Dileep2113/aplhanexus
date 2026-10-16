"""Market context snapshot rendered for the ``run.py context`` command.

``build_context`` stitches together:

* NSE index/breadth/FII-DII data (when ``nse`` is provided).
* India VIX from the Yahoo provider.

It returns a :class:`ContextReport` whose ``.render()`` produces a
human-readable console string.  The caller decides how to display it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .providers import NseContext, NseContextProvider, ProviderError, YahooProvider


# ---------------------------------------------------------------------------
# NseContext is an alias of NseContextProvider used by run.py
# ---------------------------------------------------------------------------
# providers.py exports NseContextProvider under the class name
# NseContextProvider; run.py imports it as NseContext.  We make that alias
# available from this module too, so both spellings work.
# (The actual class is in providers.py; we just re-import under the short name.)


@dataclass
class ContextReport:
    """Bundled market-context data with a plain-text renderer."""

    lines: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join(self.lines)


def build_context(
    nse: Optional["NseContext"] = None,
    vix_provider: Optional[YahooProvider] = None,
) -> ContextReport:
    """Fetch and format the market context.

    Parameters
    ----------
    nse:
        An :class:`NseContext` / ``NseContextProvider`` instance.  When
        ``None`` the NSE section is skipped.
    vix_provider:
        A :class:`YahooProvider` used to fetch the India VIX bar.  When
        ``None`` VIX is skipped.
    """
    report = ContextReport()
    add = report.lines.append

    # ---- NSE indices, breadth, FII/DII -----------------------------------
    if nse is not None:
        try:
            ctx = nse.fetch()
        except ProviderError as exc:
            add(f"! NSE context unavailable: {exc}")
            ctx = None

        if ctx is not None:
            status = ctx.exchange_status or "?"
            tdate = ctx.trade_date or "?"
            add(f"Exchange: {status}  trade-date: {tdate}")

            # Indices we care about, in display order
            _WANT = [
                "NIFTY 50", "NIFTY BANK", "NIFTY MIDCAP 100",
                "NIFTY SMLCAP 100", "NIFTY IT", "NIFTY AUTO",
                "NIFTY PHARMA", "NIFTY FMCG", "NIFTY METAL", "INDIA VIX",
            ]
            for name in _WANT:
                ix = ctx.index(name)
                if ix is None:
                    continue
                br = ""
                if ix.breadth_ratio is not None:
                    br = (f"  A/D {ix.advances}/{ix.declines} "
                          f"({ix.breadth_ratio:.2f})")
                add(f"  {name:<22} {ix.last:>12,.2f}  {ix.pct_change:+.2f}%{br}")

            if ctx.flows:
                f = ctx.flows
                add(f"  FII net {f.fii_net_cr:+,.0f} cr  "
                    f"DII net {f.dii_net_cr:+,.0f} cr  ({f.date}, provisional)")

            add(f"  data: {ctx.meta.label()}")
            for w in ctx.meta.warnings:
                add(f"  ! {w}")
    else:
        add("(NSE context disabled -- set data.context: nse in config.yaml to enable)")

    # ---- VIX from Yahoo --------------------------------------------------
    if vix_provider is not None:
        add("")
        try:
            bars, meta = vix_provider.intraday_bars("VIX", interval="1d", lookback_days=5)
            if bars:
                last = bars[-1]
                add(f"INDIA VIX (Yahoo daily): {last.close:.2f}  "
                    f"[{last.ts.strftime('%Y-%m-%d')}]")
            else:
                add("INDIA VIX: no daily bar available from Yahoo")
            add(f"  data: {meta.label()}")
        except ProviderError as exc:
            add(f"! VIX unavailable: {exc}")

    return report
