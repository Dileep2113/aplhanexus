"""Indian Pre-Market Intelligence & Scenario Generator.

Synthesizes overnight international markets, GIFT Nifty proxy, commodities,
currencies, and domestic institutional flows into an executive pre-market briefing
with Bullish, Bearish, and Neutral scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from analyst.global_events import get_upcoming_macro_events, MacroEvent
from analyst.global_macro import GlobalSnapshot, PricePoint


@dataclass
class PreMarketScenario:
    name: str                          # BULLISH | BEARISH | NEUTRAL
    probability_bias: str              # HIGH | MODERATE | LOW
    opening_range_nifty: str           # Expected opening level
    primary_catalysts: list[str] = field(default_factory=list)
    actionable_intraday_plan: str = ""


@dataclass
class PreMarketReport:
    generated_at: datetime
    gift_nifty_implied: Optional[float]
    gift_nifty_change_pct: Optional[float]
    nifty_prev_close: Optional[float]
    gap_points: Optional[float]
    gap_type: str                      # GAP_UP | GAP_DOWN | FLAT
    regime: str
    overnight_us_summary: str
    overnight_asia_summary: str
    overnight_europe_summary: str
    commodities_fx_summary: str
    fii_dii_summary: str
    scenarios: list[PreMarketScenario] = field(default_factory=list)
    upcoming_events: list[MacroEvent] = field(default_factory=list)

    def render_console(self) -> str:
        lines: list[str] = []
        add = lines.append

        add("═" * 78)
        add(f" 🇮🇳 INDIAN PRE-MARKET EXECUTIVE BRIEFING  |  {self.generated_at.strftime('%Y-%m-%d %H:%M IST')}")
        add("═" * 78)

        add(f" • Regime            : {self.regime}")
        gap_sym = "+" if (self.gap_points or 0) >= 0 else ""
        add(f" • GIFT Nifty Implied: {self.gift_nifty_implied or 0:,.2f} ({self.gift_nifty_change_pct or 0:+.2f}%)")
        add(f" • Indicated Gap     : {gap_sym}{self.gap_points or 0:,.1f} pts ({self.gap_type})")
        add(f" • Overnight US      : {self.overnight_us_summary}")
        add(f" • Overnight Asia    : {self.overnight_asia_summary}")
        add(f" • European Close    : {self.overnight_europe_summary}")
        add(f" • Commodities & FX  : {self.commodities_fx_summary}")
        add(f" • Institutional Flow: {self.fii_dii_summary}")

        add("─" * 78)
        add(" SCENARIO ANALYSIS (NON-GUARANTEED PROBABILISTIC ROADMAP):")
        for sc in self.scenarios:
            add(f"   [{sc.name} - {sc.probability_bias} BIAS] Zone: {sc.opening_range_nifty}")
            for cat in sc.primary_catalysts:
                add(f"     • {cat}")
            add(f"     Strategy: {sc.actionable_intraday_plan}")

        add("─" * 78)
        add(" ⚠️ GOVERNANCE DISCLAIMER: GIFT Nifty is an indicative pre-market barometer,")
        add("    NOT a guaranteed forecast of opening or settlement levels.")
        add("═" * 78)
        return "\n".join(lines)


def generate_premarket_report(macro: GlobalSnapshot) -> PreMarketReport:
    """Generate the structured pre-market scenario report from global snapshot."""
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)

    nifty = macro.nse.nifty50
    nifty_close = nifty.value if nifty else 23500.0

    # GIFT Nifty estimate (uses US Futures & S&P change if synthetic)
    us_avg_chg = 0.0
    us_valid = [p.change_pct for p in macro.us if p.change_pct is not None]
    if us_valid:
        us_avg_chg = sum(us_valid) / len(us_valid)

    # Calculate indicative GIFT Nifty
    gift_chg = us_avg_chg * 0.75  # Beta-adjusted correlation
    gift_implied = nifty_close * (1 + (gift_chg / 100.0))
    gap_pts = gift_implied - nifty_close

    if gap_pts > 45:
        gap_type = "GAP_UP"
    elif gap_pts < -45:
        gap_type = "GAP_DOWN"
    else:
        gap_type = "FLAT_OPEN"

    # Summaries
    us_lead = ", ".join(f"{p.label} {p.change_pct:+.2f}%" for p in macro.us[:3] if p.change_pct is not None) or "Stable"
    asia_lead = ", ".join(f"{p.label} {p.change_pct:+.2f}%" for p in macro.asian[:3] if p.change_pct is not None) or "Mixed"
    eu_lead = ", ".join(f"{p.label} {p.change_pct:+.2f}%" for p in macro.europe[:3] if p.change_pct is not None) or "Steady"
    comm_lead = ", ".join(f"{p.label} {p.change_pct:+.2f}%" for p in macro.commodities[:3] if p.change_pct is not None) or "Neutral"

    fii_str = f"FII: {macro.nse.fii_net_cr:+,.0f} Cr" if macro.nse.fii_net_cr is not None else "FII: Pending"
    dii_str = f"DII: {macro.nse.dii_net_cr:+,.0f} Cr" if macro.nse.dii_net_cr is not None else "DII: Pending"
    inst_str = f"{fii_str} | {dii_str} (Provisional previous session)"

    # Build scenarios
    scenarios: list[PreMarketScenario] = []

    # Scenario 1: Base Case
    if gap_type == "GAP_UP":
        scenarios.append(PreMarketScenario(
            name="GAP_UP_AND_HOLD (BULLISH)",
            probability_bias="MODERATE",
            opening_range_nifty=f"{gift_implied:,.0f} - {gift_implied + 50:,.0f}",
            primary_catalysts=[
                f"Global risk-on lead: {us_lead}",
                "Follow-through institutional buying above pre-market levels",
            ],
            actionable_intraday_plan="Allow opening 15m to settle (09:15-09:30). Look for opening range breakout (ORB) on strong sector leaders.",
        ))
        scenarios.append(PreMarketScenario(
            name="GAP_FADE / PULLBACK (MEAN REVERSION)",
            probability_bias="SECONDARY",
            opening_range_nifty=f"{nifty_close:,.0f} - {gift_implied:,.0f}",
            primary_catalysts=[
                "Overnight profit taking at overhead resistance",
                "FII supply on gap opens if valuations stretched",
            ],
            actionable_intraday_plan="Monitor VWAP breakdown if price slips below opening low with declining volume.",
        ))
    elif gap_type == "GAP_DOWN":
        scenarios.append(PreMarketScenario(
            name="GAP_DOWN_AND_EXTEND (BEARISH)",
            probability_bias="MODERATE",
            opening_range_nifty=f"{gift_implied - 50:,.0f} - {gift_implied:,.0f}",
            primary_catalysts=[
                f"Global selloff pressure: {us_lead}",
                "FII liquidations and elevated India VIX",
            ],
            actionable_intraday_plan="Exercise capital preservation. Focus strictly on short setups under prior-day low or defensive sectors.",
        ))
        scenarios.append(PreMarketScenario(
            name="SUPPORT BOUNCE (DIP BUYING)",
            probability_bias="SECONDARY",
            opening_range_nifty=f"{gift_implied:,.0f} - {nifty_close:,.0f}",
            primary_catalysts=[
                "DII institutional value buying near 20-day moving average support",
                "Oversold intraday conditions",
            ],
            actionable_intraday_plan="Wait for 5m candle close above intraday VWAP with volume confirmation before taking long reversal trades.",
        ))
    else:
        scenarios.append(PreMarketScenario(
            name="RANGE_BOUND_CONSOLIDATION (NEUTRAL)",
            probability_bias="HIGH",
            opening_range_nifty=f"{nifty_close - 40:,.0f} - {nifty_close + 40:,.0f}",
            primary_catalysts=[
                "Balanced global cues and sideways commodity prices",
                "Market awaiting key scheduled macro catalyst",
            ],
            actionable_intraday_plan="Avoid aggressive directional bets. Trade only clear relative-strength breakouts in leading sectors.",
        ))

    events = get_upcoming_macro_events()

    return PreMarketReport(
        generated_at=now,
        gift_nifty_implied=round(gift_implied, 2),
        gift_nifty_change_pct=round(gift_chg, 2),
        nifty_prev_close=round(nifty_close, 2),
        gap_points=round(gap_pts, 2),
        gap_type=gap_type,
        regime=macro.regime,
        overnight_us_summary=us_lead,
        overnight_asia_summary=asia_lead,
        overnight_europe_summary=eu_lead,
        commodities_fx_summary=comm_lead,
        fii_dii_summary=inst_str,
        scenarios=scenarios,
        upcoming_events=events,
    )
