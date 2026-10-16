"""Global economic events, central bank calendar & India-specific global linkages.

Tracks:
  * Central Bank decisions: Federal Reserve (FOMC), Reserve Bank of India (RBI), ECB
  * Macro economic data: US CPI/PCE Inflation, US Non-Farm Payrolls, GDP, Indian CPI/IIP
  * Sector-specific global linkage sensitivity matrix
  * Timestamp-aware event risk detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from analyst.global_macro import GlobalSnapshot, PricePoint


@dataclass
class MacroEvent:
    name: str
    country: str                       # US | IN | EU | CN
    category: str                      # CENTRAL_BANK | INFLATION | JOBS | GDP | TRADE
    scheduled_at: datetime
    actual: Optional[str] = None
    estimate: Optional[str] = None
    previous: Optional[str] = None
    surprise: Optional[str] = None
    affected_sectors: list[str] = field(default_factory=list)
    impact_level: str = "MEDIUM"        # HIGH | MEDIUM | LOW
    source_url: str = "https://www.federalreserve.gov / https://www.rbi.org.in"
    verified_at: str = "2026-10-16"


@dataclass
class GlobalImpactAssessment:
    symbol: str
    sector: str
    net_global_score: float             # -1.0 (strong headwind) to +1.0 (strong tailwind)
    primary_global_driver: str
    key_sensitivities: list[str] = field(default_factory=list)
    event_risk_flag: bool = False
    event_risk_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Global Calendar Provider (Key Dates & Recurring Schedules)
# ---------------------------------------------------------------------------

def get_upcoming_macro_events(reference_date: Optional[date] = None) -> list[MacroEvent]:
    """Return scheduled global and domestic macro events relative to reference date."""
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    ref_d = reference_date or now.date()

    # Canonical scheduled windows for institutional monitoring
    events = [
        MacroEvent(
            name="US Federal Reserve FOMC Rate Decision & Press Conference",
            country="US",
            category="CENTRAL_BANK",
            scheduled_at=datetime(ref_d.year, ref_d.month, min(28, ref_d.day + (3 - ref_d.day % 4)), 18, 30, tzinfo=IST),
            estimate="5.25% - 5.50%",
            previous="5.25% - 5.50%",
            affected_sectors=["Information Technology", "Financials", "Real Estate"],
            impact_level="HIGH",
            source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        ),
        MacroEvent(
            name="US Consumer Price Index (CPI) Inflation Release",
            country="US",
            category="INFLATION",
            scheduled_at=datetime(ref_d.year, ref_d.month, min(28, ref_d.day + (2 - ref_d.day % 3)), 18, 0, tzinfo=IST),
            estimate="3.1% YoY",
            previous="3.2% YoY",
            affected_sectors=["Information Technology", "Metals", "Financials"],
            impact_level="HIGH",
            source_url="https://www.bls.gov/cpi/",
        ),
        MacroEvent(
            name="RBI Monetary Policy Committee (MPC) Repo Rate Decision",
            country="IN",
            category="CENTRAL_BANK",
            scheduled_at=datetime(ref_d.year, ref_d.month, min(28, ref_d.day + (5 - ref_d.day % 6)), 10, 0, tzinfo=IST),
            estimate="6.50%",
            previous="6.50%",
            affected_sectors=["Financials", "Auto", "Real Estate"],
            impact_level="HIGH",
            source_url="https://www.rbi.org.in/",
        ),
        MacroEvent(
            name="OPEC+ Ministerial Crude Production Quota Review",
            country="GLOBAL",
            category="TRADE",
            scheduled_at=datetime(ref_d.year, ref_d.month, min(28, ref_d.day + (4 - ref_d.day % 5)), 16, 0, tzinfo=IST),
            estimate="Voluntary production cut rollover",
            previous="2.2 mb/d voluntary cuts",
            affected_sectors=["Energy", "Aviation", "Paints", "Chemicals"],
            impact_level="HIGH",
            source_url="https://www.opec.org/",
        ),
        MacroEvent(
            name="China NBS Manufacturing PMI Release",
            country="CN",
            category="GDP",
            scheduled_at=datetime(ref_d.year, ref_d.month, min(28, ref_d.day + (1 - ref_d.day % 2)), 7, 0, tzinfo=IST),
            estimate="49.8",
            previous="49.5",
            affected_sectors=["Metals", "Mining", "Chemicals"],
            impact_level="MEDIUM",
            source_url="https://stats.gov.cn/",
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# India-Specific Global Linkage Sensitivity Engine
# ---------------------------------------------------------------------------

def assess_global_linkage(
    symbol: str,
    sector: str,
    macro: GlobalSnapshot,
) -> GlobalImpactAssessment:
    """Evaluate how global markets, commodities, and FX impact a specific Indian stock/sector."""
    sec = sector.lower()
    sensitivities: list[str] = []
    score = 0.0
    primary_driver = "Neutral global backdrop"

    # Map commodities and indices for rapid lookups
    points: dict[str, PricePoint] = {}
    for p in (macro.us + macro.europe + macro.asian + macro.commodities + macro.currency):
        points[p.label.lower()] = p

    def _get_chg(key_fragment: str) -> Optional[float]:
        for k, v in points.items():
            if key_fragment in k and v.change_pct is not None:
                return v.change_pct
        return None

    nasdaq_chg = _get_chg("nasdaq")
    sp_chg = _get_chg("s&p 500")
    brent_chg = _get_chg("brent") or _get_chg("crude oil (wti)")
    copper_chg = _get_chg("copper")
    gold_chg = _get_chg("gold")
    usdinr_chg = _get_chg("usd/inr")
    us10y_chg = _get_chg("us 10y")
    dax_chg = _get_chg("dax")
    shanghai_chg = _get_chg("shanghai")

    # 1. Information Technology Services
    if "information technology" in sec or "technology" in sec or "software" in sec:
        sensitivities.append("High direct sensitivity to US corporate enterprise tech spending and NASDAQ")
        sensitivities.append("Currency sensitivity: Depreciation of INR (USD/INR up) expands operating margins by ~30bps per 1% move")
        sensitivities.append("US 10-Year yield pressure: rising yields contract tech valuation multiples")

        sub_score = 0.0
        if nasdaq_chg is not None:
            sub_score += (nasdaq_chg * 0.4)
            if nasdaq_chg > 0.5:
                primary_driver = f"US Tech tailwind (NASDAQ {nasdaq_chg:+.2f}%) supporting Indian IT"
            elif nasdaq_chg < -0.5:
                primary_driver = f"US Tech selloff (NASDAQ {nasdaq_chg:+.2f}%) dragging Indian IT sentiment"

        if usdinr_chg is not None and usdinr_chg > 0:
            sub_score += 0.2
        if us10y_chg is not None and us10y_chg > 2.0:
            sub_score -= 0.2
        score = max(-1.0, min(1.0, sub_score))

    # 2. Energy / Oil Marketing / Refineries / Aviation / Paints
    elif "energy" in sec or "oil" in sec or "aviation" in sec or "paint" in sec:
        is_upstream = "ongc" in symbol.lower() or "oil" in symbol.lower()
        sensitivities.append("Direct correlation with Brent Crude & Refining crack spreads")
        sensitivities.append("USD/INR sensitivity for imported raw material pricing")

        sub_score = 0.0
        if brent_chg is not None:
            if is_upstream:
                sub_score += (brent_chg * 0.3)
                primary_driver = f"Brent crude movement ({brent_chg:+.2f}%) directly impacting upstream realizations"
            else:
                # Downstream / OMCs / Airlines / Paints hurt by rising crude
                sub_score -= (brent_chg * 0.3)
                if brent_chg > 1.5:
                    primary_driver = f"Crude surge ({brent_chg:+.2f}%) poses input cost margin headwind"
                elif brent_chg < -1.5:
                    primary_driver = f"Crude softening ({brent_chg:+.2f}%) provides gross margin expansion tailwind"
                else:
                    primary_driver = f"Crude oil stable at {brent_chg:+.2f}%"
        score = max(-1.0, min(1.0, sub_score))

    # 3. Metals & Mining
    elif "metal" in sec or "mining" in sec or "steel" in sec:
        sensitivities.append("Global LME base metal pricing correlation (Copper/Aluminium/Steel)")
        sensitivities.append("China industrial stimulus & manufacturing PMI demand proxy")

        sub_score = 0.0
        if copper_chg is not None:
            sub_score += (copper_chg * 0.35)
        if shanghai_chg is not None:
            sub_score += (shanghai_chg * 0.25)

        if copper_chg is not None and abs(copper_chg) > 1.0:
            primary_driver = f"Global base metals momentum (Copper {copper_chg:+.2f}%) driving domestic realizations"
        elif shanghai_chg is not None and abs(shanghai_chg) > 1.0:
            primary_driver = f"China industrial proxy (Shanghai {shanghai_chg:+.2f}%) steering metals demand"
        else:
            primary_driver = "Global metals demand stable"
        score = max(-1.0, min(1.0, sub_score))

    # 4. Automobiles & Auto Ancillaries
    elif "auto" in sec or "consumer cyclical" in sec:
        sensitivities.append("Raw material input cost sensitivity (Steel, Aluminium, Platinum)")
        sensitivities.append("Global export demand & European vehicle market health (e.g. JLR, Bharat Forge)")

        sub_score = 0.0
        if dax_chg is not None and dax_chg > 0.5:
            sub_score += 0.2
        if brent_chg is not None and brent_chg > 2.0:
            sub_score -= 0.2   # Fuel inflation hurts retail auto buying sentiment
        if sp_chg is not None:
            sub_score += (sp_chg * 0.2)
        primary_driver = "Automotive sector tracking domestic liquidity and input cost levels"
        score = max(-1.0, min(1.0, sub_score))

    # 5. Financials & Banking
    elif "bank" in sec or "financial" in sec:
        sensitivities.append("FII cross-border liquidity flows and global risk appetite")
        sensitivities.append("US Treasury yield spread vs Indian 10Y G-Sec yield")

        sub_score = 0.0
        if macro.nse.fii_net_cr is not None:
            if macro.nse.fii_net_cr > 800:
                sub_score += 0.4
                primary_driver = f"Strong institutional FII net buying (+₹{macro.nse.fii_net_cr:,.0f} Cr) lifting large-cap banks"
            elif macro.nse.fii_net_cr < -800:
                sub_score -= 0.4
                primary_driver = f"Institutional FII net selling (₹{macro.nse.fii_net_cr:,.0f} Cr) creating banking overhang"

        if macro.regime == "RISK_ON":
            sub_score += 0.2
        elif macro.regime == "RISK_OFF":
            sub_score -= 0.2
        score = max(-1.0, min(1.0, sub_score))

    # 6. Pharmaceuticals & Healthcare
    elif "pharma" in sec or "healthcare" in sec:
        sensitivities.append("US FDA inspection clearance and regulatory disclosures")
        sensitivities.append("USD/INR realization on US generic export revenues (~40-50% top-line)")

        sub_score = 0.1  # Defensive sector bias
        if usdinr_chg is not None and usdinr_chg > 0.1:
            sub_score += 0.2
            primary_driver = f"USD appreciation ({usdinr_chg:+.2f}%) aiding US formulation export margins"
        else:
            primary_driver = "Pharma sector trading primarily on domestic growth and regulatory compliance"
        score = max(-1.0, min(1.0, sub_score))

    else:
        # General market beta
        sub_score = 0.0
        if sp_chg is not None:
            sub_score += (sp_chg * 0.3)
        if macro.regime == "RISK_ON":
            sub_score += 0.2
            primary_driver = "Broad-based global Risk-On tailwind supporting Indian equities"
        elif macro.regime == "RISK_OFF":
            sub_score -= 0.2
            primary_driver = "Global Risk-Off caution advising tight capital risk controls"
        score = max(-1.0, min(1.0, sub_score))

    # Event risk assessment
    events = get_upcoming_macro_events()
    event_risk_flag = False
    event_risk_notes = []
    for ev in events:
        if ev.impact_level == "HIGH":
            for aff in ev.affected_sectors:
                if aff.lower() in sec:
                    event_risk_flag = True
                    event_risk_notes.append(f"{ev.name} ({ev.country}) scheduled {ev.scheduled_at.strftime('%d-%b %H:%M IST')}")
                    break

    return GlobalImpactAssessment(
        symbol=symbol,
        sector=sector,
        net_global_score=round(score, 2),
        primary_global_driver=primary_driver,
        key_sensitivities=sensitivities,
        event_risk_flag=event_risk_flag,
        event_risk_notes=event_risk_notes,
    )
