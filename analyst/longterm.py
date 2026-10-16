"""Long-Term Institutional Investment Analysis & Scoring Engine.

Evaluates companies for 6-24 month investment horizon across 10 transparent criteria:
  1. Revenue Growth YoY (>= 12%)
  2. PAT / Earnings Growth YoY (>= 15%)
  3. Operating Profit Margin OPM (>= 15%)
  4. Return on Equity ROE (>= 15%)
  5. Debt-to-Equity Ratio (<= 1.0x / < 100 on YF)
  6. Structural Price Trend (Price > 200 EMA)
  7. 52-Week High Proximity (Within 25% of 52W High)
  8. Valuation Multiple & Fair Value (P/E < 45 or Attractive PEG)
  9. Trend Strength (ADX >= 20)
  10. Institutional & Macro Backdrop (Risk-On / Mixed regime & strong sponsorship)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from analyst.company import CompanyProfile
from analyst.global_macro import GlobalSnapshot
from analyst.technicals import TechnicalProfile


@dataclass
class CriterionCheck:
    name: str
    requirement: str
    actual: str
    status: str       # "PASS" | "WARN" | "FAIL"
    detail: str


@dataclass
class LongTermOpportunity:
    symbol: str
    name: str
    sector: str
    industry: str
    ltp: float
    market_cap_cr: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    roe: Optional[float]
    debt_to_equity: Optional[float]
    revenue_growth_yoy: Optional[float]
    earnings_growth_yoy: Optional[float]
    operating_margin: Optional[float]
    high_52w: Optional[float]
    low_52w: Optional[float]
    ema200: Optional[float]
    above_200_ema: bool
    adx: Optional[float]
    score: int                         # 0 to 100
    verdict: str                       # STRONG BUY | ACCUMULATE | HOLD | AVOID
    valuation_verdict: str             # Attractive | Fair | Premium | Overvalued
    operating_cashflow_cr: Optional[float] = None
    free_cashflow_cr: Optional[float] = None
    fcf_yield_pct: Optional[float] = None
    cash_conversion_status: str = "N/A"
    valuation_scenarios: dict[str, float] = field(default_factory=dict)
    criteria: list[CriterionCheck] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    thesis: str = ""


def evaluate_longterm_candidate(
    company: Optional[CompanyProfile],
    profile: TechnicalProfile,
    macro: GlobalSnapshot,
) -> Optional[LongTermOpportunity]:
    """Evaluate a stock against 10 rigorous institutional long-term criteria."""
    if not profile or profile.ltp <= 0:
        return None

    symbol = profile.symbol
    ltp = profile.ltp
    d = profile.daily

    name = company.name if company else symbol
    sector = company.sector if company else "General"
    industry = company.industry if company else "General"

    criteria: list[CriterionCheck] = []
    score = 0
    strengths: list[str] = list(company.strengths) if company else []
    risks: list[str] = list(company.risks) if company else []

    # -------------------------------------------------------------------------
    # 1. Revenue Growth YoY (Max 12 pts)
    # -------------------------------------------------------------------------
    rev_g = company.revenue_growth_yoy if company else None
    if rev_g is not None:
        pct = rev_g * 100
        if pct >= 15.0:
            status = "PASS"
            pts = 12
            det = f"Robust top-line expansion (+{pct:.1f}% YoY)"
        elif pct >= 8.0:
            status = "PASS"
            pts = 9
            det = f"Moderate revenue growth (+{pct:.1f}% YoY)"
        elif pct >= 0.0:
            status = "WARN"
            pts = 5
            det = f"Subdued positive growth (+{pct:.1f}% YoY)"
        else:
            status = "FAIL"
            pts = 0
            det = f"Revenue contraction ({pct:.1f}% YoY)"
        criteria.append(CriterionCheck(
            name="Revenue Growth YoY",
            requirement="≥ +12.0% YoY",
            actual=f"{pct:+.1f}%",
            status=status,
            detail=det,
        ))
    else:
        pts = 6
        criteria.append(CriterionCheck(
            name="Revenue Growth YoY",
            requirement="≥ +12.0% YoY",
            actual="Data Pending",
            status="WARN",
            detail="Quarterly revenue growth metric not reported",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 2. PAT / Earnings Growth YoY (Max 12 pts)
    # -------------------------------------------------------------------------
    earn_g = company.earnings_growth_yoy if company else None
    if earn_g is not None:
        pct = earn_g * 100
        if pct >= 20.0:
            status = "PASS"
            pts = 12
            det = f"Strong profit expansion (+{pct:.1f}% YoY)"
        elif pct >= 10.0:
            status = "PASS"
            pts = 9
            det = f"Healthy earnings growth (+{pct:.1f}% YoY)"
        elif pct >= 0.0:
            status = "WARN"
            pts = 5
            det = f"Flat/slow earnings (+{pct:.1f}% YoY)"
        else:
            status = "FAIL"
            pts = 0
            det = f"Earnings de-growth ({pct:.1f}% YoY)"
        criteria.append(CriterionCheck(
            name="Earnings (PAT) Growth YoY",
            requirement="≥ +15.0% YoY",
            actual=f"{pct:+.1f}%",
            status=status,
            detail=det,
        ))
    else:
        pts = 6
        criteria.append(CriterionCheck(
            name="Earnings (PAT) Growth YoY",
            requirement="≥ +15.0% YoY",
            actual="Data Pending",
            status="WARN",
            detail="Earnings growth rate not disclosed",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 3. Operating Profit Margin (OPM) (Max 10 pts)
    # -------------------------------------------------------------------------
    opm = company.operating_margin if company else None
    if opm is not None:
        pct = opm * 100
        if pct >= 20.0:
            status = "PASS"
            pts = 10
            det = f"Exceptional operating pricing power ({pct:.1f}%)"
        elif pct >= 14.0:
            status = "PASS"
            pts = 8
            det = f"Healthy operating margins ({pct:.1f}%)"
        elif pct >= 8.0:
            status = "WARN"
            pts = 5
            det = f"Moderate operating margin ({pct:.1f}%)"
        else:
            status = "FAIL"
            pts = 2
            det = f"Thin operating margin ({pct:.1f}%)"
        criteria.append(CriterionCheck(
            name="Operating Margin (OPM)",
            requirement="≥ 15.0%",
            actual=f"{pct:.1f}%",
            status=status,
            detail=det,
        ))
    else:
        pts = 5
        criteria.append(CriterionCheck(
            name="Operating Margin (OPM)",
            requirement="≥ 15.0%",
            actual="N/A",
            status="WARN",
            detail="Operating margin unavailable",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 4. Return on Equity (ROE) (Max 10 pts)
    # -------------------------------------------------------------------------
    roe = company.roe if company else None
    if roe is not None:
        pct = roe * 100
        if pct >= 18.0:
            status = "PASS"
            pts = 10
            det = f"High capital productivity (ROE {pct:.1f}%)"
        elif pct >= 12.0:
            status = "PASS"
            pts = 7
            det = f"Acceptable return on equity ({pct:.1f}%)"
        elif pct >= 5.0:
            status = "WARN"
            pts = 4
            det = f"Sub-par capital efficiency ({pct:.1f}%)"
        else:
            status = "FAIL"
            pts = 0
            det = f"Poor return on equity ({pct:.1f}%)"
        criteria.append(CriterionCheck(
            name="Return on Equity (ROE)",
            requirement="≥ 15.0%",
            actual=f"{pct:.1f}%",
            status=status,
            detail=det,
        ))
    else:
        pts = 5
        criteria.append(CriterionCheck(
            name="Return on Equity (ROE)",
            requirement="≥ 15.0%",
            actual="N/A",
            status="WARN",
            detail="ROE data not reported",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 5. Debt to Equity & Balance Sheet Solvency (Max 10 pts)
    # -------------------------------------------------------------------------
    de = company.debt_to_equity if company else None
    # Financials / Banks typically carry high leverage by design
    is_financial = "bank" in sector.lower() or "financial" in sector.lower() or "bank" in industry.lower()
    if is_financial:
        pts = 8
        criteria.append(CriterionCheck(
            name="Debt-to-Equity (Solvency)",
            requirement="Financial Sector Basel Capital Norms",
            actual="Bank/NBFC",
            status="PASS",
            detail="Leverage evaluated through Capital Adequacy / NIM rather than standard D/E",
        ))
    elif de is not None:
        # de on Yahoo is reported as percentage, e.g. 50 = 0.50x
        ratio = de / 100.0 if de > 5.0 else de
        if ratio <= 0.5:
            status = "PASS"
            pts = 10
            det = f"Prudent low leverage (D/E {ratio:.2f}x)"
        elif ratio <= 1.0:
            status = "PASS"
            pts = 7
            det = f"Manageable leverage (D/E {ratio:.2f}x)"
        elif ratio <= 1.5:
            status = "WARN"
            pts = 4
            det = f"Elevated debt level (D/E {ratio:.2f}x)"
        else:
            status = "FAIL"
            pts = 0
            det = f"High solvency risk (D/E {ratio:.2f}x)"
        criteria.append(CriterionCheck(
            name="Debt-to-Equity (Solvency)",
            requirement="≤ 1.0x (Ideal < 0.5x)",
            actual=f"{ratio:.2f}x",
            status=status,
            detail=det,
        ))
    else:
        pts = 5
        criteria.append(CriterionCheck(
            name="Debt-to-Equity (Solvency)",
            requirement="≤ 1.0x",
            actual="N/A",
            status="WARN",
            detail="Debt/equity ratio unlisted",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 6. Structural Trend: Price > 200 EMA (Max 12 pts)
    # -------------------------------------------------------------------------
    ema200 = d.ema200 if d else None
    above_200 = False
    if ema200 and ema200 > 0:
        above_200 = ltp >= ema200
        dist_pct = ((ltp - ema200) / ema200) * 100
        if above_200 and dist_pct >= 3.0:
            status = "PASS"
            pts = 12
            det = f"Trading +{dist_pct:.1f}% above 200 EMA (Long-term Bullish structure)"
        elif above_200:
            status = "PASS"
            pts = 9
            det = f"Testing 200 EMA support from above (+{dist_pct:.1f}%)"
        elif dist_pct >= -3.0:
            status = "WARN"
            pts = 5
            det = f"Consolidating right at 200 EMA ({dist_pct:.1f}%)"
        else:
            status = "FAIL"
            pts = 0
            det = f"Below 200 EMA ({dist_pct:.1f}%). Long-term downtrend"
        criteria.append(CriterionCheck(
            name="Structural Trend (200 EMA)",
            requirement="Price > 200 EMA",
            actual=f"₹{ema200:,.1f} ({'+' if dist_pct>=0 else ''}{dist_pct:.1f}%)",
            status=status,
            detail=det,
        ))
    else:
        # Fallback to daily trend
        trend_name = d.trend if d else "NEUTRAL"
        if trend_name == "BULLISH":
            status = "PASS"
            pts = 8
            above_200 = True
            det = "Daily trend BULLISH (200 EMA calculating)"
        elif trend_name == "BEARISH":
            status = "FAIL"
            pts = 2
            det = "Daily trend BEARISH"
        else:
            status = "WARN"
            pts = 5
            det = "Trend NEUTRAL/Consolidation"
        criteria.append(CriterionCheck(
            name="Structural Trend (200 EMA)",
            requirement="Price > 200 EMA",
            actual=trend_name,
            status=status,
            detail=det,
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 7. 52-Week High Proximity (Relative Strength) (Max 10 pts)
    # -------------------------------------------------------------------------
    high52 = company.high_52w if company else None
    low52 = company.low_52w if company else None
    if high52 and high52 > 0:
        discount_from_high = ((high52 - ltp) / high52) * 100
        if discount_from_high <= 15.0:
            status = "PASS"
            pts = 10
            det = f"Within {discount_from_high:.1f}% of 52W High (Strong leadership)"
        elif discount_from_high <= 25.0:
            status = "PASS"
            pts = 8
            det = f"Within {discount_from_high:.1f}% of 52W High (Healthy consolidation)"
        elif discount_from_high <= 40.0:
            status = "WARN"
            pts = 4
            det = f"{discount_from_high:.1f}% off 52W High (Accumulation zone)"
        else:
            status = "FAIL"
            pts = 1
            det = f"Lagging stock ({discount_from_high:.1f}% below 52W High)"
        criteria.append(CriterionCheck(
            name="52-Week High Proximity",
            requirement="Within 25% of 52W High",
            actual=f"-{discount_from_high:.1f}% off High",
            status=status,
            detail=det,
        ))
    else:
        pts = 5
        criteria.append(CriterionCheck(
            name="52-Week High Proximity",
            requirement="Within 25% of 52W High",
            actual="N/A",
            status="WARN",
            detail="52-week range data unavailable",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 8. Valuation Multiple & P/E (Max 10 pts)
    # -------------------------------------------------------------------------
    pe = company.pe_ratio if company else None
    val_verdict = company.valuation_verdict if company else "Fair Value"
    if pe is not None and pe > 0:
        if pe <= 25.0:
            status = "PASS"
            pts = 10
            det = f"Attractive institutional multiple (P/E {pe:.1f}x)"
            val_verdict = "Attractive"
        elif pe <= 45.0:
            status = "PASS"
            pts = 8
            det = f"Fair growth valuation (P/E {pe:.1f}x)"
            val_verdict = "Fair Value"
        elif pe <= 70.0:
            status = "WARN"
            pts = 5
            det = f"Growth premium multiple (P/E {pe:.1f}x)"
            val_verdict = "Premium"
        else:
            status = "FAIL"
            pts = 2
            det = f"Excessive multiple (P/E {pe:.1f}x)"
            val_verdict = "Overvalued"
        criteria.append(CriterionCheck(
            name="Valuation (P/E Multiple)",
            requirement="P/E ≤ 45.0x (or Attractive Growth)",
            actual=f"{pe:.1f}x",
            status=status,
            detail=det,
        ))
    else:
        pts = 5
        criteria.append(CriterionCheck(
            name="Valuation (P/E Multiple)",
            requirement="P/E ≤ 45.0x",
            actual="N/A",
            status="WARN",
            detail="Trailing P/E ratio not meaningful or loss-making",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 9. Trend Strength (ADX) (Max 8 pts)
    # -------------------------------------------------------------------------
    adx_val = d.adx if d else None
    if adx_val is not None:
        if adx_val >= 25.0:
            status = "PASS"
            pts = 8
            det = f"Strong persistent directional trend (ADX {adx_val:.1f})"
        elif adx_val >= 18.0:
            status = "PASS"
            pts = 6
            det = f"Emerging trend strength (ADX {adx_val:.1f})"
        elif adx_val >= 12.0:
            status = "WARN"
            pts = 4
            det = f"Range-bound / consolidating (ADX {adx_val:.1f})"
        else:
            status = "FAIL"
            pts = 1
            det = f"Weak/flat trend (ADX {adx_val:.1f})"
        criteria.append(CriterionCheck(
            name="Trend Strength (ADX)",
            requirement="ADX ≥ 20.0",
            actual=f"{adx_val:.1f}",
            status=status,
            detail=det,
        ))
    else:
        pts = 4
        criteria.append(CriterionCheck(
            name="Trend Strength (ADX)",
            requirement="ADX ≥ 20.0",
            actual="N/A",
            status="WARN",
            detail="ADX calculation in progress",
        ))
    score += pts

    # -------------------------------------------------------------------------
    # 10. Institutional & Macro Backdrop (Max 6 pts)
    # -------------------------------------------------------------------------
    regime = macro.regime
    if regime == "RISK_ON":
        status = "PASS"
        pts = 6
        det = "Macro environment is RISK_ON with favorable institutional tailwinds"
    elif regime == "MIXED":
        status = "PASS"
        pts = 5
        det = "Macro environment is MIXED; selective stock-specific accumulation"
    else:
        status = "WARN"
        pts = 2
        det = "Macro environment is RISK_OFF; institutional caution advised"
    criteria.append(CriterionCheck(
        name="Macro Regime & Domestic Liquidity",
        requirement="Risk-On or Stable Domestic Inflows",
        actual=regime,
        status=status,
        detail=det,
    ))
    score += pts

    total_score = max(0, min(100, score))

    # Verdict assignment
    pass_count = sum(1 for c in criteria if c.status == "PASS")
    if total_score >= 80 and pass_count >= 7:
        verdict = "STRONG BUY"
    elif total_score >= 65 and pass_count >= 5:
        verdict = "ACCUMULATE"
    elif total_score >= 50:
        verdict = "HOLD"
    else:
        verdict = "AVOID"

    # Multi-year valuation scenarios (12-month forward intrinsic targets)
    val_scenarios = {}
    growth_rate = (earn_g if earn_g is not None else (rev_g if rev_g is not None else 0.12))
    base_target = ltp * (1.0 + max(-0.15, min(0.35, growth_rate)))
    bull_target = ltp * (1.0 + max(0.05, min(0.60, growth_rate * 1.35)))
    bear_target = ltp * (1.0 - max(0.05, min(0.30, 0.15 if growth_rate >= 0 else abs(growth_rate))))

    val_scenarios["base"] = round(base_target, 2)
    val_scenarios["bull"] = round(bull_target, 2)
    val_scenarios["bear"] = round(bear_target, 2)

    # Cash conversion calculation
    ocf = company.operating_cashflow_cr if company else None
    fcf = company.free_cashflow_cr if company else None
    mcap = company.market_cap_cr if company else None
    fcf_yield = round((fcf / mcap * 100.0), 2) if (fcf and mcap and mcap > 0) else None

    if ocf and ocf > 0 and fcf and fcf > 0:
        conversion = "High Quality (Cash Generative)"
        strengths.append(f"Strong Cash Generator (FCF ₹{fcf:,.0f} Cr)")
    elif ocf and ocf > 0:
        conversion = "Capex Reinvestment Phase"
    elif ocf and ocf <= 0:
        conversion = "Negative OCF Warning"
        risks.append("Negative Operating Cash Flow")
    else:
        conversion = "Data Not Disclosed"

    # Banking / Financials specific ROA metrics
    if is_financial and company and company.roa:
        if company.roa >= 0.012:
            strengths.append(f"Healthy ROA ({company.roa*100:.2f}%)")
        elif company.roa < 0.007:
            risks.append(f"Sub-par Banking ROA ({company.roa*100:.2f}%)")

    # Multi-year thesis synthesis
    thesis_parts = []
    thesis_parts.append(f"{name} ({symbol}) shows a long-term institutional rating of {verdict} (Score: {total_score}/100, {pass_count}/10 criteria passed).")
    if rev_g and rev_g > 0.10:
        thesis_parts.append(f"Revenue is expanding at {rev_g*100:+.1f}% YoY with operating margins at {((opm or 0)*100):.1f}%.")
    if above_200:
        thesis_parts.append(f"Price is structurally supported above its 200 EMA at ₹{ltp:,.2f}.")
    else:
        thesis_parts.append(f"Price is currently below or near the 200 EMA; require base confirmation before aggressive allocation.")
    if pe:
        thesis_parts.append(f"Valuation stands at P/E {pe:.1f}x ({val_verdict}). Intrinsic targets: Base ₹{base_target:,.1f}, Bull ₹{bull_target:,.1f}, Bear ₹{bear_target:,.1f}.")
    full_thesis = " ".join(thesis_parts)

    return LongTermOpportunity(
        symbol=symbol,
        name=name,
        sector=sector,
        industry=industry,
        ltp=ltp,
        market_cap_cr=company.market_cap_cr if company else None,
        pe_ratio=pe,
        pb_ratio=company.pb_ratio if company else None,
        roe=roe,
        debt_to_equity=company.debt_to_equity if company else None,
        revenue_growth_yoy=rev_g,
        earnings_growth_yoy=earn_g,
        operating_margin=opm,
        high_52w=high52,
        low_52w=low52,
        ema200=ema200,
        above_200_ema=above_200,
        adx=adx_val,
        score=total_score,
        verdict=verdict,
        valuation_verdict=val_verdict,
        operating_cashflow_cr=ocf,
        free_cashflow_cr=fcf,
        fcf_yield_pct=fcf_yield,
        cash_conversion_status=conversion,
        valuation_scenarios=val_scenarios,
        criteria=criteria,
        strengths=strengths,
        risks=risks,
        thesis=full_thesis,
    )
