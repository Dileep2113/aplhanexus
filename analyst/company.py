"""Company profile & fundamental analysis engine.

Extracts business details, key financial metrics, valuation, earnings performance,
and corporate event indicators via Yahoo Finance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompanyProfile:
    symbol: str
    name: str
    sector: str = "Unknown"
    industry: str = "Unknown"
    business_summary: str = ""
    market_cap_cr: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    promoter_holding_pct: Optional[float] = None
    institutional_holding_pct: Optional[float] = None
    operating_cashflow_cr: Optional[float] = None
    free_cashflow_cr: Optional[float] = None
    roa: Optional[float] = None
    current_ratio: Optional[float] = None
    fundamental_score: int = 50   # 0 to 100
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    valuation_verdict: str = "Fair"

    def render_summary(self) -> str:
        lines = [
            f"  {self.name} ({self.symbol}) | Sector: {self.sector} | Industry: {self.industry}",
        ]
        mcap = f"₹{self.market_cap_cr:,.0f} Cr" if self.market_cap_cr else "N/A"
        pe = f"{self.pe_ratio:.1f}x" if self.pe_ratio else "N/A"
        fpe = f"{self.forward_pe:.1f}x" if self.forward_pe else "N/A"
        pb = f"{self.pb_ratio:.1f}x" if self.pb_ratio else "N/A"
        roe = f"{self.roe * 100:.1f}%" if self.roe else "N/A"
        de = f"{self.debt_to_equity:.2f}" if self.debt_to_equity is not None else "N/A"
        rev_g = f"{self.revenue_growth_yoy * 100:+.1f}%" if self.revenue_growth_yoy is not None else "N/A"
        earn_g = f"{self.earnings_growth_yoy * 100:+.1f}%" if self.earnings_growth_yoy is not None else "N/A"
        opm = f"{self.operating_margin * 100:.1f}%" if self.operating_margin is not None else "N/A"

        lines.append(f"    Cap: {mcap} | P/E: {pe} (Fwd: {fpe}) | P/B: {pb} | D/E: {de} | Beta: {self.beta or 'N/A'}")
        lines.append(f"    ROE: {roe} | OPM: {opm} | Rev Growth: {rev_g} | PAT Growth: {earn_g}")
        if self.strengths:
            lines.append(f"    Strengths: {', '.join(self.strengths[:3])}")
        if self.risks:
            lines.append(f"    Risks: {', '.join(self.risks[:3])}")
        return "\n".join(lines)


def fetch_company_profile(yahoo_symbol: str) -> Optional[CompanyProfile]:
    """Fetch fundamental profile and business metrics for a symbol."""
    import yfinance as yf

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(yahoo_symbol)
            info = t.info or {}
            # Fallback to BSE (.BO) if needed
            if (not info or ("shortName" not in info and "longName" not in info and "regularMarketPrice" not in info)) and yahoo_symbol.endswith(".NS"):
                bse_sym = yahoo_symbol.replace(".NS", ".BO")
                t_bse = yf.Ticker(bse_sym)
                bse_info = t_bse.info or {}
                if bse_info:
                    info = bse_info
    except Exception:
        return None

    if not info or ("shortName" not in info and "longName" not in info and "regularMarketPrice" not in info):
        return None

    name = info.get("longName") or info.get("shortName") or yahoo_symbol.replace(".NS", "")
    sector = info.get("sector") or info.get("sectorDisp") or "General"
    industry = info.get("industry") or info.get("industryDisp") or "General"
    summary = info.get("longBusinessSummary", "")

    def _safe_float(v) -> Optional[float]:
        if v is None:
            return None
        try:
            import math
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None

    raw_mcap = _safe_float(info.get("marketCap"))
    mcap_cr = (raw_mcap / 10000000) if raw_mcap else None

    raw_ocf = _safe_float(info.get("operatingCashflow"))
    ocf_cr = (raw_ocf / 10000000) if raw_ocf else None

    raw_fcf = _safe_float(info.get("freeCashflow"))
    fcf_cr = (raw_fcf / 10000000) if raw_fcf else None

    profile = CompanyProfile(
        symbol=yahoo_symbol,
        name=name,
        sector=sector,
        industry=industry,
        business_summary=summary[:350] + ("..." if len(summary) > 350 else ""),
        market_cap_cr=mcap_cr,
        pe_ratio=_safe_float(info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forwardPE")),
        pb_ratio=_safe_float(info.get("priceToBook")),
        roe=_safe_float(info.get("returnOnEquity")),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        revenue_growth_yoy=_safe_float(info.get("revenueGrowth")),
        earnings_growth_yoy=_safe_float(info.get("earningsGrowth")),
        operating_margin=_safe_float(info.get("operatingMargins")),
        profit_margin=_safe_float(info.get("profitMargins")),
        dividend_yield=_safe_float(info.get("dividendYield")),
        beta=_safe_float(info.get("beta")),
        high_52w=_safe_float(info.get("fiftyTwoWeekHigh")),
        low_52w=_safe_float(info.get("fiftyTwoWeekLow")),
        promoter_holding_pct=_safe_float(info.get("heldPercentInsiders")),
        institutional_holding_pct=_safe_float(info.get("heldPercentInstitutions")),
        operating_cashflow_cr=ocf_cr,
        free_cashflow_cr=fcf_cr,
        roa=_safe_float(info.get("returnOnAssets")),
        current_ratio=_safe_float(info.get("currentRatio")),
    )

    # Fundamental Scoring and Strengths/Risks evaluation
    score = 50
    strengths = []
    risks = []

    # Growth & Margins
    if profile.revenue_growth_yoy is not None:
        if profile.revenue_growth_yoy > 0.15:
            score += 10
            strengths.append(f"Strong YoY Rev Growth ({profile.revenue_growth_yoy * 100:+.1f}%)")
        elif profile.revenue_growth_yoy < -0.05:
            score -= 10
            risks.append(f"Declining Revenue ({profile.revenue_growth_yoy * 100:+.1f}%)")

    if profile.earnings_growth_yoy is not None:
        if profile.earnings_growth_yoy > 0.20:
            score += 10
            strengths.append(f"Robust Profit Growth ({profile.earnings_growth_yoy * 100:+.1f}%)")
        elif profile.earnings_growth_yoy < -0.10:
            score -= 10
            risks.append(f"Earnings Contraction ({profile.earnings_growth_yoy * 100:+.1f}%)")

    if profile.operating_margin is not None:
        if profile.operating_margin > 0.20:
            score += 10
            strengths.append(f"High Operating Margin ({profile.operating_margin * 100:.1f}%)")
        elif profile.operating_margin < 0.05:
            score -= 5
            risks.append(f"Thin Margins ({profile.operating_margin * 100:.1f}%)")

    # Solvency / Leverage
    if profile.debt_to_equity is not None:
        if profile.debt_to_equity < 50:
            score += 10
            strengths.append("Conservative Debt (<50 D/E)")
        elif profile.debt_to_equity > 150:
            score -= 10
            risks.append("High Leverage (>150 D/E)")

    # Valuation verdict
    if profile.pe_ratio:
        if profile.pe_ratio < 15 and (profile.earnings_growth_yoy or 0) >= 0:
            profile.valuation_verdict = "Attractive / Undervalued"
            score += 5
        elif profile.pe_ratio > 60:
            profile.valuation_verdict = "Premium / High Multiple"
            score -= 5
        else:
            profile.valuation_verdict = "Fair Value"

    profile.fundamental_score = max(10, min(95, score))
    profile.strengths = strengths
    profile.risks = risks

    return profile


if __name__ == "__main__":
    p = fetch_company_profile("RELIANCE.NS")
    if p:
        print(p.render_summary())
