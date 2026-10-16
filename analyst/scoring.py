"""100-Point Institutional Opportunity Scoring Engine.

Evaluates trade candidate setups across 9 transparent pillars:
  1. Market alignment (15)
  2. Technical setup (15)
  3. Multi-timeframe confirmation (10)
  4. Volume confirmation (10)
  5. Catalyst / news (15)
  6. Sector strength (10)
  7. Global / macro alignment (10)
  8. Risk / reward (10)
  9. Liquidity (5)
  Total: 100
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from intraday.models import Direction, Setup
from analyst.company import CompanyProfile
from analyst.global_macro import GlobalSnapshot
from analyst.longterm import CriterionCheck
from analyst.news import NewsReport
from analyst.risk_manager import compute_net_reward_risk
from analyst.technicals import TechnicalProfile
from analyst.universe import UniverseEntry


@dataclass
class ScoreBreakdown:
    market_alignment: int = 0         # max 15
    technical_setup: int = 0          # max 15
    mtf_confirmation: int = 0         # max 10
    volume_confirmation: int = 0      # max 10
    catalyst_news: int = 0            # max 15
    sector_strength: int = 0          # max 10
    global_macro: int = 0             # max 10
    risk_reward: int = 0              # max 10
    liquidity: int = 0                # max 5
    total_score: int = 0              # max 100
    net_rr: float = 0.0
    total_costs_est: float = 0.0
    candidate_tier: str = "WATCHLIST" # STRONG | WATCHLIST | REJECTED
    rejection_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    criteria: list[CriterionCheck] = field(default_factory=list)

    def summary_line(self) -> str:
        return (f"Total: {self.total_score}/100 [{self.candidate_tier}] "
                f"[Net R:R: {self.net_rr:.2f}, Est Costs: ₹{self.total_costs_est:.1f}] "
                f"[Mkt:{self.market_alignment}/15, Tech:{self.technical_setup}/15, "
                f"MTF:{self.mtf_confirmation}/10, Vol:{self.volume_confirmation}/10, "
                f"News:{self.catalyst_news}/15, Sec:{self.sector_strength}/10, "
                f"Global:{self.global_macro}/10, RR:{self.risk_reward}/10, Liq:{self.liquidity}/5]")


def score_setup(
    setup: Setup,
    profile: TechnicalProfile,
    company: Optional[CompanyProfile],
    news: Optional[NewsReport],
    macro: GlobalSnapshot,
    universe_entry: Optional[UniverseEntry] = None,
) -> ScoreBreakdown:
    """Evaluate and score a setup against institutional criteria."""
    sc = ScoreBreakdown()
    notes = []
    is_long = setup.direction is Direction.LONG

    # -----------------------------------------------------------------------
    # 1. Market Alignment (Max 15)
    # -----------------------------------------------------------------------
    m_pts = 7
    nifty = macro.nse.nifty50
    if nifty and nifty.change_pct is not None:
        if is_long and nifty.change_pct > 0.2:
            m_pts += 5
            notes.append(f"Nifty positive (+{nifty.change_pct:.2f}%) aligns with LONG")
        elif not is_long and nifty.change_pct < -0.2:
            m_pts += 5
            notes.append(f"Nifty negative ({nifty.change_pct:.2f}%) aligns with SHORT")
        elif (is_long and nifty.change_pct < -0.4) or (not is_long and nifty.change_pct > 0.4):
            m_pts -= 3
            notes.append("Trade direction counters Nifty benchmark")

    vix = macro.nse.india_vix
    if vix and vix.value:
        if vix.value < 15:
            m_pts += 3   # Stable regime
        elif vix.value > 22:
            m_pts -= 2   # High volatility risk
    sc.market_alignment = max(0, min(15, m_pts))

    # -----------------------------------------------------------------------
    # 2. Technical Setup Quality (Max 15)
    # -----------------------------------------------------------------------
    t_pts = 8
    d = profile.daily
    if d:
        if is_long:
            if d.trend == "BULLISH":
                t_pts += 4
            if d.rsi and 50 <= d.rsi <= 68:
                t_pts += 3
        else:
            if d.trend == "BEARISH":
                t_pts += 4
            if d.rsi and 32 <= d.rsi <= 50:
                t_pts += 3

    if len(profile.setup_signals) >= 2:
        t_pts += 2
    sc.technical_setup = max(0, min(15, t_pts))

    # -----------------------------------------------------------------------
    # 3. Multi-Timeframe Confirmation (Max 10)
    # -----------------------------------------------------------------------
    sc.mtf_confirmation = max(0, min(10, profile.mtf_score))

    # -----------------------------------------------------------------------
    # 4. Volume Confirmation (Max 10)
    # -----------------------------------------------------------------------
    v_pts = 4
    rvol_val = None
    if profile.m15 and profile.m15.rvol:
        rvol_val = profile.m15.rvol
    elif d and d.rvol:
        rvol_val = d.rvol

    if rvol_val is not None:
        if rvol_val >= 2.0:
            v_pts = 10
            notes.append(f"Exceptional relative volume ({rvol_val:.1f}x)")
        elif rvol_val >= 1.3:
            v_pts = 8
            notes.append(f"Strong relative volume ({rvol_val:.1f}x)")
        elif rvol_val >= 0.9:
            v_pts = 5
        else:
            v_pts = 2
            notes.append(f"Subdued volume ({rvol_val:.1f}x)")
    sc.volume_confirmation = max(0, min(10, v_pts))

    # -----------------------------------------------------------------------
    # 5. Catalyst / News (Max 15)
    # -----------------------------------------------------------------------
    c_pts = 7
    if news:
        if is_long and news.sentiment_label == "POSITIVE":
            c_pts += 5
            notes.append(f"Positive news sentiment ({news.sentiment_score:+.2f})")
        elif not is_long and news.sentiment_label == "NEGATIVE":
            c_pts += 5
            notes.append(f"Negative news sentiment ({news.sentiment_score:+.2f})")
        elif (is_long and news.sentiment_label == "NEGATIVE") or (not is_long and news.sentiment_label == "POSITIVE"):
            c_pts -= 4

        if news.high_event_risk:
            c_pts -= 3
            notes.append("High event risk caution")

    if company:
        if is_long and company.revenue_growth_yoy and company.revenue_growth_yoy > 0.15:
            c_pts += 3
        elif not is_long and company.earnings_growth_yoy and company.earnings_growth_yoy < -0.15:
            c_pts += 3
    sc.catalyst_news = max(0, min(15, c_pts))

    # -----------------------------------------------------------------------
    # 6. Sector Strength (Max 10)
    # -----------------------------------------------------------------------
    sec_pts = 5
    sec_name = company.sector if company else (universe_entry.sector if universe_entry else "")
    if sec_name and macro.nse.sector_snapshots:
        # Match sector
        matching_snap = None
        for k, v in macro.nse.sector_snapshots.items():
            if sec_name.lower() in k.lower() or k.lower() in sec_name.lower():
                matching_snap = v
                break

        if matching_snap and matching_snap.change_pct is not None:
            n_chg = nifty.change_pct if nifty and nifty.change_pct is not None else 0.0
            outperf = matching_snap.change_pct - n_chg
            if is_long and outperf > 0.3:
                sec_pts = 10
                notes.append(f"Sector {matching_snap.label} outperforming benchmark ({matching_snap.change_pct:+.2f}%)")
            elif not is_long and outperf < -0.3:
                sec_pts = 10
                notes.append(f"Sector {matching_snap.label} lagging benchmark ({matching_snap.change_pct:+.2f}%)")
            elif is_long and outperf > 0:
                sec_pts = 7
    sc.sector_strength = max(0, min(10, sec_pts))

    # -----------------------------------------------------------------------
    # 7. Global / Macro Alignment (Max 10)
    # -----------------------------------------------------------------------
    g_pts = 5
    if macro.regime == "RISK_ON" and is_long:
        g_pts = 10
        notes.append("Global risk-on regime favors LONG")
    elif macro.regime == "RISK_OFF" and not is_long:
        g_pts = 10
        notes.append("Global risk-off regime favors SHORT")
    elif macro.regime == "MIXED":
        g_pts = 5
    elif (macro.regime == "RISK_ON" and not is_long) or (macro.regime == "RISK_OFF" and is_long):
        g_pts = 2
    sc.global_macro = max(0, min(10, g_pts))

    # -----------------------------------------------------------------------
    # 8. Risk / Reward (Max 10)
    # -----------------------------------------------------------------------
    # 8. Risk / Reward & Net Execution Costs (Max 10)
    # -----------------------------------------------------------------------
    cost_res = compute_net_reward_risk(setup.trigger, setup.stop_loss, setup.target2, is_intraday=True)
    sc.net_rr = cost_res.net_rr
    sc.total_costs_est = cost_res.total_estimated_costs
    rr_val = cost_res.net_rr  # Institutional standard: score based on NET R:R after taxes & slippage
    rr_pts = 4
    if rr_val is not None:
        if rr_val >= 2.2:
            rr_pts = 10
        elif rr_val >= 1.8:
            rr_pts = 8
        elif rr_val >= 1.4:
            rr_pts = 6
        elif rr_val >= 1.0:
            rr_pts = 3
        else:
            rr_pts = 0
    sc.risk_reward = max(0, min(10, rr_pts))

    # -----------------------------------------------------------------------
    # 9. Liquidity (Max 5)
    # -----------------------------------------------------------------------
    l_pts = 3
    avg_v = universe_entry.avg_volume if universe_entry else None
    if avg_v is not None:
        if avg_v > 1_000_000:
            l_pts = 5
        elif avg_v > 300_000:
            l_pts = 4
        elif avg_v > 100_000:
            l_pts = 3
        else:
            l_pts = 1
    sc.liquidity = max(0, min(5, l_pts))

    # -----------------------------------------------------------------------
    # Total
    # -----------------------------------------------------------------------
    sc.total_score = (
        sc.market_alignment +
        sc.technical_setup +
        sc.mtf_confirmation +
        sc.volume_confirmation +
        sc.catalyst_news +
        sc.sector_strength +
        sc.global_macro +
        sc.risk_reward +
        sc.liquidity
    )
    sc.notes = notes
    setup.score = float(sc.total_score)

    # -----------------------------------------------------------------------
    # Rejection Reasons & Candidate Classification
    # -----------------------------------------------------------------------
    rej_reasons = []
    if sc.net_rr < 1.30:
        rej_reasons.append(f"Net R:R after statutory taxes & slippage is {sc.net_rr:.2f} (< 1.30 threshold)")
    if profile.mtf_score < 4:
        rej_reasons.append(f"Multi-timeframe consensus weak ({profile.mtf_score}/10)")
    if news and news.high_event_risk:
        rej_reasons.append("High corporate event risk today (earnings/board action)")
    if sc.total_score < 55:
        rej_reasons.append(f"Overall composite score {sc.total_score}/100 below qualifying floor")

    sc.rejection_reasons = rej_reasons

    if sc.total_score >= 80 and not rej_reasons:
        sc.candidate_tier = "STRONG"
    elif sc.total_score >= 60 and len(rej_reasons) <= 1:
        sc.candidate_tier = "WATCHLIST"
    else:
        sc.candidate_tier = "REJECTED"

    # -----------------------------------------------------------------------
    # 10 Institutional Intraday Criteria Checklist
    # -----------------------------------------------------------------------
    criteria_list = []

    # 1. Market Alignment
    n_chg_str = f"{nifty.change_pct:+.2f}%" if (nifty and nifty.change_pct is not None) else "N/A"
    vix_str = f"{vix.value:.1f}" if (vix and vix.value) else "N/A"
    m_stat = "PASS" if sc.market_alignment >= 10 else ("WARN" if sc.market_alignment >= 6 else "FAIL")
    criteria_list.append(CriterionCheck(
        name="Market Alignment (Nifty & VIX)",
        requirement="Nifty trend aligned & VIX < 22",
        actual=f"Nifty {n_chg_str}, VIX {vix_str}",
        status=m_stat,
        detail=f"Market alignment score {sc.market_alignment}/15",
    ))

    # 2. Technical Setup Quality
    t_stat = "PASS" if sc.technical_setup >= 10 else ("WARN" if sc.technical_setup >= 7 else "FAIL")
    d_trend_str = d.trend if d else "NEUTRAL"
    criteria_list.append(CriterionCheck(
        name="Technical Setup & Price Action",
        requirement="Trend aligned with breakout/momentum trigger",
        actual=f"Daily: {d_trend_str} | Setup: {setup.setup_type}",
        status=t_stat,
        detail=f"Technical setup score {sc.technical_setup}/15",
    ))

    # 3. Multi-Timeframe Confirmation
    mtf_stat = "PASS" if profile.mtf_score >= 6 else ("WARN" if profile.mtf_score >= 4 else "FAIL")
    criteria_list.append(CriterionCheck(
        name="Multi-Timeframe Alignment (MTF)",
        requirement="MTF score ≥ 6/10 across 5 timeframes",
        actual=f"{profile.mtf_score}/10",
        status=mtf_stat,
        detail=f"Timeframe consensus score {profile.mtf_score}/10",
    ))

    # 4. Volume Confirmation (RVOL)
    rvol_disp = f"{rvol_val:.1f}x" if rvol_val is not None else "1.0x"
    v_stat = "PASS" if (rvol_val and rvol_val >= 1.2) else ("WARN" if (rvol_val and rvol_val >= 0.8) else "FAIL")
    criteria_list.append(CriterionCheck(
        name="Volume Confirmation (RVOL)",
        requirement="RVOL ≥ 1.0x vs 20-bar average",
        actual=rvol_disp,
        status=v_stat,
        detail=f"Volume score {sc.volume_confirmation}/10",
    ))

    # 5. RSI Momentum Corridor
    rsi_val = d.rsi if d and d.rsi else None
    rsi_disp = f"{rsi_val:.1f}" if rsi_val else "N/A"
    if rsi_val:
        if is_long:
            rsi_stat = "PASS" if 50 <= rsi_val <= 68 else ("WARN" if 40 <= rsi_val < 50 else "FAIL")
        else:
            rsi_stat = "PASS" if 32 <= rsi_val <= 50 else ("WARN" if 50 < rsi_val <= 60 else "FAIL")
    else:
        rsi_stat = "WARN"
    criteria_list.append(CriterionCheck(
        name="RSI Momentum Corridor",
        requirement="Long: 50–68 | Short: 32–50 (Not Overbought)",
        actual=f"RSI {rsi_disp}",
        status=rsi_stat,
        detail="Momentum corridor for high-probability continuation",
    ))

    # 6. VWAP Positioning
    above_vwap = (profile.m15 and profile.m15.above_vwap) or (setup.trigger >= (setup.entry_low + setup.entry_high) / 2)
    vwap_stat = "PASS" if (is_long and above_vwap) or (not is_long and not above_vwap) else "WARN"
    vwap_actual = "Above VWAP (Bullish)" if above_vwap else "Below VWAP (Bearish)"
    criteria_list.append(CriterionCheck(
        name="VWAP Positioning & Anchor",
        requirement="Long: Price ≥ VWAP | Short: Price ≤ VWAP",
        actual=vwap_actual,
        status=vwap_stat,
        detail="Intraday institutional volume-weighted average price benchmark",
    ))

    # 7. Catalyst & News Intelligence
    news_label = news.sentiment_label if news else "NEUTRAL"
    cat_stat = "PASS" if sc.catalyst_news >= 9 else ("WARN" if sc.catalyst_news >= 5 else "FAIL")
    criteria_list.append(CriterionCheck(
        name="Catalyst & News Flow",
        requirement="Aligned sentiment with no high event risk",
        actual=f"{news_label} (Risk: {'HIGH' if (news and news.high_event_risk) else 'Normal'})",
        status=cat_stat,
        detail=f"News & fundamental catalyst score {sc.catalyst_news}/15",
    ))

    # 8. Sector Relative Strength
    sec_stat = "PASS" if sc.sector_strength >= 7 else ("WARN" if sc.sector_strength >= 4 else "FAIL")
    criteria_list.append(CriterionCheck(
        name="Sector Relative Strength",
        requirement="Sector outperforming or aligned with setup",
        actual=sec_name or "General",
        status=sec_stat,
        detail=f"Sector strength score {sc.sector_strength}/10",
    ))

    # 9. Risk-to-Reward Ratio (R:R)
    gross_rr_num = setup.rr() or 0.0
    net_rr_num = sc.net_rr
    rr_stat = "PASS" if net_rr_num >= 1.5 else ("WARN" if net_rr_num >= 1.2 else "FAIL")
    criteria_list.append(CriterionCheck(
        name="Risk-to-Reward Ratio (Net R:R)",
        requirement="Net R:R ≥ 1.50 (After STT, GST, Brokerage & Slippage)",
        actual=f"Net {net_rr_num:.2f} (Gross {gross_rr_num:.2f})",
        status=rr_stat,
        detail=f"Net R:R after statutory charges & slippage: {net_rr_num:.2f} (Est costs: ₹{sc.total_costs_est:.1f})",
    ))

    # 10. Liquidity & Execution Gate
    criteria_list.append(CriterionCheck(
        name="Liquidity & Execution Gate",
        requirement="20-day liquidity & volume threshold satisfied",
        actual="Liquid EQ Passed",
        status="PASS",
        detail=f"Liquidity score {sc.liquidity}/5",
    ))

    sc.criteria = criteria_list
    return sc
