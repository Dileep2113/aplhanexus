"""Institutional Research, Global Intelligence & Trade Opportunity Report Formatter.

Formats the complete output specified by institutional guidelines:
  * Global Macro & Regional Snapshot (US, Europe, Asia, Commodities, Currencies, Bond Yields)
  * Indian Pre-Market Outlook & GIFT Nifty Scenario Analysis
  * Dedicated Table 1: High-Conviction Intraday Trading Setups (with Net R:R, Costs, Rejection Logs)
  * Dedicated Table 2: Long-Term Investment Screener (with Cash Flow Quality, Moat & Valuation Scenarios)
  * Modular Strategy Backtest & Historical Validation Dashboard
  * Master Criteria Reference & Methodology Matrix
  * Governance, Attribution & Data Freshness Badges
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
from pathlib import Path
from typing import Optional

from intraday.models import Direction, Setup
from analyst.company import CompanyProfile
from analyst.global_macro import GlobalSnapshot
from analyst.longterm import LongTermOpportunity, CriterionCheck
from analyst.news import NewsReport
from analyst.premarket import PreMarketReport, generate_premarket_report
from analyst.scoring import ScoreBreakdown
from analyst.technicals import TechnicalProfile


@dataclass
class ScoredOpportunity:
    setup: Setup
    profile: TechnicalProfile
    company: Optional[CompanyProfile]
    news: Optional[NewsReport]
    score_breakdown: ScoreBreakdown
    macro: GlobalSnapshot


def render_trade_thesis(opp: ScoredOpportunity) -> str:
    """Generate the structured 9-step trade thesis for a selected opportunity."""
    s = opp.setup
    c = opp.company
    p = opp.profile
    n = opp.news
    m = opp.macro
    sb = opp.score_breakdown

    lines = []
    add = lines.append

    add(f"╔═══════════════════════════════════════════════════════════════════════════════════════╗")
    add(f"║ TRADE THESIS: {s.symbol:<12} | {s.direction.value:<5} | SCORE: {sb.total_score:>2}/100 [{sb.candidate_tier}] | {s.setup_type:<20} ║")
    add(f"╚═══════════════════════════════════════════════════════════════════════════════════════╝")

    # Company
    c_name = c.name if c else s.symbol
    c_sec = c.sector if c else "Unknown"
    c_pe = f"P/E {c.pe_ratio:.1f}x" if c and c.pe_ratio else ""
    c_rev = f"Rev Growth {c.revenue_growth_yoy*100:+.1f}%" if c and c.revenue_growth_yoy is not None else ""
    add(f"  🏢 Company   → {c_name} ({c_sec}). {c_pe} {c_rev}. {c.business_summary[:160] if c and c.business_summary else 'Core sector constituent.'}")

    # Industry
    add(f"  🏭 Industry  → {c.industry if c else 'Broader Sector'}. Trend aligned with institutional flows.")

    # News
    if n and n.items:
        top_news = n.items[0].title
        add(f"  📰 News      → {top_news} (Sentiment: {n.sentiment_label})")
    else:
        add(f"  📰 News      → No fresh breaking headlines in last 24h; moving on structural flow.")

    # Global
    us_lead = m.us[0].as_line().strip() if m.us else "Global stable"
    add(f"  🌍 Global    → Regime is {m.regime}. US benchmark: {us_lead}.")

    # Market
    nifty = m.nse.nifty50
    n_str = f"Nifty at {nifty.value:,.2f} ({nifty.change_pct:+.2f}%)" if nifty and nifty.change_pct is not None else "NSE active"
    vix = m.nse.india_vix
    v_str = f", India VIX {vix.value:.2f}" if vix else ""
    add(f"  📈 Market    → {n_str}{v_str}. A/D Breadth: {m.nse.advance_decline or 'N/A'}.")

    # Technical
    d = p.daily
    d_trend = d.trend if d else "NEUTRAL"
    rsi_str = f"RSI={d.rsi:.1f}" if d and d.rsi else ""
    adx_str = f"ADX={d.adx:.1f}" if d and d.adx else ""
    add(f"  📐 Technical → MTF Alignment: {p.mtf_score}/10. Daily trend {d_trend}. {rsi_str} {adx_str}. Key signals: {', '.join(p.setup_signals[:3]) or 'Clean break'}.")

    # Volume
    rvol_val = d.rvol if d and d.rvol else (p.m15.rvol if p.m15 and p.m15.rvol else None)
    vol_str = f"RVOL {rvol_val:.1f}x relative to 20-bar average." if rvol_val else "Volume within normal session limits."
    add(f"  📊 Volume    → {vol_str}")

    # Catalyst
    cat = n.primary_catalyst if n and n.primary_catalyst else ("Technical Breakout + Structural Momentum" if s.direction is Direction.LONG else "Technical Breakdown + Overhead Supply")
    add(f"  ⚡ Catalyst  → {cat}")

    # Exact Trade Trigger & Costs
    entry_zone = f"₹{s.entry_low:,.2f} – ₹{s.entry_high:,.2f}"
    add(f"  🎯 Trade     → {s.direction.value} ENTRY on 5-minute close beyond trigger ₹{s.trigger:,.2f}.")
    add(f"                 Entry Zone: {entry_zone}")
    add(f"                 Stop Loss : ₹{s.stop_loss:,.2f} (strictly non-negotiable)")
    add(f"                 Target 1  : ₹{s.target1:,.2f} (partial booking & trail)")
    add(f"                 Target 2  : ₹{s.target2:,.2f} (move: {s.expected_move_pct():.2f}%, Gross R:R {s.rr() or 0:.2f}, Net R:R {sb.net_rr:.2f})")
    add(f"                 Est Costs : ₹{sb.total_costs_est:.1f} (Brokerage, STT, GST, Exchange charges & slippage)")
    add(f"                 Cancel/Inv: ₹{s.invalidate:,.2f} (structure collapsed; cancel setup)")

    if sb.rejection_reasons:
        add(f"  ⚠️ Caveats   → {', '.join(sb.rejection_reasons)}")

    return "\n".join(lines)


def generate_full_report(
    macro: GlobalSnapshot,
    opportunities: list[ScoredOpportunity],
    total_scanned: int,
    min_score_threshold: int = 60,
    max_picks: int = 5,
    longterm_opportunities: Optional[list[LongTermOpportunity]] = None,
    premarket_report: Optional[PreMarketReport] = None,
    backtest_report: Optional[object] = None,
) -> str:
    """Generate the complete executive report for console output."""
    lines = []
    add = lines.append

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    add("═" * 89)
    add(f"   GLOBAL-MARKET-AWARE INSTITUTIONAL TRADING & RESEARCH DESK   |   {now_str}")
    add("═" * 89)

    # 1. Macro and Global Overview
    add(macro.render())
    add("")

    # 2. Pre-market summary
    pm = premarket_report or generate_premarket_report(macro)
    add(pm.render_console())
    add("")

    # Filter qualified opportunities
    qualified = [o for o in opportunities if o.score_breakdown.total_score >= min_score_threshold]
    qualified.sort(key=lambda o: o.score_breakdown.total_score, reverse=True)
    top_picks = qualified[:max_picks]

    add(f"🔍 SCAN AUDIT: {total_scanned} liquid symbols analyzed | {len(opportunities)} intraday setups identified | {len(qualified)} qualified (≥{min_score_threshold}/100)")
    add("─" * 89)

    # 3. Opportunities Table or NO TRADE
    if not top_picks:
        add("")
        add("*****************************************************************************************")
        add("  🛑 NO TRADE — INTRADAY SETUPS DO NOT MEET THE REQUIRED CONDITIONS.")
        add("*****************************************************************************************")
        add("  Reason: No screened candidate satisfied all fundamental, technical, macro, and")
        add(f"          risk-reward quality thresholds (minimum score {min_score_threshold}/100 required).")
        add("  Principle: Capital preservation takes precedence over forcing low-conviction trades.")
        add("")
    else:
        add(f"⚡ TOP {len(top_picks)} HIGH-CONVICTION INTRADAY SETUPS (AFTER STATUTORY TAXES & SLIPPAGE)")
        add("─" * 89)

        header = (f"{'Symbol':<12} {'Dir':<5} {'LTP':>8} {'Entry Zone':<17} {'Stop Loss':>9} "
                  f"{'Target 2':>9} {'Net R:R':>7} {'Score':>5} {'Tier':<10} {'Setup Type':<16}")
        add(header)
        add("─" * 89)

        for o in top_picks:
            s = o.setup
            sb = o.score_breakdown
            ez = f"{s.entry_low:,.1f}-{s.entry_high:,.1f}"
            row = (f"{s.symbol:<12} {s.direction.value:<5} {o.profile.ltp:>8.2f} {ez:<17} "
                   f"{s.stop_loss:>9.2f} {s.target2:>9.2f} {sb.net_rr:>7.2f} "
                   f"{int(s.score or 0):>4}  {sb.candidate_tier:<10} {s.setup_type:<16}")
            add(row)

        add("─" * 89)
        add("")

        # Detailed Theses for each top pick
        for o in top_picks:
            add(render_trade_thesis(o))
            add("")

    # Long-Term Section (if present)
    if longterm_opportunities:
        lt_qualified = [o for o in longterm_opportunities if o.score >= 60]
        lt_qualified.sort(key=lambda o: o.score, reverse=True)
        top_lt = lt_qualified[:max_picks]
        if top_lt:
            add("─" * 89)
            add(f"🏛️ TOP {len(top_lt)} LONG-TERM INVESTMENT CANDIDATES (STRUCTURAL & CASH FLOW QUALITY)")
            add("─" * 89)
            lt_header = f"{'Symbol':<12} {'Verdict':<12} {'LTP':>8} {'M.Cap(Cr)':>10} {'P/E':>6} {'ROE%':>6} {'D/E':>6} {'Rev YoY':>8} {'Score':>5}"
            add(lt_header)
            add("─" * 89)
            for lo in top_lt:
                mc_str = f"{lo.market_cap_cr:,.0f}" if lo.market_cap_cr else "N/A"
                pe_str = f"{lo.pe_ratio:.1f}" if lo.pe_ratio else "N/A"
                roe_str = f"{lo.roe*100:.1f}%" if lo.roe else "N/A"
                de_str = f"{lo.debt_to_equity/100:.2f}" if (lo.debt_to_equity and lo.debt_to_equity > 5) else (f"{lo.debt_to_equity:.2f}" if lo.debt_to_equity is not None else "N/A")
                rev_str = f"{lo.revenue_growth_yoy*100:+.1f}%" if lo.revenue_growth_yoy is not None else "N/A"
                lt_row = f"{lo.symbol:<12} {lo.verdict:<12} {lo.ltp:>8.2f} {mc_str:>10} {pe_str:>6} {roe_str:>6} {de_str:>6} {rev_str:>8} {lo.score:>5}"
                add(lt_row)
            add("─" * 89)
            add("")

    # Backtest Summary (if present)
    if backtest_report and hasattr(backtest_report, "summary_lines"):
        add("─" * 89)
        add("📊 STRATEGY BACKTESTING VALIDATION (HISTORICAL SIMULATION WITH TAXES & SLIPPAGE)")
        add("─" * 89)
        for line in backtest_report.summary_lines():
            add(f"  {line}")
        add("─" * 89)
        add("")

    # Mandatory Provenance & Disclaimers
    add("⚖️  GOVERNANCE & DATA INTEGRITY NOTICE:")
    add("  • All market prices from Yahoo Finance / NSE India keyless feeds (delayed ~15m / historical).")
    add("  • Net R:R calculated after Indian statutory taxes: STT (0.025%), GST (18%), NSE txn & slippage.")
    add("  • In accordance with institutional guidelines, ENTRY signals require live tick confirmation.")
    add("  • No guaranteed profits. Trade decisions remain subject to strict capital-at-risk limits.")
    add("═" * 89)

    return "\n".join(lines)


def generate_html_report(
    macro: GlobalSnapshot,
    opportunities: list[ScoredOpportunity],
    total_scanned: int,
    min_score_threshold: int = 55,
    max_picks: int = 25,
    output_path: Optional[object] = None,
    longterm_opportunities: Optional[list[LongTermOpportunity]] = None,
    premarket_report: Optional[PreMarketReport] = None,
    backtest_report: Optional[object] = None,
) -> str:
    """Generate an institutional-grade interactive HTML report with complete global intelligence."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    pm = premarket_report or generate_premarket_report(macro)

    # 1. Filter and sort Intraday Opportunities
    qualified = [o for o in opportunities if o.score_breakdown.total_score >= min_score_threshold]
    qualified.sort(key=lambda o: o.score_breakdown.total_score, reverse=True)
    top_intraday = qualified[:max_picks] if qualified else [o for o in sorted(opportunities, key=lambda x: x.score_breakdown.total_score, reverse=True)[:max_picks]]

    # 2. Filter and sort Long-Term Opportunities
    lt_list = longterm_opportunities or []
    lt_sorted = sorted(lt_list, key=lambda o: o.score, reverse=True)
    top_longterm = [o for o in lt_sorted if o.score >= 50][:max_picks] if any(o.score >= 50 for o in lt_sorted) else lt_sorted[:max_picks]

    # Helper for formatting macro price cards
    def _pt_card(label: str, pt, tag: str = "DELAYED ~15m"):
        if not pt:
            return ""
        chg = pt.change_pct if pt.change_pct is not None else 0.0
        chg_cls = "text-emerald-400" if chg >= 0 else "text-rose-400"
        chg_sign = "+" if chg > 0 else ""
        val_str = f"{pt.value:,.2f}" if pt.value is not None else "N/A"
        return f"""
        <div class="bg-slate-800/80 border border-slate-700/60 rounded-lg p-2.5 flex flex-col justify-between">
          <div class="flex items-center justify-between text-[10px] text-slate-400">
            <span class="truncate font-medium">{html.escape(label)}</span>
            <span class="text-[9px] px-1 py-0.2 rounded bg-slate-900 text-slate-500">{tag}</span>
          </div>
          <div class="text-sm font-bold text-slate-100 mt-1">{val_str}</div>
          <div class="text-xs font-semibold {chg_cls} mt-0.5">{chg_sign}{chg:.2f}%</div>
        </div>
        """

    us_cards = "".join(_pt_card(p.label, p) for p in macro.us[:4])
    fut_cards = "".join(_pt_card(p.label, p, "FUTURES") for p in macro.us_futures[:2])
    eu_cards = "".join(_pt_card(p.label, p, "EU CLOSE") for p in macro.europe[:3])
    asia_cards = "".join(_pt_card(p.label, p, "ASIA CLOSE") for p in macro.asian[:3])
    comm_cards = "".join(_pt_card(p.label, p, "COMMODITY") for p in macro.commodities[:4])
    curr_cards = "".join(_pt_card(p.label, p, "FOREX") for p in macro.currency[:2])

    nifty = macro.nse.nifty50
    bank = macro.nse.bank_nifty
    vix = macro.nse.india_vix
    nifty_card = _pt_card("NIFTY 50", nifty, "NSE DELAYED")
    bank_card = _pt_card("BANK NIFTY", bank, "NSE DELAYED")
    vix_card = _pt_card("INDIA VIX", vix, "NSE DELAYED")

    fii_val = macro.nse.fii_net_cr or 0
    dii_val = macro.nse.dii_net_cr or 0
    fii_str = f"+{fii_val:,} cr" if fii_val > 0 else f"{fii_val:,} cr"
    dii_str = f"+{dii_val:,} cr" if dii_val > 0 else f"{dii_val:,} cr"
    fii_cls = "text-emerald-400" if fii_val >= 0 else "text-rose-400"
    dii_cls = "text-emerald-400" if dii_val >= 0 else "text-rose-400"

    regime_color = "emerald" if macro.regime == "RISK_ON" else ("rose" if macro.regime == "RISK_OFF" else "amber")

    # -------------------------------------------------------------------------
    # Render Pre-Market Scenario Cards
    # -------------------------------------------------------------------------
    pm_scenarios_html = []
    for sc in pm.scenarios:
        b_color = "emerald" if "BULLISH" in sc.name else ("rose" if "BEARISH" in sc.name else "amber")
        cat_items = "".join(f"<li class='text-[11px] text-slate-300'>• {html.escape(c)}</li>" for c in sc.primary_catalysts)
        pm_scenarios_html.append(f"""
        <div class="bg-slate-900/80 p-3.5 rounded-xl border border-{b_color}-800/50 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-xs font-bold text-{b_color}-400">{html.escape(sc.name)}</span>
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-{b_color}-950 text-{b_color}-300 border border-{b_color}-700/50">{sc.probability_bias} BIAS</span>
            </div>
            <div class="text-[11px] font-mono text-slate-400 mb-2">Target Zone: <strong class="text-white">{html.escape(sc.opening_range_nifty)}</strong></div>
            <ul class="space-y-1 mb-2">
              {cat_items}
            </ul>
          </div>
          <div class="text-[10px] text-slate-400 border-t border-slate-800 pt-1.5">
            <strong>Plan:</strong> {html.escape(sc.actionable_intraday_plan)}
          </div>
        </div>
        """)
    all_pm_scenarios_html = "".join(pm_scenarios_html)

    # -------------------------------------------------------------------------
    # Render Intraday Table Rows with Deep Criteria Checklists
    # -------------------------------------------------------------------------
    intraday_rows = []
    for idx, opp in enumerate(top_intraday, 1):
        s = opp.setup
        c = opp.company
        p = opp.profile
        n = opp.news
        sb = opp.score_breakdown

        dir_cls = "bg-emerald-950/80 text-emerald-400 border border-emerald-700/60" if s.direction.value == "LONG" else "bg-rose-950/80 text-rose-400 border border-rose-700/60"
        tier_cls = "bg-emerald-950/80 text-emerald-300 border-emerald-700/50" if sb.candidate_tier == "STRONG" else (
            "bg-blue-950/80 text-blue-300 border-blue-700/50" if sb.candidate_tier == "WATCHLIST" else "bg-rose-950/80 text-rose-300 border-rose-700/50"
        )
        score_val = int(sb.total_score)
        score_bar_cls = "bg-emerald-500" if score_val >= 85 else ("bg-amber-500" if score_val >= 70 else "bg-slate-500")

        company_name = html.escape(c.name if c else s.symbol)
        sector_name = html.escape(c.sector if c else "General")
        ez = f"₹{s.entry_low:,.1f}–{s.entry_high:,.1f}"
        move_pct = s.expected_move_pct()

        # Criteria checklist pills summary
        passed_count = sum(1 for cr in sb.criteria if cr.status == "PASS")
        warn_count = sum(1 for cr in sb.criteria if cr.status == "WARN")
        fail_count = sum(1 for cr in sb.criteria if cr.status == "FAIL")

        criteria_pills = f"""
        <div class="flex items-center gap-1.5 flex-wrap">
          <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700/50">✓ {passed_count} PASS</span>
          {f'<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/80 text-amber-300 border border-amber-700/50">⚠ {warn_count} WARN</span>' if warn_count else ''}
          {f'<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950/80 text-rose-300 border border-rose-700/50">✗ {fail_count} FAIL</span>' if fail_count else ''}
        </div>
        """

        # Build detailed criteria audit table rows for drawer
        criteria_tbody_rows = []
        for cr in sb.criteria:
            st_cls = "text-emerald-400 bg-emerald-950/60 border-emerald-700/60" if cr.status == "PASS" else ("text-amber-400 bg-amber-950/60 border-amber-700/60" if cr.status == "WARN" else "text-rose-400 bg-rose-950/60 border-rose-700/60")
            icon = "✓" if cr.status == "PASS" else ("⚠" if cr.status == "WARN" else "✗")
            criteria_tbody_rows.append(f"""
            <tr class="border-b border-slate-800/80 text-xs hover:bg-slate-800/30">
              <td class="py-2 px-3 font-semibold text-slate-200">{html.escape(cr.name)}</td>
              <td class="py-2 px-3 text-slate-400 font-mono text-[11px]">{html.escape(cr.requirement)}</td>
              <td class="py-2 px-3 text-cyan-300 font-mono text-[11px] font-bold">{html.escape(cr.actual)}</td>
              <td class="py-2 px-3 text-center">
                <span class="px-2 py-0.5 rounded text-[10px] font-extrabold border {st_cls}">{icon} {cr.status}</span>
              </td>
              <td class="py-2 px-3 text-slate-300 text-[11px]">{html.escape(cr.detail)}</td>
            </tr>
            """)
        criteria_audit_html = "".join(criteria_tbody_rows)

        # Build thesis & execution summary
        news_title = html.escape(n.items[0].title if n and n.items else "No fresh breaking news; setup moving on technical & structural volume flows.")
        biz_summary = html.escape(c.business_summary[:220] + "..." if c and c.business_summary else "Core exchange constituent.")
        cat_disp = html.escape(n.primary_catalyst if n and n.primary_catalyst else "Technical Momentum Breakout")

        rejection_box = ""
        if sb.rejection_reasons:
            rejection_box = f"""
            <div class="bg-rose-950/60 border border-rose-800/60 p-2.5 rounded-lg text-rose-300 text-[11px] mt-2">
              <strong>Risk/Rejection Factors:</strong> {html.escape(', '.join(sb.rejection_reasons))}
            </div>
            """

        drawer_html = f"""
        <div class="space-y-4">
          <!-- Criteria Checklist Section -->
          <div class="bg-slate-900/80 rounded-xl p-4 border border-slate-800 shadow-inner">
            <div class="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
              <div class="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-1.5">
                <span>🔍</span> Intraday Criteria Checklist Audit (10-Point Institutional Verification)
              </div>
              <div class="text-[11px] text-slate-400 font-mono">
                Symbol: <span class="text-white font-bold">{html.escape(s.symbol)}</span> | Direction: <span class="font-bold text-{ 'emerald' if s.direction.value == 'LONG' else 'rose' }-400">{s.direction.value}</span> | Net R:R: <span class="text-emerald-400 font-bold">{sb.net_rr:.2f}</span>
              </div>
            </div>
            <div class="overflow-x-auto">
              <table class="w-full text-left border-collapse">
                <thead>
                  <tr class="text-[10px] uppercase font-bold text-slate-400 bg-slate-950/60 border-b border-slate-800">
                    <th class="py-2 px-3">Criterion Evaluated</th>
                    <th class="py-2 px-3">Institutional Requirement</th>
                    <th class="py-2 px-3">Stock Live Reading</th>
                    <th class="py-2 px-3 text-center">Audit Status</th>
                    <th class="py-2 px-3">Institutional Diagnostic Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {criteria_audit_html}
                </tbody>
              </table>
            </div>
            {rejection_box}
          </div>

          <!-- Execution Plan & Thesis Section -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-300">
            <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
              <div class="font-bold text-slate-200 mb-1.5 flex items-center gap-1.5">
                <span>🏢</span> Company & Context
              </div>
              <p class="text-slate-400 text-[11px] leading-relaxed mb-2">{biz_summary}</p>
              <div class="flex flex-wrap gap-2 text-[10px] font-mono text-cyan-400">
                <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">P/E: {f'{c.pe_ratio:.1f}x' if c and c.pe_ratio else 'N/A'}</span>
                <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">Rev YoY: {f'{c.revenue_growth_yoy*100:+.1f}%' if c and c.revenue_growth_yoy is not None else 'N/A'}</span>
              </div>
            </div>

            <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
              <div class="font-bold text-slate-200 mb-1.5 flex items-center gap-1.5">
                <span>⚡</span> Catalyst & News Intelligence
              </div>
              <p class="text-slate-200 font-semibold mb-1 text-[11px]">{cat_disp}</p>
              <p class="text-slate-400 text-[11px] leading-relaxed mb-2">{news_title}</p>
              <div class="text-[11px] text-slate-400">
                Sentiment: <span class="font-bold text-slate-200">{n.sentiment_label if n else 'NEUTRAL'}</span> |
                Event Risk: <span class="font-bold {'text-rose-400' if n and n.high_event_risk else 'text-emerald-400'}">{'HIGH' if n and n.high_event_risk else 'LOW'}</span>
              </div>
            </div>

            <div class="bg-slate-900/60 p-3.5 rounded-xl border border-indigo-900/40">
              <div class="font-bold text-indigo-300 mb-1.5 flex items-center gap-1.5">
                <span>🎯</span> Exact Execution Plan & Statutory Costs
              </div>
              <div class="space-y-1 font-mono text-[11px]">
                <div class="text-slate-200"><strong class="text-indigo-400">Trigger:</strong> 5m close beyond ₹{s.trigger:,.2f}</div>
                <div class="text-slate-300"><strong class="text-slate-400">Entry Zone:</strong> {ez}</div>
                <div class="text-rose-400 font-bold"><strong>Stop Loss:</strong> ₹{s.stop_loss:,.2f} (Strict)</div>
                <div class="text-emerald-400 font-semibold"><strong>Target 1:</strong> ₹{s.target1:,.2f} | <strong>Target 2:</strong> ₹{s.target2:,.2f} (+{move_pct:.1f}%)</div>
                <div class="text-cyan-300"><strong>Net R:R:</strong> {sb.net_rr:.2f} (Est Taxes & Slip: ₹{sb.total_costs_est:.1f})</div>
                <div class="text-amber-400 text-[10px] mt-1"><strong>Cancel if:</strong> below ₹{s.invalidate:,.2f}</div>
              </div>
            </div>
          </div>
        </div>
        """

        row_html = f"""
        <tr class="table-row intraday-row hover:bg-slate-800/50 transition border-b border-slate-800/80" data-direction="{s.direction.value}" data-score="{score_val}">
          <td class="px-4 py-3 text-slate-500 font-mono text-xs">{idx}</td>
          <td class="px-4 py-3">
            <div class="font-bold text-slate-100 flex items-center gap-1.5">
              <span>{html.escape(s.symbol)}</span>
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">{sector_name}</span>
            </div>
            <div class="text-[11px] text-slate-400 truncate max-w-xs">{company_name}</div>
          </td>
          <td class="px-4 py-3">
            <span class="px-2.5 py-1 rounded text-xs font-extrabold {dir_cls}">{s.direction.value}</span>
          </td>
          <td class="px-4 py-3 font-mono font-bold text-slate-100">₹{p.ltp:,.2f}</td>
          <td class="px-4 py-3 font-mono text-xs text-indigo-300">
            <div>₹{s.trigger:,.2f}</div>
            <div class="text-[10px] text-slate-400">{ez}</div>
          </td>
          <td class="px-4 py-3 font-mono text-xs text-rose-400 font-bold">₹{s.stop_loss:,.2f}</td>
          <td class="px-4 py-3 font-mono text-xs text-emerald-400 font-semibold">₹{s.target1:,.2f}</td>
          <td class="px-4 py-3 font-mono text-xs text-emerald-300 font-bold">
            ₹{s.target2:,.2f}
            <span class="text-[10px] text-emerald-400/80 block">+{move_pct:.1f}%</span>
          </td>
          <td class="px-4 py-3 font-mono text-xs font-bold text-emerald-400">
            {sb.net_rr:.2f}
            <span class="text-[10px] text-slate-400 block font-normal">Gross {s.rr() or 0:.2f}</span>
          </td>
          <td class="px-4 py-3">
            <div class="flex items-center gap-2">
              <span class="font-bold font-mono text-xs text-slate-100">{score_val}</span>
              <div class="w-12 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div class="{score_bar_cls} h-1.5 rounded-full" style="width: {min(100, score_val)}%"></div>
              </div>
            </div>
          </td>
          <td class="px-4 py-3">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold border {tier_cls}">{sb.candidate_tier}</span>
          </td>
          <td class="px-4 py-3">{criteria_pills}</td>
          <td class="px-4 py-3 text-right">
            <button onclick="toggleRow('intra-drawer-{idx}')" class="text-xs px-3 py-1.5 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 font-medium transition shadow-sm">
              Audit & Plan ▼
            </button>
          </td>
        </tr>
        <tr id="intra-drawer-{idx}" class="thesis-row hidden bg-slate-900/90 border-b border-indigo-950/60">
          <td colspan="13" class="px-6 py-4">
            {drawer_html}
          </td>
        </tr>
        """
        intraday_rows.append(row_html)

    all_intraday_rows_html = "".join(intraday_rows) if intraday_rows else """
    <tr><td colspan="13" class="text-center py-8 text-slate-400">No intraday setups met the minimum qualifying threshold for today's session.</td></tr>
    """

    # -------------------------------------------------------------------------
    # Render Long-Term Table Rows with Deep Fundamental Criteria Checklists
    # -------------------------------------------------------------------------
    longterm_rows = []
    for idx, lo in enumerate(top_longterm, 1):
        v_cls = "bg-emerald-950/80 text-emerald-400 border border-emerald-700/60" if lo.verdict == "STRONG BUY" else (
            "bg-blue-950/80 text-blue-400 border border-blue-700/60" if lo.verdict == "ACCUMULATE" else (
                "bg-amber-950/80 text-amber-400 border border-amber-700/60" if lo.verdict == "HOLD" else "bg-rose-950/80 text-rose-400 border border-rose-700/60"
            )
        )
        score_val = int(lo.score)
        score_bar_cls = "bg-emerald-500" if score_val >= 80 else ("bg-blue-500" if score_val >= 65 else ("bg-amber-500" if score_val >= 50 else "bg-slate-500"))

        company_name = html.escape(lo.name)
        sector_name = html.escape(lo.sector)
        mcap_str = f"₹{lo.market_cap_cr:,.0f} Cr" if lo.market_cap_cr else "N/A"
        pe_str = f"{lo.pe_ratio:.1f}x" if lo.pe_ratio else "N/A"
        roe_str = f"{lo.roe * 100:.1f}%" if lo.roe else "N/A"
        de_str = f"{lo.debt_to_equity / 100:.2f}x" if (lo.debt_to_equity and lo.debt_to_equity > 5) else (f"{lo.debt_to_equity:.2f}x" if lo.debt_to_equity is not None else "N/A")
        rev_str = f"{lo.revenue_growth_yoy * 100:+.1f}%" if lo.revenue_growth_yoy is not None else "N/A"

        # Criteria checklist pills summary
        passed_count = sum(1 for cr in lo.criteria if cr.status == "PASS")
        warn_count = sum(1 for cr in lo.criteria if cr.status == "WARN")
        fail_count = sum(1 for cr in lo.criteria if cr.status == "FAIL")

        criteria_pills = f"""
        <div class="flex items-center gap-1.5 flex-wrap">
          <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700/50">✓ {passed_count} PASS</span>
          {f'<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/80 text-amber-300 border border-amber-700/50">⚠ {warn_count} WARN</span>' if warn_count else ''}
          {f'<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950/80 text-rose-300 border border-rose-700/50">✗ {fail_count} FAIL</span>' if fail_count else ''}
        </div>
        """

        # Build detailed criteria audit table rows for long-term drawer
        lt_criteria_rows = []
        for cr in lo.criteria:
            st_cls = "text-emerald-400 bg-emerald-950/60 border-emerald-700/60" if cr.status == "PASS" else ("text-amber-400 bg-amber-950/60 border-amber-700/60" if cr.status == "WARN" else "text-rose-400 bg-rose-950/60 border-rose-700/60")
            icon = "✓" if cr.status == "PASS" else ("⚠" if cr.status == "WARN" else "✗")
            lt_criteria_rows.append(f"""
            <tr class="border-b border-slate-800/80 text-xs hover:bg-slate-800/30">
              <td class="py-2 px-3 font-semibold text-slate-200">{html.escape(cr.name)}</td>
              <td class="py-2 px-3 text-slate-400 font-mono text-[11px]">{html.escape(cr.requirement)}</td>
              <td class="py-2 px-3 text-cyan-300 font-mono text-[11px] font-bold">{html.escape(cr.actual)}</td>
              <td class="py-2 px-3 text-center">
                <span class="px-2 py-0.5 rounded text-[10px] font-extrabold border {st_cls}">{icon} {cr.status}</span>
              </td>
              <td class="py-2 px-3 text-slate-300 text-[11px]">{html.escape(cr.detail)}</td>
            </tr>
            """)
        lt_criteria_audit_html = "".join(lt_criteria_rows)

        strengths_html = "".join(f"<li class='text-emerald-300'>✓ {html.escape(s)}</li>" for s in lo.strengths) or "<li class='text-slate-400'>Core sector constituent</li>"
        risks_html = "".join(f"<li class='text-rose-300'>⚠ {html.escape(r)}</li>" for r in lo.risks) or "<li class='text-slate-400'>Standard market beta & macro risks</li>"

        # Valuation Scenarios
        base_t = lo.valuation_scenarios.get("base", lo.ltp * 1.15)
        bull_t = lo.valuation_scenarios.get("bull", lo.ltp * 1.30)
        bear_t = lo.valuation_scenarios.get("bear", lo.ltp * 0.85)

        lt_drawer_html = f"""
        <div class="space-y-4">
          <!-- Long-Term Criteria Checklist Section -->
          <div class="bg-slate-900/80 rounded-xl p-4 border border-slate-800 shadow-inner">
            <div class="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
              <div class="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <span>🏛️</span> Long-Term Fundamental & Structural Criteria Audit (10-Point Checklist)
              </div>
              <div class="text-[11px] text-slate-400 font-mono">
                Symbol: <span class="text-white font-bold">{html.escape(lo.symbol)}</span> | Rating: <span class="font-bold text-emerald-400">{lo.verdict}</span> | Score: <span class="text-white font-bold">{score_val}/100</span>
              </div>
            </div>
            <div class="overflow-x-auto">
              <table class="w-full text-left border-collapse">
                <thead>
                  <tr class="text-[10px] uppercase font-bold text-slate-400 bg-slate-950/60 border-b border-slate-800">
                    <th class="py-2 px-3">Fundamental / Structural Pillar</th>
                    <th class="py-2 px-3">Institutional Requirement</th>
                    <th class="py-2 px-3">Company Actual Metric</th>
                    <th class="py-2 px-3 text-center">Audit Status</th>
                    <th class="py-2 px-3">Evaluation Diagnostic</th>
                  </tr>
                </thead>
                <tbody>
                  {lt_criteria_audit_html}
                </tbody>
              </table>
            </div>
          </div>

          <!-- Thesis and Fundamental Details -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-300">
            <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
              <div class="font-bold text-slate-200 mb-1.5 flex items-center gap-1.5">
                <span>📖</span> Multi-Year Investment Thesis
              </div>
              <p class="text-slate-300 text-[11px] leading-relaxed mb-2">{html.escape(lo.thesis)}</p>
              <div class="text-[11px] text-cyan-400 font-mono border-t border-slate-800 pt-1.5">
                Cash Conversion: <span class="text-slate-200 font-semibold">{html.escape(lo.cash_conversion_status)}</span>
              </div>
            </div>

            <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
              <div class="font-bold text-emerald-300 mb-1.5 flex items-center gap-1.5">
                <span>🎯</span> Intrinsic Valuation Scenarios (12M Targets)
              </div>
              <div class="space-y-1.5 text-[11px] font-mono">
                <div class="text-emerald-400 font-bold">Bull Target : ₹{bull_t:,.1f} (+{((bull_t-lo.ltp)/lo.ltp*100):.1f}%)</div>
                <div class="text-cyan-300 font-semibold">Base Target : ₹{base_t:,.1f} (+{((base_t-lo.ltp)/lo.ltp*100):.1f}%)</div>
                <div class="text-rose-400 font-semibold">Bear Target : ₹{bear_t:,.1f} ({((bear_t-lo.ltp)/lo.ltp*100):.1f}%)</div>
              </div>
              <div class="text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800">
                Valuation Verdict: <span class="font-bold text-slate-200">{html.escape(lo.valuation_verdict)}</span>
              </div>
            </div>

            <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
              <div class="font-bold text-slate-200 mb-1.5 flex items-center gap-1.5">
                <span>🛡️</span> Strengths & Risk Factors
              </div>
              <ul class="space-y-1 text-[11px] mb-2">
                {strengths_html}
                {risks_html}
              </ul>
            </div>
          </div>
        </div>
        """

        lt_row_html = f"""
        <tr class="table-row longterm-row hover:bg-slate-800/50 transition border-b border-slate-800/80" data-verdict="{lo.verdict}" data-score="{score_val}">
          <td class="px-4 py-3 text-slate-500 font-mono text-xs">{idx}</td>
          <td class="px-4 py-3">
            <div class="font-bold text-slate-100 flex items-center gap-1.5">
              <span>{html.escape(lo.symbol)}</span>
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">{sector_name}</span>
            </div>
            <div class="text-[11px] text-slate-400 truncate max-w-xs">{company_name}</div>
          </td>
          <td class="px-4 py-3">
            <span class="px-2.5 py-1 rounded text-xs font-extrabold {v_cls}">{lo.verdict}</span>
          </td>
          <td class="px-4 py-3 font-mono font-bold text-slate-100">₹{lo.ltp:,.2f}</td>
          <td class="px-4 py-3 font-mono text-xs text-slate-300">{mcap_str}</td>
          <td class="px-4 py-3 font-mono text-xs text-cyan-300 font-semibold">{pe_str}</td>
          <td class="px-4 py-3 font-mono text-xs text-emerald-400 font-semibold">{roe_str}</td>
          <td class="px-4 py-3 font-mono text-xs text-slate-300">{de_str}</td>
          <td class="px-4 py-3 font-mono text-xs text-emerald-300 font-semibold">{rev_str}</td>
          <td class="px-4 py-3">
            <div class="flex items-center gap-2">
              <span class="font-bold font-mono text-xs text-slate-100">{score_val}</span>
              <div class="w-12 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div class="{score_bar_cls} h-1.5 rounded-full" style="width: {min(100, score_val)}%"></div>
              </div>
            </div>
          </td>
          <td class="px-4 py-3">{criteria_pills}</td>
          <td class="px-4 py-3 text-right">
            <button onclick="toggleRow('lt-drawer-{idx}')" class="text-xs px-3 py-1.5 rounded-lg bg-cyan-600/30 hover:bg-cyan-600/50 text-cyan-300 border border-cyan-500/40 font-medium transition shadow-sm">
              Audit & Thesis ▼
            </button>
          </td>
        </tr>
        <tr id="lt-drawer-{idx}" class="thesis-row hidden bg-slate-900/90 border-b border-cyan-950/60">
          <td colspan="12" class="px-6 py-4">
            {lt_drawer_html}
          </td>
        </tr>
        """
        longterm_rows.append(lt_row_html)

    all_longterm_rows_html = "".join(longterm_rows) if longterm_rows else """
    <tr><td colspan="12" class="text-center py-8 text-slate-400">No long-term candidates qualified in this scan universe.</td></tr>
    """

    # -------------------------------------------------------------------------
    # Render Backtest Section (if available)
    # -------------------------------------------------------------------------
    backtest_html = ""
    if backtest_report and hasattr(backtest_report, "total_trades") and backtest_report.total_trades > 0:
        bt = backtest_report
        backtest_html = f"""
        <section id="backtest-section" class="glass-card rounded-2xl p-5 shadow-xl border-slate-700/70 scroll-mt-6 space-y-4">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 rounded text-[10px] font-extrabold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 uppercase">Point-in-Time Simulation</span>
                <h2 class="text-base font-bold text-white flex items-center gap-2">
                  <span>📊</span> Strategy Backtest & Validation ({html.escape(bt.strategy_name)})
                </h2>
              </div>
              <p class="text-xs text-slate-400 mt-0.5">Historical simulation across {len(bt.symbols_tested)} symbols from {bt.start_date} to {bt.end_date} with exact Indian statutory taxes & slippage.</p>
            </div>
            <span class="text-[10px] px-2.5 py-1 rounded bg-slate-900 text-slate-400 border border-slate-800 font-mono">BACKTESTED_SIMULATION</span>
          </div>

          <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-center">
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[10px] text-slate-400 uppercase font-medium">Win Rate</div>
              <div class="text-lg font-bold text-emerald-400 mt-1">{bt.win_rate_pct:.1f}%</div>
              <div class="text-[10px] text-slate-500 font-mono">{bt.winning_trades}W / {bt.losing_trades}L</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[10px] text-slate-400 uppercase font-medium">Profit Factor</div>
              <div class="text-lg font-bold text-indigo-400 mt-1">{bt.profit_factor:.2f}</div>
              <div class="text-[10px] text-slate-500 font-mono">Gross W/L Ratio</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[10px] text-slate-400 uppercase font-medium">Net P&L</div>
              <div class="text-lg font-bold text-emerald-300 mt-1">₹{bt.total_net_pnl:,.0f}</div>
              <div class="text-[10px] text-rose-400/80 font-mono">-₹{bt.total_statutory_costs:,.0f} taxes</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[10px] text-slate-400 uppercase font-medium">Max Drawdown</div>
              <div class="text-lg font-bold text-amber-400 mt-1">{bt.max_drawdown_pct:.1f}%</div>
              <div class="text-[10px] text-slate-500 font-mono">Peak-to-trough</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[10px] text-slate-400 uppercase font-medium">Sharpe Ratio</div>
              <div class="text-lg font-bold text-cyan-400 mt-1">{bt.sharpe_ratio:.2f}</div>
              <div class="text-[10px] text-slate-500 font-mono">Risk-adjusted return</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[10px] text-slate-400 uppercase font-medium">Expectancy</div>
              <div class="text-lg font-bold text-white mt-1">₹{bt.expectancy_per_trade:,.1f}</div>
              <div class="text-[10px] text-slate-500 font-mono">Net per trade</div>
            </div>
          </div>
        </section>
        """

    # -------------------------------------------------------------------------
    # Assemble Full Standalone HTML Document
    # -------------------------------------------------------------------------
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Global-Market-Aware Trading Intelligence Platform — Institutional Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{ background-color: #0b1120; color: #f8fafc; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    .glass-card {{ background: rgba(15, 23, 42, 0.78); backdrop-filter: blur(12px); border: 1px solid rgba(51, 65, 85, 0.65); }}
  </style>
</head>
<body class="min-h-screen p-4 md:p-8">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Header Section -->
    <header class="glass-card rounded-2xl p-6 shadow-xl flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-slate-700/70">
      <div>
        <div class="flex items-center gap-2.5 mb-1">
          <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span class="text-xs font-semibold tracking-wider text-emerald-400 uppercase">Global-Market-Aware Trading Intelligence Platform</span>
        </div>
        <h1 class="text-2xl md:text-3xl font-extrabold text-white tracking-tight">Institutional Research, Scanners & Trade Engine</h1>
        <p class="text-xs md:text-sm text-slate-400 mt-1">Generated: <span class="font-mono text-slate-300">{now_str}</span> | Universe Scanned: <span class="font-bold text-slate-200">{total_scanned:,} Equities</span></p>
      </div>
      <div class="flex items-center gap-3">
        <div class="bg-slate-800/90 border border-slate-700 rounded-xl px-4 py-2.5 text-center">
          <div class="text-[11px] text-slate-400 uppercase tracking-wider font-medium">Market Regime</div>
          <div class="text-base font-extrabold text-{regime_color}-400 flex items-center justify-center gap-1.5 mt-0.5">
            <span>●</span> {macro.regime}
          </div>
        </div>
        <div class="bg-slate-800/90 border border-slate-700 rounded-xl px-4 py-2.5 text-center">
          <div class="text-[11px] text-slate-400 uppercase tracking-wider font-medium">Intraday Setups</div>
          <div class="text-base font-extrabold text-indigo-400 mt-0.5">{len(top_intraday)} Setups</div>
        </div>
        <div class="bg-slate-800/90 border border-slate-700 rounded-xl px-4 py-2.5 text-center">
          <div class="text-[11px] text-slate-400 uppercase tracking-wider font-medium">Long-Term Picks</div>
          <div class="text-base font-extrabold text-cyan-400 mt-0.5">{len(top_longterm)} Stocks</div>
        </div>
      </div>
    </header>

    <!-- Global Macro Grid -->
    <section class="glass-card rounded-2xl p-5 shadow-lg border-slate-700/70 space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
          <span>🌍</span> Global Macro, Commodities & Domestic Market Context
        </h2>
        <span class="text-xs text-slate-400 font-mono">Cross-Border Correlation Engine</span>
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
        {nifty_card}
        {bank_card}
        {vix_card}
        <div class="bg-slate-800/80 border border-slate-700/60 rounded-lg p-2.5 flex flex-col justify-between">
          <div class="flex items-center justify-between text-[10px] text-slate-400">
            <span>FII Net</span>
            <span class="text-[9px] px-1 rounded bg-slate-900 text-slate-500">PROVISIONAL</span>
          </div>
          <div class="text-sm font-bold {fii_cls} mt-1">{fii_str}</div>
          <div class="text-[10px] text-slate-400">Cash Flow</div>
        </div>
        <div class="bg-slate-800/80 border border-slate-700/60 rounded-lg p-2.5 flex flex-col justify-between">
          <div class="flex items-center justify-between text-[10px] text-slate-400">
            <span>DII Net</span>
            <span class="text-[9px] px-1 rounded bg-slate-900 text-slate-500">PROVISIONAL</span>
          </div>
          <div class="text-sm font-bold {dii_cls} mt-1">{dii_str}</div>
          <div class="text-[10px] text-slate-400">Cash Flow</div>
        </div>
        <div class="bg-slate-800/80 border border-slate-700/60 rounded-lg p-2.5 flex flex-col justify-between">
          <div class="flex items-center justify-between text-[10px] text-slate-400">
            <span>A/D Ratio</span>
            <span class="text-[9px] px-1 rounded bg-slate-900 text-slate-500">NSE EQ</span>
          </div>
          <div class="text-sm font-bold text-slate-200 mt-1">{macro.nse.advance_decline or 'N/A'}</div>
          <div class="text-[10px] text-slate-400">Breadth</div>
        </div>
        {us_cards}
        {fut_cards}
        {eu_cards}
        {asia_cards}
        {comm_cards}
        {curr_cards}
      </div>
    </section>

    <!-- Indian Pre-Market & GIFT Nifty Outlook -->
    <section class="glass-card rounded-2xl p-5 shadow-lg border-slate-700/70 space-y-3">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div class="flex items-center gap-2">
          <span class="px-2 py-0.5 rounded text-[10px] font-extrabold bg-indigo-500/20 text-indigo-400 border border-indigo-500/40 uppercase">Pre-Market Briefing</span>
          <h2 class="text-sm font-bold uppercase tracking-wider text-white">GIFT Nifty Implied Open & Scenarios</h2>
        </div>
        <div class="text-xs font-mono text-slate-300">
          GIFT Nifty Implied: <strong class="text-white">₹{pm.gift_nifty_implied:,.1f}</strong> ({pm.gift_nifty_change_pct:+.2f}%) | Indicated Gap: <strong class="{ 'text-emerald-400' if (pm.gap_points or 0)>=0 else 'text-rose-400' }">{pm.gap_points:+.1f} pts</strong> ({pm.gap_type})
        </div>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {all_pm_scenarios_html}
      </div>
    </section>

    <!-- Navigation Quick-Switch -->
    <div class="flex flex-wrap items-center gap-2.5 bg-slate-900/80 p-2 rounded-xl border border-slate-800">
      <a href="#intraday-section" class="px-4 py-2 rounded-lg text-xs font-bold bg-indigo-600 text-white flex items-center gap-2 shadow transition hover:bg-indigo-500">
        <span>⚡</span> Table 1: Intraday Trading Setups ({len(top_intraday)})
      </a>
      <a href="#longterm-section" class="px-4 py-2 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 flex items-center gap-2 transition">
        <span>🏛️</span> Table 2: Long-Term Investment Picks ({len(top_longterm)})
      </a>
      { '<a href="#backtest-section" class="px-4 py-2 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 flex items-center gap-2 transition"><span>📊</span> Strategy Backtest & Validation</a>' if backtest_html else '' }
      <a href="#criteria-reference" class="px-4 py-2 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 flex items-center gap-2 transition">
        <span>📋</span> Methodology & Statutory Cost Matrix
      </a>
    </div>

    <!-- ================================================================= -->
    <!-- TABLE 1: INTRADAY TRADING SETUPS & ANALYSIS                      -->
    <!-- ================================================================= -->
    <section id="intraday-section" class="glass-card rounded-2xl shadow-xl overflow-hidden border-slate-700/70 scroll-mt-6">
      <div class="p-5 border-b border-slate-800 flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-gradient-to-r from-indigo-950/40 to-slate-900/60">
        <div>
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded text-[10px] font-extrabold bg-indigo-500/20 text-indigo-400 border border-indigo-500/40 uppercase">High Priority Intraday Desk</span>
            <h2 class="text-lg font-bold text-white flex items-center gap-2">
              <span>⚡</span> Table 1: Intraday Trading Setups & Statutory Cost Audit
            </h2>
          </div>
          <p class="text-xs text-slate-400 mt-1">
            Evaluates 10 institutional criteria including Net Reward-to-Risk strictly after statutory charges (STT, GST, Exchange fees, Stamp Duty, Slippage).
          </p>
        </div>

        <!-- Filter Controls -->
        <div class="flex flex-wrap items-center gap-2">
          <input type="text" id="intradaySearch" onkeyup="filterIntraday()" placeholder="Search symbol, setup..."
                 class="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-48">
          <button onclick="setIntradayFilter('ALL')" class="intra-filter-btn text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-medium transition">All</button>
          <button onclick="setIntradayFilter('LONG')" class="intra-filter-btn text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 font-medium transition">Long Only</button>
          <button onclick="setIntradayFilter('SHORT')" class="intra-filter-btn text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 font-medium transition">Short Only</button>
          <button onclick="setIntradayFilter('TOP')" class="intra-filter-btn text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 font-medium transition">Score ≥ 80</button>
        </div>
      </div>

      <!-- Intraday Table -->
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-slate-900/95 text-[11px] font-bold uppercase tracking-wider text-slate-400 border-b border-slate-800">
              <th class="px-4 py-3">#</th>
              <th class="px-4 py-3">Symbol / Company</th>
              <th class="px-4 py-3">Dir</th>
              <th class="px-4 py-3">LTP</th>
              <th class="px-4 py-3">Trigger / Zone</th>
              <th class="px-4 py-3">Stop Loss</th>
              <th class="px-4 py-3">Target 1</th>
              <th class="px-4 py-3">Target 2</th>
              <th class="px-4 py-3">Net R:R</th>
              <th class="px-4 py-3">Score</th>
              <th class="px-4 py-3">Tier</th>
              <th class="px-4 py-3">Criteria Checks</th>
              <th class="px-4 py-3 text-right">Audit & Plan</th>
            </tr>
          </thead>
          <tbody id="intradayTableBody">
            {all_intraday_rows_html}
          </tbody>
        </table>
      </div>

      <div class="p-4 bg-slate-900/80 border-t border-slate-800/80 text-[11px] text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-2">
        <span>Displaying institutional intraday setups. Net R:R includes all exchange & statutory charges.</span>
        <span>Click <strong>Audit & Plan ▼</strong> on any row to inspect all 10 verified intraday criteria, execution triggers, and cost details.</span>
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- TABLE 2: LONG-TERM INVESTMENT PICKS & FUNDAMENTAL ANALYSIS       -->
    <!-- ================================================================= -->
    <section id="longterm-section" class="glass-card rounded-2xl shadow-xl overflow-hidden border-slate-700/70 scroll-mt-6">
      <div class="p-5 border-b border-slate-800 flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-gradient-to-r from-cyan-950/40 to-slate-900/60">
        <div>
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded text-[10px] font-extrabold bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 uppercase">Multi-Quarter Horizon</span>
            <h2 class="text-lg font-bold text-white flex items-center gap-2">
              <span>🏛️</span> Table 2: Long-Term Investment Picks & Cash Flow Quality
            </h2>
          </div>
          <p class="text-xs text-slate-400 mt-1">
            Evaluates 10 institutional long-term criteria: YoY Revenue Growth, Earnings Growth, OPM, ROE, Solvency (D/E), Structural 200 EMA, 52W Proximity, Valuation Multiple, ADX Trend, and Cash Flow Quality.
          </p>
        </div>

        <!-- Filter Controls -->
        <div class="flex flex-wrap items-center gap-2">
          <input type="text" id="longtermSearch" onkeyup="filterLongterm()" placeholder="Search long-term symbol..."
                 class="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-48">
          <button onclick="setLongtermFilter('ALL')" class="lt-filter-btn text-xs px-3 py-1.5 rounded-lg bg-cyan-600 text-white font-medium transition">All</button>
          <button onclick="setLongtermFilter('STRONG BUY')" class="lt-filter-btn text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 font-medium transition">Strong Buy</button>
          <button onclick="setLongtermFilter('ACCUMULATE')" class="lt-filter-btn text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 font-medium transition">Accumulate</button>
          <button onclick="setLongtermFilter('TOP')" class="lt-filter-btn text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 font-medium transition">Score ≥ 75</button>
        </div>
      </div>

      <!-- Long-Term Table -->
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-slate-900/95 text-[11px] font-bold uppercase tracking-wider text-slate-400 border-b border-slate-800">
              <th class="px-4 py-3">#</th>
              <th class="px-4 py-3">Symbol / Company</th>
              <th class="px-4 py-3">Rating</th>
              <th class="px-4 py-3">LTP</th>
              <th class="px-4 py-3">Market Cap</th>
              <th class="px-4 py-3">P/E Ratio</th>
              <th class="px-4 py-3">ROE (%)</th>
              <th class="px-4 py-3">Debt/Equity</th>
              <th class="px-4 py-3">YoY Rev Growth</th>
              <th class="px-4 py-3">Score</th>
              <th class="px-4 py-3">Criteria Checks</th>
              <th class="px-4 py-3 text-right">Audit & Thesis</th>
            </tr>
          </thead>
          <tbody id="longtermTableBody">
            {all_longterm_rows_html}
          </tbody>
        </table>
      </div>

      <div class="p-4 bg-slate-900/80 border-t border-slate-800/80 text-[11px] text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-2">
        <span>Long-term structural holdings intended for 6-24 month capital appreciation and compound growth.</span>
        <span>Click <strong>Audit & Thesis ▼</strong> on any row to inspect all 10 verified fundamental criteria, cash flow quality, and intrinsic target scenarios.</span>
      </div>
    </section>

    <!-- Optional Backtest Section -->
    {backtest_html}

    <!-- ================================================================= -->
    <!-- SECTION 4: MASTER CRITERIA & STATUTORY COST MATRIX               -->
    <!-- ================================================================= -->
    <section id="criteria-reference" class="glass-card rounded-2xl p-6 shadow-xl border-slate-700/70 scroll-mt-6 space-y-4">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <h2 class="text-base font-bold text-white flex items-center gap-2">
          <span>📋</span> Master Criteria Checklist & Statutory Cost Rulebook
        </h2>
        <span class="text-xs text-slate-400">Institutional Governance & Regulatory Parameters</span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Intraday Criteria Guide -->
        <div class="bg-slate-900/70 rounded-xl p-4 border border-indigo-900/40 space-y-3">
          <div class="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-1.5 border-b border-slate-800 pb-2">
            <span>⚡</span> Intraday Criteria & Statutory Cost Modeling
          </div>
          <div class="space-y-2 text-xs text-slate-300">
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">1. Market Alignment:</span>
              <span class="text-slate-400">Nifty direction aligns with trade; India VIX strictly &lt; 22 (stable volatility regime).</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">2. Technical Price Action:</span>
              <span class="text-slate-400">Trend alignment + breakout triggers (Prior-Day High/Low or VWAP Momentum).</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">3. Multi-Timeframe (MTF):</span>
              <span class="text-slate-400">Agreement score &ge; 6/10 across Daily, Weekly, Hourly, 15m, and 5m charts.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">4. Volume Confirmation (RVOL):</span>
              <span class="text-slate-400">Relative Volume &ge; 1.0x (optimal &ge; 1.3x) relative to 20-period moving average.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">5. RSI Momentum Corridor:</span>
              <span class="text-slate-400">Longs: 50–68; Shorts: 32–50 (avoids exhausted overbought &gt;70 or oversold &lt;30).</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">6. VWAP Anchor:</span>
              <span class="text-slate-400">Price must trade on correct side of intraday VWAP for institutional confirmation.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">7. Catalyst & News Flow:</span>
              <span class="text-slate-400">Fresh earnings/order wins without imminent high event risks (e.g. results today).</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">8. Sector Strength:</span>
              <span class="text-slate-400">Sector index demonstrating positive relative strength or outperforming broader Nifty.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">9. Net Reward-to-Risk (R:R):</span>
              <span class="text-slate-400">Net R:R &ge; 1.50 calculated strictly AFTER STT (0.025%), GST (18%), NSE txn fees, and 5 bps slippage.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-indigo-400 font-bold">10. Session Liquidity Gate:</span>
              <span class="text-slate-400">Eligible EQ series with 20-day traded turnover to prevent slippage and liquidity traps.</span>
            </div>
          </div>
        </div>

        <!-- Long-Term Criteria Guide -->
        <div class="bg-slate-900/70 rounded-xl p-4 border border-cyan-900/40 space-y-3">
          <div class="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5 border-b border-slate-800 pb-2">
            <span>🏛️</span> Long-Term Criteria & Cash Flow Verification
          </div>
          <div class="space-y-2 text-xs text-slate-300">
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">1. Revenue Growth YoY:</span>
              <span class="text-slate-400">Top-line business expansion &ge; 12.0% YoY (compound scaling power).</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">2. Earnings (PAT) Growth YoY:</span>
              <span class="text-slate-400">Net profit growth &ge; 15.0% YoY demonstrating operating leverage and bottom-line health.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">3. Operating Profit Margin:</span>
              <span class="text-slate-400">Operating margins (OPM) &ge; 15.0% indicating economic moat and pricing power.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">4. Return on Equity (ROE):</span>
              <span class="text-slate-400">ROE &ge; 15.0% indicating highly efficient shareholder equity reinvestment.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">5. Solvency & Leverage:</span>
              <span class="text-slate-400">Debt-to-Equity &le; 1.0x (ideal &lt; 0.5x); banks evaluated by capital adequacy & NIM.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">6. Structural Trend (200 EMA):</span>
              <span class="text-slate-400">Price structurally above 200-day EMA confirms multi-month institutional bull market.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">7. 52-Week High Proximity:</span>
              <span class="text-slate-400">Within 25% of 52W High; shows leadership momentum and eliminates lagging value traps.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">8. Valuation Multiple:</span>
              <span class="text-slate-400">P/E &le; 45.0x or attractive growth PEG ratio to ensure margin of safety on entry.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">9. Trend Strength (ADX):</span>
              <span class="text-slate-400">Daily ADX &ge; 20.0 confirms strong, non-random multi-month directional trend.</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-cyan-400 font-bold">10. Cash Flow Conversion:</span>
              <span class="text-slate-400">Positive Operating Cash Flow and Free Cash Flow (FCF) yield confirming earnings authenticity.</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Footer Notice -->
    <footer class="p-4 text-center text-xs text-slate-500 space-y-1">
      <p>⚖️ Institutional Governance Notice: Free delayed feeds (~15 min lag). Execution requires live tick volume & VWAP validation.</p>
      <p>Strict capital-at-risk parameters must be adhered to at all times. Never risk more than designated per-trade threshold.</p>
    </footer>

  </div>

  <script>
    function toggleRow(id) {{
      const el = document.getElementById(id);
      if (el) {{
        el.classList.toggle('hidden');
      }}
    }}

    // Intraday Filtering
    let currentIntradayFilter = 'ALL';
    function setIntradayFilter(type) {{
      currentIntradayFilter = type;
      document.querySelectorAll('.intra-filter-btn').forEach(btn => {{
        btn.classList.remove('bg-indigo-600', 'text-white');
        btn.classList.add('bg-slate-800', 'text-slate-300');
      }});
      event.target.classList.remove('bg-slate-800', 'text-slate-300');
      event.target.classList.add('bg-indigo-600', 'text-white');
      filterIntraday();
    }}

    function filterIntraday() {{
      const query = document.getElementById('intradaySearch').value.toLowerCase();
      const rows = document.querySelectorAll('.intraday-row');

      rows.forEach(row => {{
        const text = row.innerText.toLowerCase();
        const dir = row.getAttribute('data-direction');
        const score = parseInt(row.getAttribute('data-score') || '0');

        let matchesFilter = true;
        if (currentIntradayFilter === 'LONG' && dir !== 'LONG') matchesFilter = false;
        if (currentIntradayFilter === 'SHORT' && dir !== 'SHORT') matchesFilter = false;
        if (currentIntradayFilter === 'TOP' && score < 80) matchesFilter = false;

        const matchesQuery = text.includes(query);

        if (matchesFilter && matchesQuery) {{
          row.style.display = '';
        }} else {{
          row.style.display = 'none';
          const next = row.nextElementSibling;
          if (next && next.classList.contains('thesis-row')) {{
            next.classList.add('hidden');
          }}
        }}
      }});
    }}

    // Long-Term Filtering
    let currentLongtermFilter = 'ALL';
    function setLongtermFilter(type) {{
      currentLongtermFilter = type;
      document.querySelectorAll('.lt-filter-btn').forEach(btn => {{
        btn.classList.remove('bg-cyan-600', 'text-white');
        btn.classList.add('bg-slate-800', 'text-slate-300');
      }});
      event.target.classList.remove('bg-slate-800', 'text-slate-300');
      event.target.classList.add('bg-cyan-600', 'text-white');
      filterLongterm();
    }}

    function filterLongterm() {{
      const query = document.getElementById('longtermSearch').value.toLowerCase();
      const rows = document.querySelectorAll('.longterm-row');

      rows.forEach(row => {{
        const text = row.innerText.toLowerCase();
        const verdict = row.getAttribute('data-verdict');
        const score = parseInt(row.getAttribute('data-score') || '0');

        let matchesFilter = true;
        if (currentLongtermFilter === 'STRONG BUY' && verdict !== 'STRONG BUY') matchesFilter = false;
        if (currentLongtermFilter === 'ACCUMULATE' && verdict !== 'ACCUMULATE') matchesFilter = false;
        if (currentLongtermFilter === 'TOP' && score < 75) matchesFilter = false;

        const matchesQuery = text.includes(query);

        if (matchesFilter && matchesQuery) {{
          row.style.display = '';
        }} else {{
          row.style.display = 'none';
          const next = row.nextElementSibling;
          if (next && next.classList.contains('thesis-row')) {{
            next.classList.add('hidden');
          }}
        }}
      }});
    }}
  </script>
</body>
</html>
"""

    if output_path:
        p = Path(output_path)
        p.write_text(html_content, encoding="utf-8")

    return html_content
