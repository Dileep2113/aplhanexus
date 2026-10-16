#!/usr/bin/env python
"""CLI for the Autonomous Indian Stock Market Analyst.

Executes the complete institutional research cycle:
  * SCAN: Comprehensive universe scan (Macro + Technicals + Fundamentals + News + Scoring)
  * MONITOR: Live monitoring loop for scored opportunities
  * EOD: End-of-day journal analysis and learning system review

Usage:
  python run_analyst.py scan [--universe nifty500|nifty100|nifty50] [--limit N] [--min-score 60] [--save-setups]
  python run_analyst.py eod
  python run_analyst.py monitor [--symbols RELIANCE,TCS]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 on Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import yaml

from analyst.backtest import run_strategy_backtest
from analyst.company import fetch_company_profile
from analyst.global_macro import fetch_global_snapshot
from analyst.journal import log_opportunities, summarize_journal
from analyst.longterm import evaluate_longterm_candidate, LongTermOpportunity
from analyst.news import fetch_news
from analyst.premarket import generate_premarket_report
from analyst.report import ScoredOpportunity, generate_full_report, generate_html_report
from analyst.scoring import score_setup
from analyst.setup_builder import build_candidate_setups
from analyst.technicals import analyse_symbol
from analyst.universe import apply_liquidity_filter, fetch_universe
from intraday.providers import NseContextProvider
from intraday.session import SessionClock
from analyst.predictions import PredictionRegistry, PredictionRecord
from analyst.evening_review import EveningReviewer
from analyst.stock_research import StockResearcher
from analyst.learning_engine import LearningEngine
from analyst.storage import StorageManager
from analyst.web_app import run_web_server

ROOT = Path(__file__).resolve().parent



def cmd_scan(args) -> int:
    print("\n" + "═" * 78)
    print("🚀 LAUNCHING AUTONOMOUS INSTITUTIONAL RESEARCH & OPPORTUNITY SCAN")
    print("═" * 78)

    # 1. Global Macro & NSE Context
    print("\n[1/6] Fetching Global Macro, Commodities, Currencies & NSE Context...", flush=True)
    nse_prov = NseContextProvider(SessionClock())
    macro = fetch_global_snapshot(nse_provider=nse_prov)

    # 2. Universe Ingestion
    print(f"\n[2/6] Loading eligible NSE universe ({args.universe}{', excluding Nifty 50' if args.exclude_nifty50 else ''})...", flush=True)
    raw_universe = fetch_universe(tier=args.universe, exclude_nifty50=args.exclude_nifty50)
    if args.limit > 0:
        raw_universe = raw_universe[:args.limit]
        print(f"  (Limiting universe to first {args.limit} symbols for this run)")

    # 3. Liquidity Gate
    print("\n[3/6] Applying 20-day liquidity and volume filtration...", flush=True)
    universe = apply_liquidity_filter(raw_universe, sample_size=args.limit)

    # 4. Technical Analysis & Candidate Setup Generation
    print(f"\n[4/6] Analyzing multi-timeframe price action & indicators across {len(universe)} symbols...", flush=True)
    candidates: list[tuple] = []

    if len(universe) > 30 or args.fast:
        from analyst.technicals import batch_analyse_symbols
        syms_to_scan = [e.yahoo_symbol for e in universe]
        profile_map = batch_analyse_symbols(syms_to_scan, chunk_size=100)
        for entry in universe:
            profile = profile_map.get(entry.yahoo_symbol)
            if not profile or profile.ltp <= 0:
                continue
            candidate_setups = build_candidate_setups(profile)
            if candidate_setups:
                candidates.append((entry, profile, candidate_setups))
    else:
        for idx, entry in enumerate(universe, 1):
            if idx % 20 == 0 or idx == len(universe):
                print(f"  ... processed {idx}/{len(universe)} symbols", flush=True)

            profile = analyse_symbol(entry.yahoo_symbol, fast_mode=args.fast)
            if not profile or profile.ltp <= 0:
                continue

            candidate_setups = build_candidate_setups(profile)
            if candidate_setups:
                candidates.append((entry, profile, candidate_setups))

    print(f"  Identified candidate chart setups across {len(candidates)} symbols.")

    # 5. Deep Fundamental, News, and 100-Point Scoring
    print(f"\n[5/6] Performing fundamental valuation, news catalyst check & 100-pt scoring...", flush=True)
    all_scored: list[ScoredOpportunity] = []
    all_longterm: list[LongTermOpportunity] = []
    seen_symbols: set[str] = set()

    for entry, profile, setups in candidates:
        # Fetch company profile and news
        comp = fetch_company_profile(entry.yahoo_symbol)
        news_rep = fetch_news(entry.yahoo_symbol)

        # 1. Intraday setups scoring (with 10-point intraday criteria verification)
        for s in setups:
            score_res = score_setup(
                setup=s,
                profile=profile,
                company=comp,
                news=news_rep,
                macro=macro,
                universe_entry=entry,
            )
            all_scored.append(ScoredOpportunity(
                setup=s,
                profile=profile,
                company=comp,
                news=news_rep,
                score_breakdown=score_res,
                macro=macro,
            ))

        # 2. Long-term structural evaluation (once per symbol with 10-point fundamental criteria)
        if entry.yahoo_symbol not in seen_symbols:
            seen_symbols.add(entry.yahoo_symbol)
            lt_opp = evaluate_longterm_candidate(comp, profile, macro)
            if lt_opp:
                all_longterm.append(lt_opp)

    # 6. Executive Report Generation
    print("\n[6/6] Generating Executive Trade Report & Theses...", flush=True)
    qualified = [o for o in all_scored if o.score_breakdown.total_score >= args.min_score]
    pm_report = generate_premarket_report(macro)

    # Strategy Backtesting Validation
    test_syms = [o.setup.symbol + ".NS" for o in qualified[:3]] if qualified else ["RELIANCE.NS", "TCS.NS"]
    bt_report = run_strategy_backtest(test_syms, period="30d")

    report_text = generate_full_report(
        macro=macro,
        opportunities=all_scored,
        total_scanned=len(universe),
        min_score_threshold=args.min_score,
        max_picks=args.max_picks,
        longterm_opportunities=all_longterm,
        premarket_report=pm_report,
        backtest_report=bt_report,
    )
    print("\n" + report_text)

    # Log to journal and Immutable Prediction Registry
    if qualified:
        logged = log_opportunities(qualified)
        print(f"\n📝 Journal updated: {logged} high-conviction setup(s) logged to analyst_journal.csv")

        # Immutable Prediction Registry persistence
        try:
            import uuid
            from datetime import datetime as _pdt
            reg = PredictionRegistry()
            today_str = _pdt.now().strftime("%Y-%m-%d")
            saved_preds = []
            for o in qualified:
                s = o.setup
                rec_id = f"pred_{_pdt.now().strftime('%Y%m%d')}_{s.symbol}_{uuid.uuid4().hex[:6]}"
                rec = PredictionRecord(
                    id=rec_id,
                    symbol=s.symbol,
                    exchange="NSE",
                    timestamp=_pdt.now().isoformat(),
                    session_date=today_str,
                    direction=s.direction.value if hasattr(s.direction, "value") else str(s.direction),
                    setup_type=s.setup_type,
                    entry_price=float(s.trigger),
                    entry_zone_low=float(s.entry_low),
                    entry_zone_high=float(s.entry_high),
                    stop_loss=float(s.stop_loss),
                    target1=float(s.target1),
                    target2=float(s.target2),
                    holding_period="INTRADAY",
                    market_regime=(macro.regime.value if hasattr(macro.regime, "value") else str(macro.regime)) if macro and hasattr(macro, "regime") else "NEUTRAL",
                    technical_indicators={
                        "rsi": o.profile.daily.rsi if o.profile and o.profile.daily else None,
                        "adx": o.profile.daily.adx if o.profile and o.profile.daily else None,
                        "ema_20": o.profile.daily.ema20 if o.profile and o.profile.daily else None,
                        "ema_50": o.profile.daily.ema50 if o.profile and o.profile.daily else None,
                    },
                    rvol=o.profile.daily.rvol if o.profile and o.profile.daily else 1.0,
                    confidence_score=float(o.score_breakdown.total_score),
                    calibrated_confidence=float(o.score_breakdown.total_score * 0.95),
                    selection_reasons="; ".join(s.reasons),
                    failure_reasons="Reversal breach of stop-loss or intraday market chop",
                    status="ACTIVE",
                )
                reg.register(rec)
                saved_preds.append(rec.__dict__)
            StorageManager().save_predictions_snapshot(today_str, saved_preds)
            print(f"🔒 Prediction Registry: {len(saved_preds)} immutable predictions registered in SQLite & storage/predictions/")
        except Exception as e:
            print(f"⚠️ Warning registering predictions: {e}")


    # Generate HTML Tabular Report each time (with separate Intraday and Long-Term tables)
    out_html = ROOT / "report.html"
    generate_html_report(
        macro=macro,
        opportunities=all_scored,
        total_scanned=len(universe),
        min_score_threshold=args.min_score,
        max_picks=max(args.max_picks, 25),
        output_path=out_html,
        longterm_opportunities=all_longterm,
        premarket_report=pm_report,
        backtest_report=bt_report,
    )
    print(f"📊 Tabular HTML report generated: {out_html.resolve()}")

    # Also archive timestamped copy in reports/
    try:
        from datetime import datetime as _dt
        rep_dir = ROOT / "reports"
        rep_dir.mkdir(exist_ok=True)
        ts_name = rep_dir / f"report_{_dt.now().strftime('%Y-%m-%d_%H%M%S')}.html"
        ts_name.write_text(out_html.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Copy to current conversation artifact directory if available
    try:
        art_dir = Path(r"C:\Users\workJ\.gemini\antigravity\brain\81804217-2729-4e79-abbb-866a11cb1411")
        if art_dir.exists():
            (art_dir / "report.html").write_text(out_html.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Optionally write top setups into setups.yaml
    if args.save_setups and qualified:
        qualified.sort(key=lambda o: o.score_breakdown.total_score, reverse=True)
        top = qualified[:args.max_picks]
        setups_data = {"setups": []}
        for o in top:
            s = o.setup
            setups_data["setups"].append({
                "symbol": s.symbol,
                "name": s.name,
                "direction": s.direction.value,
                "setup_type": s.setup_type,
                "trigger": float(s.trigger),
                "entry_low": float(s.entry_low),
                "entry_high": float(s.entry_high),
                "stop_loss": float(s.stop_loss),
                "target1": float(s.target1),
                "target2": float(s.target2),
                "invalidate": float(s.invalidate),
                "atr": float(s.atr),
                "score": float(s.score or 0),
                "reasons": list(s.reasons),
                "expiry_ist": "14:45",
            })
        out_file = ROOT / "setups.yaml"
        out_file.write_text(yaml.safe_dump(setups_data, sort_keys=False), encoding="utf-8")
        print(f"💾 Top {len(top)} setup(s) exported to {out_file.name} for live monitoring!")

    return 0


def cmd_premarket(args) -> int:
    print("\n" + "═" * 78)
    print("🌅 FETCHING GLOBAL MACRO & INDIAN PRE-MARKET EXECUTIVE BRIEFING")
    print("═" * 78 + "\n")
    nse_prov = NseContextProvider(SessionClock())
    macro = fetch_global_snapshot(nse_provider=nse_prov)
    pm = generate_premarket_report(macro)
    print(pm.render_console())
    return 0


def cmd_backtest(args) -> int:
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    print("\n" + "═" * 78)
    print(f"📊 HISTORICAL STRATEGY VALIDATION: {args.strategy}")
    print(f"   Symbols ({len(syms)}): {', '.join(syms[:5])}...")
    print(f"   Period: {args.period} | Statutory Taxes & Slippage Applied: Yes")
    print("═" * 78 + "\n")
    bt = run_strategy_backtest(syms, period=args.period, strategy_type=args.strategy)
    for l in bt.summary_lines():
        print(f"  {l}")
    print("\n" + "─" * 78)
    return 0


def cmd_eod(args) -> int:
    print("\n" + "═" * 78)
    print("📈 END-OF-DAY JOURNAL & LEARNING SYSTEM AUDIT")
    print("═" * 78 + "\n")
    print(summarize_journal())
    print("\n" + "─" * 78)
    return 0


def cmd_monitor(args) -> int:
    from run_monitor import cmd_live, load_config
    cfg = load_config(ROOT / "config.yaml")
    setups_path = Path(args.setups) if args.setups else (ROOT / "setups.yaml")
    if not setups_path.exists():
        print(f"Error: {setups_path} not found. Run 'python run_analyst.py scan --save-setups' first!")
        return 1
    args.setups = str(setups_path)
    return cmd_live(args, cfg)


def cmd_morning(args) -> int:
    print("\n" + "═" * 78)
    print("🌅 EXECUTING AUTOMATED PRE-MARKET MORNING BRIEFING & MACRO INTELLIGENCE")
    print("═" * 78 + "\n")
    from datetime import datetime as _dt
    import json
    from analyst.database import get_db_connection
    nse_prov = NseContextProvider(SessionClock())
    macro = fetch_global_snapshot(nse_provider=nse_prov)
    pm = generate_premarket_report(macro)
    print(pm.render_console())

    # Persist to database & storage
    today_str = _dt.now().strftime("%Y-%m-%d")
    html_content = f"<html><body><h1>Pre-Market Briefing {today_str}</h1><pre>{pm.render_console()}</pre></body></html>"
    scenarios_dicts = [{"name": s.name, "bias": s.probability_bias, "range": s.opening_range_nifty, "plan": s.actionable_intraday_plan} for s in pm.scenarios]
    json_data = {"date": today_str, "sentiment": pm.gap_type, "regime": pm.regime, "scenarios": scenarios_dicts}
    hfile, _, checksum = StorageManager().save_morning_report(today_str, html_content, json_data)
    with get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO morning_reports (
                report_date, created_at, market_regime, global_risk_assessment,
                gift_nifty_sentiment, scenarios_json, html_path, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (today_str, _dt.now().isoformat(), pm.regime, pm.gap_type,
              pm.gap_type, json.dumps(scenarios_dicts), str(hfile), checksum))
    print(f"\n💾 Morning Report persisted to {hfile.resolve()} (SHA-256: {checksum[:12]}...)")
    return 0


def cmd_evening(args) -> int:
    print("\n" + "═" * 78)
    print("📊 EXECUTING EVENING OUTCOME AUDIT & PERFORMANCE EVALUATION")
    print("═" * 78 + "\n")
    reviewer = EveningReviewer()
    report = reviewer.evaluate_date(args.date)
    print(f"Trading Session: {report.report_date}")
    print(f"Total Predictions: {report.total_predictions} | Triggered Trades: {report.triggered_trades}")
    print(f"Win Rate: {report.win_rate_pct:.1f}% ({report.win_count} Wins / {report.loss_count} Losses)")
    print(f"Profit Factor: {report.profit_factor:.2f} | Avg R-Multiple: {report.avg_r_multiple:+.2f}R")
    print(f"Net Realized P&L (After Statutory Taxes): ₹{report.net_pnl_sum:+,.2f}")
    print(f"Ambiguous Trades (Quarantined): {report.ambiguous_count}")
    if report.failure_summary:
        print(f"Failure Taxonomy Breakdown: {report.failure_summary}")
    if report.html_path:
        print(f"\n📑 Report HTML generated: {report.html_path}")
    print("\n" + "─" * 78)
    return 0


def cmd_research(args) -> int:
    if not args.symbol:
        print("Error: --symbol is required (e.g. --symbol RELIANCE or TCS.NS)")
        return 1
    print("\n" + "═" * 78)
    print(f"🔍 CONDUCTING DEEP INSTITUTIONAL RESEARCH: {args.symbol.upper()}")
    print("═" * 78 + "\n")
    researcher = StockResearcher()
    dossier = researcher.research_symbol(args.symbol, force_refresh=args.refresh)
    print(f"Company: {dossier.company_name} ({dossier.symbol})")
    print(f"Sector: {dossier.sector} | Industry: {dossier.industry} | LTP: ₹{dossier.ltp:,.2f}")
    print(f"Market Cap: ₹{dossier.market_cap_cr:,.0f} Cr\n")
    print(f"Business Moats & Model:\n  {dossier.business_model[:300]}...\n")
    v = dossier.valuation_triad
    print(f"Intrinsic Valuation Triad:")
    print(f"  • Bear Case: ₹{v['bear_case']['fair_value']:,.2f} ({v['bear_case']['upside_pct']:+.1f}%)")
    print(f"  • Base Case: ₹{v['base_case']['fair_value']:,.2f} ({v['base_case']['upside_pct']:+.1f}%)")
    print(f"  • Bull Case: ₹{v['bull_case']['fair_value']:,.2f} ({v['bull_case']['upside_pct']:+.1f}%)\n")
    print(f"Strategic Outlook:\n  {dossier.long_term_outlook}")
    if dossier.html_path:
        print(f"\n📑 Detailed Dossier HTML: {dossier.html_path}")
    print("\n" + "─" * 78)
    return 0


def cmd_learn(args) -> int:
    print("\n" + "═" * 78)
    print("🧠 RUNNING SELF-LEARNING STRATEGY OPTIMIZATION CYCLE")
    print("═" * 78 + "\n")
    learner = LearningEngine()
    active_ver, params = learner.get_active_strategy()
    print(f"Current Active Strategy: {active_ver}")
    print(f"Active Parameters: {params}\n")
    exp = learner.run_learning_cycle()
    print(f"Experiment ID: {exp.id}")
    print(f"Candidate Version: {exp.candidate_strategy_version}")
    print(f"Mutated Parameters: {exp.proposed_parameters}")
    print(f"Identified Primary Flaw: {exp.error_patterns.get('primary_failure_mode')}")
    m = exp.walk_forward_metrics
    print(f"\nWalk-Forward Validation Results:")
    print(f"  Trades: {m.get('total_trades')} | Win Rate: {m.get('win_rate_pct'):.1f}% | Sharpe: {m.get('sharpe_ratio'):.2f}")
    print(f"  Max Drawdown: {m.get('max_drawdown_pct'):.1f}% | Profit Factor: {m.get('profit_factor'):.2f} | Net P&L: ₹{m.get('total_net_pnl'):,.2f}")
    print(f"\nDecision: {exp.decision}")
    if exp.rejection_reason:
        print(f"Rejection Reason: {exp.rejection_reason}")
    print("\n" + "─" * 78)
    return 0


def cmd_serve(args) -> int:
    server = run_web_server(port=args.port, enable_scheduler=not args.no_scheduler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...")
        server.shutdown()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Run institutional universe research scan")
    p_scan.add_argument("--universe", choices=["nifty50", "nifty100", "nifty500", "nse_all", "all"],
                        default="nse_all", help="Universe scope (default nse_all for all listed NSE stocks)")
    p_scan.add_argument("--exclude-nifty50", action="store_true",
                        help="Exclude Nifty 50 constituents to focus strictly on broader market")
    p_scan.add_argument("--limit", type=int, default=0,
                        help="Limit symbol count for testing (0 = full)")
    p_scan.add_argument("--fast", action="store_true",
                        help="Fast scan mode using daily bars")
    p_scan.add_argument("--min-score", type=int, default=55,
                        help="Minimum score threshold (out of 100)")
    p_scan.add_argument("--max-picks", type=int, default=5,
                        help="Maximum top opportunities to display")
    p_scan.add_argument("--save-setups", action="store_true",
                        help="Save top picks to setups.yaml for live monitoring")
    p_scan.set_defaults(func=cmd_scan)

    # morning
    p_morn = sub.add_parser("morning", help="Execute automated pre-market morning briefing & persist")
    p_morn.set_defaults(func=cmd_morning)

    # premarket (alias)
    p_prem = sub.add_parser("premarket", help="Run Indian pre-market & GIFT Nifty analysis")
    p_prem.set_defaults(func=cmd_premarket)

    # evening
    p_eve = sub.add_parser("evening", help="Execute evening outcome audit & performance review")
    p_eve.add_argument("--date", default=None, help="Trading date to evaluate (YYYY-MM-DD, default today)")
    p_eve.set_defaults(func=cmd_evening)

    # research
    p_res = sub.add_parser("research", help="Deep institutional individual stock research")
    p_res.add_argument("--symbol", required=True, help="NSE symbol to research (e.g. RELIANCE, TCS, INFY)")
    p_res.add_argument("--refresh", action="store_true", help="Force fresh data fetch instead of cache")
    p_res.set_defaults(func=cmd_research)

    # learn
    p_lrn = sub.add_parser("learn", help="Run self-learning strategy optimizer & walk-forward validation")
    p_lrn.set_defaults(func=cmd_learn)

    # serve
    p_srv = sub.add_parser("serve", help="Launch interactive Web Dashboard & Background Scheduler")
    p_srv.add_argument("--port", type=int, default=8050, help="Port to listen on (default 8050)")
    p_srv.add_argument("--no-scheduler", action="store_true", help="Disable the background scheduler")
    p_srv.set_defaults(func=cmd_serve)

    # backtest
    p_bt = sub.add_parser("backtest", help="Run historical strategy backtest & validation")
    p_bt.add_argument("--symbols", default="", help="Comma-separated Yahoo symbols (default top large-caps)")
    p_bt.add_argument("--period", default="60d", help="Historical data period (e.g. 30d, 60d, 1y)")
    p_bt.add_argument("--strategy", default="Prior_Day_Breakout", help="Strategy to test")
    p_bt.set_defaults(func=cmd_backtest)

    # monitor
    p_mon = sub.add_parser("monitor", help="Start real-time monitoring of generated setups")
    p_mon.add_argument("--setups", default=str(ROOT / "setups.yaml"),
                       help="Path to setups.yaml")
    p_mon.add_argument("--once", action="store_true",
                       help="Single poll then exit")
    p_mon.set_defaults(func=cmd_monitor)

    # eod
    p_eod = sub.add_parser("eod", help="End-of-day journal and learning summary")
    p_eod.set_defaults(func=cmd_eod)

    args = parser.parse_args(argv)
    return args.func(args)



if __name__ == "__main__":
    sys.exit(main())
