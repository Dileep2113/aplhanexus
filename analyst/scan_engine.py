"""Scan Engine for Institutional Research & Opportunity Detection.

Callable programmatically by:
  - Web Application UI (One-click trigger)
  - Background Automation Scheduler (Every 30m during market hours)
  - CLI (run_analyst.py scan)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from analyst.backtest import run_strategy_backtest
from analyst.company import fetch_company_profile
from analyst.global_macro import fetch_global_snapshot
from analyst.journal import log_opportunities
from analyst.longterm import LongTermOpportunity, evaluate_longterm_candidate
from analyst.news import fetch_news
from analyst.predictions import PredictionRecord, PredictionRegistry
from analyst.premarket import generate_premarket_report
from analyst.report import ScoredOpportunity, generate_full_report, generate_html_report
from analyst.scoring import score_setup
from analyst.setup_builder import build_candidate_setups
from analyst.storage import StorageManager
from analyst.technicals import analyse_symbol, batch_analyse_symbols
from analyst.universe import apply_liquidity_filter, fetch_universe
from intraday.providers import NseContextProvider
from intraday.session import SessionClock

logger = logging.getLogger("ScanEngine")
ROOT = Path(__file__).resolve().parent.parent


def execute_research_scan(
    universe_tier: str = "nifty50",
    limit: int = 0,
    min_score: int = 55,
    max_picks: int = 25,
    fast_mode: bool = True,
    exclude_nifty50: bool = False,
    save_setups: bool = True,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Execute end-to-end institutional scan across Indian equity universe."""

    def _progress(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _progress("Fetching Global Macro, Commodities, Currencies & NSE Context...")
    nse_prov = NseContextProvider(SessionClock())
    macro = fetch_global_snapshot(nse_provider=nse_prov)

    _progress(f"Loading eligible NSE universe ({universe_tier})...")
    raw_universe = fetch_universe(tier=universe_tier, exclude_nifty50=exclude_nifty50)
    if limit > 0:
        raw_universe = raw_universe[:limit]

    _progress(f"Applying liquidity filtration across {len(raw_universe)} symbols...")
    universe = apply_liquidity_filter(raw_universe, sample_size=limit)

    _progress(f"Analyzing technical setups across {len(universe)} symbols...")
    candidates: list[tuple] = []

    if len(universe) > 10 or fast_mode:
        syms_to_scan = [e.yahoo_symbol for e in universe]
        profile_map = batch_analyse_symbols(syms_to_scan, chunk_size=50)
        for entry in universe:
            profile = profile_map.get(entry.yahoo_symbol)
            if not profile or profile.ltp <= 0:
                continue
            setups = build_candidate_setups(profile)
            if setups:
                candidates.append((entry, profile, setups))
    else:
        for entry in universe:
            profile = analyse_symbol(entry.yahoo_symbol, fast_mode=fast_mode)
            if not profile or profile.ltp <= 0:
                continue
            setups = build_candidate_setups(profile)
            if setups:
                candidates.append((entry, profile, setups))

    _progress(f"Scoring {len(candidates)} chart setups with fundamentals & news...")
    all_scored: list[ScoredOpportunity] = []
    all_longterm: list[LongTermOpportunity] = []
    seen_symbols: set[str] = set()

    for entry, profile, setups in candidates:
        comp = fetch_company_profile(entry.yahoo_symbol)
        news_rep = fetch_news(entry.yahoo_symbol)

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

        if entry.yahoo_symbol not in seen_symbols:
            seen_symbols.add(entry.yahoo_symbol)
            lt_opp = evaluate_longterm_candidate(comp, profile, macro)
            if lt_opp:
                all_longterm.append(lt_opp)

    qualified = [o for o in all_scored if o.score_breakdown.total_score >= min_score]
    pm_report = generate_premarket_report(macro)

    # Strategy Backtesting Validation
    test_syms = [o.setup.symbol + ".NS" for o in qualified[:3]] if qualified else ["RELIANCE.NS", "TCS.NS"]
    bt_report = run_strategy_backtest(test_syms, period="30d")

    # Generate HTML Tabular Report
    out_html = ROOT / "report.html"
    generate_html_report(
        macro=macro,
        opportunities=all_scored,
        total_scanned=len(universe),
        min_score_threshold=min_score,
        max_picks=max_picks,
        output_path=out_html,
        longterm_opportunities=all_longterm,
        premarket_report=pm_report,
        backtest_report=bt_report,
    )

    # Archive to storage/reports/
    now_dt = datetime.now()
    rep_dir = ROOT / "reports"
    rep_dir.mkdir(exist_ok=True)
    ts_name = rep_dir / f"report_{now_dt.strftime('%Y-%m-%d_%H%M%S')}.html"
    try:
        ts_name.write_text(out_html.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Copy to brain artifact directory if exists
    try:
        art_dir = Path(r"C:\Users\workJ\.gemini\antigravity\brain\81804217-2729-4e79-abbb-866a11cb1411")
        if art_dir.exists():
            (art_dir / "report.html").write_text(out_html.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Register immutable predictions
    saved_preds = []
    if qualified:
        log_opportunities(qualified)
        reg = PredictionRegistry()
        today_str = now_dt.strftime("%Y-%m-%d")
        for o in qualified:
            s = o.setup
            rec_id = f"pred_{now_dt.strftime('%Y%m%d')}_{s.symbol}_{uuid.uuid4().hex[:6]}"
            rec = PredictionRecord(
                id=rec_id,
                symbol=s.symbol,
                exchange="NSE",
                timestamp=now_dt.isoformat(),
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
            try:
                reg.register(rec)
                saved_preds.append(rec.__dict__)
            except Exception:
                pass
        StorageManager().save_predictions_snapshot(today_str, saved_preds)

    # Save structured results for web dashboard native rendering
    intraday_data = []
    for o in qualified:
        s = o.setup
        intraday_data.append({
            "symbol": s.symbol,
            "name": s.name,
            "direction": s.direction.value if hasattr(s.direction, "value") else str(s.direction),
            "setup_type": s.setup_type,
            "trigger": float(s.trigger),
            "entry_low": float(s.entry_low),
            "entry_high": float(s.entry_high),
            "stop_loss": float(s.stop_loss),
            "target1": float(s.target1),
            "target2": float(s.target2),
            "atr": float(s.atr),
            "score": float(o.score_breakdown.total_score),
            "tier": o.score_breakdown.candidate_tier.value if hasattr(o.score_breakdown.candidate_tier, "value") else str(o.score_breakdown.candidate_tier),
            "net_rr": round(o.score_breakdown.net_rr, 2) if hasattr(o.score_breakdown, "net_rr") else 2.5,
            "reasons": list(s.reasons),
            "rejection_reasons": list(o.score_breakdown.rejection_reasons),
        })

    longterm_data = []
    for lt in all_longterm:
        longterm_data.append({
            "symbol": lt.symbol,
            "name": lt.name,
            "sector": lt.sector,
            "industry": lt.industry,
            "ltp": float(lt.ltp),
            "market_cap_cr": float(lt.market_cap_cr or 0),
            "pe_ratio": float(lt.pe_ratio or 0),
            "roe": float((lt.roe or 0) * 100.0) if lt.roe else None,
            "debt_to_equity": float(lt.debt_to_equity or 0),
            "rev_growth": float((lt.revenue_growth_yoy or 0) * 100.0) if lt.revenue_growth_yoy else None,
            "score": int(lt.score),
            "verdict": lt.verdict,
            "valuation_verdict": lt.valuation_verdict,
            "valuation_scenarios": lt.valuation_scenarios,
            "thesis": lt.thesis,
            "strengths": lt.strengths,
            "risks": lt.risks,
        })

    import json as _json
    cache_payload = {
        "timestamp": now_dt.isoformat(),
        "total_scanned": len(universe),
        "macro": {
            "regime": macro.regime.value if hasattr(macro.regime, "value") else str(macro.regime),
            "nifty_close": macro.nse.nifty50.value if (macro.nse and macro.nse.nifty50) else 23146.15,
            "vix": macro.nse.india_vix.value if (macro.nse and macro.nse.india_vix) else 14.0,
            "gift_nifty_gap": pm_report.gap_points if pm_report else 0,
            "gap_type": pm_report.gap_type if pm_report else "FLAT",
            "scenarios": [{"name": sc.name, "bias": sc.probability_bias, "range": sc.opening_range_nifty, "plan": sc.actionable_intraday_plan} for sc in (pm_report.scenarios if pm_report else [])],
        },
        "intraday": intraday_data,
        "longterm": longterm_data,
    }
    (ROOT / "_latest_scan.json").write_text(_json.dumps(cache_payload, indent=2, default=str), encoding="utf-8")


    # Optionally write top setups to setups.yaml
    if save_setups and qualified:
        qualified.sort(key=lambda o: o.score_breakdown.total_score, reverse=True)
        top = qualified[:max_picks]
        setups_data = {"setups": []}
        for o in top:
            s = o.setup
            setups_data["setups"].append({
                "symbol": s.symbol,
                "name": s.name,
                "direction": s.direction.value if hasattr(s.direction, "value") else str(s.direction),
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
        (ROOT / "setups.yaml").write_text(yaml.safe_dump(setups_data, sort_keys=False), encoding="utf-8")

    _progress(f"Scan complete: {len(universe)} symbols scanned, {len(qualified)} qualified setups found.")

    return {
        "status": "COMPLETED",
        "total_scanned": len(universe),
        "candidates_found": len(candidates),
        "qualified_setups": len(qualified),
        "longterm_opportunities": len(all_longterm),
        "predictions_registered": len(saved_preds),
        "html_report_path": str(out_html),
    }
