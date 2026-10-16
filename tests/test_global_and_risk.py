"""Unit & Integration tests for Global Intelligence, Premarket, Risk, and Backtesting modules."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import pytest

from analyst.global_macro import GlobalSnapshot, PricePoint, NseContext
from analyst.global_events import get_upcoming_macro_events, assess_global_linkage
from analyst.premarket import generate_premarket_report
from analyst.risk_manager import calculate_trade_costs, compute_net_reward_risk, size_intraday_position
from analyst.backtest import run_strategy_backtest, BacktestTrade, BacktestReport


def test_global_events_calendar():
    events = get_upcoming_macro_events()
    assert len(events) >= 4
    names = [e.name for e in events]
    assert any("FOMC" in n or "Federal Reserve" in n for n in names)
    assert any("CPI" in n for n in names)
    assert any("RBI" in n for n in names)
    for e in events:
        assert e.scheduled_at is not None
        assert e.impact_level in ("HIGH", "MEDIUM", "LOW")
        assert len(e.affected_sectors) > 0


def test_global_linkage_sensitivities():
    IST = ZoneInfo("Asia/Kolkata")
    macro = GlobalSnapshot(
        fetched_at=datetime.now(IST),
        regime="RISK_ON",
        us=[PricePoint("^IXIC", "NASDAQ", 18500.0, 1.8), PricePoint("^GSPC", "S&P 500", 5800.0, 0.8)],
        commodities=[PricePoint("BZ=F", "Brent Crude", 74.5, -2.5), PricePoint("HG=F", "Copper", 4.4, 1.9)],
        currency=[PricePoint("USDINR=X", "USD/INR", 83.9, 0.25)],
    )

    # 1. IT sensitivity test
    it_assess = assess_global_linkage("INFY.NS", "Information Technology", macro)
    assert it_assess.net_global_score > 0
    assert "US Tech tailwind" in it_assess.primary_global_driver
    assert any("USD/INR" in s for s in it_assess.key_sensitivities)

    # 2. Downstream oil/paint sensitivity test (softening crude = positive)
    paint_assess = assess_global_linkage("ASIANPAINT.NS", "Paints", macro)
    assert paint_assess.net_global_score > 0
    assert "softening" in paint_assess.primary_global_driver.lower() or "crude" in paint_assess.primary_global_driver.lower()

    # 3. Metals sensitivity test (rising copper = positive)
    metal_assess = assess_global_linkage("TATASTEEL.NS", "Metals", macro)
    assert metal_assess.net_global_score > 0
    assert "Copper" in metal_assess.primary_global_driver


def test_premarket_scenario_generation():
    IST = ZoneInfo("Asia/Kolkata")
    macro = GlobalSnapshot(
        fetched_at=datetime.now(IST),
        regime="RISK_ON",
        us=[PricePoint("^GSPC", "S&P 500", 5800.0, 0.8)],
        asian=[PricePoint("^N225", "Nikkei 225", 39000.0, 0.9)],
        nse=NseContext(nifty50=PricePoint("^NSEI", "Nifty 50", 24000.0, 0.4)),
    )

    pm = generate_premarket_report(macro)
    assert pm.gift_nifty_implied is not None
    assert pm.gap_points is not None
    assert len(pm.scenarios) >= 1
    assert pm.regime == "RISK_ON"

    console_out = pm.render_console()
    assert "INDIAN PRE-MARKET EXECUTIVE BRIEFING" in console_out
    assert "SCENARIO ANALYSIS" in console_out
    assert "GIFT Nifty" in console_out


def test_statutory_trade_costs():
    # 100 shares at ₹2,500 buy, ₹2,550 sell
    costs = calculate_trade_costs(entry_price=2500.0, exit_price=2550.0, shares=100, is_intraday=True)

    assert costs.brokerage > 0
    assert costs.brokerage <= 40.0  # ₹20 max per leg
    assert costs.stt > 0            # 0.025% on sell
    assert costs.gst > 0            # 18% on fees
    assert costs.stamp_duty > 0     # 0.003% on buy
    assert costs.exchange_charges > 0
    assert costs.slippage > 0
    assert costs.total_statutory_and_slippage > (costs.brokerage + costs.stt)


def test_net_reward_risk_calculation():
    # Entry 1000, SL 980 (risk 20), Target 1050 (reward 50) -> Gross R:R = 2.5
    res = compute_net_reward_risk(entry_price=1000.0, stop_loss=980.0, target=1050.0, shares=100, is_intraday=True)

    assert res.gross_rr == 2.50
    # Net R:R must be strictly less than Gross R:R due to statutory taxes and slippage
    assert res.net_rr < res.gross_rr
    assert res.net_rr > 1.80  # Still very viable
    assert res.viable is True
    assert res.total_estimated_costs > 0


def test_position_sizing_and_daily_lockout():
    capital = 500000.0
    risk_pct = 1.0       # ₹5,000 budget
    entry = 1000.0
    sl = 990.0           # stop distance = 10 -> 500 shares
    target = 1030.0      # reward = 30 -> gross R:R = 3.0

    # Normal sizing
    s1 = size_intraday_position(
        capital=capital,
        risk_pct_per_trade=risk_pct,
        entry_price=entry,
        stop_loss=sl,
        target=target,
        daily_realized_loss=0.0,
    )
    assert s1.is_allowed is True
    assert s1.shares > 0
    assert s1.shares <= 500

    # Daily loss lockout breach (realized loss > max allowed e.g. -₹8,000 vs 1.5% limit = ₹7,500)
    s2 = size_intraday_position(
        capital=capital,
        risk_pct_per_trade=risk_pct,
        entry_price=entry,
        stop_loss=sl,
        target=target,
        daily_realized_loss=-8000.0,
        max_daily_loss_pct=1.5,
    )
    assert s2.is_allowed is False
    assert s2.shares == 0
    assert any("DAILY_LOSS_LOCKOUT" in r for r in s2.rejection_reasons)
