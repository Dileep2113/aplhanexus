"""Unit tests for the autonomous analyst modules."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from intraday.models import Direction
from analyst.company import CompanyProfile
from analyst.global_macro import GlobalSnapshot, PricePoint, NseContext
from analyst.news import NewsItem, NewsReport, _classify_article
from analyst.report import ScoredOpportunity, generate_full_report, render_trade_thesis
from analyst.scoring import score_setup
from analyst.setup_builder import build_candidate_setups
from analyst.technicals import LevelSet, TechnicalProfile, TimeframeView
from analyst.universe import UniverseEntry, _sector_from_industry


def test_sector_from_industry_mapping():
    assert _sector_from_industry("Private Sector Bank") == "Financials"
    assert _sector_from_industry("Computers - Software") == "Information Technology"
    assert _sector_from_industry("Refineries") == "Energy"
    assert _sector_from_industry("Pharmaceuticals") == "Healthcare"
    assert _sector_from_industry("Automobiles - Passenger Cars") == "Auto"
    assert _sector_from_industry("Unknown Gizmos") == "Others"


def test_news_classification_keywords():
    cat1, sent1 = _classify_article("Company Q2 Net Profit up 45% YoY to record high")
    assert cat1 == "RESULTS"
    assert sent1 == "POSITIVE"

    cat2, sent2 = _classify_article("SEBI issues regulatory penalty notice on disclosure lapse")
    assert cat2 == "REGULATORY"
    assert sent2 == "NEGATIVE"

    cat3, sent3 = _classify_article("Company bags Rs 1500 Cr railway procurement order win")
    assert cat3 == "ORDER"
    assert sent3 == "POSITIVE"


def test_setup_builder_generates_strictly_valid_setups():
    levels = LevelSet(
        prev_day_high=1500.0,
        prev_day_low=1460.0,
        week_high=1520.0,
        week_low=1440.0,
    )
    daily = TimeframeView(
        name="Daily",
        interval="1d",
        trend="BULLISH",
        atr=25.0,
        rsi=60.0,
        rvol=1.5,
    )
    profile = TechnicalProfile(
        symbol="TEST.NS",
        ltp=1495.0,
        levels=levels,
        daily=daily,
        mtf_score=8,
    )

    setups = build_candidate_setups(profile)
    assert len(setups) >= 2  # Both LONG breakout and SHORT breakdown

    for s in setups:
        # Every setup MUST pass core structural validation
        errs = s.validate()
        assert not errs, f"Validation failed for {s.name}: {errs}"

        # Risk reward must be strictly positive
        rr = s.rr()
        assert rr is not None and rr > 1.5

        # Stop loss must be on proper side
        if s.direction is Direction.LONG:
            assert s.stop_loss < s.trigger
            assert s.target1 > s.trigger
            assert s.target2 > s.target1
        else:
            assert s.stop_loss > s.trigger
            assert s.target1 < s.trigger
            assert s.target2 < s.target1


def test_scoring_engine_and_bounds():
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)

    macro = GlobalSnapshot(
        fetched_at=now,
        regime="RISK_ON",
        nse=NseContext(
            nifty50=PricePoint("^NSEI", "Nifty 50", 23500.0, 0.5),
            india_vix=PricePoint("^INDIAVIX", "India VIX", 12.0, -2.0),
        )
    )

    levels = LevelSet(prev_day_high=1000.0, prev_day_low=970.0)
    daily = TimeframeView(name="Daily", interval="1d", trend="BULLISH", atr=15.0, rvol=1.6, rsi=62.0)
    tech = TechnicalProfile(symbol="TEST.NS", ltp=995.0, levels=levels, daily=daily, mtf_score=8)
    setups = build_candidate_setups(tech)
    assert setups

    comp = CompanyProfile(
        symbol="TEST.NS",
        name="Test Corp",
        sector="Financials",
        revenue_growth_yoy=0.20,
        earnings_growth_yoy=0.25,
        operating_margin=0.22,
        debt_to_equity=30.0,
        pe_ratio=18.0,
    )

    news = NewsReport(
        symbol="TEST.NS",
        sentiment_score=0.5,
        sentiment_label="POSITIVE",
        primary_catalyst="RESULTS: Record Q2 PAT growth",
    )

    entry = UniverseEntry(
        symbol="TEST",
        yahoo_symbol="TEST.NS",
        company_name="Test Corp",
        sector="Financials",
        industry="Private Sector Bank",
        avg_volume=1_500_000,
    )

    long_setup = [s for s in setups if s.direction is Direction.LONG][0]
    score_res = score_setup(long_setup, tech, comp, news, macro, entry)

    assert 0 <= score_res.total_score <= 100
    assert score_res.total_score >= 60  # Good alignment should score well
    assert score_res.market_alignment > 0
    assert score_res.technical_setup > 0
    assert score_res.catalyst_news > 0
    assert score_res.risk_reward > 0


def test_report_no_trade_fallback():
    IST = ZoneInfo("Asia/Kolkata")
    macro = GlobalSnapshot(fetched_at=datetime.now(IST))
    report = generate_full_report(macro=macro, opportunities=[], total_scanned=50, min_score_threshold=65)
    assert "NO TRADE" in report
    assert "Capital preservation" in report


def test_report_renders_trade_thesis():
    IST = ZoneInfo("Asia/Kolkata")
    macro = GlobalSnapshot(
        fetched_at=datetime.now(IST),
        regime="RISK_ON",
        us=[PricePoint("^GSPC", "S&P 500", 7600.0, 0.5)],
        nse=NseContext(nifty50=PricePoint("^NSEI", "Nifty 50", 23500.0, 0.4)),
    )

    levels = LevelSet(prev_day_high=500.0, prev_day_low=480.0)
    tech = TechnicalProfile(symbol="TEST.NS", ltp=498.0, levels=levels, mtf_score=8)
    setups = build_candidate_setups(tech)
    s = setups[0]

    comp = CompanyProfile(symbol="TEST.NS", name="Test Inc", sector="Energy")
    news = NewsReport(symbol="TEST.NS", primary_catalyst="Capex expansion")
    score_res = score_setup(s, tech, comp, news, macro)

    opp = ScoredOpportunity(setup=s, profile=tech, company=comp, news=news, score_breakdown=score_res, macro=macro)
    thesis = render_trade_thesis(opp)

    assert "TRADE THESIS" in thesis
    assert "Company" in thesis
    assert "Industry" in thesis
    assert "Technical" in thesis
    assert "Trade" in thesis


def test_intraday_criteria_checklist_populated():
    from analyst.technicals import LevelSet, TechnicalProfile, TimeframeView
    from analyst.company import CompanyProfile
    from analyst.news import NewsReport
    from analyst.scoring import score_setup
    from analyst.global_macro import GlobalSnapshot, NseContext, PricePoint
    from intraday.models import Direction
    from zoneinfo import ZoneInfo

    macro = GlobalSnapshot(
        fetched_at=datetime.now(ZoneInfo("Asia/Kolkata")),
        regime="RISK_ON",
        nse=NseContext(nifty50=PricePoint("^NSEI", "Nifty 50", 24000.0, 0.4), india_vix=PricePoint("^INDIAVIX", "VIX", 13.5, -1.0))
    )
    daily = TimeframeView(name="Daily", interval="1d", trend="BULLISH", atr=10.0, rvol=1.5, rsi=60.0)
    profile = TechnicalProfile(symbol="TEST.NS", ltp=100.0, levels=LevelSet(prev_day_high=101.0, prev_day_low=97.0), daily=daily, mtf_score=8)
    setups = build_candidate_setups(profile)
    long_s = [s for s in setups if s.direction is Direction.LONG][0]

    comp = CompanyProfile(symbol="TEST.NS", name="Test Co", sector="Technology")
    news = NewsReport(symbol="TEST.NS", sentiment_score=0.4, sentiment_label="POSITIVE")
    sc = score_setup(long_s, profile, comp, news, macro)

    # 10 Intraday criteria must be populated
    assert len(sc.criteria) == 10
    names = [c.name for c in sc.criteria]
    assert any("Market Alignment" in n for n in names)
    assert any("Technical Setup" in n for n in names)
    assert any("Multi-Timeframe" in n for n in names)
    assert any("Volume Confirmation" in n for n in names)
    assert any("RSI Momentum" in n for n in names)
    assert any("VWAP" in n for n in names)
    assert any("Catalyst" in n for n in names)
    assert any("Sector" in n for n in names)
    assert any("Risk-to-Reward" in n for n in names)
    assert any("Liquidity" in n for n in names)

    # Check status values are strictly valid
    for c in sc.criteria:
        assert c.status in ("PASS", "WARN", "FAIL")
        assert len(c.requirement) > 0
        assert len(c.actual) > 0


def test_longterm_evaluation_and_criteria():
    from analyst.longterm import evaluate_longterm_candidate
    from analyst.company import CompanyProfile
    from analyst.technicals import TechnicalProfile, TimeframeView, LevelSet
    from analyst.global_macro import GlobalSnapshot
    from zoneinfo import ZoneInfo

    macro = GlobalSnapshot(fetched_at=datetime.now(ZoneInfo("Asia/Kolkata")), regime="RISK_ON")
    daily = TimeframeView(name="Daily", interval="1d", trend="BULLISH", ema200=450.0, adx=26.0)
    profile = TechnicalProfile(symbol="QUALITY.NS", ltp=500.0, levels=LevelSet(), daily=daily)

    comp = CompanyProfile(
        symbol="QUALITY.NS",
        name="Quality Industries",
        sector="Industrials",
        market_cap_cr=25000.0,
        pe_ratio=28.0,
        roe=0.22,
        debt_to_equity=25.0,  # 0.25x D/E
        revenue_growth_yoy=0.18,
        earnings_growth_yoy=0.24,
        operating_margin=0.21,
        high_52w=520.0,
        low_52w=380.0,
    )

    lt = evaluate_longterm_candidate(comp, profile, macro)
    assert lt is not None
    assert lt.score >= 70
    assert lt.verdict in ("STRONG BUY", "ACCUMULATE")
    assert lt.above_200_ema is True

    # 10 Long-term criteria must be present
    assert len(lt.criteria) == 10
    cr_names = [c.name for c in lt.criteria]
    assert any("Revenue Growth" in n for n in cr_names)
    assert any("Earnings (PAT) Growth" in n for n in cr_names)
    assert any("Operating Margin" in n for n in cr_names)
    assert any("Return on Equity" in n for n in cr_names)
    assert any("Debt-to-Equity" in n for n in cr_names)
    assert any("200 EMA" in n for n in cr_names)
    assert any("52-Week High" in n for n in cr_names)
    assert any("Valuation" in n for n in cr_names)
    assert any("ADX" in n for n in cr_names)
    assert any("Macro Regime" in n for n in cr_names)


def test_html_report_renders_both_tables_and_criteria():
    from analyst.report import generate_html_report, ScoredOpportunity
    from analyst.longterm import evaluate_longterm_candidate
    from analyst.technicals import TechnicalProfile, TimeframeView, LevelSet
    from analyst.company import CompanyProfile
    from analyst.news import NewsReport
    from analyst.scoring import score_setup
    from analyst.global_macro import GlobalSnapshot, NseContext, PricePoint
    from zoneinfo import ZoneInfo

    macro = GlobalSnapshot(
        fetched_at=datetime.now(ZoneInfo("Asia/Kolkata")),
        regime="RISK_ON",
        nse=NseContext(nifty50=PricePoint("^NSEI", "Nifty 50", 24000.0, 0.5))
    )
    daily = TimeframeView(name="Daily", interval="1d", trend="BULLISH", atr=10.0, rvol=1.5, rsi=60.0, ema200=90.0, adx=25.0)
    profile = TechnicalProfile(symbol="TEST.NS", ltp=100.0, levels=LevelSet(prev_day_high=101.0, prev_day_low=97.0), daily=daily, mtf_score=8)
    setups = build_candidate_setups(profile)
    comp = CompanyProfile(symbol="TEST.NS", name="Test Co", sector="Technology", revenue_growth_yoy=0.15, earnings_growth_yoy=0.20, operating_margin=0.18, roe=0.19, debt_to_equity=30.0, pe_ratio=25.0, high_52w=105.0)
    news = NewsReport(symbol="TEST.NS", sentiment_score=0.4, sentiment_label="POSITIVE")

    sc = score_setup(setups[0], profile, comp, news, macro)
    opp = ScoredOpportunity(setup=setups[0], profile=profile, company=comp, news=news, score_breakdown=sc, macro=macro)
    lt_opp = evaluate_longterm_candidate(comp, profile, macro)

    html_out = generate_html_report(
        macro=macro,
        opportunities=[opp],
        total_scanned=10,
        min_score_threshold=50,
        longterm_opportunities=[lt_opp] if lt_opp else [],
    )

    # Must contain both distinct table sections
    assert 'id="intraday-section"' in html_out
    assert 'id="longterm-section"' in html_out
    assert 'id="criteria-reference"' in html_out

    # Must list criteria verification
    assert "Intraday Criteria Checklist Audit" in html_out
    assert "Long-Term Fundamental & Structural Criteria Audit" in html_out
    assert "Revenue Growth YoY" in html_out
    assert "Market Alignment (Nifty &amp; VIX)" in html_out

