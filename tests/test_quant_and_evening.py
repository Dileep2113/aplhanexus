"""Unit tests for SEBI Quantitative Engine, Gann Square of 9, Evening Review Failure Diagnostics, and Scheduler Execution Logs.
"""

import pytest
import math
from pathlib import Path
from analyst.gann import compute_gann_square9
from analyst.technicals import (
    compute_pivots,
    detect_wyckoff_phase,
    detect_fvg,
    detect_bos_choch,
    compute_ema_variances,
)
from analyst.database import init_database
from analyst.storage import StorageManager
from analyst.predictions import PredictionRecord, PredictionRegistry
from analyst.evening_review import EveningReviewer
from analyst.scheduler import BackgroundScheduler
from analyst.learning_engine import LearningEngine


def test_gann_square_of_9():
    cmp = 2500.0  # sqrt(2500) = 50.0
    g = compute_gann_square9(cmp)
    
    assert g["base_factor"] == 50
    # T1 (90°): (50 + 90/180)^2 = (50.5)^2 = 2550.25
    assert g["t1_price"] == 2550.25
    # T2 (180°): (50 + 180/180)^2 = (51.0)^2 = 2601.0
    assert g["t2_price"] == 2601.0
    # T3 (360°): (50 + 360/180)^2 = (52.0)^2 = 2704.0
    assert g["t3_price"] == 2704.0
    # Invalidation floor: (50 - 0.5)^2 = (49.5)^2 = 2450.25
    assert g["invalidation_floor"] == 2450.25
    assert g["t1_upside_pct"] == round((2550.25 - 2500.0) / 2500.0 * 100.0, 2)
    assert g["t1_horizon_days"] == 22


def test_pivots_calculation():
    p = compute_pivots(prev_high=105.0, prev_low=95.0, prev_close=100.0)
    # Pivot = (105 + 95 + 100) / 3 = 100.0
    assert p["pivot"] == 100.0
    # R1 = 2*100 - 95 = 105.0
    assert p["r1"] == 105.0
    # S1 = 2*100 - 105 = 95.0
    assert p["s1"] == 95.0


def test_fvg_detection():
    # Bullish FVG: Low[2] > High[0]
    highs = [100.0, 104.0, 108.0]
    lows = [95.0, 101.0, 102.0]
    res = detect_fvg(highs, lows)
    assert res["fvg_detected"] is True
    assert res["bullish_fvg_lower"] == 100.0
    assert res["bullish_fvg_upper"] == 102.0


def test_bos_choch_detection():
    closes = [100.0 + i for i in range(25)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    res = detect_bos_choch(closes, highs, lows, swing_bars=10)
    assert res["bos_type"] == "BULLISH_BOS"


def test_evening_review_diagnoses_and_populates_failures(tmp_path):
    db_path = tmp_path / "test.db"
    init_database(db_path)
    storage = StorageManager(tmp_path / "storage")
    reg = PredictionRegistry(db_path)
    reviewer = EveningReviewer(db_path, storage)

    # Register an intentionally failing prediction
    pred = PredictionRecord(
        id="pred_fail_test",
        symbol="RELIANCE.NS",
        exchange="NSE",
        timestamp="2026-10-22T09:30:00",
        session_date="2026-10-22",
        direction="LONG",
        setup_type="Prior_Day_High_Breakout",
        entry_price=2500.0,
        entry_zone_low=2495.0,
        entry_zone_high=2505.0,
        stop_loss=2450.0,
        target1=2560.0,
        target2=2600.0,
    )
    reg.register(pred)

    report = reviewer.evaluate_date("2026-10-22")
    assert report.total_predictions >= 1
    # Check that outcomes have rich metadata
    out = next(o for o in report.outcomes if o.prediction_id == "pred_fail_test")
    assert out.symbol == "RELIANCE.NS"
    assert out.direction == "LONG"
    assert out.setup_type == "Prior_Day_High_Breakout"
    assert out.entry_price == 2500.0


def test_scheduler_outcome_message_and_execution_logs(tmp_path):
    db_path = tmp_path / "test_sched.db"
    init_database(db_path)
    sched = BackgroundScheduler(db_path)

    # Register mock job that returns structured message
    sched.register_handler("morning_report", lambda: "🌅 Morning scan complete: 15 symbols analyzed.")
    sched._execute_job("morning_report", sched._handlers["morning_report"])

    statuses = sched.get_job_statuses()
    jobs = statuses["jobs"]
    history = statuses["history"]

    assert "morning_report" in jobs
    assert jobs["morning_report"]["last_status"] == "SUCCESS"
    assert "Morning scan complete" in jobs["morning_report"]["last_message"]
    assert len(history) >= 1
    assert history[0]["job_name"] == "morning_report"
    assert history[0]["status"] == "SUCCESS"
