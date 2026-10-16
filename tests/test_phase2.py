"""Tests for Phase 2: Complete, Persistent, Self-Learning Stock Research Web Application."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from analyst.database import get_db_connection, init_database
from analyst.evening_review import EveningReviewer, PredictionOutcome
from analyst.learning_engine import LearningEngine
from analyst.predictions import PredictionRecord, PredictionRegistry
from analyst.scheduler import BackgroundScheduler
from analyst.stock_research import StockResearcher
from analyst.storage import StorageManager, compute_sha256


@pytest.fixture
def temp_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test_analyst.db"
        storage_dir = tmp_path / "test_storage"
        init_database(db_path)
        yield db_path, storage_dir


def test_database_init(temp_env):
    db_path, _ = temp_env
    with get_db_connection(db_path) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "predictions" in tables
        assert "prediction_outcomes" in tables
        assert "morning_reports" in tables
        assert "evening_reports" in tables
        assert "strategy_versions" in tables
        assert "learning_experiments" in tables
        assert "stock_research" in tables
        assert "scheduler_jobs" in tables

        # Baseline strategy exists
        strat = conn.execute("SELECT * FROM strategy_versions WHERE version = 'v1.0.0'").fetchone()
        assert strat is not None
        assert strat["is_active"] == 1


def test_storage_manager_and_retention(temp_env):
    _, storage_dir = temp_env
    storage = StorageManager(storage_dir)

    # Test report save and checksum
    html_file, json_file, csum = storage.save_morning_report(
        "2026-10-16",
        "<html>Test Morning</html>",
        {"sentiment": "BULLISH"}
    )
    assert html_file.exists()
    assert json_file.exists()
    assert compute_sha256("<html>Test Morning</html>") == csum

    # Retention policy
    retention_res = storage.apply_retention_policy(retention_days=365)
    assert "retention_days" in retention_res
    assert retention_res["retention_days"] == 365


def test_prediction_registry_immutability(temp_env):
    db_path, _ = temp_env
    reg = PredictionRegistry(db_path)

    p1 = PredictionRecord(
        id="pred_001",
        symbol="RELIANCE.NS",
        exchange="NSE",
        timestamp="2026-10-16T09:30:00",
        session_date="2026-10-16",
        direction="LONG",
        setup_type="Prior_Day_High_Breakout",
        entry_price=3000.0,
        entry_zone_low=2995.0,
        entry_zone_high=3005.0,
        stop_loss=2950.0,
        target1=3060.0,
        target2=3120.0,
        confidence_score=78.5,
    )
    reg.register(p1)

    # Fetch
    fetched = reg.get("pred_001")
    assert fetched is not None
    assert fetched.symbol == "RELIANCE.NS"
    assert fetched.direction == "LONG"
    assert fetched.entry_price == 3000.0

    # Test Immutability: attempting to re-register same ID must raise ValueError
    with pytest.raises(ValueError):
        reg.register(p1)

    # Test Revision: creates new linked record
    rev_id = reg.create_revision("pred_001", {"target1": 3075.0})
    assert rev_id != "pred_001"
    revised = reg.get(rev_id)
    assert revised.parent_prediction_id == "pred_001"
    assert revised.target1 == 3075.0


def test_evening_reviewer_ambiguity_rule(temp_env):
    db_path, storage_dir = temp_env
    storage = StorageManager(storage_dir)
    reg = PredictionRegistry(db_path)
    reviewer = EveningReviewer(db_path, storage)

    p = PredictionRecord(
        id="pred_ambig",
        symbol="TCS.NS",
        exchange="NSE",
        timestamp="2026-10-16T09:30:00",
        session_date="2026-10-16",
        direction="LONG",
        setup_type="ORB",
        entry_price=4000.0,
        entry_zone_low=3990.0,
        entry_zone_high=4010.0,
        stop_loss=3900.0,
        target1=4100.0,
        target2=4200.0,
    )
    reg.register(p)

    # Construct synthetic daily bar where High >= Target1 (4120) and Low <= StopLoss (3880)
    df_daily = pd.DataFrame([{
        "open": 4000.0,
        "high": 4150.0,
        "low": 3850.0,
        "close": 4020.0,
    }], index=["2026-10-16"])

    outcome = reviewer._evaluate_with_bars(p, df_daily, is_intraday=False)

    # Per institutional specification: on daily bars where both are breached,
    # it must be marked AMBIGUOUS_OUTCOME, never assuming target hit first!
    assert outcome.outcome_status == "AMBIGUOUS_OUTCOME"
    assert outcome.first_hit == "AMBIGUOUS"


def test_learning_engine_promotion_gates(temp_env):
    db_path, storage_dir = temp_env
    storage = StorageManager(storage_dir)
    learner = LearningEngine(db_path, storage)

    active_ver, params = learner.get_active_strategy()
    assert active_ver == "v1.0.0"
    assert "preferred_rvol" in params

    # Test hypothesis mutation
    cand_ver, mutated = learner._generate_hypothesis(
        active_ver,
        params,
        {"primary_failure_mode": "FALSE_BREAKOUT"}
    )
    assert cand_ver != active_ver
    assert mutated["min_rvol"] >= params["min_rvol"]


def test_stock_research_dossier_structure(temp_env):
    db_path, storage_dir = temp_env
    storage = StorageManager(storage_dir)
    researcher = StockResearcher(db_path, storage)

    dossier = researcher.research_symbol("INFY.NS")
    assert dossier.symbol == "INFY.NS"
    assert dossier.company_name
    assert "bear_case" in dossier.valuation_triad
    assert "base_case" in dossier.valuation_triad
    assert "bull_case" in dossier.valuation_triad
    assert "promoter_holding_pct" in dossier.governance
    assert dossier.html_path is not None


def test_scheduler_lifecycle(temp_env):
    db_path, _ = temp_env
    sched = BackgroundScheduler(db_path)

    executed = []
    sched.register_handler("test_job", lambda: executed.append(True))
    sched.start()

    res = sched.trigger_job_now("test_job")
    assert res["status"] == "TRIGGERED"

    import time
    time.sleep(0.5)
    assert len(executed) == 1

    sched.stop()
