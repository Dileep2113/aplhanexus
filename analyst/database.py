"""SQLite database persistence for the Autonomous AI Trading Platform.

Maintains relational tables for:
  - predictions (immutable audit log of all trade signals)
  - prediction_outcomes (evening evaluations with MFE/MAE, statutory costs, and ambiguity tracking)
  - morning_reports (pre-market global macro and scenario briefings)
  - evening_reports (post-market daily performance reviews)
  - strategy_versions (versioned parameters and promotion history)
  - learning_experiments (self-learning hypotheses and walk-forward validation)
  - stock_research (cached deep fundamental and valuation research per symbol)
  - scheduler_jobs (job execution history and status)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "analyst_data.db"


@contextmanager
def get_db_connection(db_path: Path | str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite connection with foreign keys and dict-like rows."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Initialize all database tables and indexes if they do not exist."""
    with get_db_connection(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            parent_prediction_id TEXT,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL DEFAULT 'NSE',
            timestamp TEXT NOT NULL,
            session_date TEXT NOT NULL,
            direction TEXT NOT NULL,
            setup_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_zone_low REAL NOT NULL,
            entry_zone_high REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target1 REAL NOT NULL,
            target2 REAL NOT NULL,
            holding_period TEXT NOT NULL DEFAULT 'INTRADAY',
            market_regime TEXT,
            technical_indicators_json TEXT,
            rvol REAL,
            vwap_relation TEXT,
            sector_signal TEXT,
            global_signal TEXT,
            catalysts_json TEXT,
            fundamental_context_json TEXT,
            strategy_version TEXT NOT NULL DEFAULT 'v1.0.0',
            scoring_weights_json TEXT,
            confidence_score REAL,
            calibrated_confidence REAL,
            cost_estimate_json TEXT,
            selection_reasons TEXT,
            failure_reasons TEXT,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            created_at TEXT NOT NULL,
            FOREIGN KEY (parent_prediction_id) REFERENCES predictions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_pred_session_date ON predictions(session_date);
        CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol);
        CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);

        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            prediction_id TEXT PRIMARY KEY,
            evaluated_at TEXT NOT NULL,
            outcome_status TEXT NOT NULL,
            triggered INTEGER NOT NULL DEFAULT 0,
            trigger_time TEXT,
            exit_time TEXT,
            exit_price REAL,
            mfe REAL,
            mae REAL,
            mfe_pct REAL,
            mae_pct REAL,
            first_hit TEXT,
            gross_pnl REAL,
            gross_pnl_pct REAL,
            statutory_costs REAL,
            net_pnl REAL,
            net_pnl_pct REAL,
            r_multiple REAL,
            failure_category TEXT,
            evaluation_notes TEXT,
            raw_data_json TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_outcomes_status ON prediction_outcomes(outcome_status);
        CREATE INDEX IF NOT EXISTS idx_outcomes_failure ON prediction_outcomes(failure_category);

        CREATE TABLE IF NOT EXISTS morning_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            market_regime TEXT,
            global_risk_assessment TEXT,
            gift_nifty_sentiment TEXT,
            scenarios_json TEXT,
            strong_sectors_json TEXT,
            weak_sectors_json TEXT,
            ranked_longs_json TEXT,
            ranked_shorts_json TEXT,
            watchlist_json TEXT,
            no_trade_recommended INTEGER NOT NULL DEFAULT 0,
            html_path TEXT,
            checksum TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_morning_date ON morning_reports(report_date);

        CREATE TABLE IF NOT EXISTS evening_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            total_predictions INTEGER NOT NULL DEFAULT 0,
            triggered_trades INTEGER NOT NULL DEFAULT 0,
            win_count INTEGER NOT NULL DEFAULT 0,
            loss_count INTEGER NOT NULL DEFAULT 0,
            ambiguous_count INTEGER NOT NULL DEFAULT 0,
            win_rate_pct REAL,
            profit_factor REAL,
            net_pnl_sum REAL,
            avg_r_multiple REAL,
            best_trade_json TEXT,
            worst_trade_json TEXT,
            factor_attribution_json TEXT,
            failure_summary_json TEXT,
            html_path TEXT,
            checksum TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_evening_date ON evening_reports(report_date);

        CREATE TABLE IF NOT EXISTS strategy_versions (
            version TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            validation_metrics_json TEXT,
            is_active INTEGER NOT NULL DEFAULT 0,
            promoted_at TEXT,
            rolled_back_at TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS learning_experiments (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source_strategy_version TEXT NOT NULL,
            candidate_strategy_version TEXT NOT NULL,
            proposed_parameters_json TEXT NOT NULL,
            error_patterns_identified_json TEXT,
            walk_forward_metrics_json TEXT,
            decision TEXT NOT NULL,
            rejection_reason TEXT,
            FOREIGN KEY (source_strategy_version) REFERENCES strategy_versions(version)
        );

        CREATE TABLE IF NOT EXISTS stock_research (
            symbol TEXT PRIMARY KEY,
            company_name TEXT,
            sector TEXT,
            industry TEXT,
            updated_at TEXT NOT NULL,
            business_model_summary TEXT,
            segments_json TEXT,
            management_governance_json TEXT,
            financial_metrics_json TEXT,
            valuation_models_json TEXT,
            catalysts_json TEXT,
            short_term_outlook TEXT,
            long_term_outlook TEXT,
            raw_json TEXT,
            checksum TEXT
        );

        CREATE TABLE IF NOT EXISTS scheduler_jobs (
            job_name TEXT PRIMARY KEY,
            last_run TEXT,
            next_run TEXT,
            last_status TEXT,
            last_duration_seconds REAL,
            last_message TEXT
        );
        """)

        # Ensure default baseline strategy v1.0.0 exists
        row = conn.execute("SELECT version FROM strategy_versions WHERE version = 'v1.0.0'").fetchone()
        if not row:
            default_params = {
                "min_intraday_score": 55,
                "preferred_rvol": 1.3,
                "min_rvol": 1.0,
                "rsi_long_min": 50,
                "rsi_long_max": 68,
                "rsi_short_min": 32,
                "rsi_short_max": 50,
                "min_rr_ratio": 1.5,
                "mtf_min_agreement": 6,
                "max_chase_atr_pct": 0.35,
                "max_vix_threshold": 22.0,
            }
            conn.execute("""
                INSERT INTO strategy_versions (version, created_at, parameters_json, is_active, notes)
                VALUES ('v1.0.0', ?, ?, 1, 'Initial baseline institutional strategy')
            """, (datetime.now().isoformat(), json.dumps(default_params)))


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully at:", DEFAULT_DB_PATH)
