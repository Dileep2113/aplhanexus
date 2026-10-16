"""Persistent Background Automation Scheduler.

Runs scheduled jobs in a background daemon thread independently of active browser tabs:
  - Morning Analysis: 8:15 AM IST (pre-market briefing)
  - Intraday Opportunity Scan: Every 30 minutes during trading hours (9:30 AM - 3:00 PM IST)
  - Evening Review: 4:30 PM IST (post-market settlement & outcome audit)
  - Strategy Learning Cycle: 6:00 PM IST (walk-forward optimization & promotion)
  - Retention Reaper: 7:00 PM IST (365-day storage cleanup)

Maintains execution health and timestamps in SQLite scheduler_jobs table.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytz

from analyst.database import DEFAULT_DB_PATH, get_db_connection

logger = logging.getLogger("AnalystScheduler")
IST = pytz.timezone("Asia/Kolkata")


class BackgroundScheduler:
    """Manages scheduled background trading research jobs."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable[[], Any]] = {}
        self._last_executed_day: Dict[str, str] = {}
        self._load_last_execution_state()

    def _load_last_execution_state(self) -> None:
        """Hydrate last execution state from database across service restarts/wakeups."""
        try:
            with get_db_connection(self.db_path) as conn:
                rows = conn.execute("SELECT job_name, last_run FROM scheduler_jobs WHERE last_run IS NOT NULL").fetchall()
                for r in rows:
                    if r["last_run"]:
                        self._last_executed_day[r["job_name"]] = r["last_run"][:10]
        except Exception as e:
            logger.warning(f"Could not load past scheduler state: {e}")

    def is_alive(self) -> bool:
        """Check if background scheduler thread is running."""
        return bool(self._thread and self._thread.is_alive())

    def register_handler(self, job_name: str, handler: Callable[[], Any]) -> None:
        """Register a callback for a specific job."""
        self._handlers[job_name] = handler

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="AnalystSchedulerThread")
        self._thread.start()
        logger.info("Background scheduler thread started.")

    def stop(self) -> None:
        """Stop the background scheduler thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Background scheduler stopped.")

    def trigger_job_now(self, job_name: str) -> Dict[str, Any]:
        """Manually trigger a job immediately in a worker thread."""
        handler = self._handlers.get(job_name)
        if not handler:
            return {"status": "ERROR", "message": f"No handler registered for '{job_name}'"}

        def _worker():
            self._execute_job(job_name, handler)

        t = threading.Thread(target=_worker, daemon=True, name=f"ManualJob-{job_name}")
        t.start()
        return {"status": "TRIGGERED", "job_name": job_name, "message": "Job initiated in background."}

    def get_job_statuses(self) -> Dict[str, Any]:
        """Query current status and computed next runs of all scheduled jobs."""
        now_ist = datetime.now(IST)
        next_runs = self._compute_next_runs(now_ist)
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM scheduler_jobs").fetchall()
            row_map = {r["job_name"]: dict(r) for r in rows}

        jobs = {}
        job_catalog = [
            ("morning_report", "Morning Pre-Market Briefing", "08:15 IST (Weekdays)"),
            ("intraday_scan", "Intraday Universe Scan & Signals", "Every 30m (09:30-15:00 IST)"),
            ("evening_review", "Evening Outcome Audit & P&L", "16:30 IST (Weekdays)"),
            ("learning_cycle", "Strategy Self-Learning & Tuning", "18:00 IST (Weekdays)"),
            ("retention_reaper", "365-Day Retention Reaper", "19:00 IST (Daily)"),
        ]
        for j_id, j_name, j_sched in job_catalog:
            db_entry = row_map.get(j_id, {})
            jobs[j_id] = {
                "job_name": j_name,
                "schedule": j_sched,
                "last_run": db_entry.get("last_run"),
                "next_run": next_runs.get(j_id),
                "last_status": db_entry.get("last_status", "IDLE"),
                "last_duration_seconds": db_entry.get("last_duration_seconds"),
                "last_message": db_entry.get("last_message", "Scheduled and ready"),
            }
        return jobs

    def _compute_next_runs(self, now: datetime) -> Dict[str, str]:
        """Calculate the next estimated execution timestamp in IST for each job."""
        from datetime import timedelta
        res = {}
        # Morning report: next weekday 08:15
        d = now.date()
        target = IST.localize(datetime(d.year, d.month, d.day, 8, 15))
        if now >= target or now.weekday() >= 5:
            days_ahead = 1
            while (now + timedelta(days=days_ahead)).weekday() >= 5:
                days_ahead += 1
            d_next = (now + timedelta(days=days_ahead)).date()
            target = IST.localize(datetime(d_next.year, d_next.month, d_next.day, 8, 15))
        res["morning_report"] = target.strftime("%Y-%m-%d %H:%M IST")

        # Intraday scan
        if now.weekday() < 5 and 9 <= now.hour < 15:
            mins = 30 - (now.minute % 30)
            t_scan = now + timedelta(minutes=mins)
            res["intraday_scan"] = t_scan.strftime("%Y-%m-%d %H:%M IST")
        else:
            res["intraday_scan"] = (target.replace(hour=9, minute=30)).strftime("%Y-%m-%d %H:%M IST")

        # Evening review: 16:30
        target_eve = IST.localize(datetime(d.year, d.month, d.day, 16, 30))
        if now >= target_eve or now.weekday() >= 5:
            days_ahead = 1
            while (now + timedelta(days=days_ahead)).weekday() >= 5:
                days_ahead += 1
            d_next = (now + timedelta(days=days_ahead)).date()
            target_eve = IST.localize(datetime(d_next.year, d_next.month, d_next.day, 16, 30))
        res["evening_review"] = target_eve.strftime("%Y-%m-%d %H:%M IST")

        # Learning cycle: 18:00
        target_lrn = IST.localize(datetime(d.year, d.month, d.day, 18, 0))
        if now >= target_lrn or now.weekday() >= 5:
            days_ahead = 1
            while (now + timedelta(days=days_ahead)).weekday() >= 5:
                days_ahead += 1
            d_next = (now + timedelta(days=days_ahead)).date()
            target_lrn = IST.localize(datetime(d_next.year, d_next.month, d_next.day, 18, 0))
        res["learning_cycle"] = target_lrn.strftime("%Y-%m-%d %H:%M IST")

        # Retention reaper: 19:00 daily
        target_rep = IST.localize(datetime(d.year, d.month, d.day, 19, 0))
        if now >= target_rep:
            target_rep = target_rep + timedelta(days=1)
        res["retention_reaper"] = target_rep.strftime("%Y-%m-%d %H:%M IST")

        return res

    def _scheduler_loop(self) -> None:
        """Main periodic poll loop."""
        while not self._stop_event.is_set():
            try:
                now_ist = datetime.now(IST)
                today_str = now_ist.strftime("%Y-%m-%d")
                weekday = now_ist.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
                is_weekday = weekday < 5
                hour, minute = now_ist.hour, now_ist.minute

                # 1. Morning Report: 8:15 AM IST (Weekdays) - with catch-up recovery upon server wake/restart
                if is_weekday and ((hour == 8 and minute >= 15) or (8 < hour < 15)):
                    if self._last_executed_day.get("morning_report") != today_str:
                        logger.info(f"Triggering morning report for {today_str} (catch-up / scheduled at {hour:02d}:{minute:02d} IST)")
                        self._last_executed_day["morning_report"] = today_str
                        self._run_job_if_registered("morning_report")

                # 2. Intraday Scan: Every 30 mins between 9:30 AM and 3:00 PM IST (Weekdays)
                if is_weekday and (9 <= hour < 15):
                    if (hour == 9 and minute >= 30) or (hour > 9):
                        slot_key = f"{today_str}_{hour}_{minute // 30}"
                        if self._last_executed_day.get("intraday_scan") != slot_key:
                            self._last_executed_day["intraday_scan"] = slot_key
                            self._run_job_if_registered("intraday_scan")

                # 3. Evening Review: 4:30 PM IST (Weekdays) - with catch-up recovery upon server wake/restart
                if is_weekday and ((hour == 16 and minute >= 30) or (hour > 16)):
                    if self._last_executed_day.get("evening_review") != today_str:
                        logger.info(f"Triggering evening review for {today_str} (catch-up / scheduled at {hour:02d}:{minute:02d} IST)")
                        self._last_executed_day["evening_review"] = today_str
                        self._run_job_if_registered("evening_review")

                # 4. Learning Cycle: 6:00 PM IST (Weekdays)
                if is_weekday and ((hour == 18 and minute >= 0) or (hour > 18)):
                    if self._last_executed_day.get("learning_cycle") != today_str:
                        self._last_executed_day["learning_cycle"] = today_str
                        self._run_job_if_registered("learning_cycle")

                # 5. Retention Reaper: 7:00 PM IST (Daily)
                if hour >= 19:
                    if self._last_executed_day.get("retention_reaper") != today_str:
                        self._last_executed_day["retention_reaper"] = today_str
                        self._run_job_if_registered("retention_reaper")

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

            # Sleep 15 seconds before next check
            self._stop_event.wait(15.0)

    def _run_job_if_registered(self, job_name: str) -> None:
        handler = self._handlers.get(job_name)
        if handler:
            t = threading.Thread(target=self._execute_job, args=(job_name, handler), daemon=True)
            t.start()

    def _execute_job(self, job_name: str, handler: Callable[[], Any]) -> None:
        start_time = time.time()
        start_iso = datetime.now(IST).isoformat()
        status = "SUCCESS"
        msg = "Job completed successfully."

        try:
            handler()
        except Exception as e:
            status = "ERROR"
            msg = str(e)
            logger.error(f"Error executing job '{job_name}': {e}", exc_info=True)

        duration = round(time.time() - start_time, 2)
        with get_db_connection(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scheduler_jobs (
                    job_name, last_run, next_run, last_status,
                    last_duration_seconds, last_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (job_name, start_iso, None, status, duration, msg))
