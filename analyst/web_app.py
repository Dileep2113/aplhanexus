"""AlphaNexus - Institutional Quantitative Intelligence & Execution Platform.

High-fidelity web interface and background automation server featuring:
  - Industry-standard Dark Slate / Zinc palette (Bloomberg & Linear design system)
  - Dedicated persistent sidebar navigation with category groupings
  - Separated, purpose-built data tables for Intraday and Long-Term strategies
  - Pre-Market Probabilistic Scenarios, Deep Stock Research, and Post-Market Audits
  - Auto-running background scheduler daemon with live execution controls
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from analyst.database import DEFAULT_DB_PATH, get_db_connection
from analyst.evening_review import EveningReviewer
from analyst.global_macro import fetch_global_snapshot
from analyst.learning_engine import LearningEngine
from analyst.predictions import PredictionRegistry
from analyst.premarket import generate_premarket_report
from analyst.scan_engine import execute_research_scan
from analyst.scheduler import BackgroundScheduler
from analyst.stock_research import StockResearcher
from analyst.storage import StorageManager

ROOT_DIR = Path(__file__).resolve().parent.parent


class AnalystWebHandler(BaseHTTPRequestHandler):
    """AlphaNexus REST API & Institutional Web Application Server."""

    db_path = DEFAULT_DB_PATH
    storage = StorageManager()
    registry = PredictionRegistry(db_path)
    reviewer = EveningReviewer(db_path, storage)
    learner = LearningEngine(db_path, storage)
    researcher = StockResearcher(db_path, storage)
    scheduler: Optional[BackgroundScheduler] = None
    _active_scan_status: Dict[str, Any] = {"status": "IDLE", "message": "Scanner ready"}
    _server_start_time: datetime = datetime.now()

    def log_message(self, format, *args):
        pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path in ("/", "/index.html"):
                self._serve_html_dashboard()
            elif path in ("/health", "/api/health"):
                self._serve_health()
            elif path == "/report.html":
                self._serve_file(ROOT_DIR / "report.html", "text/html")
            elif path == "/api/status":
                self._serve_json(self._get_system_status())
            elif path == "/api/scan/latest":
                self._serve_json(self._get_latest_scan_data())
            elif path == "/api/scan/status":
                self._serve_json(self._active_scan_status)
            elif path == "/api/predictions":
                date_filter = query.get("date", [None])[0]
                preds = self.registry.list_for_date(date_filter) if date_filter else self.registry.list_all(100)
                self._serve_json([p.__dict__ for p in preds])
            elif path == "/api/morning":
                self._serve_json(self._get_latest_morning_report())
            elif path == "/api/evening":
                date_filter = query.get("date", [None])[0]
                report = self.reviewer.evaluate_date(date_filter)
                self._serve_json(report.__dict__)
            elif path == "/api/research":
                sym = query.get("symbol", ["RELIANCE.NS"])[0]
                dossier = self.researcher.research_symbol(sym)
                self._serve_json(dossier.__dict__)
            elif path == "/api/learning":
                versions = self.learner.list_strategy_versions()
                self._serve_json({"versions": versions})
            elif path == "/api/scheduler":
                statuses = self.scheduler.get_job_statuses() if self.scheduler else {}
                self._serve_json({"jobs": statuses})
            elif path == "/api/reports/list":
                self._serve_json(self._get_report_archive())
            elif path == "/api/reports/view":
                file_rel = query.get("file", [""])[0]
                safe_file = (ROOT_DIR / file_rel).resolve()
                if ROOT_DIR in safe_file.parents and safe_file.exists():
                    self._serve_file(safe_file, "text/html")
                else:
                    self.send_error(404, "Report file not found")
            else:
                self.send_error(404, "Endpoint Not Found")
        except Exception as e:
            self._serve_json({"error": str(e)}, status=500)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/api/scan/trigger":
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                params = json.loads(body) if body.strip() else {}

                universe = params.get("universe", query.get("universe", ["nifty50"])[0])
                limit = int(params.get("limit", query.get("limit", [0])[0]))
                min_score = int(params.get("min_score", query.get("min_score", [55])[0]))
                fast = bool(params.get("fast", query.get("fast", [True])[0]))

                self._trigger_background_scan(universe, limit, min_score, fast)
                self._serve_json({"status": "TRIGGERED", "message": f"Universe scan initiated for {universe.upper()}."})
            elif path == "/api/scheduler/trigger":
                job_name = query.get("job", [""])[0]
                if self.scheduler and job_name:
                    res = self.scheduler.trigger_job_now(job_name)
                    self._serve_json(res)
                else:
                    self._serve_json({"error": "Scheduler or job name missing"}, status=400)
            elif path == "/api/learning/run":
                exp = self.learner.run_learning_cycle()
                self._serve_json(exp.__dict__)
            elif path == "/api/learning/rollback":
                target = query.get("version", [""])[0]
                success = self.learner.rollback_to_version(target)
                self._serve_json({"success": success, "target_version": target})
            else:
                self.send_error(404, "Endpoint Not Found")
        except Exception as e:
            self._serve_json({"error": str(e)}, status=500)

    def _serve_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _serve_health(self) -> None:
        db_ok = False
        try:
            with get_db_connection(self.db_path) as conn:
                conn.execute("SELECT 1").fetchone()
                db_ok = True
        except Exception:
            pass

        uptime_s = round((datetime.now() - self._server_start_time).total_seconds(), 1)
        sched_ok = self.scheduler is not None and self.scheduler.is_alive()

        scan_cache_file = ROOT_DIR / "_latest_scan.json"
        data_freshness = "FRESH"
        data_lag_s = None
        if scan_cache_file.exists():
            data_lag_s = round(time.time() - scan_cache_file.stat().st_mtime, 1)
            if data_lag_s > 3600:
                data_freshness = "STALE"
        else:
            data_freshness = "NOT_INITIALIZED"

        status_code = 200 if db_ok else 503
        self._serve_json({
            "status": "HEALTHY" if db_ok else "DEGRADED",
            "platform": "AlphaNexus Institutional Platform",
            "version": "v1.0.0",
            "uptime_seconds": uptime_s,
            "database": "CONNECTED" if db_ok else "DISCONNECTED",
            "scheduler": "ACTIVE" if sched_ok else "INACTIVE",
            "broker_execution": "DISABLED_OBSERVER_ONLY",
            "data_freshness": data_freshness,
            "data_lag_seconds": data_lag_s,
            "timestamp": datetime.now().isoformat(),
        }, status=status_code)

    def _get_system_status(self) -> Dict[str, Any]:
        active_ver, params = self.learner.get_active_strategy()
        job_stats = self.scheduler.get_job_statuses() if self.scheduler else {}
        return {
            "platform": "AlphaNexus Institutional Platform",
            "active_strategy_version": active_ver,
            "active_parameters": params,
            "scheduler_jobs": job_stats,
            "scan_status": self._active_scan_status,
            "database_path": str(self.db_path),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_latest_scan_data(self) -> Dict[str, Any]:
        scan_cache_file = ROOT_DIR / "_latest_scan.json"
        if scan_cache_file.exists():
            try:
                return json.loads(scan_cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"timestamp": datetime.now().isoformat(), "macro": {}, "intraday": [], "longterm": []}

    def _get_latest_morning_report(self) -> Dict[str, Any]:
        with get_db_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM morning_reports ORDER BY report_date DESC LIMIT 1").fetchone()
            if row:
                return dict(row)
        macro = fetch_global_snapshot()
        pm = generate_premarket_report(macro)
        return {
            "report_date": pm.generated_at.strftime("%Y-%m-%d"),
            "market_regime": pm.regime,
            "global_risk_assessment": pm.gap_type,
            "gift_nifty_sentiment": pm.gap_type,
            "scenarios_json": json.dumps([{"name": s.name, "bias": s.probability_bias, "range": s.opening_range_nifty, "plan": s.actionable_intraday_plan} for s in pm.scenarios]),
            "no_trade_recommended": 0,
        }

    def _get_report_archive(self) -> List[Dict[str, Any]]:
        archive = []
        main_rep = ROOT_DIR / "report.html"
        if main_rep.exists():
            archive.append({
                "name": "Full Universe Dual-Table Research Report",
                "type": "SCAN",
                "date": datetime.fromtimestamp(main_rep.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "size_kb": round(main_rep.stat().st_size / 1024, 1),
                "url": "/report.html",
            })
        storage_root = ROOT_DIR / "storage"
        if storage_root.exists():
            for p in sorted(storage_root.rglob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
                rel = p.relative_to(ROOT_DIR)
                rep_type = "RESEARCH" if "research" in str(rel) else ("MORNING" if "morning" in str(rel) else "EVENING")
                archive.append({
                    "name": p.stem.replace("_", " ").title() + f" ({p.parent.name})",
                    "type": rep_type,
                    "date": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "url": f"/api/reports/view?file={str(rel).replace(chr(92), '/')}",
                })
        return archive

    def _trigger_background_scan(self, universe: str, limit: int, min_score: int, fast: bool) -> None:
        def _scan_worker():
            AnalystWebHandler._active_scan_status = {
                "status": "RUNNING",
                "message": f"Scanning {universe.upper()} universe...",
                "started_at": datetime.now().isoformat(),
            }
            try:
                res = execute_research_scan(
                    universe_tier=universe,
                    limit=limit,
                    min_score=min_score,
                    fast_mode=fast,
                    progress_callback=lambda m: AnalystWebHandler._active_scan_status.update({"message": m}),
                )
                AnalystWebHandler._active_scan_status = {
                    "status": "SUCCESS",
                    "message": f"Scan completed: {res.get('total_scanned')} stocks evaluated, {res.get('qualified_setups')} qualified setups.",
                    "completed_at": datetime.now().isoformat(),
                }
            except Exception as e:
                AnalystWebHandler._active_scan_status = {
                    "status": "ERROR",
                    "message": f"Scan error: {e}",
                    "failed_at": datetime.now().isoformat(),
                }

        t = threading.Thread(target=_scan_worker, daemon=True, name="WebScannerThread")
        t.start()

    def _serve_html_dashboard(self) -> None:
        html = self._render_full_dashboard()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _render_full_dashboard(self) -> str:
        scan_data = self._get_latest_scan_data()
        macro = scan_data.get("macro", {})
        intraday = scan_data.get("intraday", [])
        longterm = scan_data.get("longterm", [])
        try:
            active_ver, _ = self.learner.get_active_strategy()
        except Exception:
            active_ver = "v1.0.0"

        # Intraday Table Rows
        intraday_rows = "".join([
            f"""<tr class="table-row" data-direction="{o.get('direction')}" data-tier="{o.get('tier')}">
                <td>
                    <div style="font-weight:700; color:#0f172a; font-size:0.95rem;">{o.get('symbol')}</div>
                    <div style="font-size:0.75rem; color:#64748b; max-width:180px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{o.get('name')}</div>
                </td>
                <td>
                    <span class="pill {'pill-long' if o.get('direction') == 'LONG' else 'pill-short'}">
                        {'LONG (BUY)' if o.get('direction') == 'LONG' else 'SHORT (SELL)'}
                    </span>
                </td>
                <td><span class="pill pill-neutral">{o.get('setup_type')}</span></td>
                <td>
                    <div style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#0f172a;">₹{o.get('trigger', 0):,.2f}</div>
                    <div style="font-size:0.72rem; color:#64748b;">Zone: ₹{o.get('entry_low', 0):,.1f}–{o.get('entry_high', 0):,.1f}</div>
                </td>
                <td>
                    <div style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#dc2626;">₹{o.get('stop_loss', 0):,.2f}</div>
                </td>
                <td>
                    <div style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#059669;">T1: ₹{o.get('target1', 0):,.2f}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#047857;">T2: ₹{o.get('target2', 0):,.2f}</div>
                </td>
                <td>
                    <div style="font-family:'JetBrains Mono',monospace; font-weight:700; color:#2563eb;">{o.get('net_rr', 2.5):.2f}x</div>
                    <div style="font-size:0.7rem; color:#64748b;">Net of costs</div>
                </td>
                <td>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-weight:700; font-size:0.95rem; color:#0f172a;">{o.get('score', 0):.0f}</span>
                        <div style="width:48px; background:#e2e8f0; height:6px; border-radius:3px; overflow:hidden;">
                            <div style="width:{min(100, o.get('score',0))}%; background:{'#059669' if o.get('score',0)>=70 else '#2563eb'}; height:100%;"></div>
                        </div>
                    </div>
                </td>
                <td>
                    <span class="pill {'pill-strong' if o.get('tier')=='STRONG' else 'pill-watchlist'}">{o.get('tier')}</span>
                </td>
                <td>
                    <button class="btn-action" onclick='showThesisModal({json.dumps(o)})'>Details</button>
                </td>
            </tr>"""
            for o in intraday
        ])

        # Long-Term Table Rows
        longterm_rows = "".join([
            f"""<tr class="table-row-lt" data-verdict="{lt.get('verdict')}">
                <td>
                    <div style="font-weight:700; color:#0f172a; font-size:0.95rem;">{lt.get('symbol')}</div>
                    <div style="font-size:0.75rem; color:#64748b;">{lt.get('name')}</div>
                </td>
                <td>
                    <div style="font-weight:600; font-size:0.85rem; color:#334155;">{lt.get('sector')}</div>
                    <div style="font-size:0.72rem; color:#64748b;">{lt.get('industry')}</div>
                </td>
                <td style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#0f172a;">₹{lt.get('ltp', 0):,.2f}</td>
                <td style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#334155;">₹{lt.get('market_cap_cr', 0):,.0f} Cr</td>
                <td style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#334155;">{lt.get('pe_ratio', 0):.1f}x</td>
                <td style="font-family:'JetBrains Mono',monospace; color:#059669; font-weight:600;">{f"{lt.get('roe'):.1f}%" if lt.get('roe') else '—'}</td>
                <td style="font-family:'JetBrains Mono',monospace; color:#2563eb; font-weight:600;">{f"{lt.get('rev_growth'):+.1f}%" if lt.get('rev_growth') else '—'}</td>
                <td>
                    <span class="pill {'pill-strong' if 'STRONG' in lt.get('verdict','') else ('pill-long' if 'ACCUMULATE' in lt.get('verdict','') else 'pill-neutral')}">
                        {lt.get('verdict')}
                    </span>
                </td>
                <td>
                    <div style="font-size:0.78rem; font-family:'JetBrains Mono',monospace;">
                        <span style="color:#dc2626;">₹{lt.get('valuation_scenarios', {}).get('Bear_Case', 0):,.0f}</span> |
                        <span style="color:#2563eb; font-weight:600;">₹{lt.get('valuation_scenarios', {}).get('Base_Case', 0):,.0f}</span> |
                        <span style="color:#059669;">₹{lt.get('valuation_scenarios', {}).get('Bull_Case', 0):,.0f}</span>
                    </div>
                    <div style="font-size:0.7rem; color:#64748b;">Bear / Base / Bull</div>
                </td>
                <td style="font-weight:700; font-size:0.95rem; color:#0f172a;">{lt.get('score')}</td>
            </tr>"""
            for lt in longterm
        ])

        # Pre-Market Scenario Cards
        scenarios = macro.get("scenarios", [])
        scenario_cards = "".join([
            f"""<div class="scenario-card" style="border-left: 4px solid {'#059669' if 'BULL' in sc.get('name','').upper() else ('#dc2626' if 'BEAR' in sc.get('name','').upper() else '#d97706')};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="font-weight:700; font-size:0.95rem; color:#0f172a;">{sc.get('name')}</div>
                    <span class="pill {'pill-strong' if sc.get('bias')=='HIGH' else 'pill-neutral'}">{sc.get('bias')} BIAS</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.05rem; color:#2563eb; font-weight:600; margin-bottom:6px;">
                    Expected Zone: {sc.get('range')}
                </div>
                <div style="font-size:0.84rem; color:#334155; line-height:1.5;">
                    <b style="color:#0f172a;">Gameplan:</b> {sc.get('plan')}
                </div>
            </div>"""
            for sc in scenarios
        ])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>AlphaNexus - Institutional Quantitative Suite</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #f8fafc;
            --bg-sidebar: #ffffff;
            --bg-card: #ffffff;
            --bg-card-hover: #f8fafc;
            --border: #e2e8f0;
            --border-subtle: #f1f5f9;
            --text-main: #0f172a;
            --text-secondary: #334155;
            --text-dimmed: #64748b;
            --primary: #2563eb;
            --primary-tint: #eff6ff;
            --green: #059669;
            --green-tint: #ecfdf5;
            --green-border: #a7f3d0;
            --red: #dc2626;
            --red-tint: #fef2f2;
            --red-border: #fecaca;
            --amber: #d97706;
            --amber-tint: #fffbeb;
            --amber-border: #fde68a;
            --indigo: #4f46e5;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-base);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
            font-size: 0.88rem;
        }}
        /* Left Sidebar & Mobile Drawer */
        .sidebar {{
            width: 250px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            box-shadow: 1px 0 2px rgba(0, 0, 0, 0.02);
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .brand {{
            padding: 18px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .brand-badge {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            color: #ffffff;
            font-size: 0.95rem;
            letter-spacing: -0.5px;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
            flex-shrink: 0;
        }}
        .brand-name {{
            font-weight: 700;
            font-size: 1.05rem;
            letter-spacing: -0.3px;
            color: var(--text-main);
        }}
        .brand-sub {{
            font-size: 0.68rem;
            color: var(--text-dimmed);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 600;
        }}
        .mobile-close-btn {{
            display: none;
            background: transparent;
            border: none;
            color: var(--text-dimmed);
            font-size: 1.3rem;
            cursor: pointer;
            margin-left: auto;
            padding: 4px 8px;
            line-height: 1;
            border-radius: 4px;
        }}
        .sidebar-backdrop {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(3px);
            z-index: 150;
            opacity: 0;
            transition: opacity 0.25s ease;
            pointer-events: none;
        }}
        .sidebar-backdrop.show {{
            display: block;
            opacity: 1;
            pointer-events: auto;
        }}
        .nav-menu {{
            padding: 16px 10px;
            display: flex;
            flex-direction: column;
            gap: 3px;
            flex: 1;
            overflow-y: auto;
        }}
        .nav-category {{
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            padding: 12px 12px 6px 12px;
        }}
        .nav-btn {{
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid transparent;
            padding: 9px 12px;
            border-radius: 6px;
            font-size: 0.84rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            text-align: left;
            transition: all 0.15s;
        }}
        .nav-btn:hover {{
            background: #f1f5f9;
            color: var(--text-main);
        }}
        .nav-btn.active {{
            background: #eff6ff;
            color: #1d4ed8;
            font-weight: 600;
            border-color: #bfdbfe;
        }}
        .nav-count {{
            margin-left: auto;
            background: #f1f5f9;
            color: var(--text-secondary);
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 10px;
            border: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }}
        /* Main Viewport */
        .main-viewport {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            background: var(--bg-base);
            width: 100%;
            min-width: 0;
        }}
        /* Header Topbar */
        .topbar {{
            background: #ffffff;
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            gap: 12px;
        }}
        .mobile-toggle-btn {{
            display: none;
            background: #ffffff;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 9px;
            border-radius: 6px;
            cursor: pointer;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        }}
        .mobile-brand-title {{
            display: none;
            align-items: center;
            gap: 8px;
        }}
        .topbar-actions {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }}
        .ticker-strip {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .ticker-cell {{
            background: #f8fafc;
            border: 1px solid var(--border);
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.78rem;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }}
        .ticker-lbl {{ color: var(--text-dimmed); font-weight: 600; }}
        .ticker-num {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text-main); }}
        /* Buttons */
        .btn {{
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
            padding: 7px 14px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            white-space: nowrap;
        }}
        .btn:hover {{
            border-color: #94a3b8;
            background: #f8fafc;
            color: #0f172a;
        }}
        .btn-primary {{
            background: #2563eb;
            border-color: #1d4ed8;
            color: #ffffff;
            font-weight: 600;
            box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2);
        }}
        .btn-primary:hover {{
            background: #1d4ed8;
            border-color: #1e40af;
            color: #ffffff;
        }}
        .btn-action {{
            background: #f1f5f9;
            color: #2563eb;
            border: 1px solid #cbd5e1;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.74rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
        }}
        .btn-action:hover {{
            background: #2563eb;
            color: #ffffff;
            border-color: #1d4ed8;
        }}
        /* Content Panels */
        .panel {{ display: none; padding: 24px; }}
        .panel.active {{ display: block; }}
        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: box-shadow 0.15s, border-color 0.15s;
        }}
        .metric-card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border-color: #cbd5e1;
        }}
        .metric-card-lbl {{
            font-size: 0.72rem;
            color: var(--text-dimmed);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .metric-card-val {{
            font-size: 1.6rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }}
        /* Tables & Responsive Sticky Columns */
        .table-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            position: relative;
        }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            text-align: left;
            white-space: nowrap;
        }}
        thead th {{
            background: #f8fafc;
            color: #475569;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
        }}
        tbody td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-subtle);
            font-size: 0.84rem;
            color: var(--text-secondary);
        }}
        tbody tr:hover {{ background: var(--bg-card-hover); }}
        tbody tr:last-child td {{ border-bottom: none; }}
        /* Sticky First Column for Horizontal Table Swiping */
        .table-panel table thead th:first-child,
        .table-panel table tbody td:first-child {{
            position: sticky;
            left: 0;
            background: #ffffff;
            z-index: 2;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.04);
        }}
        .table-panel table thead th:first-child {{
            background: #f8fafc;
            z-index: 3;
        }}
        .table-panel table tbody tr:hover td:first-child {{
            background: var(--bg-card-hover);
        }}
        /* Pills & Badges */
        .pill {{
            padding: 3px 9px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            display: inline-block;
            font-family: 'Inter', sans-serif;
            white-space: nowrap;
        }}
        .pill-long {{ background: var(--green-tint); color: #047857; border: 1px solid var(--green-border); }}
        .pill-short {{ background: var(--red-tint); color: #b91c1c; border: 1px solid var(--red-border); }}
        .pill-strong {{ background: var(--primary-tint); color: #1d4ed8; border: 1px solid #bfdbfe; }}
        .pill-watchlist {{ background: var(--amber-tint); color: #b45309; border: 1px solid var(--amber-border); }}
        .pill-neutral {{ background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }}
        /* Scenario Cards */
        .scenario-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }}
        /* Modals */
        .modal-bg {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(4px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 12px;
        }}
        .modal-box {{
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            width: 92%;
            max-width: 600px;
            max-height: 88vh;
            overflow-y: auto;
            padding: 24px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
        }}
        /* Filter Controls & Horizontal Chip Container */
        .filter-chip-row {{
            display: flex;
            gap: 6px;
            overflow-x: auto;
            white-space: nowrap;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            padding-bottom: 2px;
        }}
        .filter-chip-row::-webkit-scrollbar {{ display: none; }}
        /* Bottom Navigation Bar for Mobile */
        .bottom-nav {{
            display: none;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #ffffff;
            border-top: 1px solid var(--border);
            height: 60px;
            z-index: 95;
            padding-bottom: env(safe-area-inset-bottom, 0px);
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.04);
        }}
        .bottom-nav-item {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: transparent;
            border: none;
            color: #64748b;
            padding: 6px 0;
            cursor: pointer;
            transition: color 0.15s;
            gap: 3px;
            text-decoration: none;
            font-family: inherit;
        }}
        .bottom-nav-item.active {{
            color: #2563eb;
            font-weight: 600;
        }}
        .bottom-nav-icon {{
            font-size: 1.15rem;
            line-height: 1;
        }}
        .bottom-nav-lbl {{
            font-size: 0.65rem;
            letter-spacing: -0.2px;
        }}

        /* Responsive Breakpoints */
        @media (max-width: 900px) {{
            .sidebar {{
                position: fixed;
                top: 0;
                left: 0;
                bottom: 0;
                width: 270px;
                z-index: 200;
                transform: translateX(-100%);
                box-shadow: 4px 0 24px rgba(0, 0, 0, 0.12);
            }}
            .sidebar.open {{
                transform: translateX(0);
            }}
            .mobile-close-btn {{
                display: block;
            }}
            .mobile-toggle-btn {{
                display: flex;
            }}
            .mobile-brand-title {{
                display: flex;
            }}
            .topbar {{
                padding: 10px 14px;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .ticker-strip {{
                order: 3;
                width: 100%;
                overflow-x: auto;
                white-space: nowrap;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
                padding-bottom: 2px;
            }}
            .ticker-strip::-webkit-scrollbar {{ display: none; }}
            .panel {{
                padding: 14px;
            }}
            .bottom-nav {{
                display: flex;
            }}
            .main-viewport {{
                padding-bottom: 70px;
            }}
        }}

        @media (max-width: 640px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: 8px;
                margin-bottom: 14px;
            }}
            .metric-card {{
                padding: 12px;
            }}
            .metric-card-lbl {{
                font-size: 0.68rem;
            }}
            .metric-card-val {{
                font-size: 1.25rem;
            }}
            .btn {{
                padding: 6px 10px;
                font-size: 0.78rem;
            }}
            .topbar-actions .btn {{
                padding: 5px 8px;
                font-size: 0.75rem;
            }}
            .modal-box {{
                width: 95%;
                padding: 16px;
                max-height: 85vh;
            }}
        }}
    </style>
</head>
<body>

    <!-- Backdrop for Mobile Drawer -->
    <div id="sidebar-backdrop" class="sidebar-backdrop" onclick="toggleSidebar(false)"></div>

    <!-- Left Sidebar -->
    <aside class="sidebar" id="app-sidebar">
        <div class="brand">
            <div class="brand-badge">AN</div>
            <div>
                <div class="brand-name">AlphaNexus</div>
                <div class="brand-sub">Quantitative Suite</div>
            </div>
            <button class="mobile-close-btn" onclick="toggleSidebar(false)" aria-label="Close navigation">&times;</button>
        </div>

        <div class="nav-menu">
            <div class="nav-category">Live Desk</div>
            <button class="nav-btn active" data-tab="tab-intraday" onclick="switchTab('tab-intraday')">
                <span>🎯</span> Intraday Desk
                <span class="nav-count">{len(intraday)}</span>
            </button>
            <button class="nav-btn" data-tab="tab-longterm" onclick="switchTab('tab-longterm')">
                <span>🏛️</span> Long-Term Portfolio
                <span class="nav-count">{len(longterm)}</span>
            </button>

            <div class="nav-category">Market Intelligence</div>
            <button class="nav-btn" data-tab="tab-premarket" onclick="switchTab('tab-premarket')">
                <span>🌅</span> Pre-Market Scenarios
            </button>
            <button class="nav-btn" data-tab="tab-research" onclick="switchTab('tab-research')">
                <span>🔍</span> Stock Deep Dive
            </button>

            <div class="nav-category">Quant & Governance</div>
            <button class="nav-btn" data-tab="tab-evening" onclick="switchTab('tab-evening')">
                <span>📈</span> Post-Market Audit
            </button>
            <button class="nav-btn" data-tab="tab-learning" onclick="switchTab('tab-learning')">
                <span>🧠</span> Model Self-Learning
            </button>
            <button class="nav-btn" data-tab="tab-scheduler" onclick="switchTab('tab-scheduler')">
                <span>⚙️</span> Automation & Timers
            </button>
            <button class="nav-btn" data-tab="tab-archive" onclick="switchTab('tab-archive')">
                <span>📁</span> Reports Vault
            </button>
        </div>

        <div style="padding:16px; border-top:1px solid var(--border); font-size:0.74rem; color:var(--text-dimmed); background:#ffffff;">
            <div>Strategy: <b style="color:#2563eb;">{active_ver}</b></div>
            <div style="margin-top:3px;">Scheduler: <b style="color:#059669;">DAEMON ACTIVE</b></div>
        </div>
    </aside>

    <!-- Main Viewport -->
    <div class="main-viewport">

        <!-- Top Status Bar -->
        <header class="topbar">
            <div style="display:flex; align-items:center; gap:10px;">
                <button class="mobile-toggle-btn" onclick="toggleSidebar(true)" aria-label="Open Navigation">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
                </button>
                <div class="mobile-brand-title">
                    <div class="brand-badge" style="width:28px; height:28px; font-size:0.85rem;">AN</div>
                    <div style="font-weight:700; font-size:1rem; color:var(--text-main); letter-spacing:-0.2px;">AlphaNexus</div>
                </div>
            </div>

            <div class="ticker-strip">
                <div class="ticker-cell">
                    <span class="ticker-lbl">NIFTY 50</span>
                    <span class="ticker-num">{macro.get('nifty_close', 23146.15):,.2f}</span>
                </div>
                <div class="ticker-cell">
                    <span class="ticker-lbl">GIFT GAP</span>
                    <span class="ticker-num" style="color:{'#059669' if macro.get('gift_nifty_gap',0)>=0 else '#dc2626'};">
                        {macro.get('gift_nifty_gap', 0):+,.1f} pts
                    </span>
                </div>
                <div class="ticker-cell">
                    <span class="ticker-lbl">INDIA VIX</span>
                    <span class="ticker-num" style="color:#d97706;">{macro.get('vix', 14.08):.2f}</span>
                </div>
                <div class="ticker-cell">
                    <span class="ticker-lbl">REGIME</span>
                    <span style="font-weight:600; color:#059669;">{macro.get('regime', 'RISK_ON')}</span>
                </div>
            </div>

            <div class="topbar-actions">
                <button class="btn btn-primary" onclick="showScanModal()">⚡ Scan</button>
                <button class="btn" onclick="triggerJob('morning_report')">🌅 AM</button>
                <button class="btn" onclick="triggerJob('evening_review')">📊 PM</button>
            </div>
        </header>

        <!-- PANEL 1: INTRADAY DESK -->
        <div id="tab-intraday" class="panel active">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-card-lbl">Qualified Setups</div>
                    <div class="metric-card-val" style="color:#2563eb;">{len(intraday)}</div>
                    <div style="font-size:0.75rem; color:var(--text-dimmed); margin-top:3px;">Intraday Score &ge; 55</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-lbl">High Conviction</div>
                    <div class="metric-card-val" style="color:#059669;">{sum(1 for o in intraday if o.get('tier') == 'STRONG')}</div>
                    <div style="font-size:0.75rem; color:var(--text-dimmed); margin-top:3px;">Multi-timeframe consensus</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-lbl">Avg Net R:R (After Friction)</div>
                    <div class="metric-card-val" style="color:#d97706;">
                        {(sum(o.get('net_rr', 2.5) for o in intraday) / len(intraday)):.2f}x
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-dimmed); margin-top:3px;">Net of STT, GST, brokerage & slippage</div>
                </div>
            </div>

            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; gap:10px; flex-wrap:wrap;">
                <div class="filter-chip-row">
                    <button class="btn" onclick="filterIntraday('ALL', this)" style="background:#eff6ff; border-color:#3b82f6; color:#1d4ed8; font-weight:600;">All Setups ({len(intraday)})</button>
                    <button class="btn" onclick="filterIntraday('LONG', this)">🟢 Long (Buy)</button>
                    <button class="btn" onclick="filterIntraday('SHORT', this)">🔴 Short (Sell)</button>
                    <button class="btn" onclick="filterIntraday('STRONG', this)">⭐ High Conviction Only</button>
                </div>
                <div style="flex-shrink:0;">
                    <input type="text" id="intraday-search" placeholder="Search ticker..." onkeyup="searchIntraday()" style="padding:6px 12px; font-size:0.82rem; width:160px; background:#ffffff; border:1px solid #cbd5e1; color:#0f172a; border-radius:6px;">
                </div>
            </div>

            <div class="table-panel">
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Setup Type</th>
                            <th>Trigger / Entry Zone</th>
                            <th>Stop Loss</th>
                            <th>Targets (T1 / T2)</th>
                            <th>Net R:R</th>
                            <th>Score</th>
                            <th>Tier</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="intraday-table-body">
                        {intraday_rows if intraday_rows else '<tr><td colspan="10" style="padding:30px; text-align:center; color:var(--text-dimmed);">No setups currently active. Launch a universe scan above.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- PANEL 2: LONG-TERM WEALTH PORTFOLIO -->
        <div id="tab-longterm" class="panel">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-card-lbl">Structural Candidates</div>
                    <div class="metric-card-val" style="color:#2563eb;">{len(longterm)}</div>
                    <div style="font-size:0.75rem; color:var(--text-dimmed); margin-top:3px;">10-Point Fundamental Criteria</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-lbl">Strong Buy Count</div>
                    <div class="metric-card-val" style="color:#059669;">{sum(1 for lt in longterm if 'STRONG' in lt.get('verdict',''))}</div>
                    <div style="font-size:0.75rem; color:var(--text-dimmed); margin-top:3px;">Score &ge; 80 with high cash conversion</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-lbl">Investment Horizon</div>
                    <div class="metric-card-val" style="color:#4f46e5;">6–24 M</div>
                    <div style="font-size:0.75rem; color:var(--text-dimmed); margin-top:3px;">Intrinsic Fair Value Triad</div>
                </div>
            </div>

            <div class="filter-chip-row" style="margin-bottom:12px;">
                <button class="btn" onclick="filterLongterm('ALL', this)" style="background:#eff6ff; border-color:#3b82f6; color:#1d4ed8; font-weight:600;">All Opportunities</button>
                <button class="btn" onclick="filterLongterm('STRONG BUY', this)">💎 Strong Buy</button>
                <button class="btn" onclick="filterLongterm('ACCUMULATE', this)">📈 Accumulate</button>
                <button class="btn" onclick="filterLongterm('HOLD', this)">⏸️ Hold</button>
            </div>

            <div class="table-panel">
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Sector & Industry</th>
                            <th>LTP</th>
                            <th>Market Cap</th>
                            <th>P/E</th>
                            <th>ROE %</th>
                            <th>Rev YoY</th>
                            <th>Verdict</th>
                            <th>Intrinsic Valuation Triad</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody id="longterm-table-body">
                        {longterm_rows if longterm_rows else '<tr><td colspan="10" style="padding:30px; text-align:center; color:var(--text-dimmed);">No candidates evaluated yet. Run universe scan to populate.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- PANEL 3: PRE-MARKET SCENARIOS -->
        <div id="tab-premarket" class="panel">
            <div style="margin-bottom:18px;">
                <h2 style="font-size:1.15rem; font-weight:700; color:var(--text-main);">Indian Pre-Market Execution Roadmaps</h2>
                <div style="color:var(--text-dimmed); font-size:0.82rem; margin-top:4px;">
                    Synthesizes GIFT Nifty, overnight global cues, and institutional flows into probabilistic roadmaps.
                </div>
            </div>
            <div>
                {scenario_cards if scenario_cards else '<div class="scenario-card">No pre-market scenarios calculated yet. Click Morning in the top bar to run.</div>'}
            </div>
        </div>

        <!-- PANEL 4: DEEP STOCK RESEARCH -->
        <div id="tab-research" class="panel">
            <div class="metric-card" style="margin-bottom:20px;">
                <div class="metric-card-lbl">On-Demand Institutional Equity Research</div>
                <div style="display:flex; gap:10px; margin-top:10px;">
                    <input type="text" id="research-sym-input" placeholder="Enter NSE Symbol (e.g. RELIANCE, TCS, INFY, HDFCBANK, TATAMOTORS)" style="flex:1; background:#ffffff; border:1px solid #cbd5e1; color:#0f172a; padding:9px 14px; border-radius:6px;">
                    <button class="btn btn-primary" onclick="executeResearch()">Generate Dossier</button>
                </div>
                <div style="display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; align-items:center;">
                    <span style="font-size:0.72rem; color:var(--text-dimmed); font-weight:600;">POPULAR:</span>
                    <button class="btn-action" onclick="quickResearch('RELIANCE.NS')">RELIANCE</button>
                    <button class="btn-action" onclick="quickResearch('TCS.NS')">TCS</button>
                    <button class="btn-action" onclick="quickResearch('INFY.NS')">INFY</button>
                    <button class="btn-action" onclick="quickResearch('HDFCBANK.NS')">HDFCBANK</button>
                    <button class="btn-action" onclick="quickResearch('TATAMOTORS.NS')">TATAMOTORS</button>
                </div>
            </div>
            <div id="research-results-container" style="color:var(--text-dimmed); padding:30px; text-align:center;">
                Enter any symbol above to evaluate business moats, segment revenue, management governance, cash flow conversion, Bear/Base/Bull valuation triad, and verified news.
            </div>
        </div>

        <!-- PANEL 5: POST-MARKET P&L AUDIT -->
        <div id="tab-evening" class="panel">
            <div class="metric-card" style="margin-bottom:18px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="metric-card-lbl">Post-Market Settlement & Friction Audit</div>
                        <div style="font-size:0.82rem; color:var(--text-dimmed); margin-top:3px;">
                            Compares signals against settlement data, deducting statutory taxes (STT 0.025%, GST 18%, NSE turnover, slippage).
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="loadEveningReview()">Evaluate Outcomes</button>
                </div>
            </div>
            <div id="evening-results-container"></div>
        </div>

        <!-- PANEL 6: SELF-LEARNING OPTIMIZER -->
        <div id="tab-learning" class="panel">
            <div class="metric-card" style="margin-bottom:18px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="metric-card-lbl">Evidence-Based Strategy Optimizer</div>
                        <div style="font-size:0.82rem; color:var(--text-dimmed); margin-top:3px;">
                            Diagnoses failure taxonomy, mutates parameters, and runs walk-forward validation (Sharpe &ge; 1.0, Win Rate &ge; 50%).
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="triggerLearningCycle()">Run Learning Cycle</button>
                </div>
            </div>
            <div id="learning-results-container"></div>
        </div>

        <!-- PANEL 7: SCHEDULER & AUTOMATION -->
        <div id="tab-scheduler" class="panel">
            <div class="metric-card" style="margin-bottom:18px;">
                <div class="metric-card-lbl">Autonomous Background Daemon</div>
                <div style="font-size:0.82rem; color:var(--text-dimmed); margin-top:3px;">
                    Scheduled jobs run on persistent background timers independently of open browser tabs.
                </div>
            </div>
            <div class="table-panel">
                <table>
                    <thead>
                        <tr>
                            <th>Job Description</th>
                            <th>Schedule (IST)</th>
                            <th>Next Execution</th>
                            <th>Last Status</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="scheduler-table-body">
                        <tr><td colspan="5" style="padding:20px; text-align:center;">Loading scheduled jobs...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- PANEL 8: REPORTS ARCHIVE -->
        <div id="tab-archive" class="panel">
            <div class="metric-card" style="margin-bottom:18px;">
                <div class="metric-card-lbl">Historical Research Archive</div>
                <div style="font-size:0.82rem; color:var(--text-dimmed); margin-top:3px;">
                    Saved dual-table scans, morning briefings, evening audits, and equity dossiers.
                </div>
            </div>
            <div class="table-panel">
                <table>
                    <thead>
                        <tr>
                            <th>Report Name</th>
                            <th>Type</th>
                            <th>Created Date</th>
                            <th>Size</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="archive-table-body">
                        <tr><td colspan="5" style="padding:20px; text-align:center;">Loading archive...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <!-- Details Modal -->
    <div id="thesis-modal" class="modal-bg" onclick="closeModal(event)">
        <div class="modal-box" onclick="event.stopPropagation()">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e2e8f0; padding-bottom:12px; margin-bottom:16px;">
                <h3 id="modal-symbol" style="color:#0f172a; font-size:1.15rem; font-weight:700;"></h3>
                <button class="btn-action" onclick="document.getElementById('thesis-modal').style.display='none'">✕ Close</button>
            </div>
            <div id="modal-body" style="font-size:0.86rem; line-height:1.6; color:#334155;"></div>
        </div>
    </div>

    <!-- Scan Launch Modal -->
    <div id="scan-modal" class="modal-bg" onclick="closeScanModal(event)">
        <div class="modal-box" onclick="event.stopPropagation()">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e2e8f0; padding-bottom:12px; margin-bottom:16px;">
                <h3 style="color:#0f172a; font-size:1.1rem; font-weight:700;">Launch Universe Scan</h3>
                <button class="btn-action" onclick="document.getElementById('scan-modal').style.display='none'">✕ Close</button>
            </div>
            <div style="display:flex; flex-direction:column; gap:14px;">
                <div>
                    <label style="font-size:0.75rem; color:var(--text-dimmed); font-weight:600; text-transform:uppercase;">Universe Scope</label>
                    <select id="modal-universe" style="width:100%; margin-top:6px; padding:9px; background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; border-radius:6px;">
                        <option value="nifty50">Nifty 50 (Apex Large-Caps)</option>
                        <option value="nifty100">Nifty 100</option>
                        <option value="nifty500" selected>Nifty 500 (Institutional Breadth)</option>
                        <option value="nse_all">All Listed NSE Equities (2000+)</option>
                    </select>
                </div>
                <div>
                    <label style="font-size:0.75rem; color:var(--text-dimmed); font-weight:600; text-transform:uppercase;">Minimum Score Filter</label>
                    <select id="modal-minscore" style="width:100%; margin-top:6px; padding:9px; background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; border-radius:6px;">
                        <option value="50">50 (Permissive)</option>
                        <option value="55" selected>55 (Institutional Baseline)</option>
                        <option value="65">65 (High Conviction Only)</option>
                        <option value="75">75 (Strict Apex Only)</option>
                    </select>
                </div>
                <div style="margin-top:10px;">
                    <button class="btn btn-primary" onclick="launchScanFromModal()" style="width:100%; padding:10px; justify-content:center;">
                        🚀 Launch Scan
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleSidebar(forceOpen) {{
            const sidebar = document.getElementById('app-sidebar');
            const backdrop = document.getElementById('sidebar-backdrop');
            if (!sidebar || !backdrop) return;
            const isOpen = sidebar.classList.contains('open');
            const open = forceOpen !== undefined ? forceOpen : !isOpen;
            if (open) {{
                sidebar.classList.add('open');
                backdrop.classList.add('show');
                document.body.style.overflow = 'hidden';
            }} else {{
                sidebar.classList.remove('open');
                backdrop.classList.remove('show');
                document.body.style.overflow = '';
            }}
        }}

        function switchTab(tabId) {{
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-btn, .bottom-nav-item').forEach(b => b.classList.remove('active'));
            const targetPanel = document.getElementById(tabId);
            if (targetPanel) targetPanel.classList.add('active');

            document.querySelectorAll(`[data-tab="${{tabId}}"]`).forEach(b => b.classList.add('active'));

            toggleSidebar(false);

            if (tabId === 'tab-scheduler') fetchSchedulerStatuses();
            if (tabId === 'tab-archive') loadReportsArchive();
        }}

        function filterIntraday(type, btn) {{
            document.querySelectorAll('#tab-intraday .btn').forEach(b => b.style.borderColor = 'var(--border)');
            btn.style.borderColor = '#3b82f6';
            const rows = document.querySelectorAll('#intraday-table-body .table-row');
            rows.forEach(r => {{
                const dir = r.getAttribute('data-direction');
                const tier = r.getAttribute('data-tier');
                if (type === 'ALL') r.style.display = '';
                else if (type === 'LONG' && dir === 'LONG') r.style.display = '';
                else if (type === 'SHORT' && dir === 'SHORT') r.style.display = '';
                else if (type === 'STRONG' && tier === 'STRONG') r.style.display = '';
                else r.style.display = 'none';
            }});
        }}

        function searchIntraday() {{
            const query = document.getElementById('intraday-search').value.toUpperCase();
            const rows = document.querySelectorAll('#intraday-table-body .table-row');
            rows.forEach(r => {{
                r.style.display = r.innerText.toUpperCase().includes(query) ? '' : 'none';
            }});
        }}

        function filterLongterm(verdict, btn) {{
            document.querySelectorAll('#tab-longterm .btn').forEach(b => b.style.borderColor = 'var(--border)');
            btn.style.borderColor = '#3b82f6';
            const rows = document.querySelectorAll('#longterm-table-body .table-row-lt');
            rows.forEach(r => {{
                const v = r.getAttribute('data-verdict');
                if (verdict === 'ALL') r.style.display = '';
                else if (v.includes(verdict)) r.style.display = '';
                else r.style.display = 'none';
            }});
        }}

        function showThesisModal(o) {{
            document.getElementById('modal-symbol').innerText = o.symbol + ' — ' + o.setup_type.replace('_',' ').toUpperCase();
            document.getElementById('modal-body').innerHTML = `
                <div><b>Direction:</b> <span style="color:${{o.direction==='LONG'?'#047857':'#b91c1c'}}">${{o.direction==='LONG'?'LONG (BUY)':'SHORT (SELL)'}}</span> | <b>Net R:R:</b> ${{o.net_rr}}x</div>
                <div style="margin-top:10px;"><b>Execution Levels:</b></div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; color:#0f172a; padding:12px; border-radius:6px; font-family:'JetBrains Mono',monospace; margin-top:4px;">
                    Trigger: ₹${{o.trigger}} | Entry Zone: ₹${{o.entry_low}} – ₹${{o.entry_high}}<br>
                    Stop Loss: ₹${{o.stop_loss}} | Target 1: ₹${{o.target1}} | Target 2: ₹${{o.target2}}
                </div>
                <div style="margin-top:12px;"><b>Trade Selection Theses:</b></div>
                <ul style="padding-left:18px; margin-top:4px; color:#334155;">${{(o.reasons || []).map(r => `<li>${{r}}</li>`).join('')}}</ul>
                ${{o.rejection_reasons && o.rejection_reasons.length ? `
                    <div style="margin-top:10px; color:#b45309;"><b>Risk Caveats:</b></div>
                    <ul style="padding-left:18px; color:#b45309; margin-top:4px;">${{o.rejection_reasons.map(r => `<li>${{r}}</li>`).join('')}}</ul>
                ` : ''}}
            `;
            document.getElementById('thesis-modal').style.display = 'flex';
        }}

        function closeModal(e) {{
            document.getElementById('thesis-modal').style.display = 'none';
        }}

        function showScanModal() {{
            document.getElementById('scan-modal').style.display = 'flex';
        }}

        function closeScanModal(e) {{
            document.getElementById('scan-modal').style.display = 'none';
        }}

        async function launchScanFromModal() {{
            const universe = document.getElementById('modal-universe').value;
            const minScore = document.getElementById('modal-minscore').value;
            document.getElementById('scan-modal').style.display = 'none';
            alert('🚀 Scan initiated for ' + universe.toUpperCase() + '. The table will update automatically when complete.');
            try {{
                await fetch('/api/scan/trigger', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ universe, min_score: parseInt(minScore), fast: true }})
                }});
            }} catch(e) {{
                alert('Error launching scan: ' + e);
            }}
        }}

        function quickResearch(sym) {{
            document.getElementById('research-sym-input').value = sym;
            executeResearch();
        }}

        async function executeResearch() {{
            const sym = document.getElementById('research-sym-input').value.trim();
            if (!sym) return;
            const container = document.getElementById('research-results-container');
            container.innerHTML = '<div style="padding:40px; color:#2563eb; font-weight:600;">Conducting institutional analysis on ' + sym + '...</div>';
            try {{
                const res = await fetch('/api/research?symbol=' + encodeURIComponent(sym));
                const d = await res.json();
                renderResearchDossier(d);
            }} catch (e) {{
                container.innerHTML = '<div style="color:#dc2626;">Error: ' + e + '</div>';
            }}
        }}

        function renderResearchDossier(d) {{
            const container = document.getElementById('research-results-container');
            const v = d.valuation_triad || {{}};
            const bear = v.bear_case || {{}};
            const base = v.base_case || {{}};
            const bull = v.bull_case || {{}};

            container.innerHTML = `
                <div class="metric-card" style="text-align:left; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="font-size:1.2rem; color:#0f172a;">${{d.company_name}} (${{d.symbol}})</h2>
                        <div style="font-size:0.82rem; color:#2563eb; font-weight:600;">Sector: ${{d.sector}} | Industry: ${{d.industry}}</div>
                    </div>
                    <div style="display:flex; gap:18px; font-size:1rem; margin:10px 0; font-family:'JetBrains Mono',monospace;">
                        <div><b>LTP:</b> ₹${{d.ltp.toLocaleString()}}</div>
                        <div><b>Market Cap:</b> ₹${{(d.market_cap_cr || 0).toLocaleString()}} Cr</div>
                    </div>
                    <p style="color:#475569; line-height:1.6; font-size:0.86rem;">${{d.business_model}}</p>
                </div>

                <div class="metrics-grid" style="text-align:left;">
                    <div class="metric-card" style="border-top: 3px solid #ef4444;">
                        <div style="color:#b91c1c; font-weight:600; font-size:0.78rem;">Bear Case Valuation</div>
                        <div style="font-size:1.6rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:6px 0; color:#0f172a;">₹${{(bear.fair_value || 0).toLocaleString()}}</div>
                        <div style="font-size:0.75rem; color:#64748b;">${{bear.assumptions || ''}}</div>
                    </div>
                    <div class="metric-card" style="border-top: 3px solid #2563eb;">
                        <div style="color:#1d4ed8; font-weight:600; font-size:0.78rem;">Base Case Fair Value</div>
                        <div style="font-size:1.6rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:6px 0; color:#0f172a;">₹${{(base.fair_value || 0).toLocaleString()}}</div>
                        <div style="font-size:0.75rem; color:#64748b;">${{base.assumptions || ''}}</div>
                    </div>
                    <div class="metric-card" style="border-top: 3px solid #059669;">
                        <div style="color:#047857; font-weight:600; font-size:0.78rem;">Bull Case Expansion</div>
                        <div style="font-size:1.6rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:6px 0; color:#0f172a;">₹${{(bull.fair_value || 0).toLocaleString()}}</div>
                        <div style="font-size:0.75rem; color:#64748b;">${{bull.assumptions || ''}}</div>
                    </div>
                </div>

                <div class="metric-card" style="text-align:left;">
                    <div class="metric-card-lbl" style="margin-bottom:10px;">Verified News & Catalysts (Clickable Sources)</div>
                    <table>
                        <thead><tr><th>Category</th><th>Headline & Source</th><th>Publisher</th></tr></thead>
                        <tbody>
                            ${{(d.catalysts || []).map(c => `
                                <tr>
                                    <td style="padding:10px;"><span class="pill pill-strong">${{c.category}}</span></td>
                                    <td style="padding:10px;"><a href="${{c.url}}" target="_blank" style="color:#2563eb; text-decoration:none; font-weight:500;">${{c.title}}</a></td>
                                    <td style="padding:10px; font-size:0.8rem; color:#64748b;">${{c.publisher}}</td>
                                </tr>
                            `).join('') || '<tr><td colspan="3" style="padding:14px; text-align:center;">No recent news recorded.</td></tr>'}}
                        </tbody>
                    </table>
                </div>
            `;
        }}

        async function loadEveningReview() {{
            const container = document.getElementById('evening-results-container');
            container.innerHTML = '<div style="padding:30px; color:#2563eb; font-weight:600;">Auditing daily outcomes against actual settlement data...</div>';
            try {{
                const res = await fetch('/api/evening');
                const d = await res.json();
                container.innerHTML = `
                    <div class="metrics-grid">
                        <div class="metric-card"><div class="metric-card-lbl">Win Rate</div><div class="metric-card-val" style="color:${{d.win_rate_pct >= 50 ? '#059669' : '#dc2626'}}">${{d.win_rate_pct}}%</div><div>${{d.win_count}} W / ${{d.loss_count}} L</div></div>
                        <div class="metric-card"><div class="metric-card-lbl">Profit Factor</div><div class="metric-card-val">${{d.profit_factor}}</div><div>Gross W / Gross L</div></div>
                        <div class="metric-card"><div class="metric-card-lbl">Net P&L (After Taxes)</div><div class="metric-card-val" style="color:${{d.net_pnl_sum >= 0 ? '#059669' : '#dc2626'}}">₹${{d.net_pnl_sum.toLocaleString()}}</div><div>Net of STT, GST, NSE fees</div></div>
                        <div class="metric-card"><div class="metric-card-lbl">Ambiguous Trades</div><div class="metric-card-val" style="color:#d97706">${{d.ambiguous_count}}</div><div>Quarantined</div></div>
                    </div>
                `;
            }} catch (e) {{
                container.innerHTML = '<div style="color:#dc2626;">Error: ' + e + '</div>';
            }}
        }}

        async function triggerLearningCycle() {{
            const container = document.getElementById('learning-results-container');
            container.innerHTML = '<div style="padding:30px; color:#2563eb; font-weight:600;">Running walk-forward validation on mutated parameters...</div>';
            try {{
                const res = await fetch('/api/learning/run', {{ method: 'POST' }});
                const d = await res.json();
                container.innerHTML = `
                    <div class="metric-card" style="text-align:left;">
                        <h3>Experiment ${{d.id}}: Decision <span style="color:${{d.decision==='PROMOTED'?'#059669':'#dc2626'}}">${{d.decision}}</span></h3>
                        <div style="margin-top:6px;"><b>Source:</b> ${{d.source_strategy_version}} &rarr; <b>Candidate:</b> ${{d.candidate_strategy_version}}</div>
                        ${{d.rejection_reason ? '<div style="color:#dc2626; margin-top:8px;"><b>Reason:</b> ' + d.rejection_reason + '</div>' : ''}}
                        <div style="margin-top:8px;"><b>Walk-Forward Metrics:</b> Sharpe: ${{d.walk_forward_metrics.sharpe_ratio?.toFixed(2)}}, Win Rate: ${{d.walk_forward_metrics.win_rate_pct?.toFixed(1)}}%, Max DD: ${{d.walk_forward_metrics.max_drawdown_pct?.toFixed(1)}}%</div>
                    </div>
                `;
            }} catch (e) {{
                container.innerHTML = '<div style="color:#dc2626;">Error: ' + e + '</div>';
            }}
        }}

        async function fetchSchedulerStatuses() {{
            const tbody = document.getElementById('scheduler-table-body');
            try {{
                const res = await fetch('/api/scheduler');
                const d = await res.json();
                const jobs = d.jobs || {{}};
                tbody.innerHTML = Object.keys(jobs).map(k => {{
                    const j = jobs[k];
                    const statusColor = j.last_status === 'SUCCESS' ? '#059669' : (j.last_status === 'ERROR' ? '#dc2626' : '#64748b');
                    return `
                        <tr>
                            <td style="padding:12px; font-weight:600; color:#0f172a;">${{j.job_name}}</td>
                            <td style="padding:12px; color:var(--text-dimmed); font-size:0.82rem;">${{j.schedule}}</td>
                            <td style="padding:12px; color:#2563eb; font-family:'JetBrains Mono',monospace; font-size:0.82rem; font-weight:600;">${{j.next_run || 'Pending'}}</td>
                            <td style="padding:12px;"><span style="background:#f1f5f9; color:${{statusColor}}; border:1px solid #e2e8f0; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600;">${{j.last_status}}</span></td>
                            <td style="padding:12px;"><button class="btn-action" onclick="triggerJob('${{k}}')">Run Now</button></td>
                        </tr>
                    `;
                }}).join('') || '<tr><td colspan="5" style="padding:18px; text-align:center;">No scheduled jobs found.</td></tr>';
            }} catch (e) {{
                tbody.innerHTML = '<tr><td colspan="5" style="padding:18px; text-align:center; color:#dc2626;">Error loading jobs: ' + e + '</td></tr>';
            }}
        }}

        async function loadReportsArchive() {{
            const tbody = document.getElementById('archive-table-body');
            try {{
                const res = await fetch('/api/reports/list');
                const list = await res.json();
                tbody.innerHTML = list.map(r => `
                    <tr>
                        <td style="padding:12px; font-weight:600; color:#0f172a;">${{r.name}}</td>
                        <td style="padding:12px;"><span class="pill pill-strong">${{r.type}}</span></td>
                        <td style="padding:12px; color:var(--text-dimmed); font-size:0.82rem;">${{r.date}}</td>
                        <td style="padding:12px; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">${{r.size_kb}} KB</td>
                        <td style="padding:12px;"><a href="${{r.url}}" target="_blank" class="btn-action" style="text-decoration:none;">Open Report ↗</a></td>
                    </tr>
                `).join('') || '<tr><td colspan="5" style="padding:18px; text-align:center;">No reports archived.</td></tr>';
            }} catch (e) {{
                tbody.innerHTML = '<tr><td colspan="5" style="padding:18px; text-align:center; color:#dc2626;">Error: ' + e + '</td></tr>';
            }}
        }}

        async function triggerJob(jobName) {{
            try {{
                const res = await fetch('/api/scheduler/trigger?job=' + encodeURIComponent(jobName), {{ method: 'POST' }});
                const data = await res.json();
                alert('✓ ' + data.message);
                if (document.getElementById('tab-scheduler').classList.contains('active')) fetchSchedulerStatuses();
            }} catch (e) {{
                alert('✕ Error triggering job: ' + e);
            }}
        }}
    </script>
    <!-- Bottom Navigation Bar for Mobile -->
    <nav class="bottom-nav">
        <button class="bottom-nav-item active" data-tab="tab-intraday" onclick="switchTab('tab-intraday')">
            <span class="bottom-nav-icon">🎯</span>
            <span class="bottom-nav-lbl">Intraday</span>
        </button>
        <button class="bottom-nav-item" data-tab="tab-longterm" onclick="switchTab('tab-longterm')">
            <span class="bottom-nav-icon">🏛️</span>
            <span class="bottom-nav-lbl">Portfolio</span>
        </button>
        <button class="bottom-nav-item" data-tab="tab-premarket" onclick="switchTab('tab-premarket')">
            <span class="bottom-nav-icon">🌅</span>
            <span class="bottom-nav-lbl">Pre-Market</span>
        </button>
        <button class="bottom-nav-item" data-tab="tab-research" onclick="switchTab('tab-research')">
            <span class="bottom-nav-icon">🔍</span>
            <span class="bottom-nav-lbl">Research</span>
        </button>
        <button class="bottom-nav-item" onclick="toggleSidebar(true)">
            <span class="bottom-nav-icon">☰</span>
            <span class="bottom-nav-lbl">More</span>
        </button>
    </nav>
</body>
</html>"""


def run_web_server(port: int = 8050, db_path: Path | str = DEFAULT_DB_PATH, enable_scheduler: bool = True) -> ThreadingHTTPServer:
    """Start the AlphaNexus web server and background automation daemon."""
    from analyst.database import init_database
    init_database(db_path)
    AnalystWebHandler.db_path = Path(db_path)

    # If fresh deploy with no cache, spawn baseline scan in background
    if not (ROOT_DIR / "_latest_scan.json").exists():
        print("[*] No existing scan cache found. Initiating baseline scan in background...", flush=True)
        boot_thread = threading.Thread(
            target=lambda: execute_research_scan(universe_tier="nifty50", limit=15, fast_mode=True),
            daemon=True,
            name="BootScannerThread"
        )
        boot_thread.start()

    if enable_scheduler:
        sched = BackgroundScheduler(db_path)
        storage = StorageManager()
        reviewer = EveningReviewer(db_path, storage)
        learner = LearningEngine(db_path, storage)

        sched.register_handler("morning_report", lambda: execute_research_scan(universe_tier="nifty50", limit=15, fast_mode=True))
        sched.register_handler("intraday_scan", lambda: execute_research_scan(universe_tier="nifty500", limit=50, fast_mode=True))
        sched.register_handler("evening_review", lambda: reviewer.evaluate_date())
        sched.register_handler("learning_cycle", lambda: learner.run_learning_cycle())
        sched.register_handler("retention_reaper", lambda: storage.apply_retention_policy(365))
        sched.start()
        AnalystWebHandler.scheduler = sched

    server = ThreadingHTTPServer(("0.0.0.0", port), AnalystWebHandler)
    print(f"[*] AlphaNexus Web Dashboard running at http://localhost:{port}")
    return server
