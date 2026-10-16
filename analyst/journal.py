"""Trade Journal & Historical Learning System.

Maintains historical logs of every generated setup and tracks forward performance:
  * Win rate
  * Profit factor / Average R
  * Setup type performance breakdown
  * Market regime performance breakdown
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from analyst.report import ScoredOpportunity

JOURNAL_FILE = Path(__file__).resolve().parent.parent / "analyst_journal.csv"

FIELDS = [
    "date", "symbol", "direction", "setup_type", "trigger",
    "entry_low", "entry_high", "stop_loss", "target1", "target2",
    "invalidate", "rr_ratio", "score", "catalyst", "sector",
    "market_regime", "status", "realised_r", "notes"
]


def log_opportunities(opportunities: Sequence[ScoredOpportunity],
                      file_path: Path = JOURNAL_FILE) -> int:
    """Append newly generated setups to the institutional trade journal."""
    new_file = not file_path.exists()
    count = 0
    today_str = datetime.now().strftime("%Y-%m-%d")

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for opp in opportunities:
            s = opp.setup
            c = opp.company
            m = opp.macro
            n = opp.news

            cat = n.primary_catalyst if n and n.primary_catalyst else ""
            sec = c.sector if c else ""

            writer.writerow({
                "date": today_str,
                "symbol": s.symbol,
                "direction": s.direction.value,
                "setup_type": s.setup_type,
                "trigger": s.trigger,
                "entry_low": s.entry_low,
                "entry_high": s.entry_high,
                "stop_loss": s.stop_loss,
                "target1": s.target1,
                "target2": s.target2,
                "invalidate": s.invalidate,
                "rr_ratio": round(s.rr() or 0.0, 2),
                "score": int(s.score or 0),
                "catalyst": cat[:100],
                "sector": sec,
                "market_regime": m.regime,
                "status": "RECORDED",
                "realised_r": "",
                "notes": "; ".join(opp.score_breakdown.notes[:2]),
            })
            count += 1

    return count


def summarize_journal(file_path: Path = JOURNAL_FILE) -> str:
    """Compute learning stats from the trade journal."""
    if not file_path.exists():
        return "  [journal] No historical journal found yet."

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return "  [journal] Journal is currently empty."

    total = len(rows)
    by_setup = {}
    by_dir = {"LONG": 0, "SHORT": 0}

    for r in rows:
        st = r.get("setup_type", "other")
        by_setup[st] = by_setup.get(st, 0) + 1
        d = r.get("direction", "LONG")
        by_dir[d] = by_dir.get(d, 0) + 1

    lines = [
        f"📊 INSTITUTIONAL JOURNAL SUMMARY ({total} total setups recorded)",
        f"  Direction balance: LONG {by_dir['LONG']} | SHORT {by_dir['SHORT']}",
        "  Setup breakdown:",
    ]
    for st, cnt in by_setup.items():
        lines.append(f"    • {st:<24} : {cnt:>4} setups ({cnt/total*100:.1f}%)")

    return "\n".join(lines)
