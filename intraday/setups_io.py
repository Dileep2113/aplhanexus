"""Load monitored setups from YAML and persist outcomes to the journal."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

from .models import Direction, Setup
from .monitor import SetupMonitor
from .session import parse_hhmm


def load_setups(path: str | Path) -> list[Setup]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: list[Setup] = []
    for item in raw.get("setups", []):
        out.append(Setup(
            symbol=str(item["symbol"]).upper(),
            name=str(item.get("name", item["symbol"])),
            direction=Direction(str(item["direction"]).upper()),
            setup_type=str(item.get("setup_type", "unspecified")),
            trigger=float(item["trigger"]),
            entry_low=float(item["entry_low"]),
            entry_high=float(item["entry_high"]),
            stop_loss=float(item["stop_loss"]),
            target1=float(item["target1"]),
            target2=float(item["target2"]),
            target3=float(item["target3"]) if item.get("target3") is not None else None,
            invalidate=float(item["invalidate"]),
            atr=float(item["atr"]),
            score=float(item["score"]) if item.get("score") is not None else None,
            reasons=tuple(item.get("reasons", [])),
            expiry_ist=parse_hhmm(item["expiry_ist"]) if item.get("expiry_ist") else None,
        ))
    return out


JOURNAL_FIELDS = [
    "date", "symbol", "name", "direction", "setup_type", "score", "state",
    "trigger", "entry_low", "entry_high", "stop_loss", "target1", "target2", "invalidate",
    "atr", "planned_rr_t2", "expected_move_pct",
    "entry_price", "entry_time", "shares", "risk_amount", "exit_price", "exit_time",
    "exit_reason", "pnl", "r_multiple", "mfe", "mae", "data_freshness", "alerts",
]


def append_journal(path: str | Path, monitors: Iterable[SetupMonitor], day: datetime) -> int:
    """Append one row per setup (spec section 25).  Returns rows written."""
    path = Path(path)
    new = not path.exists()
    n = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if new:
            w.writeheader()
        for m in monitors:
            s, p = m.setup, m.position
            pnl = None
            rmul = None
            if p and p.exit_price is not None:
                pnl = s.direction.sign * (p.exit_price - p.entry_price) * p.shares
                rmul = pnl / p.sizing.risk_amount if p.sizing.risk_amount else None
            fresh = m.history[-1].meta.freshness.value if m.history else ""
            w.writerow({
                "date": day.date().isoformat(), "symbol": s.symbol, "name": s.name,
                "direction": s.direction.value, "setup_type": s.setup_type, "score": s.score,
                "state": s.state.value, "trigger": s.trigger, "entry_low": s.entry_low,
                "entry_high": s.entry_high, "stop_loss": s.stop_loss, "target1": s.target1,
                "target2": s.target2, "invalidate": s.invalidate, "atr": s.atr,
                "planned_rr_t2": round(s.rr() or 0, 3),
                "expected_move_pct": round(s.expected_move_pct(), 3),
                "entry_price": p.entry_price if p else None,
                "entry_time": p.entry_time.isoformat() if p else None,
                "shares": p.shares if p else None,
                "risk_amount": round(p.sizing.risk_amount, 2) if p else None,
                "exit_price": p.exit_price if p else None,
                "exit_time": p.exit_time.isoformat() if p and p.exit_time else None,
                "exit_reason": p.exit_reason if p else None,
                "pnl": round(pnl, 2) if pnl is not None else None,
                "r_multiple": round(rmul, 3) if rmul is not None else None,
                "mfe": round(p.mfe, 2) if p else None, "mae": round(p.mae, 2) if p else None,
                "data_freshness": fresh,
                "alerts": json.dumps([a.kind.value for a in m.history]),
            })
            n += 1
    return n
