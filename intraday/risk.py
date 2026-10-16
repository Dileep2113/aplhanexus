"""Position sizing against configured hard limits (spec section 17).

``size()`` never rounds a breach away.  If the smallest tradable quantity
still violates a limit the answer is zero shares and a stated reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import Direction, Setup


@dataclass(frozen=True)
class RiskLimits:
    account_capital: float
    risk_per_trade_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    max_trades_per_day: int
    max_exposure_pct: float
    lot_rounding: int = 1

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskLimits":
        r = cfg["risk"]
        return cls(
            account_capital=float(r["account_capital"]),
            risk_per_trade_pct=float(r["risk_per_trade_pct"]),
            max_daily_loss_pct=float(r["max_daily_loss_pct"]),
            max_open_positions=int(r["max_open_positions"]),
            max_trades_per_day=int(r["max_trades_per_day"]),
            max_exposure_pct=float(r["max_exposure_pct"]),
            lot_rounding=int(r.get("lot_rounding", 1)),
        )

    @property
    def max_risk_per_trade(self) -> float:
        return self.account_capital * self.risk_per_trade_pct / 100.0

    @property
    def max_daily_loss(self) -> float:
        return self.account_capital * self.max_daily_loss_pct / 100.0

    @property
    def max_exposure(self) -> float:
        return self.account_capital * self.max_exposure_pct / 100.0


@dataclass
class DayLedger:
    """Running totals the sizer checks against.  Reset each session."""

    realised_pnl: float = 0.0
    trades_taken: int = 0
    open_exposure: float = 0.0
    open_positions: int = 0

    def record_entry(self, notional: float) -> None:
        self.trades_taken += 1
        self.open_positions += 1
        self.open_exposure += notional

    def record_exit(self, notional: float, pnl: float) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.open_exposure = max(0.0, self.open_exposure - notional)
        self.realised_pnl += pnl


@dataclass(frozen=True)
class SizingResult:
    shares: int
    risk_amount: float          # what is actually at risk with this size
    notional: float             # shares * entry
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def allowed(self) -> bool:
        return self.shares > 0


def size(setup: Setup, limits: RiskLimits, ledger: Optional[DayLedger] = None) -> SizingResult:
    """Shares to trade so that risk <= max_risk_per_trade and all caps hold."""
    ledger = ledger or DayLedger()
    entry = setup.entry_reference
    rps = setup.risk_per_share
    reasons: list[str] = []

    if rps <= 0:
        return SizingResult(0, 0.0, 0.0, ("zero risk per share",))

    # Daily kill-switches first -- these do not depend on the setup.
    if ledger.realised_pnl <= -limits.max_daily_loss:
        reasons.append(f"daily loss limit hit ({ledger.realised_pnl:,.0f} <= -{limits.max_daily_loss:,.0f})")
    if ledger.trades_taken >= limits.max_trades_per_day:
        reasons.append(f"max trades per day reached ({limits.max_trades_per_day})")
    if ledger.open_positions >= limits.max_open_positions:
        reasons.append(f"max open positions reached ({limits.max_open_positions})")
    if reasons:
        return SizingResult(0, 0.0, 0.0, tuple(reasons))

    # Remaining loss budget for the day bounds this trade's risk too.
    remaining_loss_budget = limits.max_daily_loss + ledger.realised_pnl
    risk_budget = min(limits.max_risk_per_trade, remaining_loss_budget)
    if risk_budget <= 0:
        return SizingResult(0, 0.0, 0.0, ("no loss budget remaining today",))

    shares = int(risk_budget // rps)
    # Exposure cap.
    room = limits.max_exposure - ledger.open_exposure
    by_exposure = int(room // entry) if entry > 0 else 0
    if by_exposure < shares:
        reasons.append(f"capped by exposure limit (room {room:,.0f})")
        shares = max(0, by_exposure)

    lot = max(1, limits.lot_rounding)
    shares = (shares // lot) * lot
    if shares <= 0:
        reasons.append("risk per share exceeds allowed risk for even one lot")
        return SizingResult(0, 0.0, 0.0, tuple(reasons))

    return SizingResult(shares, shares * rps, shares * entry, tuple(reasons))


# ---------------------------------------------------------------------------
# Convenience aliases used by run.py
# ---------------------------------------------------------------------------

#: ``RiskConfig`` is an alternate name for :class:`RiskLimits`.
RiskConfig = RiskLimits


class RiskBook:
    """Combines :class:`RiskLimits` and a :class:`DayLedger` with a ``summary()`` method.

    ``run.py`` constructs one with ``RiskBook(config=RiskConfig.from_config(cfg))``
    and passes it to :class:`~intraday.monitor.MonitorEngine`.

    The engine accesses ``risk.limits`` for position sizing; ``run.py``
    calls ``risk.summary()`` at the end of a session to print the day's P&L.
    """

    def __init__(self, config: RiskLimits):
        self.limits: RiskLimits = config
        self.ledger: DayLedger = DayLedger()

    def summary(self) -> str:
        l = self.ledger
        cap = self.limits.account_capital
        dd_pct = abs(l.realised_pnl) / cap * 100.0 if l.realised_pnl < 0 else 0.0
        return (
            f"trades={l.trades_taken}  open={l.open_positions}  "
            f"exposure={l.open_exposure:,.0f}  "
            f"realised_pnl={l.realised_pnl:+,.0f}  "
            f"dd%={dd_pct:.2f}  "
            f"daily_loss_limit={self.limits.max_daily_loss:,.0f}"
        )

