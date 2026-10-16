"""Real-time monitoring loop (spec section 21).

One :class:`SetupMonitor` per setup.  Each ``update()`` receives the bars
known so far and moves the setup through its lifecycle::

    WATCHING -> PRE_ENTRY -> ACTIVE -> T1_HIT -> CLOSED_TARGET
             \\                    \\-> CLOSED_STOP
              \\-> INVALIDATED / EXPIRED

Decisions use *closed* bars only.  The most recent bar from a polling
provider may still be forming; we never trigger on it.

Honesty rules baked in:

* An ENTRY alert is ``actionable`` only when the data is REALTIME.  On the
  free feeds it is emitted as an observation ("trigger was crossed on the
  bar closing 10:20") -- the spec forbids claiming an entry the feed did
  not actually confirm, and a 15-minute-old bar cannot confirm anything.
* When the feed goes STALE the setup freezes: no transitions, one
  DATA_WARNING, until fresh bars arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Callable, Optional, Sequence

from .indicators import atr as _atr, rvol, vwap_snapshot
from .models import (Alert, AlertKind, Bar, DataMeta, Direction, Freshness,
                     Setup, SetupState)
from .risk import DayLedger, RiskLimits, SizingResult, size
from .session import IST_TZ, SessionClock


@dataclass
class MonitorConfig:
    pre_entry_atr_fraction: float = 0.35
    confirm_bars: int = 1
    require_volume_confirmation: bool = True
    volume_confirm_rvol: float = 1.2
    alert_cooldown_seconds: int = 120
    rvol_lookback: int = 20

    @classmethod
    def from_config(cls, cfg: dict) -> "MonitorConfig":
        m = cfg.get("monitor", {})
        return cls(
            pre_entry_atr_fraction=float(m.get("pre_entry_atr_fraction", 0.35)),
            confirm_bars=int(m.get("confirm_bars", 1)),
            require_volume_confirmation=bool(m.get("require_volume_confirmation", True)),
            volume_confirm_rvol=float(m.get("volume_confirm_rvol", 1.2)),
            alert_cooldown_seconds=int(m.get("alert_cooldown_seconds", 120)),
        )


@dataclass
class Position:
    """Notional position opened when the trigger confirmed."""

    entry_price: float
    entry_time: datetime
    shares: int
    sizing: SizingResult
    mfe: float = 0.0   # max favourable excursion, per share
    mae: float = 0.0   # max adverse excursion, per share
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    def pnl_per_share(self, px: float, direction: Direction) -> float:
        return direction.sign * (px - self.entry_price)


@dataclass
class SetupMonitor:
    setup: Setup
    clock: SessionClock
    limits: RiskLimits
    ledger: DayLedger
    cfg: MonitorConfig = field(default_factory=MonitorConfig)
    position: Optional[Position] = None
    last_bar_ts: Optional[datetime] = None
    _last_alert_at: dict[AlertKind, datetime] = field(default_factory=dict)
    _bars_beyond_trigger: int = 0
    _stale_warned: bool = False
    history: list[Alert] = field(default_factory=list)

    # ------------------------------------------------------------------
    def _alert(self, kind: AlertKind, msg: str, px: Optional[float], at: datetime,
               meta: DataMeta, actionable: bool = False) -> Optional[Alert]:
        last = self._last_alert_at.get(kind)
        if last and (at - last).total_seconds() < self.cfg.alert_cooldown_seconds:
            return None
        self._last_alert_at[kind] = at
        a = Alert(kind, self.setup.symbol, msg, px, at, meta, actionable)
        self.history.append(a)
        return a

    def _set(self, state: SetupState) -> None:
        self.setup.state = state

    # ------------------------------------------------------------------
    def update(self, bars: Sequence[Bar], meta: DataMeta,
               now: Optional[datetime] = None) -> list[Alert]:
        """Consume the latest bars; return any new alerts."""
        s = self.setup
        if s.state.is_terminal:
            return []
        now = (now or meta.fetched_at).astimezone(IST_TZ)
        out: list[Alert] = []

        # --- data-quality gate --------------------------------------------
        if meta.freshness is Freshness.STALE or len(bars) < 2:
            if not self._stale_warned:
                a = self._alert(AlertKind.DATA_WARNING,
                                "feed stale or insufficient; setup frozen until fresh bars arrive",
                                bars[-1].close if bars else None, now, meta)
                if a:
                    out.append(a)
                self._stale_warned = True
            return out
        self._stale_warned = False

        # Closed bars only.  With a polling provider the last bar is forming
        # unless the market is shut, in which case every bar is complete.
        # Replay bars are historical by construction, so all are closed.
        last_is_forming = (meta.market_state.is_tradable
                           and meta.freshness is not Freshness.REPLAY)
        closed = list(bars[:-1]) if last_is_forming else list(bars)
        if not closed:
            return out
        last = closed[-1]
        if self.last_bar_ts is not None and last.ts <= self.last_bar_ts:
            return out  # nothing new
        self.last_bar_ts = last.ts
        actionable = meta.freshness.is_actionable

        # --- session-time rules -------------------------------------------
        if s.state in (SetupState.WATCHING, SetupState.PRE_ENTRY):
            expiry = s.expiry_ist or self.clock.times.entry_expiry
            if last.ts.time() >= expiry:
                self._set(SetupState.EXPIRED)
                a = self._alert(AlertKind.EXPIRY,
                                f"entry window closed at {expiry.strftime('%H:%M')} without a trigger",
                                last.close, last.ts, meta)
                return out + ([a] if a else [])

        if s.state.is_open and self.clock.is_past_square_off(last.ts):
            return out + self._close(last.close, last.ts, meta, SetupState.CLOSED_TARGET
                                     if self._pnl(last.close) > 0 else SetupState.CLOSED_STOP,
                                     AlertKind.SQUARE_OFF, "mandatory intraday square-off", actionable)

        # --- pre-entry stage ----------------------------------------------
        if s.state in (SetupState.WATCHING, SetupState.PRE_ENTRY):
            sign = s.direction.sign
            # Invalidation: structure broke before we got in.
            if sign * (last.close - s.invalidate) <= 0:
                self._set(SetupState.INVALIDATED)
                a = self._alert(AlertKind.INVALIDATION,
                                f"closed through invalidation {s.invalidate:,.2f}; cancel the idea",
                                last.close, last.ts, meta)
                return out + ([a] if a else [])

            crossed = sign * (last.close - s.trigger) > 0
            if crossed:
                self._bars_beyond_trigger += 1
            else:
                self._bars_beyond_trigger = 0

            if self._bars_beyond_trigger >= self.cfg.confirm_bars:
                ok, why = self._confirm(closed, last)
                if ok:
                    return out + self._enter(last, meta, actionable, why)
                a = self._alert(AlertKind.PRE_ENTRY,
                                f"trigger {s.trigger:,.2f} crossed but not confirmed: {why}",
                                last.close, last.ts, meta)
                if a:
                    out.append(a)
                return out

            # Approaching?
            dist = sign * (s.trigger - last.close)
            if 0 < dist <= s.atr * self.cfg.pre_entry_atr_fraction:
                if s.state is SetupState.WATCHING:
                    self._set(SetupState.PRE_ENTRY)
                a = self._alert(AlertKind.PRE_ENTRY,
                                f"approaching {s.direction.value} trigger {s.trigger:,.2f} "
                                f"({dist:,.2f} away, {dist / s.atr:.2f} ATR)",
                                last.close, last.ts, meta)
                if a:
                    out.append(a)
            elif s.state is SetupState.PRE_ENTRY and dist > s.atr * self.cfg.pre_entry_atr_fraction:
                self._set(SetupState.WATCHING)
            return out

        # --- open position stage ------------------------------------------
        if s.state.is_open and self.position:
            p = self.position
            pnl_hi = p.pnl_per_share(last.high if s.direction is Direction.LONG else last.low, s.direction)
            pnl_lo = p.pnl_per_share(last.low if s.direction is Direction.LONG else last.high, s.direction)
            p.mfe = max(p.mfe, pnl_hi)
            p.mae = min(p.mae, pnl_lo)
            sign = s.direction.sign

            # Stop first: if a bar spans both stop and target, assume the worse.
            if sign * (last.low if sign > 0 else last.high) <= sign * s.stop_loss:
                return out + self._close(s.stop_loss, last.ts, meta, SetupState.CLOSED_STOP,
                                         AlertKind.STOP_LOSS, "stop-loss traded", actionable)
            far = last.high if sign > 0 else last.low
            if sign * far >= sign * s.target2:
                return out + self._close(s.target2, last.ts, meta, SetupState.CLOSED_TARGET,
                                         AlertKind.TARGET2, "target 2 reached", actionable)
            if s.state is SetupState.ACTIVE and sign * far >= sign * s.target1:
                self._set(SetupState.T1_HIT)
                a = self._alert(AlertKind.TARGET1,
                                f"target 1 {s.target1:,.2f} reached; consider trailing stop to entry",
                                s.target1, last.ts, meta, actionable)
                if a:
                    out.append(a)
        return out

    # ------------------------------------------------------------------
    def _confirm(self, closed: Sequence[Bar], last: Bar) -> tuple[bool, str]:
        """Volume + VWAP + entry-band checks on the trigger bar."""
        s = self.setup
        sign = s.direction.sign
        notes: list[str] = []
        # Entry band: a close far past the band means we're chasing.
        if sign > 0 and last.close > s.entry_high:
            return False, f"close {last.close:,.2f} beyond entry band top {s.entry_high:,.2f} (chasing)"
        if sign < 0 and last.close < s.entry_low:
            return False, f"close {last.close:,.2f} beyond entry band bottom {s.entry_low:,.2f} (chasing)"
        if self.cfg.require_volume_confirmation:
            rv = rvol(closed, self.cfg.rvol_lookback)
            if rv is None:
                return False, "not enough bars to measure relative volume"
            if rv < self.cfg.volume_confirm_rvol:
                return False, f"trigger-bar RVOL {rv:.2f} < {self.cfg.volume_confirm_rvol}"
            notes.append(f"RVOL {rv:.2f}")
        day_bars = [b for b in closed if b.ts.date() == last.ts.date()]
        snap = vwap_snapshot(day_bars)
        if snap is not None:
            if sign * (last.close - snap.value) < 0:
                return False, f"close on wrong side of VWAP {snap.value:,.2f}"
            notes.append(f"VWAP {snap.value:,.2f} {'rising' if snap.rising else 'flat/falling'}")
        if not self.clock.is_entry_window(last.ts):
            return False, "outside the entry window (09:30-expiry)"
        return True, ", ".join(notes) or "price confirmation"

    def _enter(self, bar: Bar, meta: DataMeta, actionable: bool, why: str) -> list[Alert]:
        s = self.setup
        sizing = size(s, self.limits, self.ledger)
        if not sizing.allowed:
            self._set(SetupState.INVALIDATED)
            a = self._alert(AlertKind.INVALIDATION,
                            "trigger confirmed but risk limits forbid a position: "
                            + "; ".join(sizing.reasons), bar.close, bar.ts, meta)
            return [a] if a else []
        # Use the bar close as the notional fill; a real broker fill replaces it.
        self.position = Position(bar.close, bar.ts, sizing.shares, sizing)
        self.ledger.record_entry(sizing.notional)
        self._set(SetupState.ACTIVE)
        verb = "trigger confirmed" if actionable else "trigger observed on closed bar"
        a = self._alert(AlertKind.ENTRY,
                        f"{s.direction.value} {verb} ({why}); notional {sizing.shares} sh, "
                        f"risk {sizing.risk_amount:,.0f}, SL {s.stop_loss:,.2f}, "
                        f"T1 {s.target1:,.2f}, T2 {s.target2:,.2f}",
                        bar.close, bar.ts, meta, actionable)
        return [a] if a else []

    def _pnl(self, px: float) -> float:
        if not self.position:
            return 0.0
        return self.position.pnl_per_share(px, self.setup.direction) * self.position.shares

    def _close(self, px: float, at: datetime, meta: DataMeta, state: SetupState,
               kind: AlertKind, why: str, actionable: bool) -> list[Alert]:
        p = self.position
        assert p is not None
        pnl = self._pnl(px)
        p.exit_price, p.exit_time, p.exit_reason = px, at, why
        self.ledger.record_exit(p.sizing.notional, pnl)
        self._set(state)
        r = pnl / p.sizing.risk_amount if p.sizing.risk_amount else 0.0
        a = self._alert(kind, f"{why}; P&L {pnl:,.0f} ({r:+.2f}R), MFE {p.mfe:,.2f}, MAE {p.mae:,.2f}",
                        px, at, meta, actionable)
        return [a] if a else []


# ----------------------------------------------------------------------
class MonitorBook:
    """All monitors for the day plus the shared ledger."""

    def __init__(self, setups: Sequence[Setup], clock: SessionClock, limits: RiskLimits,
                 cfg: MonitorConfig):
        self.clock, self.limits, self.cfg = clock, limits, cfg
        self.ledger = DayLedger()
        self.monitors: dict[str, SetupMonitor] = {}
        self.rejected: dict[str, list[str]] = {}
        for s in setups:
            errs = s.validate()
            if errs:
                self.rejected[s.symbol] = errs
                continue
            self.monitors[s.symbol] = SetupMonitor(s, clock, limits, self.ledger, cfg)

    def active_symbols(self) -> list[str]:
        return [k for k, m in self.monitors.items() if not m.setup.state.is_terminal]

    def all_done(self) -> bool:
        return not self.active_symbols()

    def summary(self) -> str:
        lines = ["symbol         dir    state           trig       SL        T1        T2      R:R"]
        for m in self.monitors.values():
            s = m.setup
            lines.append(f"{s.symbol:<14} {s.direction.value:<6} {s.state.value:<15} "
                         f"{s.trigger:>8.2f} {s.stop_loss:>8.2f} {s.target1:>9.2f} {s.target2:>9.2f} "
                         f"{(s.rr() or 0):>6.2f}")
        for sym, errs in self.rejected.items():
            lines.append(f"{sym:<14} REJECTED: {'; '.join(errs)}")
        l = self.ledger
        lines.append(f"ledger: trades {l.trades_taken}, open {l.open_positions}, "
                     f"exposure {l.open_exposure:,.0f}, realised P&L {l.realised_pnl:,.0f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MonitorEngine -- high-level facade used by run.py
# ---------------------------------------------------------------------------

class MonitorEngine:
    """Orchestrates a :class:`MonitorBook` against a data provider.

    This is the higher-level object that ``run.py`` instantiates.  It holds
    the provider and interval so that ``poll()`` can fetch bars and dispatch
    them to each active :class:`SetupMonitor` without the caller having to
    manage the per-symbol loop.

    Parameters
    ----------
    provider:
        Any :class:`MarketDataProvider` instance (Yahoo, Replay, …).
    clock:
        Shared :class:`SessionClock` / :class:`MarketClock`.
    config:
        :class:`MonitorConfig` loaded from config.yaml.
    risk:
        A ``RiskBook`` (or ``RiskLimits``) object; its ``limits`` attribute
        is used for position sizing.
    interval:
        Bar interval string forwarded to the provider (e.g. ``"5m"``).
    """

    def __init__(self, provider, clock: SessionClock, config: MonitorConfig,
                 risk, interval: str = "5m"):
        self._provider = provider
        self._clock = clock
        self._config = config
        self._interval = interval
        # Support both RiskBook (has .limits) and bare RiskLimits.
        self._limits: RiskLimits = getattr(risk, "limits", risk)
        self._book: Optional[MonitorBook] = None
        self._rejected_list: list[tuple] = []

    # ------------------------------------------------------------------
    def register_all(self, setups: Sequence[Setup]) -> int:
        """Register *setups*, return the number accepted (without validation errors)."""
        self._book = MonitorBook(setups, self._clock, self._limits, self._config)
        self._rejected_list = [
            (s, errs)
            for s in setups
            for sym, errs in self._book.rejected.items()
            if s.symbol == sym
        ]
        return len(self._book.monitors)

    @property
    def rejected(self) -> list[tuple]:
        """List of ``(Setup, [error_strings])`` for every rejected setup."""
        return self._rejected_list

    @property
    def active(self) -> bool:
        """True when at least one setup is still non-terminal."""
        if self._book is None:
            return False
        return not self._book.all_done()

    # ------------------------------------------------------------------
    def poll(self) -> list:
        """Fetch the latest bars and update every active monitor.

        Returns all :class:`~intraday.models.Alert` objects emitted this
        cycle across all symbols.
        """
        if self._book is None:
            return []
        alerts = []
        for sym in self._book.active_symbols():
            m = self._book.monitors[sym]
            try:
                bars, meta = self._provider.intraday_bars(
                    sym, interval=self._interval, lookback_days=5)
            except Exception as exc:
                # Surface the error as a DATA_WARNING alert so the loop keeps running.
                from .models import Alert, AlertKind, DataMeta, Freshness, MarketState
                from .session import now_ist
                now = now_ist()
                dummy_meta = DataMeta(
                    source=f"error:{sym}", fetched_at=now, data_timestamp=None,
                    market_state=self._clock.state_at(now), freshness=Freshness.STALE,
                    warnings=(str(exc),))
                alerts.append(Alert(
                    AlertKind.DATA_WARNING, sym,
                    f"provider error: {exc}", None, now, dummy_meta))
                continue
            alerts.extend(m.update(bars, meta))
        return alerts

    # ------------------------------------------------------------------
    def status_table(self) -> str:
        """Return a formatted table of all setup states (delegates to MonitorBook)."""
        if self._book is None:
            return "(no setups registered)"
        return self._book.summary()

