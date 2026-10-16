"""Institutional Historical Backtesting & Strategy Validation Engine.

Simulates strategy execution over historical point-in-time session data without
look-ahead bias, applying realistic execution slippage and exact Indian statutory taxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Optional, Sequence

from intraday.models import Direction
from analyst.risk_manager import calculate_trade_costs


@dataclass
class BacktestTrade:
    symbol: str
    entry_date: str
    direction: str                     # LONG | SHORT
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    statutory_costs: float
    net_pnl: float
    return_pct: float
    holding_bars: int
    exit_reason: str                   # TARGET_1 | TARGET_2 | STOP_LOSS | SQUARE_OFF_1515
    market_regime: str                 # BULL | BEAR | SIDEWAYS


@dataclass
class BacktestReport:
    strategy_name: str
    symbols_tested: list[str]
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_per_trade: float
    total_gross_pnl: float
    total_statutory_costs: float
    total_net_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    regime_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    performance_type: str = "BACKTESTED_SIMULATION"

    def summary_lines(self) -> list[str]:
        lines = [
            f"STRATEGY PERFORMANCE: {self.strategy_name} ({self.start_date} to {self.end_date})",
            f"Total Trades: {self.total_trades} | Win Rate: {self.win_rate_pct:.1f}% | Profit Factor: {self.profit_factor:.2f}",
            f"Net P&L: ₹{self.total_net_pnl:,.2f} (Total Costs Paid: ₹{self.total_statutory_costs:,.2f})",
            f"Max Drawdown: {self.max_drawdown_pct:.1f}% | Sharpe Ratio: {self.sharpe_ratio:.2f} | Expectancy: ₹{self.expectancy_per_trade:.2f}/trade",
            f"Avg Win: ₹{self.avg_win:,.2f} | Avg Loss: ₹{self.avg_loss:,.2f}",
        ]
        return lines


def run_strategy_backtest(
    symbols: list[str],
    period: str = "60d",
    capital_per_trade: float = 100000.0,
    risk_pct: float = 1.0,
    strategy_type: str = "Prior_Day_Breakout",
) -> BacktestReport:
    """Execute multi-period historical backtesting across specified symbols."""
    import yfinance as yf
    import pandas as pd

    all_trades: list[BacktestTrade] = []

    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            df = t.history(period=period, interval="1d")
            if df is None or len(df) < 5:
                continue

            # Iterate day by day (point-in-time)
            for i in range(1, len(df)):
                prev_bar = df.iloc[i - 1]
                curr_bar = df.iloc[i]

                pdh = float(prev_bar["High"])
                pdl = float(prev_bar["Low"])
                c_open = float(curr_bar["Open"])
                c_high = float(curr_bar["High"])
                c_low = float(curr_bar["Low"])
                c_close = float(curr_bar["Close"])
                trade_date = str(curr_bar.name.date())

                # Regime proxy: 20-day SMA slope if enough history
                regime = "SIDEWAYS"
                if i >= 10:
                    sma10 = float(df["Close"].iloc[max(0, i-10):i].mean())
                    if pdh > sma10 * 1.01:
                        regime = "BULL"
                    elif pdl < sma10 * 0.99:
                        regime = "BEAR"

                # Check LONG setup: Curr High breaks Prev Day High
                if c_high > pdh and c_open <= pdh * 1.015:
                    entry = pdh
                    # ATR proxy
                    atr_approx = max(0.01 * entry, (pdh - pdl))
                    stop_loss = entry - (0.8 * atr_approx)
                    target1 = entry + (1.2 * atr_approx)
                    target2 = entry + (2.0 * atr_approx)

                    stop_dist = max(0.5, entry - stop_loss)
                    risk_budget = capital_per_trade * (risk_pct / 100.0)
                    shares = max(1, int(risk_budget / stop_dist))

                    # Evaluate outcome inside current day bar
                    exit_reason = "SQUARE_OFF_1515"
                    exit_price = c_close

                    if c_low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "STOP_LOSS"
                    elif c_high >= target2:
                        exit_price = target2
                        exit_reason = "TARGET_2"
                    elif c_high >= target1:
                        exit_price = target1
                        exit_reason = "TARGET_1"

                    gross_pnl = (exit_price - entry) * shares
                    costs = calculate_trade_costs(entry, exit_price, shares, is_intraday=True)
                    net_pnl = gross_pnl - costs.total_statutory_and_slippage
                    ret_pct = ((exit_price - entry) / entry) * 100

                    all_trades.append(BacktestTrade(
                        symbol=sym,
                        entry_date=trade_date,
                        direction="LONG",
                        entry_price=round(entry, 2),
                        exit_price=round(exit_price, 2),
                        shares=shares,
                        gross_pnl=round(gross_pnl, 2),
                        statutory_costs=round(costs.total_statutory_and_slippage, 2),
                        net_pnl=round(net_pnl, 2),
                        return_pct=round(ret_pct, 2),
                        holding_bars=1,
                        exit_reason=exit_reason,
                        market_regime=regime,
                    ))

                # Check SHORT setup: Curr Low breaks Prev Day Low
                elif c_low < pdl and c_open >= pdl * 0.985:
                    entry = pdl
                    atr_approx = max(0.01 * entry, (pdh - pdl))
                    stop_loss = entry + (0.8 * atr_approx)
                    target1 = entry - (1.2 * atr_approx)
                    target2 = entry - (2.0 * atr_approx)

                    stop_dist = max(0.5, stop_loss - entry)
                    risk_budget = capital_per_trade * (risk_pct / 100.0)
                    shares = max(1, int(risk_budget / stop_dist))

                    exit_reason = "SQUARE_OFF_1515"
                    exit_price = c_close

                    if c_high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "STOP_LOSS"
                    elif c_low <= target2:
                        exit_price = target2
                        exit_reason = "TARGET_2"
                    elif c_low <= target1:
                        exit_price = target1
                        exit_reason = "TARGET_1"

                    gross_pnl = (entry - exit_price) * shares
                    costs = calculate_trade_costs(entry, exit_price, shares, is_intraday=True)
                    net_pnl = gross_pnl - costs.total_statutory_and_slippage
                    ret_pct = ((entry - exit_price) / entry) * 100

                    all_trades.append(BacktestTrade(
                        symbol=sym,
                        entry_date=trade_date,
                        direction="SHORT",
                        entry_price=round(entry, 2),
                        exit_price=round(exit_price, 2),
                        shares=shares,
                        gross_pnl=round(gross_pnl, 2),
                        statutory_costs=round(costs.total_statutory_and_slippage, 2),
                        net_pnl=round(net_pnl, 2),
                        return_pct=round(ret_pct, 2),
                        holding_bars=1,
                        exit_reason=exit_reason,
                        market_regime=regime,
                    ))
        except Exception:
            continue

    total_trades = len(all_trades)
    if total_trades == 0:
        return BacktestReport(
            strategy_name=strategy_type,
            symbols_tested=symbols,
            start_date="N/A",
            end_date="N/A",
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            expectancy_per_trade=0.0,
            total_gross_pnl=0.0,
            total_statutory_costs=0.0,
            total_net_pnl=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            avg_win=0.0,
            avg_loss=0.0,
        )

    wins = [t for t in all_trades if t.net_pnl > 0]
    losses = [t for t in all_trades if t.net_pnl <= 0]
    win_rate = (len(wins) / total_trades) * 100

    total_win_amt = sum(t.net_pnl for t in wins)
    total_loss_amt = abs(sum(t.net_pnl for t in losses))

    profit_factor = round(total_win_amt / total_loss_amt, 2) if total_loss_amt > 0 else 9.99
    tot_gross = sum(t.gross_pnl for t in all_trades)
    tot_costs = sum(t.statutory_costs for t in all_trades)
    tot_net = sum(t.net_pnl for t in all_trades)
    expectancy = tot_net / total_trades

    avg_win = (total_win_amt / len(wins)) if wins else 0.0
    avg_loss = (total_loss_amt / len(losses)) if losses else 0.0

    # Drawdown calculation
    equity_curve = [0.0]
    running = 0.0
    for t in all_trades:
        running += t.net_pnl
        equity_curve.append(running)

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = (max_dd / capital_per_trade) * 100 if capital_per_trade > 0 else 0.0
    tot_ret_pct = (tot_net / capital_per_trade) * 100 if capital_per_trade > 0 else 0.0

    # Sharpe ratio estimate
    pnl_series = [t.net_pnl for t in all_trades]
    mean_pnl = sum(pnl_series) / len(pnl_series)
    variance = sum((x - mean_pnl) ** 2 for x in pnl_series) / len(pnl_series) if len(pnl_series) > 1 else 1.0
    std_pnl = math.sqrt(variance) if variance > 0 else 1.0
    daily_rf = (0.065 / 252) * capital_per_trade  # RBI repo rate proxy
    sharpe = round(((mean_pnl - daily_rf) / std_pnl) * math.sqrt(252), 2) if std_pnl > 0 else 0.0

    # Regime breakdown
    regimes = ["BULL", "BEAR", "SIDEWAYS"]
    regime_stats = {}
    for r in regimes:
        r_trades = [t for t in all_trades if t.market_regime == r]
        if r_trades:
            r_wins = sum(1 for t in r_trades if t.net_pnl > 0)
            regime_stats[r] = {
                "trades": float(len(r_trades)),
                "win_rate": round(r_wins / len(r_trades) * 100, 1),
                "net_pnl": round(sum(t.net_pnl for t in r_trades), 2),
            }

    start_date = all_trades[0].entry_date
    end_date = all_trades[-1].entry_date

    return BacktestReport(
        strategy_name=strategy_type,
        symbols_tested=symbols,
        start_date=start_date,
        end_date=end_date,
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate_pct=round(win_rate, 1),
        profit_factor=profit_factor,
        expectancy_per_trade=round(expectancy, 2),
        total_gross_pnl=round(tot_gross, 2),
        total_statutory_costs=round(tot_costs, 2),
        total_net_pnl=round(tot_net, 2),
        total_return_pct=round(tot_ret_pct, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        sharpe_ratio=sharpe,
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        regime_breakdown=regime_stats,
        trades=all_trades,
    )
