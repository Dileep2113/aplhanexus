"""Institutional Risk Management, Statutory Cost & Net Reward-to-Risk Engine.

Accurately models Indian regulatory and execution costs:
  * Brokerage (Discount broker model: ₹20 or 0.03% max per executed order)
  * Securities Transaction Tax (STT): 0.025% on Sell side for intraday; 0.1% on both sides for delivery
  * Exchange Transaction Charges: NSE 0.00297%
  * Goods & Services Tax (GST): 18% on (Brokerage + Txn Charges + SEBI charges)
  * SEBI Turnover Charges: ₹10 per crore (0.0001%)
  * Stamp Duty: 0.003% on Buy side (intraday)
  * Realistic bid-ask spread & execution slippage model
  * Net Reward-to-Risk computation after all statutory charges
  * Centralized risk enforcement: sizing, exposure limits, daily loss lockout
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage: float
    total_statutory_and_slippage: float

    def summary(self) -> str:
        return (f"Total Costs: ₹{self.total_statutory_and_slippage:.2f} "
                f"[Brk: ₹{self.brokerage:.2f}, STT: ₹{self.stt:.2f}, "
                f"Exch+GST: ₹{self.exchange_charges + self.gst:.2f}, "
                f"Slip: ₹{self.slippage:.2f}]")


@dataclass
class NetRRResult:
    gross_rr: float
    net_rr: float
    total_estimated_costs: float
    gross_reward: float
    net_reward: float
    gross_risk: float
    net_risk: float
    viable: bool                         # True if net_rr >= threshold


def calculate_trade_costs(
    entry_price: float,
    exit_price: float,
    shares: int,
    is_intraday: bool = True,
    slippage_bps: float = 5.0,           # 5 bps (0.05%) default for liquid equities
) -> CostBreakdown:
    """Calculate exact Indian cash equity execution charges."""
    if shares <= 0 or entry_price <= 0 or exit_price <= 0:
        return CostBreakdown(0, 0, 0, 0, 0, 0, 0, 0)

    buy_turnover = entry_price * shares
    sell_turnover = exit_price * shares
    total_turnover = buy_turnover + sell_turnover

    # 1. Brokerage: ₹20 flat or 0.03% whichever is lower per executed leg
    b_buy = min(20.0, buy_turnover * 0.0003)
    b_sell = min(20.0, sell_turnover * 0.0003)
    brokerage = b_buy + b_sell

    # 2. STT
    if is_intraday:
        stt = sell_turnover * 0.00025    # 0.025% on sell side only
    else:
        stt = total_turnover * 0.001     # 0.1% on buy and sell for delivery

    # 3. Exchange Transaction charges: NSE cash 0.00297%
    exchange_charges = total_turnover * 0.0000297

    # 4. SEBI turnover charges: ₹10 per crore (0.0001%)
    sebi_charges = total_turnover * 0.000001

    # 5. GST: 18% on (Brokerage + Exchange charges + SEBI charges)
    gst = (brokerage + exchange_charges + sebi_charges) * 0.18

    # 6. Stamp duty: 0.003% on buy side only (cash intraday)
    stamp_duty = buy_turnover * 0.00003 if is_intraday else buy_turnover * 0.00015

    # 7. Slippage estimate
    slippage = total_turnover * (slippage_bps / 10000.0)

    total_costs = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty + slippage

    return CostBreakdown(
        brokerage=round(brokerage, 2),
        stt=round(stt, 2),
        exchange_charges=round(exchange_charges, 2),
        gst=round(gst, 2),
        sebi_charges=round(sebi_charges, 2),
        stamp_duty=round(stamp_duty, 2),
        slippage=round(slippage, 2),
        total_statutory_and_slippage=round(total_costs, 2),
    )


def compute_net_reward_risk(
    entry_price: float,
    stop_loss: float,
    target: float,
    shares: int = 100,
    is_intraday: bool = True,
    min_net_rr_threshold: float = 1.30,
) -> NetRRResult:
    """Calculate Net Reward-to-Risk strictly accounting for execution friction."""
    gross_risk_per_share = abs(entry_price - stop_loss)
    gross_reward_per_share = abs(target - entry_price)

    if gross_risk_per_share <= 0:
        return NetRRResult(0, 0, 0, 0, 0, 0, 0, False)

    gross_rr = gross_reward_per_share / gross_risk_per_share
    gross_risk_total = gross_risk_per_share * shares
    gross_reward_total = gross_reward_per_share * shares

    # Costs if target hit
    win_costs = calculate_trade_costs(entry_price, target, shares, is_intraday)
    # Costs if stop loss hit
    loss_costs = calculate_trade_costs(entry_price, stop_loss, shares, is_intraday)

    net_reward = gross_reward_total - win_costs.total_statutory_and_slippage
    net_risk = gross_risk_total + loss_costs.total_statutory_and_slippage

    if net_risk <= 0:
        net_rr = 0.0
    else:
        net_rr = max(0.0, net_reward / net_risk)

    viable = net_rr >= min_net_rr_threshold

    return NetRRResult(
        gross_rr=round(gross_rr, 2),
        net_rr=round(net_rr, 2),
        total_estimated_costs=round(win_costs.total_statutory_and_slippage, 2),
        gross_reward=round(gross_reward_total, 2),
        net_reward=round(net_reward, 2),
        gross_risk=round(gross_risk_total, 2),
        net_risk=round(net_risk, 2),
        viable=viable,
    )


# ---------------------------------------------------------------------------
# Institutional Sizing & Capital Protection
# ---------------------------------------------------------------------------

@dataclass
class SizingDecision:
    shares: int
    entry_price: float
    stop_loss: float
    risk_budget: float
    notional_value: float
    net_rr: float
    is_allowed: bool
    rejection_reasons: list[str] = field(default_factory=list)


def size_intraday_position(
    capital: float,
    risk_pct_per_trade: float,
    entry_price: float,
    stop_loss: float,
    target: float,
    max_exposure_pct: float = 40.0,
    daily_realized_loss: float = 0.0,
    max_daily_loss_pct: float = 1.5,
    min_net_rr: float = 1.30,
) -> SizingDecision:
    """Size position using strict risk-budgeting and circuit limits."""
    rejection_reasons = []

    # 1. Daily Loss Lockout Check
    max_daily_loss = capital * (max_daily_loss_pct / 100.0)
    if abs(daily_realized_loss) >= max_daily_loss and daily_realized_loss < 0:
        rejection_reasons.append(f"DAILY_LOSS_LOCKOUT: Cumulative realized loss ₹{abs(daily_realized_loss):,.2f} exceeds threshold ₹{max_daily_loss:,.2f}")
        return SizingDecision(0, entry_price, stop_loss, 0, 0, 0, False, rejection_reasons)

    # 2. Stop distance check
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        rejection_reasons.append("INVALID_STOP: Stop-loss price matches or crosses entry price")
        return SizingDecision(0, entry_price, stop_loss, 0, 0, 0, False, rejection_reasons)

    # 3. Maximum risk budget
    risk_budget = capital * (risk_pct_per_trade / 100.0)
    raw_shares = int(risk_budget / stop_distance)

    if raw_shares <= 0:
        rejection_reasons.append(f"INSUFFICIENT_CAPITAL: Stop distance ₹{stop_distance:.2f} requires more risk than budget ₹{risk_budget:.2f}")
        return SizingDecision(0, entry_price, stop_loss, risk_budget, 0, 0, False, rejection_reasons)

    # 4. Exposure limit ceiling
    max_exposure = capital * (max_exposure_pct / 100.0)
    max_shares_by_exposure = int(max_exposure / entry_price)
    final_shares = min(raw_shares, max_shares_by_exposure)

    if final_shares <= 0:
        rejection_reasons.append("EXPOSURE_CEILING: Trade violates maximum single-position exposure")
        return SizingDecision(0, entry_price, stop_loss, risk_budget, 0, 0, False, rejection_reasons)

    # 5. Net Reward-to-Risk check after statutory costs
    cost_res = compute_net_reward_risk(entry_price, stop_loss, target, final_shares, is_intraday=True)
    if cost_res.net_rr < min_net_rr:
        rejection_reasons.append(f"UNVIABLE_NET_RR: Net R:R after statutory taxes & slippage ({cost_res.net_rr:.2f}) is below minimum {min_net_rr:.2f}")

    is_allowed = len(rejection_reasons) == 0
    notional = final_shares * entry_price

    return SizingDecision(
        shares=final_shares if is_allowed else 0,
        entry_price=entry_price,
        stop_loss=stop_loss,
        risk_budget=risk_budget,
        notional_value=notional,
        net_rr=cost_res.net_rr,
        is_allowed=is_allowed,
        rejection_reasons=rejection_reasons,
    )
