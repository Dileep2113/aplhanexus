"""CLI for the intraday monitoring loop.

    python run_monitor.py status                  # market state + context, no setups
    python run_monitor.py live [--once]           # poll free feeds, alert on setups.yaml
    python run_monitor.py replay --symbol RELIANCE --date 2026-09-04
                                                  # rehearse a setup against a past session

Every alert prints the data provenance line beneath it.  On the free feeds
nothing is ever labelled REALTIME, and ENTRY alerts are marked informational.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import yaml

from intraday.monitor import MonitorBook, MonitorConfig
from intraday.providers import (NseContextProvider, ProviderError, ReplayProvider,
                                YahooProvider)
from intraday.risk import RiskLimits
from intraday.session import IST_TZ, SessionClock, SessionTimes, now_ist
from intraday.setups_io import append_journal, load_setups

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def print_context(nse: NseContextProvider) -> None:
    try:
        ctx = nse.fetch()
    except ProviderError as e:
        print(f"NSE context unavailable: {e}")
        return
    print(f"Exchange: {ctx.exchange_status or '?'}  trade date {ctx.trade_date or '?'}")
    for name in ("NIFTY 50", "NIFTY BANK", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "INDIA VIX"):
        ix = ctx.index(name)
        if ix:
            br = "" if ix.breadth_ratio is None else f"  A/D {ix.advances}/{ix.declines} ({ix.breadth_ratio:.2f})"
            print(f"  {name:<18} {ix.last:>12,.2f}  {ix.pct_change:+.2f}%{br}")
    if ctx.flows:
        print(f"  FII net {ctx.flows.fii_net_cr:+,.0f} cr   DII net {ctx.flows.dii_net_cr:+,.0f} cr   ({ctx.flows.date}, provisional)")
    print(f"  data: {ctx.meta.label()}")
    for w in ctx.meta.warnings:
        print(f"  warning: {w}")


def cmd_status(args, cfg) -> int:
    clock = SessionClock(SessionTimes.from_config(cfg))
    now = now_ist()
    print(f"Now (IST): {now:%Y-%m-%d %H:%M:%S}  market state: {clock.state_at(now).value}")
    print(f"Last trading day: {clock.last_trading_day(now)}")
    if cfg["data"].get("context") == "nse":
        print_context(NseContextProvider(clock, cfg["data"]["request_timeout_seconds"],
                                         cfg["data"]["nse_min_request_gap_seconds"]))
    return 0


def cmd_live(args, cfg) -> int:
    clock = SessionClock(SessionTimes.from_config(cfg))
    limits = RiskLimits.from_config(cfg)
    mcfg = MonitorConfig.from_config(cfg)
    setups = load_setups(args.setups)
    book = MonitorBook(setups, clock, limits, mcfg)
    d = cfg["data"]
    provider = YahooProvider(clock, d["max_bar_age_seconds"], d["request_timeout_seconds"])
    interval = d.get("intraday_interval", "5m")

    print("=" * 78)
    print("LIVE MONITOR  -- free delayed feed; ENTRY alerts are observations, not fills")
    print("=" * 78)
    if cfg["data"].get("context") == "nse":
        print_context(NseContextProvider(clock, d["request_timeout_seconds"],
                                         d["nse_min_request_gap_seconds"]))
    print(book.summary())
    print("-" * 78)

    state = clock.state_at()
    if not state.is_tradable:
        print(f"Market state is {state.value}. No live session to monitor.")
        if not args.once:
            print("Exiting; re-run during 09:15-15:30 IST on a trading day.")
            return 0

    poll = int(cfg["monitor"]["poll_seconds"])
    while True:
        now = now_ist()
        for sym in book.active_symbols():
            m = book.monitors[sym]
            try:
                bars, meta = provider.intraday_bars(sym, interval=interval, lookback_days=5)
            except ProviderError as e:
                print(f"[{now:%H:%M:%S}] {sym}: {e}")
                continue
            for a in m.update(bars, meta, now):
                print(a.render())
        if args.once or book.all_done() or not clock.state_at().is_tradable:
            break
        time.sleep(poll)

    print("-" * 78)
    print(book.summary())
    n = append_journal(ROOT / "journal.csv", book.monitors.values(), now_ist())
    print(f"journal: {n} row(s) appended to journal.csv")
    return 0


def cmd_replay(args, cfg) -> int:
    """Replay one past session for one symbol using Yahoo's stored 5m bars."""
    clock = SessionClock(SessionTimes.from_config(cfg))
    limits = RiskLimits.from_config(cfg)
    mcfg = MonitorConfig.from_config(cfg)
    setups = [s for s in load_setups(args.setups) if s.symbol == args.symbol.upper()]
    if not setups:
        print(f"no setup for {args.symbol} in {args.setups}")
        return 2
    d = cfg["data"]
    yahoo = YahooProvider(clock, d["max_bar_age_seconds"], d["request_timeout_seconds"])
    day = date.fromisoformat(args.date)
    bars, meta = yahoo.intraday_bars(args.symbol, interval=d.get("intraday_interval", "5m"),
                                     lookback_days=10)
    # Keep prior days for RVOL/ATR context, then step through the chosen day.
    prior = [b for b in bars if b.ts.date() < day]
    target = [b for b in bars if b.ts.date() == day]
    if not target:
        print(f"no {d.get('intraday_interval', '5m')} bars for {args.symbol} on {day} (source: {meta.label()})")
        return 2
    replay = ReplayProvider({args.symbol: prior + target}, clock)
    replay.seek(args.symbol, len(prior))
    book = MonitorBook(setups, clock, limits, mcfg)
    m = book.monitors.get(args.symbol.upper())
    if m is None:
        print(book.summary())
        return 2
    print(f"REPLAY {args.symbol} {day}  ({len(target)} bars; source {meta.source}, "
          f"{len(prior)} prior bars for context)")
    print(book.summary())
    print("-" * 78)
    while replay.advance(args.symbol):
        rb, rmeta = replay.intraday_bars(args.symbol)
        for a in m.update(rb, rmeta, rb[-1].ts):
            print(a.render())
        if m.setup.state.is_terminal:
            break
    print("-" * 78)
    print(book.summary())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--setups", default=str(ROOT / "setups.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    live = sub.add_parser("live")
    live.add_argument("--once", action="store_true", help="single poll then exit")
    rp = sub.add_parser("replay")
    rp.add_argument("--symbol", required=True)
    rp.add_argument("--date", required=True, help="YYYY-MM-DD, a past trading day within ~59 days")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    return {"status": cmd_status, "live": cmd_live, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
