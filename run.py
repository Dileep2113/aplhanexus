#!/usr/bin/env python
"""CLI for the intraday monitoring loop.

    python run.py context
    python run.py replay  --symbols RELIANCE,TCS [--date 2026-09-04]
    python run.py monitor --symbols RELIANCE,TCS

``replay`` re-runs a past session bar by bar and is the way to verify the loop
while the market is shut.  ``monitor`` polls the live feed, which is free and
therefore delayed -- every alert it prints says so.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

from intraday.clock import IST_TZ, MarketClock, to_ist
from intraday.context import build_context
from intraday.models import Direction, Freshness, MarketState
from intraday.monitor import MonitorConfig, MonitorEngine
from intraday.presets import previous_day_range_breakout
from intraday.providers import NseContext, ReplayProvider, YahooProvider

ROOT = Path(__file__).resolve().parent


def load_config(path: Optional[str]) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    if not p.exists():
        raise SystemExit(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def make_nse(cfg: dict) -> Optional[NseContext]:
    d = cfg.get("data", {})
    if str(d.get("context", "nse")).lower() != "nse":
        return None
    return NseContext(min_gap_seconds=float(d.get("nse_min_request_gap_seconds", 1.5)),
                      timeout=float(d.get("request_timeout_seconds", 25)))


def make_clock(cfg: dict, nse: Optional[NseContext]) -> MarketClock:
    """Clock with the exchange holiday list when it can be fetched."""
    holidays = None
    if nse is not None:
        holidays = nse.trading_holidays()
        if holidays is None:
            print("! exchange holiday calendar unavailable; weekday holidays "
                  "may be reported as an ordinary closed session\n",
                  file=sys.stderr)
    return MarketClock.from_config(cfg, holidays)


def make_yahoo(cfg: dict, clock: MarketClock) -> YahooProvider:
    d = cfg.get("data", {})
    return YahooProvider(clock,
                         max_bar_age_seconds=float(d.get("max_bar_age_seconds", 1200)),
                         timeout=float(d.get("request_timeout_seconds", 25)))


def parse_symbols(raw: str) -> list[str]:
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not out:
        raise SystemExit("no symbols given")
    return out


def banner(clock: MarketClock, now: datetime, mode: str) -> None:
    state = clock.state_at(now)
    phase = clock.phase_at(now)
    print("=" * 99)
    print(f"MODE {mode} | {now.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
          f"market {state.value} | phase {phase.value}")
    for w in clock.state_warnings(now):
        print(f"  ! {w}")
    print("=" * 99)


# --- commands ------------------------------------------------------------
def cmd_context(args) -> int:
    cfg = load_config(args.config)
    nse = make_nse(cfg)
    clock = make_clock(cfg, nse)
    yahoo = make_yahoo(cfg, clock)
    banner(clock, to_ist(datetime.now()), "context")
    print(build_context(nse=nse, vix_provider=yahoo).render())
    return 0


def _build_engine(cfg: dict, clock: MarketClock, provider, interval: str):
    from intraday.risk import RiskBook, RiskConfig
    risk = RiskBook(config=RiskConfig.from_config(cfg))
    engine = MonitorEngine(provider, clock, MonitorConfig.from_config(cfg),
                           risk, interval=interval)
    return engine, risk


def _derive_setups(bars_by_symbol: dict, session: date, directions
                   ) -> list:
    setups = []
    for sym, bars in bars_by_symbol.items():
        for direction in directions:
            try:
                setups.append(previous_day_range_breakout(
                    sym, sym, bars, session, direction))
            except ValueError as exc:
                print(f"  ! skipped {sym} {direction.value}: {exc}")
    return setups


def cmd_replay(args) -> int:
    cfg = load_config(args.config)
    interval = str(cfg.get("data", {}).get("intraday_interval", "5m"))
    nse = make_nse(cfg) if args.holidays else None
    clock = make_clock(cfg, nse)
    yahoo = make_yahoo(cfg, clock)
    symbols = parse_symbols(args.symbols)

    raw: dict[str, list] = {}
    for sym in symbols:
        try:
            bars, meta = yahoo.intraday_bars(sym, interval=interval,
                                             lookback_days=args.lookback)
        except Exception as exc:
            print(f"  ! {sym}: {exc}")
            continue
        if bars:
            raw[sym] = bars
            print(f"  loaded {sym}: {len(bars)} {interval} bars "
                  f"({bars[0].ts.date()} .. {bars[-1].ts.date()})")
        for w in meta.warnings:
            print(f"    ! {sym}: {w}")
    if not raw:
        print("no data loaded; nothing to replay")
        return 1

    sessions = sorted({b.ts.date() for bars in raw.values() for b in bars})
    session = date.fromisoformat(args.date) if args.date else sessions[-1]
    if session not in sessions:
        print(f"session {session} not in the fetched range "
              f"({sessions[0]} .. {sessions[-1]})")
        return 1

    directions = ([Direction.LONG] if args.side == "long" else
                  [Direction.SHORT] if args.side == "short" else
                  [Direction.LONG, Direction.SHORT])
    print(f"\nreplaying {session} ({len(raw)} symbol(s), "
          f"{'/'.join(d.value for d in directions)})\n")

    setups = _derive_setups(raw, session, directions)
    if not setups:
        print("no setups could be derived from the available data")
        return 1

    replay = ReplayProvider.from_provider(yahoo, list(raw), clock,
                                          interval=interval, session=session,
                                          warmup=args.warmup)
    engine, risk = _build_engine(cfg, clock, replay, interval)
    accepted = engine.register_all(setups)
    print(f"registered {accepted}/{len(setups)} setups\n")
    for setup, errs in engine.rejected:
        print(f"  ! rejected {setup.symbol} {setup.direction.value}: "
              + "; ".join(errs))

    banner(clock, replay.now(), "replay")
    print(engine.status_table())
    print()

    emitted = 0
    while replay.advance():
        for alert in engine.poll():
            print(alert.render())
            emitted += 1
        if not engine.active:
            break

    print(f"\n{emitted} alert(s) emitted over the session")
    print("\nFINAL STATE")
    print(engine.status_table())
    print(f"\nRISK BOOK: {risk.summary()}")
    print("\nNOTE: replayed from stored bars. Outcomes here are a check that "
          "the state machine works, not a performance claim.")
    return 0


def cmd_monitor(args) -> int:
    cfg = load_config(args.config)
    interval = str(cfg.get("data", {}).get("intraday_interval", "5m"))
    nse = make_nse(cfg)
    clock = make_clock(cfg, nse)
    yahoo = make_yahoo(cfg, clock)
    symbols = parse_symbols(args.symbols)
    now = to_ist(datetime.now())
    banner(clock, now, "monitor")

    state = clock.state_at(now)
    if state is not MarketState.OPEN:
        print(f"market is {state.value}: there is no live session to monitor.")
        print("run `python run.py replay` to exercise the loop on a past "
              "session instead.")
        if not args.force:
            return 2
        print("--force given: continuing against last-session data.\n")

    print(f"! {yahoo.name} is a free feed with no published delay figure. "
          f"Alerts are marked INFORMATIONAL and must not be treated as live "
          f"execution signals.\n")

    raw: dict[str, list] = {}
    for sym in symbols:
        try:
            bars, _ = yahoo.intraday_bars(sym, interval=interval, lookback_days=5)
        except Exception as exc:
            print(f"  ! {sym}: {exc}")
            continue
        if bars:
            raw[sym] = bars
    if not raw:
        print("no data for any requested symbol")
        return 1

    sessions = sorted({b.ts.date() for bars in raw.values() for b in bars})
    session = sessions[-1]
    directions = ([Direction.LONG] if args.side == "long" else
                  [Direction.SHORT] if args.side == "short" else
                  [Direction.LONG, Direction.SHORT])
    setups = _derive_setups(raw, session, directions)
    engine, risk = _build_engine(cfg, clock, yahoo, interval)
    print(f"registered {engine.register_all(setups)}/{len(setups)} setups")
    print(engine.status_table())
    print()

    poll_seconds = args.poll or int(cfg.get("monitor", {}).get("poll_seconds", 60))
    print(f"polling every {poll_seconds}s; Ctrl-C to stop\n")
    try:
        while engine.active:
            for alert in engine.poll():
                print(alert.render())
            now = to_ist(datetime.now())
            if clock.state_at(now) is not MarketState.OPEN and not args.force:
                print(f"\nsession closed at {now.strftime('%H:%M')}; stopping")
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\ninterrupted")

    print("\nFINAL STATE")
    print(engine.status_table())
    print(f"\nRISK BOOK: {risk.summary()}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("context", help="market regime and sector strength")
    c.set_defaults(func=cmd_context)

    for name, fn, helptext in (("replay", cmd_replay, "re-run a past session"),
                               ("monitor", cmd_monitor, "poll the live feed")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--symbols", required=True,
                       help="comma-separated NSE symbols, e.g. RELIANCE,TCS")
        s.add_argument("--side", choices=("long", "short", "both"),
                       default="both")
        s.set_defaults(func=fn)
        if name == "replay":
            s.add_argument("--date", help="IST session date, YYYY-MM-DD")
            s.add_argument("--lookback", type=int, default=5)
            s.add_argument("--warmup", type=int, default=20)
            s.add_argument("--holidays", action="store_true",
                           help="fetch the NSE holiday calendar first")
        else:
            s.add_argument("--poll", type=int, help="override poll seconds")
            s.add_argument("--force", action="store_true",
                           help="run even though the market is shut")

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
