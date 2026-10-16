"""Market-data providers.

Every public method returns data *together with* a :class:`DataMeta`.  The
providers are the only place that knows how stale a source is, so they are
the only place allowed to assign a :class:`Freshness`.

Free sources used (no API key):

* Yahoo Finance via ``yfinance`` -- 5m/1d OHLCV for ``.NS`` tickers and the
  ``^NSEI`` / ``^NSEBANK`` / ``^INDIAVIX`` indices.  Delayed; the lag is
  measured from the last bar's timestamp and reported, never assumed.
* nseindia.com JSON endpoints -- market status, all indices with breadth,
  FII/DII provisional figures, Nifty 500 constituents.  Requires browser
  impersonation (``curl_cffi``); the per-symbol quote endpoint is blocked
  for non-browser clients and is deliberately *not* used.

A broker adapter (Kite/Upstox/Dhan/Fyers) would subclass
:class:`MarketDataProvider` and be the only one permitted to return
``Freshness.REALTIME``.
"""

from __future__ import annotations

import csv
import io
import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence

from .models import Bar, DataMeta, Freshness, MarketState, Quote
from .session import IST_TZ, SessionClock, now_ist


class ProviderError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Abstract interface
# --------------------------------------------------------------------------

class MarketDataProvider(ABC):
    """What the monitor needs from any data source."""

    name: str = "abstract"

    def __init__(self, clock: SessionClock, max_bar_age_seconds: int = 1200):
        self.clock = clock
        self.max_bar_age_seconds = max_bar_age_seconds

    @abstractmethod
    def intraday_bars(self, symbol: str, interval: str = "5m",
                      lookback_days: int = 5) -> tuple[list[Bar], DataMeta]:
        """Ascending intraday bars, tz-aware IST."""

    @abstractmethod
    def daily_bars(self, symbol: str, lookback_days: int = 60) -> tuple[list[Bar], DataMeta]:
        """Ascending daily bars."""

    def quote(self, symbol: str, interval: str = "5m") -> Quote:
        """Latest price derived from the most recent bar plus session aggregates.

        Default implementation builds it from bars; providers with a proper
        quote endpoint override this.
        """
        bars, meta = self.intraday_bars(symbol, interval=interval, lookback_days=5)
        if not bars:
            raise ProviderError(f"{self.name}: no bars for {symbol}")
        last = bars[-1]
        day = last.ts.date()
        today = [b for b in bars if b.ts.date() == day]
        prev = [b for b in bars if b.ts.date() < day]
        from .indicators import vwap  # local import avoids a cycle at import time
        return Quote(
            symbol=symbol,
            ltp=last.close,
            meta=meta,
            previous_close=prev[-1].close if prev else None,
            day_open=today[0].open,
            day_high=max(b.high for b in today),
            day_low=min(b.low for b in today),
            volume=sum(b.volume for b in today),
            vwap=vwap(today),
        )

    # -- helpers shared by implementations ---------------------------------

    def _freshness_for(self, data_ts: Optional[datetime], fetched: datetime,
                       base: Freshness) -> tuple[Freshness, MarketState, list[str]]:
        """Decide the freshness label from the market state and observed lag."""
        state = self.clock.state_at(fetched)
        warns: list[str] = []
        if data_ts is None:
            return Freshness.STALE, state, ["data carries no timestamp"]
        lag = (fetched - data_ts).total_seconds()
        if not state.is_tradable:
            # Market shut: whatever we have is a completed-session record.
            return Freshness.HISTORICAL, state, warns
        if lag > self.max_bar_age_seconds:
            warns.append(f"last bar is {lag / 60:.0f} min old (> {self.max_bar_age_seconds // 60} min tolerance)")
            return Freshness.STALE, state, warns
        return base, state, warns


# --------------------------------------------------------------------------
# Yahoo Finance (delayed, keyless)
# --------------------------------------------------------------------------

def to_yahoo_symbol(nse_symbol: str) -> str:
    """RELIANCE -> RELIANCE.NS; index aliases pass through."""
    aliases = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK",
               "INDIAVIX": "^INDIAVIX", "VIX": "^INDIAVIX"}
    s = nse_symbol.upper().strip()
    if s in aliases:
        return aliases[s]
    if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + ".NS"


class YahooProvider(MarketDataProvider):
    """OHLCV via yfinance.  Never claims better than DELAYED."""

    name = "yahoo"
    _bar_len = {"1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800,
                "60m": 3600, "1h": 3600, "1d": 86400}

    def __init__(self, clock: SessionClock, max_bar_age_seconds: int = 1200,
                 timeout: int = 25):
        super().__init__(clock, max_bar_age_seconds)
        self.timeout = timeout

    def _download(self, ysym: str, interval: str, period: str):
        import warnings
        import pandas as pd
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ysym, interval=interval, period=period, progress=False,
                             auto_adjust=False, timeout=self.timeout)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _frame_to_bars(self, df, interval: str) -> list[Bar]:
        bars: list[Bar] = []
        for ts, row in df.iterrows():
            ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                # Daily bars come back naive in exchange-local date terms.
                ts = ts.replace(tzinfo=IST_TZ)
            else:
                ts = ts.astimezone(IST_TZ)
            vals = [row.get(k) for k in ("Open", "High", "Low", "Close", "Volume")]
            if any(v is None or v != v for v in vals):  # NaN guard
                continue
            bars.append(Bar(ts, float(vals[0]), float(vals[1]), float(vals[2]),
                            float(vals[3]), float(vals[4])))
        bars.sort(key=lambda b: b.ts)
        return bars

    def _meta(self, ysym: str, bars: Sequence[Bar], interval: str, fetched: datetime) -> DataMeta:
        if not bars:
            return DataMeta(f"{self.name}:{ysym}", fetched, None,
                            self.clock.state_at(fetched), Freshness.STALE,
                            ("empty response",))
        # A bar stamped 10:00 on a 5m interval describes prices up to 10:05.
        last_ts = bars[-1].ts + timedelta(seconds=self._bar_len.get(interval, 300))
        last_ts = min(last_ts, fetched)
        fresh, state, warns = self._freshness_for(last_ts, fetched, Freshness.DELAYED)
        from .indicators import coherent_only
        _, dropped = coherent_only(bars)
        if dropped:
            warns.append(f"{dropped} incoherent OHLC bar(s) dropped")
        return DataMeta(f"{self.name}:{ysym}", fetched, last_ts, state, fresh, tuple(warns))

    def intraday_bars(self, symbol: str, interval: str = "5m",
                      lookback_days: int = 5) -> tuple[list[Bar], DataMeta]:
        ysym = to_yahoo_symbol(symbol)
        fetched = now_ist()
        # Yahoo caps 5m history at 60 days and 1m at 7; keep requests modest.
        period = f"{min(max(lookback_days, 1), 59)}d"
        try:
            df = self._download(ysym, interval, period)
        except Exception as e:  # network / parsing -- surface, don't guess
            raise ProviderError(f"yahoo {ysym} {interval}: {type(e).__name__}: {e}") from e
        bars = self._frame_to_bars(df, interval) if df is not None else []
        from .indicators import coherent_only
        meta = self._meta(ysym, bars, interval, fetched)
        bars, _ = coherent_only(bars)
        return bars, meta

    def daily_bars(self, symbol: str, lookback_days: int = 60) -> tuple[list[Bar], DataMeta]:
        ysym = to_yahoo_symbol(symbol)
        fetched = now_ist()
        # Ask for calendar days generously so we get `lookback_days` sessions.
        period = f"{max(lookback_days * 2, 5)}d" if lookback_days * 2 < 365 else "1y"
        try:
            df = self._download(ysym, "1d", period)
        except Exception as e:
            raise ProviderError(f"yahoo {ysym} 1d: {type(e).__name__}: {e}") from e
        bars = self._frame_to_bars(df, "1d") if df is not None else []
        meta = self._meta(ysym, bars, "1d", fetched)
        # Daily bars are by nature end-of-day records.
        meta = meta.degraded_to(Freshness.HISTORICAL, "daily interval")
        return bars, meta


# --------------------------------------------------------------------------
# NSE public JSON (market context only)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexSnapshot:
    symbol: str
    last: float
    pct_change: float
    advances: Optional[int]
    declines: Optional[int]
    unchanged: Optional[int]

    @property
    def breadth_ratio(self) -> Optional[float]:
        if self.advances is None or not self.declines:
            return None
        return self.advances / self.declines


@dataclass(frozen=True)
class InstitutionalFlow:
    date: str
    fii_net_cr: float
    dii_net_cr: float


@dataclass
class MarketContext:
    meta: DataMeta
    exchange_status: Optional[str] = None       # "Open" / "Close"
    trade_date: Optional[str] = None
    indices: dict[str, IndexSnapshot] = field(default_factory=dict)
    flows: Optional[InstitutionalFlow] = None

    def index(self, name: str) -> Optional[IndexSnapshot]:
        return self.indices.get(name.upper())


class NseContextProvider:
    """Index levels, breadth and FII/DII from nseindia.com.

    Not a :class:`MarketDataProvider` -- it has no per-symbol bars.  The
    quote-equity endpoint returns 403 to non-browser sessions and is not
    attempted; symbol prices come from Yahoo.
    """

    BASE = "https://www.nseindia.com"
    ARCHIVES = "https://nsearchives.nseindia.com"

    def __init__(self, clock=None, timeout: int = 25, min_gap_seconds: float = 1.5):
        # clock may be None when NseContext is constructed before the clock is built.
        # In that case we create a minimal default clock.
        if clock is None:
            from .session import SessionClock
            clock = SessionClock()
        self.clock = clock
        self.timeout = timeout
        self.min_gap = min_gap_seconds
        self._last_call = 0.0
        self._session = None

    def _sess(self):
        if self._session is None:
            from curl_cffi import requests as cr
            self._session = cr.Session(impersonate="chrome")
            self._session.get(self.BASE + "/", timeout=self.timeout)  # sets cookies
        return self._session

    def _get(self, url: str):
        gap = _time.monotonic() - self._last_call
        if gap < self.min_gap:
            _time.sleep(self.min_gap - gap)
        r = self._sess().get(url, timeout=self.timeout)
        self._last_call = _time.monotonic()
        if r.status_code != 200:
            raise ProviderError(f"nse {url.split('/api/')[-1]}: HTTP {r.status_code}")
        return r

    def fetch(self) -> MarketContext:
        fetched = now_ist()
        warns: list[str] = []
        ctx = MarketContext(meta=DataMeta("nse:context", fetched, None,
                                          self.clock.state_at(fetched), Freshness.STALE))
        # 1. market status -- authoritative on whether today is a session.
        try:
            ms = self._get(self.BASE + "/api/marketStatus").json()
            for m in ms.get("marketState", []):
                if m.get("market") == "Capital Market":
                    ctx.exchange_status = m.get("marketStatus")
                    ctx.trade_date = m.get("tradeDate")
                    break
            if ctx.exchange_status:
                self.clock.exchange_says_closed = ctx.exchange_status.lower().startswith("close")
        except Exception as e:
            warns.append(f"marketStatus unavailable: {e}")

        # 2. all indices with breadth.
        try:
            data = self._get(self.BASE + "/api/allIndices").json().get("data", [])
            for row in data:
                sym = str(row.get("indexSymbol", "")).upper()
                try:
                    ctx.indices[sym] = IndexSnapshot(
                        symbol=sym,
                        last=float(row.get("last")),
                        pct_change=float(row.get("percentChange") or 0.0),
                        advances=_int_or_none(row.get("advances")),
                        declines=_int_or_none(row.get("declines")),
                        unchanged=_int_or_none(row.get("unchanged")),
                    )
                except (TypeError, ValueError):
                    continue
        except Exception as e:
            warns.append(f"allIndices unavailable: {e}")

        # 3. FII / DII provisional.
        try:
            rows = self._get(self.BASE + "/api/fiidiiTradeReact").json()
            fii = dii = None
            d = None
            for r in rows:
                d = r.get("date", d)
                if r.get("category", "").startswith("FII"):
                    fii = float(r.get("netValue"))
                elif r.get("category") == "DII":
                    dii = float(r.get("netValue"))
            if fii is not None and dii is not None:
                ctx.flows = InstitutionalFlow(d or "", fii, dii)
        except Exception as e:
            warns.append(f"fiidii unavailable: {e}")

        # Freshness: index values are near-live during the session but this
        # is still an unofficial feed, so DELAYED is the ceiling.
        state = self.clock.state_at(fetched)
        if ctx.indices:
            fresh = Freshness.DELAYED if state.is_tradable else Freshness.HISTORICAL
            data_ts = fetched if state.is_tradable else None
            if not state.is_tradable and ctx.trade_date:
                try:
                    data_ts = datetime.strptime(ctx.trade_date, "%d-%b-%Y").replace(
                        hour=15, minute=30, tzinfo=IST_TZ)
                except ValueError:
                    pass
        else:
            fresh, data_ts = Freshness.STALE, None
            warns.append("no index data")
        ctx.meta = DataMeta("nse:context", fetched, data_ts, state, fresh, tuple(warns))
        return ctx

    def nifty500_symbols(self) -> list[str]:
        r = self._get(self.ARCHIVES + "/content/indices/ind_nifty500list.csv")
        reader = csv.DictReader(io.StringIO(r.text))
        return [row["Symbol"].strip() for row in reader if row.get("Series", "EQ") == "EQ"]

    def trading_holidays(self) -> Optional[list]:
        """Fetch the NSE trading holiday list for the current year.

        Returns a list of :class:`datetime.date` objects, or ``None`` if the
        endpoint is unavailable.  The caller uses the result to seed the
        :class:`MarketClock` holiday set.
        """
        from datetime import date as _date, datetime as _datetime
        try:
            r = self._get(self.BASE + "/api/holiday-master?type=trading")
            data = r.json()
            dates: list[_date] = []
            rows = data if isinstance(data, list) else data.get("CM", data.get("FO", []))
            for row in rows:
                raw = row.get("tradingDate") or row.get("holidayDate") or ""
                for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                    try:
                        dates.append(_datetime.strptime(raw, fmt).date())
                        break
                    except (ValueError, AttributeError):
                        continue
            return dates if dates else None
        except Exception:
            return None


def _int_or_none(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Convenience alias: run.py imports NseContext; run_monitor.py uses the full name.
# ---------------------------------------------------------------------------
NseContext = NseContextProvider


# --------------------------------------------------------------------------
# Replay (offline, deterministic)
# --------------------------------------------------------------------------

class ReplayProvider(MarketDataProvider):
    """Serves stored bars up to a movable cursor, for tests and dry runs.

    ``advance()`` reveals the next bar; ``intraday_bars`` returns everything
    at or before the cursor.  Freshness is always REPLAY so nothing built on
    top can mistake a rehearsal for a live session.

    When multiple symbols are loaded the no-argument ``advance()`` and
    ``now()`` operate on all symbols together (advancing the one with the
    earliest next bar), so callers don't need to know the symbol list.
    """

    name = "replay"

    def __init__(self, bars_by_symbol: dict[str, list[Bar]], clock: SessionClock):
        super().__init__(clock)
        self._bars = {k.upper(): sorted(v, key=lambda b: b.ts) for k, v in bars_by_symbol.items()}
        self._cursor: dict[str, int] = {k: 0 for k in self._bars}

    # ------------------------------------------------------------------
    # Classmethod factory used by run.py
    # ------------------------------------------------------------------
    @classmethod
    def from_provider(cls, provider: "MarketDataProvider", symbols: list[str],
                      clock: "SessionClock", interval: str = "5m",
                      session=None, warmup: int = 20) -> "ReplayProvider":
        """Fetch bars from *provider*, then build a replay restricted to *session*.

        Parameters
        ----------
        provider:
            Live provider (e.g. YahooProvider) to pull historical bars from.
        symbols:
            NSE symbols to load.
        clock:
            Shared clock instance.
        interval:
            Bar interval string (e.g. "5m").
        session:
            :class:`datetime.date` for the session to replay.  If None the
            latest date found across all symbols is used.
        warmup:
            Number of bars *before* the session kept so indicators (ATR,
            RVOL) have history to work against.  The replay cursor starts
            at the first bar of *session*.
        """
        from datetime import date as _date
        bars_by_sym: dict[str, list[Bar]] = {}
        for sym in symbols:
            try:
                bars, _ = provider.intraday_bars(sym, interval=interval, lookback_days=60)
            except Exception:
                continue
            if bars:
                bars_by_sym[sym.upper()] = bars

        if not bars_by_sym:
            raise ProviderError("ReplayProvider.from_provider: no bars fetched")

        if session is None:
            all_dates = sorted({b.ts.date() for v in bars_by_sym.values() for b in v})
            session = all_dates[-1]

        # Build per-symbol bar lists: warmup prior bars + session bars.
        trimmed: dict[str, list[Bar]] = {}
        cursors: dict[str, int] = {}
        for sym, bars in bars_by_sym.items():
            prior = [b for b in bars if b.ts.date() < session]
            sess_bars = [b for b in bars if b.ts.date() == session]
            warmup_bars = prior[-warmup:] if warmup else []
            combined = warmup_bars + sess_bars
            trimmed[sym] = combined
            cursors[sym] = len(warmup_bars)   # cursor starts at first session bar

        obj = cls(trimmed, clock)
        # Set cursors to beginning of the session for each symbol.
        for sym, c in cursors.items():
            obj._cursor[sym] = c
        return obj

    # ------------------------------------------------------------------
    # Per-symbol navigation
    # ------------------------------------------------------------------
    def advance(self, symbol: Optional[str] = None, n: int = 1) -> bool:
        """Advance the cursor.

        * ``advance(symbol)`` -- advance one symbol.
        * ``advance()`` -- advance the symbol whose *next* bar has the
          earliest timestamp across all symbols; returns False when all
          symbols are exhausted.
        """
        if symbol is not None:
            k = symbol.upper()
            if self._cursor[k] >= len(self._bars[k]):
                return False
            self._cursor[k] = min(self._cursor[k] + n, len(self._bars[k]))
            return True

        # No-arg: find the symbol with the earliest un-revealed bar.
        best_sym = None
        best_ts = None
        for k, bars in self._bars.items():
            i = self._cursor[k]
            if i >= len(bars):
                continue
            ts = bars[i].ts
            if best_ts is None or ts < best_ts:
                best_ts = ts
                best_sym = k
        if best_sym is None:
            return False
        self._cursor[best_sym] += 1
        return True

    def seek(self, symbol: str, index: int) -> None:
        k = symbol.upper()
        self._cursor[k] = max(0, min(index, len(self._bars[k])))

    def exhausted(self, symbol: str) -> bool:
        k = symbol.upper()
        return self._cursor[k] >= len(self._bars[k])

    def now(self, symbol: Optional[str] = None) -> Optional[datetime]:
        """Timestamp of the most recently revealed bar.

        * ``now(symbol)`` -- for one symbol.
        * ``now()`` -- the latest of all per-symbol now-times.
        """
        if symbol is not None:
            k = symbol.upper()
            i = self._cursor[k]
            return self._bars[k][i - 1].ts if i else None
        # No-arg: the latest revealed timestamp across all symbols.
        ts = None
        for k, bars in self._bars.items():
            i = self._cursor[k]
            if i > 0:
                t = bars[i - 1].ts
                if ts is None or t > ts:
                    ts = t
        return ts

    def _meta(self, symbol: str, bars: Sequence[Bar]) -> DataMeta:
        ts = bars[-1].ts if bars else None
        return DataMeta(f"replay:{symbol}", ts or now_ist(), ts,
                        MarketState.OPEN if bars else MarketState.CLOSED,
                        Freshness.REPLAY)

    def intraday_bars(self, symbol: str, interval: str = "5m",
                      lookback_days: int = 5) -> tuple[list[Bar], DataMeta]:
        k = symbol.upper()
        bars = self._bars[k][: self._cursor[k]]
        return list(bars), self._meta(k, bars)

    def daily_bars(self, symbol: str, lookback_days: int = 60) -> tuple[list[Bar], DataMeta]:
        # Aggregate revealed intraday bars into daily candles.
        bars, meta = self.intraday_bars(symbol)
        out: list[Bar] = []
        for b in bars:
            if out and out[-1].ts.date() == b.ts.date():
                p = out[-1]
                out[-1] = Bar(p.ts, p.open, max(p.high, b.high), min(p.low, b.low),
                              b.close, p.volume + b.volume)
            else:
                out.append(Bar(b.ts.replace(hour=0, minute=0), b.open, b.high, b.low,
                               b.close, b.volume))
        return out, meta
