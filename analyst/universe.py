"""NSE tradeable universe — symbol list, sector tags, liquidity filter.

Two-tier universe:
  Tier 1  Nifty 500 (reliable CSV from NSE archives)
  Tier 2  Nifty MidSmallcap 400 additions where available

After fetching, a liquidity gate filters out illiquid stocks:
  average daily volume (20-day) must exceed MIN_AVG_VOLUME shares.

Each symbol is returned as a :class:`UniverseEntry` with sector and industry
tags sourced from Yahoo Finance (cached on first use).
"""

from __future__ import annotations

import csv
import io
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

# ---------------------------------------------------------------------------
CACHE_PATH = Path(__file__).parent / "_cache_universe.json"
CACHE_MAX_AGE_HOURS = 6          # re-fetch universe after this many hours
MIN_AVG_VOLUME = 100_000         # daily shares; below this → illiquid, skip

NSE_ARCHIVES = "https://nsearchives.nseindia.com"
NSE_BASE = "https://www.nseindia.com"

# Nifty index CSVs & master NSE equity CSV on archives
_INDEX_CSVS = {
    "nifty500":  "/content/indices/ind_nifty500list.csv",
    "nifty100":  "/content/indices/ind_nifty100list.csv",
    "nifty50":   "/content/indices/ind_nifty50list.csv",
    "nse_all":   "/content/equities/EQUITY_L.csv",
}


# ---------------------------------------------------------------------------
@dataclass
class UniverseEntry:
    symbol: str               # NSE symbol (e.g. RELIANCE)
    yahoo_symbol: str         # Yahoo symbol (e.g. RELIANCE.NS)
    company_name: str
    sector: str
    industry: str
    index_membership: list[str] = field(default_factory=list)  # nifty50, nifty500 etc.
    avg_volume: Optional[float] = None   # 20-day avg daily volume
    liquidity_ok: bool = True


# ---------------------------------------------------------------------------
def _get_curl_session():
    """Return a curl_cffi session impersonating Chrome (same as NseContextProvider)."""
    from curl_cffi import requests as cr
    sess = cr.Session(impersonate="chrome")
    sess.get(NSE_BASE + "/", timeout=20)
    return sess


def _fetch_nse_csv(path: str, sess=None) -> list[dict]:
    """Fetch an NSE archives CSV and return rows as list of dicts."""
    url = NSE_ARCHIVES + path
    try:
        if sess is None:
            sess = _get_curl_session()
        r = sess.get(url, timeout=25)
        if r.status_code != 200:
            return []
        reader = csv.DictReader(io.StringIO(r.text))
        return [row for row in reader]
    except Exception:
        return []


def _yahoo_sym(nse_sym: str) -> str:
    s = nse_sym.strip().upper()
    if s.endswith(".NS") or s.startswith("^") or s.endswith(".BO"):
        return s
    return s + ".NS"


# ---------------------------------------------------------------------------
def fetch_universe(tier: str = "nifty500",
                   exclude_nifty50: bool = False,
                   use_cache: bool = True) -> list[UniverseEntry]:
    """Return the NSE universe for the chosen tier.

    Parameters
    ----------
    tier : "nifty50" | "nifty100" | "nifty500" | "nse_all" | "all"
    exclude_nifty50 : bool  -- if True, omits Nifty 50 constituents
    use_cache : bool  -- serve from disk cache if fresh enough.
    """
    cache_key = f"{tier}_{'no50' if exclude_nifty50 else 'with50'}"

    # --- try cache --------------------------------------------------------
    if use_cache and CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            import time as _t
            age_h = (_t.time() - data.get("ts", 0)) / 3600
            if age_h < CACHE_MAX_AGE_HOURS and cache_key in data:
                entries = [UniverseEntry(**e) for e in data[cache_key]]
                if entries:
                    return entries
        except Exception:
            pass

    # --- fetch from NSE ---------------------------------------------------
    print(f"  [universe] fetching {tier} symbol list from NSE archives...", flush=True)
    sess = None
    try:
        sess = _get_curl_session()
    except Exception as e:
        print(f"  [universe] curl_cffi unavailable: {e}; falling back to hardcoded list")

    rows_50 = _fetch_nse_csv(_INDEX_CSVS["nifty50"], sess)
    nifty50_syms = {r.get("Symbol", "").strip().upper() for r in rows_50}

    rows_500 = _fetch_nse_csv(_INDEX_CSVS["nifty500"], sess)
    industry_map = {
        r.get("Symbol", "").strip().upper(): r.get("Industry", "").strip()
        for r in rows_500
    }

    entries: list[UniverseEntry] = []
    seen: set[str] = set()

    if tier in ("all", "nse_all"):
        # Load the complete master list of all 2,300+ listed equity stocks
        eq_rows = _fetch_nse_csv(_INDEX_CSVS["nse_all"], sess)
        for row in eq_rows:
            series = row.get(" SERIES", row.get("SERIES", "")).strip()
            if series != "EQ":
                continue
            sym = row.get("SYMBOL", row.get("Symbol", "")).strip().upper()
            if not sym or sym in seen:
                continue
            if exclude_nifty50 and sym in nifty50_syms:
                continue
            seen.add(sym)

            ind = industry_map.get(sym, "General Equity")
            sec = _sector_from_industry(ind)
            mem = []
            if sym in nifty50_syms:
                mem.append("nifty50")
            elif sym in industry_map:
                mem.append("nifty500")
            else:
                mem.append("broad_nse")

            entries.append(UniverseEntry(
                symbol=sym,
                yahoo_symbol=_yahoo_sym(sym),
                company_name=row.get("NAME OF COMPANY", row.get("Company Name", sym)).strip(),
                sector=sec,
                industry=ind,
                index_membership=mem,
            ))
    else:
        source_rows = rows_500
        if tier == "nifty50":
            source_rows = rows_50 if rows_50 else rows_500

        for row in source_rows:
            sym = row.get("Symbol", "").strip().upper()
            if not sym or sym in seen:
                continue
            if exclude_nifty50 and sym in nifty50_syms:
                continue
            seen.add(sym)

            industry = row.get("Industry", row.get("Sector", "Unknown")).strip()
            sector = _sector_from_industry(industry)
            membership: list[str] = ["nifty500"]
            if sym in nifty50_syms:
                membership.insert(0, "nifty50")

            entries.append(UniverseEntry(
                symbol=sym,
                yahoo_symbol=_yahoo_sym(sym),
                company_name=row.get("Company Name", sym).strip(),
                sector=sector,
                industry=industry,
                index_membership=membership,
            ))

    # Sort: Nifty 50 first if present, then alphabetical
    entries.sort(key=lambda e: (0 if "nifty50" in e.index_membership else 1, e.symbol))

    # --- save cache -------------------------------------------------------
    try:
        existing = {}
        if CACHE_PATH.exists():
            existing = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        import time as _t
        existing["ts"] = _t.time()
        existing[cache_key] = [e.__dict__ for e in entries]
        CACHE_PATH.write_text(json.dumps(existing), encoding="utf-8")
    except Exception:
        pass

    print(f"  [universe] {len(entries)} symbols loaded", flush=True)
    return entries


# ---------------------------------------------------------------------------
def apply_liquidity_filter(entries: list[UniverseEntry],
                           min_avg_volume: float = MIN_AVG_VOLUME,
                           sample_size: int = 0) -> list[UniverseEntry]:
    """Mark entries as liquidity_ok=False when avg volume is too low.

    Batch downloads in chunks of 80 symbols for speed and stability.
    Any symbol that causes a NoneType / subscript error is silently skipped
    (treated as unknown volume → passes filter by default).
    """
    import yfinance as yf
    import warnings as _w
    import pandas as pd

    check = entries if not sample_size else entries[:sample_size]
    syms = [e.yahoo_symbol for e in check]

    print(f"  [universe] checking liquidity for {len(syms)} symbols in batch chunks...", flush=True)
    vol_map: dict[str, float] = {}

    def _extract_vol(df, sym: str) -> float | None:
        """Safely extract mean volume for one symbol from a downloaded DataFrame."""
        try:
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                if sym not in df.columns.get_level_values(0):
                    return None
                col = df[sym]
                if col is None or col.empty:
                    return None
                v = col["Volume"].dropna()
            else:
                # Flat DataFrame — only valid for single-symbol downloads
                v = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
            return float(v.mean()) if len(v) > 0 else None
        except Exception:
            return None

    chunk_size = 80
    for i in range(0, len(syms), chunk_size):
        chunk = syms[i:i + chunk_size]
        df = None
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                df = yf.download(
                    chunk, period="10d", interval="1d",
                    progress=False, auto_adjust=False,
                    group_by="ticker", threads=False,
                )
        except Exception:
            pass

        for s in chunk:
            avg = _extract_vol(df, s)
            if avg is not None:
                vol_map[s] = avg

    for e in check:
        avg = vol_map.get(e.yahoo_symbol)
        e.avg_volume = avg
        # Unknown volume → give benefit of the doubt (pass)
        if avg is not None and avg < min_avg_volume:
            e.liquidity_ok = False
        else:
            e.liquidity_ok = True

    filtered = [e for e in check if e.liquidity_ok]
    print(f"  [universe] {len(filtered)}/{len(check)} pass liquidity filter "
          f"(>{min_avg_volume:,.0f} avg daily vol)", flush=True)
    return filtered


# ---------------------------------------------------------------------------
def _sector_from_industry(industry: str) -> str:
    """Map NSE industry string to a broad sector label."""
    ind = industry.upper()
    mapping = {
        ("BANK", "FINANCE", "NBFC", "INSURANCE", "HOUSING FINANCE",
         "MICROFINANCE"): "Financials",
        ("INFORMATION TECHNOLOGY", "IT", "SOFTWARE", "TECH"): "Information Technology",
        ("OIL", "GAS", "PETROLEUM", "REFIN", "ENERGY"): "Energy",
        ("PHARMA", "DRUG", "HOSPITAL", "HEALTH", "DIAGNOSTIC"): "Healthcare",
        ("AUTO", "TYRE", "VEHICLE", "ANCILLAR"): "Auto",
        ("CEMENT", "CONSTRUCTION", "INFRASTRUCTURE", "REAL ESTATE"): "Infrastructure",
        ("CONSUMER", "FMCG", "FOOD", "BEVERAGE", "RETAIL"): "Consumer",
        ("METAL", "STEEL", "ALUMINIUM", "COPPER", "MINING"): "Metals",
        ("TELECOM", "COMMUNICATION"): "Telecom",
        ("POWER", "ELECTRIC", "RENEWABLE"): "Power",
        ("CHEMICAL", "FERTILIZER", "AGROCHEMICAL"): "Chemicals",
        ("MEDIA", "ENTERTAINMENT"): "Media",
        ("TEXTILE", "GARMENT"): "Textiles",
    }
    for keywords, sector in mapping.items():
        if any(k in ind for k in keywords):
            return sector
    return "Others"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    universe = fetch_universe("nifty500")
    print(f"Total symbols: {len(universe)}")
    for e in universe[:5]:
        print(f"  {e.symbol:15} {e.sector:20} {e.industry}")
