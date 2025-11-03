# -*- coding: utf-8 -*-
r"""
code/data_ingestion/get_benchmark_data.py

Dependencies:
  pip install requests pandas python-dateutil python-dotenv

What this script does:
  - Reads Twelve Data API key from .env (TWELVE_DATA_API_KEY and common variants).
  - Downloads 1-minute OHLCV bars for TWO ETFs: SPY and SMH.
  - Default window: 2024-04-29 ~ 2025-10-28 (can be overridden via CLI).
  - Adds required columns:
        * symbol
        * year, month, day, minute (HH:MM)
        * minutes_of_day: 1..1440 with 00:00 as minute #1
  - Writes two CSVs to data/benchmark/:
        data/benchmark/SPY_benchmark_data.csv
        data/benchmark/SMH_benchmark_data.csv
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from dotenv import load_dotenv, dotenv_values

# ============================== Config & Defaults ==============================

BASE_URL = "https://api.twelvedata.com/time_series"

# Conservative per-minute throttle & API return cap
REQS_PER_MIN = 8
MAX_ROWS_PER_CALL = 5000
DAILY_CREDITS = 800  # informational guard

# Default request window (can be changed via CLI)
DEFAULT_START_DATE = "2024-04-29"
DEFAULT_END_DATE   = "2025-10-28"
DEFAULT_INTERVAL   = "1min"
DEFAULT_TZ         = "America/New_York"
DEFAULT_CHUNK_BDAYS = 10
DEFAULT_ORDER = "ASC"

# Target ETFs (symbol only; exchange/country optional in params if you want to pin)
TARGETS = ["SPY", "SMH"]

# ============================== CLI ===========================================

import argparse

def build_args():
    p = argparse.ArgumentParser("Fetch 1-min benchmark ETF bars (SPY, SMH) from Twelve Data.")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date",   default=DEFAULT_END_DATE)
    p.add_argument("--interval",   default=DEFAULT_INTERVAL)
    p.add_argument("--timezone",   default=DEFAULT_TZ)
    p.add_argument("--chunk-bdays", type=int, default=DEFAULT_CHUNK_BDAYS)
    p.add_argument("--order", default=DEFAULT_ORDER, choices=["ASC", "DESC"])
    p.add_argument("--include-prepost", action="store_true", help="Include pre/post market if your plan supports.")
    p.add_argument("--outdir", default=str(Path("data") / "benchmark"), help="Output directory for CSV files.")
    return p.parse_args()

# ============================== API Key Loader ================================

def load_api_key() -> str:
    """
    Load Twelve Data API key from environment and/or .env.
    Accepts common variants; prefers env, then .env.
    """
    load_dotenv()  # loads .env into environment
    candidates = [
        "TWELVE_DATA_API_KEY",
        "TWELVEDATA_API_KEY",
        "TWELVE_DATA_APIKEY",
        "TD_API_KEY",
        "TWD_API_KEY",
    ]
    for k in candidates:
        v = os.getenv(k)
        if v:
            return v.strip()

    # Fallback: scan raw .env map for something that looks like a TD API key
    m = dotenv_values()
    for k, v in (m or {}).items():
        if not v:
            continue
        name = (k or "").lower()
        if ("twelve" in name or "tw" in name) and ("api" in name and "key" in name):
            return str(v).strip()

    raise RuntimeError(
        "Twelve Data API key not found. Please set TWELVE_DATA_API_KEY in your project .env."
    )

# ============================== Rate Limiter ==================================

class MinuteLimiter:
    def __init__(self, per_min: int):
        self.per_min = per_min
        self.t0 = time.monotonic()
        self.n = 0

    def wait(self):
        now = time.monotonic()
        if now - self.t0 >= 60:
            self.t0, self.n = now, 0
        if self.n >= self.per_min:
            time.sleep(max(0.0, 60 - (now - self.t0) + 0.2))
            self.t0, self.n = time.monotonic(), 0
        self.n += 1

limiter = MinuteLimiter(REQS_PER_MIN)

# ============================== HTTP Helper ===================================

def td_get(api_key: str, params: Dict, retries: int = 6):
    """
    GET with throttling, retries, and Twelve Data error normalization.
    """
    p = dict(params or {})
    p.setdefault("apikey", api_key)
    p.setdefault("format", "JSON")

    for i in range(retries):
        try:
            limiter.wait()
            r = requests.get(BASE_URL, params=p, timeout=60)
            js = r.json()
            if isinstance(js, dict) and js.get("status") == "error":
                raise RuntimeError(f"{js.get('code')}: {js.get('message')}")
            return js
        except Exception as e:
            if i < retries - 1:
                # gentle backoff
                time.sleep(2 * (i + 1))
                continue
            raise

# ============================== Utilities =====================================

def chunk_by_bdays(start_str: str, end_str: str, chunk_bdays: int = 10) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split [start, end] into business-day chunks so each request stays below ~5000 rows.
    ~10 BD ≈ 3900 rows at 1-min bars (regular session), safely below the cap.
    """
    out = []
    cur = pd.to_datetime(start_str).normalize()
    end = pd.to_datetime(end_str).normalize()
    while cur <= end:
        nxt = (cur + BDay(chunk_bdays)).normalize()
        if nxt > end:
            nxt = end
        out.append((cur, nxt))
        cur = (nxt + BDay(1)).normalize()
    return out

def parse_values(js) -> List[Dict]:
    """
    Extract 'values' array from either {values:[...]} or {SYMBOL:{values:[...]}} shapes.
    """
    if isinstance(js, dict) and "values" in js:
        return js["values"]
    if isinstance(js, dict):
        for v in js.values():
            if isinstance(v, dict) and "values" in v:
                return v["values"]
    return []

def _to_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def to_rows(values) -> List[Dict]:
    rows = []
    for v in values or []:
        rows.append({
            "datetime": v.get("datetime"),
            "open":     _to_float(v.get("open")),
            "high":     _to_float(v.get("high")),
            "low":      _to_float(v.get("low")),
            "close":    _to_float(v.get("close")),
            "volume":   _to_float(v.get("volume")),
        })
    return rows

# ============================== Fetcher =======================================

def fetch_time_series_for_symbol(api_key: str,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str,
                                 interval: str,
                                 timezone: str,
                                 include_prepost: bool,
                                 order: str,
                                 chunk_bdays: int) -> pd.DataFrame:
    """
    Pull 1-min time_series for one symbol over [start_date, end_date] in BD chunks.
    """
    base_params = {
        "symbol": symbol,
        "interval": interval,
        "timezone": timezone,
        "order": order,
    }
    if include_prepost:
        base_params["prepost"] = "true"

    chunks = chunk_by_bdays(start_date, end_date, chunk_bdays)
    print(f"[{symbol}] total chunks: {len(chunks)}")
    all_rows: List[Dict] = []

    for i, (a, b) in enumerate(chunks, 1):
        sub_a, sub_b = a, b
        while True:
            params = dict(base_params)
            params["start_date"] = sub_a.strftime("%Y-%m-%d")
            params["end_date"]   = sub_b.strftime("%Y-%m-%d")

            js = td_get(api_key, params)
            values = parse_values(js)
            n = len(values)

            # If at cap, bisect the window and refetch.
            if n >= MAX_ROWS_PER_CALL:
                mid = sub_a + (sub_b - sub_a) / 2
                sub_b = pd.to_datetime(mid).normalize()
                continue

            all_rows.extend(to_rows(values))
            print(f"[{symbol}] chunk {i}/{len(chunks)} {a.date()} ~ {b.date()} -> {n} rows")
            break

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}. Check API key/plan/symbol/time range.")

    df = df.dropna(subset=["datetime"]).drop_duplicates("datetime")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ============================== Column Enrichment =============================

def add_calendar_and_symbol_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add symbol and calendar-derived columns:
      - symbol
      - year, month, day, minute (HH:MM)
      - minutes_of_day: 1..1440 with 00:00 as minute #1
    """
    out = df.copy()
    if not np.issubdtype(out["datetime"].dtype, np.datetime64):
        out["datetime"] = pd.to_datetime(out["datetime"])

    out["symbol"] = symbol
    out["year"]   = out["datetime"].dt.year
    out["month"]  = out["datetime"].dt.month
    out["day"]    = out["datetime"].dt.day
    out["minute"] = out["datetime"].dt.strftime("%H:%M")

    # minutes_of_day: 1..1440, counting from 00:00
    out["minutes_of_day"] = (out["datetime"].dt.hour * 60 + out["datetime"].dt.minute) + 1

    return out

# ============================== Runner ========================================

def run_for_symbol(api_key: str,
                   symbol: str,
                   start_date: str,
                   end_date: str,
                   interval: str,
                   timezone: str,
                   include_prepost: bool,
                   order: str,
                   chunk_bdays: int,
                   outdir: Path) -> Path:
    df = fetch_time_series_for_symbol(
        api_key=api_key,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        timezone=timezone,
        include_prepost=include_prepost,
        order=order,
        chunk_bdays=chunk_bdays,
    )
    df = add_calendar_and_symbol_columns(df, symbol)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{symbol}_benchmark_data.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] {symbol}: wrote {len(df):,} rows -> {out_path}")
    return out_path

def main():
    args = build_args()
    api_key = load_api_key()

    start_date = args.start_date
    end_date   = args.end_date
    interval   = args.interval
    timezone   = args.timezone
    chunk_bdays = args.chunk_bdays
    order = args.order
    include_prepost = args.include_prepost
    outdir = Path(args.outdir)

    # Rough call estimation across both ETFs (informational only)
    bdays = pd.bdate_range(start_date, end_date).size
    est_calls = math.ceil(max(1, bdays) / chunk_bdays) * len(TARGETS)
    if est_calls > DAILY_CREDITS:
        print(f"[WARN] Estimated {est_calls} calls > {DAILY_CREDITS} soft guard across both ETFs.")

    for sym in TARGETS:
        try:
            run_for_symbol(
                api_key=api_key,
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                timezone=timezone,
                include_prepost=include_prepost,
                order=order,
                chunk_bdays=chunk_bdays,
                outdir=outdir,
            )
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")

if __name__ == "__main__":
    main()
