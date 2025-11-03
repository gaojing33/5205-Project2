# -*- coding: utf-8 -*-
"""
data_ingestion/get_market_data.py

Dependencies:
  pip install requests pandas python-dateutil python-dotenv

What this script does:
  - Reads Twelve Data API key from .env (TWELVE_DATA_API_KEY; a few variants are accepted).
  - Fetches 1-minute OHLCV bars for FIVE NASDAQ-listed U.S. tickers:
        AMAT, AMD, AVGO, MU, NVDA
  - Time window defaults: 2024-01-01 ~ 2025-10-28 (configurable via CLI).
  - Always computes the requested local technical features and rolling stats:
        open, high, low, close, volume,
        typical_price, hl_range, oc_change,
        ret_1, lret_1, ret_5, lret_5, vol_5, vwap_5, volu_mean_5, volu_z_5,
        ret_15, lret_15, vol_15, vwap_15, volu_mean_15, volu_z_15,
        ret_30, lret_30, vol_30, vwap_30, volu_mean_30, volu_z_30,
        ret_60, lret_60, vol_60, vwap_60, volu_mean_60, volu_z_60,
        ema_20, ema_50, ema_200, rsi_14, macd, macd_signal, macd_hist, bb_mid, bb_up, bb_dn
  - Adds required time-derived fields:
        minute_of_day (1..1440, first minute from 00:00 is 1),
        dow_0..dow_6 (one-hot weekday, 0=Mon,...,6=Sun),
        year, month, day, minute (HH:MM),
        date_ny ("YYYY/M/D" with no leading zeros),
        symbol (plain ticker: NVDA, etc.)
  - Writes FIVE CSV files into data/market:
        data/market/AMAT_market_data_origin.csv
        data/market/AMD_market_data_origin.csv
        data/market/AVGO_market_data_origin.csv
        data/market/MU_market_data_origin.csv
        data/market/NVDA_market_data_origin.csv

Notes:
  - We pin symbols explicitly to NASDAQ (exchange="NASDAQ") and country="US"
    to avoid cross-listing ambiguity.
  - The Twelve Data /time_series endpoint supports timezone and prepost parameters,
    and typically returns up to ~5,000 records per call; we window by business days
    to stay within that limit.
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv, dotenv_values
from pandas.tseries.offsets import BDay

# ============================== Config & Limits ==============================

BASE = "https://api.twelvedata.com"
REQS_PER_MIN = 8       # conservative per-minute throttle
DAILY_CREDITS = 800    # informational soft guard

DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE   = "2025-10-28"
DEFAULT_INTERVAL   = "1min"
DEFAULT_TZ         = "America/New_York"
DEFAULT_CHUNK_BDAYS = 10
DEFAULT_ORDER = "ASC"

# Canonical list of target tickers (NASDAQ, US)
TARGETS = [
    ("AMAT", "NASDAQ", "US"),  # Applied Materials Inc.
    ("AMD",  "NASDAQ", "US"),  # Advanced Micro Devices, Inc.
    ("AVGO", "NASDAQ", "US"),  # Broadcom Inc.
    ("MU",   "NASDAQ", "US"),  # Micron Technology Inc.
    ("NVDA", "NASDAQ", "US"),  # NVIDIA Corporation
]

# Columns we guarantee to include (deduped if already present)
REQUESTED_FEATURES = [
    "open","high","low","close","volume",
    "typical_price","hl_range","oc_change",
    "ret_1","lret_1",
    "ret_5","lret_5","vol_5","vwap_5","volu_mean_5","volu_z_5",
    "ret_15","lret_15","vol_15","vwap_15","volu_mean_15","volu_z_15",
    "ret_30","lret_30","vol_30","vwap_30","volu_mean_30","volu_z_30",
    "ret_60","lret_60","vol_60","vwap_60","volu_mean_60","volu_z_60",
    "ema_20","ema_50","ema_200","rsi_14","macd","macd_signal","macd_hist",
    "bb_mid","bb_up","bb_dn",
]

# ============================== CLI =========================================

def build_args():
    p = argparse.ArgumentParser("Fetch 1-min NASDAQ US bars from Twelve Data and compute features.")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date",   default=DEFAULT_END_DATE)
    p.add_argument("--interval",   default=DEFAULT_INTERVAL)
    p.add_argument("--timezone",   default=DEFAULT_TZ)
    p.add_argument("--chunk-bdays", type=int, default=DEFAULT_CHUNK_BDAYS)
    p.add_argument("--order", default=DEFAULT_ORDER, choices=["ASC", "DESC"])
    p.add_argument("--include-prepost", action="store_true", help="Include pre/post market if plan supports.")
    p.add_argument("--outdir", default=str(Path("data") / "market"), help="Output directory for CSV files.")
    return p.parse_args()

# ============================== API Key Loader ==============================

def load_api_key() -> str:
    """Load Twelve Data API key from environment and/or .env."""
    load_dotenv()
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

    m = dotenv_values()
    for k, v in (m or {}).items():
        if not v:
            continue
        name = (k or "").lower()
        if ("twelve" in name or "tw" in name) and ("api" in name and "key" in name):
            return str(v).strip()

    raise RuntimeError("Twelve Data API key not found. Set TWELVE_DATA_API_KEY in your project .env.")

# ============================== Rate Limiter ==============================

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

# ============================== HTTP Helper ==============================

def call(api_key: str, endpoint: str, params: Dict, retry: int = 6):
    """GET wrapper with throttling and backoff."""
    p = dict(params or {})
    p.setdefault("apikey", api_key)
    p.setdefault("format", "JSON")

    for i in range(retry):
        try:
            limiter.wait()
            r = requests.get(f"{BASE}/{endpoint}", params=p, timeout=60)
            js = r.json()
            if isinstance(js, dict) and js.get("status") == "error":
                raise RuntimeError(f"{endpoint} error {js.get('code')}: {js.get('message')}")
            return js
        except Exception as e:
            msg = str(e)
            if i < retry - 1 and ("429" in msg or "rate" in msg or "Server error" in msg):
                time.sleep(5 * (i + 1)); continue
            if i < retry - 1:
                time.sleep(2 * (i + 1)); continue
            raise

# ============================== Utilities ===================================

def chunks_by_bdays(start_ts: str, end_ts: str, bdays: int = 10) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start_ts, end_ts] into business-day chunks."""
    cur, out = pd.to_datetime(start_ts).normalize(), []
    end_ts = pd.to_datetime(end_ts).normalize()
    while cur <= end_ts:
        nxt = (cur + BDay(bdays)).normalize()
        if nxt > end_ts:
            nxt = end_ts
        out.append((cur, nxt))
        cur = (nxt + BDay(1)).normalize()
    return out

def _to_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def clean_values(values: List[Dict]) -> List[Dict]:
    """Keep only the OHLCV fields from /time_series response."""
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

# ============================== Fetcher ======================================

def fetch_time_series(api_key: str,
                      symbol: str,
                      exchange: str,
                      country: str,
                      start_date: str,
                      end_date: str,
                      interval: str = DEFAULT_INTERVAL,
                      timezone: str = DEFAULT_TZ,
                      include_prepost: bool = False,
                      order: str = DEFAULT_ORDER,
                      chunk_bdays: int = DEFAULT_CHUNK_BDAYS) -> pd.DataFrame:
    """
    Pull 1-min bars via /time_series in business-day chunks.
    We pin listing to NASDAQ (exchange) and US (country) explicitly.
    """
    symbol_with_exch = f"{symbol}:{exchange}"
    base_params = {
        "symbol":   symbol_with_exch,
        "exchange": exchange,
        "country":  country,
        "interval": interval,
        "timezone": timezone,
        "order":    order,
    }
    if include_prepost:
        base_params["prepost"] = "true"

    chunks = chunks_by_bdays(start_date, end_date, chunk_bdays)
    all_rows: List[Dict] = []
    for i, (a, b) in enumerate(chunks, 1):
        sub_a, sub_b = a, b
        while True:
            params = base_params | {
                "start_date": sub_a.strftime("%Y-%m-%d"),
                "end_date":   sub_b.strftime("%Y-%m-%d"),
            }
            js = call(api_key, "time_series", params)
            values = None
            if isinstance(js, dict) and "values" in js:
                values = js["values"]
            elif isinstance(js, dict):
                for v in js.values():
                    if isinstance(v, dict) and "values" in v:
                        values = v["values"]; break

            n = len(values or [])
            # If we hit ~5,000 cap, bisect the sub-window and refetch.
            if n >= 5000:
                mid = sub_a + (sub_b - sub_a) / 2
                sub_b = pd.to_datetime(mid).normalize()
                continue

            all_rows.extend(clean_values(values))
            print(f"[{symbol}] chunk {i}/{len(chunks)} {a.date()} ~ {b.date()} -> {n} rows")
            break

    df = pd.DataFrame(all_rows).dropna(subset=["datetime"]).drop_duplicates("datetime")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ============================== Local Features ===============================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all requested local features on 1-min bars.
    NOTE: Rolling windows (5/15/30/60) are in minutes.
    """
    df = df.copy()

    # Basic constructs
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["hl_range"]      = df["high"] - df["low"]
    df["oc_change"]     = df["close"] - df["open"]

    # Returns & rolling stats
    df["ret_1"]  = df["close"].pct_change()
    df["lret_1"] = np.log(df["close"]).diff()

    for w in [5, 15, 30, 60]:
        df[f"ret_{w}"]  = df["close"].pct_change(w)
        df[f"lret_{w}"] = np.log(df["close"]).diff(w)
        # Annualized (approx.) minute vol using 252 trading days * 390 mins/day:
        df[f"vol_{w}"]  = df["lret_1"].rolling(w).std() * np.sqrt(252 * 390 / w)
        df[f"vwap_{w}"] = (
            (df["typical_price"] * df["volume"]).rolling(w).sum()
            / (df["volume"].rolling(w).sum() + 1e-9)
        )
        df[f"volu_mean_{w}"] = df["volume"].rolling(w).mean()
        df[f"volu_z_{w}"]    = (
            df["volume"] - df[f"volu_mean_{w}"]
        ) / (df["volume"].rolling(w).std() + 1e-9)

    # EMA
    df["ema_20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema_50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, signal, macd - signal

    # Bollinger(20,2)
    ma20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    df["bb_mid"], df["bb_up"], df["bb_dn"] = ma20, ma20 + 2*sd20, ma20 - 2*sd20

    return df

# ============================== Time Columns =================================

def add_time_columns(df: pd.DataFrame, symbol_str: str) -> pd.DataFrame:
    """
    Adds: minute_of_day (1..1440), dow_0..dow_6, year/month/day, minute(HH:MM), date_ny, symbol.
    """
    df = df.copy()
    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # minute_of_day: from 00:00 (first minute is 1)
    df["minute_of_day"] = (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute) + 1

    # One-hot weekday
    dow = df["datetime"].dt.dayofweek  # 0=Mon ... 6=Sun
    dummies = pd.get_dummies(dow, prefix="dow")
    for i in range(7):
        col = f"dow_{i}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"dow_{i}" for i in range(7)]].astype(int)
    df = pd.concat(
        [df.drop(columns=[c for c in df.columns if c.startswith("dow_")], errors="ignore"), dummies],
        axis=1
    )

    # year/month/day/minute and date_ny
    df["year"]   = df["datetime"].dt.year
    df["month"]  = df["datetime"].dt.month
    df["day"]    = df["datetime"].dt.day
    df["minute"] = df["datetime"].dt.strftime("%H:%M")
    df["date_ny"] = (
        df["year"].astype(str) + "/" + df["month"].astype(str) + "/" + df["day"].astype(str)
    )

    df["symbol"] = symbol_str
    return df

# ============================== Output Shaping ===============================

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place identifier/time columns first, then requested features, then the rest."""
    id_time_cols = [
        "symbol", "datetime", "date_ny", "year", "month", "day", "minute", "minute_of_day",
        "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5","dow_6",
    ]
    features = [c for c in REQUESTED_FEATURES if c in df.columns]
    others = [c for c in df.columns if c not in id_time_cols + features]
    return df[id_time_cols + features + others]

# ============================== Runner =======================================

def run_for_one(api_key: str,
                symbol: str,
                exchange: str,
                country: str,
                start_date: str,
                end_date: str,
                interval: str,
                timezone: str,
                include_prepost: bool,
                order: str,
                chunk_bdays: int,
                outdir: Path) -> Path:
    """Fetch bars → compute features → add time cols → reorder → CSV."""
    df = fetch_time_series(
        api_key=api_key,
        symbol=symbol,
        exchange=exchange,
        country=country,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        timezone=timezone,
        include_prepost=include_prepost,
        order=order,
        chunk_bdays=chunk_bdays
    )

    if df.empty:
        # Create empty frame with the expected schema
        df = pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

    df = add_features(df)
    df = add_time_columns(df, symbol_str=symbol)
    df = reorder_columns(df)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{symbol}_market_data_origin.csv"
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

    # Rough call estimate (info only)
    bdays = pd.bdate_range(start_date, end_date).size
    est_calls = math.ceil(max(1, bdays) / chunk_bdays) * len(TARGETS)
    if est_calls > DAILY_CREDITS:
        print(f"[WARN] Estimated {est_calls} calls across all tickers > {DAILY_CREDITS} soft guard.")

    for sym, exch, ctry in TARGETS:
        try:
            run_for_one(
                api_key=api_key,
                symbol=sym,
                exchange=exch,
                country=ctry,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                timezone=timezone,
                include_prepost=include_prepost,
                order=order,
                chunk_bdays=chunk_bdays,
                outdir=outdir
            )
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")

if __name__ == "__main__":
    main()
