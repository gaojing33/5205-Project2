# -*- coding: utf-8 -*-
"""
Build <symbol>_fun_f-style fundamental features from AMAT_fundamental_data.csv.

Steps
------
1. Load raw fundamental daily data from:
   data/fundamental/AMAT_fundamental_data.csv

2. Clean and engineer features:
   - dividend / split flags and amounts
   - shares_out forward-fill, log and 252-day rolling z-score
   - EPS surprise, winsorized surprise, event-robust z (8-event window)
   - IQR outlier flags for key numeric columns
   - Additional event-robust z for:
       * div_amount (dividend strength)
       * eps_estimate
       * eps_actual

3. Align fundamental dates to the trading calendar:
   - Trading calendar is built from a reference market features CSV.
   - Events that happen on non-trading days are moved forward
     to the next trading day.
   - Non-trading dates are then dropped.
   - 5d_before_earnings_flag is recomputed on the trading-day grid.

4. Cut the sample to dates >= 2024-04-30 (inclusive).

"""

import os
import math
import bisect
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ================= Configuration =================
# Project root = "code" directory
# ...\5205 Reminder-Data Selection\code\preprocessing\preprocess_fundamental_data.py
# -> parents[0] = ...\code\preprocessing
# -> parents[1] = ...\code
ROOT_DIR = Path(__file__).resolve().parents[1]

# Input directory for raw fundamental data
DATA_FUND_DIR = ROOT_DIR / "data" / "fundamental"

# Output directory for processed fundamental features
OUT_DIR = ROOT_DIR / "data" / "features" / "fundamental_features"

# Reference trading calendar: data\features\market_features\AMAT_mar_f.csv
MARKET_REF_FILE = ROOT_DIR / "data" / "features" / "market_features" / "AMAT_mar_f.csv"
MARKET_DATE_COL = "date_ny"

# Tickers to process
TICKERS = ["AMAT", "NVDA", "MU", "AVGO", "AMD"]

# Rolling window and robust-z parameters
ROLL_DAYS = 252               # rolling window for shares_out (in trading days)
RZ_WINDOW_EVENTS = 8          # robust z for EPS/dividend over last 8 events
RZ_MIN_EVENTS = 4             # minimum number of events to compute robust z
IQR_K = 1.5                   # IQR outlier threshold
WINS_QUANTILES = (0.01, 0.99) # winsorization quantiles for EPS surprise
FALLBACK_WINS_ABS = 100.0     # absolute fallback winsor bound (±100%)

CUT_START_DATE = "2024-04-30"  # keep rows with date_ny >= this date (inclusive)

# ================= Helper functions =================

def to_num(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(s, errors="coerce")


def iqr_flags(x: pd.Series, k: float = IQR_K) -> pd.Series:
    """Return a boolean Series indicating IQR outliers."""
    x = x.dropna()
    if x.empty:
        return pd.Series([], dtype=bool)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (x < lo) | (x > hi)


def mad(arr: np.ndarray) -> float:
    """Median absolute deviation."""
    med = np.median(arr)
    return np.median(np.abs(arr - med))


# ---------- Fundamental feature builders ----------

def build_div_split_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build dividend and split related flags and magnitudes."""
    df = df.copy()

    # Dividend features
    if "dividend" in df.columns:
        df["dividend"] = to_num(df["dividend"])
        df["div_event_flag"] = (df["dividend"] > 0).astype(int)
        df["div_amount"] = df["dividend"].fillna(0.0)
        df["div_negative_flag"] = (df["dividend"] < 0).astype(int)
    else:
        df["div_event_flag"] = 0
        df["div_amount"] = 0.0
        df["div_negative_flag"] = 0

    # Split features
    if "split_ratio" in df.columns:
        df["split_ratio"] = to_num(df["split_ratio"])
        df["split_mult"] = df["split_ratio"].fillna(1.0)
        df["split_flag"] = (df["split_ratio"].notna() & (df["split_mult"] != 1.0)).astype(int)
        df["split_nonpos_flag"] = (df["split_mult"] <= 0).astype(int)
    else:
        df["split_mult"] = 1.0
        df["split_flag"] = 0
        df["split_nonpos_flag"] = 0

    return df


def build_shares_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build shares_out related features (forward fill, log, 252-day rolling z)."""
    df = df.copy()

    if "shares_out" in df.columns:
        df["shares_out"] = to_num(df["shares_out"])
        df["shares_out_nonpos_flag"] = (df["shares_out"] <= 0).astype(int)

        # Forward fill to avoid cold start / missing values
        df["shares_out_ffill"] = df["shares_out"].ffill()

        # Log transform and rolling statistics
        df["log_shares_out"] = np.log1p(df["shares_out_ffill"])

        roll_mean = df["log_shares_out"].rolling(ROLL_DAYS, min_periods=20).mean().shift(1)
        roll_std = df["log_shares_out"].rolling(ROLL_DAYS, min_periods=20).std(ddof=0).shift(1)

        df["shares_out_roll_mean_252"] = roll_mean
        df["shares_out_roll_std_252"] = roll_std
        df["shares_out_roll_z_252"] = (df["log_shares_out"] - roll_mean) / roll_std
    else:
        df["shares_out_nonpos_flag"] = 0
        df["shares_out_ffill"] = np.nan
        df["log_shares_out"] = np.nan
        df["shares_out_roll_mean_252"] = np.nan
        df["shares_out_roll_std_252"] = np.nan
        df["shares_out_roll_z_252"] = np.nan

    return df


def build_eps_surprise_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build EPS surprise related features, including winsorized surprise and robust z."""
    df = df.copy()

    # Earnings flag
    if "earnings_flag" in df.columns:
        earn = to_num(df["earnings_flag"]).fillna(0).astype(int)
    else:
        earn = pd.Series(0, index=df.index)

    # Keep EPS information only on earnings days
    for col in ["eps_estimate", "eps_actual", "eps_surp_pct"]:
        if col in df.columns:
            df[col] = to_num(df[col])
            df.loc[earn != 1, col] = np.nan

    # Compute surprise percentage if needed
    if "eps_estimate" in df.columns and "eps_actual" in df.columns:
        est = df["eps_estimate"]
        act = df["eps_actual"]
        with np.errstate(divide="ignore", invalid="ignore"):
            calc = 100.0 * (act - est) / np.where(est.abs() > 0, est.abs(), np.nan)
        df["eps_surp_pct_calc"] = calc
    else:
        df["eps_surp_pct_calc"] = np.nan

    if "eps_surp_pct" in df.columns:
        df["eps_surp_pct_final"] = df["eps_surp_pct"].where(
            df["eps_surp_pct"].notna(), df["eps_surp_pct_calc"]
        )
    else:
        df["eps_surp_pct_final"] = df["eps_surp_pct_calc"]

    # Winsorize surprise
    surp = df["eps_surp_pct_final"].dropna()
    if len(surp) >= 20:
        lo = surp.quantile(WINS_QUANTILES[0])
        hi = surp.quantile(WINS_QUANTILES[1])
    else:
        lo, hi = -FALLBACK_WINS_ABS, FALLBACK_WINS_ABS
    df["eps_surp_winsor"] = df["eps_surp_pct_final"].clip(lower=lo, upper=hi)

    # Robust z on event-only series (past 8 events, median/MAD, using only past data)
    rz = pd.Series(index=df.index, dtype="float64")
    mask_evt = (earn == 1) & df["eps_surp_winsor"].notna()
    surp_evt = df.loc[mask_evt, "eps_surp_winsor"]

    if not surp_evt.empty:
        roll_med = surp_evt.rolling(RZ_WINDOW_EVENTS, min_periods=RZ_MIN_EVENTS).median().shift(1)
        roll_mad = surp_evt.rolling(RZ_WINDOW_EVENTS, min_periods=RZ_MIN_EVENTS).apply(
            lambda x: mad(np.asarray(x)), raw=True
        ).shift(1)

        denom = 1.4826 * roll_mad.replace(0, np.nan)
        rz_evt = (surp_evt - roll_med) / denom
        rz.loc[rz_evt.index] = rz_evt

    df["eps_surp_rz_8"] = rz

    return df


def add_iqr_outlier_flags(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Add *_iqr_outlier boolean columns for the given numeric columns."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = to_num(df[c])
            flags = iqr_flags(s)
            out_col = f"{c}_iqr_outlier"
            df[out_col] = False
            df.loc[flags.index, out_col] = flags
    return df


# ---------- Robust event z for selected columns ----------

def robust_event_z_on_series(
    event_series: pd.Series,
    window: int = RZ_WINDOW_EVENTS,
    min_events: int = RZ_MIN_EVENTS,
) -> pd.Series:
    """Compute rolling robust z (median/MAD) on a 1D series of event values."""
    roll_med = event_series.rolling(window, min_periods=min_events).median().shift(1)
    roll_mad = event_series.rolling(window, min_periods=min_events).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    ).shift(1)
    denom = 1.4826 * roll_mad.replace(0, np.nan)
    return (event_series - roll_med) / denom


def add_event_robust_z(
    df: pd.DataFrame,
    value_col: str,
    flag_col: str,
    out_col: str,
    window: int = RZ_WINDOW_EVENTS,
    min_events: int = RZ_MIN_EVENTS,
) -> pd.DataFrame:
    """
    Add an event-only robust z column.

    - Only rows with flag_col == 1 are used to compute z.
    - Other rows are NaN for out_col.
    """
    df = df.copy()
    if value_col not in df.columns or flag_col not in df.columns:
        df[out_col] = np.nan
        return df

    v = to_num(df[value_col])
    flag = to_num(df[flag_col]).fillna(0).astype(int)

    evt_idx = df.index[flag == 1]
    if len(evt_idx) == 0:
        df[out_col] = np.nan
        return df

    evt_series = v.loc[evt_idx]
    rz_evt = robust_event_z_on_series(evt_series, window=window, min_events=min_events)

    # 关键修改在这里：只往 out_col 这一列写
    df[out_col] = np.nan
    df.loc[rz_evt.index, out_col] = rz_evt.values

    return df

# ---------- Trading calendar alignment ----------

def build_trading_calendar(market_path: Path, date_col: str = MARKET_DATE_COL):
    """Build a sorted unique list of trading dates (datetime.date) from a market features CSV."""
    df_mkt = pd.read_csv(market_path)
    if date_col not in df_mkt.columns:
        raise ValueError(f"{market_path} does not contain column {date_col}")
    dates = pd.to_datetime(df_mkt[date_col]).dt.date
    trade_dates = sorted(dates.unique())
    return trade_dates


def next_trading_date(d, trade_dates):
    """Find the smallest trading date strictly greater than d. Return None if not found."""
    i = bisect.bisect_left(trade_dates, d)
    if i >= len(trade_dates):
        return None
    return trade_dates[i]


def shift_events_to_trading_days(
    df_fund: pd.DataFrame,
    trade_dates,
    date_col: str = "date_ny",
    earn_col: str = "earnings_flag",
) -> pd.DataFrame:
    """
    Align fundamental events to trading days.

    1. Identify fundamental dates that are not trading days.
    2. If that date has an earnings_flag == 1, propagate EPS-related columns
       to the next trading day.
    3. For dividend / split columns, propagate non-zero values to the next trading day.
    4. Drop non-trading dates.
    5. Recompute 5d_before_earnings_flag on the trading-day grid.
    """
    df = df_fund.copy()

    # Normalise date column to datetime.date
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df = df.sort_values(date_col).reset_index(drop=True)

    trade_date_set = set(trade_dates)
    fund_date_set = set(df[date_col])

    extra_dates = sorted(fund_date_set - trade_date_set)

    # Columns to propagate when moving events from non-trading day to next trading day
    earn_prop_cols = [
        "earnings_flag",
        "eps_estimate",
        "eps_actual",
        "eps_surp_pct",
        "eps_surp_pct_calc",
        "eps_surp_pct_final",
        "eps_surp_winsor",
        "eps_surp_rz_8",
        "eps_estimate_rz_8",
        "eps_actual_rz_8",
    ]
    div_prop_cols = [
        "dividend",
        "div_event_flag",
        "div_amount",
        "div_negative_flag",
        "split_ratio",
        "split_mult",
        "split_flag",
        "split_nonpos_flag",
        "div_amount_rz_8",
    ]

    for d in extra_dates:
        next_d = next_trading_date(d, trade_dates)
        if next_d is None:
            # Out of sample at the end; nothing to propagate
            continue

        src_rows = df[df[date_col] == d]
        if src_rows.empty:
            continue
        src_row = src_rows.iloc[-1]

        dest_idx = df.index[df[date_col] == next_d]
        if dest_idx.empty:
            # Trading date not present in fundamentals; skip
            continue

        # Earnings-related propagation
        if earn_col in df.columns and (src_row.get(earn_col, 0) == 1):
            for col in earn_prop_cols:
                if col in df.columns:
                    df.loc[dest_idx, col] = src_row[col]

        # Dividend / split propagation for non-zero values
        for col in div_prop_cols:
            if col not in df.columns:
                continue
            val = src_row[col]
            if pd.isna(val):
                continue

            nonzero = False
            if isinstance(val, (int, float)):
                nonzero = (val != 0)
            elif isinstance(val, bool):
                nonzero = val
            else:
                s = str(val).strip()
                nonzero = s not in ("", "0", "0.0")

            if nonzero:
                df.loc[dest_idx, col] = val

    # Keep only trading dates
    df = df[df[date_col].isin(trade_date_set)].copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    # Rebuild 5d_before_earnings_flag on trading-day grid
    if "5d_before_earnings_flag" not in df.columns:
        df["5d_before_earnings_flag"] = 0
    else:
        df["5d_before_earnings_flag"] = 0

    if earn_col in df.columns:
        earn_indices = df.index[df[earn_col] == 1].tolist()
        for i in earn_indices:
            # Look back up to 5 trading days
            for k in range(1, 6):
                j = i - k
                if j < 0:
                    break
                dist = k
                current = df.at[j, "5d_before_earnings_flag"]
                if (current == 0) or (dist < current):
                    df.at[j, "5d_before_earnings_flag"] = dist

    return df


def format_date_to_ymd_slash(x):
    """Format datetime.date as 'YYYY/M/D'."""
    return f"{x.year}/{x.month}/{x.day}"


# ================= Per-ticker processing =================

def process_one_ticker(ticker: str, trade_dates) -> None:
    """Process one ticker from *_fundamental_data.csv to *_fun_f.csv."""
    in_path = DATA_FUND_DIR / f"{ticker}_fundamental_data.csv"
    if not in_path.is_file():
        raise FileNotFoundError(f"Input fundamental file not found: {in_path}")

    print(f"[LOAD] {ticker}: {in_path}")
    df = pd.read_csv(in_path)

    # Ensure we have a symbol column and enforce ticker as symbol
    if "symbol" not in df.columns:
        df["symbol"] = ticker
    else:
        df["symbol"] = df["symbol"].fillna(ticker)

    # Parse date_ny and sort
    if "date_ny" not in df.columns:
        raise ValueError(f"Input fundamental_data for {ticker} must contain a 'date_ny' column.")
    df["date_ny"] = pd.to_datetime(df["date_ny"])
    df = df.sort_values("date_ny").reset_index(drop=True)

    # Optional: drop helper columns that are not needed downstream
    for col in ["eps_surp_pctz"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ---------- Feature engineering on the full history ----------
    df = build_div_split_features(df)
    df = build_shares_features(df)
    df = build_eps_surprise_features(df)

    # IQR outlier flags on key numeric columns
    iqr_candidates = [
        "dividend",
        "div_amount",
        "split_ratio",
        "split_mult",
        "shares_out",
        "log_shares_out",
        "eps_estimate",
        "eps_actual",
        "eps_surp_pct_final",
        "eps_surp_winsor",
    ]
    df = add_iqr_outlier_flags(df, iqr_candidates)

    # Additional event-robust z features
    df = add_event_robust_z(df, "div_amount", "div_event_flag", "div_amount_rz_8")
    df = add_event_robust_z(df, "eps_estimate", "earnings_flag", "eps_estimate_rz_8")
    df = add_event_robust_z(df, "eps_actual", "earnings_flag", "eps_actual_rz_8")

    # ---------- Align to trading calendar and recompute 5d_before_earnings_flag ----------
    df = shift_events_to_trading_days(df, trade_dates, date_col="date_ny", earn_col="earnings_flag")

    # ---------- Cut sample to dates >= CUT_START_DATE ----------
    cut_date = pd.to_datetime(CUT_START_DATE).date()
    df = df[df["date_ny"] >= cut_date].copy()

    # ---------- Format date_ny and add year / month / day ----------
    df["date_ny"] = df["date_ny"].apply(format_date_to_ymd_slash)

    dt_parsed = pd.to_datetime(df["date_ny"])
    df["year"] = dt_parsed.dt.year
    df["month"] = dt_parsed.dt.month
    df["day"] = dt_parsed.dt.day

    # ---------- Fill NaNs with 0 for EPS / surprise / robust-z columns ----------
    zero_fill_cols = [
        "eps_estimate",
        "eps_actual",
        "eps_surp_pct",
        "eps_surp_pct_calc",
        "eps_surp_pct_final",
        "eps_surp_winsor",
        "eps_surp_rz_8",
        "div_amount_rz_8",
        "eps_estimate_rz_8",
        "eps_actual_rz_8",
    ]
    for c in zero_fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ---------- Convert IQR outlier flags from bool to 0/1 ----------
    iqr_flag_cols = [
        "dividend_iqr_outlier",
        "div_amount_iqr_outlier",
        "split_ratio_iqr_outlier",
        "split_mult_iqr_outlier",
        "shares_out_iqr_outlier",
        "log_shares_out_iqr_outlier",
        "eps_estimate_iqr_outlier",
        "eps_actual_iqr_outlier",
        "eps_surp_pct_final_iqr_outlier",
        "eps_surp_winsor_iqr_outlier",
    ]
    for c in iqr_flag_cols:
        if c in df.columns:
            # False/NaN -> 0, True -> 1
            df[c] = df[c].fillna(False).astype(int)

    # ---------- Standardise column order to match fun_f ----------
    final_cols = [
        "date_ny",
        "symbol",
        "dividend",
        "div_event_flag",
        "div_amount",
        "div_negative_flag",
        "split_ratio",
        "split_flag",
        "split_mult",
        "split_nonpos_flag",
        "shares_out",
        "shares_out_nonpos_flag",
        "shares_out_ffill",
        "log_shares_out",
        "shares_out_roll_mean_252",
        "shares_out_roll_std_252",
        "shares_out_roll_z_252",
        "earnings_flag",
        "eps_estimate",
        "eps_actual",
        "eps_surp_pct",
        "eps_surp_pct_calc",
        "eps_surp_pct_final",
        "eps_surp_winsor",
        "eps_surp_rz_8",
        "dividend_iqr_outlier",
        "div_amount_iqr_outlier",
        "split_ratio_iqr_outlier",
        "split_mult_iqr_outlier",
        "shares_out_iqr_outlier",
        "log_shares_out_iqr_outlier",
        "eps_estimate_iqr_outlier",
        "eps_actual_iqr_outlier",
        "eps_surp_pct_final_iqr_outlier",
        "eps_surp_winsor_iqr_outlier",
        "div_amount_rz_8",
        "eps_estimate_rz_8",
        "eps_actual_rz_8",
        "5d_before_earnings_flag",
        "year",
        "month",
        "day",
    ]

    # Ensure all columns exist; missing ones will be created as NaN
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[final_cols].copy()

    # ---------- Save to disk ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{ticker}_fun_f.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {ticker}: Fundamental features written to: {out_path}")
    print(f"[INFO] {ticker}: Shape = {df.shape}")


# ================= Main pipeline =================

def main():
    # Build trading calendar once and reuse for all tickers
    if not MARKET_REF_FILE.is_file():
        raise FileNotFoundError(f"Market reference file not found: {MARKET_REF_FILE}")
    trade_dates = build_trading_calendar(MARKET_REF_FILE, date_col=MARKET_DATE_COL)
    print(f"[INFO] Trading calendar loaded from {MARKET_REF_FILE}")
    print(f"[INFO] Trading dates: {len(trade_dates)} (from {trade_dates[0]} to {trade_dates[-1]})")

    for ticker in TICKERS:
        process_one_ticker(ticker, trade_dates)


if __name__ == "__main__":
    main()
