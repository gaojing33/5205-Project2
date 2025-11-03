# -*- coding: utf-8 -*-
r"""
Build sen_core_f / sen_sat_f from origin sentiment CSVs (AMAT/AMD/NVDA/MU/AVGO)

What this script does (end-to-end):
1) Load origin minute-level sentiment events from:
     <IN_DIR>\{SYMBOL}_sentiment_data_otigin.csv
   (NOTE: The filename intentionally uses "otigin" to match your files.)
   The origin file is sparse (only minutes that have news) and includes:
     ['symbol','datetime','date_ny','year','month','day','minute',
      'minute_of_day_1based','dow_0'..'dow_6','n_news','tone_sum','tone_mean','pos_sum','neg_sum']

2) Create a complete trading-minute panel (09:30-15:59, Monday–Friday) for each business day
   from FETCH_START_DATE (warm-up to avoid cold start) to END_DATE.
   Then, map each news timestamp to a “tradable minute” using:
     - pre-open (<09:30) → same day 09:31
     - regular hours [09:30,16:00) → that minute (floor)
     - after close (>=16:00) → next business day 09:31
   Aggregate minute counts/sums onto this panel.

3) Feature engineering:
   - Rolling counts: cnt_5m / 15m / 30m / 60m
   - EWMs: tone_ewm15 / tone_ewm30
   - Has-news mask; minute embeddings minute_sin / minute_cos (by wall-clock day length)
   - “Distance since last hit”: mins_since_last_news
   - Level & changes: abs_tone_mean, tone_mean_delta, tone_mean_ewm_hl15
   - Direction proxies: n_pos_raw / n_neg_raw
   - Carry features per trading_date using the raw events (NOT the trading panel):
       overnight_* (16:00 ~ next 09:30), morning_* (09:30 ~ 12:50)
     Missing carry → set to 0 to keep schema stable.

4) Robust normalization (no look-ahead):
   - Group by 'minute_from_open' (09:30=1, …, 15:59=390), across the last ROLL_DAYS trading days
     (require at least MINP valid days)
   - Counts/timers: log1p → IQR robust z  → *_rs
   - Sums (e.g., tone_sum): signed_log → IQR robust z → *_rs
   - Means/smoothed/deltas: symmetric clip (1%/99%) within group → IQR robust z → *_rs
   - Local (same-group) MAD robust z for n_news and tone_mean: local_rz_*

5) Warm-up deletion: drop the first WARMUP_DROP_MINUTES minutes after open (ranked by minute_from_open),
   which removes 09:30–12:49 so the final output starts at 12:50+ for each day.

6) Final time range cut: keep EFFECTIVE_START_DATE ~ END_DATE inclusive. (You asked to start from 2024-04-30.)

7) Core vs Satellite feature selection across all 5 symbols:
   - BASE_COLS are always kept.
   - CORE = features that pass minimal activity/variance checks in ALL symbols.
   - SAT  = features that pass in at least K_SAT symbols (default 3).
   - Save:
       <OUT_DIR>\{SYMBOL}_sen_core_f.csv
       <OUT_DIR>\{SYMBOL}_sen_sat_f.csv
     (and the ALL_* counterparts if you want to enable them at the bottom)

Author’s note:
- This script is self-contained (no BigQuery dependencies).
- It intentionally mirrors the logic of your three helper scripts while adapting to the
  available columns in the origin CSVs. Missing publisher/url → source_div features default to 0.
"""

import os
import re
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# -------- CONFIG ---------
# =========================
TICKERS = ["AMAT", "AMD", "AVGO", "MU", "NVDA"]
IN_DIR  = r"data\sentiment"
OUT_DIR = r"data\features\sentiment_features"

# Build with a long warm-up to avoid cold start, but cut to the effective range at the very end
FETCH_START_DATE     = "2024-01-01"   # warm-up start (build all features first)
EFFECTIVE_START_DATE = "2024-04-30"   # final output must start from this date (inclusive)
END_DATE             = "2025-10-28"   # final output end date (inclusive)

# Minute windows & normalization knobs
ROLL_DAYS   = 60            # cross-day rolling window for robust baselines
MINP        = 10            # minimal days in window
CLIP_Q_LOW  = 0.01          # symmetric clipping for mean-like series
CLIP_Q_HIGH = 0.99

# Drop the first 200 minutes each day (09:30~12:49) AFTER feature construction
WARMUP_DROP_MINUTES = 200

# Satellite selection: feature must be “valid” in at least K symbols
K_SAT             = 3
NONZERO_RATIO_MIN = 1e-3
VAR_MIN           = 1e-12

# Base columns we always keep in outputs (schema stability)
BASE_COLS = [
    "datetime","symbol","date_ny",
    "minute_of_day_1based","minute_sin","minute_cos",
    "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5","dow_6",
    "has_news",
    # carry features (filled with zeros if unavailable)
    "overnight_n","overnight_tone_sum","overnight_tone_mean","overnight_source_div",
    "morning_n","morning_tone_sum","morning_tone_mean","morning_source_div",
]
# Canonical market calendar (use AMAT market features as trading-day reference)
MARKET_CAL_PATH = r"data\features\market_features\AMAT_mar_f.csv"

# ==============================
# -------- UTILITIES -----------
# ==============================
def bday_range(start: date, end: date) -> List[date]:
    """Business-day range (Mon-Fri). This is a simple weekday filter (US holidays not excluded)."""
    idx = pd.bdate_range(start=start, end=end, freq="B")
    return [d.date() for d in idx]


def minute_grid_for_day(d: date) -> pd.DatetimeIndex:
    """Return 390 trading minutes [09:30..15:59] for the given date (naive local time)."""
    start_min = datetime(d.year, d.month, d.day, 9, 30)
    return pd.date_range(start=start_min, periods=390, freq="min")  # 09:30..15:59


def minute_grid(d_start: date, d_end: date) -> pd.DatetimeIndex:
    """Concatenate trading-minute grids for all business days in [d_start, d_end]."""
    all_idx = []
    for d in bday_range(d_start, d_end):
        all_idx.append(minute_grid_for_day(d))
    if not all_idx:
        return pd.DatetimeIndex([])
    out = all_idx[0]
    for r in all_idx[1:]:
        out = out.append(r)
    return out


def next_business_day(d: date) -> date:
    """Next weekday (Mon-Fri)."""
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:  # 5=Sat, 6=Sun
        nd += timedelta(days=1)
    return nd


def map_to_trading_min(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Map a naive local timestamp to a trading-minute bucket:
      - if time < 09:30  → same day 09:31
      - if 09:30 ≤ t <16:00 → keep that minute (floor)
      - if t ≥ 16:00 → map to NEXT business day 09:31
    """
    t = ts.time()
    open_t  = datetime(ts.year, ts.month, ts.day, 9, 30).time()
    close_t = datetime(ts.year, ts.month, ts.day, 16, 0).time()
    if t >= close_t:
        nd = next_business_day(ts.date())
        return datetime(nd.year, nd.month, nd.day, 9, 31)
    elif t < open_t:
        return datetime(ts.year, ts.month, ts.day, 9, 31)
    else:
        # already in regular hours; floor to minute
        return ts.replace(second=0, microsecond=0)


def minute_from_open(ts: pd.Timestamp) -> int:
    """Return 1..390 for 09:30..15:59; may return <0 or >390 if ts outside trading hours (we guard callers)."""
    return int(((ts - ts.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() // 60) - 9*60 - 30) + 1


def signed_log(x: pd.Series) -> pd.Series:
    """Signed log transform: sign(x)*log(1+|x|)"""
    xv = x.astype(float)
    return np.sign(xv) * np.log1p(np.abs(xv))


def robust_z_from_iqr(x: pd.Series, med: pd.Series, iqr: pd.Series) -> pd.Series:
    z = (x - med) / iqr.replace(0, np.nan)
    return z.fillna(0.0)


def robust_z_from_mad(x: pd.Series, med: pd.Series, mad: pd.Series) -> pd.Series:
    z = (x - med) / (1.4826 * mad.replace(0, np.nan))
    return z.fillna(0.0)


def group_rolling_quantile(df: pd.DataFrame, value_col: str, q: float) -> pd.Series:
    """
    Group by ['minute_from_open'] across days; order by date_ny; shift() to avoid lookahead;
    then do rolling quantile over ROLL_DAYS with MINP.
    """
    s = (df.sort_values(["minute_from_open", "date_ny"])
           .groupby("minute_from_open", group_keys=False)[value_col]
           .apply(lambda x: x.shift().rolling(ROLL_DAYS, min_periods=MINP).quantile(q)))
    return s.reindex(df.index)


def group_rolling_median(df: pd.DataFrame, value_col: str) -> pd.Series:
    s = (df.sort_values(["minute_from_open", "date_ny"])
           .groupby("minute_from_open", group_keys=False)[value_col]
           .apply(lambda x: x.shift().rolling(ROLL_DAYS, min_periods=MINP).median()))
    return s.reindex(df.index)


def group_rolling_iqr(df: pd.DataFrame, value_col: str) -> pd.Series:
    q1 = group_rolling_quantile(df, value_col, 0.25)
    q3 = group_rolling_quantile(df, value_col, 0.75)
    return (q3 - q1)


def group_rolling_mad(df: pd.DataFrame, value_col: str) -> Tuple[pd.Series, pd.Series]:
    """
    MAD = median(|x - median(x)|) within the rolling window grouped by minute_from_open across days.
    """
    def _mad_window(x):
        m = np.median(x)
        return np.median(np.abs(x - m))
    s = (df.sort_values(["minute_from_open", "date_ny"])
           .groupby("minute_from_open", group_keys=False)[value_col]
           .apply(lambda x: x.shift().rolling(ROLL_DAYS, min_periods=MINP).apply(_mad_window, raw=False)))
    med = group_rolling_median(df, value_col)
    return med, s.reindex(df.index)


# ========================================
# -------- CARRY FEATURE BUILDING --------
# ========================================
def build_daily_carry(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build carry features by trading_date using raw event timestamps (naive local time).
    Windows:
      - overnight: [prev 16:00, this 09:30)
      - morning : [this 09:30, this 12:50)
    Return columns:
      ['trading_date',
       'overnight_n','overnight_tone_sum','overnight_tone_mean','overnight_source_div',
       'morning_n','morning_tone_sum','morning_tone_mean','morning_source_div']
    """
    cols = ["trading_date","overnight_n","overnight_tone_sum","overnight_tone_mean","overnight_source_div",
            "morning_n","morning_tone_sum","morning_tone_mean","morning_source_div"]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    df = raw.copy()
    ts = pd.to_datetime(df["datetime"])
    df["ts"] = ts
    df["t_only"] = ts.dt.time

    # Map each row to "trading_date" consistent with panel alignment
    # (after close counts towards NEXT day)
    close_t = datetime(2000,1,1,16,0).time()
    def _trading_date_for_row(r_ts: pd.Timestamp) -> date:
        if r_ts.time() >= close_t:
            return next_business_day(r_ts.date())
        return r_ts.date()

    df["trading_date"] = df["ts"].apply(_trading_date_for_row)

    # Helper masks
    def is_overnight(t):
        return (t >= datetime(2000,1,1,16,0).time()) or (t < datetime(2000,1,1,9,30).time())
    def is_morning(t):
        return (t >= datetime(2000,1,1,9,30).time()) and (t < datetime(2000,1,1,12,50).time())

    df["is_overnight"] = df["t_only"].apply(is_overnight)
    df["is_morning"]   = df["t_only"].apply(is_morning)

    # Aggregate
    ov = (df[df["is_overnight"]]
          .groupby("trading_date", as_index=False)
          .agg(overnight_n=("n_news","sum"),
               overnight_tone_sum=("tone_sum","sum")))
    mo = (df[df["is_morning"]]
          .groupby("trading_date", as_index=False)
          .agg(morning_n=("n_news","sum"),
               morning_tone_sum=("tone_sum","sum")))

    carry = pd.merge(ov, mo, on="trading_date", how="outer").fillna(0.0)
    # means + dummy source_div (0; origin files lack publisher/url)
    carry["overnight_tone_mean"] = np.where(carry["overnight_n"]>0,
                                            carry["overnight_tone_sum"]/carry["overnight_n"], 0.0)
    carry["morning_tone_mean"]   = np.where(carry["morning_n"]>0,
                                            carry["morning_tone_sum"]/carry["morning_n"], 0.0)
    carry["overnight_source_div"] = 0.0
    carry["morning_source_div"]   = 0.0

    # Ensure column order
    carry = carry[["trading_date",
                   "overnight_n","overnight_tone_sum","overnight_tone_mean","overnight_source_div",
                   "morning_n","morning_tone_sum","morning_tone_mean","morning_source_div"]]
    return carry


# ============================================
# -------- MINUTE PANEL + FEATURES -----------
# ============================================
def build_minute_panel_from_origin(origin: pd.DataFrame,
                                   start_date: date,
                                   end_date: date,
                                   symbol: str) -> pd.DataFrame:
    """
    1) Build complete trading-minute grid for [start_date, end_date].
    2) Map origin events to tradable minutes, aggregate onto the grid.
    3) Create primary time scaffolding columns.
    """
    idx = minute_grid(start_date, end_date)
    panel = pd.DataFrame(index=idx)
    panel.index.name = "datetime"
    panel["symbol"]  = symbol
    panel["n_news"]  = 0.0
    panel["tone_sum"] = 0.0

    # Map sparse events -> tradable minutes
    if not origin.empty:
        df = origin.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["bucket_dt"] = df["datetime"].apply(map_to_trading_min)
        grp = df.groupby("bucket_dt").agg(n=("n_news","sum"),
                                          t=("tone_sum","sum"))
        # intersect on index to be safe
        inter = grp.index.intersection(panel.index)
        panel.loc[inter, "n_news"]   += grp.loc[inter, "n"].values
        panel.loc[inter, "tone_sum"] += grp.loc[inter, "t"].values

    # Derived basics
    panel["n_news"]   = panel["n_news"].fillna(0.0)
    panel["tone_sum"] = panel["tone_sum"].fillna(0.0)
    panel["tone_mean"] = np.where(panel["n_news"]>0, panel["tone_sum"]/panel["n_news"], 0.0)

    # time scaffolding
    dt = panel.index
    panel["date_ny"] = dt.normalize()
    # minute_from_open (for grouping) and rank-from-open (for warm-up drop)
    panel["minute_from_open"] = ((dt - dt.normalize())/pd.Timedelta(minutes=1)).astype(int) - (9*60+30) + 1
    # day length (wall-clock, minutes) to build sin/cos
    minutes_in_day = ((dt.normalize()+pd.Timedelta(days=1))-dt.normalize())/pd.Timedelta(minutes=1)
    minutes_in_day = minutes_in_day.astype(int)
    # embeddings
    panel["minute_of_day_1based"] = (((dt - dt.normalize())/pd.Timedelta(minutes=1)).astype(int) + 1)
    panel["minute_sin"] = np.sin(2*np.pi*(panel["minute_of_day_1based"]-1)/minutes_in_day)
    panel["minute_cos"] = np.cos(2*np.pi*(panel["minute_of_day_1based"]-1)/minutes_in_day)
    # DOW one-hot (Mon=0..Sun=6, but grid only contains weekdays)
    dow = pd.Index(dt).dayofweek
    for k in range(7):
        panel[f"dow_{k}"] = (dow==k).astype(int)

    # Rolling counts
    for w in [5, 15, 30, 60]:
        panel[f"cnt_{w}m"] = panel["n_news"].rolling(w, min_periods=1).sum()

    # EWMs & helpers
    panel["tone_ewm15"] = panel["tone_mean"].ewm(span=15, adjust=False, min_periods=1).mean()
    panel["tone_ewm30"] = panel["tone_mean"].ewm(span=30, adjust=False, min_periods=1).mean()
    panel["has_news"]   = (panel["n_news"]>0).astype(int)

    # Distance since last hit (within each day)
    last_hit = panel["minute_of_day_1based"].where(panel["n_news"]>0).groupby(panel["date_ny"]).ffill()
    panel["mins_since_last_news"] = (panel["minute_of_day_1based"] - last_hit).fillna(1e6).astype(int)

    # Level / change / smoothing
    panel["abs_tone_mean"]        = panel["tone_mean"].abs()
    panel["tone_mean_delta"]      = panel.groupby("date_ny")["tone_mean"].diff().fillna(0.0)
    panel["tone_mean_ewm_hl15"]   = panel["tone_mean"].ewm(halflife=15, adjust=False, min_periods=1).mean()

    # Direction counters
    panel["n_pos_raw"] = panel["n_news"] * (panel["tone_mean"] > 0).astype(int)
    panel["n_neg_raw"] = panel["n_news"] * (panel["tone_mean"] < 0).astype(int)

    panel.reset_index(inplace=True)  # keep 'datetime' as a visible column
    return panel


def add_carry_and_fill(panel: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    """Join carry features and ensure all carry columns exist (fill NaN → 0)."""
    carry = build_daily_carry(origin)
    out = panel.copy()
    out["trading_date"] = out["date_ny"].dt.date

    out = out.merge(carry, on="trading_date", how="left")
    carry_cols = ["overnight_n","overnight_tone_sum","overnight_tone_mean","overnight_source_div",
                  "morning_n","morning_tone_sum","morning_tone_mean","morning_source_div"]
    for c in carry_cols:
        if c not in out.columns:
            out[c] = 0.0
    out[carry_cols] = out[carry_cols].fillna(0.0)
    return out.drop(columns=["trading_date"])


def add_robust_normalizations(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add *_rs robust z features by grouping on minute_from_open across days.
    NOTE: Assumes panel already contains: date_ny (datetime64[ns]), minute_from_open (int)
    """
    df = panel.copy()

    # Guard: minute_from_open should be in [1,390] for regular-hours grid
    df = df[(df["minute_from_open"]>=1) & (df["minute_from_open"]<=390)].copy()

    # 1) Counts / timers: log1p → IQR robust z
    counts_cols = ["n_news","cnt_5m","cnt_15m","cnt_30m","cnt_60m","n_pos_raw","n_neg_raw",
                   "overnight_n","overnight_source_div","morning_n","morning_source_div",
                   "mins_since_last_news"]
    for col in counts_cols:
        if col in df.columns:
            xv = np.log1p(df[col].astype(float))
            med = group_rolling_median(df.assign(_v=xv), "_v")
            iqr = group_rolling_iqr(df.assign(_v=xv), "_v")
            df[f"{col}_rs"] = robust_z_from_iqr(xv, med, iqr)

    # 2) Sums: signed_log → IQR robust z
    sum_cols = ["tone_sum","overnight_tone_sum","morning_tone_sum"]
    for col in sum_cols:
        if col in df.columns:
            xv = signed_log(df[col])
            med = group_rolling_median(df.assign(_v=xv), "_v")
            iqr = group_rolling_iqr(df.assign(_v=xv), "_v")
            df[f"{col}_rs"] = robust_z_from_iqr(xv, med, iqr)

    # 3) Mean-like: symmetric clip → IQR robust z
    mean_cols = ["tone_mean","tone_ewm15","tone_ewm30","tone_mean_delta","tone_mean_ewm_hl15",
                 "overnight_tone_mean","morning_tone_mean"]
    for col in mean_cols:
        if col in df.columns:
            ql = group_rolling_quantile(df, col, CLIP_Q_LOW)
            qh = group_rolling_quantile(df, col, CLIP_Q_HIGH)
            xv = df[col].astype(float).clip(lower=ql, upper=qh)
            med = group_rolling_median(df.assign(_v=xv), "_v")
            iqr = group_rolling_iqr(df.assign(_v=xv), "_v")
            df[f"{col}_rs"] = robust_z_from_iqr(xv, med, iqr)

    # Local surprises via MAD
    med_tone, mad_tone = group_rolling_mad(df.assign(_v=df["tone_mean"].astype(float)), "_v")
    df["local_rz_tone_mean"] = robust_z_from_mad(df["tone_mean"].astype(float), med_tone, mad_tone)

    med_cnt, mad_cnt = group_rolling_mad(df.assign(_v=np.log1p(df["n_news"].astype(float))), "_v")
    df["local_rz_n_news"] = robust_z_from_mad(np.log1p(df["n_news"].astype(float)), med_cnt, mad_cnt)

    # surprise (deviation from group median)
    med_for_tone = group_rolling_median(df, "tone_mean")
    df["surprise_tone_mean"] = (df["tone_mean"].astype(float) - med_for_tone).fillna(0.0)

    # convenience slog column for tone_sum (useful for audits)
    df["tone_sum_slog"] = signed_log(df["tone_sum"])

    return df


def apply_warmup_and_cut(df: pd.DataFrame) -> pd.DataFrame:
    """Drop warm-up minutes (first 200 per day) and then cut final effective range."""
    out = df.copy()
    # rank by minute_from_open within day to delete first 200 minutes (09:30~12:49)
    out["_rank_from_open"] = out.groupby(out["date_ny"])["minute_from_open"].rank(method="first")
    out = out.loc[~((out["_rank_from_open"]>=1) & (out["_rank_from_open"]<=WARMUP_DROP_MINUTES))].copy()
    out = out.drop(columns=["_rank_from_open"])

    # final cut
    eff_start = pd.to_datetime(EFFECTIVE_START_DATE)
    eff_end   = pd.to_datetime(END_DATE)
    out = out.loc[(out["date_ny"]>=eff_start) & (out["date_ny"]<=eff_end)].copy()
    return out

def load_market_calendar(path: str) -> pd.DatetimeIndex:
    """Load canonical trading dates from AMAT_mar_f.csv (column: date_ny)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Market calendar file not found: {path}")
    mkt = pd.read_csv(path, usecols=["date_ny"])
    mkt["date_ny"] = pd.to_datetime(mkt["date_ny"])
    # normalize to date level, drop duplicates, sort
    return mkt["date_ny"].dt.normalize().drop_duplicates().sort_values()


def align_to_market_calendar(df: pd.DataFrame,
                             allowed_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Keep only rows whose date_ny belongs to the canonical market calendar.
    This removes sentiment-only business days (e.g., US holidays with no trading).
    """
    # ensure datetime dtype
    df = df.copy()
    df["date_ny"] = pd.to_datetime(df["date_ny"])
    mask = df["date_ny"].dt.normalize().isin(allowed_dates)
    out = df.loc[mask].copy()
    return out


# ===========================================
# -------- CORE / SAT FEATURE PICKER --------
# ===========================================
def is_numeric_series(s: pd.Series) -> bool:
    return s.dtype.kind in "biufc"


def per_symbol_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for c in df.columns:
        if c in BASE_COLS:
            continue
        s = df[c]
        if not is_numeric_series(s):
            continue
        nz_ratio = float((s != 0).sum() / max(1, len(s)))
        var = float(np.nanvar(s))
        stats[c] = {"nonzero_ratio": nz_ratio, "var": var}
    return pd.DataFrame(stats).T


def build_schema(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    per_stats = {sym: per_symbol_stats(df) for sym, df in dfs.items()}
    all_feats = sorted(set().union(*[st.index for st in per_stats.values() if not st.empty]))
    rows = []
    for f in all_feats:
        row = {"feature": f}
        valid_cnt = 0
        for sym in dfs.keys():
            if f in per_stats[sym].index:
                nzr = per_stats[sym].loc[f, "nonzero_ratio"]
                var = per_stats[sym].loc[f, "var"]
                ok = int((nzr >= NONZERO_RATIO_MIN) and (var >= VAR_MIN))
            else:
                ok = 0
            row[f"ok_{sym}"] = ok
            valid_cnt += ok
        row["ok_sum"] = valid_cnt
        rows.append(row)
    schema = pd.DataFrame(rows).set_index("feature") if rows else pd.DataFrame(columns=["feature"]).set_index("feature")

    core = [f for f in schema.index if schema.loc[f, "ok_sum"] == len(dfs)]
    sat  = [f for f in schema.index if schema.loc[f, "ok_sum"] >= K_SAT]
    # remove base cols if they appeared
    core = [c for c in core if c not in BASE_COLS]
    sat  = [c for c in sat  if c not in BASE_COLS]
    return schema, core, sat


def finalize_and_save(per_symbol_df: Dict[str, pd.DataFrame],
                      core: List[str], sat: List[str]) -> None:
    """
    Save per-symbol {sen_core_f, sen_sat_f}. Add year/month/day/minute at the end for your downstream scripts.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    for tag, chosen in [("core", core), ("sat", sat)]:
        sel_cols = BASE_COLS + sorted(chosen)
        for sym, df in per_symbol_df.items():
            out = df.copy()

            # Ensure base columns exist (already ensured, but double guard)
            for c in BASE_COLS:
                if c not in out:
                    out[c] = 0

            # Column ordering
            out = out[sel_cols].copy()

            # Append friendly columns (year/month/day/minute string "HH:MM")
            out["year"]  = out["date_ny"].dt.year.astype(int)
            out["month"] = out["date_ny"].dt.month.astype(int)
            out["day"]   = out["date_ny"].dt.day.astype(int)
            out["minute"] = out["datetime"].dt.strftime("%H:%M")

            fname = os.path.join(OUT_DIR, f"{sym}_sen_{tag}_f.csv")
            out.to_csv(fname, index=False, encoding="utf-8-sig")
            print(f"[OK] {sym} -> {fname}  shape={out.shape}")

        # If you want ALL_* exports as well, uncomment below:
        # all_df = pd.concat([per_symbol_df[s][sel_cols].assign(year=per_symbol_df[s]["date_ny"].dt.year.astype(int),
        #                                                       month=per_symbol_df[s]["date_ny"].dt.month.astype(int),
        #                                                       day=per_symbol_df[s]["date_ny"].dt.day.astype(int),
        #                                                       minute=per_symbol_df[s]["datetime"].dt.strftime("%H:%M"),
        #                                                       symbol=s)
        #                     for s in per_symbol_df], axis=0, ignore_index=True)
        # fname_all = os.path.join(OUT_DIR, f"ALL_sen_{tag}_f.csv")
        # all_df.to_csv(fname_all, index=False, encoding="utf-8-sig")
        # print(f"[OK] ALL -> {fname_all}  shape={all_df.shape}")


# ============================
# -------- PIPELINE ----------
# ============================
def process_symbol(sym: str) -> pd.DataFrame:
    """Load origin, build panel, engineer features, normalize, warm-up drop, final cut, return dataframe."""
    in_path = os.path.join(IN_DIR, f"{sym}_sentiment_data_otigin.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing input file: {in_path}")

    origin = pd.read_csv(in_path)
    # Sanity: keep only necessary columns; coerce dtypes
    need_cols = {"datetime","n_news","tone_sum"}
    miss = need_cols - set(origin.columns)
    if miss:
        raise RuntimeError(f"{in_path} missing required columns: {sorted(miss)}")

    # Limit origin to warm-up and effective end (carry needs the full raw timeline)
    origin["datetime"] = pd.to_datetime(origin["datetime"])
    origin = origin.loc[(origin["datetime"].dt.date >= pd.to_datetime(FETCH_START_DATE).date())
                        & (origin["datetime"].dt.date <= pd.to_datetime(END_DATE).date())].copy()

    # Build complete trading-minute panel & aggregate
    start_d = pd.to_datetime(FETCH_START_DATE).date()
    end_d   = pd.to_datetime(END_DATE).date()
    panel = build_minute_panel_from_origin(origin, start_d, end_d, sym)

    # Merge carry
    panel = add_carry_and_fill(panel, origin)

    # Add robust normalizations (grouped by minute_from_open across days)
    panel = add_robust_normalizations(panel)

    # Apply warm-up deletion and final effective cut
    panel = apply_warmup_and_cut(panel)

    # Keep a consistent column ordering: put BASE_COLS first, then engineered columns (will be pruned later)
    # Ensure BASE_COLS exist
    for c in BASE_COLS:
        if c not in panel:
            panel[c] = 0
    ordered = BASE_COLS + [c for c in panel.columns if c not in BASE_COLS]
    panel = panel[ordered].copy()

    return panel


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) load canonical trading calendar from AMAT market features
    market_dates = load_market_calendar(MARKET_CAL_PATH)
    print(f"[INFO] market calendar loaded, {len(market_dates)} unique trading days")

    per_symbol_df: Dict[str, pd.DataFrame] = {}
    for sym in TICKERS:
        print(f"[RUN] {sym}  {FETCH_START_DATE} → {END_DATE}")
        df_sym = process_symbol(sym)

        # 2) align sentiment dates to market trading calendar
        before = len(df_sym)
        df_sym = align_to_market_calendar(df_sym, market_dates)
        after = len(df_sym)
        print(f"[CAL] {sym} calendar alignment: {before} -> {after} rows "
              f"({before - after} removed)")

        per_symbol_df[sym] = df_sym
        print(f"[DONE] {sym} shape={per_symbol_df[sym].shape}")

    # Build schema / decide core & satellite features
    schema, core, sat = build_schema(per_symbol_df)
    # (Optional) Save schema audits for your inspection:
    schema_path = os.path.join(OUT_DIR, "schema_stats_sentiment.csv")
    schema.to_csv(schema_path, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "schema_core_features.txt"), "w", encoding="utf-8") as f:
        for c in sorted(core):
            f.write(c + "\n")
    with open(os.path.join(OUT_DIR, "schema_sat_features.txt"), "w", encoding="utf-8") as f:
        for c in sorted(sat):
            f.write(c + "\n")
    print(f"[INFO] core={len(core)}  sat={len(sat)}  (K_SAT={K_SAT})  -> {schema_path}")

    # Export per-symbol CSVs
    finalize_and_save(per_symbol_df, core, sat)

    print("[ALL DONE]")

if __name__ == "__main__":
    main()
