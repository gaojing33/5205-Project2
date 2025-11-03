# -*- coding: utf-8 -*-
"""
pipeline_market_origin_to_mar_f.py

End-to-end pipeline:

- Read data/market/{SYMBOL}_market_data_origin.csv
- Clean missing values / basic anomalies
- Ensure there is a row for every minute within the existing intraday range
  for each trading day; if a minute is missing, insert a new row and fill
  all numeric columns by taking the linear interpolation between previous
  and next rows (equivalent to averaging neighbors for a single-gap case)
- Build rich intraday features (returns, volume, volatility, EMA/VWAP
  deviations, RSI, MACD, intraday seasonality, etc.)
- Drop intraday warm-up (first 200 minutes after daily open)
- Add spillover features from competitor stocks (their lret_1 lags 1–5),
  with robust non-leaky filling
- Apply IQR-based robust z-standardization to vol_* and spill_* families
- Drop redundant / collinear raw columns, keep a compact feature set
- Filter out data before 2024-04-30 to avoid cold-start
- Write final file to data/features/market_features/{SYMBOL}_mar_f.csv
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

# ========= PATHS & GLOBAL CONFIG =========

# Project root: adjust if needed
BASE = Path("data")
MARKET_DIR = BASE / "market"
OUT_DIR = BASE / "features" / "market_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Symbols to process / use for spillover
SYMBOLS = ["AMAT", "AVGO", "NVDA", "AMD", "MU"]
TS_COL = "datetime"
RET1_COL = "lret_1"
MOD_COL = "minute_of_day"

# Cleaning / feature engineering parameters
WARMUP_MIN = 200            # Drop first 200 minutes after daily open (per day)
RET_WINSOR_Q = (0.01, 0.99)
HL_WINSOR_Q = (0.01, 0.99)
VWAP_DEV_CAP = 0.10
RSI_CLIP = (1.0, 99.0)
BANDPOS_CLIP = (-3.0, 3.0)
KEEP_RET_FAMILY = "lret"    # Keep log-return family
DROP_LEVEL_COLUMNS = True   # Drop ema_*, vwap_*, bb_* (level columns)
DROP_MACD_COMPONENTS = True # Drop macd, macd_signal (keep macd_hist & z)
EPS = 1e-12

# IQR robust z parameters
WINDOW = 120
MINP = 60
WARM_MINP = 10
KEEP_META = True            # If False, also drop datetime / symbol

# Start date to avoid cold-start region
START_DATE = pd.Timestamp("2024-04-30")


# ========= SMALL UTILITIES =========

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast floats/ints to float32/int32 to reduce memory."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64", "Int64"]).columns:
        df[c] = df[c].astype("int32", errors="ignore")
    return df


def add_report(rows, **kw):
    """Append a dict row to report list."""
    rows.append(kw)


def find_ts_column(df: pd.DataFrame) -> str:
    """Heuristically find a timestamp column name."""
    for c in ["datetime", "timestamp", "time", "Date", "date"]:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "date" in cl:
            return c
    raise ValueError("Cannot find a timestamp column; need datetime/timestamp/date.")


def parse_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Parse timestamp column and add _date (calendar date) and _dow (day-of-week)."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().all():
        raise ValueError(f"{ts_col} cannot be parsed as datetime")
    df = df.sort_values(ts_col).reset_index(drop=True)
    df["_date"] = df[ts_col].dt.date
    df["_dow"] = df[ts_col].dt.dayofweek.astype("int16")
    return df


def ensure_minute_as_is(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a minute_of_day column (integer minutes since local midnight).
    - If exists, coerce to int and recompute where invalid.
    - If not, compute from timestamp column.
    """
    df = df.copy()
    ts_col = find_ts_column(df)
    if "minute_of_day" in df.columns:
        df["minute_of_day"] = pd.to_numeric(
            df["minute_of_day"], errors="coerce"
        ).astype("Int64")
        if df["minute_of_day"].isna().any():
            day0 = df[ts_col].dt.normalize()
            df["minute_of_day"] = (
                df[ts_col] - day0
            ).dt.total_seconds().div(60).astype("Int64")
    else:
        day0 = df[ts_col].dt.normalize()
        df["minute_of_day"] = (
            df[ts_col] - day0
        ).dt.total_seconds().div(60).astype("Int64")
    df["minute_of_day"] = df["minute_of_day"].astype("int32")
    return df


def winsorize_series(s: pd.Series, q_low=0.01, q_high=0.99):
    """Winsorize a series at given quantiles."""
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)


def robust_z(series: pd.Series):
    """Median + MAD robust z-score."""
    med = series.median()
    mad = (series - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(
            np.zeros(len(series)), index=series.index, dtype="float32"
        )
    return (1.4826 * (series - med) / (mad + EPS)).astype("float32")


def pct_dev(a: pd.Series, b: pd.Series):
    """Percentage deviation a/b - 1 with safe denominator."""
    return a / (b.replace(0, np.nan) + EPS) - 1.0


def safe_div(a, b):
    """Safe division a/b avoiding zero denominator."""
    return a / (b.replace(0, np.nan) + EPS)


# ========= ENSURE CONTIGUOUS MINUTE GRID =========

def fill_missing_minutes_per_day(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Ensure that for each calendar date, the intraday series has one row per minute
    between the first and last timestamp of that day.

    For missing minutes:
      - Insert a new row at the missing timestamp.
      - For all numeric columns (except minute_of_day), fill with linear interpolation
        over time (for single missing minute, this is exactly the average of the
        previous and next rows).
      - Recompute minute_of_day from timestamp after reindexing.
      - Non-numeric columns are left as-is (can be filled later or are unused).

    This is applied before deeper cleaning / feature engineering so that rolling
    windows and return-like features see a regular 1-minute grid.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)

    # Group by calendar date to avoid filling overnight gaps
    groups = []
    for date_val, day_df in df.groupby(df[ts_col].dt.date, sort=True):
        if day_df.empty:
            continue
        day_df = day_df.sort_values(ts_col)

        start_ts = day_df[ts_col].min()
        end_ts = day_df[ts_col].max()
        full_index = pd.date_range(start=start_ts, end=end_ts, freq="T")

        # Remove minute_of_day before reindex; we will recompute it
        if "minute_of_day" in day_df.columns:
            day_df = day_df.drop(columns=["minute_of_day"])

        day_df = day_df.set_index(ts_col).reindex(full_index)

        # Restore timestamp column
        day_df[ts_col] = day_df.index

        # Interpolate numeric columns (except minute_of_day, which is not present now)
        num_cols = day_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            day_df[num_cols] = day_df[num_cols].interpolate(
                method="linear", limit_direction="both"
            )

        groups.append(day_df.reset_index(drop=True))

    if not groups:
        return df

    merged = pd.concat(groups, axis=0, ignore_index=True)

    # Recompute minute_of_day after reindexing
    day0 = merged[ts_col].dt.normalize()
    merged["minute_of_day"] = (
        merged[ts_col] - day0
    ).dt.total_seconds().div(60).astype("int32")

    return merged


# ========= DOMAIN RULE CHECKS (OPTIONAL REPORTING) =========

def domain_rule_checks(df: pd.DataFrame, ts_col: str, symbol: str) -> pd.DataFrame:
    """
    Perform basic domain sanity checks (non-negative prices, OHLC order consistency,
    Bollinger band ordering, RSI bounds, etc.). Returns a report DataFrame.
    """
    rows = []
    has = set(df.columns).__contains__

    for r in df.itertuples(index=True):
        ts = getattr(r, ts_col)

        # Non-negative prices/volume
        for feat in ("open", "high", "low", "close", "volume"):
            if has(feat):
                v = getattr(r, feat)
                if pd.notna(v) and v < 0:
                    add_report(
                        rows,
                        symbol=symbol,
                        timestamp=ts,
                        feature=feat,
                        value=v,
                        rule="non-negative",
                        details=">=0",
                    )

        # OHLC ordering
        if all(has(c) for c in ("open", "high", "low", "close")):
            o, h, l, c = r.open, r.high, r.low, r.close
            if pd.notna(o) and pd.notna(h) and pd.notna(l) and pd.notna(c):
                if h < max(o, c, l):
                    add_report(
                        rows,
                        symbol=symbol,
                        timestamp=ts,
                        feature="high",
                        value=h,
                        rule="high>=max(open,close,low)",
                        details=f"max={max(o, c, l)}",
                    )
                if l > min(o, c, h):
                    add_report(
                        rows,
                        symbol=symbol,
                        timestamp=ts,
                        feature="low",
                        value=l,
                        rule="low<=min(open,close,high)",
                        details=f"min={min(o, c, h)}",
                    )

        # hl_range consistency
        if all(has(c) for c in ("high", "low", "hl_range")):
            H, L, R = r.high, r.low, r.hl_range
            if pd.notna(H) and pd.notna(L) and pd.notna(R):
                if R < 0:
                    add_report(
                        rows,
                        symbol=symbol,
                        timestamp=ts,
                        feature="hl_range",
                        value=R,
                        rule="hl_range>=0",
                        details="negative",
                    )
                if abs((H - L) - R) > 1e-6:
                    add_report(
                        rows,
                        symbol=symbol,
                        timestamp=ts,
                        feature="hl_range",
                        value=R,
                        rule="hl_range≈H-L",
                        details=f"expected≈{H-L}",
                    )

        # typical_price consistency
        if all(has(c) for c in ("high", "low", "close", "typical_price")):
            H, L, C, TP = r.high, r.low, r.close, r.typical_price
            if (
                pd.notna(H)
                and pd.notna(L)
                and pd.notna(C)
                and pd.notna(TP)
            ):
                exp = (H + L + C) / 3.0
                if abs(exp - TP) > 1e-6:
                    add_report(
                        rows,
                        symbol=symbol,
                        timestamp=ts,
                        feature="typical_price",
                        value=TP,
                        rule="tp≈(H+L+C)/3",
                        details=f"expected≈{exp}",
                    )

        # macd_hist consistency
        if all(has(c) for c in ("macd", "macd_signal", "macd_hist")):
            m, s, h = r.macd, r.macd_signal, r.macd_hist
            if pd.notna(m) and pd.notna(s) and pd.notna(h):
                if abs((m - s) - h) > 1e-6:
                    add_report(
                        rows,
                        symbol=symbol,
                        timestamp=ts,
                        feature="macd_hist",
                        value=h,
                        rule="macd_hist=macd-macd_signal",
                        details=f"expected≈{m-s}",
                    )

        # Bollinger band ordering
        if all(has(c) for c in ("bb_dn", "bb_mid", "bb_up")):
            dn, mid, up = r.bb_dn, r.bb_mid, r.bb_up
            if (
                pd.notna(dn)
                and pd.notna(mid)
                and pd.notna(up)
                and not (dn <= mid <= up)
            ):
                add_report(
                    rows,
                    symbol=symbol,
                    timestamp=ts,
                    feature="bb_*",
                    value=f"({dn},{mid},{up})",
                    rule="bb_dn<=bb_mid<=bb_up",
                    details="ordering violated",
                )

        # RSI bounds
        if has("rsi_14"):
            rr = r.rsi_14
            if pd.notna(rr) and (rr < 0 or rr > 100):
                add_report(
                    rows,
                    symbol=symbol,
                    timestamp=ts,
                    feature="rsi_14",
                    value=rr,
                    rule="0<=RSI<=100",
                    details="out of bounds",
                )

    return pd.DataFrame(rows)


# ========= BASE CLEANING =========

def clean_base(df: pd.DataFrame, ts_col: str, symbol: str):
    """
    Base cleaning:
      - Forward-fill and repair OHLC per day
      - Volume repair and outlier winsorization (log scale, per minute-of-day)
      - Build volume_wins
      - Build robust z reports for return family (not used as columns)
      - Clip RSI and create rsi_14_clip
      - Create f_volu_z_*_clipped from volu_z_* columns
    """
    df = df.sort_values([ts_col, "minute_of_day"]).reset_index(drop=True)

    # OHLC cleaning per day
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["close"] = df.groupby("_date", observed=True)["close"].ffill()

    if "open" in df.columns and "close" in df.columns:
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        op_na = df["open"].isna()
        df.loc[op_na, "open"] = df["close"].shift(1)
        df["open"] = df.groupby("_date", observed=True)["open"].ffill()

    if all(c in df.columns for c in ("high", "low", "open", "close")):
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        hi_na = df["high"].isna()
        lo_na = df["low"].isna()
        df.loc[hi_na, "high"] = np.maximum(
            df.loc[hi_na, "open"].values, df.loc[hi_na, "close"].values
        )
        df.loc[lo_na, "low"] = np.minimum(
            df.loc[lo_na, "open"].values, df.loc[lo_na, "close"].values
        )
        df["high"] = df.groupby("_date", observed=True)["high"].ffill()
        df["low"] = df.groupby("_date", observed=True)["low"].ffill()

    # Volume cleaning
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["volume"] = (
            df.groupby("_date", observed=True)["volume"]
            .ffill()
            .fillna(0.0)
        )

    # Volume winsorization on log scale per minute-of-day
    if "volume" in df.columns:
        vlog = np.log1p(df["volume"])
        lo = vlog.groupby(df["minute_of_day"], observed=True).transform(
            lambda g: g.quantile(0.005)
        )
        hi = vlog.groupby(df["minute_of_day"], observed=True).transform(
            lambda g: g.quantile(0.995)
        )
        vlog_w = vlog.clip(lower=lo, upper=hi)
        df["volume_wins"] = np.expm1(vlog_w).astype("float32")

    # Return-family robust z report (not kept as features)
    reports = []
    ret_cols = [c for c in df.columns if c.startswith("ret_")] + [
        c for c in df.columns if c.startswith("lret_")
    ]

    for col in ret_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        sw = winsorize_series(s, *RET_WINSOR_Q)

        def _rz(x: pd.Series):
            med = x.rolling(120, min_periods=60).median()
            rmad = x.rolling(120, min_periods=60).apply(
                lambda y: (
                    pd.Series(y) - pd.Series(y).median()
                ).abs().median(),
                raw=False,
            )
            return 1.4826 * (x - med) / (rmad + EPS)

        z = (
            df.groupby("_date", observed=True)[
                sw.name if hasattr(sw, "name") else col
            ]
            .apply(lambda g: _rz(sw.loc[g.index]))
            .reset_index(level=0, drop=True)
        )
        bad = z.abs() > 5.0
        for idx in bad[bad].index:
            add_report(
                reports,
                symbol=symbol,
                timestamp=df.loc[idx, ts_col],
                feature=col,
                value=float(df.loc[idx, col])
                if pd.notna(df.loc[idx, col])
                else np.nan,
                rule="rolling-robust-z>5 (w=120)",
                details=f"z={float(z.loc[idx]):.2f}",
            )

    stat_df = pd.DataFrame(reports)

    # RSI clipping
    if "rsi_14" in df.columns:
        rsi = pd.to_numeric(df["rsi_14"], errors="coerce")
        df["rsi_14_clip"] = rsi.clip(*RSI_CLIP).astype("float32")

    # volu_z_* -> f_volu_z_*_clipped
    for k in [5, 15, 30, 60]:
        col = f"volu_z_{k}"
        if col in df.columns:
            df[f"f_{col}_clipped"] = (
                pd.to_numeric(df[col], errors="coerce")
                .clip(-5, 5)
                .astype("float32")
            )

    return df, stat_df


# ========= ATR20 =========

def add_atr20(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR_20 based on high/low/close."""
    if not all(c in df.columns for c in ("high", "low", "close")):
        return df
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    df["ATR_20"] = (
        tr.ewm(span=20, adjust=False, min_periods=5).mean().astype("float32")
    )
    return df


# ========= FEATURE ENGINEERING =========

def build_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Build higher-level features:
      - Volume seasonality adjustment and volume ratio/z
      - Rolling robust z for return family
      - Volatility features (hl_range_pct, hl_over_ATR)
      - EMA/VWAP deviations and robust z
      - OC change and z
      - Bollinger band position
      - RSI position and z
      - MACD histogram z
      - Intraday phase encoding (sin/cos)
    """
    df = df.sort_values([ts_col, "minute_of_day"]).reset_index(drop=True)

    # Volume seasonality and ratios
    if "volume_wins" in df.columns:
        grp = df.groupby("minute_of_day", observed=True)["volume_wins"]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, np.nan)
        df["f_volume_norm"] = (
            (df["volume_wins"] - mu) / (sd + EPS)
        ).astype("float32")

        for k in [5, 15, 30, 60]:
            mk = f"volu_mean_{k}"
            if mk in df.columns:
                ratio = safe_div(
                    pd.to_numeric(df["volume_wins"], errors="coerce"),
                    pd.to_numeric(df[mk], errors="coerce"),
                ) - 1.0
                df[f"f_vol_ratio_{k}"] = ratio.astype("float32")
                df[f"fz_vol_ratio_{k}"] = robust_z(df[f"f_vol_ratio_{k}"])

    # Rolling robust z for return family
    ret_cols = [c for c in df.columns if c.startswith("ret_")] + [
        c for c in df.columns if c.startswith("lret_")
    ]
    for col in ret_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        sw = winsorize_series(s, *RET_WINSOR_Q)

        def _rz(x: pd.Series):
            med = x.rolling(120, min_periods=60).median()
            rmad = x.rolling(120, min_periods=60).apply(
                lambda y: (
                    pd.Series(y) - pd.Series(y).median()
                ).abs().median(),
                raw=False,
            )
            return 1.4826 * (x - med) / (rmad + EPS)

        z = (
            df.groupby("_date", observed=True)[
                sw.name if hasattr(sw, "name") else col
            ]
            .apply(lambda g: _rz(sw.loc[g.index]))
            .reset_index(level=0, drop=True)
        )
        df[f"fz_{col}_rolling"] = z.astype("float32")

    # Volatility features
    if all(c in df.columns for c in ("high", "low", "close")):
        h = pd.to_numeric(df["high"], errors="coerce")
        l = pd.to_numeric(df["low"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")
        hl_pct = winsorize_series(
            (h - l) / (c.replace(0, np.nan) + EPS), *HL_WINSOR_Q
        )
        df["f_hl_range_pct"] = hl_pct.astype("float32")
        df["fz_hl_range_pct"] = robust_z(df["f_hl_range_pct"])
        df = add_atr20(df)
        if "ATR_20" in df.columns:
            f = (h - l) / (
                pd.to_numeric(df["ATR_20"], errors="coerce") + EPS
            )
            df["f_hl_over_atr20"] = f.astype("float32")
            df["fz_hl_over_atr20"] = robust_z(df["f_hl_over_atr20"])

    # Price vs EMA/VWAP deviations
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")

        for n in [20, 50, 200]:
            col = f"ema_{n}"
            if col in df.columns:
                base = pd.to_numeric(df[col], errors="coerce")
                dev = pct_dev(close, base)
                df[f"f_dev_close_ema{n}"] = dev.astype("float32")
                df[f"fz_dev_close_ema{n}"] = robust_z(
                    df[f"f_dev_close_ema{n}"]
                )

        for k in [5, 15, 30, 60]:
            col = f"vwap_{k}"
            if col in df.columns:
                base = pd.to_numeric(df[col], errors="coerce")
                dev = pct_dev(close, base).clip(
                    -VWAP_DEV_CAP, VWAP_DEV_CAP
                )
                df[f"f_dev_close_vwap_{k}"] = dev.astype("float32")
                df[f"fz_dev_close_vwap_{k}"] = robust_z(
                    df[f"f_dev_close_vwap_{k}"]
                )

        if "typical_price" in df.columns and "ema_20" in df.columns:
            tp = pd.to_numeric(df["typical_price"], errors="coerce")
            e20 = pd.to_numeric(df["ema_20"], errors="coerce")
            dev = pct_dev(tp, e20)
            df["f_dev_tp_ema20"] = dev.astype("float32")
            df["fz_dev_tp_ema20"] = robust_z(df["f_dev_tp_ema20"])

        # OC change
        oc = None
        if "oc_change" in df.columns:
            oc_raw = pd.to_numeric(df["oc_change"], errors="coerce")
            if oc_raw.abs().median(skipna=True) < 1:
                oc = oc_raw
        if oc is None and "open" in df.columns:
            op = pd.to_numeric(df["open"], errors="coerce")
            oc = safe_div(close, op) - 1.0
        if oc is not None:
            ocw = winsorize_series(oc, *RET_WINSOR_Q)
            df["f_oc_change_pct"] = ocw.astype("float32")
            df["fz_oc_change_pct"] = robust_z(df["f_oc_change_pct"])

    # Bollinger band position
    if all(c in df.columns for c in ("bb_mid", "bb_up", "bb_dn")) and "close" in df.columns:
        mid = pd.to_numeric(df["bb_mid"], errors="coerce")
        up = pd.to_numeric(df["bb_up"], errors="coerce")
        dn = pd.to_numeric(df["bb_dn"], errors="coerce")
        width = (up - dn).replace(0, np.nan)
        bandpos = (
            pd.to_numeric(df["close"], errors="coerce") - mid
        ) / (width + EPS)
        df["f_bb_pos"] = bandpos.clip(*BANDPOS_CLIP).astype("float32")
        df["fz_bb_pos"] = robust_z(df["f_bb_pos"])

    # RSI strength
    if "rsi_14" in df.columns:
        src = (
            pd.to_numeric(df.get("rsi_14_clip", df["rsi_14"]), errors="coerce")
            .clip(*RSI_CLIP)
        )
        pos = (src - 50.0) / 50.0
        df["f_rsi_pos"] = pos.astype("float32")
        df["fz_rsi_pos"] = robust_z(df["f_rsi_pos"])

    # MACD
    if "macd_hist" in df.columns or (
        "macd" in df.columns and "macd_signal" in df.columns
    ):
        if "macd_hist" not in df.columns:
            m = pd.to_numeric(df.get("macd"), errors="coerce")
            ms = pd.to_numeric(df.get("macd_signal"), errors="coerce")
            df["macd_hist"] = (m - ms).astype("float32")
        df["fz_macd_hist"] = robust_z(
            pd.to_numeric(df["macd_hist"], errors="coerce")
        )

    # Intraday phase encoding
    md = df["minute_of_day"].astype("float32")
    L = (
        df.groupby("_date", observed=True)["minute_of_day"]
        .transform("max")
        .astype("float32")
        + 1.0
    )
    phase = 2.0 * math.pi * (md / (L + EPS))
    df["f_minsin"] = np.sin(phase).astype("float32")
    df["f_mincos"] = np.cos(phase).astype("float32")

    return df


# ========= ORIGIN -> BASE FEATURES (WITH MINUTE FILL & WARM-UP DROP) =========

def process_symbol_origin_to_features(sym: str, origin_df: pd.DataFrame):
    """
    From origin CSV for one symbol:
      - Normalize column names
      - Parse timestamps
      - Ensure minute grid is contiguous within each day (insert missing minutes)
      - Recompute _date / _dow / minute_of_day after filling
      - Run domain checks (optional)
      - Base cleaning
      - Feature engineering
      - Drop warm-up region (first WARMUP_MIN minutes after daily open)
      - Drop helper columns and attach symbol
    """
    df = origin_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: c.lower() for c in df.columns})

    ts_col = find_ts_column(df)
    df = parse_timestamp(df, ts_col)
    df = ensure_minute_as_is(df)

    # Ensure every minute within the intraday range has a row (per day)
    df = fill_missing_minutes_per_day(df, ts_col)

    # Rebuild _date / _dow / minute_of_day on the updated grid
    df = parse_timestamp(df, ts_col)
    df = ensure_minute_as_is(df)

    # Optional domain checks (not written to disk here; can be extended)
    _rule_df = domain_rule_checks(df, ts_col, sym)

    # Base cleaning
    df_clean, _stat_df = clean_base(df, ts_col, sym)

    # Feature engineering
    df_feat = build_features(df_clean, ts_col)

    # Drop EMA/VWAP/Bollinger level columns
    if DROP_LEVEL_COLUMNS:
        drop_cols = [
            c
            for c in df_feat.columns
            if c.startswith(("ema_", "vwap_", "bb_"))
        ]
        if "typical_price" in df_feat.columns:
            drop_cols.append("typical_price")
        df_feat = df_feat.drop(columns=drop_cols, errors="ignore")

    # Keep only one return family
    if KEEP_RET_FAMILY.lower() == "lret":
        drop_ret = [
            c
            for c in df_feat.columns
            if c.startswith("ret_") and not c.startswith("lret_")
        ]
        drop_ret += [
            f"fz_{c}_rolling"
            for c in [d.replace("_wins", "") for d in drop_ret]
        ]
        df_feat = df_feat.drop(columns=drop_ret, errors="ignore")
    else:
        drop_lret = [c for c in df_feat.columns if c.startswith("lret_")]
        drop_lret += [
            f"fz_{c}_rolling"
            for c in [d.replace("_wins", "") for d in drop_lret]
        ]
        df_feat = df_feat.drop(columns=drop_lret, errors="ignore")

    if DROP_MACD_COMPONENTS:
        df_feat = df_feat.drop(
            columns=[c for c in ("macd", "macd_signal") if c in df_feat.columns],
            errors="ignore",
        )

    # Warm-up drop: first WARMUP_MIN minutes after daily open
    day_open = df_feat.groupby("_date", observed=True)[
        "minute_of_day"
    ].transform("min")
    warm_mask = df_feat["minute_of_day"] < (day_open + WARMUP_MIN)
    df_feat = df_feat.loc[~warm_mask].reset_index(drop=True)

    # Drop helper columns
    for c in ["_date", "_dow"]:
        if c in df_feat.columns:
            df_feat.drop(columns=[c], inplace=True)

    df_feat["symbol"] = sym
    df_feat = downcast(df_feat)
    return df_feat


# ========= TIME NORMALIZATION: NEW YORK LOCAL, NO TZ =========

def to_ny_naive(series: pd.Series) -> pd.Series:
    """
    Convert a timestamp series to naive America/New_York local time
    (drop timezone after conversion).
    """
    dt = pd.to_datetime(series, errors="coerce", utc=None)
    if getattr(dt.dtype, "tz", None) is not None:
        dt_ny = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        dt_local = dt.dt.tz_localize(
            "America/New_York",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        dt_ny = dt_local.dt.tz_localize(None)
    return dt_ny


# ========= ORIGINAL DATA FOR SPILLOVER (datetime, lret_1) =========

def read_original_for_spill(src_df: pd.DataFrame) -> pd.DataFrame:
    """
    From original market CSV:
      - Parse datetime as New York local time
      - Ensure lret_1 exists (compute from close if missing)
      - Keep only [datetime, lret_1]
    """
    df = src_df.copy()
    if TS_COL not in df.columns:
        raise ValueError(f"original data missing {TS_COL}")
    df[TS_COL] = to_ny_naive(df[TS_COL])

    if RET1_COL not in df.columns:
        if "close" not in df.columns:
            raise ValueError(
                f"original data missing {RET1_COL} and close for fallback"
            )
        c = pd.to_numeric(df["close"], errors="coerce")
        df[RET1_COL] = np.log(c).diff()

    df = (
        df[[TS_COL, RET1_COL]]
        .copy()
        .sort_values(TS_COL)
        .reset_index(drop=True)
    )
    return df


# ========= NEAREST LEFT/RIGHT INDEX HELPERS =========

def nearest_left_right(
    s_idx: pd.DatetimeIndex, s_vals: np.ndarray, x_time: pd.Timestamp
):
    """
    Given a time index and values, find nearest valid value to the left and to the
    right of x_time. Return (left_time, left_val, right_time, right_val); any side
    can be None if no valid value exists.
    """
    pos = s_idx.searchsorted(x_time, side="left")

    # Left side
    li = pos - 1
    left_time, left_val = None, None
    while li >= 0:
        v = s_vals[li]
        if pd.notna(v):
            left_time, left_val = s_idx[li], v
            break
        li -= 1

    # Right side
    ri = pos
    right_time, right_val = None, None
    n = len(s_idx)
    while ri < n:
        v = s_vals[ri]
        if pd.notna(v):
            right_time, right_val = s_idx[ri], v
            break
        ri += 1

    return left_time, left_val, right_time, right_val


def safe_fill_one_target(
    s_idx, s_vals, x_time: pd.Timestamp, t_time: pd.Timestamp, k: int
):
    """
    For spillover lag target x = t - k minutes, construct a non-leaky fill:
      - Right-side value must satisfy <= t-1min; for k=1, right side is not allowed.
      - If both left and right are valid, use their average.
      - If only one side is valid, use that side.
      - If neither side is valid, return None (caller applies column-level fallback).
    """
    l_t, l_v, r_t, r_v = nearest_left_right(s_idx, s_vals, x_time)

    # Right-side non-leakage constraint
    if r_t is not None and r_t > (t_time - pd.Timedelta(minutes=1)):
        r_t, r_v = None, None
    if k == 1:
        # For k=1, any right-side value is considered too close (may be at t itself)
        r_t, r_v = None, None

    if (l_v is not None) and (r_v is not None):
        return (l_v + r_v) / 2.0
    if l_v is not None:
        return l_v
    if r_v is not None:
        return r_v
    return None


# ========= SPILLOVER (WITH ROBUST FILLING) =========

def add_spillover(
    base_df: pd.DataFrame, comp_map: dict, base_sym: str, all_symbols: list
) -> pd.DataFrame:
    """
    For each competitor symbol, construct spillover features:
      spill_{COMP}_lret_lag{k}, k=1..5

    For each (t, k):
      - Target time x = t - k minutes.
      - First try to get exact lret_1 at x.
      - If missing, search nearest left/right valid values (with non-leaky
        constraints) via safe_fill_one_target.
      - If still missing, fallback to the median lret_1 of that competitor.
      - Final guarantee: no NaN in spill_* columns.
    """
    out = base_df.copy()
    times = out[TS_COL].values  # ndarray[datetime64[ns]]

    for comp in [s for s in all_symbols if s != base_sym]:
        if comp not in comp_map:
            print(f"[WARN] Skip spillover {base_sym} <- {comp}: no original data.")
            continue

        src = comp_map[comp].dropna(subset=[TS_COL]).copy()
        s = src.set_index(TS_COL)[RET1_COL].astype("float64")
        s_idx = s.index
        s_vals = s.values
        series_median = (
            np.nanmedian(s_vals)
            if np.isfinite(np.nanmedian(s_vals))
            else 0.0
        )

        for k in range(1, 6):
            colname = f"spill_{comp}_lret_lag{k}"
            filled = np.empty(len(times), dtype="float64")
            filled[:] = np.nan
            k_delta = np.timedelta64(k, "m")

            for i, t_time in enumerate(times):
                x_time = (t_time - k_delta).astype("datetime64[ns]")
                val = s.get(pd.Timestamp(x_time))
                if pd.isna(val):
                    v = safe_fill_one_target(
                        s_idx,
                        s_vals,
                        pd.Timestamp(x_time),
                        pd.Timestamp(t_time),
                        k,
                    )
                    if v is None or not np.isfinite(v):
                        v = series_median
                else:
                    v = float(val)
                filled[i] = v

            # Defensive column-level fallback
            if np.isnan(filled).any():
                col_median = np.nanmedian(filled)
                if not np.isfinite(col_median):
                    col_median = 0.0
                nan_mask = np.isnan(filled)
                filled[nan_mask] = col_median

            out[colname] = filled.astype("float32")

    return out


# ========= META FEATURES: date_ny / minute_of_day+1 / DOW ONE-HOT =========

def add_date_ny(df: pd.DataFrame) -> pd.DataFrame:
    """Add date_ny as 'YYYY/M/D' string based on datetime."""
    dt = df[TS_COL]
    df["date_ny"] = (
        dt.dt.year.astype(str)
        + "/"
        + dt.dt.month.astype(str)
        + "/"
        + dt.dt.day.astype(str)
    )
    return df


def bump_minute_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift minute_of_day by +1 (keeping as Int64).
    This matches previously used conventions where minute_of_day starts from 1.
    """
    df[MOD_COL] = pd.to_numeric(df[MOD_COL], errors="coerce").astype("Int64")
    df[MOD_COL] = (df[MOD_COL] + 1).astype("Int64")
    return df


def add_dow_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add day-of-week one-hot features dow_0..dow_6.
    Drop any existing 'dow' column to avoid duplication.
    """
    if "dow" in df.columns:
        df = df.drop(columns=["dow"])
    dow = df[TS_COL].dt.dayofweek  # 0=Mon, 6=Sun
    for d in range(7):
        df[f"dow_{d}"] = (dow == d).astype("int8")
    return df


def add_spillover_and_meta(
    sym: str, base_df: pd.DataFrame, orig_for_spill: dict, all_syms: list
) -> pd.DataFrame:
    """
    Starting from per-symbol features:
      - Normalize datetime to New York naive
      - Add date_ny
      - Shift minute_of_day by +1
      - Add spillover features
      - Add DOW one-hot
      - Downcast dtypes and ensure no NaN in spill_* columns
    """
    df = base_df.copy()
    df[TS_COL] = to_ny_naive(df[TS_COL])
    df = add_date_ny(df)
    df = bump_minute_of_day(df)
    df = add_spillover(df, orig_for_spill, sym, all_syms)
    df = add_dow_onehot(df)
    df = downcast(df)

    # Final NaN check for spill_* columns
    spill_cols = [c for c in df.columns if c.startswith("spill_")]
    for c in spill_cols:
        if df[c].isna().any():
            med = df[c].median()
            if pd.isna(med):
                med = 0.0
            df[c] = df[c].fillna(med).astype("float32")
    return df


# ========= IQR-BASED ROBUST Z FOR BLOCKS =========

def robust_z_iqr_block_past_only(
    block: pd.DataFrame, window=120, minp=60, warm_minp=10
) -> pd.DataFrame:
    """
    IQR-based robust z for a block of numeric features, using only historical data:
      - Use rolling quantiles on shifted values (S = X.shift(1))
      - median = rolling Q50(S), IQR = Q75(S) - Q25(S)
      - z_roll = 1.349 * (X - median) / (IQR + eps)
      - If rolling window is not available (early part), fallback to expanding
        quantiles with min_periods=warm_minp
      - Non-finite values are replaced with 0.0; final output is float32
    """
    if block.empty:
        return block.copy()

    X = block.apply(pd.to_numeric, errors="coerce")
    S = X.shift(1)  # only use historical values

    q50 = S.rolling(window=window, min_periods=minp).quantile(
        0.5, interpolation="linear"
    )
    q25 = S.rolling(window=window, min_periods=minp).quantile(
        0.25, interpolation="linear"
    )
    q75 = S.rolling(window=window, min_periods=minp).quantile(
        0.75, interpolation="linear"
    )
    iqr = q75 - q25

    z_roll = 1.349 * (X - q50) / (iqr + EPS)

    need_exp = q50.isna() | iqr.isna()
    if need_exp.any().any():
        q50e = S.expanding(min_periods=warm_minp).quantile(
            0.5, interpolation="linear"
        )
        q25e = S.expanding(min_periods=warm_minp).quantile(
            0.25, interpolation="linear"
        )
        q75e = S.expanding(min_periods=warm_minp).quantile(
            0.75, interpolation="linear"
        )
        iqre = q75e - q25e
        z_exp = 1.349 * (X - q50e) / (iqre + EPS)
        Z = z_roll.where(~need_exp, z_exp)
    else:
        Z = z_roll

    return (
        Z.where(np.isfinite(Z), 0.0)
        .fillna(0.0)
        .astype("float32")
    )


# ========= FINAL CLEANING / STANDARDIZATION / DATE FILTER =========

def finalize_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning:
      - Apply IQR-based robust z to spill_* and vol_* families
      - Drop raw spill_* and vol-related raw/collinear features
      - Keep selected fz_dev_close_ema* / fz_dev_close_vwap* subset
      - Drop dow_6 (keep others)
      - Downcast types
      - Filter out rows before START_DATE
    """
    df = df.copy()
    blocks = []

    # spill_* -> fz_spill_*
    spill_cols = [c for c in df.columns if c.startswith("spill_")]
    if spill_cols:
        z_spill = robust_z_iqr_block_past_only(
            df[spill_cols], window=WINDOW, minp=MINP, warm_minp=WARM_MINP
        )
        z_spill.columns = [
            c.replace("spill_", "fz_spill_") for c in z_spill.columns
        ]
        blocks.append(z_spill)

    # vol_* -> fz_vol_*
    vol_cols = [c for c in df.columns if re.fullmatch(r"vol_\d+", c)]
    if vol_cols:
        z_vol = robust_z_iqr_block_past_only(
            df[vol_cols], window=WINDOW, minp=MINP, warm_minp=WARM_MINP
        )
        z_vol.columns = [f"fz_{c}" for c in z_vol.columns]
        blocks.append(z_vol)

    if blocks:
        df = pd.concat([df] + blocks, axis=1)

    # Drop redundant / collinear features
    drop_cols = []
    raw_like_patterns = [
        r"^spill_.*$",                      # keep only fz_spill_* standardized
        r"^f_vol_ratio_\d+$",
        r"^f_hl_range_pct$",
        r"^ATR_20$",
        r"^f_hl_over_atr20$",
        r"^f_dev_close_ema(20|50|200)$",
        r"^f_dev_close_vwap_(5|15|30|60)$",
        r"^f_dev_tp_ema20$",
        r"^f_oc_change_pct$",
        r"^f_bb_pos$",
        r"^rsi_14$",
        r"^f_rsi_pos$",
        r"^macd_hist$",
        r"^lret_.*$",
        r"^volu_mean_(5|15|30|60)$",
    ]
    for pat in raw_like_patterns:
        drop_cols += [c for c in df.columns if re.fullmatch(pat, c)]

    # Keep only selected ema/vwap z features
    keep_ema = {"fz_dev_close_ema20", "fz_dev_close_ema200"}
    drop_cols += [
        c
        for c in df.columns
        if re.fullmatch(r"fz_dev_close_ema(20|50|200)", c)
        and c not in keep_ema
    ]

    keep_vwap = {"fz_dev_close_vwap_15", "fz_dev_close_vwap_60"}
    drop_cols += [
        c
        for c in df.columns
        if re.fullmatch(r"fz_dev_close_vwap_(5|15|30|60)", c)
        and c not in keep_vwap
    ]

    # Drop dow_6; keep date_ny and minute_of_day
    base_drop = ["dow_6"]
    drop_cols += [c for c in base_drop if c in df.columns]

    if not KEEP_META:
        drop_cols += [k for k in [TS_COL, "symbol"] if k in df.columns]

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df = downcast(df)

    # Filter by START_DATE (cold-start cut)
    df = df[df[TS_COL] >= START_DATE].reset_index(drop=True)

    return df


# ========= MAIN PIPELINE =========

def main():
    # Pre-read original data for spillover (per symbol)
    orig_for_spill = {}
    for s in SYMBOLS:
        path = MARKET_DIR / f"{s}_market_data_origin.csv"
        if not path.exists():
            print(f"[WARN] {s}: missing {path}; will not be used as spillover source.")
            continue
        raw = pd.read_csv(path)
        orig_for_spill[s] = read_original_for_spill(raw)

    all_syms_for_spill = list(orig_for_spill.keys())
    if not all_syms_for_spill:
        print("[WARN] No original data available for spillover; will build single-symbol features only.")

    # Run full pipeline per symbol
    for sym in SYMBOLS:
        origin_path = MARKET_DIR / f"{sym}_market_data_origin.csv"
        if not origin_path.exists():
            print(f"[WARN] {sym}: skip, missing {origin_path}")
            continue

        print(f"[INFO] {sym}: reading {origin_path}")
        origin_df = pd.read_csv(origin_path)

        # 1) origin -> base features (with minute filling and warm-up drop)
        feat_df = process_symbol_origin_to_features(sym, origin_df)

        # 2) Add meta & spillover
        feat_spill = add_spillover_and_meta(
            sym, feat_df, orig_for_spill, all_syms_for_spill
        )

        # 3) Final standardization & pruning & date filter
        final_df = finalize_clean(feat_spill)

        # 4) Save to {SYMBOL}_mar_f.csv
        out_path = OUT_DIR / f"{sym}_mar_f.csv"
        final_df.to_csv(out_path, index=False)
        print(
            f"[OK] {sym}: wrote {out_path} "
            f"(rows={len(final_df)}, cols={final_df.shape[1]})"
        )

    print("[DONE] All symbols processed.")


if __name__ == "__main__":
    main()
