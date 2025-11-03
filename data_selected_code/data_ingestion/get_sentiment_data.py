# -*- coding: utf-8 -*-
"""
data_ingestion/get_sentiment_data.py

Dependencies:
  pip install google-cloud-bigquery google-cloud-bigquery-storage pandas python-dateutil pytz

What this script does:
  - Queries GDELT (via BigQuery) for five tickers' company mentions and tone fields.
  - Converts raw articles to per-minute sentiment rows WITHOUT any extra time mapping:
      * We only aggregate by the exact New York local minute each article was published in
        (UTC → America/New_York → floor to minute). We do NOT map pre/post to other minutes/days.
  - Builds the required time-derived columns and outputs one CSV per ticker:
        data/sentiment/AMAT_sentiment_data_otigin.csv
        data/sentiment/AMD_sentiment_data_otigin.csv
        data/sentiment/AVGO_sentiment_data_otigin.csv
        data/sentiment/MU_sentiment_data_otigin.csv
        data/sentiment/NVDA_sentiment_data_otigin.csv

Date window:
  - Start: 2023-04-01 (inclusive)
  - End  : 2025-10-28 (inclusive)

Output columns (per minute with ≥1 news):
  - symbol
  - datetime             (e.g., "2024/4/30 12:50:00", NY local time; date parts have no leading zeros)
  - date_ny              (e.g., "2024/4/30")
  - year, month, day     (integers, e.g., 2024, 4, 30)
  - minute               ("HH:MM")
  - minute_of_day_1based (1..1440, first minute from 00:00 is 1)
  - dow_0..dow_6         (one-hot weekday, 0=Mon,...,6=Sun)
  - n_news               (# articles in that minute)
  - tone_sum             (sum of GDELT V2Tone.tone in that minute)
  - tone_mean            (mean of GDELT V2Tone.tone in that minute)
  - pos_sum              (sum of GDELT V2Tone.positive in that minute)
  - neg_sum              (sum of GDELT V2Tone.negative in that minute)

Notes:
  - Requires ADC credentials for BigQuery (e.g., `gcloud auth application-default login`)
    or GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON.
  - We keep a finance-domain whitelist to reduce noise; adjust FIN_DOMAIN_WHITELIST if needed.
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
from dateutil import tz

from google.cloud import bigquery
try:
    from google.cloud import bigquery_storage
    _HAS_BQ_STORAGE = True
except Exception:
    _HAS_BQ_STORAGE = False

# ============================== Configuration ===============================

# Five NASDAQ US tickers
TICKERS = ["AMAT", "AMD", "AVGO", "MU", "NVDA"]

# Date window (inclusive)
START_DATE = "2023-04-01"
END_DATE   = "2025-10-28"

# Output directory
OUT_DIR = Path("data") / "sentiment"

# Optional publisher whitelist to increase financial relevance
FIN_DOMAIN_WHITELIST = [
    "reuters.com","bloomberg.com","wsj.com","cnbc.com",
    "marketwatch.com","barrons.com","seekingalpha.com",
    "finance.yahoo.com","investorplace.com","thestreet.com",
    "nasdaq.com","zacks.com","fool.com"
]

# Time zones
TZ_NY  = tz.gettz("America/New_York")
TZ_UTC = tz.gettz("UTC")

# GCP settings (GDELT is hosted in location=US)
PROJECT_ID  = os.getenv("GOOGLE_CLOUD_PROJECT") or "orbital-gantry-476707-q6"
BQ_LOCATION = "US"

# Company-name regex patterns for GDELT V2Organizations (lowercased and comma-offsets removed)
ORG_REGEX = {
    "NVDA": r"(^|;)\s*nvidia(\s+corporation|(\s+corp\.?)?)?\s*(;|$)",
    "AMD":  r"(^|;)\s*advanced\ micro\ devices(,\ inc\.?)?\s*(;|$)|(^|;)\s*amd\s*(;|$)",
    "AVGO": r"(^|;)\s*broadcom(\s+inc\.?|(\s+corp\.?)?)\s*(;|$)|(^|;)\s*avago\s*(;|$)",
    "AMAT": r"(^|;)\s*applied\ materials(,\ inc\.?)?\s*(;|$)",
    "MU":   r"(^|;)\s*micron\ technology(,\ inc\.?)?\s*(;|$)|(^|;)\s*micron\s*(;|$)",
}

# ============================== Helpers =====================================

def _build_domain_filter_sql() -> str:
    """Builds SQL predicate for a domain allow-list on the URL field."""
    if not FIN_DOMAIN_WHITELIST:
        return ""
    conds = []
    for dom in FIN_DOMAIN_WHITELIST:
        esc = re.escape(dom)
        conds.append(f"REGEXP_CONTAINS(url, r'https?://([^/]*{esc})')")
    return "AND (" + " OR ".join(conds) + ")"

def build_query(ticker: str, start: str, end: str, use_domain_filter: bool = True):
    """
    Returns (SQL, job_config) to pull publisher/url/v2tone fields from GDELT GKG partitioned table
    within [start, end] by partition date, filtering rows mentioning the target company.
    """
    org_regex = ORG_REGEX[ticker]
    domain_filter_sql = _build_domain_filter_sql() if use_domain_filter else ""
    sql = f"""
    WITH base AS (
      SELECT
        DATE,
        SourceCommonName AS publisher,
        DocumentIdentifier AS url,
        REGEXP_REPLACE(LOWER(V2Organizations), r',\\d+', '') AS orgs_lc,
        V2Tone AS v2tone
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE DATE(_PARTITIONTIME) >= @d_start
        AND DATE(_PARTITIONTIME) <= @d_end
    )
    SELECT
      DATE,
      publisher,
      url,
      SAFE_CAST(SPLIT(v2tone, ',')[SAFE_ORDINAL(1)] AS FLOAT64) AS tone,
      SAFE_CAST(SPLIT(v2tone, ',')[SAFE_ORDINAL(2)] AS FLOAT64) AS pos,
      SAFE_CAST(SPLIT(v2tone, ',')[SAFE_ORDINAL(3)] AS FLOAT64) AS neg
    FROM base
    WHERE REGEXP_CONTAINS(orgs_lc, @org_re)
      {domain_filter_sql}
    """
    params = [
        bigquery.ScalarQueryParameter("d_start", "DATE", start),
        bigquery.ScalarQueryParameter("d_end",   "DATE", end),
        bigquery.ScalarQueryParameter("org_re",  "STRING", org_regex),
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return sql, job_config

def bq_client():
    """Creates a BigQuery client (ADC or service-account JSON required)."""
    return bigquery.Client(project=PROJECT_ID)

def yyyymmddhhmmss_to_utc(s: str) -> pd.Timestamp:
    """Parses GDELT DATE like 20250101123456 as a UTC timestamp."""
    return pd.to_datetime(s, format="%Y%m%d%H%M%S", utc=True)

# ============================== Core Logic ===================================

def fetch_raw_articles(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Executes the BigQuery job and returns a de-duplicated article-level DataFrame."""
    client = bq_client()
    try:
        bqs = bigquery_storage.BigQueryReadClient() if _HAS_BQ_STORAGE else None
    except Exception:
        bqs = None

    sql, cfg = build_query(ticker, start, end, use_domain_filter=True)
    job = client.query(sql, job_config=cfg, location=BQ_LOCATION)
    df = job.result().to_dataframe(bqstorage_client=bqs) if bqs is not None else job.result().to_dataframe()

    if df.empty:
        # Return consistent schema with empty timestamp column
        df["published_utc"] = pd.to_datetime([])
        return df[["publisher", "url", "published_utc", "tone", "pos", "neg"]]

    # Parse & clean
    df["published_utc"] = df["DATE"].astype(str).apply(yyyymmddhhmmss_to_utc)
    df = df.drop(columns=["DATE"]).drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["tone"] = df["tone"].fillna(0.0)
    df["pos"]  = df["pos"].fillna(0.0)
    df["neg"]  = df["neg"].fillna(0.0)
    return df[["publisher", "url", "published_utc", "tone", "pos", "neg"]]

def aggregate_to_minute_sentiment(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Aggregates raw articles to New York local minutes WITHOUT any extra re-mapping:
      - Convert published_utc → America/New_York
      - Drop tz info (keep wall-clock NY time), then floor to minute
      - Group by minute and build features
    """
    if raw_df.empty:
        cols = [
            "symbol","datetime","date_ny","year","month","day","minute",
            "minute_of_day_1based",
            "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5","dow_6",
            "n_news","tone_sum","tone_mean","pos_sum","neg_sum"
        ]
        return pd.DataFrame(columns=cols)

    # 1) UTC → 纽约本地时间（带时区）
    ts_ny = raw_df["published_utc"].dt.tz_convert(TZ_NY)

    # 2) 去掉时区信息（保留纽约本地“挂钟时间”），再按分钟截断
    ts_ny_min = ts_ny.dt.tz_localize(None).dt.floor("min")

    # 3) 按分钟聚合
    grp = (
        raw_df.assign(ts_ny_min=ts_ny_min)
        .groupby("ts_ny_min", as_index=False)
        .agg(
            n_news=("url", "count"),
            tone_sum=("tone", "sum"),
            tone_mean=("tone", "mean"),
            pos_sum=("pos", "sum"),
            neg_sum=("neg", "sum"),
        )
    )

    dt = grp["ts_ny_min"]
    year  = dt.dt.year
    month = dt.dt.month
    day   = dt.dt.day
    hour  = dt.dt.hour
    minute_num = dt.dt.minute

    datetime_str = (
        year.astype(str) + "/" + month.astype(str) + "/" + day.astype(str) + " " +
        hour.astype(str).str.zfill(2) + ":" + minute_num.astype(str).str.zfill(2) + ":00"
    )
    date_ny_str = year.astype(str) + "/" + month.astype(str) + "/" + day.astype(str)
    minute_of_day_1based = (hour * 60 + minute_num) + 1

    dow = dt.dt.dayofweek
    dummies = pd.get_dummies(dow, prefix="dow")
    for i in range(7):
        col = f"dow_{i}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"dow_{i}" for i in range(7)]].astype(int)

    minute_str = hour.astype(str).str.zfill(2) + ":" + minute_num.astype(str).str.zfill(2)

    out = pd.DataFrame({
        "symbol": symbol,
        "datetime": datetime_str,
        "date_ny": date_ny_str,
        "year": year.values,
        "month": month.values,
        "day": day.values,
        "minute": minute_str.values,
        "minute_of_day_1based": minute_of_day_1based.values,
        "n_news": grp["n_news"].values,
        "tone_sum": grp["tone_sum"].values,
        "tone_mean": grp["tone_mean"].values,
        "pos_sum": grp["pos_sum"].values,
        "neg_sum": grp["neg_sum"].values,
    })
    out = pd.concat([out, dummies.reset_index(drop=True)], axis=1)

    cols = [
        "symbol","datetime","date_ny","year","month","day","minute",
        "minute_of_day_1based",
        "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5","dow_6",
        "n_news","tone_sum","tone_mean","pos_sum","neg_sum"
    ]
    return out[cols]

def run_for_symbol(symbol: str, start: str, end: str, outdir: Path) -> Path:
    """Fetches raw articles for a symbol, aggregates to minute sentiment, writes CSV, and returns its path."""
    raw = fetch_raw_articles(symbol, start, end)
    minute_df = aggregate_to_minute_sentiment(raw, symbol)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{symbol}_sentiment_data_otigin.csv"
    minute_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {symbol}: wrote {len(minute_df):,} rows -> {out_path}")
    return out_path

# ============================== Main =========================================

def main():
    print(f"[INFO] GCP project={PROJECT_ID}, location={BQ_LOCATION}")
    for tic in TICKERS:
        try:
            run_for_symbol(tic, START_DATE, END_DATE, OUT_DIR)
        except Exception as e:
            print(f"[ERROR] {tic}: {e}")
    print("[DONE]")

if __name__ == "__main__":
    main()
