# -*- coding: utf-8 -*-
"""
Check column consistency and build ALL_* feature files.

Directory structure (relative to this script):
  ../data/features/fundamental_features
  ../data/features/market_features
  ../data/features/sentiment_features

Outputs:
  ../data/features/fundamental_features/ALL_fun_f.csv
  ../data/features/market_features/ALL_mark_f.csv
  ../data/features/sentiment_features/ALL_sen_core_f.csv
  ../data/features/sentiment_features/ALL_sen_sat_f.csv

Note: For sentiment features we assume file patterns:
  <SYMBOL>_sen_core_f.csv
  <SYMBOL>_sen_sat_f.csv
where SYMBOL in {MU, AMD, NVDA, AVGO, AMAT}.
"""

import sys
from pathlib import Path

import pandas as pd


def check_columns_for_files(csv_files, group_name=""):
    """
    Check whether all CSV files in this list share identical column names.

    Parameters
    ----------
    csv_files : list[pathlib.Path]
        List of CSV file paths to be checked.
    group_name : str
        A label for logging (e.g. folder name or subgroup name).

    Returns
    -------
    bool
        True if all files share the same columns (in the same order),
        False otherwise.
    """
    if not csv_files:
        print(f"[WARN] {group_name}: no CSV files found, skip column check.")
        return True

    if len(csv_files) == 1:
        print(f"[INFO] {group_name}: only one CSV file, skip column consistency check.")
        return True

    print(f"[CHECK] Column consistency for group: {group_name}")
    ref_file = csv_files[0]
    ref_cols = pd.read_csv(ref_file, nrows=0).columns.tolist()
    print(f"  Reference file: {ref_file.name}")
    ok = True

    for f in csv_files[1:]:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if cols != ref_cols:
            ok = False
            ref_set = set(ref_cols)
            cols_set = set(cols)
            missing = sorted(ref_set - cols_set)
            extra = sorted(cols_set - ref_set)
            print(f"  [MISMATCH] {f.name}")
            if missing:
                print(f"    - Missing columns (vs reference): {missing}")
            if extra:
                print(f"    - Extra columns (vs reference): {extra}")

    if ok:
        print(f"[OK] {group_name}: all CSV files share the same columns.")

    return ok


def concat_files(csv_files, out_path):
    """
    Concatenate a list of CSV files row-wise and save to out_path.

    Parameters
    ----------
    csv_files : list[pathlib.Path]
        List of CSV file paths to be concatenated.
    out_path : pathlib.Path
        Output CSV path.
    """
    if not csv_files:
        print(f"[WARN] No files to concatenate for {out_path.name}.")
        return

    dfs = []
    for f in csv_files:
        print(f"[LOAD] {f}")
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path} | shape = {combined.shape}")


def main():
    # Base directory: folder where this script is located
    script_dir = Path(__file__).resolve().parent

    # Project root is assumed to be the parent of data_alignment
    project_root = script_dir.parent

    # Feature folders (relative to project root)
    features_dir = project_root / "data" / "features"
    fund_dir = features_dir / "fundamental_features"
    market_dir = features_dir / "market_features"
    sent_dir = features_dir / "sentiment_features"

    # --- Fundamental features ---
    if fund_dir.exists():
        fund_files = sorted(
            f for f in fund_dir.glob("*.csv")
            if not f.name.startswith("ALL_")
        )
        check_columns_for_files(fund_files, "fundamental_features")
        concat_files(fund_files, fund_dir / "ALL_fun_f.csv")
    else:
        print(f"[WARN] Directory not found: {fund_dir}")

    # --- Market features ---
    if market_dir.exists():
        market_files = sorted(
            f for f in market_dir.glob("*.csv")
            if not f.name.startswith("ALL_")
        )
        check_columns_for_files(market_files, "market_features")
        concat_files(market_files, market_dir / "ALL_mark_f.csv")
    else:
        print(f"[WARN] Directory not found: {market_dir}")

    # --- Sentiment features ---
    if sent_dir.exists():
        # Core sentiment files: <SYMBOL>_sen_core_f.csv
        core_files = sorted(
            f for f in sent_dir.glob("*_sen_core_f.csv")
            if not f.name.startswith("ALL_")
        )
        check_columns_for_files(core_files, "sentiment_features (core)")
        concat_files(core_files, sent_dir / "ALL_sen_core_f.csv")

        # Satellite sentiment files: <SYMBOL>_sen_sat_f.csv
        sat_files = sorted(
            f for f in sent_dir.glob("*_sen_sat_f.csv")
            if not f.name.startswith("ALL_")
        )
        check_columns_for_files(sat_files, "sentiment_features (sat)")
        concat_files(sat_files, sent_dir / "ALL_sen_sat_f.csv")
    else:
        print(f"[WARN] Directory not found: {sent_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
