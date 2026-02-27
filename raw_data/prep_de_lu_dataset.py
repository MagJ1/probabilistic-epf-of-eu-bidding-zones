#!/usr/bin/env python3
"""
Build 7-day lagged features FIRST, then split DE-LU train_val into train + val at cutoff.

Input (default):
  raw_data/single_bid_zones/de_lu/de_lu_train_val.csv

Process:
  1) Load CSV
  2) Add lagged feature columns using "right order" handling for duplicate ds per unique_id
     (drops rows where lag not available, i.e., first `lag_in_days` days)
  3) Split by cutoff timestamp:
       train: ds <= cutoff   (cutoff INCLUDED)
       val:   ds >  cutoff

Outputs (default, beside input):
  de_lu_train.csv
  de_lu_val.csv

Usage examples:
  python split_train_val_with_lags.py

  python split_train_val_with_lags.py \
    --features y is_weekend is_holiday hour_sin hour_cos day_of_week_sin day_of_week_cos month_sin month_cos

  python split_train_val_with_lags.py \
    --lag-in-days 7 --suffix _lag7d --cutoff "2022-12-31 23:00" --sort

Notes:
- If your dataset has DST duplicates (same local ds twice), the _occ logic keeps alignment stable.
- Make sure the columns you pass in --features exist in the input CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd


DEFAULT_INPATH = "raw_data/single_bid_zones/de_lu/de_lu_train_val.csv"
DEFAULT_CUTOFF = "2022-12-31 23:00"
DEFAULT_LAG_DAYS = 7
DEFAULT_SUFFIX = "_lag7d"


def lag_features_simple(
    df: pd.DataFrame,
    features: List[str],
    lag_in_days: int = 7,
    *,
    ds_col: str = "ds",
    id_col: str = "unique_id",
    suffix: str = "_lag7d",
) -> pd.DataFrame:
    out = df.copy()
    out[ds_col] = pd.to_datetime(out[ds_col])

    # 1) Define "right order" for duplicates of the same ds
    out = out.sort_values([id_col, ds_col], kind="mergesort")
    out["_occ"] = out.groupby([id_col, ds_col], sort=False).cumcount()

    # 2) Build lookup table and shift timestamps forward by lag
    lag = out[[id_col, ds_col, "_occ"] + features].copy()
    lag[ds_col] = lag[ds_col] + pd.Timedelta(days=lag_in_days)

    # 3) Rename lagged columns and merge 1:1 by (id, ds, occurrence)
    lag = lag.rename(columns={c: f"{c}{suffix}" for c in features})

    merged = out.merge(
        lag,
        on=[id_col, ds_col, "_occ"],
        how="left",
        validate="many_to_one",
    )

    # 4) Drop rows where lag not available (first `lag_in_days` days)
    lag_cols = [f"{c}{suffix}" for c in features]
    merged = merged.dropna(subset=lag_cols)

    return merged.drop(columns=["_occ"])


def parse_args():
    p = argparse.ArgumentParser(description="Add lagged features then split train/val by cutoff.")

    p.add_argument("--inpath", type=str, default=DEFAULT_INPATH, help="Input CSV path")
    p.add_argument(
        "--cutoff",
        type=str,
        default=DEFAULT_CUTOFF,
        help='Cutoff timestamp (inclusive for train), e.g. "2022-12-31 23:00"',
    )

    p.add_argument("--ds-col", type=str, default="ds", help="Datetime column name (default: ds)")
    p.add_argument("--id-col", type=str, default="unique_id", help="ID column name (default: unique_id)")

    p.add_argument(
        "--features",
        nargs="+",
        default=[
            "gas_price",
            "cross_border_trading",
            "non_renewable",
            "renewable",
            "load",
            "co2_emission_allowances",
            "synthetic_price",
        ],
        help="List of columns to lag (space-separated).",
    )
    p.add_argument("--lag-in-days", type=int, default=DEFAULT_LAG_DAYS, help="Lag in days (default 7)")
    p.add_argument("--suffix", type=str, default=DEFAULT_SUFFIX, help="Suffix for lagged columns (default _lag7d)")

    p.add_argument("--sort", action="store_true", help="Sort by ds before splitting (recommended)")
    p.add_argument("--train-out", type=str, default="raw_data/single_bid_zones/de_lu/", help="Output path for train CSV (default: beside input)")
    p.add_argument("--val-out", type=str, default="raw_data/single_bid_zones/de_lu/", help="Output path for val CSV (default: beside input)")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    inpath = Path(args.inpath)
    if not inpath.exists():
        print(f"[ERR] Input file not found: {inpath}", file=sys.stderr)
        return 1

    df = pd.read_csv(inpath)

    # basic column checks
    missing = [c for c in [args.ds_col, args.id_col] if c not in df.columns]
    if missing:
        print(f"[ERR] Missing required columns: {missing}. Available: {list(df.columns)}", file=sys.stderr)
        return 1

    feat_missing = [c for c in args.features if c not in df.columns]
    if feat_missing:
        print(f"[ERR] Features not found in CSV: {feat_missing}", file=sys.stderr)
        return 1

    # Ensure datetime
    df[args.ds_col] = pd.to_datetime(df[args.ds_col])

    # Optional sort (good hygiene; lag fn will stable-sort anyway)
    if args.sort:
        df = df.sort_values([args.id_col, args.ds_col], kind="mergesort").reset_index(drop=True)

    # 1) ADD LAGS FIRST (this also drops the first lag window rows)
    df_lagged = lag_features_simple(
        df,
        features=args.features,
        lag_in_days=args.lag_in_days,
        ds_col=args.ds_col,
        id_col=args.id_col,
        suffix=args.suffix,
    )

    # 2) SPLIT AFTER lagging
    cutoff = pd.to_datetime(args.cutoff)
    train_df = df_lagged[df_lagged[args.ds_col] <= cutoff].copy()
    val_df = df_lagged[df_lagged[args.ds_col] > cutoff].copy()

    out_dir = inpath.parent

    train_out = Path(args.train_out) if args.train_out else (out_dir / "de_lu_train.csv")
    val_out   = Path(args.val_out)   if args.val_out   else (out_dir / "de_lu_val.csv")

    # If user provided a directory, append default filenames
    if args.train_out and train_out.suffix == "":
        train_out = train_out / "de_lu_train.csv"
    if args.val_out and val_out.suffix == "":
        val_out = val_out / "de_lu_val.csv"

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    # Logging
    print("[OK] Lagging + split complete")
    print(f"  lag: {args.lag_in_days} days | suffix: {args.suffix}")
    print(f"  cutoff (inclusive train): {cutoff}")
    print(f"  lagged total: {len(df_lagged)} rows | {df_lagged[args.ds_col].min()} -> {df_lagged[args.ds_col].max()}")
    print(f"  train: {len(train_df)} rows | {train_df[args.ds_col].min()} -> {train_df[args.ds_col].max()} | wrote {train_out}")
    print(f"  val:   {len(val_df)} rows | {val_df[args.ds_col].min()} -> {val_df[args.ds_col].max()} | wrote {val_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())