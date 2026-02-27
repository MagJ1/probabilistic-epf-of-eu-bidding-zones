#!/usr/bin/env python3
"""
Energy-Charts price crawler + calendar feature builder + transfer-learning dataset builder.

Outputs (in --outdir, default: raw_data/transfer_learning/):
Per-zone (filesystem-safe names):
  {ZONE_SAFE}_train.csv
  {ZONE_SAFE}_test.csv   (optional via --write-tests)

Merged cross-border:
  cross_border_electricity_prices_zero_shot.csv
  cross_border_electricity_prices_one_shot.csv
  cross_border_electricity_prices_few_shot.csv

Extra DE-LU slices:
  de_lu_one_shot.csv
  de_lu_few_shot.csv

Notes:
- Uses API zone codes (e.g., "DE-LU", "IT-North"), but filenames use filesystem-safe
  aliases (e.g., "de_lu", "it_north"). Mapping is handled by zone_to_safe_name().
- Timestamps are converted UTC -> Europe/Berlin and then tz info is dropped (local clock time).
- Adds zone-aware holidays by union of national holiday calendars for the zone.

Requirements:
  pip install requests pandas numpy holidays
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import holidays


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_OUTDIR = "raw_data/transfer_learning/"
DEFAULT_TZ = "Europe/Berlin"

DEFAULT_TRAIN_START = "2018-10-01"
DEFAULT_TRAIN_END = "2023-12-31"
DEFAULT_TEST_START = "2024-01-01"
DEFAULT_TEST_END = "2024-12-31"

DEFAULT_BIDDING_ZONES_TRAIN = [
    "AT", "BE", "CH", "CZ", "DK1", "DK2", "FR", "HU", "IT-North",
    "NL", "NO2", "PL", "SE4", "SI", "DE-LU"
]

DEFAULT_BIDDING_ZONES_CROSS_BORDER = [
    "AT", "BE", "CH", "CZ", "DK1", "DK2", "FR", "HU", "IT-North",
    "NL", "NO2", "PL", "SE4", "SI"
]

# Map bidding zone -> list of (country_code, subdiv)
ZONE_TO_HOLIDAY_ENTITIES: Dict[str, List[Tuple[str, Optional[str]]]] = {
    "AT": [("AT", None)],
    "BE": [("BE", None)],
    "CH": [("CH", None)],
    "CZ": [("CZ", None)],
    "DK1": [("DK", None)],
    "DK2": [("DK", None)],
    "FR": [("FR", None)],
    "HU": [("HU", None)],
    "IT-North": [("IT", None)],     # “North” spans multiple regions; using national holidays
    "NL": [("NL", None)],
    "NO2": [("NO", None)],
    "PL": [("PL", None)],
    "SE4": [("SE", None)],
    "SI": [("SI", None)],
    "DE-LU": [("DE", None), ("LU", None)],
    "DE-AT-LU": [("DE", None), ("AT", None), ("LU", None)],
}


# -----------------------------
# Filesystem-safe naming
# -----------------------------
def zone_to_safe_name(zone: str) -> str:
    """
    Convert an API zone code to a filesystem-safe lowercase token.
    Examples:
      "DE-LU"    -> "de_lu"
      "IT-North" -> "it_north"
      "DK1"      -> "dk1"
      "NO2"      -> "no2"
    Rule:
      - lowercase
      - replace any non [a-z0-9] with underscore
      - collapse multiple underscores
      - strip leading/trailing underscores
    """
    s = zone.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# -----------------------------
# Feature engineering
# -----------------------------
def cyclic_encode(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def build_holiday_calendar(bidding_zone: str, years: np.ndarray) -> holidays.HolidayBase:
    if bidding_zone not in ZONE_TO_HOLIDAY_ENTITIES:
        raise ValueError(
            f"Unknown bidding zone '{bidding_zone}'. Add it to ZONE_TO_HOLIDAY_ENTITIES."
        )

    combined = holidays.HolidayBase()
    for country_code, subdiv in ZONE_TO_HOLIDAY_ENTITIES[bidding_zone]:
        cal = holidays.country_holidays(country_code, years=years, subdiv=subdiv) if subdiv \
            else holidays.country_holidays(country_code, years=years)
        combined.update(cal)
    return combined


def add_calendar_features(df: pd.DataFrame, bidding_zone: str) -> pd.DataFrame:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    years = df["ds"].dt.year.unique()
    holiday_calendar = build_holiday_calendar(bidding_zone, years=years)
    holiday_dates = set(holiday_calendar.keys())  # datetime.date

    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["hour"] = df["ds"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["ds"].dt.date.isin(holiday_dates).astype(int)

    df = cyclic_encode(df, "month", 12)
    df = cyclic_encode(df, "day_of_week", 7)
    df = cyclic_encode(df, "hour", 24)
    return df


# -----------------------------
# Fetching / formatting
# -----------------------------
def fetch_prices(
    bidding_zone: str,
    start_date: str,
    end_date: str,
    tz: str = DEFAULT_TZ,
    timeout: int = 60,
) -> pd.DataFrame:
    url = f"https://api.energy-charts.info/price?bzn={bidding_zone}&start={start_date}&end={end_date}"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {bidding_zone}: {resp.text[:300]}")

    data = resp.json()
    if "unix_seconds" not in data or "price" not in data:
        raise RuntimeError(f"Unexpected response schema for {bidding_zone}. Keys: {list(data.keys())}")

    ts = pd.to_datetime(data["unix_seconds"], unit="s", utc=True).tz_convert(tz)
    df = pd.DataFrame({"ds": ts, "y": data["price"]})

    # Keep local clock time, drop tz info
    df["ds"] = df["ds"].dt.tz_localize(None)

    # Drop missing prices
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    return df


def to_nf_format(df: pd.DataFrame, bidding_zone: str) -> pd.DataFrame:
    df = df.copy()
    df["unique_id"] = bidding_zone  # keep API code inside data
    base_cols = ["unique_id", "ds", "y"]
    rest = [c for c in df.columns if c not in base_cols]
    return df[base_cols + rest]


def build_zone_split_csv(
    *,
    bidding_zone: str,
    split: str,  # "train" or "test"
    start_date: str,
    end_date: str,
    outdir: Path,
    tz: str,
    timeout: int,
) -> Path:
    df = fetch_prices(bidding_zone, start_date, end_date, tz=tz, timeout=timeout)
    df = to_nf_format(df, bidding_zone=bidding_zone)
    df = add_calendar_features(df, bidding_zone=bidding_zone)

    outdir.mkdir(parents=True, exist_ok=True)
    zone_safe = zone_to_safe_name(bidding_zone)
    outpath = outdir / f"{zone_safe}_{split}.csv"
    df.to_csv(outpath, index=False)

    print(f"[OK] {bidding_zone} ({zone_safe}) {split}: {df['ds'].min()} -> {df['ds'].max()} | rows={len(df)} | wrote {outpath}")
    return outpath


# -----------------------------
# Merging / transfer learning splits
# -----------------------------
def merge_cross_border(
    *,
    outdir: Path,
    bidding_zones: List[str],
    split: str,  # "train"
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for zone in bidding_zones:
        zone_safe = zone_to_safe_name(zone)
        fpath = outdir / f"{zone_safe}_{split}.csv"
        if not fpath.exists():
            print(f"[WARN] merge: missing {fpath}, skipping.")
            continue
        dfs.append(pd.read_csv(fpath))

    if not dfs:
        raise RuntimeError("merge_cross_border: no input files found.")
    return pd.concat(dfs, ignore_index=True)


def write_learning_paradigm_files(
    *,
    outdir: Path,
    df_cross_border_base: pd.DataFrame,
    df_de_lu_train: pd.DataFrame,
    one_shot_rows: int = 192,  # 168 + 24
    few_shot_rows: int = 720,  # 30*24
) -> None:
    # Ensure proper ordering for DE-LU
    df_de = df_de_lu_train.copy()
    df_de["ds"] = pd.to_datetime(df_de["ds"])
    df_de = df_de.sort_values("ds").reset_index(drop=True)

    de_lu_one = df_de.iloc[-one_shot_rows:].copy()
    de_lu_few = df_de.iloc[-few_shot_rows:].copy()

    # Save DE-LU slices explicitly
    p_one = outdir / "de_lu_one_shot.csv"
    p_few = outdir / "de_lu_few_shot.csv"
    de_lu_one.to_csv(p_one, index=False)
    de_lu_few.to_csv(p_few, index=False)
    print(f"[OK] wrote {p_one} (rows={len(de_lu_one)})")
    print(f"[OK] wrote {p_few} (rows={len(de_lu_few)})")

    # Zero-shot = cross-border base without DE-LU (enforce)
    df_zero = df_cross_border_base[df_cross_border_base["unique_id"] != "DE-LU"].copy()

    # One-shot / Few-shot = zero-shot + slices
    df_one = pd.concat([df_zero, de_lu_one], ignore_index=True)
    df_few = pd.concat([df_zero, de_lu_few], ignore_index=True)

    df_zero.to_csv(outdir / "cross_border_electricity_prices_zero_shot.csv", index=False)
    df_one.to_csv(outdir / "cross_border_electricity_prices_one_shot.csv", index=False)
    df_few.to_csv(outdir / "cross_border_electricity_prices_few_shot.csv", index=False)

    print("[OK] wrote cross-border learning-paradigm datasets:")
    print(f"  {outdir / 'cross_border_electricity_prices_zero_shot.csv'} (rows={len(df_zero)})")
    print(f"  {outdir / 'cross_border_electricity_prices_one_shot.csv'} (rows={len(df_one)})")
    print(f"  {outdir / 'cross_border_electricity_prices_few_shot.csv'} (rows={len(df_few)})")


# -----------------------------
# CLI / main
# -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Energy-Charts crawler + transfer-learning dataset builder")

    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                   help=f"Export path for CSVs (default: {DEFAULT_OUTDIR})")
    p.add_argument("--tz", type=str, default=DEFAULT_TZ, help="Timezone to convert to (default Europe/Berlin)")
    p.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")

    p.add_argument("--train-start", type=str, default=DEFAULT_TRAIN_START)
    p.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END)

    p.add_argument("--write-tests", action="store_true", default=True,
                   help="Also fetch and write {ZONE_SAFE}_test.csv for zones in --bidding-zones-test")
    p.add_argument("--test-start", type=str, default=DEFAULT_TEST_START)
    p.add_argument("--test-end", type=str, default=DEFAULT_TEST_END)

    p.add_argument("--bidding-zones-train", nargs="*", default=DEFAULT_BIDDING_ZONES_TRAIN)
    p.add_argument("--bidding-zones-test", nargs="*", default=["DE-LU"])

    p.add_argument("--bidding-zones-cross-border", nargs="*", default=DEFAULT_BIDDING_ZONES_CROSS_BORDER,
                   help="Zones that form the cross-border base dataset (typically excludes DE-LU)")

    p.add_argument("--skip-cross-border", action="store_true", help="Skip building cross-border datasets")
    p.add_argument("--one-shot-rows", type=int, default=192, help="Rows of DE-LU appended for one-shot (default 192)")
    p.add_argument("--few-shot-rows", type=int, default=720, help="Rows of DE-LU appended for few-shot (default 720)")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Crawl train splits
    for zone in args.bidding_zones_train:
        try:
            build_zone_split_csv(
                bidding_zone=zone,
                split="train",
                start_date=args.train_start,
                end_date=args.train_end,
                outdir=outdir,
                tz=args.tz,
                timeout=args.timeout,
            )
        except Exception as e:
            print(f"[ERR] {zone} train: {e}", file=sys.stderr)

    # 2) Optional: crawl test splits
    if args.write_tests:
        for zone in args.bidding_zones_test:
            try:
                build_zone_split_csv(
                    bidding_zone=zone,
                    split="test",
                    start_date=args.test_start,
                    end_date=args.test_end,
                    outdir=outdir,
                    tz=args.tz,
                    timeout=args.timeout,
                )
            except Exception as e:
                print(f"[ERR] {zone} test: {e}", file=sys.stderr)

    # 3) Build cross-border datasets (zero/one/few)
    if not args.skip_cross_border:
        try:
            df_cross_border_base = merge_cross_border(
                outdir=outdir,
                bidding_zones=args.bidding_zones_cross_border,
                split="train",
            )

            # Load DE-LU train (must exist)
            de_lu_safe = zone_to_safe_name("DE-LU")
            de_lu_train_path = outdir / f"{de_lu_safe}_train.csv"
            if not de_lu_train_path.exists():
                raise RuntimeError(
                    f"Missing {de_lu_train_path}. Ensure DE-LU is in --bidding-zones-train."
                )
            df_de_lu_train = pd.read_csv(de_lu_train_path)

            write_learning_paradigm_files(
                outdir=outdir,
                df_cross_border_base=df_cross_border_base,
                df_de_lu_train=df_de_lu_train,
                one_shot_rows=args.one_shot_rows,
                few_shot_rows=args.few_shot_rows,
            )

        except Exception as e:
            print(f"[ERR] cross-border build: {e}", file=sys.stderr)

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())