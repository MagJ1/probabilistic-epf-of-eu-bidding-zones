# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import textwrap as tw

def _pick_length_from_counts(counts: pd.Series, label: str) -> int:
    if counts.empty:
        print(f"[infer_split] {label}: no rows after split; length=0.", file=sys.stderr)
        return 0
    if counts.nunique() == 1:
        tl = int(counts.iloc[0])
        print(f"[infer_split] {label}: length={tl} (uniform across ids).")
        return tl
    min_value = int(counts.min())
    print(
        f"[infer_split] WARNING: {label}: non-uniform per-id lengths. "
        f"Using min(length)={min_value}. "
        f"(min={int(counts.min())}, max={int(counts.max())})",
        file=sys.stderr,
    )
    return min_value

def _infer_full_length(csv_path, id_col, time_col) -> int:
    df = pd.read_csv(csv_path, usecols=[id_col, time_col])
    df[time_col] = pd.to_datetime(df[time_col], errors="raise", utc=False)
    counts = df.groupby(id_col)[time_col].nunique(dropna=True)
    if counts.empty:
        return 0
    if counts.nunique() == 1:
        return int(counts.iloc[0])
    return int(counts.min())  

def infer_lengths_with_split(
    csv_path: Path,
    id_col: str,
    time_col: str,
    split_date: str | pd.Timestamp,
) -> tuple[int, int]:
    """
    Split by `split_date` where the first part includes rows with time <= split_date.
    Return (len_part1, len_part2) as timesteps-per-series, using min across ids if non-uniform.
    """
    df = pd.read_csv(csv_path, usecols=[id_col, time_col])
    df[time_col] = pd.to_datetime(df[time_col], errors="raise", utc=False)

    sd = pd.to_datetime(split_date, utc=False)

    part1 = df[df[time_col] <= sd]
    part2 = df[df[time_col] >  sd]

    counts1 = part1.groupby(id_col)[time_col].nunique(dropna=True)
    counts2 = part2.groupby(id_col)[time_col].nunique(dropna=True)

    len1 = _pick_length_from_counts(counts1, label="part1 (<= split_date)")
    len2 = _pick_length_from_counts(counts2, label="part2 (> split_date)")
    return len1, len2


def _write_atomic(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser(
        description="Generate Hydra finetune data YAMLs for Moirai from a CSV."
    )
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--id-col", default="unique_id")
    ap.add_argument("--time-col", default="ds")
    ap.add_argument("--target-col", default="y")
    ap.add_argument("--context-length", type=int, required=True)
    ap.add_argument("--prediction-length", type=int, required=True)
    ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--mode", default="custom")
    ap.add_argument("--distance", type=int, default=1)
    # eval settings
    ap.add_argument("--make-eval", action="store_true")
    ap.add_argument("--eval_date_offset")
    ap.add_argument("--eval-distance", type=int, default=24)
    ap.add_argument("--offset", type=int)
    ap.add_argument("--out-dir-train",
                    default="src/models/moirai/conf/finetune/data")
    ap.add_argument("--out-dir-val",
                    default="src/models/moirai/conf/finetune/val_data")

    args = ap.parse_args()

    if args.make_eval and not args.eval_date_offset:
        ap.error("--eval_date_offset is required when --make-eval is set")

    csv_path = Path(args.csv)
    out_dir_train = Path(args.out_dir_train)
    out_dir_val   = Path(args.out_dir_val)
    base = args.dataset_name

    # --- lengths ---
    if args.make_eval:
        train_length, eval_length = infer_lengths_with_split(
            csv_path, args.id_col, args.time_col, args.eval_date_offset
        )
        if train_length <= 0 or eval_length <= 0:
            raise ValueError(
                f"Invalid lengths after split (train_length={train_length}, eval_length={eval_length}). "
                "Pick an earlier --eval_date_offset."
            )
    else:
        train_length = _infer_full_length(csv_path, args.id_col, args.time_col)
        if train_length <= 0:
            raise ValueError("Could not infer a positive train_length from CSV.")
        eval_length = None  # unused when not making eval

    # --- write TRAIN yaml ---
    train_yaml = tw.dedent(f"""\
        _target_: models.moirai.simple_panel_exo.generate_finetune_builder
        dataset: {base}
        train_length: {train_length}
        prediction_length: {args.prediction_length}
        context_length: {args.context_length}
        patch_size: {args.patch_size}
        mode: {args.mode}
        distance: {args.distance}
    """)
    train_path = out_dir_train / f"{base}.yaml"
    _write_atomic(train_path, train_yaml)
    print(f"[make_ft_data_yaml] Wrote {train_path}")

    # --- write EVAL yaml (only if requested) ---
    if args.make_eval:
        offset = args.offset if args.offset is not None else args.context_length
        eval_name = f"{base}_eval"
        eval_yaml = tw.dedent(f"""\
            _target_: models.moirai.simple_panel_exo.generate_eval_builder
            dataset: {eval_name}
            offset: {offset}
            eval_length: {eval_length}
            prediction_length: {args.prediction_length}
            context_length: {args.context_length}
            patch_size: {args.patch_size}
            mode: {args.mode}
            distance: {args.eval_distance}
        """)
        eval_path = out_dir_val / f"{eval_name}.yaml"
        _write_atomic(eval_path, eval_yaml)
        print(f"[make_ft_data_yaml] Wrote {eval_path}")

    print("[make_ft_data_yaml] Done.")


if __name__ == "__main__":
    main()