# scripts/build_moirai_dataset.py
import argparse
from pathlib import Path
import pandas as pd

from models.moirai.simple_panel_exo import (
    SimpleFinetuneDatasetBuilder,
    SimpleEvalDatasetBuilder,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_name", type=str)
    p.add_argument("file_path", type=str)
    p.add_argument("--dataset_type", 
                   type=str, 
                   choices=["wide","long","wide_multivariate","panel_exo"], default="panel_exo")
    p.add_argument("--offset", type=int, default=None)
    p.add_argument("--date_offset", type=str, default=None)
    p.add_argument("--freq", default="H")
    p.add_argument("--normalize", action="store_true")

    p.add_argument("--id_col", default="unique_id")
    p.add_argument("--time_col", default="ds")
    p.add_argument("--target_col", default="y")
    p.add_argument("--ck_cols", nargs="*", default=[
        "is_weekend","is_holiday","month_sin","month_cos",
        "day_of_week_sin","day_of_week_cos","hour_sin","hour_cos"
    ])
    p.add_argument("--cu_cols", nargs="*", default=[])

    args = p.parse_args()

    train_dataset_builder = SimpleFinetuneDatasetBuilder(
        dataset=args.dataset_name,
        windows=None, distance=None,
        prediction_length=None, context_length=None, patch_size=None,
    )
    train_dataset_builder.build_dataset(
        file=Path(args.file_path),
        dataset_type=args.dataset_type,
        offset=args.offset,
        date_offset=pd.Timestamp(args.date_offset) if args.date_offset else None,
        freq=args.freq,
        normalize=args.normalize,
        id_col=args.id_col, time_col=args.time_col, target_col=args.target_col,
        ck_cols=args.ck_cols, cu_cols=args.cu_cols,
    )

    if args.offset is not None or args.date_offset is not None:
        SimpleEvalDatasetBuilder(
            f"{args.dataset_name}_eval",
            offset=None, windows=None, distance=None,
            prediction_length=None, context_length=None, patch_size=None,
        ).build_dataset(
            file=Path(args.file_path),
            dataset_type=args.dataset_type,
            freq=args.freq,
            mean=train_dataset_builder.mean, std=train_dataset_builder.std,
            id_col=args.id_col, time_col=args.time_col, target_col=args.target_col,
            ck_cols=args.ck_cols, cu_cols=args.cu_cols,
        )

if __name__ == "__main__":
    main()