import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Generator, Optional
import numpy as np

import datasets
import pandas as pd
from datasets import Features, Sequence, Value
from torch.utils.data import Dataset

from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc
from uni2ts.data.builder._base import DatasetBuilder
from uni2ts.data.dataset import (
    EvalDataset,
    FinetuneDataset,
    SampleTimeSeriesType,
    TimeSeriesDataset,
)
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation
from uni2ts.data.builder.simple import _from_long_dataframe, _from_wide_dataframe, _from_wide_dataframe_multivariate


def _from_panel_exo_dataframe(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    ck_cols: list[str] = None,   # known past+future (calendar sin/cos)
    cu_cols: list[str] = None,   # known past-only (gas, load, ...)
    offset: Optional[int] = None,
    date_offset: Optional[pd.Timestamp] = None,
    freq: str = "H",
) -> tuple[GenFunc, Features]:
    ck_cols = ck_cols or []
    cu_cols = cu_cols or []

    # Ensure time is the index like their other helpers expect
    if df.index.name != time_col:
        if time_col in df.columns:
            df = df.set_index(time_col)
        else:
            raise ValueError(f"time_col '{time_col}' not found")

    df = df.sort_values([id_col, df.index.name])

    # Optional trimming
    if offset is not None:
        df = df.groupby(id_col, group_keys=False).apply(lambda g: g.iloc[:offset])
    elif date_offset is not None:
        df = df[df.index <= date_offset]

    inferred_freq = pd.infer_freq(df.index)
    use_freq = inferred_freq or freq

    # Validate columns
    missing = [c for c in [target_col, id_col] + (ck_cols + cu_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic NA handling
    feat_cols = ck_cols + cu_cols
    if feat_cols:
        df[feat_cols] = (
            df.groupby(id_col)[feat_cols]
            .transform(lambda g: g.ffill())
            .fillna(0.0)
        )

    # One HF example per unique_id
    items = df[id_col].unique()

    has_ck = len(ck_cols) > 0
    has_cu = len(cu_cols) > 0

    def example_gen_func():
        for uid in items:
            g = df[df[id_col] == uid]
            ex = {
                "item_id": str(uid),
                "start": g.index[0],
                "freq": use_freq,
                "target": g[target_col].astype("float32").to_numpy(),
            }
            if has_ck:
                ex["feat_dynamic_real"] = g[ck_cols].astype("float32").to_numpy().T
            if has_cu:
                ex["past_feat_dynamic_real"] = g[cu_cols].astype("float32").to_numpy().T
            yield ex

    # Build Features dict only for present fields
    feats = dict(
        item_id=Value("string"),
        start=Value("timestamp[s]"),
        freq=Value("string"),
        target=Sequence(Value("float32")),
    )
    if has_ck:
        feats["feat_dynamic_real"] = Sequence(Sequence(Value("float32")), length=len(ck_cols))
    if has_cu:
        feats["past_feat_dynamic_real"] = Sequence(Sequence(Value("float32")), length=len(cu_cols))

    features = Features(feats)
    return example_gen_func, features


@dataclass
class SimpleDatasetBuilder(DatasetBuilder):
    dataset: str
    weight: float = 1.0
    sample_time_series: Optional[SampleTimeSeriesType] = SampleTimeSeriesType.NONE
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        offset: Optional[int] = None,
        date_offset: Optional[pd.Timestamp] = None,
        freq: str = "H",
        *,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
        ck_cols: Optional[list[str]] = None,
        cu_cols: Optional[list[str]] = None,
    ):
        assert offset is None or date_offset is None, (
            "One or neither offset and date_offset must be specified, but not both. "
            f"Got offset: {offset}, date_offset: {date_offset}"
        )

        # safer read: don't force index here; the helper will set it
        if dataset_type == "panel_exo":
            df = pd.read_csv(file, parse_dates=[time_col])
        else:
            df = pd.read_csv(file, index_col=0, parse_dates=True)

        if dataset_type == "panel_exo":
            example_gen_func, features = _from_panel_exo_dataframe(
                df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                ck_cols=ck_cols or [],
                cu_cols=cu_cols or [],
                freq=freq,
                offset=offset,
                date_offset=date_offset,
            )
        elif dataset_type == "long":
            example_gen_func, features = _from_long_dataframe(
                df, freq=freq, offset=offset, date_offset=date_offset
            )
        elif dataset_type == "wide":
            example_gen_func, features = _from_wide_dataframe(
                df, freq=freq, offset=offset, date_offset=date_offset
            )
        elif dataset_type == "wide_multivariate":
            example_gen_func, features = _from_wide_dataframe_multivariate(
                df, freq=freq, offset=offset, date_offset=date_offset
            )
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', 'wide_multivariate', 'panel_exo'."
            )

        hf_dataset = datasets.Dataset.from_generator(example_gen_func, features=features)
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform_map[self.dataset](),
            dataset_weight=self.weight,
            sample_time_series=self.sample_time_series,
        )
    
@dataclass
class SimpleFinetuneDatasetBuilder(DatasetBuilder):
    dataset: str
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    mode: Optional[str] = "S"
    storage_path: Path = env.CUSTOM_DATA_PATH
    mean = None
    std = None

    """
    Databuilder class for LSF fine-tuning, which is modified from SimpleEvalDatasetBuilder. 
    'mean' and 'std' are accepted for data normalization.
    """

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        offset: Optional[int] = None,
        date_offset: Optional[pd.Timestamp] = None,
        freq: str = "H",
        normalize: Optional[bool] = False,
        *,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
        ck_cols: Optional[list[str]] = None,
        cu_cols: Optional[list[str]] = None,
    ):
        assert offset is None or date_offset is None, (
            "One or neither offset and date_offset must be specified, but not both. "
            f"Got offset: {offset}, date_offset: {date_offset}"
        )

        if dataset_type == "panel_exo":
            df = pd.read_csv(file, parse_dates=[time_col])
        else:
            df = pd.read_csv(file, index_col=0, parse_dates=True)

        if normalize and dataset_type != "panel_exo":
            end = (
                offset if offset is not None
                else (len(df[df.index <= date_offset]) if date_offset is not None else len(df.index))
            )
            df = self.scale(df, 0, end)

        if dataset_type == "panel_exo":
            example_gen_func, features = _from_panel_exo_dataframe(
                df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                ck_cols=ck_cols or [],
                cu_cols=cu_cols or [],
                freq=freq,
                offset=offset,
                date_offset=date_offset,
            )
            # TODO: ADJUSTED PATHS FROM MY SIDE, MIGHT NEED TO MAKE IT INDEPENDENT OF MOIRAI
            # save_dir = self.storage_path / "lsf" / "panel_exo" / self.dataset
            save_dir = self.storage_path / self.dataset
        elif dataset_type == "long":
            example_gen_func, features = _from_long_dataframe(df, freq=freq, offset=offset, date_offset=date_offset)
            save_dir = self.storage_path / "lsf" / "long" / self.dataset
        elif dataset_type == "wide":
            example_gen_func, features = _from_wide_dataframe(df, freq=freq, offset=offset, date_offset=date_offset)
            save_dir = self.storage_path / "lsf" / "wide" / self.dataset
        elif dataset_type == "wide_multivariate":
            example_gen_func, features = _from_wide_dataframe_multivariate(df, freq=freq, offset=offset, date_offset=date_offset)
            save_dir = self.storage_path / "lsf" / "wide_multivariate" / self.dataset
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', 'wide_multivariate', 'panel_exo'."
            )

        hf_dataset = datasets.Dataset.from_generator(example_gen_func, features=features)
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(save_dir)

    def load_dataset(
    self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        # Prefer panel_exo if it exists; otherwise fall back to original layout
        # TODO: ADJUSTED PATHS FROM MY SIDE, MIGHT NEED TO MAKE IT INDEPENDENT OF MOIRAI
        # base = Path(self.storage_path) / "lsf"
        # candidates = [
        #     base / "panel_exo" / self.dataset,
        #     base / "wide" / self.dataset,
        #     base / "wide_multivariate" / self.dataset,
        #     base / "long" / self.dataset,
        # ]
        base = Path(self.storage_path)
        candidates = [
            base / self.dataset,
            base / "wide" / self.dataset,
            base / "wide_multivariate" / self.dataset,
            base / "long" / self.dataset,
        ]

        for p in candidates:
            if p.exists():
                disk_path = p
                break
        else:
            raise FileNotFoundError(
                f"No saved dataset found for '{self.dataset}' under {base} "
                f"(looked in: {', '.join(str(c) for c in candidates)})"
            )

        return FinetuneDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(str(disk_path))
            ),
            transform=transform_map[self.dataset](
                distance=self.distance,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
            ),
        )

    def scale(self, data, start, end):
        train = data[start:end]
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0)
        return (data - self.mean) / (self.std + 1e-10)


@dataclass
class SimpleEvalDatasetBuilder(DatasetBuilder):
    dataset: str
    offset: Optional[int]
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    mode: Optional[str] = "S"
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        freq: str = "H",
        mean: pd.Series = None,
        std: pd.Series = None,
        *,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
        ck_cols: Optional[list[str]] = None,
        cu_cols: Optional[list[str]] = None,
    ):
        if dataset_type == "panel_exo":
            df = pd.read_csv(file, parse_dates=[time_col])
        else:
            df = pd.read_csv(file, index_col=0, parse_dates=True)

        if (mean is not None) and (std is not None) and dataset_type != "panel_exo":
            df = (df - mean) / (std + 1e-10)

        if dataset_type == "panel_exo":
            example_gen_func, features = _from_panel_exo_dataframe(
                df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                ck_cols=ck_cols or [],
                cu_cols=cu_cols or [],
                freq=freq,
            )
            # TODO: ADJUSTED PATHS FROM MY SIDE, MIGHT NEED TO MAKE IT INDEPENDENT OF MOIRAI
            # save_dir = self.storage_path / "lsf" / "panel_exo" / self.dataset
            save_dir = self.storage_path / self.dataset
        elif dataset_type == "long":
            example_gen_func, features = _from_long_dataframe(df, freq=freq)
            save_dir = self.storage_path / "lsf" / "long" / self.dataset
        elif dataset_type == "wide":
            example_gen_func, features = _from_wide_dataframe(df, freq=freq)
            save_dir = self.storage_path / "lsf" / "wide" / self.dataset
        elif dataset_type == "wide_multivariate":
            example_gen_func, features = _from_wide_dataframe_multivariate(df, freq=freq)
            save_dir = self.storage_path / "lsf" / "wide_multivariate" / self.dataset
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', 'wide_multivariate', 'panel_exo'."
            )

        hf_dataset = datasets.Dataset.from_generator(example_gen_func, features=features)
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(save_dir)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        # Prefer panel_exo if it exists; otherwise fall back to original layout
        # TODO: ADJUSTED PATHS FROM MY SIDE, MIGHT NEED TO MAKE IT INDEPENDENT OF MOIRAI
        # base = Path(self.storage_path) / "lsf"
        # candidates = [
        #     base / "panel_exo" / self.dataset,
        #     base / "wide" / self.dataset,
        #     base / "wide_multivariate" / self.dataset,
        #     base / "long" / self.dataset,
        # ]
        base = Path(self.storage_path)
        candidates = [
            base / self.dataset,
            base / "wide" / self.dataset,
            base / "wide_multivariate" / self.dataset,
            base / "long" / self.dataset,
        ]

        for p in candidates:
            if p.exists():
                disk_path = p
                break
        else:
            raise FileNotFoundError(
                f"No saved dataset found for '{self.dataset}' under {base} "
                f"(looked in: {', '.join(str(c) for c in candidates)})"
            )

        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(str(disk_path))
            ),
            transform=transform_map[self.dataset](
                offset=self.offset,
                distance=self.distance,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
            ),
        )
    

def generate_finetune_builder(
    dataset: str,
    train_length: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    mode: str,
    storage_path: Path = env.CUSTOM_DATA_PATH,
    distance=1,
) -> SimpleFinetuneDatasetBuilder:
    """
    By default, 'distance' is set to 1 for sliding window. A larger value can be used to reduce computational cost.
    """

    windows = (train_length - context_length - prediction_length) // distance + 1
    return SimpleFinetuneDatasetBuilder(
        dataset=dataset,
        windows=windows,
        distance=distance,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        mode=mode,
        storage_path=storage_path,
    )


def generate_eval_builder(
    dataset: str,
    offset: int,
    eval_length: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    mode: str,
    storage_path: Path = env.CUSTOM_DATA_PATH,
    distance=None,
) -> SimpleEvalDatasetBuilder:
    """
    By default, 'distance' is set to prediction length for rolling evaluation.
    Offer specific 'distance' to decrease the number of validation samples and to reduce computational cost.
    """

    if distance is not None:
        windows = (eval_length - prediction_length) // distance + 1
    else:
        distance = prediction_length
        windows = eval_length // prediction_length

    return SimpleEvalDatasetBuilder(
        dataset=dataset,
        offset=offset,
        windows=windows,
        distance=distance,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        mode=mode,
        storage_path=storage_path,
    )


def generate_eval_builders(
    dataset: str,
    offset: int,
    eval_length: int,
    prediction_lengths: list[int],
    context_lengths: list[int],
    patch_sizes: list[int],
    storage_path: Path = env.CUSTOM_DATA_PATH,
) -> list[SimpleEvalDatasetBuilder]:
    return [
        SimpleEvalDatasetBuilder(
            dataset=dataset,
            offset=offset,
            windows=eval_length // pred,
            distance=pred,
            prediction_length=pred,
            context_length=ctx,
            patch_size=psz,
            storage_path=storage_path,
        )
        for pred, ctx, psz in product(prediction_lengths, context_lengths, patch_sizes)
    ]