# src/data/base_datamodule.py
from __future__ import annotations
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.seeds import seed_worker
import torch

class BaseForecastingDataModule(pl.LightningDataModule):
    """
    Generic DataModule:
    - Children set `dataset_cls`
    - Children pass extra dataset kwargs via `dataset_kwargs_extra`
    - Base builds train/val/test and (optionally) shares scaler state if exposed
    """
    dataset_cls = None                           # child MUST set
    dataset_kwargs_extra: Dict[str, Any] = {}    # child MAY override

    def __init__(self,
                 train_csv_path: str,
                 test_csv_path: str,
                 context_length: int = 168,
                 forecast_horizon: int = 24,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 persistent_workers: bool = False,
                 val_split_date: dict[str, Union[str, pd.Timestamp]] = {},
                 date_col: str = "ds",
                 id_col: str = "unique_id",
                 y_col: str = "y",
                 ck_cols: Optional[List[str]] = None,
                 cu_cols: Optional[List[str]] = None,
                 static_cols: Optional[List[str]] = None,
                 past_unknown_cov_cutoff: int = 14,
                 realistic_mode: bool = True,
                 share_scalers: bool = True,   # only if dataset exposes get_/set_ methods
                 seed: int = 0,
                 **dataset_kwargs_extra,       # pass-through to dataset
                 ):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.test_csv_path  = test_csv_path
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        val_split_date = val_split_date or {}
        self.val_split_date = {
            str(k): (pd.Timestamp(v) if isinstance(v, str) else v)
            for k, v in val_split_date.items()
        }

        self.date_col = date_col
        self.id_col   = id_col
        self.y_col    = y_col
        self.ck_cols  = ck_cols
        self.cu_cols  = cu_cols
        self.static_cols = static_cols or [] 

        self.past_unknown_cov_cutoff = past_unknown_cov_cutoff
        self.realistic_mode = realistic_mode

        self.share_scalers = share_scalers
        self.seed = int(seed)
        # stored and forwarded to dataset on construction
        self._dataset_kwargs_extra = dataset_kwargs_extra

        # will be populated in setup()
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    # --------------------------
    # Helpers
    # --------------------------
    def _make_ds(self, split: str, csv_path: str, **overrides) -> Any:
        assert self.dataset_cls is not None, "Child must set dataset_cls"
        kw = dict(
            csv_path=csv_path,
            context_length=self.context_length,
            forecast_horizon=self.forecast_horizon,
            val_split_date=self.val_split_date,
            split_type=split,
            date_col=self.date_col,
            id_col=self.id_col,
            y_col=self.y_col,
            ck_cols=self.ck_cols,
            cu_cols=self.cu_cols,
            static_cols=self.static_cols,
            past_unknown_cov_cutoff=self.past_unknown_cov_cutoff,
            realistic_mode=self.realistic_mode,
        )
        # child-provided defaults + caller overrides
        kw.update(self.dataset_kwargs_extra)
        kw.update(self._dataset_kwargs_extra)
        kw.update(overrides)
        return self.dataset_cls(**kw)

    def _maybe_share_scalers(self, src_ds, dst_ds):
        if not self.share_scalers or src_ds is None or dst_ds is None:
            return
        # Feature-detect: only share if the dataset exposes these hooks
        if hasattr(src_ds, "get_target_scaler") and hasattr(dst_ds, "set_target_scaler"):
            dst_ds.set_target_scaler(src_ds.get_target_scaler())
        if hasattr(src_ds, "get_scalers") and hasattr(dst_ds, "set_scalers"):
            dst_ds.set_scalers(src_ds.get_scalers())

    # --------------------------
    # Lightning hooks
    # --------------------------
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            # Train first (lets it fit scalers internally, if any)
            self.train_dataset = self._make_ds("train", self.train_csv_path)
            # Build val and share states if available
            self.val_dataset = self._make_ds("val", self.train_csv_path)
            self._maybe_share_scalers(self.train_dataset, self.val_dataset)

        if stage in (None, "test"):
            # If fit not called yet, build a temp train to get states
            src_for_states = self.train_dataset
            if src_for_states is None:
                src_for_states = self._make_ds("train", self.train_csv_path)
            self.test_dataset = self._make_ds("test", self.test_csv_path)
            self._maybe_share_scalers(src_for_states, self.test_dataset)

    # --------------------------
    # DataLoaders
    # --------------------------
    def _make_loader(self, dataset, *, shuffle: bool, seed_offset: int = 0) -> DataLoader:
        # If shuffle=False, generator isn't strictly needed, but harmless.
        g = torch.Generator()
        g.manual_seed(self.seed + seed_offset)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=False, seed_offset=0)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False, seed_offset=1)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False, seed_offset=2)