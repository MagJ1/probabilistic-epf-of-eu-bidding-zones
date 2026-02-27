# FlowForecasterDataModule
from typing import Optional, Union
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.normalizing_flows.dataset import FlowForecasterDataset
from data.base_datamodule import BaseForecastingDataModule
from sklearn.preprocessing import StandardScaler
import numpy as np

class FlowForecasterDataModule(BaseForecastingDataModule):
    """Builds datasets and returns dataloaders"""
    def __init__(self,
                 train_csv_path: str,
                 test_csv_path: str,
                 context_length: int = 168,
                 forecast_horizon: int = 24,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 persistent_workers: bool = False,
                 val_split_date: dict[str, Union[str, pd.Timestamp, None]] = {},
                 date_col: str = "ds",
                 id_col: str = "unique_id",
                 y_col: str = "y",
                 ck_cols: list[str] = None,
                 cu_cols: list[str] = None,
                 past_unknown_cov_cutoff: int = 14,
                 scale_data: bool = True,
                 scale_cols: Optional[list[str]] = None,
                 realistic_mode: bool = True,
                 origin_stride_train: int = 1,
                 origin_stride_val: int = 1,
                 origin_stride_test: int = 24,   # <- daily forecasts on test
                 origin_anchor_hour_train: Optional[int] = None,
                 origin_anchor_hour_val:   Optional[int] = None,
                 origin_anchor_hour_test:  Optional[int] = 0,   # e.g., midnight on test
                 ):
        super().__init__(train_csv_path=train_csv_path,
                         test_csv_path=test_csv_path,
                         context_length=context_length,
                         forecast_horizon=forecast_horizon,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         persistent_workers=persistent_workers,
                         val_split_date=val_split_date,
                         date_col=date_col,
                         id_col=id_col,
                         y_col=y_col,
                         ck_cols=ck_cols,
                         cu_cols=cu_cols,
                         past_unknown_cov_cutoff=past_unknown_cov_cutoff,
                         realistic_mode=realistic_mode)

        self.scale_data = scale_data
        self.scale_cols = scale_cols or ((self.cu_cols or []) + [self.y_col])

        self.origin_stride_train = max(1, int(origin_stride_train))
        self.origin_stride_val   = max(1, int(origin_stride_val))
        self.origin_stride_test  = max(1, int(origin_stride_test))
        self.origin_anchor_hour_train = origin_anchor_hour_train
        self.origin_anchor_hour_val   = origin_anchor_hour_val
        self.origin_anchor_hour_test  = origin_anchor_hour_test

        if self.scale_data:
            self.scalers: dict[str, StandardScaler] = {}
        else:
            self.scalers = None

    def setup(self, stage=None):
        """Builds datasets"""
        def make_dataset(split_type: str,
                         csv_path: str,
                         val_split_date: dict[str, Union[str, pd.Timestamp, None]],
                         scale_data: bool,
                         scalers: Optional[dict[str,StandardScaler]],
                         scale_cols: Optional[list[str]],
                         origin_stride: int, 
                         origin_anchor_hour: Optional[int]) -> FlowForecasterDataset:
            return FlowForecasterDataset(
                csv_path=csv_path,
                context_length=self.context_length,
                forecast_horizon=self.forecast_horizon,
                val_split_date=val_split_date,
                split_type=split_type,
                date_col=self.date_col,
                id_col=self.id_col,
                y_col=self.y_col,
                ck_cols=self.ck_cols,
                cu_cols=self.cu_cols,
                past_unknown_cov_cutoff=self.past_unknown_cov_cutoff,
                scale_data=scale_data,
                scalers=scalers,
                scale_cols=scale_cols,
                realistic_mode=self.realistic_mode,
                origin_stride=origin_stride,
                origin_anchor_hour=origin_anchor_hour
            )

        # Fit scalers (unscaled view of TRAIN)
        if self.scale_data:
            unscaled = make_dataset(
                "train", self.train_csv_path, val_split_date=self.val_split_date,
                scale_data=False, scalers=None, scale_cols=None,
                origin_stride=self.origin_stride_train,
                origin_anchor_hour=self.origin_anchor_hour_train,
            )
            for col in self.scale_cols:
                v = unscaled.get_col(col).reshape(-1, 1)
                v = v[np.isfinite(v[:, 0])]
                self.scalers[col] = StandardScaler().fit(v)

        if stage == "fit" or stage is None:
            self.train_dataset = make_dataset(
                "train", self.train_csv_path, self.val_split_date,
                self.scale_data, self.scalers, self.scale_cols,
                origin_stride=self.origin_stride_train,
                origin_anchor_hour=self.origin_anchor_hour_train,
            )
            self.val_dataset = make_dataset(
                "val", self.train_csv_path, self.val_split_date,
                self.scale_data, self.scalers, self.scale_cols,
                origin_stride=self.origin_stride_val,
                origin_anchor_hour=self.origin_anchor_hour_val,
            )

        if stage == "test" or stage is None:
            self.test_dataset = make_dataset(
                "test", self.test_csv_path, {},
                self.scale_data, self.scalers, self.scale_cols,
                origin_stride=self.origin_stride_test,
                origin_anchor_hour=self.origin_anchor_hour_test,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers, 
                          persistent_workers=self.persistent_workers)

    def get_scalers(self):
        return self.scalers