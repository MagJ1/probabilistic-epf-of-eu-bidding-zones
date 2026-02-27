from typing import TypedDict, Optional, Union
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data.base_dataset import BaseForecastingDataset

class SampleBatch(TypedDict):
    """Dictionary returned by FlowForecasterDataset.__getitem__()"""
    ds: str
    unique_id: int
    y_past: torch.Tensor
    c_ctx_future_unknown: torch.Tensor
    c_ctx_future_known: torch.Tensor
    c_fct_future_known: torch.Tensor
    y_future: torch.Tensor

class FlowForecasterDataset(BaseForecastingDataset):
    """Dataset class for the normalizing flows model"""
    def __init__(self,
                 csv_path: str,
                 context_length: int = 168,
                 forecast_horizon: int = 24,
                 val_split_date: dict[Union[str, pd.Timestamp]] = {},
                 split_type: str = "train",  
                 date_col: str = "ds",
                 id_col: str = "unique_id",
                 y_col: str = "y",
                 ck_cols: list[str] = None,
                 cu_cols: list[str] = None,
                 past_unknown_cov_cutoff: int = 14,
                 scale_data: bool = True,
                 scalers: Optional[dict[str, StandardScaler]] =None,
                 scale_cols: Optional[list[str]] = None,
                 realistic_mode: bool = True, 
                 origin_stride: int = 1,
                 origin_anchor_hour: Optional[int] = None):
        
        """Constructor

        Args:
            csv_path: Path to the csv file, containing the dataset
            context_length: Steps the model can see in the past
            forecast_horizon: Steps the model has to predict into the future
            scaler: Scaler that is used to scale the electricity prices
            val_split_date: dict of split dates for train/val set.
            split_type: What type of data it is. With "train", steps <= val_split_date is taken. With "val", steps > val_split_date is taken. "test" does the same as "train", but was implemented for code claritiy. 
            date_col: Column that holds the dates in the csv file
            id_col: Column that holds the ids in the csv file
            y_col: Column that holds the target columns, probably electricity prices
            ck_cols: Columns that hold covariates known in the future, i.e. calendar features
            cu_cols: Columsn that hold covariates known only in the past, e.g. load
            past_unknown_cov_cutoff: Tells how much of the unknown covariates (e.g. load) is masked, to generate batches. With a forecast at 12:00, the standard value would be 14, because information is only known until 9:00, making 10:00 and 11:00 unknown. 
            realistic_mode: True, when a realistic forecasting scenario should be trained, where data of covariates only known of the past is not known after 9:00, while prices of the same day of the forecast are known between 12:00 and 23:00. 
        """
    
        super().__init__(csv_path=csv_path, 
                            context_length=context_length, 
                            forecast_horizon=forecast_horizon,
                            val_split_date=val_split_date, 
                            split_type=split_type, 
                            date_col=date_col, 
                            id_col=id_col, 
                            y_col=y_col, 
                            ck_cols=ck_cols, 
                            cu_cols=cu_cols,
                            past_unknown_cov_cutoff=past_unknown_cov_cutoff,
                            realistic_mode=realistic_mode, 
                            origin_stride=origin_stride,
                            origin_anchor_hour=origin_anchor_hour)

        self.scale_data = scale_data
        self.scalers = scalers
        self.scale_cols = scale_cols

    
    def __getitem__(self, idx: int) -> SampleBatch:
        """
        Returns:
            dict:
                "ds": str
                "unique_id": int
                "y_past": shape (context_length,)
                "c_ctx_future_unknown": shape (context_length,len(cu_cols))
                "ctx_future_known": shape (context_length,len(ck_cols))
                "c_fct_future_known": shape (forecast_horizon,len(ck_cols))
        """
        sidx, t0 = self.indices[idx]
        time_series = self.ts_collection[sidx]

        ds = str(time_series["ds"][t0])
        unique_id = time_series["unique_id"]
        y_past = time_series["y"][t0-self.context_length:t0].copy()
        y_fut  = time_series["y"][t0:t0+self.forecast_horizon].copy()

        ck_past = time_series["ck"][t0-self.context_length:t0].copy()
        ck_fut  = time_series["ck"][t0:t0+self.forecast_horizon].copy()

        cu_past = time_series["cu"][t0-self.context_length:t0].copy()
        # if self.realistic_mode:
        #     cu_past[self.past_unknown_cov_cutoff:,:] = 0

        if self.scale_data and self.scalers:
            if self.scale_cols and (self.y_col in self.scale_cols):
                sc = self.scalers.get(self.y_col)
                if sc and self._ok_scale(sc):
                    y_past = sc.transform(y_past[:, None]).ravel()
                    y_fut  = sc.transform(y_fut[:,  None]).ravel()

            if self.cu_cols:
                for name in self.cu_cols:
                    if self.scale_cols and (name not in self.scale_cols):
                        continue
                    sc = self.scalers.get(name)
                    if not (sc and self._ok_scale(sc)):
                        continue
                    _, j = self.feature_map[name]
                    cu_past[:, j:j+1] = sc.transform(cu_past[:, j:j+1])

        # if idx < 3:
        #     print("Troubleshoot y_past mean: ", y_past.mean().item())
        #     print("Troubleshoot y_past std: ", y_past.std().item())
        #     print("Troubleshoot y_fut mean: ", y_fut.mean().item())
        #     print("Troubleshoot y_fut std: ", y_fut.std().item())

        return {
            "ds": ds,
            "unique_id":unique_id,
            "y_past": torch.from_numpy(y_past),
            "c_ctx_future_unknown": torch.from_numpy(cu_past),
            "c_ctx_future_known": torch.from_numpy(ck_past),
            "c_fct_future_known": torch.from_numpy(ck_fut),
            "y_future": torch.from_numpy(y_fut),
        }
    
    def _ok_scale(self, sc):
        s = getattr(sc, "scale_", None)
        if s is None: 
            return False
        s0 = float(s[0]) if np.ndim(s) == 1 else float(s)
        return s0 > 1e-8
    

# def _safe_transform(sc, x):
#     scale = getattr(sc, "scale_", None)
#     if scale is None:  
#         return x
#     # if single feature, scale is scalar-like
#     s = float(scale) if np.ndim(scale)==0 else float(scale[0])
#     if s < 1e-8:
#         return x  # skip scaling constant feature
#     return sc.transform(x)