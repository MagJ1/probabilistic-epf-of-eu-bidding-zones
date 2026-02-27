# src/models/nhits_qra/dataset.py
from typing import TypedDict, Optional, Union, Dict, Any, Iterable, Tuple, List
import torch
import pandas as pd
import numpy as np

from data.base_dataset import BaseForecastingDataset

class SampleBatch(TypedDict):
    """Dictionary returned by FlowForecasterDataset.__getitem__()"""
    ds: list[str]
    unique_id: int
    y_past: torch.Tensor
    c_ctx_future_unknown: torch.Tensor
    c_ctx_future_known: torch.Tensor
    c_fct_future_known: torch.Tensor
    flags_ctx_cu_known: torch.Tensor
    static: torch.Tensor
    y_future: torch.Tensor

class NHITSQRAForecasterDataset(BaseForecastingDataset):
    """Dataset class for the NHITS+QRA model"""
    def __init__(self,
                 csv_path: str,
                 context_length: int = 168,
                 forecast_horizon: int = 24,
                 val_split_date: dict[str, Union[str, pd.Timestamp, None]] = {},
                 split_type: str = "train",  
                 date_col: str = "ds",
                 id_col: str = "unique_id",
                 y_col: str = "y",
                 ck_cols: list[str] = None,
                 cu_cols: list[str] = None,
                 static_cols: list[str] = None,
                 past_unknown_cov_cutoff: int = 14,
                 scale_data: bool = True,
                 realistic_mode: bool = True, 
                 registry: Optional[Dict[str, Dict[str, Any]]] = None, 
                 enable_registry: bool = True, 
                 fit_on: str = "train", 
                 scale_target: bool = True, 
                 target_scaler_state: Optional[Tuple[str, float, float]] = None
                 ):
        
        """Constructor

        Args:
            csv_path: Path to the csv file, containing the dataset
            context_length: Steps the model can see in the past
            forecast_horizon: Steps the model has to predict into the future
            scaler: Scaler that is used to scale the electricity prices
            val_split_date: Dict of dates per TS where so split into train and val.
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
                            static_cols=static_cols,
                            past_unknown_cov_cutoff=past_unknown_cov_cutoff,
                            realistic_mode=realistic_mode, 
                            registry=registry, 
                            enable_registry=enable_registry, 
                            fit_on=fit_on, 
                            scale_target=scale_target, 
                            target_scaler_state=target_scaler_state)

        self.scale_data = scale_data

    
    def __getitem__(self, idx: int) -> SampleBatch:
        """
        Returns:
            dict:
                "ds": list[str]  
                "unique_id": int
                "y_past": (context_length,)
                "c_ctx_future_unknown": (context_length, C_cu)
                "c_ctx_future_known":   (context_length, C_ck)
                "c_fct_future_known":   (forecast_horizon, C_ck)
                "flags_ctx_cu_known":   (context_length, C_cu)   # 1=available, 0=masked
                "static": (S,)
                "y_future": (forecast_horizon,)
        """
        sidx, t0 = self.indices[idx]
        ts = self.ts_collection[sidx]
        C, H = self.context_length, self.forecast_horizon

        # timestamps for the forecast window
        ds_window = ts["ds"][t0-C:t0+H]
        ds = [str(x) for x in ds_window]
        unique_id = ts["unique_id"]

        # raw slices
        y_past_raw  = ts["y"][t0-C:t0].copy()          # (C,)
        y_fut_raw   = ts["y"][t0:t0+H].copy()          # (H,)
        ck_past_raw = ts["ck"][t0-C:t0, :].copy()      # (C, C_ck)
        ck_fut_raw  = ts["ck"][t0:t0+H, :].copy()      # (H, C_ck)
        cu_past_raw = ts["cu"][t0-C:t0, :].copy()      # (C, C_cu)

        # scale / transform (no-op if registry disabled or scale_data=False)
        if self.scale_data:
            ck_past = self.transform_stack(self.ck_cols, ck_past_raw)
            ck_fut  = self.transform_stack(self.ck_cols, ck_fut_raw)
            cu_past = self.transform_stack(self.cu_cols, cu_past_raw)
            y_past  = self.transform_target(y_past_raw)
            y_fut   = self.transform_target(y_fut_raw)
        else:
            ck_past, ck_fut, cu_past = ck_past_raw, ck_fut_raw, cu_past_raw
            y_past, y_fut = y_past_raw, y_fut_raw

        # availability flags for cu in the CONTEXT (future-unknown latency)
        # 1 = available, 0 = not available (masked)
        if cu_past.shape[1] > 0:
            flags_ctx_cu_known = np.ones_like(cu_past, dtype="float32")
            if self.realistic_mode:
                cutoff = int(min(max(self.past_unknown_cov_cutoff, 0), C))
                if cutoff > 0:
                    flags_ctx_cu_known[-cutoff:, :] = 0.0
                    # neutralize masked values *after scaling* to mean (0 in normalized space)
                    cu_past[-cutoff:, :] = 0.0
        else:
            # keep shapes consistent if no cu features
            flags_ctx_cu_known = np.zeros((C, 0), dtype="float32")

        # static vector (already numeric from base; includes uid_int if configured)
        static_vec = ts["static"] if ts["static"].size else np.zeros((0,), dtype="float32")

        return {
            "ds": ds,
            "unique_id": unique_id,
            "y_past": torch.from_numpy(y_past),
            "c_ctx_future_unknown": torch.from_numpy(cu_past),
            "c_ctx_future_known": torch.from_numpy(ck_past),
            "c_fct_future_known": torch.from_numpy(ck_fut),
            "flags_ctx_cu_known": torch.from_numpy(flags_ctx_cu_known),
            "static": torch.from_numpy(static_vec),
            "y_future": torch.from_numpy(y_fut),
        }
    
    

# def _safe_transform(sc, x):
#     scale = getattr(sc, "scale_", None)
#     if scale is None:  
#         return x
#     # if single feature, scale is scalar-like
#     s = float(scale) if np.ndim(scale)==0 else float(scale[0])
#     if s < 1e-8:
#         return x  # skip scaling constant feature
#     return sc.transform(x)