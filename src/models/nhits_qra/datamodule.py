# src/models/nhits_qra/datamodule.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
from data.base_datamodule import BaseForecastingDataModule
from models.nhits_qra.dataset import NHITSQRAForecasterDataset
from utils.feature_select import select_features

class NHITSQRADataModule(BaseForecastingDataModule):
    dataset_cls = NHITSQRAForecasterDataset
    dataset_kwargs_extra = {}  # potentially setting defaults

    def __init__(self,
                 *args,
                 registry: Optional[Dict[str, Dict[str, Any]]] = None,
                 enable_registry: bool = True,
                 fit_on: str = "train",
                 scale_target: bool = True,
                 scale_data: bool = True,
                 share_scalers: bool = True,
                 active_ck: Optional[List[str]] = None,
                 active_cu: Optional[List[str]] = None,
                 active_static: Optional[List[str]] = None,
                 include_feature: Optional[List[str]] = None,
                 exclude_feature: Optional[List[str]] = None,
                 **kwargs):
        
        super().__init__(*args, share_scalers=share_scalers, **kwargs)

        self.registry = registry or {}
        self.enable_registry = enable_registry
        self.fit_on = fit_on
        self.scale_target = scale_target
        self.scale_data = scale_data

        # active lists (explicit wins over auto-derivation)
        self.active_ck = active_ck or []
        self.active_cu = active_cu or []
        self.active_static = active_static or []

        # optional quick overrides
        self.include_feature = include_feature or []
        self.exclude_feature = exclude_feature or []

        self._dataset_kwargs_extra.update(dict(
            registry=self.registry,
            enable_registry=self.enable_registry,
            fit_on=self.fit_on,
            scale_target=self.scale_target,
            scale_data=self.scale_data,
        ))


    # ---- extend setup: pick features, then call base.setup ----
    def setup(self, stage=None):
        ck, cu, st = select_features(
            enable_registry=self.enable_registry,
            registry=self.registry,
            base_ck=self.ck_cols, base_cu=self.cu_cols, base_static=getattr(self, "static_cols", []),
            active_ck=self.active_ck, active_cu=self.active_cu, active_static=self.active_static,
            include=self.include_feature or getattr(self, "include", []),
            exclude=self.exclude_feature or getattr(self, "exclude", []),
        )
        self.ck_cols, self.cu_cols, self.static_cols = ck, cu, st
        super().setup(stage)