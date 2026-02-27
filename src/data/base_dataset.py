# src/data/base_dataset.py
from typing import Optional, Union, Dict, Any, Iterable, Tuple, List
import copy
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

ScalerState = Dict[str, Tuple[str, float, float]]  # name -> (kind, a, b)

class BaseForecastingDataset(Dataset):
    """Base dataset class for forecasting models"""

    def __init__(self, 
                 csv_path: str, 
                 context_length: int = 168, 
                 forecast_horizon: int = 24, 
                 val_split_date: dict[str, Union[pd.Timestamp, str, None]] = {},
                 split_type: str = "train",  
                 date_col: str = "ds",
                 id_col: str = "unique_id",
                 y_col: str = "y",
                 ck_cols: Optional[List[str]] = None,
                 cu_cols: Optional[List[str]] = None,
                 flags_cols: Optional[List[str]] = None,
                 static_cols: Optional[List[str]] = None,
                 past_unknown_cov_cutoff: int = 14,
                 realistic_mode: bool = False,
                 registry: Optional[Dict[str, Dict[str, Any]]] = None,
                 enable_registry: bool = False,
                 fit_on: str = "train",
                 scale_target: bool = True, 
                 target_scaler_state: Optional[Tuple[str, float, float]] = None,
                 origin_stride: int = 1,        # step between forecast origins (in rows)
                 origin_anchor_hour: Optional[int] = None   # restrict origins to a specific hour (0-23)
                 ):
        """
        If `registry` is provided and enable_registry=True, roles and scaling behaviour
        are taken from it; otherwise ck_cols/cu_cols/static_cols are used as before.
        """

        self.date_col = date_col
        self.id_col = id_col
        self.y_col = y_col

        # registry for per-feature standardization, not implement in NF model yet
        self.enable_registry = bool(enable_registry)
        self.registry = registry or {}
        self.fit_on = fit_on

        # Derive role lists from registry if enabled; else use provided lists
        # if self.enable_registry and self.registry:
        #     self.ck_cols, self.cu_cols, self.flags_cols, self.static_cols = \
        #         self._roles_from_registry(self.registry)
        # else:
        self.ck_cols    = ck_cols or []
        self.cu_cols    = cu_cols or []
        self.flags_cols = flags_cols or []
        self.static_cols= static_cols or []

        # id mapping
        self._uid2int: Dict[str, int] = {}
        self._int2uid: Dict[int, str] = {}

        # feature map for time-varying only
        self.feature_map: Dict[str, Tuple[str, Optional[int]]] = {self.y_col: ("y", None)}
        self.feature_map.update({name: ("ck",    i) for i, name in enumerate(self.ck_cols)})
        self.feature_map.update({name: ("cu",    i) for i, name in enumerate(self.cu_cols)})
        self.feature_map.update({name: ("flags", i) for i, name in enumerate(self.flags_cols)})

        self.feature_names: List[str] = [*self.ck_cols, *self.cu_cols, *self.flags_cols]

        self.val_split_date = {
            str(k): (pd.Timestamp(v) if isinstance(v, str) else v)
            for k, v in (val_split_date or {}).items()
        }

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.split_type = split_type

        self.origin_stride = max(1, int(origin_stride))
        if origin_anchor_hour is not None:
            ah = int(origin_anchor_hour)
            if not (0 <= ah <= 23):
                raise ValueError("origin_anchor_hour must be in [0, 23].")
            self.origin_anchor_hour = ah
        else:
            self.origin_anchor_hour = None
        self.past_unknown_cov_cutoff = past_unknown_cov_cutoff
        self.realistic_mode = realistic_mode

        # Scalers are learned on the fly via fit_transforms(); default no-op
        self._scalers: ScalerState = {}

        df = self._load_and_preprocess(csv_path)
        self.ts_collection = self._construct_ts_collection(df)

        self._scale_target = bool(scale_target)
        self._y_scaler: Optional[Tuple[str, float, float]] = target_scaler_state

        # fit scaler on target, either on train or train+val
        if self._scale_target and self._y_scaler is None and self.split_type == "train":
            if self.fit_on not in {"train", "train+val"}:
                raise ValueError(f"Unknown fit_on policy: {self.fit_on}")

            if self.fit_on == "train":
                # per-UID default cutoff = that UID's last timestamp
                last_by_uid = df.groupby(self.id_col)[self.date_col].max()
                cuts = df[self.id_col].map(lambda u: self.val_split_date.get(str(u), last_by_uid.loc[u]))
                fit_df = df[df[self.date_col] <= cuts]
            else:  # "train+val"
                fit_df = df

            y = fit_df[self.y_col].to_numpy(dtype="float32")
            mu, std = float(np.nanmean(y)), float(np.nanstd(y) + 1e-8)
            self._y_scaler = ("z", mu, std)

        # If registry-enabled, optionally fit scalers here (only for the chosen split)
        if self.enable_registry:
            self._maybe_fit_transforms(df)

        self.indices = self._build_index()

    # -------------------------------------------------------------------------
    # Loading / construction
    # -------------------------------------------------------------------------
    def _load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, parse_dates=[self.date_col])
        df.sort_values([self.id_col, self.date_col], inplace=True)
        return df
    
    def _construct_ts_collection(self, df: pd.DataFrame):
        """Create time series collection out of dataframe. Therefore, dataframe is partitioned according to unique_id column. Then, optionally split into train and validation sets. Subsquently, columns are grouped into date (ds), unique_id, target (y), covariate known in the future (ck), covariates not known in the future (cu), flags and static features."""
        ts_collection = []
        for uid, g in df.groupby(self.id_col):
            if uid not in self._uid2int:
                idx = len(self._uid2int)
                self._uid2int[uid] = idx
                self._int2uid[idx] = uid

            uid_int = self._uid2int[uid]
            cut = self.val_split_date.get(str(uid))
            if cut is None:
                train_df = g
                val_df   = pd.DataFrame(columns=g.columns)
            else:
                train_df = g[g[self.date_col] <= cut]
                val_df   = g[g[self.date_col] >  cut]

            if self.split_type in {"train"}:
                split_df = train_df
            elif self.split_type == "val":
                split_df = val_df
            elif self.split_type == "test":
                split_df = g
            else:
                continue

            if split_df.empty:
                continue  # skip this series
            static_vec = self._static_vector(split_df, uid_int)
            ts_collection.append({
                "ds": split_df[self.date_col].to_numpy(),
                "unique_id": uid_int,
                "y":  split_df[self.y_col].to_numpy().astype("float32"),
                "ck": split_df[self.ck_cols].to_numpy().astype("float32") if self.ck_cols else np.zeros((len(split_df),0),dtype="float32"),
                "cu": split_df[self.cu_cols].to_numpy().astype("float32") if self.cu_cols else np.zeros((len(split_df),0),dtype="float32"),
                "flags": split_df[self.flags_cols].to_numpy().astype("float32") if self.flags_cols else np.zeros((len(split_df),0),dtype="float32"),
                "static": static_vec
            })
        if not ts_collection:
            raise ValueError(f"No data for split='{self.split_type}'. Check val_split_date.")
        return ts_collection

    def _build_index(self):
        """Build (series_idx, t0) honoring context, horizon, stride, and optional anchor.

        Semantics:
        - origin_stride is in HOURS.
        - Without anchor: take every `origin_stride`-th hour.
        - With anchor_hour=h: keep only t0 where hour(t0)==h; the effective minimum
            spacing is 24h. We then step by `max(1, origin_stride // 24)` on that
            anchored list (i.e., 24→daily, 168→weekly, etc.).
        - The last valid origin t0 = T - forecast_horizon is INCLUDED.
        """
        indices = []
        for s_idx, ts in enumerate(self.ts_collection):
            T = len(ts[self.y_col])
            start = self.context_length
            last  = T - self.forecast_horizon               # inclusive
            if last < start:
                continue

            if self.origin_anchor_hour is None:
                # hourly stride, inclusive upper bound
                for t0 in range(start, last + 1, self.origin_stride):
                    indices.append((s_idx, t0))
            else:
                # collect all anchored positions (hour == anchor), inclusive upper bound
                ds_hours = pd.to_datetime(ts["ds"]).hour
                anchored_positions = [t0 for t0 in range(start, last + 1)
                                    if ds_hours[t0] == self.origin_anchor_hour]
                if not anchored_positions:
                    continue
                # interpret stride as HOURS; reduce to days on the anchored grid
                step_days = max(1, self.origin_stride // 24)
                for k in range(0, len(anchored_positions), step_days):
                    indices.append((s_idx, anchored_positions[k]))

        return indices

    # -------------------------------------------------------------------------
    # Registry & transforms
    # -------------------------------------------------------------------------
    @staticmethod
    def _roles_from_registry(registry: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Hydra feature registry is resolved. There, every feature is assigend a role and whether to be standardized or not and if so, with which method. """

        ck, cu, flags, static = [], [], [], []
        for name, meta in registry.items():
            role = (meta.get("role") or "").lower()

            # disallow standardization for flags/static
            if role in {"flags", "static"} and meta.get("standardize", False):
                raise ValueError(f"Feature '{name}' has role '{role}' and must not be standardized.")

            # assign feature to the right role list
            if role == "ck":     ck.append(name)
            elif role == "cu":   cu.append(name)
            elif role == "flags": flags.append(name)
            elif role == "static": static.append(name)
            # ignore unknown roles

        return ck, cu, flags, static

    def _maybe_fit_transforms(self, df: pd.DataFrame):
        """Fit feature scalers on train (or train+val) rows only, using per-UID cutoffs."""

        # only fit on the training split
        if self.split_type != "train":
            return

        if self.fit_on == "train":
            # per-UID max timestamp as fallback
            per_uid_last = df.groupby(self.id_col)[self.date_col].transform("max")

            # map UID -> dict cutoff (may be NaT). dict is keyed by str(uid)
            mapped = df[self.id_col].astype(str).map(self.val_split_date)
            mapped = pd.to_datetime(mapped)      # ensure Timestamp/NaT dtype

            # choose dict cutoff if present, else per-UID last
            cuts = mapped.fillna(per_uid_last)

            # keep rows up to the cutoff for *that row's UID*
            fit_df = df[df[self.date_col] <= cuts]

        elif self.fit_on == "train+val":
            fit_df = df
        else:
            raise ValueError(f"Unknown fit_on policy: {self.fit_on}")

        names = self.ck_cols + self.cu_cols + self.static_cols
        names_to_fit = [
            n for n, m in self.registry.items()
            if n in names and m.get("standardize", False)
        ]
        if not names_to_fit:
            return

        for name in names_to_fit:
            role = (self.registry[name].get("role") or "").lower()
            if role not in {"ck", "cu"}:
                continue  # skip flags/static

            x = fit_df[name].to_numpy(dtype="float32")
            scaler = (self.registry[name].get("scaler") or "zscore").lower()

            if scaler == "zscore":
                mu, std = float(np.nanmean(x)), float(np.nanstd(x) + 1e-8)
                self._scalers[name] = ("z", mu, std)
            elif scaler == "robust":
                q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
                self._scalers[name] = ("r", float(q1), float((q3 - q1) + 1e-8))
            elif scaler == "minmax":
                mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
                self._scalers[name] = ("m", mn, float((mx - mn) + 1e-8))
            elif scaler in {"none", None}:
                continue
            else:
                raise ValueError(f"Unsupported scaler '{scaler}' for feature '{name}'")

    def _transform_feat(self, name: str, arr: np.ndarray) -> np.ndarray:
        """Apply per-feature transform if configured; otherwise no-op. Expects 1-D array."""
        s = self._scalers.get(name)
        if not s:
            return arr.astype("float32")
        kind, a, b = s
        if kind == "z":   out = (arr - a) / b
        elif kind == "r": out = (arr - a) / b
        elif kind == "m": out = (arr - a) / b
        else:             out = arr
        return out.astype("float32")

    def transform_stack(self, names: Iterable[str], seq2d: np.ndarray) -> np.ndarray:
        """Apply per-feature transforms column-wise and return stacked 2D array.
        seq2d: shape (T, C) matching names order."""
        if not self.enable_registry or not names:
            return seq2d.astype("float32")
        cols = []
        for j, name in enumerate(names):
            cols.append(self._transform_feat(name, seq2d[:, j]))
        return np.stack(cols, axis=-1)

    def transform_target(self, arr1d: np.ndarray) -> np.ndarray:
        if not self._scale_target or self._y_scaler is None:
            return arr1d.astype("float32")
        kind, mu, std = self._y_scaler
        if kind == "z":
            return ((arr1d - mu) / std).astype("float32")
        return arr1d.astype("float32")

    def inverse_target(self, arr1d: np.ndarray) -> np.ndarray:
        if not self._scale_target or self._y_scaler is None:
            return arr1d.astype("float32")
        kind, mu, std = self._y_scaler
        if kind == "z":
            return (arr1d * std + mu).astype("float32")
        return arr1d.astype("float32")
    
    def _static_vector(self, split_df: pd.DataFrame, uid_int: int) -> np.ndarray:
        if not self.static_cols:
            return np.zeros((0,), dtype="float32")

        out = np.empty((len(self.static_cols),), dtype="float32")
        for i, col in enumerate(self.static_cols):
            if col == self.id_col:
                out[i] = float(uid_int)  # mapped ID
            else:
                # assumes numeric; if not, add a per-column encoder
                out[i] = np.float32(split_df[col].iloc[0])
        return out
    # -------------------------------------------------------------------------
    # Public helpers (intended for child datasets / DM)
    # -------------------------------------------------------------------------
    def get_scalers(self) -> ScalerState:
        """Deep-copy of learned scalers for sharing/persistence."""
        return copy.deepcopy(self._scalers)

    def set_scalers(self, state: ScalerState):
        """Load scalers learned elsewhere (e.g., from train split)."""
        self._scalers = copy.deepcopy(state) if state is not None else {}

    def set_target_scaler(self, state: Optional[Tuple[str, float, float]]):
        self._y_scaler = state

    def get_target_scaler(self) -> Optional[Tuple[str, float, float]]:
        return self._y_scaler
    
    def uid_to_int(self, uid: str) -> int:
        return self._uid2int[uid]

    def int_to_uid(self, idx: int) -> str:
        return self._int2uid[idx]


    # -------------------------------------------------------------------------
    # Torch dataset plumbing
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        raise NotImplementedError

    # Column access utilities (time-varying only)
    def get_col(self, name: str) -> np.ndarray:
        if name == self.y_col:
            return np.concatenate([ts["y"] for ts in self.ts_collection], axis=0)
        kind, j = self.feature_map.get(name, (None, None))
        if kind in {"ck","cu","flags"}:
            return np.concatenate([ts[kind][:, j] for ts in self.ts_collection], axis=0)
        raise KeyError(f"Unknown column: {name}")
    
    def slice_feature(self, ts: dict, name: str, start: int, end: int) -> np.ndarray:
        if name == self.y_col:
            return ts["y"][start:end]
        kind, j = self.feature_map[name]
        if kind in {"ck","cu","flags"}:
            return ts[kind][start:end, j]
        raise KeyError(f"Unknown column: {name}")