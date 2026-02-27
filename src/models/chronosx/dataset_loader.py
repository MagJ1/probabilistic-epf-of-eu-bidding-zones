from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from gluonts.dataset.field_names import FieldName

def _is_nullish(x) -> bool:
    # YAML `null` becomes Python None via OmegaConf.
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in ("null", "none", ""):
        return True
    return False

def _normalize_probs(ps: List[float], n: int) -> List[float]:
    if not ps:
        return [1.0 / n] * n
    vals = [float(p) for p in ps]
    s = sum(vals)
    if s <= 0:
        raise ValueError("Sum of dataset probabilities must be > 0.")
    return [p / s for p in vals]

def _ensure_freq(period_index: pd.PeriodIndex) -> str:
    # period freq string like "H"
    return period_index.freqstr

def _to_period(ts: pd.Series) -> pd.PeriodIndex:
    # convert datetime-like to PeriodIndex with inferred freq
    dt = pd.DatetimeIndex(ts)
    try:
        freq = pd.infer_freq(dt)
    except Exception:
        freq = None
    if freq is None:
        # silently default to hourly; DST can break infer_freq
        freq = "h"
    return dt.to_period(freq=freq)

def _ffill_then_zero(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df[cols].ffill()
    # if leading NaNs remain after ffill, fill with 0
    df[cols] = df[cols].fillna(0.0)
    return df

def _build_entries(
    frame: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    fillna_forward: bool,
) -> list[dict]:
    if fillna_forward:
        frame = frame.sort_values([id_col, date_col])
        frame = _ffill_then_zero(frame, [target_col] + feature_cols)

    groups = frame.groupby(id_col, sort=False)
    entries = []
    for uid, g in groups:
        g = g.sort_values(date_col)
        # start as Period
        pidx = _to_period(g[date_col])
        start = pidx[0]
        # arrays
        target = g[target_col].to_numpy(dtype=float)  # (T,)
        feats = g[feature_cols].to_numpy(dtype=float).T  # (C, T)  <-- time last, Instance_Splitter will change order later
        entry = {
            "start": start,
            "target": target,
            FieldName.FEAT_DYNAMIC_REAL: feats,  # (C, T)
        }
        entries.append(entry)
    return entries

def load_from_csv(
    train_csv_path: str,
    eval_csv_path: Optional[str],
    id_col: str,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    fillna_forward: bool = True,
    include_covariates: bool = True,
) -> Tuple[list[dict], Optional[list[dict]]]:
    """
    Returns (train_entries, eval_entries). `eval_entries` can be validation or test, depending on the caller.

    Each entry is a dict with start (pd.Period), target (np.ndarray T,), and
    feat_dynamic_real as (T, C) if include_covariates=True.
    """
    train_df = pd.read_csv(train_csv_path)
    if include_covariates:
        missing = [c for c in feature_cols if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in train CSV: {missing}")

    train_entries = _build_entries(
        frame=train_df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        feature_cols=feature_cols if include_covariates else [],
        fillna_forward=fillna_forward,
    )

    eval_entries = None
    _eval_set = (eval_csv_path is not None) and (str(eval_csv_path).strip().lower() not in {"", "none", "null"})
    if _eval_set:
        test_df = pd.read_csv(eval_csv_path)
        if include_covariates:
            missing = [c for c in feature_cols if c not in test_df.columns]
            if missing:
                raise ValueError(f"Missing feature columns in test CSV: {missing}")

        eval_entries = _build_entries(
            frame=test_df,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            feature_cols=feature_cols if include_covariates else [],
            fillna_forward=fillna_forward,
        )

    return train_entries, eval_entries


def _load_single_dataset(cfg_data: DictConfig,
                         features_order: List[str],
                         fillna_forward: bool):
    
    tr_raw = cfg_data.get("train_csv_path", None)
    va_raw = cfg_data.get("val_csv_path", None)

    if _is_nullish(tr_raw):
        raise ValueError("data.train_csv_path is null/None/empty in single-dataset mode.")
    
    tr_abs = to_absolute_path(tr_raw)
    va_abs = None if _is_nullish(va_raw) else to_absolute_path(va_raw)
    
    t_gts, v_gts = load_from_csv(
        train_csv_path=tr_abs,
        eval_csv_path=va_abs,
        id_col=cfg_data.id_col,
        date_col=cfg_data.date_col,
        target_col=cfg_data.target_col,
        feature_cols=features_order,
        fillna_forward=bool(fillna_forward),
        include_covariates=True,
    )
    train_list = [t_gts]
    val_list = [v_gts] if v_gts else None
    probs = [1.0]
    names = ["single"]
    return train_list, val_list, probs, names

def _load_concatenated_dataset(cfg_data: DictConfig,
                          features_order: List[str],
                          fillna_forward: bool):
    srcs = cfg_data.concatenated_dataset
    if not srcs:
        raise ValueError(
            "Provide data.concatenated_dataset when data.train_csv_path and data.val_csv_path are null."
        )

    train_list, val_list, names, probs = [], [], [], []

    for i, src in enumerate(srcs):
        name  = str(getattr(src, "name", f"src_dataset{i}"))
        trRaw = getattr(src, "train_csv_path", None)
        vaRaw = getattr(src, "val_csv_path", None)
        pRaw  = getattr(src, "prob", None)

        if _is_nullish(trRaw):
            raise ValueError(f"concatenated_dataset[{i}] '{name}': train_csv_path is required (got null).")
        
        tr_abs = to_absolute_path(trRaw)
        va_abs = None if _is_nullish(vaRaw) else to_absolute_path(vaRaw)

        t_gts, v_gts = load_from_csv(
        train_csv_path=tr_abs,
        eval_csv_path=va_abs,          # may be None
        id_col=cfg_data.id_col,
        date_col=cfg_data.date_col,
        target_col=cfg_data.target_col,
        feature_cols=features_order,
        fillna_forward=bool(fillna_forward),
        include_covariates=True,
        )

        if not t_gts:
            raise RuntimeError(f"concatenated_dataset[{i}] '{name}': train set produced 0 series after loading.")

        train_list.append(t_gts)
        if v_gts:
            val_list.append(v_gts)

        names.append(name)
        probs.append(1.0 if pRaw is None else float(pRaw))

    probs = _normalize_probs(probs, len(train_list))
    if len(val_list) == 0:
        val_list = None
    return train_list, val_list, probs, names