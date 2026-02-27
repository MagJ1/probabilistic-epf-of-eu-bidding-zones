# src/models/nhits_qra/qra/qra_io.py
from __future__ import annotations
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np
import pandas as pd
import json
from dataclasses import dataclass
from omegaconf import OmegaConf

from models.nhits_qra.qra.qra_time import ensure_dtindex


def prepare_qra_subdirs(dirs: dict) -> dict:
    out_qra      = Path(dirs["data"]) / "qra"
    design_dir   = out_qra / "design_matrices"
    val_raw_dir  = out_qra / "val_pred_raw"
    val_fix_dir  = out_qra / "val_pred_postprocessed"
    qra_ckpt_dir = Path(dirs["ckpt"]) / "qra"
    metrics_qra  = Path(dirs["metrics"]) / "qra"
    pred_dir     = Path(dirs["pred"]) if "pred" in dirs else None

    for p in [out_qra, design_dir, val_raw_dir, val_fix_dir, qra_ckpt_dir, metrics_qra]:
        p.mkdir(parents=True, exist_ok=True)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    return {
        "out_qra": out_qra,
        "design_dir": design_dir,
        "val_raw_dir": val_raw_dir,
        "val_fix_dir": val_fix_dir,
        "qra_ckpt_dir": qra_ckpt_dir,
        "metrics_qra": metrics_qra,
        "pred": pred_dir,   # may be None during train/val
    }


def save_design(design_dir: Path, Ztr, ytr, Zval, yval):
    design_dir.mkdir(parents=True, exist_ok=True)
    for h, (Xt, yt, Xv, yv) in enumerate(zip(Ztr, ytr, Zval, yval)):
        np.save(design_dir / f"train_X_h{h}.npy", Xt)
        np.save(design_dir / f"train_y_h{h}.npy", yt)
        np.save(design_dir / f"val_X_h{h}.npy",   Xv)
        np.save(design_dir / f"val_y_h{h}.npy",   yv)


def save_pred_bundles(prefix_raw: Path, prefix_fix: Path, Q_raw_per_h, Q_fix_per_h, idx_df: Optional[pd.DataFrame]=None, taus=None):
    prefix_raw.mkdir(parents=True, exist_ok=True)
    prefix_fix.mkdir(parents=True, exist_ok=True)
    for h, Qr in enumerate(Q_raw_per_h):
        if isinstance(Qr, np.ndarray) and Qr.size:
            np.save(prefix_raw / f"Q_raw_h{h}.npy", Qr)
    for h, Qf in enumerate(Q_fix_per_h):
        if isinstance(Qf, np.ndarray) and Qf.size:
            np.save(prefix_fix / f"Q_fixed_h{h}.npy", Qf)
    if idx_df is not None:
        idx_df.to_parquet(prefix_raw / "index.parquet", index=False)
        idx_df.to_parquet(prefix_fix / "index.parquet", index=False)
    if taus is not None:
        (prefix_fix / "taus_used.json").write_text(json.dumps(list(taus), indent=2))


def emit_per_origin_files(
    pred_dir: Path,
    uid: np.ndarray,
    origin_ds,
    es_bh: Optional[np.ndarray] = None,    # (N, H_eff)
    crps_bh: Optional[np.ndarray] = None,  # (N, H_eff)
    write: str = "csv",
):
    """Write per-origin mean ES/CRPS once (arrays already masked and aligned)."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    ods = ensure_dtindex(origin_ds)

    def _write(df: pd.DataFrame, name: str):
        if write == "parquet":
            df.to_parquet(pred_dir / f"{name}.parquet", index=False)
        elif write == "csv":
            df.to_csv(pred_dir / f"{name}.csv", index=False)
        else:
            raise ValueError("write must be 'csv' or 'parquet'")

    if isinstance(es_bh, np.ndarray) and es_bh.size:
        df = pd.DataFrame({
            "unique_id": uid,
            "origin_ds": ods.strftime("%Y-%m-%d %H:%M:%S"),
            "es_mean":   es_bh.mean(axis=1).astype(float),
        })
        _write(df, "es_per_origin")

    if isinstance(crps_bh, np.ndarray) and crps_bh.size:
        df = pd.DataFrame({
            "unique_id": uid,
            "origin_ds": ods.strftime("%Y-%m-%d %H:%M:%S"),
            "crps_mean": crps_bh.mean(axis=1).astype(float),
        })
        _write(df, "crps_per_origin")


@dataclass(frozen=True)
class QRAPaths:
    # base
    out_qra: Path
    design_dir: Path
    val_raw_dir: Path
    val_fix_dir: Path
    qra_ckpt_dir: Path
    metrics_dir: Path

    # derived
    pred_raw_dir: Path
    pred_fix_dir: Path
    general_pred_dir: Optional[Path]  # only for test

def prepare_paths(ctx, dirs: Dict[str, Path]) -> tuple[Dict[str, Path], QRAPaths]:
    """
    Returns:
      - paths_dict (from prepare_qra_subdirs)
      - QRAPaths (strongly-typed convenience bundle)
    """
    paths = prepare_qra_subdirs(dirs)

    out_qra      = Path(paths["out_qra"])
    design_dir   = Path(paths["design_dir"])
    val_raw_dir  = Path(paths["val_raw_dir"])
    val_fix_dir  = Path(paths["val_fix_dir"])
    qra_ckpt_dir = Path(paths["qra_ckpt_dir"])
    metrics_dir  = Path(paths["metrics_qra"])

    # mkdirs
    for p in (out_qra, design_dir, val_raw_dir, val_fix_dir, qra_ckpt_dir, metrics_dir):
        p.mkdir(parents=True, exist_ok=True)

    general_pred_dir = None
    if ctx.split == "test":
        general_pred_dir = Path(paths["pred"])
        general_pred_dir.mkdir(parents=True, exist_ok=True)

    pred_raw_dir = val_raw_dir if ctx.split != "test" else (out_qra / "test_pred_raw")
    pred_fix_dir = val_fix_dir if ctx.split != "test" else (out_qra / "test_pred_postprocessed")
    pred_raw_dir.mkdir(parents=True, exist_ok=True)
    pred_fix_dir.mkdir(parents=True, exist_ok=True)

    p = QRAPaths(
        out_qra=out_qra,
        design_dir=design_dir,
        val_raw_dir=val_raw_dir,
        val_fix_dir=val_fix_dir,
        qra_ckpt_dir=qra_ckpt_dir,
        metrics_dir=metrics_dir,
        pred_raw_dir=pred_raw_dir,
        pred_fix_dir=pred_fix_dir,
        general_pred_dir=general_pred_dir,
    )
    return paths, p

def first_nonempty(models_per_h: List[Dict]) -> Optional[Dict]:
    for m in models_per_h:
        if m:
            return m
    return None


def safe_get(cfg: Any, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def resolve_tb_root(train_cfg: OmegaConf) -> Path:
    """
    Resolve to:
      outputs/nhits_qra/<model_size>/<experiment>/logs_tb
    using hydra runtime cwd if available.
    """
    # Prefer hydra original cwd (same as used in yaml via ${hydra:runtime.cwd})
    runtime_cwd = safe_get(train_cfg, "hydra.runtime.cwd", None)
    base = Path(runtime_cwd).expanduser().resolve() if runtime_cwd else Path.cwd().expanduser().resolve()

    model_size = str(safe_get(train_cfg, "model.size", "unknown"))
    experiment = str(safe_get(train_cfg, "experiment", "default"))

    return base / "outputs" / "nhits_qra" / model_size / experiment / "logs_tb"