# src/tools/postprocess_berkowitz.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Prefer your own utilities if available
from omegaconf import OmegaConf
from scipy.stats import chi2

# These exist in your repo:
# - dataset loader
from models.chronosx.dataset_loader import load_from_csv
# - metric helpers (PITs + Berkowitz; CRPS optional sanity check)
from utils.metrics import pits_from_samples, berkowitz_lr_test, crps_terms_fast

from gluonts.dataset.field_names import FieldName


def recover_berkowitz_lr(
    p_table: pd.DataFrame,
    df: int = 3,
    eps: float = 1e-300,
) -> pd.DataFrame:
    """
    Recover Berkowitz LR statistics from a wide table of p-values.

    Parameters
    ----------
    p_table : pd.DataFrame
        Wide table: index = horizons (e.g., 'h0','h1',...), columns = model names,
        values = p-values from Berkowitz LR test.
    df : int, default 3
        Degrees of freedom used in the LR test (Berkowitz uses 3).
    eps : float, default 1e-300
        Lower clipping for p to avoid inf LR when p == 0. Values <= eps are treated as eps.

    Returns
    -------
    pd.DataFrame
        Wide table of LR values aligned to p_table (same index/columns).
    """
    p = p_table.astype(float)
    # clip to (0,1) open interval to keep finite LR
    p_clipped = p.clip(lower=eps, upper=1.0 - 1e-16)
    lr = pd.DataFrame(
        chi2.isf(p_clipped, df=df),
        index=p.index,
        columns=p.columns,
    )
    return lr

def combine_p_lr_and_save(
    p_table: pd.DataFrame,
    out_csv: str | Path,
    df: int = 3,
    eps: float = 1e-300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute LR from p_table, create a tidy (long) table with both p and LR,
    and save it to CSV.

    Returns
    -------
    (lr_wide, tidy)
      - lr_wide: wide LR table (same shape as p_table)
      - tidy: long DataFrame with columns: ['horizon','model','p_value','LR','df','p_was_clipped']
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    lr_wide = recover_berkowitz_lr(p_table, df=df, eps=eps)

    # Build tidy long-form with both p and LR
    p_long = p_table.stack().rename("p_value").to_frame()
    lr_long = lr_wide.stack().rename("LR").to_frame()
    tidy = (
        p_long.join(lr_long)
              .reset_index()
              .rename(columns={"level_0": "horizon", "level_1": "model"})
    )
    # mark clipped p-values (use same eps as in recover_berkowitz_lr)
    tidy["p_was_clipped"] = tidy["p_value"] <= eps
    tidy["df"] = int(df)

    tidy.to_csv(out_csv, index=False)
    return lr_wide, tidy







# ---------- small helpers (copied logic from your test script) ----------

def _nonempty(x) -> bool:
    return x is not None and str(x).lower() not in {"", "none", "null"}

def _normalize_feats_to_time_channel(
    feats: np.ndarray,
    target_len: int,
    num_covariates: int,
) -> np.ndarray:
    """
    Ensure feats has shape (T, C). Accepts (T,C) or (C,T).
    """
    if int(num_covariates) == 0:
        return np.empty((int(target_len), 0), dtype=np.float32)

    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if feats.ndim != 2:
        raise ValueError(f"feat_dynamic_real must be 2D, got {feats.shape}")

    r, c = feats.shape
    if c == num_covariates:
        pass  # (T,C)
    elif r == num_covariates:
        feats = feats.T  # (C,T) -> (T,C)
    else:
        raise ValueError(
            f"feat_dynamic_real incompatible shape {feats.shape}; "
            f"expected (T,{num_covariates}) or ({num_covariates},T)"
        )

    return feats

def _iter_sliding_windows(
    entries: List[Dict[str, Any]],
    context_len: int,
    pred_len: int,
    stride_hours: int,
    num_covariates: int,
    anchor_hour: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Minimal replay of test iterator to recover only what we need:
      - future_target (length H)
      - forecast_start
      - item_id
    """
    for ts in entries:
        target = np.asarray(ts["target"], dtype=np.float32)           # (T,)
        feats_raw = ts.get(FieldName.FEAT_DYNAMIC_REAL, None)
        if feats_raw is None:
            feats = np.empty((len(target), 0), dtype=np.float32)
        else:
            feats = _normalize_feats_to_time_channel(feats_raw, len(target), num_covariates)

        T = int(min(len(target), feats.shape[0]))
        t_min = context_len
        t_max = T - pred_len
        if t_max < t_min:
            continue

        # stride + optional anchor
        if anchor_hour is None:
            first_t = t_min
            step = max(1, int(stride_hours))
        else:
            step = int(stride_hours) if int(stride_hours) >= 24 else 24
            start_period = ts["start"]
            found = None
            for t in range(t_min, min(t_min + 24, t_max + 1)):
                try:
                    if (start_period + t).hour == int(anchor_hour):
                        found = t; break
                except Exception:
                    if (start_period.to_timestamp() + pd.Timedelta(hours=t)).hour == int(anchor_hour):
                        found = t; break
            if found is None:
                continue
            first_t = found

        uid = ts.get("item_id", ts.get(FieldName.ITEM_ID, 1))

        t = first_t
        while t <= t_max:
            future_target = target[t : t + pred_len]  # (H,)
            yield {
                "item_id": uid,
                "future_target": future_target,
                "forecast_start": ts["start"] + t,
            }
            t += step


# ---------- config + run discovery ----------

def _find_run_root(path: Path) -> Path:
    """
    Accept a run folder (root) or a nested subfolder (e.g. .../test/DM).
    Returns the root folder containing .hydra.
    """
    p = path.resolve()
    for _ in range(5):
        if (p / ".hydra").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate a .hydra directory above the given path.")

def _load_hydra_cfg(run_root: Path) -> Any:
    cfg_path = run_root / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")
    return OmegaConf.load(str(cfg_path))

def _maybe_load_train_cfg_from_pointer(run_root: Path) -> Optional[Any]:
    """
    If the test was from a finetuned run, your test code writes
    'tested_from_run_dir.txt' at the run root. Use it to load the train config.
    """
    pointer = run_root / "tested_from_run_dir.txt"
    if not pointer.exists():
        return None
    train_dir = Path(pointer.read_text().strip())
    train_cfg_path = train_dir / "meta" / "resolved_config.yaml"
    if train_cfg_path.exists():
        return OmegaConf.load(str(train_cfg_path))
    return None

def _discover_predictions_dir(run_root: Path) -> Path:
    """
    Find the predictions folder that contains forecast_samples.npz.
    Usually: <run_root>/test/<TAG>/predictions
    """
    candidates = list(run_root.glob("**/predictions/forecast_samples.npz"))
    if not candidates:
        raise FileNotFoundError("forecast_samples.npz not found below run root.")
    # Pick the shortest path (closest to root)
    best = min(candidates, key=lambda p: len(p.as_posix()))
    return best.parent


# ---------- main postprocess steps ----------

def _resolve_data_knobs(run_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, int, int, Optional[int], int]:
    """
    Returns:
      (cfg_used, test_entries, context_len, pred_len, stride, anchor_hour, num_covariates)
    """
    cfg = _load_hydra_cfg(run_root)
    train_cfg = _maybe_load_train_cfg_from_pointer(run_root)

    # Extract test knobs (always in the test Hydra)
    D = cfg.data
    pred_len     = int(D.prediction_length)
    context_len  = int(D.context_length)
    stride       = int(getattr(D, "origin_stride_test", 24))
    anchor_raw   = getattr(D, "origin_anchor_hour_test", None)
    anchor_hour  = int(anchor_raw) if (anchor_raw is not None and str(anchor_raw) != "None") else None

    # Feature order (prefer train's persisted order if available)
    if train_cfg is not None and "features" in train_cfg and "features_order" in train_cfg.features:
        features_order = list(train_cfg.features.features_order)
    else:
        # Fall back to current cfg.features
        ck_cols = list(cfg.features.ck_cols) if "features" in cfg and "ck_cols" in cfg.features else []
        cu_cols = list(cfg.features.cu_cols) if "features" in cfg and "cu_cols" in cfg.features else []
        features_order = ck_cols + cu_cols

    use_covs = True
    if "features" in cfg and "use_covariates" in cfg.features:
        use_covs = bool(cfg.features.use_covariates)
    num_covariates = (len(features_order) if use_covs else 0)

    # Test file path (for finetuned runs you often pass train_csv_path=None; we only need the eval/test CSV)
    eval_csv_path = str(D.test_csv_path)
    train_csv_path = getattr(D, "train_csv_path", None)
    train_csv_path = (str(train_csv_path) if _nonempty(train_csv_path) else None)

    # Load test data with the SAME feature order used at test time
    _, test_entries = load_from_csv(
        train_csv_path=train_csv_path,
        eval_csv_path=eval_csv_path,
        id_col=D.id_col,
        date_col=D.date_col,
        target_col=D.target_col,
        feature_cols=features_order,
        fillna_forward=bool(getattr(D, "fillna_forward", True)),
        include_covariates=(num_covariates > 0),
    )
    if not test_entries:
        raise RuntimeError("No test data loaded.")

    return (cfg, test_entries, context_len, pred_len, stride, anchor_hour, num_covariates)

def rebuild_truth_ordered(
    run_root: Path,
) -> Tuple[np.ndarray, List[Any], List[Any]]:
    """
    Rebuild Y_{B,H} (truth per window) and aligned metadata (uids, forecast_starts).
    """
    cfg, test_entries, C, H, stride, anchor, Ccov = _resolve_data_knobs(run_root)
    windows = list(_iter_sliding_windows(
        entries=test_entries,
        context_len=C,
        pred_len=H,
        stride_hours=stride,
        num_covariates=Ccov,
        anchor_hour=anchor,
    ))
    if not windows:
        raise RuntimeError("No windows generated with the recovered knobs.")

    Y = np.stack([w["future_target"] for w in windows], axis=0)  # (B,H)
    uids = [w["item_id"] for w in windows]
    fstarts = [w["forecast_start"] for w in windows]
    return Y, uids, fstarts

def load_samples_BSH(pred_dir: Path) -> np.ndarray:
    """
    Load saved forecast samples, shape (B, S, H).
    """
    npz = np.load(pred_dir / "forecast_samples.npz")
    samples_BSH = npz["samples"]
    if samples_BSH.ndim != 3:
        raise ValueError(f"Expected (B,S,H), got {samples_BSH.shape}")
    return samples_BSH

def compute_berkowitz_per_horizon(samples_BSH: np.ndarray, Y_BH: np.ndarray) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: horizon, p_value, lr_stat, n
    """
    # S,B,H
    S_B_H = samples_BSH.transpose(1, 0, 2)
    if Y_BH.shape != (S_B_H.shape[1], S_B_H.shape[2]):
        raise ValueError(f"Shape mismatch: Y{Y_BH.shape} vs samples{S_B_H.shape}")

    # PITs: (B,H)
    U_B_H = pits_from_samples(S_B_H, Y_BH, randomized=True)

    # Berkowitz per horizon
    from scipy.stats import norm
    H = U_B_H.shape[1]
    rows = []
    for h in range(H):
        U_h = np.asarray(U_B_H[:, h], dtype=float)
        U_h = U_h[np.isfinite(U_h)]
        if U_h.size == 0:
            rows.append({"horizon": h, "p_value": np.nan, "lr_stat": np.nan, "n": 0})
            continue
        Z_h = norm.ppf(np.clip(U_h, 1e-6, 1 - 1e-6))
        out = berkowitz_lr_test(Z_h.reshape(-1))
        rows.append({
            "horizon": h,
            "p_value": float(out.get("p", np.nan)),
            "lr_stat": float(out.get("lr_stat", np.nan)),
            "n": int(Z_h.size),
        })
    return pd.DataFrame(rows)

def optional_crps_sanity_check(samples_BSH: np.ndarray, Y_BH: np.ndarray) -> float:
    """
    Recompute mean CRPS over origins to verify ordering matches your recorded metric.
    """
    import torch
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    S_B_H = samples_BSH.transpose(1, 0, 2)  # (S,B,H)
    t_samples = torch.tensor(S_B_H, device=device, dtype=torch.float32)
    t_y_true  = torch.tensor(Y_BH,  device=device, dtype=torch.float32)
    fit, spread = crps_terms_fast(t_samples, t_y_true)
    crps_per_origin = (fit - spread).mean(dim=1).detach().cpu().numpy()
    return float(np.mean(crps_per_origin))

def main():
    ap = argparse.ArgumentParser(description="Postprocess: rebuild Y and compute Berkowitz per horizon from saved samples.")
    ap.add_argument("--run", type=str, required=True,
                    help="Path to the test run root *or* a child dir inside it (the script will locate .hydra upward).")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional output dir (default: reuse the run's predictions dir).")
    ap.add_argument("--sanity_crps", action="store_true",
                    help="Also recompute mean CRPS as a sanity check.")
    args = ap.parse_args()

    run_root = _find_run_root(Path(args.run))
    cfg = _load_hydra_cfg(run_root)

    pred_dir = _discover_predictions_dir(run_root)
    samples = load_samples_BSH(pred_dir)

    # Rebuild Y and metadata
    Y, uids, fstarts = rebuild_truth_ordered(run_root)

    if Y.shape[0] != samples.shape[0] or Y.shape[1] != samples.shape[2]:
        raise RuntimeError(
            f"Samples and truth shapes incompatible: samples={samples.shape}, truth={Y.shape}"
        )

    # Compute Berkowitz per horizon
    df_berk = compute_berkowitz_per_horizon(samples, Y)

    # Optional: CRPS sanity check
    metrics = {}
    if args.sanity_crps:
        try:
            crps_mean = optional_crps_sanity_check(samples, Y)
            metrics["crps_mean_over_all_origins_recomputed"] = crps_mean
        except Exception as e:
            metrics["crps_sanity_error"] = str(e)

    # Save outputs
    out_dir = Path(args.out) if args.out else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_berk.to_parquet(out_dir / "berkowitz_per_horizon.parquet", index=False)
    df_berk.to_csv(out_dir / "berkowitz_per_horizon.csv", index=False)

    # Also drop alignment metadata if one want to inspect
    pd.DataFrame({"unique_id": uids, "origin_ds": [str(s) for s in fstarts]}).to_csv(
        out_dir / "window_order.csv", index=False
    )

    if metrics:
        import json
        (out_dir / "postprocess_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved: {out_dir/'berkowitz_per_horizon.csv'}")
    if "crps_mean_over_all_origins_recomputed" in metrics:
        print(f"Recomputed mean CRPS: {metrics['crps_mean_over_all_origins_recomputed']:.6f}")

if __name__ == "__main__":
    main()


# ((.investigations_venv) ) jlettner@mac-book-air prob-electricity-price-forecasting % python3 -m evaluation.helpers.postprocess_berkowitz \
# --run src/evaluation/E1/model_scores/outputs/chronosx/small/zero_shot/runs/2025-11-04_19-47-33-zero_shot_de_lu \
# --sanity_crps
# Saved: /Users/jlettner/Documents/Jan/Studium/KIT/Master/4Semester/MA/prob-electricity-price-forecasting/src/evaluation/E1/model_scores/outputs/chronosx/small/zero_shot/runs/2025-11-04_19-47-33-zero_shot_de_lu/predictions/berkowitz_per_horizon.csv
# Recomputed mean CRPS: 16.334452