# src/models/chronosx/test_chronosx.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd
import torch
import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from utils.paths import prepare_test_run_dirs
from utils.logging_utils import init_logging, get_logger

from models.chronosx.dataset_loader import load_from_csv
from models.chronosx.custom_pipeline import ChronosXPipelineMinimal
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import batcher
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore", message="Could not infer frequency; defaulting to hourly")

from utils.plotting import (
    _reconstruct_time_axes,
    _plot_fanchart_steps,
    _plot_fanchart_lines,
    _plot_pit_histogram,
)

from utils.metrics import (
    crps_terms_fast,
    es_terms_mv_beta,
    ece_from_samples_per_horizon,
    pits_from_samples,
    berkowitz_lr_test,
    sliced_energy_score,

)

# --- progress bar (tqdm) with safe fallback ---
try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(iterable=None, total=None, desc=None, unit=None, position=0, disable=False):
        class _Dummy:
            def __init__(self, iterable):
                self.iterable = iterable if iterable is not None else []
            def update(self, n=1): pass
            def close(self): pass
            def __iter__(self): return iter(self.iterable)
        return _Dummy(iterable)

def _nonempty(x):
    return x is not None and str(x).lower() not in {"", "none", "null"}

def _estimate_total_windows(entries, context_len, pred_len, stride_hours, anchor_hour):
    total = 0
    for ts in entries:
        T = int(len(ts["target"]))
        t_min = context_len
        t_max = T - pred_len
        if t_max < t_min:
            continue
        step = max(1, int(stride_hours)) if anchor_hour is None else (int(stride_hours) if int(stride_hours) >= 24 else 24)
        if anchor_hour is None:
            first_t = t_min
        else:
            start_period = ts["start"]
            first_t = None
            for t in range(t_min, min(t_min + 24, t_max + 1)):
                try:
                    if (start_period + t).hour == int(anchor_hour):
                        first_t = t; break
                except Exception:
                    if (start_period.to_timestamp() + pd.Timedelta(hours=t)).hour == int(anchor_hour):
                        first_t = t; break
            if first_t is None:
                continue
        total += (t_max - first_t) // step + 1
    return int(total)


def _save_df(df: pd.DataFrame, base_path: Path, fmt: str, log) -> Path:
    fmt = (fmt or "parquet").lower()
    if fmt == "parquet":
        path = base_path.with_suffix(".parquet")
        try:
            df.to_parquet(path, index=False)
            log.info("Saved %s (%d rows)", path.name, len(df))
            return path
        except Exception as e:
            path = base_path.with_suffix(".csv")
            df.to_csv(path, index=False)
            log.info("Parquet failed (%s). Saved %s (%d rows)", str(e), path.name, len(df))
            return path
    # default to CSV
    path = base_path.with_suffix(".csv")
    df.to_csv(path, index=False)
    log.info("Saved %s (%d rows)", path.name, len(df))
    return path


def _normalize_feats_to_time_channel(
    feats: np.ndarray,
    target_len: int,
    num_covariates: int,
) -> np.ndarray:
    """
    Ensure feats has shape (T, C), where C == num_covariates.
    Accepts (T,C) or (C,T). Errors on anything else.
    """
    if int(num_covariates) == 0:
        # shape (T, 0) so downstream slicing still works
        return np.empty((int(target_len), 0), dtype=np.float32)
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if feats.ndim != 2:
        raise ValueError(f"feat_dynamic_real must be 2D, got {feats.shape}")

    r, c = feats.shape  # could be (T,C) or (C,T)

    if c == num_covariates:
        pass  # (T,C)
    elif r == num_covariates:
        feats = feats.T  # (C,T) -> (T,C)
    else:
        if c == num_covariates:
            pass
        elif r == num_covariates:
            feats = feats.T
        else:
            raise ValueError(
                f"feat_dynamic_real incompatible shape {feats.shape}; "
                f"expected (T,{num_covariates}) or ({num_covariates},T)"
            )

    if feats.shape[1] != num_covariates:
        raise ValueError(f"feat_dynamic_real has C={feats.shape[1]}, expected {num_covariates}")
    return feats


def _iter_sliding_windows(
    entries: List[Dict[str, Any]],
    context_len: int,
    pred_len: int,
    stride_hours: int,
    num_covariates: int,
    anchor_hour: Optional[int] = None,
    cu_indices: Optional[List[int]] = None,
    mask_value: float = np.nan,
    mask_past_tail_len: int = 0,
    apply_future_mask: bool = False,
):
    """
    Yield dicts with:
      - item_id
      - past_target: (context_len,)
      - past_feat_dynamic_real: (context_len, C)
      - future_feat_dynamic_real: (pred_len, C)
      - future_target: (pred_len,)
      - forecast_start: pandas.Period (hourly)

    Windows step by `stride_hours`. If `anchor_hour` is set and `stride_hours < 24`,
    we use step=24. Incomplete tails are pruned.
    """
    for ts in entries:
        target = np.asarray(ts["target"], dtype=np.float32)           # (T,)
        feats_raw = ts[FieldName.FEAT_DYNAMIC_REAL]                   # (T,C) or (C,T)
        feats = _normalize_feats_to_time_channel(feats_raw, len(target), num_covariates)

        T = int(min(len(target), feats.shape[0]))
        t_min = context_len
        t_max = T - pred_len
        if t_max < t_min:
            continue

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
            past_target = target[t - context_len : t]
            future_target = target[t : t + pred_len]
            past_feat = feats[t - context_len : t, :]
            fut_feat  = feats[t : t + pred_len, :]

            # --- tail mask in the *past* window (independent of future mask) ---
            if mask_past_tail_len > 0 and cu_indices and past_feat.shape[1] > 0:
                m = min(mask_past_tail_len, past_feat.shape[0])
                if m > 0:
                    past_feat = past_feat.copy()
                    past_feat[-m:, cu_indices] = mask_value

            # --- optional future mask across the whole horizon ---
            if apply_future_mask and cu_indices and fut_feat.shape[1] > 0:
                fut_feat = fut_feat.copy()
                fut_feat[:, cu_indices] = mask_value

            yield {
                "item_id": uid,
                "past_target": past_target,
                "past_feat_dynamic_real": past_feat,
                "future_feat_dynamic_real": fut_feat,
                "future_target": future_target,
                "forecast_start": ts["start"] + t,
            }
            t += step


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_test",
)
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.seed), workers=True)

    out_dir = Path(to_absolute_path(cfg.out_dir)).resolve()
    dirs = prepare_test_run_dirs(out_dir)

    init_logging(
        log_dir=str(dirs["py"]),
        run_id=str(cfg.tag),
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False,
    )
    log = get_logger("chronosx_test")
    log.info("Test dir: %s", out_dir)

    non_finetuned = bool(getattr(cfg.test, "non_finetuned", False))

    train_cfg = None
    features_order = None
    past_only_names = None
    context_len = None
    pred_len = None
    num_covariates = None
    pretrained_src = None
    mask_enabled = None
    mask_future = None
    mask_value = None
    K_tail = None


    #### TEST FINETUNED MODEL ####
    if not non_finetuned:

        train_run_dir = Path(to_absolute_path(cfg.test.source_run_dir)).resolve()
        if not train_run_dir.exists():
            raise FileNotFoundError(f"Training run dir not found: {train_run_dir}")

        # --- load train-time resolved config (single source of truth) ---
        train_cfg_path = train_run_dir / "meta" / "resolved_config.yaml"
        if not train_cfg_path.exists():
            raise FileNotFoundError(f"Cannot find {train_cfg_path}")
        train_cfg = OmegaConf.load(str(train_cfg_path))

        D = train_cfg.data
        M = train_cfg.model
        MASK = train_cfg.mask

        # features: prefer persisted order if present
        if "features_order" in train_cfg.features and "past_only_names" in train_cfg.features:
            features_order  = list(train_cfg.features.features_order)
            past_only_names = list(train_cfg.features.past_only_names)
        else:
            ck_cols = list(train_cfg.features.ck_cols)
            cu_cols = list(train_cfg.features.cu_cols)
            if not (ck_cols or cu_cols):
                raise ValueError("Train config must contain features.ck_cols / features.cu_cols.")
            features_order  = ck_cols + cu_cols
            past_only_names = cu_cols

        pred_len = int(train_cfg.data.prediction_length)
        context_len = int(train_cfg.data.context_length)
        num_covariates = len(features_order)

        finetuned_hf_dir = train_run_dir / "train" / "final-checkpoint"
        pt_state_path    = train_run_dir / "train" / "checkpoints" / "last_state_dict.pt"
        pretrained_src   = (str(train_cfg.model.pretrained_name_or_path) if pt_state_path.exists() 
                            else str(finetuned_hf_dir))


    else:
        # ---- ZERO-SHOT: use *current* cfg.*; no train run needed ----

        D = cfg.data
        M = cfg.model
        MASK = cfg.mask

        ck_cols = list(cfg.features.ck_cols)
        cu_cols = list(cfg.features.cu_cols)

        features_order   = ck_cols + cu_cols
        use_covs         = bool(getattr(cfg.features, "use_covariates", True))
        past_only_names  = cu_cols

        pred_len        = int(cfg.data.prediction_length)
        context_len     = int(cfg.data.context_length)
        num_covariates   = (len(features_order) if use_covs else 0)
        pretrained_src  = str(cfg.model.pretrained_name_or_path)


    mask_enabled = bool(MASK.enabled)
    mask_future  = bool(MASK.mask_future) and mask_enabled
    mask_value   = np.nan if MASK.value is None else float(MASK.value)
    K_tail       = int(MASK.len) if mask_enabled else 0

    inj_cfg = str(M.covariate_injection)
    inj_eff = "none" if num_covariates == 0 else inj_cfg

    log.info("num_covariates=%d, effective_injection=%s", num_covariates, inj_eff)
    pipeline = ChronosXPipelineMinimal(
        pretrained_model_name_or_path=pretrained_src,
        prediction_length=pred_len,
        num_covariates=num_covariates,
        covariate_injection=inj_eff,
        device_map=str(cfg.model.device_map),
        hidden_dim=int(M.hidden_dim),
        num_layers=int(M.num_layers),
        layers_to_unfreeze=str(M.layers_to_unfreeze),
    )


    if not non_finetuned:
        pt_state_path = train_run_dir / "train" / "checkpoints" / "last_state_dict.pt"

        if pt_state_path.exists():
            sd = torch.load(str(pt_state_path), map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            missing, unexpected = pipeline.chronosx.load_state_dict(sd, strict=False)
            log.info("Loaded PT weights (missing=%d, unexpected=%d)", len(missing), len(unexpected))


    pipeline.chronosx.eval()
    log.info(
        "Model device summary: first param device=%s",
        next(pipeline.chronosx.parameters()).device
    )


    train_path_cfg = getattr(cfg.data, "train_csv_path", None)
    train_path_use = (to_absolute_path(train_path_cfg) if _nonempty(train_path_cfg) else None)

    include_covs = (num_covariates > 0)
    # --- load test data with SAME features order as train ---
    _, test_gts = load_from_csv(
        train_csv_path=train_path_use,
        eval_csv_path=to_absolute_path(cfg.data.test_csv_path),
        id_col=D.id_col,
        date_col=D.date_col,
        target_col=D.target_col,
        feature_cols=features_order,  # strict CK->CU
        fillna_forward=bool(getattr(D, "fillna_forward", True)),
        include_covariates=include_covs,
    )
    if not test_gts:
        raise RuntimeError("No test data loaded.")

    # --- sliding windows: stride + anchor_hour + prune tail ---
    stride = int(getattr(cfg.data, "origin_stride_test", 24))
    anchor_hour_raw = getattr(cfg.data, "origin_anchor_hour_test", None)
    anchor_hour = int(anchor_hour_raw) if (anchor_hour_raw is not None and str(anchor_hour_raw) != "None") else None


    past_only_idx = [features_order.index(n) for n in past_only_names]

    window_iter = _iter_sliding_windows(
        entries=test_gts,
        context_len=context_len,
        pred_len=pred_len,
        stride_hours=stride,
        num_covariates=num_covariates,
        anchor_hour=anchor_hour,
        cu_indices=past_only_idx,
        mask_value=mask_value,
        mask_past_tail_len=K_tail,
        apply_future_mask=mask_future
    )

    # --- batched inference ---
    B = int(cfg.test.batch_size)
    S = int(cfg.test.num_samples)

    all_preds, all_fstarts, all_uids, all_truth = [], [], [], []
    n_windows = 0

    # progress knobs
    show_bar = bool(cfg.progress_bar.enabled)
    bar_desc = str(cfg.progress_bar.desc)
    bar_pos = int(cfg.progress_bar.position)

    try:
        total_windows = _estimate_total_windows(test_gts, context_len, pred_len, stride, anchor_hour)
        log.info("Estimated total windows: %d", total_windows)
    except Exception as e:
        log.warning("Could not estimate total windows: %s", e)
        total_windows = None

    pbar = _tqdm(total=total_windows, desc=bar_desc, unit="win", position=bar_pos, disable=not show_bar)

    all_past = []

    for batch in batcher(window_iter, batch_size=B):
        # build inputs
        contexts = [torch.from_numpy(np.asarray(e["past_target"], dtype=np.float32)) for e in batch]
        covs = [
            {
                "past_feat_dynamic_real": e["past_feat_dynamic_real"],
                "future_feat_dynamic_real": e["future_feat_dynamic_real"],
            }
            for e in batch
        ]

        out = pipeline.predict(contexts, covs, num_samples=S)
        out_np = out.detach().cpu().numpy() if torch.is_tensor(out) else np.asarray(out)
        all_preds.append(out_np)
        all_fstarts.extend([e["forecast_start"] for e in batch])
        all_uids.extend([e["item_id"] for e in batch])
        all_truth.extend([e["future_target"] for e in batch])
        all_past.extend([e["past_target"] for e in batch])
        n_windows += len(batch)
        pbar.update(len(batch))

    pbar.close()

    if all_preds:
        samples_BSH = np.concatenate(all_preds, axis=0)  # (B_total, S, H)
    else:
        samples_BSH = np.empty((0, S, pred_len), dtype=np.float32)

    log.info(
        "Predictions shape: %s | windows: %d | stride: %d | anchor_hour: %s",
        tuple(samples_BSH.shape), n_windows, stride, str(anchor_hour),
    )

    # --- Save raw samples under runs/<run_id>/test/<tag>/predictions ---
    pred_dir = dirs["pred"]
    np.savez_compressed(pred_dir / "forecast_samples.npz", samples=samples_BSH)

    # choose output format via Hydra: output.format = parquet|csv  (default parquet)
    fmt = str(getattr(cfg.output, "format", "parquet")).lower()

    # ---------------- Metrics (CRPS / ES / ECE / Berkowitz) ----------------
    raw_metrics = getattr(cfg.output, "metric", ["crps", "es"])
    if isinstance(raw_metrics, str):
        metrics_req = {raw_metrics.lower()}
    else:
        try:
            metrics_req = {str(m).lower() for m in list(raw_metrics)}
        except Exception:
            metrics_req = {"crps", "es"}  # safe default

    ALLOWED_METRICS = {"crps", "es", "ece", "berkowitz"}
    unknown = metrics_req - ALLOWED_METRICS
    if unknown:
        log.warning("Unknown metrics in output.metric: %s (allowed=%s)", sorted(unknown), sorted(ALLOWED_METRICS))
    metrics_req = metrics_req & ALLOWED_METRICS  # drop unknowns

    log.info("Requested metrics: %s", sorted(metrics_req))

    # Default outputs if no windows
    H = int(pred_len)
    S_eff = int(S)
    metrics = {
        "windows": int(samples_BSH.shape[0]),
        "S": int(S_eff),
        "H": int(H),
        "stride": int(stride),
        "anchor_hour": anchor_hour,
        "crps_mean_over_all_origins": float("nan"),
        "es_mean_over_all_origins": float("nan"),
        "ece_mean_over_horizons": float("nan"),
        "ece_taus": [],
        "ece_N_per_horizon": 0,
        "berkowitz_mode": "per_horizon",
        "berkowitz_LR_per_h": [float("nan")] * H,
        "berkowitz_df_per_h": [3] * H,
        "berkowitz_p_per_h":  [float("nan")] * H,
        "berkowitz_T_per_h":  [0] * H,
        "berkowitz_mu_per_h": [float("nan")] * H,
        "berkowitz_rho_per_h":[float("nan")] * H,
        "berkowitz_sigma2_per_h":[float("nan")] * H,
    }

    # Nothing to do if no windows produced
    if samples_BSH.shape[0] == 0:
        log.warning(
            "No evaluation windows produced -> all metrics NaN. "
            "Check context_length/prediction_length/stride/anchor_hour."
        )
    else:
        # ---- prepare base arrays/tensors ----
        Btot, S_eff, H = samples_BSH.shape
        y_true_BH = np.stack(all_truth, axis=0)  # (B,H)
        assert y_true_BH.shape == (Btot, H)

        device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        )
        t_samples = torch.tensor(samples_BSH.transpose(1, 0, 2), device=device, dtype=torch.float32)  # (S,B,H)
        t_y_true  = torch.tensor(y_true_BH, device=device, dtype=torch.float32)                         # (B,H)

        origins_iso = [str(ts) for ts in all_fstarts]

        log.info("Metric base: windows=%d, S=%d, H=%d, device=%s", Btot, S_eff, H, device)

        # ---- CRPS (1D per horizon, averaged over H) ----
        crps_per_origin = None
        if "crps" in metrics_req:
            fit_crps, spread_crps = crps_terms_fast(t_samples, t_y_true)               # (B,H)
            crps_per_origin = (fit_crps - spread_crps).mean(dim=1).detach().cpu().numpy()  # (B,)
            metrics["crps_mean_over_all_origins"] = float(np.mean(crps_per_origin))
            log.info("CRPS: mean=%.6f", metrics["crps_mean_over_all_origins"])

        # ---- ES (sliced energy score) ----
        es_per_origin = None
        if "es" in metrics_req:
            es_beta = float(getattr(cfg.metrics, "es_beta", 1.0))
            K = int(getattr(cfg.metrics, "es_slices", 128))
            pair_sub = getattr(cfg.metrics, "es_pair_subsample", None)
            pair_sub = None if pair_sub is None else int(pair_sub)

            # sliced_energy_score returns (es_mean, es_bk, fit_bk, spread_bk, w_used) when return_bk=True
            es_mean, es_bk, _, _, _ = sliced_energy_score(
                t_samples, t_y_true,
                beta=es_beta,
                K=K,
                pair_subsample=pair_sub,
                return_bk=True,
            )
            es_per_origin = es_bk.mean(dim=1).detach().cpu().numpy()  # (B,)
            metrics["es_mean_over_all_origins"] = float(np.mean(es_per_origin))

            log.info(
                "ES(sliced): beta=%.3g, K=%d, pair_sub=%s, mean=%.6f",
                es_beta, K, str(pair_sub), metrics["es_mean_over_all_origins"]
            )

        # ---- ECE (optional, computed from samples on CPU) ----
        if "ece" in metrics_req:
            ece_taus = list(getattr(cfg.metrics, "ece_taus", [0.1, 0.5, 0.9]))
            S_B_H = samples_BSH.transpose(1, 0, 2)  # (S,B,H) numpy
            ece_mean, ece_per_h, ece_N = ece_from_samples_per_horizon(
                samples=S_B_H,
                y_true=y_true_BH,
                taus=ece_taus,
            )
            metrics["ece_mean_over_horizons"] = float(ece_mean)
            metrics["ece_taus"] = list(ece_taus)
            metrics["ece_N_per_horizon"] = int(ece_N)

            # optional: also persist per-h ECE in metrics json (nice for debugging)
            # (convert None -> nan)
            metrics["ece_per_h"] = [float("nan") if v is None else float(v) for v in ece_per_h]

            log.info(
                "ECE: taus=%s, mean=%.6f (N_per_h≈%d)",
                ece_taus, metrics["ece_mean_over_horizons"], metrics["ece_N_per_horizon"]
            )

        # ---- Per-origin tables + saving ----
        # Note: output.metric controls what tables get saved.
        # Keep saving logic independent from JSON summary metrics.
        fmt = str(getattr(cfg.output, "format", "parquet")).lower()

        if crps_per_origin is not None:
            df_crps = (
                pd.DataFrame({"unique_id": all_uids, "origin_ds": origins_iso, "crps_mean": crps_per_origin.astype(np.float64)})
                .sort_values(["unique_id", "origin_ds"])
            )
            _save_df(df_crps, pred_dir / "crps_per_origin", fmt, log)

        if es_per_origin is not None:
            df_es = (
                pd.DataFrame({"unique_id": all_uids, "origin_ds": origins_iso, "es_mean": es_per_origin.astype(np.float64)})
                .sort_values(["unique_id", "origin_ds"])
            )
            _save_df(df_es, pred_dir / "es_per_origin", fmt, log)

        # ---- Berkowitz (per horizon only) ----
        if "berkowitz" in metrics_req:
            U = pits_from_samples(
                t_samples.detach().cpu().numpy(),   # (S,B,H)
                t_y_true.detach().cpu().numpy(),    # (B,H)
                randomized=True,
            )
            Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))  # (B,H)
            assert Z.ndim == 2 and Z.shape == (Btot, H)

            berk_list = []
            pvals = []
            for h in range(H):
                z_h = Z[:, h]
                z_h = z_h[np.isfinite(z_h)]
                res_h = berkowitz_lr_test(z_h)
                berk_list.append(res_h)
                pvals.append(res_h.get("p", np.nan))

            metrics.update({
                "berkowitz_mode": "per_horizon",
                "berkowitz_LR_per_h":      [float(d.get("LR", np.nan)) for d in berk_list],
                "berkowitz_df_per_h":      [int(d.get("df", 3))        for d in berk_list],
                "berkowitz_p_per_h":       [float(d.get("p", np.nan))  for d in berk_list],
                "berkowitz_T_per_h":       [int(d.get("T", 0))         for d in berk_list],
                "berkowitz_mu_per_h":      [float(d.get("mu", np.nan)) for d in berk_list],
                "berkowitz_rho_per_h":     [float(d.get("rho", np.nan)) for d in berk_list],
                "berkowitz_sigma2_per_h":  [float(d.get("sigma2", np.nan)) for d in berk_list],
            })

            # better logging: summarize min/median/max p-value + a few worst horizons
            p_arr = np.asarray(pvals, dtype=float)
            finite = np.isfinite(p_arr)
            if finite.any():
                p_fin = p_arr[finite]
                # horizons are 1-based in prints
                worst_idx = np.argsort(p_arr)[:min(5, H)]
                worst_txt = ", ".join([f"h{int(i)+1}:p={p_arr[i]:.3g}" for i in worst_idx if np.isfinite(p_arr[i])])
                log.info(
                    "Berkowitz (per-h): p[min/med/max]=[%.3g, %.3g, %.3g] | worst: %s",
                    float(np.min(p_fin)), float(np.median(p_fin)), float(np.max(p_fin)),
                    worst_txt if worst_txt else "n/a",
                )
            else:
                log.info("Berkowitz: all p-values are NaN (not enough data per horizon).")

        # ---------------- Persist metrics JSON (super-runner contract) ----------------
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "test_metrics.json"

        def _jsonable(x):
            # convert numpy / torch scalars to Python scalars for JSON
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if torch.is_tensor(x) and x.ndim == 0:
                return x.item()
            return x

        # ensure all leaf values are JSON-serializable
        def _convert(obj):
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return _jsonable(obj)

        payload = _convert(metrics)

        metrics_path.write_text(json.dumps(payload, indent=2))
        log.info("Saved %s", str(metrics_path))

    # --- final compact summary log ---
    log.info(
        "Metrics summary: windows=%d | CRPS=%.6f | ES=%.6f | ECE=%.6f | Berkowitz=%s",
        int(metrics["windows"]),
        float(metrics.get("crps_mean_over_all_origins", float("nan"))),
        float(metrics.get("es_mean_over_all_origins", float("nan"))),
        float(metrics.get("ece_mean_over_horizons", float("nan"))),
        ("on" if "berkowitz" in metrics_req else "off"),
    )




    # ---------------- Fan-chart plotting (optional) ----------------
    try:
        fan_cfg = getattr(cfg.test, "plotting", None)
        fan_cfg = getattr(fan_cfg, "fan_chart", None)
    except Exception:
        fan_cfg = None

    if fan_cfg and bool(fan_cfg.enable_fan_plotting) and (len(all_fstarts) > 0):
        log.info("[chronosx_test] Fan-chart plotting enabled.")

        def _as_ts_naive_floor_h(x):
            ts = (x.to_timestamp() if isinstance(x, pd.Period) else pd.to_datetime(x))
            # if tz-aware, drop tz; if naive, this is a no-op
            try:
                ts = ts.tz_localize(None)
            except Exception:
                pass
            return ts.floor("h")

        # resolve knobs
        line_type     = str(fan_cfg.type).lower()
        origin_list = [_as_ts_naive_floor_h(s) for s in list(fan_cfg.origin_ds)]
        quantile_bands = [tuple(map(float, band)) for band in fan_cfg.quantile_ranges]
        S_plot        = int(fan_cfg.n_samples_for_plotting)
        title_single  = str(fan_cfg.title_if_single_plot).strip()
        history_hours = int(fan_cfg.history_hours)
        model_name    = str(cfg.test.plotting.model_name)
        model_size    = str(cfg.test.plotting.size)

        # dir
        fan_dir = dirs["art"]
        fan_dir.mkdir(parents=True, exist_ok=True)

        # concatenate lists to arrays aligned with samples_BSH rows
        # samples_BSH shape is (B_total, S, H) — we’ll index "b" and then (S_plot, H)
        Btot = samples_BSH.shape[0]
        H    = samples_BSH.shape[2] if samples_BSH.ndim == 3 else int(pred_len)

        fstarts    = pd.Index([_as_ts_naive_floor_h(x) for x in all_fstarts])
        uids    = np.array(all_uids)
        # we stored past + truth in python lists; keep them indexable
        # all_past[b]: (context_len,), all_truth[b]: (H,)

        def _first_match(ts: pd.Timestamp) -> List[int]:
            return list(np.flatnonzero(fstarts == ts))

        n_written = 0
        for ts in origin_list:
            idxs = _first_match(ts)
            if not idxs:
                log.info("[chronosx_test] origin %s: no matching windows (maybe pruned by stride/anchor).", ts)
                continue

            for b in idxs:
                uid_b = int(uids[b])

                # S×H samples for this origin
                S_avail = samples_BSH.shape[1]
                if S_plot <= 0:
                    continue
                S_eff = min(S_plot, S_avail)
                samples_matrix = samples_BSH[b, :S_eff, :]             # (S_eff, H)

                # history + truth
                y_past_real  = np.asarray(all_past[b], dtype=np.float32)   # (C,)
                y_true_real  = np.asarray(all_truth[b], dtype=np.float32)  # (H,)
                C = len(y_past_real)

                # time axes (origin is first forecast step)
                origin_dt, past_ds, fut_ds = _reconstruct_time_axes(str(ts), context_length=C, horizon=len(y_true_real))

                # filename + title
                bands_txt = "_".join([str(int(round((hi - lo) * 100))) for (lo, hi) in quantile_bands])
                safe_model = model_name.replace(" ", "").lower()
                safe_size  = model_size.replace(" ", "").lower()
                fname = f"fan_{safe_model}_{safe_size}_uid{uid_b}_{pd.Timestamp(origin_dt).strftime('%Y-%m-%dT%H-%M')}_S{S_eff}_bands{bands_txt}_{line_type}.png"
                out_path = fan_dir / fname

                title = title_single if (len(origin_list) == 1 and title_single) else None

                if line_type == "line":
                    _plot_fanchart_lines(
                        past_ds=past_ds,
                        past_y=y_past_real,
                        origin=origin_dt,
                        fut_ds=fut_ds,
                        samples_np=samples_matrix,
                        y_true_future_np=y_true_real,
                        quantile_bands=quantile_bands,
                        show_mean=False,
                        show_median=True,
                        history=history_hours,
                        title=title,
                        model_name=model_name,
                        model_size=model_size,
                        out_path=out_path,
                    )
                else:  # "steps" default
                    _plot_fanchart_steps(
                        past_ds=past_ds,
                        past_y=y_past_real,
                        origin=origin_dt,
                        fut_ds=None,
                        samples_np=samples_matrix,
                        y_true_future_np=y_true_real,
                        quantile_bands=quantile_bands,
                        show_mean=False,
                        show_median=True,
                        history=history_hours,
                        title=title,
                        model_name=model_name,
                        model_size=model_size,
                        out_path=out_path,
                    )
                n_written += 1

        log.info("[chronosx_test] Fan-charts written: %d → %s", n_written, str(fan_dir))


    # ---------------- PIT histogram plotting (optional) ----------------
    if cfg.test.plotting.pit.enable_pit_plotting and (samples_BSH.shape[0] > 0):
        pit_dir = dirs["art"]
        pit_dir.mkdir(parents=True, exist_ok=True)

        model_name = str(cfg.test.plotting.model_name)
        model_size = str(cfg.test.plotting.size)
        bins       = int(cfg.test.plotting.pit.bins)

        # samples_BSH: (B_total, S, H)  -> pits_from_samples expects (S,B,H)
        S_B_H = samples_BSH.transpose(1, 0, 2)             # (S, B, H)
        Y_B_H = np.stack(all_truth, axis=0)                 # (B, H) already built above

        # Compute PITs across all kept windows
        U_B_H = pits_from_samples(S_B_H, Y_B_H, randomized=True)  # (B, H)
        assert U_B_H.shape[0] == S_B_H.shape[1], "B mismatch in PIT computation."

        H = U_B_H.shape[1]
        if isinstance(cfg.test.plotting.pit.horizons, str) and cfg.test.plotting.pit.horizons.lower() == "all":
            hs_plot = list(range(H))
        else:
            hs_plot = [int(h) for h in list(cfg.test.plotting.pit.horizons)]
            invalid = [h for h in hs_plot if h < 0 or h >= H]
            if invalid:
                raise IndexError(f"PIT horizons out of range 0..{H-1}: {invalid}")

        safe_model = model_name.replace(" ", "").lower().strip()
        safe_size  = model_size.replace(" ", "").lower().strip()

        for h in hs_plot:
            U_h = np.asarray(U_B_H[:, h], dtype=float)
            U_h = U_h[np.isfinite(U_h)]
            if U_h.size == 0:
                log.info("PIT[h=%d]: skipped (no data).", h)
                continue

            title = f"{model_name} ({model_size}) — PIT histogram — h={h+1}"
            fname = f"pit_{safe_model}_{safe_size}_h{h+1}_bins{bins}.png"
            out_path = pit_dir / fname

            _plot_pit_histogram(U_h, bins=bins, title=title, out_path=out_path)

        log.info("[chronosx_test] PIT histograms written → %s", str(pit_dir))

    log.info(f"[chronosx_test] Done. Test dir: {out_dir}")


if __name__ == "__main__":
    main()