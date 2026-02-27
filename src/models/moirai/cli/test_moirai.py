# src/models/moirai/runners/test_moirai.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- macOS/MPS: allow CPU fallback for ops not implemented on MPS (e.g., gamma in StudentT) ---
import torch
try:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
except Exception:
    pass

import json
from pathlib import Path
from typing import List
import traceback

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split


from uni2ts.model.moirai import MoiraiModule
from models.moirai.moirai_finetune_lagmask import MoiraiFinetuneLagMask
from models.moirai.moirai_forecast_lagmask import MoiraiForecastLagMask

from utils.metrics import crps_terms_fast, pits_from_samples, berkowitz_lr_test, sliced_energy_score, ece_from_samples_per_horizon
from utils.paths import prepare_test_run_dirs
from utils.logging_utils import init_logging, get_logger

from utils.plotting import (
    _reconstruct_time_axes,
    _plot_fanchart_steps,
    _plot_fanchart_lines,
    _plot_pit_histogram,
)
import matplotlib.pyplot as plt

from scipy.stats import chi2, norm 

# -----------------------------
#  I/O helpers
# -----------------------------

def _resolve_hub_id(base_hub_id: str) -> str:
    # allow local HF export dir (contains config.json + model.safetensors)
    p = Path(base_hub_id)
    return str(p.resolve()) if p.exists() else base_hub_id

def _ensure_dtindex_hourly(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("H")
    return df

def _as_timestamp(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, pd.Period):
        # start of the period; for hourly freq this is right
        return x.to_timestamp(how="start")
    # strings/np.datetime64/etc.
    try:
        return pd.Timestamp(x)
    except Exception:
        return None

def _prep_pandasdataset(df: pd.DataFrame,
                        id_col: str,
                        target_col: str,
                        feat_dynamic_real: List[str],
                        past_feat_dynamic_real: List[str]) -> PandasDataset:
    
    return PandasDataset.from_long_dataframe(
        df,
        target=target_col,
        item_id=id_col,
        feat_dynamic_real=feat_dynamic_real,
        past_feat_dynamic_real=past_feat_dynamic_real,
        freq="H"
    )

# -----------------------------
#  MAIN
# -----------------------------
@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_test",
)
def main(cfg: DictConfig):

    # --- Logging dirs ---
    out_dir = Path(to_absolute_path(cfg.out_dir)).resolve()
    dirs = prepare_test_run_dirs(out_dir)
    init_logging(
        log_dir=str(dirs["py"]),
        run_id=str(cfg.tag),
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False
    )
    log = get_logger("test_moirai")
    log.info("Writing to: %s", str(out_dir))

    # --- Seed ---
    import pytorch_lightning as pl
    pl.seed_everything(int(cfg.seed), workers=True)

    # --- Load data (train tail + test) for ground truth alignment ---
    test_path  = Path(to_absolute_path(cfg.data.test_csv_path))
    id_col     = str(cfg.data.id_col)
    date_col   = str(cfg.data.date_col)
    y_col      = str(cfg.data.target_col)
    fdr_cols   = list(getattr(cfg.data, "feat_dynamic_real", []) or [])
    pfdr_cols = list(getattr(cfg.data, "past_feat_dynamic_real", []) or [])

    log.info("Loading test:  %s", str(test_path))
    df_test_raw  = pd.read_csv(test_path)

    # Build combined test frame = tail(context) of train + test
    ctx   = int(cfg.windows.context_length)
    H     = int(cfg.windows.prediction_length)
    dist  = int(cfg.windows.distance)

    # enforce datetime index & hourly
    df_te = _ensure_dtindex_hourly(df_test_raw,  date_col)

    # ensure needed columns exist
    needed = [id_col, y_col] + fdr_cols + pfdr_cols
    for c in needed:
        if c not in df_te.columns:
            raise ValueError(f"Column '{c}' must be present in both train and test CSVs.")

    # fillna forward if requested
    if bool(cfg.data.fillna_forward):
        df_te[needed] = df_te[needed].ffill()

    # --- Build GluonTS dataset for rolling windows ---
    ds_test = _prep_pandasdataset(df_te, id_col, y_col, fdr_cols, pfdr_cols)

    # number of windows such that last forecast horizon fully fits
    total_len = len(df_te)
    windows = (total_len - ctx - H) // dist + 1
    assert windows > 0, "No evaluation windows; check context_length/prediction_length/distance vs data length."

    # GluonTS split/template
    _, template = split(ds_test, offset=ctx)
    test_data = template.generate_instances(
        prediction_length=H,
        windows=windows,
        distance=dist
    )

    # --- Build/Load Moirai ---
    base_hub_id = _resolve_hub_id(getattr(cfg.model, "base_hub_id", ""))
    num_samples = int(cfg.model.num_samples)
    batch_size  = int(cfg.model.batch_size)

    # infer feature dims from dataset object
    feat_dyn_dim = getattr(ds_test,"num_feat_dynamic_real",0)
    past_feat_dyn_dim = getattr(ds_test,"num_past_feat_dynamic_real",0)

    # ckpt_path is optional (zero_shot won't pass it)
    ckpt_path = None
    if "ckpt_path" in cfg.model:
        v = cfg.model.ckpt_path
        if v not in (None, "", "null"):
            ckpt_path = to_absolute_path(str(v))

    try:
        if ckpt_path:
            log.info("Loading finetuned Lightning ckpt: %s", ckpt_path)
            log.info("Using base architecture from base_hub_id (HF/local): %s", base_hub_id)
            base_module = MoiraiModule.from_pretrained(base_hub_id)
            lm = MoiraiFinetuneLagMask.load_from_checkpoint(
                ckpt_path, map_location="cpu", module=base_module
            )
            module = lm.module
        else:
            log.info("Zero-shot: loading module from base_hub_id (HF/local): %s", base_hub_id)
            module = MoiraiModule.from_pretrained(base_hub_id)
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            "Failed to construct model.\n"
            f"base_hub_id: {base_hub_id}\n"
            f"ckpt_path: {ckpt_path}\n"
            f"Cause: {type(e).__name__}: {e}\n"
            f"Traceback:\n{tb}\n"
            "Ensure base_hub_id is a valid HF repo or a local HF export directory with "
            "config.json (+ weights), and that it matches the ckpt architecture if ckpt_path is set."
        ) from e

    model = MoiraiForecastLagMask(
        module=module,
        prediction_length=H,
        context_length=ctx,
        patch_size=32,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=feat_dyn_dim,
        past_feat_dynamic_real_dim=past_feat_dyn_dim,
        lag_mask_steps= int(cfg.model.lag_mask_steps),
        lag_mask_value= float(cfg.model.lag_mask_value)
    )
    predictor = model.create_predictor(
        batch_size=batch_size)

    # --- Forecast ---
    log.info("Forecasting: windows=%d, H=%d, S=%d, batch_size=%d", windows, H, num_samples, batch_size)
    fcst_iter = predictor.predict(test_data.input)
    forecasts = list(fcst_iter)
    assert len(forecasts) == windows, f"Expected {windows} forecasts but got {len(forecasts)}."

    # --- Build per-forecast origins & uids, stack samples (S,B,H) ---
    origins, uids = [], []
    for i, f in enumerate(forecasts):
        origin_raw = getattr(f, "start_date", None)
        origin = _as_timestamp(origin_raw)
        if origin is None:
            # robust fallback from base index
            origin = df_te.index[ctx + i * dist]
        origins.append(origin)

        uid = getattr(f, "item_id", None)
        if uid is None:
            vals = df_te[id_col].dropna().unique()
            uid = vals[0] if len(vals) else 1
        uids.append(uid)

    S = forecasts[0].samples.shape[0]
    samples = np.stack([f.samples for f in forecasts], axis=1)  # (S, B, H)

    # --- y_true aligned per (uid, origin) ---
    # build fast lookup per uid (date-indexed Series of y)
    series_by_uid = {
        uid: df_te.loc[df_te[id_col] == uid, y_col]
        for uid in pd.unique(df_te[id_col])
    }
    y_true_mat = []
    for origin, uid in zip(origins, uids):
        idx = pd.date_range(origin, periods=H, freq="H")
        y_seg = series_by_uid[uid].reindex(idx).values.astype(np.float64)
        y_true_mat.append(y_seg)
    y_true = np.stack(y_true_mat, axis=0)  # (B, H)

    # --- CRPS / ES(β=1) per origin_ds ---
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    t_samples = torch.tensor(samples, device=device, dtype=torch.float32)   # (S,B,H)
    t_y_true  = torch.tensor(y_true,  device=device, dtype=torch.float32)  # (B,H)

    fit, spread = crps_terms_fast(t_samples, t_y_true)          # (B,H)
    crps_per_origin = (fit - spread).mean(dim=1).detach().cpu().numpy()  # (B,)

    es_mean, es_bk, fit_bk, spread_bk, w_used = sliced_energy_score(
        t_samples, t_y_true,
        beta=1.0,
        use_fast_for_beta1=True,
        return_bk=True,
        )

    # per-origin ES: average over slices
    es_per_origin = es_bk.mean(dim=1).detach().cpu().numpy()  # (B,)

    # --- Berkowitz (log only, not saved as rows) ---
    U = pits_from_samples(samples, y_true, randomized=True)
    Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
    for h in range(Z.shape[1]):
        res_h = berkowitz_lr_test(Z[:, h])
        log.info(f"berkowitz_p_h{h:02d} = {res_h['p']:.6f}")
    res_pool = berkowitz_lr_test(Z.reshape(-1))
    log.info(f"berkowitz_p_pooled = {res_pool['p']:.6f}")

    # --- Build the (unique_id, origin_ds, score) table ---

    raw = cfg.output.metric
    if isinstance(raw, str):
        metrics_req = [raw.lower()]
    elif isinstance(raw, (list, ListConfig)):
        metrics_req = [str(m).lower() for m in raw]
    else:
        raise ValueError("output.metric must be a string or list of strings")

    allowed = {"crps", "es", "ece"}
    unknown = set(metrics_req) - allowed
    if unknown:
        raise ValueError(f"Unknown metrics in output.metric: {sorted(unknown)}")

    if "crps" in metrics_req:
        scores_df = pd.DataFrame({
            "unique_id": uids,
            "origin_ds": [ts.isoformat() for ts in origins],
            "crps_mean": crps_per_origin,
        }).sort_values(["unique_id", "origin_ds"])
        scores_path = dirs["pred"] / f"crps_per_origin.{cfg.output.format}"
        (scores_df.to_parquet if str(cfg.output.format).lower()=="parquet" else scores_df.to_csv)(scores_path, index=False)
        log.info("Saved per-origin CRPS to %s (rows=%d)", scores_path.name, len(scores_df))

    if "es" in metrics_req:
        scores_df = pd.DataFrame({
            "unique_id": uids,
            "origin_ds": [ts.isoformat() for ts in origins],
            "es_mean": es_per_origin,
        }).sort_values(["unique_id", "origin_ds"])
        scores_path = dirs["pred"] / f"es_per_origin.{cfg.output.format}"
        (scores_df.to_parquet if str(cfg.output.format).lower()=="parquet" else scores_df.to_csv)(scores_path, index=False)
        log.info("Saved per-origin ES to %s (rows=%d)", scores_path.name, len(scores_df))

    if "ece" in metrics_req:
        taus = list(getattr(cfg.output, "ece_taus", [10,20,30,40,50,60,70,80,90]))
        ece_mean, ece_per_h, N_used = ece_from_samples_per_horizon(samples, y_true, taus)

        ece_df = pd.DataFrame({
            "horizon": np.arange(1, H + 1, dtype=int),
            "ece": ece_per_h,              # may contain None
        })
        ece_df["N_used"] = N_used

        ece_path = dirs["pred"] / f"ece_per_h.{cfg.output.format}"
        (ece_df.to_parquet if str(cfg.output.format).lower()=="parquet" else ece_df.to_csv)(ece_path, index=False)
        log.info("Saved per-horizon ECE to %s (rows=%d)", ece_path.name, len(ece_df))

    



    # --- Also write summary metrics (optional) ---
    metrics = {
        "crps_mean_over_all_origins": float(crps_per_origin.mean()),
        "es_mean_over_all_origins": float(es_per_origin.mean()),
        "berkowitz_pooled": float(res_pool["p"]),
        "windows": int(len(origins)),
        "S": int(S),
        "H": int(H),
    }
    (dirs["metrics"] / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    if "ece" in metrics_req:
        metrics["ece_mean_over_horizons"] = float(ece_mean)
        metrics["ece_taus"] = [float(x) for x in taus]
        metrics["ece_N_used"] = int(N_used)

    # meta snapshot
    meta = {
        "base_hub_id": base_hub_id,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "context_length": ctx,
        "prediction_length": H,
        "distance": dist,
        "test_csv_path": str(test_path),
        "saved_metric_col": metrics_req,
    }
    (dirs["meta"] / "config_resolved.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    (dirs["meta"] / "test_meta.json").write_text(json.dumps(meta, indent=2))

    if "crps" in metrics_req:
        log.info("Done. mean crps = %.6f", float(crps_per_origin.mean()))
    if "es" in metrics_req:
        log.info("Done. mean es   = %.6f", float(es_per_origin.mean()))
    if "ece" in metrics_req:
        log.info("Done. mean ece   = %.6f", float(ece_mean))

    log.info("Berkowitz pooled p= %.6f", metrics["berkowitz_pooled"])


        # -------------------- Fan-chart plotting (sampling-only) --------------------
    try:
        fan_cfg = getattr(cfg, "plotting", None)
        fan_cfg = getattr(fan_cfg, "fan_chart", None)
    except Exception:
        fan_cfg = None

    if fan_cfg and bool(fan_cfg.enable_fan_plotting):
        log.info("Moirai[test]: fan-chart plotting enabled (sampling path).")

        # ---- knobs from YAML ----
        origin_list     = [str(x) for x in fan_cfg.origin_ds]
        line_type       = str(fan_cfg.type).lower()  # "steps" | "line"
        S_plot          = int(fan_cfg.n_samples_for_plotting)
        one_title       = str(fan_cfg.title_if_single_plot).strip()
        history_hours   = int(fan_cfg.history_hours)

        # Quantile bands
        # default to [[0.05,0.95],[0.25,0.75]] if not provided
        qb = getattr(fan_cfg, "quantile_ranges", None)
        if qb is None:
            quantile_bands = [(0.05, 0.95), (0.25, 0.75)]
        else:
            quantile_bands = [tuple(map(float, pair)) for pair in qb]

        # Title footer fields (optional cosmetics)
        model_name = str(cfg.plotting.model_name).strip()
        model_size = str(cfg.plotting.size).strip()

        # Where to put files (reuse artifacts dir created by prepare_test_run_dirs)
        fan_dir = dirs.get("art")
        fan_dir.mkdir(parents=True, exist_ok=True)

        # --- Build an index: (uid, origin_ts) -> column b in our stacked samples ---
        # (uids is a list aligned with 'origins' and the 2nd axis in 'samples')
        col_index = {(str(uids[b]), pd.Timestamp(origins[b]).tz_localize(None).floor("h")): b
                     for b in range(len(origins))}

        # --- Helper to extract history for a requested uid+origin from df_te ---
        def _history_and_truth(uid, origin_ts, ctx_len, H):
            # past
            past_idx = pd.date_range(origin_ts - pd.Timedelta(hours=ctx_len), periods=ctx_len, freq="h")
            past_ser = df_te.loc[df_te[id_col] == uid, y_col].reindex(past_idx).values.astype(np.float64)
            # future ground truth (already computed earlier in y_true, but fast to recalc here)
            fut_idx = pd.date_range(origin_ts, periods=H, freq="h")
            fut_ser = df_te.loc[df_te[id_col] == uid, y_col].reindex(fut_idx).values.astype(np.float64)
            return past_ser, fut_ser, past_idx, fut_idx

        n_done_total = 0
        for origin_str in origin_list:
            target_origin = pd.to_datetime(origin_str).tz_localize(None).floor("h")
            n_written = 0

            # iterate all possible uids we actually forecasted for that origin
            # (lookups via col_index)
            for uid in pd.unique(df_te[id_col]):
                key = (str(uid), target_origin)
                if key not in col_index:
                    continue
                b = col_index[key]  # the window/column index for (uid, origin)

                # (S, B, H) -> (S, H) for this origin
                S_all, _, H = samples.shape
                if S_plot <= 0 or S_plot > S_all:
                    S_eff = S_all
                else:
                    S_eff = S_plot
                samples_b = samples[:S_eff, b, :]  # (S_eff, H)

                # history & axes
                y_past, y_true_b, past_ds, fut_ds = _history_and_truth(uid, target_origin, ctx, H)
                origin_dt, past_ds_recon, fut_ds_recon = _reconstruct_time_axes(
                    str(target_origin), context_length=ctx, horizon=H
                )
                # _reconstruct_time_axes gives consistent axis objects; we’ll prefer those
                # but keep the ones we built from df_te for robustness.
                # For plotting, we’ll use the reconstructed axes.

                # Filename
                bands_txt = "_".join([str(int(round((hi - lo) * 100))) for (lo, hi) in quantile_bands])
                safe_model = model_name.replace(" ", "").lower()
                safe_size  = model_size.replace(" ", "").lower()
                fname = (
                    f"fan_{safe_model}_{safe_size}_"
                    f"uid{uid}_"
                    f"{pd.Timestamp(origin_dt).strftime('%Y-%m-%dT%H-%M')}_"
                    f"S{S_eff}_bands{bands_txt}_{line_type}.png"
                )
                out_path = fan_dir / fname

                # Title (custom if single target)
                title = one_title if (len(origin_list) == 1 and one_title) else None

                if line_type == "line":
                    _plot_fanchart_lines(
                        past_ds=past_ds_recon,
                        past_y=y_past,
                        origin=origin_dt,
                        fut_ds=fut_ds_recon,
                        samples_np=samples_b,
                        y_true_future_np=y_true_b,
                        quantile_bands=quantile_bands,
                        show_mean=False,
                        show_median=True,
                        history=history_hours,
                        title=title,
                        model_name=model_name,
                        model_size=model_size,
                        out_path=out_path,
                    )
                elif line_type=="steps":
                    _plot_fanchart_steps(
                        past_ds=past_ds_recon,
                        past_y=y_past,
                        origin=origin_dt,
                        fut_ds=None,                     # stepped plotter ignores fut_ds
                        samples_np=samples_b,            # (S_eff, H)
                        y_true_future_np=y_true_b,       # (H,)
                        quantile_bands=quantile_bands,
                        show_mean=False,
                        show_median=True,
                        history=history_hours,
                        title=title,
                        model_name=model_name,
                        model_size=model_size,
                        out_path=out_path,
                    )
                else:
                    raise ValueError(f"Type {str(cfg.test.plotting.fan_chart.type)} does not exist. Use either 'line' or 'step'.")
                n_written += 1
                n_done_total += 1

            log.info("Moirai[test] fan-plotting: origin=%s  plots_written=%d  → %s",
                     str(target_origin), n_written, str(fan_dir))

        log.info("Moirai[test] fan-plotting finished. Total plots: %d", n_done_total)


    # -------------------- PIT histogram plotting --------------------
    try:
        pit_cfg = cfg.plotting.pit
    except Exception:
        pit_cfg = None

    if pit_cfg and bool(pit_cfg.enable_pit_plotting):
        log.info("Moirai[test]: PIT histogram plotting enabled.")

        model_name = str(cfg.plotting.model_name).strip()
        model_size = str(cfg.plotting.size).strip()
        bins       = int(pit_cfg.bins)

        # `samples` is (S, B, H) and `y_true` is (B, H) above
        U_BH = pits_from_samples(samples, y_true, randomized=True)  # (B, H)

        H_total = U_BH.shape[1]
        if isinstance(pit_cfg.horizons, str) and pit_cfg.horizons.lower() == "all":
            hs_plot = list(range(H_total))
        else:
            hs_plot = [int(h) for h in list(pit_cfg.horizons)]
            invalid = [h for h in hs_plot if h < 0 or h >= H_total]
            if invalid:
                raise IndexError(f"PIT horizons out of range 0..{H_total-1}: {invalid}")

        pit_dir = dirs["art"]
        pit_dir.mkdir(parents=True, exist_ok=True)

        safe_model = model_name.replace(" ", "").lower()
        safe_size  = model_size.replace(" ", "").lower()

        for h in hs_plot:
            U_h = np.asarray(U_BH[:, h], dtype=float)
            U_h = U_h[np.isfinite(U_h)]
            if U_h.size == 0:
                log.info("Moirai[test] PIT[h=%d]: skipped (no data).", h)
                continue

            title = f"{model_name} ({model_size}) — PIT histogram — h={h+1}"
            fname = f"pit_{safe_model}_{safe_size}_h{h+1}_bins{bins}.png"
            out_path = pit_dir / fname

            _plot_pit_histogram(U_h, bins=bins, title=title, out_path=out_path)

        log.info("Moirai[test]: PIT histograms written → %s", str(pit_dir))


if __name__ == "__main__":
    main()