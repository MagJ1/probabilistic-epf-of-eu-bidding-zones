# src/models/normalizing_flows/runners/test_flows.py
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import joblib
import torch
import sys

from utils.paths import prepare_test_run_dirs, hydra_path_helper
from utils.logging_utils import init_logging, get_logger
from utils.device import pick_accelerator_and_devices
from utils.helpers import _find_best_ckpt
from utils.plotting import (
    _make_inverse_from_sklearn_scaler, 
    _get_test_dataset, 
    _plot_fanchart_lines, 
    _reconstruct_time_axes, 
    _plot_fanchart_steps, 
    _plot_pit_histogram
)

from utils.metrics import pits_from_samples

from models.normalizing_flows.FlowForecaster import FlowForecaster
from models.normalizing_flows.datamodule import FlowForecasterDataModule
from models.normalizing_flows.module import FlowForecasterModule

import matplotlib.pyplot as plt
from datetime import timedelta



@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_test",
)
def main(cfg: DictConfig):

    # seed for reproducibility of sampling metrics, dataloader order, etc.
    pl.seed_everything(int(cfg.seed), workers=True)

    # --- Where is the TRAIN run we want to evaluate? ---

    train_run_dir = Path(to_absolute_path(cfg.test.source_run_dir)).resolve()
    if not train_run_dir.exists():
        raise FileNotFoundError(f"Training run dir not found: {train_run_dir}")
    
    test_run_dir = Path(to_absolute_path(cfg.out_dir)).resolve()
    test_run_dir.mkdir(parents=True, exist_ok=True)
    dirs = prepare_test_run_dirs(test_run_dir)

    train_run_id = str(cfg.test.source_run_id)

    experiment_root = train_run_dir.parents[1]
    tb_root = experiment_root / "logs_tb"

    # --- Python logging ---
    init_logging(
        log_dir=str(dirs["py"]),
        run_id=str(cfg.tag),  # or run_id
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True, 
        unify_format=False          
    )
    
    log = get_logger("test_flows")
    log.info("Testing from train run: %s", str(cfg.test.source_run_id))

    # --- Load the train run’s *resolved* config for exact shapes/paths ---
    train_meta_dir = train_run_dir / "meta"
    train_scaler_dir = train_run_dir / "data"
    train_ckpt_dir = train_run_dir / "train" / "checkpoints"

    if not train_ckpt_dir:
        raise FileNotFoundError(f"No checkpoints found under {train_ckpt_dir}. Refusing to test.")

    with open(train_meta_dir / "params.json", "r") as f:
        train_cfg = OmegaConf.create(json.load(f))  # DictConfig-like object

    # Resolve data paths relative to current working dir (Hydra-safe)
    train_path = Path(to_absolute_path(cfg.data.train_csv_path))
    test_path  = Path(to_absolute_path(cfg.data.test_csv_path))

    # --- Lightning logger (TensorBoard) for this test run ---
    tb_logger = TensorBoardLogger(
        save_dir=str(tb_root),
        name="",  
        version=f"{train_run_id}-{cfg.tag}",
    )

    log.info(f"NUM_WORKERS: {int(cfg.test.num_workers)}")
    log.info(f"ENABLE_PROGRESS_BAR: {bool(cfg.test.enable_progress_bar)}")

    # --- DataModule built from *train* config (so shapes match the checkpoint) ---
    dm = FlowForecasterDataModule(
        train_csv_path=train_path,
        test_csv_path=test_path,
        context_length=train_cfg.data.context_length,
        forecast_horizon=train_cfg.data.forecast_horizon,
        batch_size=cfg.test.batch_size,                   # test-time batch size may differ
        num_workers=getattr(cfg.test, "num_workers", 4),
        val_split_date=cfg.data.val_split_date,
        date_col=train_cfg.data.date_col,
        id_col=train_cfg.data.id_col,
        y_col=train_cfg.data.y_col,
        ck_cols=train_cfg.features.ck_cols,
        cu_cols=train_cfg.features.cu_cols,
        past_unknown_cov_cutoff=train_cfg.model.enc_unknown_cutoff,
        scale_data=train_cfg.data.scale_data,
        realistic_mode=train_cfg.model.realistic_mode,
        origin_stride_test=cfg.data.origin_stride_test,
        origin_anchor_hour_test=cfg.data.origin_anchor_hour_test,
    )
    dm.setup(stage="test")

    # --- Load scalers saved during training ---
    scalers = joblib.load(train_scaler_dir / "scalers.pkl")

    # --- Recreate model with the *train* hyperparameters (matches checkpoint) ---
    model = FlowForecaster(
        tf_in_size=train_cfg.model.tf_in_size,
        nf_hidden_dim=train_cfg.model.nf_hidden_dim,
        n_layers=train_cfg.model.n_layers,
        n_heads=train_cfg.model.n_heads,
        n_flow_layers=train_cfg.model.n_flow_layers,
        n_made_blocks=train_cfg.model.n_made_blocks,
        tf_dropout=train_cfg.model.tf_dropout,
        c_future_known=len(train_cfg.features.ck_cols),
        c_future_unknown=len(train_cfg.features.cu_cols),
        context_length=train_cfg.data.context_length,
        forecast_horizon=train_cfg.data.forecast_horizon,
        enc_unknown_cutoff=train_cfg.model.enc_unknown_cutoff,
        dec_known_past_injection_horizon=train_cfg.model.dec_known_past_injection_horizon,
        realistic_mode=train_cfg.model.realistic_mode,
    )

    # --- Pick checkpoint ---
    ckpt_path = _find_best_ckpt(train_ckpt_dir)
    log.info("Using checkpoint: %s", ckpt_path.name)

    # --- Build the LightningModule and load weights (CPU first; PL will move it) ---
    module = FlowForecasterModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model=model,
        scalers=scalers,
        lr=train_cfg.train.lr,
        warmup_epochs=train_cfg.train.warmup_epochs,
        loss_metric=train_cfg.train.loss_metric,
        n_sample_loss=cfg.test.n_samples_eval,
        beta=train_cfg.train.beta,
        eval_metrics=train_cfg.train.eval_metrics,
        n_samples_eval=cfg.test.n_samples_eval,
        y_col=train_cfg.data.y_col,
        map_location="cpu",  # safe default; Trainer moves it to the right device, 
        detail_crps=str(cfg.test.detail.crps),
        detail_es=str(cfg.test.detail.es),
        store_series=bool(cfg.test.detail.store_series),
        ece_taus=cfg.test.detail.ece_taus
    )
    
    # --- Trainer (device helper) ---
    accel, devices, precision, _ = pick_accelerator_and_devices()
    # accel="cpu"
    log.info("Accelerator=%s devices=%s precision=%s", accel, devices, precision)
    trainer = Trainer(
        accelerator=accel,
        devices=devices,
        precision=precision,
        logger=tb_logger,
        default_root_dir=str(dirs["root"]),
        log_every_n_steps=getattr(cfg.test, "log_every_n_steps", 50),
        enable_progress_bar=bool(cfg.test.enable_progress_bar)
    )

    # --- Run test ---
    trainer.test(module, datamodule=dm)
    log.info("callback_metrics keys at end: %s", list(trainer.callback_metrics.keys()))

    module.export_predictions(dirs["pred"], write="parquet")

    # --- Save test metrics ---
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    (dirs["metrics"] / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- Save ECE per horizon (if computed) ---
    ece_detail = getattr(module, "last_test_ece_detail", None)
    if isinstance(ece_detail, dict):
        (dirs["metrics"] / "ece_detail.json").write_text(json.dumps(ece_detail, indent=2))

    # snapshot environment (nice for HPC records)
    env = {
        "pytorch": torch.__version__,
        "pl": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "train_run_dir": str(train_run_dir),
        "ckpt_used": ckpt_path.name,
    }
    (dirs["meta"] / "test_env.json").write_text(json.dumps(env, indent=2))
    (dirs["meta"] / "run_pointer.txt").write_text(str(Path(__file__).resolve()))
    (test_run_dir / "tested_from_run_dir.txt").write_text(str(train_run_dir))
    (train_run_dir / f"_done.test.{str(cfg.tag)}").write_text("ok")

    log.info("Finished test for %s", train_run_dir.name)


        # --- Fan-chart plotting (optional, controlled by Hydra knobs) ---
    # --- Fan-chart plotting (optional, controlled by Hydra knobs) ---
    if bool(cfg.test.plotting.fan_chart.enable_fan_plotting):
        # Where to put the files
        fan_dir = dirs["art"]  # .../test/artifacts
        fan_dir.mkdir(parents=True, exist_ok=True)

        # Read knobs (from test.plotting.*)
        origin_list = list(cfg.test.plotting.fan_chart.origin_ds)
        quantile_bands = [tuple(map(float, band)) for band in cfg.test.plotting.fan_chart.quantile_ranges]
        S = int(cfg.test.plotting.fan_chart.n_samples_for_plotting)
        one_title = str(cfg.test.plotting.fan_chart.title_if_single_plot).strip()
        line_type = str(cfg.test.plotting.fan_chart.type).lower()

        # inverse transform for target
        inv = None
        try:
            sc = module.scalers[module.y_col]
            inv = _make_inverse_from_sklearn_scaler(sc)
        except Exception:
            inv = None  # will plot scaled units if scaler missing

        # shape-preserving wrapper around inv()
        def _inv_like(x_np):
            y = inv(x_np) if inv is not None else x_np
            y = np.asarray(y)
            return y.reshape(np.asarray(x_np).shape)

        # fetch test dataset and scan for matches
        ds_test = _get_test_dataset(dm)
        if not origin_list:
            log.warning("Fan plotting enabled but test.plotting.origin_ds is empty; skipping.")
        else:
            log.info("Fan plotting for %d origin(s): %s", len(origin_list), origin_list)

            # Loop over requested origins; for each, iterate the test dataset once and plot all matching series
            for i, origin_str in enumerate(origin_list):
                n_plotted = 0
                target_origin = pd.to_datetime(origin_str).tz_localize(None).floor("h")

                for idx in range(len(ds_test)):
                    sample = ds_test[idx]  # dict with "ds","unique_id","y_past","y_future",...
                    sample_origin = pd.to_datetime(sample["ds"]).tz_localize(None).floor("h")
                    if sample_origin != target_origin:
                        continue

                    uid = int(sample["unique_id"])
                    device = module.device

                    # tensors: add batch dim and move
                    batch1 = {
                        "ds":         torch.tensor([0], device=device),  # placeholder; model doesn't use string ds
                        "unique_id":  torch.tensor([uid], device=device),
                        "y_past":     sample["y_past"].unsqueeze(0).to(device=device, dtype=torch.float32),                 # (1, CL)
                        "c_ctx_future_unknown": sample["c_ctx_future_unknown"].unsqueeze(0).to(device=device, dtype=torch.float32),
                        "c_ctx_future_known":   sample["c_ctx_future_known"].unsqueeze(0).to(device=device, dtype=torch.float32),
                        "c_fct_future_known":   sample["c_fct_future_known"].unsqueeze(0).to(device=device, dtype=torch.float32),
                        "y_future":   sample["y_future"].unsqueeze(0).to(device=device, dtype=torch.float32),               # (1, H)
                    }

                    # Draw scenarios (S, 1, H) -> (S, H)
                    with torch.no_grad():
                        samples_t, _ = module.model.sample(batch1, n_per_series=S, track_grad=False)
                    samples_np = samples_t[:, 0, :].detach().cpu().numpy()  # (S, H)

                    # Past/Future arrays
                    y_past = sample["y_past"].detach().cpu().numpy()            # (CL,)
                    y_true_future = sample["y_future"].detach().cpu().numpy()    # (H,)
                    H = y_true_future.shape[0]
                    CL = y_past.shape[0]

                    # inverse transform (preserve shapes)
                    y_past        = _inv_like(y_past)
                    y_true_future = _inv_like(y_true_future)
                    samples_np    = _inv_like(samples_np)

                    # time axes (uses origin at i=0 .. i=H-1)
                    origin, past_ds, fut_ds = _reconstruct_time_axes(str(target_origin), context_length=CL, horizon=H)

                    # Title and filename
                    bands_txt = "_".join([str(round((hi-lo)*100)) for (lo,hi) in quantile_bands])

                    plot_model_name = str(cfg.test.plotting.model_name).replace(" ", "")
                    plot_model_size = str(cfg.test.plotting.size).replace(" ", "")
                    fname = (
                        f"fan_{plot_model_name.lower().strip()}_{plot_model_size.lower().strip()}_"
                        f"uid{uid}_"
                        f"{pd.Timestamp(origin).strftime('%Y-%m-%dT%H-%M')}_"
                        f"S{S}_bands_{bands_txt}_{line_type}.png"
                    )
                    out_path = fan_dir / fname

                    title = one_title if (len(origin_list) == 1) else None

                    if (line_type=="line"):
                        _plot_fanchart_lines(
                            past_ds=past_ds,
                            past_y=y_past,
                            origin=origin,
                            fut_ds=fut_ds,
                            samples_np=samples_np,
                            y_true_future_np=y_true_future,
                            quantile_bands=quantile_bands,
                            show_mean=False,
                            show_median=True,
                            history=168,
                            title=title,
                            model_name=plot_model_name,
                            model_size=plot_model_size,
                            out_path=out_path,
                        )
                    elif (line_type=="steps"):
                        _plot_fanchart_steps(
                            past_ds=past_ds,
                            past_y=y_past,
                            origin=origin,
                            fut_ds=fut_ds,
                            samples_np=samples_np,
                            y_true_future_np=y_true_future,
                            quantile_bands=quantile_bands,
                            show_mean=False,
                            show_median=True,
                            history=168,
                            title=title,
                            model_name=plot_model_name,
                            model_size=plot_model_size,
                            out_path=out_path,
                        )

                    else:
                        raise ValueError(f"Type {str(cfg.test.plotting.fan_chart.type)} does not exist. Use either 'line' or 'step'.")


                    n_plotted += 1

                log.info("Fan plotting: origin=%s  plots_written=%d  → %s", str(target_origin), n_plotted, str(fan_dir))


    # --- PIT histogram plotting (optional, controlled by Hydra knobs) ---
    if bool(cfg.test.plotting.pit.enable_pit_plotting):
        pit_dir = dirs["art"]
        pit_dir.mkdir(parents=True, exist_ok=True)

        horizons_cfg = cfg.test.plotting.pit.horizons  # "all" or list[int]
        bins = int(cfg.test.plotting.pit.bins)

        model_name = str(cfg.test.plotting.model_name)
        model_size = str(cfg.test.plotting.size)

        # How many samples to use for PIT computation:
        S_pit = int(cfg.test.n_samples_eval)

        # Access the test dataset and model device
        ds_test = _get_test_dataset(dm)
        device = module.device

        # Forecast horizon from train config (consistent with checkpoint)
        H = int(train_cfg.data.forecast_horizon)

        # Decide which horizons to plot
        hs_all = list(range(H))
        if isinstance(horizons_cfg, str) and horizons_cfg.lower() == "all":
            hs_plot = hs_all
        else:
            hs_plot = [int(h) for h in list(horizons_cfg)]
            invalid = [h for h in hs_plot if h not in hs_all]
            if invalid:
                raise IndexError(f"PIT horizons out of range 0..{H-1}: {invalid}")

        # Accumulate PITs per horizon without storing giant tensors
        # We'll iterate the dataset, sample S scenarios per series, compute PITs for that series, and append.
        U_cols = [[] for _ in range(H)]  # U_cols[h] will hold many PIT values across series

        for idx in range(len(ds_test)):
            sample = ds_test[idx]

            # Build a batch of size 1 on the correct device
            uid = int(sample["unique_id"])
            batch1 = {
                "ds":         torch.tensor([0], device=device),
                "unique_id":  torch.tensor([uid], device=device),
                "y_past":     sample["y_past"].unsqueeze(0).to(device=device, dtype=torch.float32),
                "c_ctx_future_unknown": sample["c_ctx_future_unknown"].unsqueeze(0).to(device=device, dtype=torch.float32),
                "c_ctx_future_known":   sample["c_ctx_future_known"].unsqueeze(0).to(device=device, dtype=torch.float32),
                "c_fct_future_known":   sample["c_fct_future_known"].unsqueeze(0).to(device=device, dtype=torch.float32),
                "y_future":   sample["y_future"].unsqueeze(0).to(device=device, dtype=torch.float32),
            }

            with torch.no_grad():
                # samples_t: (S, B=1, H)
                samples_t, _ = module.model.sample(batch1, n_per_series=S_pit, track_grad=False)

            # Convert to numpy with the (S,B,H) layout expected by pits_from_samples
            samples_np = samples_t.detach().cpu().numpy()           # (S,1,H)
            y_true_np  = sample["y_future"].unsqueeze(0).detach().cpu().numpy()  # (1,H)

            # Compute PITs for this series across horizons; randomized=True is standard
            U_bh = pits_from_samples(samples_np, y_true_np, randomized=True)  # shape (B=1, H)
            U_1H = np.asarray(U_bh).reshape(-1)  # (H,)

            # Append each horizon’s PIT value
            for h in hs_plot:
                u = float(U_1H[h])
                if np.isfinite(u):
                    U_cols[h].append(u)

        # Now plot one histogram per requested horizon
        safe_model = model_name.replace(" ", "").lower().strip()
        safe_size  = model_size.replace(" ", "").lower().strip()

        for h in hs_plot:
            U_h = np.asarray(U_cols[h], dtype=float)
            if U_h.size == 0:
                log.info("PIT[h=%d]: skipped (no data).", h)
                continue

            title = f"{model_name} ({model_size}) — PIT histogram — h={h+1}"
            fname = f"pit_{safe_model}_{safe_size}_h{h+1}_bins{bins}.png"
            out_path = pit_dir / fname

            _plot_pit_histogram(U_h, bins=bins, title=title, out_path=out_path)

        log.info("PIT histograms written → %s", str(pit_dir))


if __name__ == "__main__":
    main()