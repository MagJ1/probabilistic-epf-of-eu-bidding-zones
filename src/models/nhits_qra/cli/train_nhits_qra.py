# src/models/nhits_qra/runners/train_nhits.py
from __future__ import annotations
from pathlib import Path
import json
import joblib
import pytorch_lightning as pl
import torch
import numpy as np
from scipy.stats import norm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from utils.logging_utils import init_logging, get_logger
from utils.device import pick_accelerator_and_devices
from utils.paths import prepare_train_run_dirs
from utils.feature_hash import feature_hash 
from utils.run_ids import default_run_id
from utils.feature_select import select_features

from models.nhits_qra.nhits.model_factory import build_nhits_model
from models.nhits_qra.nhits.nhits_module import NHITSForecasterModule
from models.nhits_qra.datamodule import NHITSQRADataModule
from models.nhits_qra.qra.qra_pipeline import run_qra
from models.nhits_qra.qra.qra_help import count_qra_params
from models.nhits_qra.logging_helpers import brief_qra_summary, log_train_run_header, _brief_train_summary
from models.nhits_qra.helpers import write_model_io


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_train",
)
def main(cfg: DictConfig):

    # ----------------- seeds & dirs -----------------
    pl.seed_everything(int(cfg.seed), workers=True)

    runs_root = Path(to_absolute_path(cfg.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    # ----------------- feature selection (for hashing + DM sanity) -----------------
    ck_cols, cu_cols, static_cols = select_features(
        enable_registry=bool(cfg.features.enable_registry),
        registry=dict(cfg.features.registry) if bool(cfg.features.enable_registry) else {},
        base_ck=list(getattr(cfg.features, "ck_cols", [])),
        base_cu=list(getattr(cfg.features, "cu_cols", [])),
        base_static=list(getattr(cfg.features, "static_cols", [])),
        active_ck=list(getattr(cfg.features, "active", {}).get("ck", [])),
        active_cu=list(getattr(cfg.features, "active", {}).get("cu", [])),
        active_static=list(getattr(cfg.features, "active", {}).get("static", [])),
        include=list(getattr(cfg.features, "include", [])),
        exclude=list(getattr(cfg.features, "exclude", [])),
    )

    fhash = feature_hash(ck_cols, cu_cols, static_cols)

    auto_id = default_run_id(f"f{fhash}")
    user_id = getattr(cfg.train, "run_id", None)
    run_id = str(user_id) if user_id else auto_id

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = prepare_train_run_dirs(run_dir)
    (dirs["meta"] / "run_id.txt").write_text(run_id)
    (dirs["meta"] / "runs_root.txt").write_text(str(runs_root))

    # ----------------- python logging -----------------
    init_logging(
        log_dir=str(dirs["py"]),
        run_id=run_id,
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False,
    )
    log = get_logger("train_nhits")
    log.info("Starting NHITS run %s", run_id)

    # ----------------- data module -----------------
    train_path = Path(to_absolute_path(cfg.data.train_csv_path))
    test_path  = Path(to_absolute_path(cfg.data.test_csv_path))

    num_workers = int(cfg.train_nhits.num_workers)
    persistent_workers = bool(getattr(cfg.train_nhits, "persistent_workers", num_workers > 0))

    dm = NHITSQRADataModule(
        # paths & windowing
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        context_length=int(cfg.data.context_length),
        forecast_horizon=int(cfg.data.forecast_horizon),
        batch_size=int(cfg.train_nhits.batch_size),
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        # split & cols
        val_split_date=cfg.data.val_split_date,
        date_col=str(cfg.data.date_col),
        id_col=str(cfg.data.id_col),
        y_col=str(cfg.data.y_col),
        # realism
        past_unknown_cov_cutoff=int(getattr(cfg.model, "past_unknown_cov_cutoff", 14)),
        realistic_mode=bool(getattr(cfg.model, "realistic_mode", True)),
        # registry-driven features
        scale_data=bool(cfg.data.scale_data),
        scale_target=bool(cfg.data.scale_target),
        enable_registry=bool(cfg.features.enable_registry),
        fit_on=str(cfg.features.fit_on),
        registry=dict(cfg.features.registry) if bool(cfg.features.enable_registry) else {},
        # active/overrides from Hydra
        active_ck=list(getattr(cfg.features, "active", {}).get("ck", [])),
        active_cu=list(getattr(cfg.features, "active", {}).get("cu", [])),
        active_static=list(getattr(cfg.features, "active", {}).get("static", [])),
        include_feature=list(getattr(cfg.features, "include", [])),   # <-- matches child DM arg name
        exclude_feature=list(getattr(cfg.features, "exclude", [])),   # <-- matches child DM arg name
        # scaler sharing to val/test
        share_scalers=True,
    )
    dm.setup()

    # Assert that the DM selection matches the runner selection
    assert ck_cols == list(dm.train_dataset.ck_cols)
    assert cu_cols == list(dm.train_dataset.cu_cols)
    assert static_cols == list(dm.train_dataset.static_cols)

    train_ds = dm.train_dataset
    log.info(f"Train dataset windows: {len(train_ds)}")
    b = train_ds[0]
    Ck, Cu = len(ck_cols), len(cu_cols)
    assert b["c_fct_future_known"].shape[-1] == Ck
    assert b["c_ctx_future_known"].shape[-1] == Ck
    assert b["c_ctx_future_unknown"].shape[-1] == Cu

    # single source of truth = dataset (for lists handed to module for checking/logging)
    # single source of truth = dataset (for lists handed to module for checking/logging)
    # lists from dataset (single source of truth)

    include_flags_ctx = bool(cfg.train_nhits.include_flags_ctx)
    ck_cols     = list(train_ds.ck_cols)
    cu_cols     = list(train_ds.cu_cols)
    static_cols = list(train_ds.static_cols)

    # add exactly one flag if (a) user wants flags and (b) there is at least one CU column
    add_flag = include_flags_ctx and (len(cu_cols) > 0)

    # model-facing lists
    hist_exog_list = ck_cols + cu_cols + (["__cu_known_flag"] if add_flag else [])
    futr_exog_list = ck_cols
    stat_exog_list = static_cols if bool(getattr(cfg.train_nhits, "include_static", True)) else []

    p = write_model_io(
        dirs["meta"],
        ck_cols=ck_cols,
        cu_cols=cu_cols,
        static_cols=static_cols,
        add_flag=add_flag,
        hist_exog_list=hist_exog_list,
        futr_exog_list=futr_exog_list,
        stat_exog_list=stat_exog_list,
    )
    log.info("Wrote model IO snapshot: %s", str(p))

    # Persist dataset scalers (from *train* split)
    feat_scalers = dm.train_dataset.get_scalers()
    targ_scaler  = dm.train_dataset.get_target_scaler()
    joblib.dump(feat_scalers, dirs["data"] / "feature_scalers.pkl")
    joblib.dump(targ_scaler,  dirs["data"] / "target_scaler.pkl")

    log.info("DM ready. ck=%d, cu=%d, static=%d",
             len(getattr(dm.train_dataset, "ck_cols", [])),
             len(getattr(dm.train_dataset, "cu_cols", [])),
             len(getattr(dm.train_dataset, "static_cols", [])))

    # ----------------- model -----------------
    # ensure input_size/context_length & h/forecast_horizon match
    assert int(cfg.model.input_size) == int(cfg.data.context_length), "input_size must equal context_length"
    assert int(cfg.model.h) == int(cfg.data.forecast_horizon), "model.h must equal forecast_horizon"
    log.info(f"hist_exog_list: {hist_exog_list}")
    log.info(f"futr_exog_list: {futr_exog_list}")
    log.info(f"stat_exog_list: {stat_exog_list}")
    nhits = build_nhits_model(
        input_size=int(cfg.model.input_size),
        h=int(cfg.model.h),
        n_blocks=list(cfg.model.n_blocks),
        mlp_units=[list(u) for u in cfg.model.mlp_units],
        dropout_prob_theta=float(cfg.model.dropout_prob_theta),
        n_pool_kernel_size=list(cfg.model.n_pool_kernel_size),
        n_freq_downsample=list(cfg.model.n_freq_downsample),
        # lists are for doc/logging/asserts in the LightningModule
        hist_exog_list=hist_exog_list,
        futr_exog_list=futr_exog_list,
        stat_exog_list=stat_exog_list,
    )



    module = NHITSForecasterModule(
        nhits_model=nhits,
        learning_rate=float(cfg.train_nhits.lr),
        loss=str(getattr(cfg.train_nhits, "loss", "mae")),
        include_flags_ctx=bool(getattr(cfg.train_nhits, "include_flags_ctx", False)),  # <- wired to Hydra
        include_static=bool(getattr(cfg.train_nhits, "include_static", True)),         # <- wired to Hydra
        hist_exog_list=hist_exog_list,
        futr_exog_list=futr_exog_list,
        stat_exog_list=stat_exog_list,
        warmup_epochs=int(cfg.train_nhits.warmup_epochs),
        swag_cfg=cfg.train_nhits.swag
    )

    # ----------------- trainer -----------------
    tb_logger = TensorBoardLogger(
        save_dir=str(runs_root.parent / "logs_tb"),
        name="",
        version=run_id,
    )

    # FIX: monitor key must match the chosen loss name
    loss_name = str(getattr(cfg.train_nhits, "loss", "mae")).lower()
    monitor_key = "val_mse" if loss_name == "mse" else "val_mae"

    log_train_run_header(
        log,
        cfg=cfg,
        run_id=run_id,
        run_dir=run_dir,
        dirs=dirs,
        monitor_key=monitor_key,
    )

    ckpt_cb = ModelCheckpoint(
        monitor=monitor_key,
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=str(dirs["ckpt"] / "nhits"),
        filename="best-{epoch:02d}-{" + monitor_key + ":.4f}",
    )

    callbacks = [ckpt_cb]
    if bool(getattr(cfg.train_nhits, "early_stopping", True)):
        es_cb = EarlyStopping(
            monitor=monitor_key,
            mode="min",
            patience=int(getattr(cfg.train_nhits, "early_patience", 10)),
            min_delta=float(getattr(cfg.train_nhits, "early_min_delta", 0.0)),
            verbose=False,
            check_on_train_epoch_end=False,
        )
        callbacks.append(es_cb)

    accel, devices, precision, _ = pick_accelerator_and_devices()
    log.info("Accelerator=%s devices=%s precision=%s", accel, devices, precision)


    trainer = Trainer(
        max_epochs=int(cfg.train_nhits.n_epochs),
        accelerator=accel,
        devices=devices,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=float(cfg.train_nhits.gradient_clip_val),
        accumulate_grad_batches=int(cfg.train_nhits.accumulate_grad_batches),
        logger=tb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=int(cfg.train_nhits.num_sanity_val_steps),
        log_every_n_steps=int(cfg.train_nhits.log_every_n_steps),
        default_root_dir=str(dirs["root"]),
        enable_progress_bar=bool(cfg.train_nhits.enable_progress_bar),
    )

    # ----------------- save resolved config & selected features -----------------
    (dirs["meta"] / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    with open(dirs["meta"] / "params.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    (dirs["meta"] / "features_selected.json").write_text(
        json.dumps({"ck": ck_cols, "cu": cu_cols, "static": static_cols}, indent=2)
    )

    # ----------------- train -----------------
    trainer.fit(module, datamodule=dm)

    

    # checkpoint sanity
    ck_new = list((dirs["ckpt"]/"nhits").glob("*.ckpt"))
    if len(ck_new) == 0:
        raise RuntimeError(f"No checkpoints written in {dirs['ckpt']}")

    # dump train metrics
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    (dirs["metrics"] / "train_metrics.json").write_text(json.dumps(metrics, indent=2))

    env = {
        "pytorch": torch.__version__,
        "pl": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    (dirs["meta"] / "env.json").write_text(json.dumps(env, indent=2))

    train_brief = _brief_train_summary(metrics, monitor_key=monitor_key, ckpt_cb=ckpt_cb)
    log.info("TRAIN summary: %s", train_brief)

    try:
        (run_dir / "_done.train").write_text("ok")
    except Exception as e:
        log.warning("Failed to write train sentinel: %s", e)

    log.info(f"Finished NHITS run {run_id}.")


    # ----------------- optional QRA -----------------
    if bool(cfg.train_qra.enabled):
        qra_summary = run_qra(cfg, module, dm, dirs, split="val",log=log)
        # store a single-line summary too
        (dirs["metrics"] / "qra" / "summary_val.json").write_text(json.dumps(qra_summary, indent=2))
        log.info("QRA summary: %s", brief_qra_summary(qra_summary))
        
    else:
        log.info("QRA disabled — skipping design/PCA/fit/metrics.")


if __name__ == "__main__":
    main()