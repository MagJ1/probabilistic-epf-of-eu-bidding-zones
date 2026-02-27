# src/models/normalizing_flows/runners/train_flows.py
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import json
import torch
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import joblib
import sys

from utils.logging_utils import init_logging, get_logger
from utils.device import pick_accelerator_and_devices
from utils.paths import prepare_train_run_dirs, hydra_path_helper
from utils.feature_hash import feature_hash
from utils.run_ids import default_run_id, pick_run_id

from models.normalizing_flows.FlowForecaster import FlowForecaster
from models.normalizing_flows.datamodule import FlowForecasterDataModule
from models.normalizing_flows.module import FlowForecasterModule


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_train",
)
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.seed), workers=True)

    runs_root = Path(to_absolute_path(cfg.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    fhash = feature_hash(cfg.features.ck_cols, cfg.features.cu_cols)
    auto_id = default_run_id(f"f{fhash}") 
    user_id = getattr(cfg.train, "run_id", None)

    if user_id:   # STRICT: if provided by super-runner, use it as-is
        run_id = str(user_id)
    else:
        run_id = auto_id

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = prepare_train_run_dirs(run_dir)

    (dirs["meta"] / "run_id.txt").write_text(run_id)
    (dirs["meta"] / "runs_root.txt").write_text(str(runs_root))


    # --- python logging
    init_logging(
        log_dir=str(dirs["py"]),
        run_id=run_id,  # or run_id
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True, 
        unify_format=False)

    log = get_logger("train_runner")
    log.info("Starting run %s", run_id)

    train_path = Path(to_absolute_path(cfg.data.train_csv_path))
    test_path  = Path(to_absolute_path(cfg.data.test_csv_path))

    # --- Lightning logger (TensorBoard) ---
    tb_logger = TensorBoardLogger(
        save_dir=str(runs_root.parent / "logs_tb"),      # logs/<tb dir>
        name="",        # e.g., tiny/small/base
        version=run_id)
    log.info(f"NUM_WORKERS: {cfg.train.num_workers}")
    log.info(f"PERSISTENT_WORKERS: {cfg.train.persistent_workers}")

    # --- DataModule from cfg ---
    dm = FlowForecasterDataModule(
        train_csv_path=train_path,
        test_csv_path=test_path,
        context_length=cfg.data.context_length,
        forecast_horizon=cfg.data.forecast_horizon,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        persistent_workers=bool(cfg.train.persistent_workers),
        val_split_date=cfg.data.val_split_date,
        date_col=cfg.data.date_col,
        id_col=cfg.data.id_col,
        y_col=cfg.data.y_col,
        ck_cols=cfg.features.ck_cols,
        cu_cols=cfg.features.cu_cols,
        past_unknown_cov_cutoff=cfg.model.enc_unknown_cutoff,
        scale_data=cfg.data.scale_data,
        realistic_mode=cfg.model.realistic_mode,
        origin_stride_train=cfg.data.origin_stride_train,
        origin_stride_val=cfg.data.origin_stride_val,
        origin_anchor_hour_train=cfg.data.origin_anchor_hour_train,
        origin_anchor_hour_val=cfg.data.origin_anchor_hour_val
    )
    dm.setup(stage="fit")


    # Save scalers for test-time
    joblib.dump(dm.get_scalers(), dirs["data"] / "scalers.pkl")

    # --- Model from cfg ---

    torch.set_default_dtype(torch.float32)   # before building the model
    # optional on MPS:
    torch.set_float32_matmul_precision("medium")
    model = FlowForecaster(
        tf_in_size=cfg.model.tf_in_size,
        nf_hidden_dim=cfg.model.nf_hidden_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        n_flow_layers=cfg.model.n_flow_layers,
        n_made_blocks=cfg.model.n_made_blocks,
        tf_dropout=cfg.model.tf_dropout,
        c_future_known=len(cfg.features.ck_cols),
        c_future_unknown=len(cfg.features.cu_cols),
        context_length=cfg.data.context_length,
        forecast_horizon=cfg.data.forecast_horizon,
        enc_unknown_cutoff=cfg.model.enc_unknown_cutoff,
        dec_known_past_injection_horizon=cfg.model.dec_known_past_injection_horizon,
        realistic_mode=cfg.model.realistic_mode,
    )
    

    log.info("Features: ck=%d, cu=%d", len(dm.ck_cols), len(dm.cu_cols))
    assert model.c_future_known == len(dm.ck_cols)
    assert model.c_future_unknown == len(dm.cu_cols)

    module = FlowForecasterModule(
        model=model,
        lr=cfg.train.lr,
        warmup_epochs=cfg.train.warmup_epochs,
        scalers=dm.get_scalers(),
        loss_metric=cfg.train.loss_metric,
        n_samples_loss=cfg.train.n_samples_loss,
        y_col=cfg.data.y_col,
        beta=cfg.train.beta,
        k_slices=cfg.train.k_slices,
        eval_metrics=cfg.train.eval_metrics,
        n_samples_eval=cfg.train.n_samples_eval,
        ece_taus=cfg.train.ece_taus
    )

    monitor_metric_key = f"val_{cfg.train.loss_metric}"
    # --- Checkpoints ---
    ckpt_cb = ModelCheckpoint(
        monitor=monitor_metric_key,
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=str(dirs["ckpt"]),
        filename="best-{epoch:02d}-{" + monitor_metric_key + ":.4f}",
    )

    # --- Early stopping ---
    es_cb = EarlyStopping(
        monitor=monitor_metric_key,   # must be logged by LightningModule on validation
        mode="min",
        patience=getattr(cfg.train, "early_patience", 10),  # tweak in config
        min_delta=getattr(cfg.train, "early_min_delta", 0.0),
        verbose=False,
        check_on_train_epoch_end=False,
    )

    # --- Trainer (device helper) ---
    accel, devices, precision, _ = pick_accelerator_and_devices()
    # accel="cpu"
    log.info("Accelerator=%s devices=%s precision=%s", accel, devices, precision)
    trainer = Trainer(
        max_epochs=cfg.train.n_epochs,
        accelerator=accel,
        devices=devices,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        logger=tb_logger,
        callbacks=[ckpt_cb] + ([es_cb] if bool(cfg.train.early_stopping) else []),
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        log_every_n_steps=cfg.train.log_every_n_steps,
        default_root_dir=str(dirs["root"]),
        enable_progress_bar=bool(cfg.train.enable_progress_bar)
    )

    # --- Save resolved config snapshot ---
    (dirs["meta"] / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    with open(dirs["meta"] / "params.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    # --- Resume or fresh ---
    last_ckpt = dirs["ckpt"] / "last.ckpt"
    resume_ckpt = str(last_ckpt) if bool(getattr(cfg, "resume", False)) and last_ckpt.exists() else None

    if resume_ckpt:
        log.info("Resuming from %s", resume_ckpt)
    else:
        log.info("Starting fresh (no resume)")

    trainer.fit(module, datamodule=dm, ckpt_path=resume_ckpt)
    log.info("callback_metrics keys at end: %s", list(trainer.callback_metrics.keys()))

    ck_new = list((dirs["ckpt"]).glob("*.ckpt"))
    if len(ck_new) == 0:
        log.error("Training finished but no checkpoints found in %s", dirs["ckpt"])
        raise RuntimeError("No checkpoints written; aborting to avoid half-baked test.")
    # --- Dump final train metrics (handy for Optuna/postprocessing) ---

    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    with open(dirs["metrics"] / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    env = {
    "pytorch": torch.__version__,
    "pl": pl.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    (dirs["meta"] / "env.json").write_text(json.dumps(env, indent=2))

    log.info("Finished run %s", run_id)
    try:
        (run_dir / "_done.train").write_text("ok")
    except Exception as e:
        log.warning("Failed to write train sentinel: %s", e)


if __name__ == "__main__":
    main()