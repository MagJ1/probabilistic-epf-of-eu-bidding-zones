# src/models/nhits_qra/runners/test_nhits_qra.py
from __future__ import annotations
import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import joblib
import torch

from utils.paths import prepare_test_run_dirs
from utils.logging_utils import init_logging, get_logger
from utils.device import pick_accelerator_and_devices

from models.nhits_qra.nhits.model_factory import build_nhits_model
from models.nhits_qra.nhits.nhits_module import NHITSForecasterModule
from models.nhits_qra.datamodule import NHITSQRADataModule
from models.nhits_qra.qra.qra_pipeline import run_qra  
from utils.helpers import _find_best_ckpt
from utils.seeds import seed_everything_hard
import numpy as np
from models.nhits_qra.logging_helpers import log_test_run_header, brief_qra_summary, _pp_path
from models.nhits_qra.helpers import read_model_io
from models.nhits_qra.helpers import build_test_metrics


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_test",
)
def main(cfg: DictConfig):
    """
    Expected Hydra keys:
      test.source_run_dir:  path to the completed training run (its run_dir)
      out_dir:              where to place this test run's outputs
      tag:                  a short label (e.g., 'test')
      test.batch_size:      eval batch size
      test.num_workers:     dataloader workers
      test.enable_progress_bar: bool
      test.n_samples_eval:  (optional) if you log any sampling-based metrics in module
    """
    seed_everything_hard(int(cfg.seed))
    # --- Locate the finished TRAIN run we will evaluate ---
    train_run_dir = Path(to_absolute_path(cfg.test.source_run_dir)).resolve()
    if not train_run_dir.exists():
        raise FileNotFoundError(f"Training run dir not found: {train_run_dir}")

    test_run_dir = Path(to_absolute_path(cfg.out_dir)).resolve()
    test_run_dir.mkdir(parents=True, exist_ok=True)
    dirs = prepare_test_run_dirs(test_run_dir)  # creates {root,metrics,py,pred,meta,art}
    train_run_id = str(cfg.test.source_run_id)
    # --- Logging ---
    init_logging(
        log_dir=str(dirs["py"]),
        run_id=str(cfg.tag),
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False,
    )
    log = get_logger("test_nhits_qra")
    log.info("Testing from train run: %s", str(cfg.test.source_run_id))

    # --- Load TRAIN run's resolved config to recreate shapes/hparams exactly ---
    train_meta_dir = train_run_dir / "meta"
    train_data_dir = train_run_dir / "data"
    train_ckpt_dir = train_run_dir / "train" / "checkpoints"

    if not train_ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints found under {train_ckpt_dir}. Refusing to test.")

    with open(train_meta_dir / "params.json", "r") as f:
        train_cfg = OmegaConf.create(json.load(f))  # DictConfig-like

    # Resolve data paths via current Hydra CWD
    train_path = Path(to_absolute_path(cfg.data.train_csv_path))
    test_path  = Path(to_absolute_path(cfg.data.test_csv_path))

    # --- TensorBoard logger for this test run (co-located with training logs root) ---
    tb_root = train_run_dir.parents[1] / "logs_tb"
    tb_logger = TensorBoardLogger(
        save_dir=str(tb_root),
        name="",
        version=f"{train_run_id}-{cfg.tag}",
    )

    # --- DataModule built with training shapes/columns ---
    dm = NHITSQRADataModule(
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        context_length=int(train_cfg.data.context_length),
        forecast_horizon=int(train_cfg.data.forecast_horizon),
        batch_size=int(cfg.test.batch_size),
        num_workers=int(cfg.test.num_workers),
        persistent_workers=bool(int(cfg.test.num_workers) > 0),
        # split & cols
        val_split_date=cfg.data.val_split_date,
        date_col=str(train_cfg.data.date_col),
        id_col=str(train_cfg.data.id_col),
        y_col=str(train_cfg.data.y_col),
        # realism
        past_unknown_cov_cutoff=int(train_cfg.model.past_unknown_cov_cutoff),
        realistic_mode=bool(train_cfg.model.realistic_mode),
        # registry-driven features (mirror training)
        scale_data=bool(train_cfg.data.scale_data),
        scale_target=bool(train_cfg.data.scale_target),
        enable_registry=bool(train_cfg.features.enable_registry),
        fit_on=str(train_cfg.features.fit_on),
        registry=dict(train_cfg.features.registry) if bool(train_cfg.features.enable_registry) else {},
        # active/overrides from TRAIN config (so shapes match)
        active_ck=list(train_cfg.features.active.get("ck")),
        active_cu=list(train_cfg.features.active.get("cu")),
        active_static=list(train_cfg.features.active.get("static")),
        include_feature=list(train_cfg.features.include),
        exclude_feature=list(train_cfg.features.exclude),
        # scalers should be reused
        share_scalers=True,
    )
    dm.setup(stage="test")

    # Use data from meta/model_io.json to derive features
    model_io = read_model_io(train_run_dir)

    add_flag      = bool(model_io["add_flag"])
    hist_exog_list = list(model_io["hist_exog_list"])
    futr_exog_list = list(model_io["futr_exog_list"])
    stat_exog_list = list(model_io["stat_exog_list"])

    # --- Recreate NHITS with training hyperparameters ---
    nhits = build_nhits_model(
        input_size=int(train_cfg.model.input_size),
        h=int(train_cfg.model.h),
        n_blocks=list(train_cfg.model.n_blocks),
        mlp_units=[list(u) for u in train_cfg.model.mlp_units],
        dropout_prob_theta=float(train_cfg.model.dropout_prob_theta),
        n_pool_kernel_size=list(train_cfg.model.n_pool_kernel_size),
        n_freq_downsample=list(train_cfg.model.n_freq_downsample),
        hist_exog_list=hist_exog_list,
        futr_exog_list=futr_exog_list,
        stat_exog_list=stat_exog_list,
    )

    # --- Load scalers saved during TRAIN (needed for module and QRA inverse-scaling) ---
    targ_scaler  = joblib.load(train_data_dir / "target_scaler.pkl")    # QRA uses this, too

    # --- Pick checkpoint & build LightningModule (CPU map; Trainer will move) ---
    ckpt_path = _find_best_ckpt(train_ckpt_dir / "nhits")

    log_test_run_header(
        log, 
        cfg=cfg, 
        train_run_dir=train_run_dir, 
        test_run_dir=test_run_dir, 
        dirs=dirs, 
        ckpt_path=ckpt_path
        )

    module = NHITSForecasterModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        nhits_model=nhits,
        learning_rate=float(train_cfg.train_nhits.lr),
        loss=str(getattr(train_cfg.train_nhits, "loss", "mae")),
        include_flags_ctx=add_flag,
        include_static=bool(getattr(train_cfg.train_nhits, "include_static", True)),
        hist_exog_list=hist_exog_list,
        futr_exog_list=futr_exog_list,
        stat_exog_list=stat_exog_list,
        warmup_epochs=int(train_cfg.train_nhits.warmup_epochs),
        map_location="cpu",
    )

    # --- Trainer on appropriate device ---
    accel, devices, precision, _ = pick_accelerator_and_devices()
    log.info("Accelerator=%s devices=%s precision=%s", accel, devices, precision)
    trainer = Trainer(
        accelerator=accel,
        devices=devices,
        precision=precision,
        logger=tb_logger,
        default_root_dir=str(dirs["root"]),
        log_every_n_steps=int(getattr(cfg.test, "log_every_n_steps", 50)),
        enable_progress_bar=bool(getattr(cfg.test, "enable_progress_bar", True)),
    )

    # --- Optional: run Lightning test loop ---
    try:
        trainer.test(module, datamodule=dm)
        log.info("callback_metrics keys at end: %s", list(trainer.callback_metrics.keys()))
    except Exception as e:
        log.warning("Lightning test() skipped/failed gracefully: %s", e)

    # --- Run QRA on TEST split by LOADING trained QRA artifacts from the train run ---
    qra_summary = None
    if bool(train_cfg.train_qra.enabled) is True:
        qra_summary = run_qra(
            train_cfg=train_cfg,
            module=module,
            dm=dm,
            dirs={
                "root": dirs["root"],
                "metrics": dirs["metrics"],
                "meta": dirs["meta"],
                "data": dirs.get("data", dirs["root"] / "data"),
                "ckpt": test_run_dir / "checkpoints",
                "pred": dirs["pred"],
                "art" : dirs["art"],
                "logs_tb": dirs["logs_tb"]
            },
            log=log,
            split="test",
            load_from_train_run=train_run_dir,
            target_scaler=targ_scaler,
            test_cfg=cfg,
        )

        (dirs["metrics"] / "qra" / "summary.json").write_text(json.dumps(qra_summary, indent=2))
        summary_path = dirs["metrics"] / "qra" / "summary.json"
        log.info("QRA summary written: %s", _pp_path(summary_path))
        log.info("QRA[test] digest: %s", brief_qra_summary(qra_summary))
    else:
        log.info("No QRA evaluation desired.")

    # ---- ONE metrics path (no else) ----
    metrics = build_test_metrics(
        trainer=trainer,
        qra_summary=qra_summary,  # None if QRA disabled
        train_cfg=train_cfg,
        test_cfg=cfg,
    )
    (dirs["metrics"] / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- Snapshot environment for reproducibility ---
    env = {
        "pytorch": torch.__version__,
        "pl": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "train_run_dir": str(train_run_dir),
        "ckpt_used": str(ckpt_path.name),
    }
    (dirs["meta"] / "test_env.json").write_text(json.dumps(env, indent=2))
    (dirs["meta"] / "run_pointer.txt").write_text(str(Path(__file__).resolve()))
    (test_run_dir / "tested_from_run_dir.txt").write_text(str(train_run_dir))

    # Sentinel to grep runs that completed
    (test_run_dir / f"_done.test.{str(getattr(cfg, 'tag', 'test'))}").write_text("ok")

    log.info("Finished TEST for %s", train_run_dir.name)


if __name__ == "__main__":
    main()