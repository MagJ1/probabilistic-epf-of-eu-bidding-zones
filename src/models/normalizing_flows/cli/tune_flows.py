# src/models/normalizing_flows/cli/tune_flows.py
import json
import math
from pathlib import Path
from typing import Dict, Any, Union, Tuple, List
import sys

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, NopPruner


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import joblib
import torch
import numpy as np

from utils.paths import prepare_train_run_dirs, hydra_path_helper
from utils.logging_utils import init_logging, get_logger
from utils.device import pick_accelerator_and_devices
from utils.optuna_helper import SimpleOptunaPruner, _make_study, _trial_run_dir

from models.normalizing_flows.FlowForecaster import FlowForecaster
from models.normalizing_flows.datamodule import FlowForecasterDataModule
from models.normalizing_flows.module import FlowForecasterModule


def _parse_metric_names(metric_cfg) -> List[str]:
    if isinstance(metric_cfg, str):
        return [metric_cfg]
    if isinstance(metric_cfg, (list, tuple, ListConfig)):
        return [str(x) for x in metric_cfg]
    raise TypeError(f"Unsupported type for cfg.tune.metric: {type(metric_cfg)}")

def _suggest_hparams(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    space = cfg.tune.space[str(cfg.model.size)]

    # Sample independently from fixed (static) spaces
    n_heads       = trial.suggest_categorical("n_heads",       list(space.n_heads))
    tf_in_size    = trial.suggest_categorical("tf_in_size",    list(space.tf_in_size))
    n_layers      = trial.suggest_categorical("n_layers",      list(space.n_layers))
    nf_hidden_dim = trial.suggest_categorical("nf_hidden_dim", list(space.nf_hidden_dim))
    n_flow_layers = trial.suggest_categorical("n_flow_layers", list(space.n_flow_layers))
    n_made_blocks = trial.suggest_categorical("n_made_blocks", list(space.n_made_blocks))
    warmup_epochs = trial.suggest_categorical("warmup_epochs", list(space.warmup_epochs))

    # Enforce constraints deterministically
    if tf_in_size % n_heads != 0:
        # Fail fast without polluting the study with an infeasible point
        raise optuna.TrialPruned(f"Infeasible combo: tf_in_size={tf_in_size}, n_heads={n_heads}")

    # Continuous ranges are fine as-is
    tf_dropout = trial.suggest_float("tf_dropout", float(space.tf_dropout[0]), float(space.tf_dropout[1]))
    lr_log10   = trial.suggest_float("lr_log10",   float(space.lr_log10[0]),   float(space.lr_log10[1]))
    lr = float(10.0 ** lr_log10)

    return dict(
        tf_in_size=tf_in_size,
        n_heads=n_heads,
        n_layers=n_layers,
        nf_hidden_dim=nf_hidden_dim,
        n_flow_layers=n_flow_layers,
        n_made_blocks=n_made_blocks,
        tf_dropout=tf_dropout,
        lr=lr,
        warmup_epochs=int(warmup_epochs),
    )

# --------------------------- objective ---------------------------------- #

def _objective(trial: optuna.Trial, cfg: DictConfig, base_dir: Path) -> Union[float, Tuple[float, ...]]:
    pl.seed_everything(int(cfg.seed) + int(trial.number), workers=True)
    hps = _suggest_hparams(trial, cfg)

    trial_dir = _trial_run_dir(base_dir, trial)
    dirs = prepare_train_run_dirs(trial_dir)

    init_logging(
        log_dir=str(dirs["py"]),
        run_id=f"trial_{trial.number:04d}",
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False,
    )
    log = get_logger(f"tune.trial{trial.number:04d}")
    log.info("Trial %d hparams: %s", trial.number, json.dumps(hps))

    metric_names = _parse_metric_names(cfg.tune.metric)
    is_multi = len(metric_names) > 1

    train_path = Path(to_absolute_path(cfg.data.train_csv_path))
    test_path  = Path(to_absolute_path(cfg.data.test_csv_path))

    tb_logger = TensorBoardLogger(
        save_dir=str(trial_dir.parent / "logs_tb"),
        name="",
        version=trial_dir.name
    )

    dm = FlowForecasterDataModule(
        train_csv_path=train_path,
        test_csv_path=test_path,
        context_length=cfg.data.context_length,
        forecast_horizon=cfg.data.forecast_horizon,
        batch_size=cfg.train.batch_size,
        num_workers=getattr(cfg.train, "num_workers", 4),
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
    joblib.dump(dm.get_scalers(), dirs["data"] / "scalers.pkl")

    torch.set_float32_matmul_precision("medium")

    model = FlowForecaster(
        tf_in_size=hps["tf_in_size"],
        nf_hidden_dim=hps["nf_hidden_dim"],
        n_layers=hps["n_layers"],
        n_heads=hps["n_heads"],
        n_flow_layers=hps["n_flow_layers"],
        n_made_blocks=hps["n_made_blocks"],
        c_future_known=len(cfg.features.ck_cols),
        c_future_unknown=len(cfg.features.cu_cols),
        context_length=cfg.data.context_length,
        forecast_horizon=cfg.data.forecast_horizon,
        enc_unknown_cutoff=cfg.model.enc_unknown_cutoff,
        dec_known_past_injection_horizon=cfg.model.dec_known_past_injection_horizon,
        realistic_mode=cfg.model.realistic_mode,
        tf_dropout=hps["tf_dropout"],
    )

    module = FlowForecasterModule(
        model=model,
        lr=hps["lr"],
        warmup_epochs=hps["warmup_epochs"],
        scalers=dm.get_scalers(),
        loss_metric=str(getattr(cfg.train, "loss_metric", "es")),  # allow cfg
        n_samples_loss=cfg.train.n_samples_loss,
        beta=cfg.train.beta,
        k_slices=cfg.train.k_slices,
        eval_metrics=cfg.train.eval_metrics,
        n_samples_eval=cfg.train.n_samples_eval,
        y_col=cfg.data.y_col,
        ece_taus=getattr(cfg.train, "ece_taus", None),
    )

    # --- monitor only primary metric for ckpt ---
    monitor_key = metric_names[0]
    mode = "min"  # all your metrics are lower-is-better

    ckpt_cb = ModelCheckpoint(
        monitor=monitor_key,
        mode=mode,
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        dirpath=str(dirs["ckpt"]),
        filename="best-{epoch:02d}-{" + monitor_key + ":.4f}",
    )

    pruning_cb = None
    if (not is_multi) and str(cfg.tune.pruner).lower() != "none":
        pruning_cb = SimpleOptunaPruner(trial, monitor=monitor_key, ignore_first_vals=3)

    accel, devices, precision, _ = pick_accelerator_and_devices()
    log.info("Accelerator=%s devices=%s precision=%s", accel, devices, precision)

    trainer = Trainer(
        max_epochs=cfg.train.n_epochs,
        accelerator=accel,
        devices=devices,
        precision=precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        logger=tb_logger,
        callbacks=[c for c in [ckpt_cb, pruning_cb] if c is not None],
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        log_every_n_steps=cfg.train.log_every_n_steps,
        default_root_dir=str(dirs["root"]),
        enable_progress_bar=bool(cfg.train.enable_progress_bar),
    )

    trainer.fit(module, datamodule=dm)

    # --- Evaluate metrics from the BEST checkpoint (so ECE matches the tuned model) ---
    best_ckpt = ckpt_cb.best_model_path
    try:
        ckpt_path = best_ckpt if best_ckpt else None
        val_list = trainer.validate(model=module, datamodule=dm, ckpt_path=ckpt_path, verbose=False)
        val_metrics_all = dict(val_list[0]) if val_list else {}
    except Exception:
        val_metrics_all = {}

    # fallback: whatever is in callback_metrics
    cb_metrics = trainer.callback_metrics
    for k, v in cb_metrics.items():
        if k not in val_metrics_all:
            try:
                val_metrics_all[k] = float(v)
            except Exception:
                pass

    # collect objectives
    metrics = {}
    for name in metric_names:
        v = val_metrics_all.get(name, np.inf)
        try:
            v = float(v)
        except Exception:
            v = float(np.inf)
        metrics[name] = v
        trial.set_user_attr(name, float(v))

    (dirs["metrics"] / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    (dirs["root"] / "trial_manifest.json").write_text(json.dumps({
        "trial": trial.number,
        "seed": int(cfg.seed) + int(trial.number),
        "hps": hps,
        "metric_names": metric_names,
        "metrics": metrics,
        "best_ckpt": ckpt_cb.best_model_path,
    }, indent=2))

    # prune only single-objective
    if (not is_multi):
        final = metrics[metric_names[0]]
        step_idx = int(trainer.current_epoch or 0)
        trial.report(final, step=step_idx)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned with {metric_names[0]}={final:.6f}")
        return final

    # multi-objective return tuple
    return tuple(metrics[n] for n in metric_names)


# ----------------------------- main ------------------------------------- #

@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_tune",
)
def main(cfg: DictConfig):
    # Base directory where Hydra chdir’d 
    base_dir = hydra_path_helper(cfg)
    # Create subfolders that look like regular training runs, but per-trial
    init_logging(log_dir=str(base_dir / "logs_py"), 
                 run_id="tuning", 
                 level=str(cfg.logging.log_level), 
                 coexist_with_hydra=True, 
                 unify_format=False)
    log = get_logger("tune_flows.main")

    study = _make_study(cfg)

    target_total = int(cfg.tune.n_trials)
    already = len(study.trials)
    remaining = max(0, target_total - already)

    log.info("cfg.tune.metric=%s (type=%s)", cfg.tune.metric, type(cfg.tune.metric))
    log.info("study.directions=%s", getattr(study, "directions", None))

    if remaining == 0:
        log.info("Study already has %d trials (>= target %d). Exiting.", already, target_total)
        return
    
    log.info("Study %s | storage=%s | directions=%s",
            cfg.tune.study_name, cfg.tune.storage, getattr(study, "directions", cfg.tune.direction))
    
    study.optimize(
        lambda tr: _objective(tr, cfg, base_dir),
        n_trials=int(cfg.tune.n_trials),
        timeout=None if cfg.tune.timeout is None else int(cfg.tune.timeout),
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[],
        n_jobs=1,
    )

    is_multi = isinstance(cfg.tune.metric, (list, tuple, ListConfig))
    if is_multi:
        pareto = [
            {"number": t.number, "values": list(t.values), "params": t.params, "user_attrs": t.user_attrs}
            for t in study.best_trials
        ]
        (base_dir / "pareto.json").write_text(json.dumps(pareto, indent=2))
        log.info("Pareto front size: %d", len(study.best_trials))
    else:
        best = study.best_trial
        summary = {"best_value": best.value, "best_params": best.params, "best_trial_number": best.number}
        (base_dir / "best.json").write_text(json.dumps(summary, indent=2))
        log.info("Best trial #%d %s=%.6f", best.number, cfg.tune.metric, best.value)


if __name__ == "__main__":
    main()