# src/models/nhits_qra/cli/tune_nhits_qra.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Union, Tuple

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, ListConfig
import ast
import time

import optuna

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import joblib
import numpy as np

from utils.paths import prepare_train_run_dirs
from utils.logging_utils import init_logging, get_logger
from utils.device import pick_accelerator_and_devices
from utils.feature_select import select_features
from utils.optuna_helper import _make_study, _trial_run_dir

from models.nhits_qra.nhits.model_factory import build_nhits_model
from models.nhits_qra.nhits.nhits_module import NHITSForecasterModule
from models.nhits_qra.datamodule import NHITSQRADataModule
from models.nhits_qra.qra.qra_pipeline import run_qra

# ----------------------------- Search space ---------------------------------- #

def _suggest_hparams_nhits(trial: optuna.Trial, cfg) -> Dict[str, Any]:
    """
    Flexible NHITS sampler:
      - samples macro knobs (n_stacks, blocks/layers/width base, profiles)
      - synthesizes per-stack lists for n_blocks and mlp_units
      - optionally samples/truncates pooling/frequency schedules
      - samples dropout + lr (log-scale)
    Expects config under: cfg.tune.nhits_space[cfg.model.size]
    """
    space = OmegaConf.to_container(cfg.tune.nhits_space[str(cfg.model.size)], resolve=True)

    # ---- 1) macro shape ----
    n_stacks         = trial.suggest_categorical("n_stacks",           list(space["n_stacks"]))
    base_blocks      = trial.suggest_categorical("blocks_per_stack",   list(space["blocks_per_stack"]))
    layers_per_block = trial.suggest_categorical("layers_per_block",   list(space["layers_per_block"]))
    base_width       = trial.suggest_categorical("width_per_layer",    list(space["width_per_layer"]))

    n_blocks = [int(base_blocks)] * int(n_stacks)


    # ---- 3) synthesize mlp_units per stack (length = n_stacks; each is a list of layer widths) ----
    # NHITS typically uses the same MLP shape for all blocks in a stack; we encode the *block* MLP once per stack.
    def widen(w: int, factor: float, cap: int = 4096) -> int:
        return int(min(cap, max(8, round(w * factor))))

    def _pad_to_len(xs, L) -> list[int]:
        xs = list(xs)
        if len(xs) >= L:
            return xs[:L]
        return xs + [xs[-1]] * (L - len(xs))

    n_stacks_i = int(n_stacks)
    n_layers   = max(1, int(layers_per_block))   # allow single-layer blocks
    width      = int(base_width)

    # For each stack: a list of ints (one hidden size per layer)
    mlp_units = [[width] * n_layers for _ in range(n_stacks_i)]

    # sanity checks (ints, not pairs)
    assert len(n_blocks) == n_stacks_i, "len(n_blocks) must equal n_stacks"
    assert len(mlp_units) == n_stacks_i, f"mlp_units malformed: {mlp_units}"
    assert all(len(stack) == n_layers for stack in mlp_units), \
        f"each stack in mlp_units must have {n_layers} layers, got {mlp_units}"
    import numpy as _np  # or put at top of file
    assert all(isinstance(u, (int, _np.integer)) and u > 0
            for stack in mlp_units for u in stack), \
        "Each MLP unit count must be a positive int"

    # ---- 4) optional pooling/frequency schedules (truncate to n_stacks) ----
    n_pool_kernel_size = None
    if "n_pool_kernel_schedules" in space and space["n_pool_kernel_schedules"]:
        # store choices as strings to make Optuna storage JSON-safe
        pool_keys = [str(s) for s in space["n_pool_kernel_schedules"]]
        pool_key  = trial.suggest_categorical("n_pool_kernel_schedule_key", pool_keys)
        n_pool_kernel_size = list(ast.literal_eval(pool_key))[:int(n_stacks)]

    if n_pool_kernel_size is None:
        n_pool_kernel_size = [2] * int(n_stacks)
    else:
        n_pool_kernel_size = _pad_to_len(n_pool_kernel_size, int(n_stacks))

    n_freq_downsample = None
    if "n_freq_downsample_schedules" in space and space["n_freq_downsample_schedules"]:
        freq_keys = [str(s) for s in space["n_freq_downsample_schedules"]]
        freq_key  = trial.suggest_categorical("n_freq_downsample_schedule_key", freq_keys)
        n_freq_downsample = list(ast.literal_eval(freq_key))[:int(n_stacks)]

    if n_freq_downsample is None:
        n_freq_downsample = [2] * int(n_stacks)
    else:
        n_freq_downsample = _pad_to_len(n_freq_downsample, int(n_stacks))

    # ---- 5) regularization + LR ----
    dr_lo, dr_hi = map(float, space["dropout_prob_theta"])
    lr_lo, lr_hi = map(float, space["lr_log10"])
    dropout      = trial.suggest_float("dropout_prob_theta", dr_lo, dr_hi)
    lr_log10     = trial.suggest_float("lr_log10", lr_lo, lr_hi)
    lr           = float(10.0 ** lr_log10)

    warmup_epochs         = trial.suggest_categorical("warmup_epochs", list(space["warmup_epochs"]))

    return dict(
        # architecture
        n_blocks=n_blocks,                 
        mlp_units=mlp_units,             
        n_pool_kernel_size=n_pool_kernel_size,
        n_freq_downsample=n_freq_downsample,
        # optimization/regularization
        dropout_prob_theta=float(dropout),
        lr=float(lr),
        # (optional) echo useful macros for logging
        n_stacks=int(n_stacks),
        layers_per_block=int(layers_per_block),
        base_width=int(base_width),
        warmup_epochs=int(warmup_epochs)
    )

def _suggest_hparams_swag(trial, cfg):
    space = OmegaConf.to_container(cfg.tune.swag_space, resolve=True)
    return dict(
        start_epoch=trial.suggest_categorical("swag_start_epoch", space["start_epoch"]),
        collect_every=trial.suggest_categorical("swag_collect_every", space["collect_every"]),
        max_rank=trial.suggest_categorical("swag_max_rank", space["max_rank"]),
        scale=float(trial.suggest_categorical("swag_scale", space["scale"])),
    )


def _suggest_hparams_qra(trial: optuna.Trial, cfg) -> Dict[str, Any]:
    space = OmegaConf.to_container(cfg.tune.qra_space[str(cfg.model.size)], resolve=True)

    # core design knobs
    S          = trial.suggest_categorical("mc_nhits_samples_qra", list(space["mc_nhits_samples_qra"]))
    sample_k   = trial.suggest_categorical("sample_k",             list(space["sample_k"]))
    stride     = trial.suggest_categorical("subsample_stride",     list(space["subsample_stride"]))

    q_grids  = space.get("quantiles", None)
    quants   = None
    if q_grids:
        q_keys = [str(q) for q in q_grids]
        q_key  = trial.suggest_categorical("qra_quantiles_key", q_keys)
        quants = list(ast.literal_eval(q_key))

    # PCA
    use_pca    = trial.suggest_categorical("qra_use_pca", list(space["use_pca"]))
    pca_var    = trial.suggest_float("qra_pca_var", float(space["pca_var"][0]), float(space["pca_var"][1])) if use_pca else None

    # choose one grid key safely
    grid_keys  = [str(g) for g in space["lambda_grids"]]
    grid_key   = trial.suggest_categorical("qra_lambda_grid_key",  grid_keys)
    lambda_grid = list(ast.literal_eval(grid_key))

    it_ne, it_bs, it_lr, it_pat = None, None, None, None
    it_ne  = trial.suggest_categorical("qra_it_n_epochs",  list(space["it_n_epochs"]))
    it_bs  = trial.suggest_categorical("qra_it_batch_size",list(space["it_batch_size"]))
    it_lr  = 10.0 ** trial.suggest_float("qra_it_lr_log10", float(space["it_lr_log10"][0]), float(space["it_lr_log10"][1]))
    it_pat = trial.suggest_categorical("qra_it_patience",  list(space["it_patience"]))


    return dict(
        mc_nhits_samples_qra=int(S),
        sample_k=int(sample_k),
        subsample_stride=int(stride),
        quantiles=quants,
        use_pca=bool(use_pca),
        pca_var=float(pca_var) if pca_var is not None else None,
        lambda_grid=list(lambda_grid),
        it_n_epochs=it_ne,
        it_batch_size=it_bs,
        it_lr=it_lr,
        it_patience=it_pat,
    )


# ------------------------------- Objective ----------------------------------- #

def _objective(trial: optuna.Trial, cfg: DictConfig, base_dir: Path) -> Union[float, Tuple[float, ...]]:
    # Repro: vary seed per trial a bit
    pl.seed_everything(int(cfg.seed) + int(trial.number), workers=True)

    # Trial-specific run dir (looks like a regular training run)
    trial_dir = _trial_run_dir(base_dir, trial)
    dirs = prepare_train_run_dirs(trial_dir)

    metric_cfg = cfg.tune.metric
    if isinstance(metric_cfg, str):
        metric_names = [metric_cfg]
    elif isinstance(metric_cfg, (list, tuple, ListConfig)):
        metric_names = [str(x) for x in metric_cfg]
    else:
        raise TypeError(...)
    n_obj = len(metric_names)
    
    # Logging for the trial
    init_logging(
        log_dir=str(dirs["py"]),
        run_id=f"trial_{trial.number:04d}",
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False,
    )
    log = get_logger(f"tune_nhits_qra.trial{trial.number:04d}")

    # ----------------- feature selection -----------------
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

    # ----------------- DataModule -----------------
    train_path = Path(to_absolute_path(cfg.data.train_csv_path))
    test_path  = Path(to_absolute_path(cfg.data.test_csv_path))

    num_workers = int(cfg.train_nhits.num_workers)
    persistent_workers = bool(getattr(cfg.train_nhits, "persistent_workers", num_workers > 0))

    dm = NHITSQRADataModule(
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
        include_feature=list(getattr(cfg.features, "include", [])),
        exclude_feature=list(getattr(cfg.features, "exclude", [])),
        share_scalers=True,
    )
    dm.setup(stage="fit")
    
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

    # ----------------- lists from dataset (single source of truth) -----------------
    include_flags_ctx = bool(cfg.train_nhits.include_flags_ctx)
    ck_cols     = list(train_ds.ck_cols)
    cu_cols     = list(train_ds.cu_cols)
    static_cols = list(train_ds.static_cols)

    # decide if we actually add a single CU-known flag channel
    add_flag = include_flags_ctx and (len(cu_cols) > 0)
    if add_flag:
        log.info("Using single CU-known flag in hist_exog (since CU>0 and include_flags_ctx=True).")
    else:
        log.info("No CU-known flag in hist_exog (either CU=0 or include_flags_ctx=False).")

    # model-facing lists (what NHITS should be built for)
    hist_exog_list = ck_cols + cu_cols + (["__cu_known_flag"] if add_flag else [])
    futr_exog_list = ck_cols
    stat_exog_list = static_cols if bool(getattr(cfg.train_nhits, "include_static", True)) else []

    # Persist dataset scalers (from *train* split)
    feat_scalers = dm.train_dataset.get_scalers()
    targ_scaler  = dm.train_dataset.get_target_scaler()
    joblib.dump(feat_scalers, dirs["data"] / "feature_scalers.pkl")
    joblib.dump(targ_scaler,  dirs["data"] / "target_scaler.pkl")

    log.info("DM ready. ck=%d, cu=%d, static=%d",
            len(getattr(dm.train_dataset, "ck_cols", [])),
            len(getattr(dm.train_dataset, "cu_cols", [])),
            len(getattr(dm.train_dataset, "static_cols", [])))

    # ----------------- Sample hparams -----------------
    hps_nhits = _suggest_hparams_nhits(trial, cfg)
    hps_qra = _suggest_hparams_qra(trial, cfg)
    posterior_sampler = trial.suggest_categorical("posterior_sampler", ["swag", "mc_dropout"])
    use_swag = (posterior_sampler == "swag")
    if use_swag:
        hps_swag = _suggest_hparams_swag(trial, cfg)
    else:
        # freeze to harmless defaults so they don't appear as "real factors"
        hps_swag = dict(enabled=False, start_epoch=0, collect_every=1, max_rank=0, scale=1.0)

    log.info("Trial %d hparams: %s", trial.number, json.dumps(hps_nhits))
    log.info("Trial %d hparams: %s", trial.number, json.dumps(hps_qra))
    log.info("Trial %d hparams SWAG: %s", trial.number, json.dumps(hps_swag))

    # ----------------- Build model -----------------
    assert int(cfg.model.input_size) == int(cfg.data.context_length), "input_size must equal context_length"
    assert int(cfg.model.h) == int(cfg.data.forecast_horizon), "model.h must equal forecast_horizon"
    log.info(f"hist_exog_list: {hist_exog_list}")
    log.info(f"futr_exog_list: {futr_exog_list}")
    log.info(f"stat_exog_list: {stat_exog_list}")

    nhits = build_nhits_model(
    input_size=int(cfg.model.input_size),
    h=int(cfg.model.h),
    n_blocks=list(hps_nhits["n_blocks"]),
    mlp_units=hps_nhits["mlp_units"],
    dropout_prob_theta=float(hps_nhits["dropout_prob_theta"]),
    n_pool_kernel_size=list(hps_nhits["n_pool_kernel_size"]),
    n_freq_downsample=list(hps_nhits["n_freq_downsample"]),
    hist_exog_list=hist_exog_list,
    futr_exog_list=futr_exog_list,
    stat_exog_list=stat_exog_list,
    )

    # mutate a copy of cfg for QRA so run_qra reads tuned values
    cfg_trial = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_trial.train_qra.mc_nhits_samples_qra = hps_qra["mc_nhits_samples_qra"]
    cfg_trial.train_qra.sample_k             = hps_qra["sample_k"]
    cfg_trial.train_qra.subsample_stride     = hps_qra["subsample_stride"]
    cfg_trial.train_qra.quantiles        = hps_qra["quantiles"]
    cfg_trial.train_qra.use_pca              = hps_qra["use_pca"]
    cfg_trial.train_qra.pca_var              = hps_qra["pca_var"]
    cfg_trial.train_qra.lambda_grid          = hps_qra["lambda_grid"]
    cfg_trial.train_qra.n_epochs  = hps_qra["it_n_epochs"]
    cfg_trial.train_qra.batch_size= hps_qra["it_batch_size"]
    cfg_trial.train_qra.lr        = hps_qra["it_lr"]
    cfg_trial.train_qra.patience  = hps_qra["it_patience"]
    cfg_trial.train_qra.solver_loss = "iterative_pinball" 

    # --- SWAG mutation ---
    cfg_trial.train_nhits.swag.enabled       = bool(use_swag)
    cfg_trial.train_nhits.swag.start_epoch   = int(hps_swag["start_epoch"])
    cfg_trial.train_nhits.swag.collect_every = int(hps_swag["collect_every"])
    cfg_trial.train_nhits.swag.max_rank      = int(hps_swag["max_rank"])
    cfg_trial.train_nhits.swag.scale         = float(hps_swag["scale"])

    cfg_trial.train_qra.sampling_backend = "swag" if use_swag else "mc_dropout"

    module = NHITSForecasterModule(
    nhits_model=nhits,
    learning_rate=float(hps_nhits["lr"]),
    loss=str(getattr(cfg.train_nhits, "loss", "mae")),
    include_flags_ctx=add_flag,
    include_static=bool(getattr(cfg.train_nhits, "include_static", True)),
    hist_exog_list=hist_exog_list,
    futr_exog_list=futr_exog_list,
    stat_exog_list=stat_exog_list,
    warmup_epochs=hps_nhits["warmup_epochs"],
    swag_cfg=cfg_trial.train_nhits.swag,
    )

    # ----------------- Trainer + callbacks -----------------
    tb_logger = TensorBoardLogger(
        save_dir=str(trial_dir.parent / "logs_tb"),
        name="",
        version=trial_dir.name,
    )

    # monitor (for checkpointing/pruning within NHITS training)
    loss_name = str(getattr(cfg.train_nhits, "loss", "mae")).lower()
    monitor_key = "val_mse" if loss_name == "mse" else "val_mae"
    mode = "min"

    ckpt_cb = ModelCheckpoint(
        monitor=monitor_key,
        mode=mode,
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
        dirpath=str(dirs["ckpt"] / "nhits"),
        filename="best-{epoch:02d}-{" + monitor_key + ":.4f}",
    )

    # in case that nhits should be early stopped. Pruning from Optuna wouldn't make sense, as we care about
    # crps/es, not MAE, which we haven't yet assessed
    es_cb = None
    if bool(getattr(cfg.train_nhits, "early_stopping", True)):
        es_cb = EarlyStopping(
            monitor=monitor_key,
            mode="min",
            patience=int(hps_nhits.get("early_patience", getattr(cfg.train_nhits, "early_patience", 10))),
            min_delta=float(getattr(cfg.train_nhits, "early_min_delta", 0.0)),
            verbose=False,
            check_on_train_epoch_end=False,
        )

    accel, devices, precision, _ = pick_accelerator_and_devices()
    trainer = Trainer(
        max_epochs=int(cfg.train_nhits.n_epochs),
        accelerator=accel,
        devices=devices,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=float(cfg.train_nhits.gradient_clip_val),
        accumulate_grad_batches=int(cfg.train_nhits.accumulate_grad_batches),
        logger=tb_logger,
        callbacks=[c for c in [ckpt_cb, es_cb] if c is not None],
        num_sanity_val_steps=int(cfg.train_nhits.num_sanity_val_steps),
        log_every_n_steps=int(cfg.train_nhits.log_every_n_steps),
        default_root_dir=str(dirs["root"]),
        enable_progress_bar=bool(cfg.train_nhits.enable_progress_bar),
    )
    t0 = time.time()
    # ----------------- Fit NHITS (train/val only) -----------------
    trainer.fit(module, datamodule=dm)
    t_train = time.time() - t0

    # checkpoint sanity (we expect at least best-*.ckpt)
    if not any((dirs["ckpt"] / "nhits").glob("*.ckpt")):
        trial.set_user_attr("fail_reason", "no_ckpt")
        return float("inf") if n_obj == 1 else tuple([float("inf")] * n_obj)
    

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        log.info("Reloading best checkpoint: %s", best_ckpt)
        module = NHITSForecasterModule.load_from_checkpoint(
            checkpoint_path=best_ckpt,
            nhits_model=nhits,
            learning_rate=float(hps_nhits["lr"]),
            loss=cfg.train_nhits.loss,
            include_flags_ctx=add_flag,
            include_static=cfg.train_nhits.include_static,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            stat_exog_list=stat_exog_list,
            warmup_epochs=int(cfg.train_nhits.warmup_epochs),
            swag_cfg=cfg_trial.train_nhits.swag
        )

    t1 = time.time()
    qra_summary = run_qra(cfg_trial, module, dm, dirs, split="val", log=log)
    t_qra = time.time() - t1

    # ----------------- Pick the optimization target(s) -----------------
    metric_names_cfg = cfg.tune.metric

    if isinstance(metric_names_cfg, str):
        metric_names = [metric_names_cfg]
    elif isinstance(metric_names_cfg, (list, tuple, ListConfig)):
        metric_names = [str(x) for x in metric_names_cfg]
    else:
        raise TypeError(f"Unsupported type for cfg.tune.metric: {type(metric_names_cfg)}")

    # Pull metrics (default to +inf if missing)
    metrics = {name: float(qra_summary.get(name, np.inf)) for name in metric_names}

    # Save into Optuna trial metadata (browsable later)
    for k, v in metrics.items():
        trial.set_user_attr(k, float(v))

    # ----------------- Persist trial metrics -----------------
    (dirs["metrics"] / "qra" / "summary.json").write_text(json.dumps(qra_summary, indent=2))
    (dirs["metrics"] / "train_metrics.json").write_text(json.dumps(metrics, indent=2))

    manifest = {
    "trial": trial.number,
    "seed": int(cfg.seed) + int(trial.number),
    "nhits_hps": hps_nhits,
    "qra_hps": hps_qra,
    "metric_names": metric_names,
    "metrics": metrics,
    "best_ckpt": best_ckpt,
    "timing": {"train_s": t_train, "qra_s": t_qra},
    }
    (dirs["root"] / "trial_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ----------------- Return objective(s) -----------------
    if len(metric_names) == 1:
        return metrics[metric_names[0]]
    else:
        return tuple(metrics[n] for n in metric_names)


# --------------------------------- main -------------------------------------- #

@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_tune",    
)
def main(cfg: DictConfig):
    # Hydra chdirs to cfg.out_root; we’ll create per-trial folders under it
    base_dir = Path(".").resolve()
    (base_dir / "logs_py").mkdir(parents=True, exist_ok=True)

    # Light logging for the study (per-trial logs live in each trial folder)
    init_logging(
        log_dir=str(base_dir / "logs_py"),
        run_id="tuning",
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False,
    )
    log = get_logger("tune_nhits_qra.main")

    # Create or load study
    study = _make_study(cfg)

    target_total = int(cfg.tune.n_trials)
    already = len(study.trials)
    remaining = max(0, target_total - already)

    if remaining == 0:
        log.info("Study already has %d trials (>= target %d). Exiting.", already, target_total)
        return

    log.info("Study %s | storage=%s | directions=%s | metric=%s",
             cfg.tune.study_name, cfg.tune.storage, cfg.tune.direction, cfg.tune.metric)
    

    study.optimize(
        lambda tr: _objective(tr, cfg, base_dir),
        n_trials=remaining,
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

        pareto = [
        {"number": t.number, "values": list(t.values), "params": t.params, "user_attrs": t.user_attrs}
        for t in study.best_trials
        ]
    else:
        best = study.best_trial
        summary = {"best_value": best.value, "best_params": best.params, "best_trial_number": best.number}
        (base_dir / "best.json").write_text(json.dumps(summary, indent=2))

        log.info("Best trial #%d %s=%.6f", best.number, cfg.tune.metric, best.value)


if __name__ == "__main__":
    main()