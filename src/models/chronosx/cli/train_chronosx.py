# src/models/chronosx/train_chronosx.py
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pytorch_lightning as pl
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
import math
# quiet Transformers console noise
from transformers.trainer_callback import PrinterCallback
PrinterCallback.on_log = lambda *args, **kwargs: None
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_default_handler()

import warnings
warnings.filterwarnings("ignore", module="transformers")

from models.chronosx.custom_pipeline import ChronosXPipelineMinimal

from utils.paths import prepare_train_run_dirs
from utils.logging_utils import init_logging, get_logger
from utils.feature_hash import feature_hash
from utils.run_ids import default_run_id, pick_run_id

from models.chronosx.dataset_loader import load_from_csv, _is_nullish, _load_single_dataset, _load_concatenated_dataset, _normalize_probs
from models.chronosx.custom_dataset import ChronosDatasetExt  # expects channel-index + mask_future args

from models.chronosx.utils.window_count import (
    estimate_total_windows_from_train_list,
    compute_steps_per_epoch,
    _get_world_size,
)

def _always_known(features_order: List[str], past_only_names: List[str]) -> List[str]:
    po = set(past_only_names)
    return [f for f in features_order if f not in po]


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_train",
)
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.seed), workers=True)

    # ----- features: strict CK -> CU order, fail fast if missing/invalid -----
    ck_cols = list(getattr(cfg.features, "ck_cols", []))
    cu_cols = list(getattr(cfg.features, "cu_cols", []))

    if ck_cols is None or cu_cols is None:
        raise ValueError("Both features.ck_cols and features.cu_cols must be defined.")

    # enforce disjointness & no duplicates
    inter = set(ck_cols).intersection(set(cu_cols))
    if inter:
        raise ValueError(f"Features cannot be both CK and CU: {sorted(inter)}")

    if len(set(ck_cols + cu_cols)) != len(ck_cols) + len(cu_cols):
        raise ValueError("Duplicate feature names detected across ck_cols/cu_cols.")

    # single source of truth used everywhere below
    features_order = ck_cols + cu_cols         # CK first, then CU
    past_only_names = cu_cols                  # CU == past only
    always_known = _always_known(features_order, past_only_names)  # == ck_cols

    runs_root = Path(to_absolute_path(cfg.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id_from_cfg = cfg.train.get("run_id", None)
    if run_id_from_cfg:
        run_id = str(run_id_from_cfg)
    else:
        # otherwise synthesize a hash-based default
        fh = feature_hash(ck_cols=always_known, cu_cols=past_only_names, static=None)
        rid = default_run_id(fh)
        run_id = pick_run_id(None, rid, runs_root)

    run_dir = runs_root / run_id

    # ----- dirs & logging -----
    dirs = prepare_train_run_dirs(run_dir)
    init_logging(
        log_dir=str(dirs["py"]),
        run_id=run_id,
        level=str(cfg.logging.log_level),
        coexist_with_hydra=True,
        unify_format=False,
    )
    log = get_logger("chronosx_train")
    log.info("Run dir: %s", run_dir)

    # ----- data (null-switch: single vs concat_dataset) -----
    fillna = bool(getattr(cfg.data, "fillna_forward", True))
    use_concat_dataset = _is_nullish(cfg.data.get("train_csv_path", None)) and _is_nullish(cfg.data.get("val_csv_path", None))

    
    if use_concat_dataset:
        train_list, val_list, probs, names = _load_concatenated_dataset(cfg.data, features_order, fillna)
        data_mode = "concat_dataset"
        log.info(f"Concatenated dataset is used.")
    else:
        train_list, val_list, probs, names = _load_single_dataset(cfg.data, features_order, fillna)
        data_mode = "single"
        log.info(f"Single dataset is used.")

    # persist resolved config (including derived order & mask list)
    cfg_persist = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_persist.features.features_order = features_order
    cfg_persist.features.past_only_names = past_only_names
    cfg_persist.data_resolved = {
        "mode": data_mode,
        "sources": [
            {
                "name": n,
                "prob": float(p),
            }
            for n, p in zip(names, probs)
        ],
    }
    (dirs["meta"] / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg_persist))
    (dirs["meta"] / "run_id.txt").write_text(run_id)
    (dirs["meta"] / "runs_root.txt").write_text(str(runs_root))

    log.info("Data mode: %s", data_mode)
    log.info("Loaded %d training source(s): %s",
            len(train_list), ", ".join(f"{n} (p={p:.3f})" for n, p in zip(names, probs)))
    if val_list is None:
        log.info("No validation source(s) provided.")
    else:
        log.info("Loaded %d validation source(s).", len(val_list))

    # ----- pipeline -----
    pred_len = int(cfg.data.prediction_length)
    num_covariates = len(features_order)

    pipeline = ChronosXPipelineMinimal(
        pretrained_model_name_or_path=str(cfg.model.pretrained_name_or_path),
        prediction_length=pred_len,
        num_covariates=num_covariates,
        covariate_injection=str(cfg.model.covariate_injection),
        device_map=str(cfg.model.device_map),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        layers_to_unfreeze=str(cfg.model.layers_to_unfreeze),
    )

    # indices of past-only channels (CU)
    past_only_idx = [features_order.index(n) for n in past_only_names]

    context_len = int(cfg.data.context_length)
    pred_len = int(cfg.data.prediction_length)

    stride = int(cfg.data.origin_stride_train)
    anchor_raw = cfg.data.origin_anchor_hour_train
    anchor_hour = None if anchor_raw in (None, "None") else int(anchor_raw)

    total_windows = estimate_total_windows_from_train_list(
        train_list,
        context_len=context_len,
        pred_len=pred_len,
        stride=stride,
        anchor_hour=anchor_hour,
    )

    world_size = _get_world_size()
    steps_per_epoch = compute_steps_per_epoch(
        total_windows=total_windows,
        per_device_bs=int(cfg.train.batch_size),
        grad_accum=int(cfg.train.grad_accum),
        world_size=world_size,
    )

    log.info(
        "Epoch geometry: total_windows=%d | world_size=%d | per_device_bs=%d | grad_accum=%d | steps_per_epoch=%d",
        total_windows, world_size, int(cfg.train.batch_size), int(cfg.train.grad_accum), steps_per_epoch
    )

    # ---- schedule: epochs OR max_steps ----
    assert len(train_list) > 0, "No training sources loaded"
    assert len(probs) == len(train_list), "probabilities length mismatch"
    assert np.isfinite(probs).all() and float(np.sum(probs)) > 0, "invalid training probabilities"

    epochs_cfg = getattr(cfg.train, "epochs", None)   # OmegaConf null -> None
    max_steps_cfg = int(getattr(cfg.train, "max_steps", -1))

    use_epoch_mode = epochs_cfg is not None

    if use_epoch_mode:
        epochs = int(epochs_cfg)
        max_steps = int(epochs * steps_per_epoch)
        log.info("Mode=epochs | epochs=%d -> max_steps=%d (derived)", epochs, max_steps)
    else:
        if max_steps_cfg <= 0:
            raise ValueError("epochs is null => max_steps must be > 0 for HF Trainer.")
        max_steps = int(max_steps_cfg)
        log.info("Mode=steps | max_steps=%d", max_steps)




    train_dataset = ChronosDatasetExt(
        datasets=train_list,
        probabilities=probs,
        tokenizer=pipeline.tokenizer,
        context_length=context_len,
        prediction_length=pred_len,
        mode="training",
        include_covariates=True,
        # realistic-mode knobs
        realistic_mode_enabled=bool(cfg.mask.enabled),
        realistic_mask_len=int(cfg.mask.len),
        realistic_apply_in=list(cfg.mask.apply_in),
        realistic_mask_value=np.nan if cfg.mask.value is None else float(cfg.mask.value),
        realistic_mask_channel_indices=past_only_idx,
        realistic_mask_future=bool(cfg.mask.mask_future),
        origin_enabled=bool(cfg.data.origin_enabled_train),
        origin_stride_hours=int(cfg.data.origin_stride_train),
        origin_anchor_hour=(None if cfg.data.origin_anchor_hour_train in (None, "None") else int(cfg.data.origin_anchor_hour_train)),
        origin_seed=int(cfg.seed),

    )


    val_dataset = None
    if not bool(cfg.model.eval_skip_pretrained):
        if val_list is not None:
            val_dataset = ChronosDatasetExt(
                datasets=val_list,           
                probabilities=_normalize_probs([1.0] * len(val_list), len(val_list)),
                tokenizer=pipeline.tokenizer,
                context_length=context_len,
                prediction_length=pred_len,
                mode="validation",
                include_covariates=True,
                realistic_mode_enabled=bool(cfg.mask.enabled),
                realistic_mask_len=int(cfg.mask.len),
                realistic_apply_in=list(cfg.mask.apply_in),
                realistic_mask_value=np.nan if cfg.mask.value is None else float(cfg.mask.value),
                realistic_mask_channel_indices=past_only_idx,
                realistic_mask_future=bool(cfg.mask.mask_future),
                origin_enabled=bool(cfg.data.origin_enabled_val),
                origin_stride_hours=int(cfg.data.origin_stride_val),
                origin_anchor_hour=(None if cfg.data.origin_anchor_hour_val in (None, "None") else int(cfg.data.origin_anchor_hour_val)),
                origin_seed=int(cfg.seed),
            )

    if bool(cfg.train.early_stopping_enabled) and bool(cfg.model.eval_skip_pretrained):
        log.warning(
            "early_stopping_enabled=True but eval_skip_pretrained=True -> no validation set. "
            "Early stopping will be disabled (HF needs eval metrics)."
        )


    _has_val = (val_dataset is not None) and (not bool(cfg.model.eval_skip_pretrained))
    eval_steps = steps_per_epoch if _has_val else 0   # 0 will map to "no" in pipeline
    save_steps = steps_per_epoch * 10                     # can still save once per epoch

    # ----- fine-tune -----
    eval_loss = pipeline.train(
        output_dir=dirs["train"],
        quantized_train_dataset=train_dataset,
        quantized_val_dataset=None if cfg.model.eval_skip_pretrained else val_dataset,
        per_device_train_batch_size=int(cfg.train.batch_size),
        per_device_eval_batch_size=int(cfg.train.eval_batch_size),
        learning_rate=float(cfg.train.learning_rate),
        lr_scheduler_type=str(cfg.train.scheduler),
        warmup_ratio=float(cfg.train.warmup_ratio),
        optim=str(cfg.train.optim),
        log_steps=int(cfg.train.log_steps),
        save_steps=save_steps,
        max_steps=max_steps,
        gradient_accumulation_steps=int(cfg.train.grad_accum),
        dataloader_num_workers=int(cfg.train.num_workers),
        eval_steps=eval_steps,
        eval_accumulation_steps=int(cfg.train.eval_accum_steps),
        tf32=True,
        torch_compile=0,
        load_best_model_at_end=True,
        save_total_limit=int(cfg.train.save_total_limit),
        seed=int(cfg.seed),
        # --- grad_accum extras ----
        save_only_model=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        disable_tqdm=not bool(cfg.train.enable_progress_bar),
        report_to=("tensorboard" if cfg.train.enable_tensorboard else "none"),
        run_name=run_id,
        logging_dir_override=(runs_root.parent / "logs_tb" / run_id),
        suppress_console_logs=False,
        early_stopping_enabled=bool(cfg.train.early_stopping_enabled) and not bool(cfg.model.eval_skip_pretrained),
        early_stopping_patience=int(cfg.train.early_stopping_patience),
        early_stopping_threshold=float(cfg.train.early_stopping_threshold),
    )

    # ----- metrics & flag -----
    (dirs["metrics"] / "train_metrics.json").write_text(
        f'{{"eval_loss": {float(eval_loss):.6f}}}\n'
    )
    (dirs["root"] / "_done.train").write_text("ok\n")

    print(f"[chronosx_train] Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()