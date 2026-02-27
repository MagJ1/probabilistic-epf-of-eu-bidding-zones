# src/models/chronosx/custom_pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

from contextlib import nullcontext
import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers import EarlyStoppingCallback

from transformers import GenerationConfig

# vendor pipeline (unchanged)
from chronosx.chronosx import ChronosXPipeline
from chronosx.utils.utils import compute_metrics, log_on_main, count_parameters
from chronosx.utils.prepare_covariates import prepare_covariates

import logging
logger = logging.getLogger(__name__)


class ChronosXPipelineMinimal(ChronosXPipeline):
    """
    Minimal extension of the vendor pipeline that ONLY customizes training
    (logging dir, progress bar, console noise). All modeling logic remains
    exactly the same.
    """

    def train(
        self,
        output_dir: Path,
        per_device_train_batch_size: int = 32,
        learning_rate: float = 1e-3,
        lr_scheduler_type: str = "linear",
        warmup_ratio: float = 0.0,
        optim: str = "adamw_torch_fused",
        log_steps: int = 50,
        save_steps: int = 200,
        max_steps: int = 5000,
        gradient_accumulation_steps: int = 2,
        dataloader_num_workers: int = 0,
        tf32: bool = True,
        torch_compile: int = 0,
        eval_steps: int = 200,
        per_device_eval_batch_size: int = 8,
        eval_accumulation_steps: int = 4,
        load_best_model_at_end: bool = True,
        save_total_limit: int = 3,
        quantized_train_dataset=None,
        quantized_val_dataset=None,
        seed: Optional[int] = None,
        # ------- our extra knobs (all optional) -------
        disable_tqdm: Optional[bool] = None,         # True -> hide HF progress bar
        report_to: Optional[str] = None,             # "tensorboard" | "none"
        run_name: Optional[str] = None,              # TB run name
        logging_dir_override: Optional[Path] = None, # TB log dir
        suppress_console_logs: bool = True,          # drop PrinterCallback spam

        early_stopping_enabled: bool = False,
        early_stopping_patience: int = 10,     # counts *evals*, not steps
        early_stopping_threshold: float = 0.0,
        save_only_model: bool = False,
        metric_for_best_model: str ="eval_loss",
        greater_is_better: bool = False,
        save_safetensors=True,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # default the extras to vendor behavior if not provided
        _disable_tqdm = bool(disable_tqdm) if disable_tqdm is not None else False
        _report_to = [] if (report_to in (None, "none")) else [report_to]
        _logging_dir = str(logging_dir_override or (output_dir / "logs"))
        _has_val = quantized_val_dataset is not None

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            optim=optim,
            logging_dir=_logging_dir,
            logging_strategy="steps",
            logging_steps=log_steps,
            save_strategy=("steps" if save_steps > 0 else "no"),
            save_steps=save_steps,
            report_to=_report_to,
            run_name=run_name,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_persistent_workers=False,
            dataloader_pin_memory=True,
            tf32=tf32,
            bf16=True,
            torch_compile=torch_compile,
            ddp_find_unused_parameters=False,
            evaluation_strategy=("steps" if quantized_val_dataset is not None else "no"),
            eval_steps=eval_steps,
            eval_accumulation_steps=eval_accumulation_steps,
            load_best_model_at_end=(load_best_model_at_end and _has_val),
            logging_first_step=True,
            save_total_limit=save_total_limit,
            disable_tqdm=_disable_tqdm,
            seed=seed,
            save_only_model=save_only_model,
            metric_for_best_model = metric_for_best_model,
            greater_is_better=greater_is_better,
            save_safetensors=save_safetensors,
        )

        # Trainer setup (same as vendor, but quiet)
        trainer = Trainer(
            model=self.chronosx,
            args=args,
            train_dataset=quantized_train_dataset,
            eval_dataset=quantized_val_dataset,
            compute_metrics=compute_metrics,
        )

        if early_stopping_enabled:
            if quantized_val_dataset is None:
                logger.warning(
                    "early_stopping_enabled=True but no validation dataset was provided; "
                    "disabling early stopping."
                )
            else:
                trainer.add_callback(
                    EarlyStoppingCallback(
                        early_stopping_patience=int(early_stopping_patience),
                        early_stopping_threshold=float(early_stopping_threshold),
                    )
                )

        # Quiet progress callback: keep the bar, suppress dict prints
        class QuietProgressCallback(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                # don't write logs dict to console
                return control

        # --- remove noisy callbacks ---
        try:
            trainer.remove_callback(PrinterCallback)
        except Exception:
            trainer.callback_handler.callbacks = [
                cb for cb in trainer.callback_handler.callbacks
                if not isinstance(cb, PrinterCallback)
            ]

        try:
            trainer.remove_callback(ProgressCallback)
        except Exception:
            trainer.callback_handler.callbacks = [
                cb for cb in trainer.callback_handler.callbacks
                if not isinstance(cb, ProgressCallback)
            ]

        # --- add  quiet version so the tqdm bar still shows ---
        trainer.add_callback(QuietProgressCallback())

        # Accept our custom batch dict keys (same as vendor)
        trainer._signature_columns = [
            "labels",
            "attention_mask",
            "input_ids",
            "past_covariates",
            "future_covariates",
            "decoder_input_ids",
        ]

        log_on_main("Training", logger)
        self.prepare_model_for_finetuning()

        # keep this quiet (vendor prints parameter_count)
        _ = count_parameters(self.chronosx)

        trainer.train()

        if quantized_val_dataset is not None:
            val_loss = trainer.evaluate(quantized_val_dataset)
        else:
            val_loss = {"eval_loss": np.nan}

        # --- save HF folder as before ---
        save_path = output_dir / "final-checkpoint"
        self.chronosx.save_pretrained(save_path)

        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # always save a portable, CPU state_dict
        state_cpu = {k: v.detach().cpu() for k, v in self.chronosx.state_dict().items()}

        # 1) raw state_dict (simple to load)
        torch.save(state_cpu, ckpt_dir / "last_state_dict.pt")

        # 2) PL-like wrapper filename (super_runner compatibility)
        torch.save({"state_dict": state_cpu}, ckpt_dir / "last.ckpt")

        # if trained with validation + load_best_model_at_end=True,
        # HF has already loaded the BEST weights into self.chronosx.
        # Save an alias with a "best" name too.
        if quantized_val_dataset is not None and load_best_model_at_end:
            torch.save(state_cpu, ckpt_dir / "best_state_dict.pt")
            torch.save({"state_dict": state_cpu}, ckpt_dir / "best.ckpt")

        return val_loss["eval_loss"]
    
    def predict(
        self,
        context: list[torch.Tensor],
        covariates: list[dict],
        num_samples: int = 20,
        *,                       # force keyword-only for the new knobs (safe defaults)
        sample_chunk: int = 16,   # how many samples to draw in parallel
        batch_chunk: int = 64,   # how many series to process in parallel
        use_bfloat16: bool = True,
    ):

        # ---- prepare context & tokenizer inputs (unchanged semantics) ----
        context_tensor = self._prepare_and_validate_context(context=context)
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)

        prepared_covariates = [prepare_covariates(entry) for entry in covariates]
        fut = np.asarray([e["future_covariates"] for e in prepared_covariates], dtype=np.float32, order="C")
        past = np.asarray([e["past_covariates"]   for e in prepared_covariates], dtype=np.float32, order="C")
        if fut.dtype == np.object_ or past.dtype == np.object_:
            raise ValueError("Covariate shapes are inconsistent; cannot stack.")

        future_covariates = torch.from_numpy(fut)
        past_covariates   = torch.from_numpy(past)

        device = self.chronosx.device
        prediction_length = self.prediction_length
        B = token_ids.size(0)
        S = int(num_samples)

        # output buffer (B, S, H) on CPU fp32 (matches return type)
        out_cpu = torch.empty((B, S, prediction_length), dtype=torch.float32, device="cpu")

        gen_cfg = GenerationConfig(
            min_new_tokens=prediction_length,
            max_new_tokens=prediction_length,
            do_sample=True,
            # num_return_sequences will be set per chunk
            eos_token_id=self.chronosx.config.eos_token_id,
            pad_token_id=self.chronosx.config.pad_token_id,
        )
        
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (use_bfloat16 and torch.cuda.is_available()) else nullcontext()

        self.chronosx.eval()
        with torch.inference_mode(), amp_ctx:
            # ---- micro-batch over series ----
            for b0 in range(0, B, batch_chunk):
                b1 = min(b0 + batch_chunk, B)

                # slice & move once per micro-batch
                ids_mb   = token_ids[b0:b1].to(device, non_blocking=True)
                mask_mb  = attention_mask[b0:b1].to(device, non_blocking=True)
                fut_mb   = future_covariates[b0:b1].to(device, non_blocking=True)
                past_mb  = past_covariates[b0:b1].to(device, non_blocking=True)

                # ---- chunk over samples (limits the B×S expansion) ----
                s_dst = 0
                for s0 in range(0, S, sample_chunk):
                    sc = min(sample_chunk, S - s0)

                    preds = self.chronosx.generate(
                        input_ids=ids_mb,
                        attention_mask=mask_mb,
                        generation_config=GenerationConfig(**{**gen_cfg.to_dict(), "num_return_sequences": sc}),
                        future_covariates=fut_mb,
                        past_covariates=past_mb,
                    )
                    preds = preds[..., 1:]                          # drop start token
                    preds = preds.reshape(ids_mb.size(0), sc, -1)   # (B_mb, sc, H)

                    # Run tokenizer’s output_transform on CPU to match centers’ device
                    preds = self.tokenizer.output_transform(
                        preds.to("cpu"),
                        scale[b0:b1].to("cpu"),
                    )
                    preds = preds.to(dtype=torch.float32, device="cpu")

                    out_cpu[b0:b1, s_dst:s_dst+sc, :] = preds
                    s_dst += sc

        return out_cpu