from __future__ import annotations
from typing import List, Optional, Any, Dict
import torch
import pytorch_lightning as pl
import torch.nn as nn
from utils.logging_utils import get_logger
import logging
import math
from models.nhits_qra.nhits.swag import SWAGState, SWAGCallback

log = get_logger(__name__)
epoch_log = logging.getLogger("epoch_summaries")

class NHITSForecasterModule(pl.LightningModule):
    def __init__(self,
                 nhits_model,
                 learning_rate: float = 1e-3,
                 loss: str = "mae",
                 include_flags_ctx: bool = True,
                 include_static: bool = True,
                 hist_exog_list: Optional[List[str]] = None,
                 futr_exog_list: Optional[List[str]] = None,
                 stat_exog_list: Optional[List[str]] = None,
                 warmup_epochs: int = None, 
                 swag_cfg: Optional[Any] = None, 
                 **kwargs):
        super().__init__()
        self.model = nhits_model
        self.learning_rate = learning_rate
        self.include_flags_ctx = include_flags_ctx
        self.include_static = include_static

        self.hist_exog_list = list(hist_exog_list or [])
        self.futr_exog_list = list(futr_exog_list or [])
        self.stat_exog_list = list(stat_exog_list or [])
        self.warmup_epochs = warmup_epochs
        self.expect_hist = len(self.hist_exog_list)   # ck + cu
        self.expect_futr = len(self.futr_exog_list)   # ck
        self.expect_stat = len(self.stat_exog_list)   # static

        if loss.lower() == "mse":
            self.loss_fn = nn.MSELoss(); self.loss_name = "mse"
        elif loss.lower() == "mae":
            self.loss_fn = nn.L1Loss();  self.loss_name = "mae"
        else:
            raise ValueError(f"Unsupported loss: {loss}")
        
        # SWAG config + state
        self.swag_cfg = swag_cfg
        self._swag_enabled = bool(getattr(swag_cfg, "enabled", False)) if swag_cfg is not None else False
        self.swag: Optional[SWAGState] = None  # created by callback on_fit_start

    def _epoch_summary_msg(self, stage: str) -> str:
        """Build a one-line epoch summary for console/TensorBoard logging."""
        cm = self.trainer.callback_metrics  # aggregated (epoch) metrics
        
        # pick out keys
        tkey = f"train_{self.loss_name}"
        vkey = f"val_{self.loss_name}"
        train_loss = float(cm[tkey].item()) if tkey in cm else float("nan")
        val_loss   = float(cm[vkey].item()) if vkey in cm else float("nan")

        # current LR (first optimizer, common case)
        try:
            lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = float("nan")

        return (
            f"epoch={self.current_epoch:03d} "
            f"stage={stage:<5} "
            f"train_{self.loss_name}={train_loss:.6f} "
            f"val_{self.loss_name}={val_loss:.6f} "
            f"lr={lr:.3g}"
        )

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self):
        # Parameter groups: decay vs no-decay
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(key in n.lower() for key in ["bias", "norm", "bn", "layernorm", "ln"]):
                no_decay.append(p)
            else:
                decay.append(p)

        optim = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.01},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_epochs = self.trainer.max_epochs

        # Warmup + CosineAnnealingLR (no restarts)
        def lr_lambda(epoch: int):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # -------------------------
    # Lightning steps
    # -------------------------
    def training_step(self, batch, batch_idx):
        x, y = self._build_windows(batch)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"train_{self.loss_name}", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._build_windows(batch)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"val_{self.loss_name}", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self._build_windows(batch)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"test_{self.loss_name}", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    

    def on_validation_epoch_end(self):
        msg = self._epoch_summary_msg(stage="val")
        epoch_log.info(msg)
        

    # -------------------------
    # Batch -> NHITS windows
    # -------------------------
    def _build_windows(self, batch):
        """
        Expects batch keys:
        y_past: (B,C)
        y_future: (B,H)
        c_ctx_future_known: (B,C,C_ck)
        c_fct_future_known: (B,H,C_ck)
        c_ctx_future_unknown: (B,C,C_cu)
        flags_ctx_cu_known: (B,C,C_cu)
        static: (B,S)  [may be empty-length]

        Returns:
        windows_batch (dict), target (B,H,1)

        NHITS windows-mode expectations:
        - insample_y:      (B,C,1)
        - insample_mask:   (B,C,1)
        - hist_exog:       (B,C,C_hist)        # past-only
        - futr_exog:       (B,C+H,C_ck)        # past+future CK!
        - stat_exog:       (B,S)
        """
        # targets
        y_past   = batch["y_past"].float().unsqueeze(-1)    # (B,C,1)
        y_future = batch["y_future"].float().unsqueeze(-1)  # (B,H,1)
        B, C = y_past.shape[0], y_past.shape[1]
        H    = y_future.shape[1]

        # exogs
        ck_past = batch["c_ctx_future_known"].float()       # (B,C,C_ck)
        ck_fut  = batch["c_fct_future_known"].float()       # (B,H,C_ck)
        cu_past = batch["c_ctx_future_unknown"].float()     # (B,C,C_cu)
        flags   = batch["flags_ctx_cu_known"].float()       # (B,C,C_cu)

        # hist_exog: ctx-known + ctx-unknown (+ optional flags), time length C
        parts = [ck_past, cu_past]
        if self.include_flags_ctx and flags.numel() > 0:
            # one flag per time step: 1 if any CU feature is known, 0 otherwise
            global_flag = flags.amax(dim=-1, keepdim=True)   # (B, C, 1)
            parts.append(global_flag)

        hist_exog = torch.cat(parts, dim=-1) if parts else torch.empty(B, C, 0, device=y_past.device)

        # futr_exog: **must be full window (C+H)** for NHITS windows mode
        assert ck_past.shape[1] == C and ck_fut.shape[1] == H
        futr_exog = torch.cat([ck_past, ck_fut], dim=1)     # (B, C+H, C_ck)

        # stat_exog
        if self.include_static and ("static" in batch):
            stat_exog = batch["static"].float()
            if stat_exog.ndim == 1:
                stat_exog = stat_exog.unsqueeze(0)
        else:
            stat_exog = torch.empty(B, 0, device=y_past.device)

        # sanity: channel counts (time length differs by design)
        assert hist_exog.shape[-1] == self.expect_hist, \
            f"hist_exog channels {hist_exog.shape[-1]} < expected {self.expect_hist}"
        assert futr_exog.shape[-1] == self.expect_futr, \
            f"futr_exog channels {futr_exog.shape[-1]} != expected {self.expect_futr}"
        assert stat_exog.shape[-1] == self.expect_stat, \
            f"stat_exog channels {stat_exog.shape[-1]} != expected {self.expect_stat}"

        windows = {
            "insample_y":    y_past,                        # (B,C,1)
            "insample_mask": torch.ones_like(y_past),       # (B,C,1)
            "hist_exog":     hist_exog,                     # (B,C, C_hist)
            "futr_exog":     futr_exog,                     # (B,C+H, C_ck)  <-- key fix
            "stat_exog":     stat_exog,                     # (B,S)
        }
        return windows, y_future                             # (B,H,1)
    

    def configure_callbacks(self):
        if not self._swag_enabled:
            return []

        return [SWAGCallback(
            start_epoch=int(getattr(self.swag_cfg, "start_epoch", 2)),
            collect_every=int(getattr(self.swag_cfg, "collect_every", 1)),
            max_rank=int(getattr(self.swag_cfg, "max_rank", 20)),
        )]
    

    

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        swag = getattr(self, "swag", None)
        if swag is None or swag.mean is None or swag.n_snapshots < 1:
            return

        dev = None
        if swag.deviations:
            dev = torch.stack([d.detach().cpu() for d in swag.deviations], dim=0)  # (K, P)

        checkpoint["swag_state"] = {
            "max_rank": int(swag.max_rank),
            "n_snapshots": int(swag.n_snapshots),
            "mean": swag.mean.detach().cpu(),
            "sq_mean": swag.sq_mean.detach().cpu(),
            "deviations": dev,  # tensor or None
        }

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        sd = checkpoint.get("swag_state", None)
        if sd is None:
            return

        swag = SWAGState(max_rank=int(sd["max_rank"]))
        swag.n_snapshots = int(sd["n_snapshots"])
        swag.mean = sd["mean"]
        swag.sq_mean = sd["sq_mean"]

        dev = sd.get("deviations", None)
        if dev is None:
            swag.deviations = []
        else:
            # store as list of (P,) tensors on CPU like SWAGState expects
            swag.deviations = [dev[i].detach().cpu() for i in range(dev.shape[0])]

        self.swag = swag