from typing import Literal, Any, Dict, Mapping, Tuple
from models.normalizing_flows.ece_accumulator import ECEAccumulator
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, CosineAnnealingLR, SequentialLR
import pandas as pd
import os
from pathlib import Path
import numpy as np
from scipy.stats import norm
import math
from properscoring import crps_ensemble
import logging
from utils.metrics import crps_terms_fast, pits_from_samples, berkowitz_lr_test, sliced_energy_score

from utils.logging_utils import get_logger

log = get_logger(__name__)
epoch_log = logging.getLogger("epoch_summaries")

class FlowForecasterModule(pl.LightningModule):
    """Deals with training of the normalizing flows model"""
    def __init__(self, 
                 model, 
                 scalers, 
                 lr: float = 1e-4, 
                 warmup_epochs=1,
                 loss_metric: Literal["es","crps","nll"] = "es", 
                 n_samples_loss: int = 256,
                 beta: float = 1,
                 k_slices: int = 128,
                 eval_metrics: tuple[str] = ("crps", "nll"),
                 n_samples_eval: int = 256,
                 y_col: str = "y",
                 detail_crps: str = None,
                 detail_es: str = None, 
                 store_series: bool = False,
                 ece_taus: list =None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.scalers = scalers
        self.flow_frozen= True
        self.warmup_epochs = warmup_epochs
        self.loss_metric = loss_metric
        self.n_samples_loss = n_samples_loss
        self.y_col = y_col
        self.beta = beta
        self.k_slices = k_slices
        self.ece_taus = ece_taus or [10,20,30,40,50,60,70,80,90]

        
        self.detail_crps = (detail_crps or "none").lower()
        self.detail_es   = (detail_es or "none").lower()
        assert self.detail_crps in ("none", "per_horizon")
        assert self.detail_es   in ("none", "per_sample")
        self.store_series = store_series
        
        self.eval_metrics = tuple(eval_metrics)
        self.n_samples_eval = n_samples_eval

        self._val_pits = []
        self._test_pits = []

        """Constructor
        Args:
            model: Normalizing flows model
            scaler: Scaler used in the data_module setup, important to calculate correct crps_metric
            max_epochs: Maximum number of epochs the model trains
            steps_per_epoch: Number of steps per epoch
            lr: Learning rate
            use_crps_metric: Whether model should be evaluated additionally on the crps metric, next to negative log likelihood
            n_crps_samples: How many samples/ scenarios the model generates to estimate the crps score
        """

        if self.loss_metric not in ("es", "crps", "nll"):
            raise ValueError(f"Loss {self.loss_metric} should be either energy score (ES) or Continuous Ranked Probability Score (CRPS). Negative Log-Likelihood (NLL) is not recommended.")

        if self.scalers is None and self.loss_metric:
            raise ValueError("Scaler is needed for CRPS/ES computation.")
        
        if self.loss_metric and self.y_col not in self.scalers:
            raise KeyError(f"y_col '{self.y_col}' not found in scalers dict.")
        
    def training_step(self, batch: dict, _) -> float:

        if log.isEnabledFor(logging.DEBUG):
            if self.global_step % 100 == 0:
                y_min = batch["y_future"].min().detach()
                y_max = batch["y_future"].max().detach()
                # avoid tensor-of-size-2 in log; log as scalars or print
                self.log("y_future_min", y_min, prog_bar=False)
                self.log("y_future_max", y_max, prog_bar=False)

        cache: dict[str, tuple[torch.Tensor, dict]] = {}

        # 1) Primary loss (with grad where needed)
        reg = self._metric_registry()
        if self.loss_metric == "crps":
            loss, extras = reg["crps"](batch, n_samples=self.n_samples_loss, track_grad=True)
            cache["crps"] = (loss, extras)

        elif self.loss_metric == "es":
            loss, extras = reg["es"](batch, n_samples=self.n_samples_loss, track_grad=True, beta=self.beta)
            cache["es"] = (loss, extras)

        elif self.loss_metric == "nll":
            loss, extras = reg["nll"](batch, teacher_force=True)  # no n_samples/track_grad needed
            cache["nll"] = (loss, extras)
        else:
            raise ValueError(f"Unknown loss_metric: {self.loss_metric}")

        self.log(f"train_{self.loss_metric}", loss, logger=True, on_epoch=True, on_step=True, prog_bar=True)
        for k, v in extras.items():
            is_scalar = isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)
            if is_scalar:
                self.log(f"train_{self.loss_metric}_{k}", v, logger=True, on_epoch=True, on_step=False)

        # 2) Log extra metrics from eval list, without grads, using eval samples
        eval_set_minus_primary = set(self.eval_metrics) - {self.loss_metric}
        _, _ = self._evaluate_and_log(
            batch,
            metrics=eval_set_minus_primary,
            prefix="train_",
            default_kwargs={"n_samples": self.n_samples_eval, "track_grad": False},
            metric_kwargs={"es": {"beta": self.beta}},   # only ES needs beta
            cache=cache,
        )
        if not torch.isfinite(loss).all():
            raise ValueError(f"Loss ({self.loss_metric}) produced non-finite values")

        self._debug_standardization(batch, tag="train_step")
        return loss
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # run only in DEBUG and only for the very first training batch
        if not log.isEnabledFor(logging.DEBUG) or self.global_step != 0:
            return

        with torch.no_grad():
            ctx0 = self.model.make_ctx(batch, teacher_force=True)
            lp0  = self.model.flow.log_prob(batch["y_future"], context=ctx0)
            all_finite = torch.isfinite(lp0).all().item()

        log.debug("init log_prob finite? %s", all_finite)
        

    def on_train_epoch_end(self):
        # log the union of the eval metrics + the training loss
        names = set(self.eval_metrics) | {self.loss_metric}
        msg = (
            f"epoch={self.current_epoch}  "
            f"{self._fmt_metric_block('train', names)}  "
            f"{self._fmt_metric_block('val',   names)}"
        )
        epoch_log.info(msg)

    def on_validation_epoch_start(self):
        if "ece" in self.eval_metrics:
            taus = torch.tensor(self.ece_taus, dtype=torch.float64, device="cpu")
            self._ece_val = ECEAccumulator(taus=taus, H=self.model.forecast_horizon)
        else:
            self._ece_val = None
    
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        metrics_to_run = set(self.eval_metrics) | {self.loss_metric}

        # One sampling pass for EVERYTHING in val/test
        S = max(self.n_samples_eval, 256)

        with torch.no_grad():
            samples, _ = self.model.sample(batch, n_per_series=S, track_grad=False)  # (S,B,H)

        self._evaluate_and_log(
            batch,
            metrics=metrics_to_run,
            prefix="val_",
            default_kwargs={
                "n_samples": S,
                "track_grad": False,
                "samples": samples,     
            },
            metric_kwargs={"es": {"beta": self.beta}},
            prog_first=True,
        )

        # PIT + ECE reuse the same samples (already no-grad)
        pits = self._collect_pits(batch, samples)
        self._val_pits.append(pits)

        if self._ece_val is not None:
            self._ece_val.update(samples=samples, y_true=batch["y_future"])

        self._debug_standardization(batch, tag="val_step")
        return

    def on_validation_epoch_end(self):
        if self._val_pits:
            U = np.concatenate(self._val_pits, axis=0)  # (T,B,H)-> here just stack batches, so N = steps*B PITS: (N,H)
            U = np.clip(U, 1e-6, 1 - 1e-6)
            Z = np.clip(norm.ppf(U), -6.0, 6.0)                              # (N,H)

            # Per-hour Berkowitz, PITs are collected per hour/ forecast, so e.g. PIT of hour +1 are grouped and tested
            for h in range(Z.shape[1]):
                res = berkowitz_lr_test(Z[:, h])
                # log a few: p-value and LR
                self.log(f"berkowitz_p_h{h:02d}", res["p"], prog_bar=False, on_epoch=True, logger=True)
            # Pooled (flatten across hours)
            res_pool = berkowitz_lr_test(Z.reshape(-1))
            self.log("berkowitz_p_pooled", res_pool["p"], prog_bar=True, on_epoch=True, logger=True)

        self._val_pits.clear()

        if self._ece_val is None:
            return
        mean_ece, ece_per_h, cover = self._ece_val.compute()
        self.log("val_ece", mean_ece, prog_bar=False, on_epoch=True, sync_dist=True)
        # store detail for runner-side saving if you want
        self.last_val_ece_detail = {
            "ece_mean": mean_ece,
            "ece_per_h": ece_per_h,
            "taus": list(map(float, self.ece_taus)),
            "n_per_h": [float(x) for x in self._ece_val.ns.tolist()],
        }



    def test_step(self, batch: dict, _):
        metrics_to_run = set(self.eval_metrics) | {self.loss_metric}
        cache = {}
         # Sample once for all sample-based metrics + PIT + ECE
        S = max(self.n_samples_eval, 256)

        with torch.no_grad():
            samples, _ = self.model.sample(batch, n_per_series=S, track_grad=False)  # (S,B,H)

        self._evaluate_and_log(
            batch,
            metrics=metrics_to_run,
            prefix="test_",
            default_kwargs={"n_samples": S, "track_grad": False, "samples": samples},
            metric_kwargs={"es": {"beta": self.beta}},
            cache=cache
        )

        if self.store_series:
            if "crps" in cache and "_series" in cache["crps"][1]:
                self._stash_series(batch, cache["crps"][1]["_series"], target="crps")
            if "es" in cache and "_series" in cache["es"][1]:
                self._stash_series(batch, cache["es"][1]["_series"],   target="es")

        if self.detail_crps == "per_horizon" and "crps" in cache and "_bh" in cache["crps"][1]:
            self._stash_bh_or_bk(batch, cache["crps"][1]["_bh"], target="crps")

        if self.detail_es == "per_sample" and "es" in cache and "_bk" in cache["es"][1]:
            self._stash_bh_or_bk(batch, cache["es"][1]["_bk"], target="es")

        pits = self._collect_pits(batch, samples)
        self._test_pits.append(pits)  # list of (B,H)

        if self._ece_test is not None:
            self._ece_test.update(samples=samples, y_true=batch["y_future"])
    

    def on_test_epoch_start(self):
        # runtime buffers (filled during test)
        self._crps_buf = []   # detailed per-horizon CRPS rows (vectorized chunks)
        self._es_buf   = []   # detailed per-horizon ES rows (via CRPS equivalence)

        # DM-ready per-origin scalar series
        self._series_buf = {"crps": [], "es": []}  # list of (uid[B], ds[B], values[B]) tuples

        if "ece" in self.eval_metrics:
            taus = torch.tensor(self.ece_taus, dtype=torch.float64, device="cpu")
            self._ece_test = ECEAccumulator(taus=taus, H=self.model.forecast_horizon)
        else:
            self._ece_test = None

    def on_test_epoch_end(self):

        # ---------- build detailed frames ----------
        if self.detail_crps == "per_horizon" and self._crps_buf:
            cols = list(zip(*self._crps_buf))
            arrs = [np.concatenate(c) for c in cols]
            self.test_crps_df = pd.DataFrame({
                "global_step": arrs[0],
                "unique_id":   arrs[1],
                "batch_idx":   arrs[2],
                "ds":          arrs[3],      # ORIGIN t0
                "horizon_idx": arrs[4],      # 0..H-1
                "crps":        arrs[5],
                "crps_mean_over_horizon": arrs[6],
            })

        if self.detail_es == "per_sample" and self._es_buf: 
            cols = list(zip(*self._es_buf))
            arrs = [np.concatenate(c) for c in cols]
            self.test_es_df = pd.DataFrame({
                "global_step": arrs[0],
                "unique_id":   arrs[1],
                "batch_idx":   arrs[2],
                "ds":          arrs[3],      # ORIGIN t0
                "sample_idx":  arrs[4],      # K-slice index
                "es":          arrs[5],
                "es_mean_over_k": arrs[6],
            })

        # ---------- per-origin series (scalar) ----------
        if self.store_series and self._series_buf["crps"]:
            uid, ds, vals = [np.concatenate(col) for col in zip(*self._series_buf["crps"])]
            self.crps_series_df = pd.DataFrame({"unique_id": uid, "ds": ds, "crps_series": vals})

        if self.store_series and self._series_buf["es"]:
            uid, ds, vals = [np.concatenate(col) for col in zip(*self._series_buf["es"])]
            self.es_series_df = pd.DataFrame({"unique_id": uid, "ds": ds, "es_series": vals})

        # ---------- normalize times & construct per-origin ----------
        # CRPS: origin_ds + optional target_ds
        if hasattr(self, "test_crps_df"):
            self.crps_det = self.test_crps_df.copy()
            self.crps_det["origin_ds"] = pd.to_datetime(self.crps_det["ds"]).dt.floor("h")
            self.crps_det["target_ds"] = (
                self.crps_det["origin_ds"]
                + pd.to_timedelta(self.crps_det["horizon_idx"].astype(int), unit="h")
            )
            self.crps_det = self.crps_det.sort_values(
                ["unique_id", "origin_ds", "horizon_idx"], kind="mergesort"
            ).reset_index(drop=True)

            self.crps_po = (
                self.crps_det
                .groupby(["unique_id", "origin_ds"], as_index=False)
                .agg(crps_mean=("crps", "mean"),
                    n_horizons=("horizon_idx", "nunique"))
                .sort_values(["unique_id", "origin_ds"], kind="mergesort")
                .reset_index(drop=True)
            )
        elif hasattr(self, "crps_series_df"):
            tmp = self.crps_series_df.copy()
            tmp["origin_ds"] = pd.to_datetime(tmp["ds"]).dt.floor("h")
            self.crps_po = (
                tmp[["unique_id", "origin_ds", "crps_series"]]
                .rename(columns={"crps_series": "crps_mean"})
                .sort_values(["unique_id", "origin_ds"], kind="mergesort")
                .reset_index(drop=True)
            )

        # ES: origin_ds only; detailed is per-slice K (no horizon)
        if hasattr(self, "test_es_df"):
            self.es_det = self.test_es_df.copy()
            self.es_det["origin_ds"] = pd.to_datetime(self.es_det["ds"]).dt.floor("h")
            self.es_det = self.es_det.sort_values(
                ["unique_id", "origin_ds", "sample_idx"], kind="mergesort"
            ).reset_index(drop=True)

            # If es_mean_over_k is already attached per origin row, keep it.
            if "es_mean_over_k" in self.es_det.columns:
                self.es_po = (
                    self.es_det
                    .drop_duplicates(subset=["unique_id", "origin_ds"])
                    .loc[:, ["unique_id", "origin_ds", "es_mean_over_k"]]
                    .rename(columns={"es_mean_over_k": "es_mean"})
                    .sort_values(["unique_id", "origin_ds"], kind="mergesort")
                    .reset_index(drop=True)
                )
            else:
                self.es_po = (
                    self.es_det
                    .groupby(["unique_id", "origin_ds"], as_index=False)
                    .agg(es_mean=("es", "mean"),
                        n_samples=("sample_idx", "nunique"))
                    .sort_values(["unique_id", "origin_ds"], kind="mergesort")
                    .reset_index(drop=True)
                )
        elif hasattr(self, "es_series_df"):
            tmp = self.es_series_df.copy()
            tmp["origin_ds"] = pd.to_datetime(tmp["ds"]).dt.floor("h")
            self.es_po = (
                tmp[["unique_id", "origin_ds", "es_series"]]
                .rename(columns={"es_series": "es_mean"})
                .sort_values(["unique_id", "origin_ds"], kind="mergesort")
                .reset_index(drop=True)
            )

        # ---------- summary + PIT ----------
        names = set(self.eval_metrics) | {self.loss_metric}
        epoch_log.info(f"(test) {self._fmt_metric_block('test', names)}")

        if self._test_pits:
            U = np.concatenate(self._test_pits, axis=0)
            U = np.clip(U, 1e-6, 1 - 1e-6)
            Z = np.clip(norm.ppf(U), -6.0, 6.0)
            for h in range(Z.shape[1]):
                res = berkowitz_lr_test(Z[:, h])
                self.log(f"berkowitz_p_h{h:02d}", res["p"], prog_bar=False, on_epoch=True, logger=True)
            res_pool = berkowitz_lr_test(Z.reshape(-1))
            self.log("berkowitz_p_pooled", res_pool["p"], prog_bar=True, on_epoch=True, logger=True)

        self._test_pits.clear()

        if self._ece_test is None:
            return
        mean_ece, ece_per_h, cover = self._ece_test.compute()
        self.log("test_ece", mean_ece, prog_bar=False, on_epoch=True, sync_dist=True)
        self.last_test_ece_detail = {
            "ece_mean": mean_ece,
            "ece_per_h": ece_per_h,
            "taus": list(map(float, self.ece_taus)),
            "n_per_h": [float(x) for x in self._ece_test.ns.tolist()],
        }

    def _as_csv_friendly(self, df: pd.DataFrame, cols=("ds","origin_ds","target_ds")) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return out

    def export_predictions(self, pred_dir: Path, write=("parquet","csv")) -> None:
        pred_dir = Path(pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "es_po"):
            if "parquet" in write: self.es_po.to_parquet(pred_dir / "es_per_origin.parquet", index=False)
            if "csv"     in write: self.es_po.to_csv(pred_dir / "es_per_origin.csv", index=False)

        if hasattr(self, "crps_po"):
            if "parquet" in write: self.crps_po.to_parquet(pred_dir / "crps_per_origin.parquet", index=False)

        if hasattr(self, "es_det"):
            if "parquet" in write: self.es_det.to_parquet(pred_dir / "es_detailed.parquet", index=False)
            if "csv"     in write:
                es_det_csv = getattr(self, "es_det_csv", self._as_csv_friendly(self.es_det, cols=("ds","origin_ds")))
                es_det_csv.to_csv(pred_dir / "es_detailed.csv", index=False)

        if hasattr(self, "crps_det"):
            if "parquet" in write: self.crps_det.to_parquet(pred_dir / "crps_detailed.parquet", index=False)
            if "csv"     in write:
                crps_det_csv = getattr(self, "crps_det_csv", self._as_csv_friendly(self.crps_det, cols=("ds","origin_ds","target_ds")))
                crps_det_csv.to_csv(pred_dir / "crps_detailed.csv", index=False)

    def _metric_registry(self):
        # Each fn returns (score, extras_dict) where extras_dict can contain fit/spread
        def _nll(batch, **kw):
            # only needs teacher_force
            tf = kw.get("teacher_force", False)
            ctx = self.model.make_ctx(batch, teacher_force=tf)
            y_fut, ctx = self._prevent_extrem_values(batch["y_future"], ctx)
            score = -self.model.flow.log_prob(y_fut, context=ctx).mean()
            return score, {}

        def _crps(batch, **kw):
            # needs n_samples, track_grad
            samples = kw.get("samples", None)
            n_samples  = kw.get("n_samples", 256)
            track_grad = kw.get("track_grad", False)
            score, crps_bh, fit, spread = self._crps_loss(
                batch, 
                n_samples=n_samples, 
                track_grad=track_grad,
                samples=samples,
            )
            return score, {
                "fit": fit.mean(), 
                "spread": spread.mean(), 
                "_bh": crps_bh.detach(),                    # (B,H)
                "_series": crps_bh.mean(dim=1).detach()     # (B,)
                }

        def _es(batch, **kw):
            # needs n_samples, track_grad, beta
            samples = kw.get("samples", None)
            n_samples  = kw.get("n_samples", 256)
            track_grad = kw.get("track_grad", False)
            beta       = kw.get("beta", 1.0)
            score, es_bk, fit, spread = self._es_loss(
                batch, n_samples=n_samples, K=self.k_slices, beta=beta, track_grad=track_grad,samples=samples,
            )
            return score, {
                "fit": fit.mean(), 
                "spread": spread.mean(), 
                "_bk": es_bk.detach(),                    # (B,K)
                "_series": es_bk.mean(dim=1).detach()      # (B,)
                }

        return {"nll": _nll, "crps": _crps, "es": _es}
    
    # Utility to evaluate only selected metrics
    def _evaluate_and_log(
        self,
        batch,
        metrics: set[str],
        *,
        prefix: str,                               # "train_", "val_", "test_"
        default_kwargs: Mapping[str, Any] | None = None,   # shared defaults
        metric_kwargs:  Mapping[str, Dict[str, Any]] | None = None,  # per-metric
        prog_first: bool = False,
        cache: Dict[str, Tuple[torch.Tensor, Dict]] | None = None,
    ):
        fns = self._metric_registry()
        cache = {} if cache is None else cache
        default_kwargs = dict(default_kwargs or {})
        metric_kwargs  = dict(metric_kwargs  or {})

        # sensible default: teacher_force on during training
        default_kwargs.setdefault("teacher_force", prefix.startswith("train_"))

        first = True
        last_score = None

        for name in sorted(metrics):
            fn = fns.get(name)
            if fn is None:
                continue
            if name in cache:
                score, extras = cache[name]
            else:
                kw = {**default_kwargs, **metric_kwargs.get(name, {})}
                score, extras = fn(batch, **kw)
                cache[name] = (score, extras)

            self.log(f"{prefix}{name}", score, logger=True, on_epoch=True, on_step=False,
                    prog_bar=(prog_first and first))
            
            for k, v in extras.items():
                is_scalar = isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)
                if is_scalar:
                    self.log(f"{prefix}{name}_{k}", v, logger=True, on_epoch=True, on_step=False)

            last_score = score
            first = False

        return last_score, cache

    def _evaluate_crps_with_lib(self, samples: torch.Tensor, y_true: torch.Tensor):
        y_true = y_true.detach().cpu().numpy()              # (B,F)
        samples = samples.detach().cpu().numpy().transpose(1, 2, 0)  # (B,F,S)
        crps_scores = crps_ensemble(y_true, samples)
        crps_mean = float(crps_scores.mean())
        return crps_mean

    def _crps_loss(
        self,
        batch: dict,
        n_samples: int | None = None,
        samples: torch.Tensor | None = None,
        unscale_to_physical: bool = True,
        track_grad: bool = False
    ) -> torch.Tensor:
        """
        Differentiable CRPS (U-CRPS) loss using the sample-based identity:
            CRPS ≈ (1/S)∑|Y_i - y|  -  (1/(2 S^2))∑∑|Y_i - Y_j|
            computed per hour and averaged over the horizon (and batch).
        Returns:
            torch scalar tensor (requires grad) suitable as a training loss.
        """
        S = int(n_samples or self.n_samples_loss)

        # Use provided samples or draw once
        if samples is None:
            samples, _ = self.model.sample(batch, n_per_series=S, track_grad=track_grad)  # (S,B,H)
        else:
            # sanity (optional but helpful)
            if samples.dim() != 3:
                raise ValueError(f"samples must be (S,B,H), got {tuple(samples.shape)}")
            # If caller passed a different S, we just accept it and override S for consistency
            S = int(samples.shape[0])

        y_true = batch["y_future"]  # (B,H)

        # Optionally unscale to physical units (linear transforms keep gradients)
        if unscale_to_physical:
            if self.scalers is None or self.y_col not in self.scalers:
                log.warning("Scaler missing for '%s'; computing CRPS in scaled units.", self.y_col)
            else:
                scaler = self.scalers[self.y_col]
                if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    # sklearn StandardScaler
                    mean = torch.as_tensor(scaler.mean_.reshape(1, 1),
                                        device=y_true.device, dtype=y_true.dtype)
                    scale = torch.as_tensor(scaler.scale_.reshape(1, 1),
                                            device=y_true.device, dtype=y_true.dtype)
                    y_true  = y_true  * scale + mean
                    samples = samples * scale + mean
                elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                    # sklearn MinMaxScaler: x_scaled = (x - min)/(max-min)
                    data_min = torch.as_tensor(scaler.data_min_.reshape(1, 1),
                                            device=y_true.device, dtype=y_true.dtype)
                    data_max = torch.as_tensor(scaler.data_max_.reshape(1, 1),
                                            device=y_true.device, dtype=y_true.dtype)
                    rng = (data_max - data_min).clamp_min(1e-12)
                    y_true  = y_true  * rng + data_min
                    samples = samples * rng + data_min
                else:
                    log.warning("Unknown scaler type for '%s'; computing CRPS in scaled units.", self.y_col)
        
        fit, spread = crps_terms_fast(samples, y_true)
        crps_bh = fit - spread # (B, H)
        crps_b = crps_bh.mean(dim=1)  # (B,)
        crps_mean = crps_b.mean()

        if log.isEnabledFor(logging.DEBUG):
            crps_lib = self._evaluate_crps_with_lib(samples.detach(), y_true.detach())
            log.debug("CRPS_mean_grad: %.6f  CRPS_mean_lib: %.6f", float(crps_mean), crps_lib)
        return crps_mean, crps_bh, fit, spread
    
    def _es_loss(self,
                batch: dict,
                n_samples: int | None = None,
                samples: torch.Tensor | None = None,
                K: int = 64,
                beta: float = 1.0,
                unscale_to_physical: bool = True,
                track_grad: bool = False,
                pair_subsample: int | None = None):
        """
        Sliced Energy Score with power beta in (0,2].
        - beta=1: classic ES (CRPS in 1D) -> fast order-stats path
        - beta=2: variance trick for spread
        - else: pairwise (O(S^2)) or subsampled pairs via `pair_subsample`
        """
        S = int(n_samples or self.n_samples_loss)

        # Use provided samples or draw once
        if samples is None:
            Y, _ = self.model.sample(batch, n_per_series=S, track_grad=track_grad)  # (S,B,H)
        else:
            if samples.dim() != 3:
                raise ValueError(f"samples must be (S,B,H), got {tuple(samples.shape)}")
            Y = samples
            S = int(Y.shape[0])

        y = batch["y_future"]  # (B,H)

        if unscale_to_physical and self.y_col in self.scalers:
            sc = self.scalers[self.y_col]
            if hasattr(sc, "mean_") and hasattr(sc, "scale_"):
                mean  = torch.as_tensor(sc.mean_.reshape(1,1),  device=y.device, dtype=y.dtype)
                scale = torch.as_tensor(sc.scale_.reshape(1,1), device=y.device, dtype=y.dtype)
                y  = y  * scale + mean
                Y  = Y  * scale + mean

        es_mean, es_bk, fit, spread, _w = sliced_energy_score(
            Y, y,
            beta=beta,
            K=K,
            pair_subsample=pair_subsample,
            use_fast_for_beta1=True,
            return_bk=True,
        )
        return es_mean, es_bk, fit, spread
    
    # add a small helper inside the module
    def _collect_pits(self, batch, samples):
        # samples: (S,B,H) torch, y_true: (B,H) torch
        samples_np = samples.detach().cpu().numpy().astype(np.float64)
        y_true = batch["y_future"].detach().cpu().numpy().astype(np.float64)
        pits = pits_from_samples(samples_np, y_true, randomized=True)  # (B,H)
        return pits  # np.ndarray


    def _debug_standardization(self, batch, tag):
        if not log.isEnabledFor(logging.DEBUG):
            return

        def stats(x):
            return float(x.mean()), float(x.std(unbiased=False))

        yp_m, yp_s = stats(batch["y_past"])
        yf_m, yf_s = stats(batch["y_future"])
        log.debug(
            "[%s] y_past mean=%.3f std=%.3f | y_future mean=%.3f std=%.3f",
            tag, yp_m, yp_s, yf_m, yf_s
        )

        # check CU columns individually
        cu = batch["c_ctx_future_unknown"]   # (B, CL, F_cu)
        cu_mean = cu.mean(dim=(0, 1))        # per-feature mean
        cu_std  = cu.std(dim=(0, 1), unbiased=False)

        log.debug("[%s] cu per-feature mean≈0? %s", tag, cu_mean.tolist())
        log.debug("[%s] cu per-feature std≈1?  %s", tag, cu_std.tolist())


    def _get_metric_scalar(self, split: str, name: str) -> float:
        """
        Returns a float from self.trainer.callback_metrics trying
        <split>_<name>_epoch, then <split>_<name>. Falls back to NaN.
        """
        m = self.trainer.callback_metrics
        return float(m.get(f"{split}_{name}_epoch",
                    m.get(f"{split}_{name}", float("nan"))))

    def _fmt_metric_block(self, split: str, names: set[str]) -> str:
        """
        Formats a 'split' block like: 'train_es=0.1234  train_crps=nan'
        Only includes keys that are in `names`.
        """
        parts = []
        for n in sorted(names):
            v = self._get_metric_scalar(split, n)
            parts.append(f"{split}_{n}={v:.4f}" if not math.isnan(v) else f"{split}_{n}=nan")
        return "  ".join(parts)

    def _prevent_extrem_values(self, y_fut, ctx):
        y_fut = torch.nan_to_num(y_fut, nan=0.0, posinf=1e6, neginf=-1e6)
        y_fut = torch.clamp(y_fut, -8.0, 8.0)

        ctx   = torch.nan_to_num(ctx,   nan=0.0, posinf=1e6, neginf=-1e6)
        ctx   = torch.clamp(ctx,   -8.0, 8.0)

        return y_fut, ctx
    
    def _to_py(self, x):
        import numpy as np, torch
        # torch → Python
        if torch.is_tensor(x):
            x = x.detach().cpu()
            if x.ndim == 0:
                return x.item()
            return x.tolist()
        # numpy scalar → Python
        if isinstance(x, (np.generic,)):
            return x.item()
        return x  # pass-through for str, Timestamp, etc.

    def _loss_log_to_parquet(self, batch, loss_bh: torch.Tensor):
        records = []
        B, F = loss_bh.shape
        for b in range(B):
            uid = self._to_py(batch["unique_id"][b])
            ds  = self._to_py(batch["ds"][b])
            row_mean = float(loss_bh[b, :].mean())
            for f in range(F):
                records.append((
                    int(self.global_step),
                    uid,
                    int(b),
                    ds,
                    int(f),
                    float(loss_bh[b, f]),
                    row_mean
                ))
        return records
    
    def _np(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return x

    def _stash_series(self, batch, series_vals: torch.Tensor, target: str):
        """
        Vectorized stash of per-origin scalar losses.
        series_vals: torch (B,)
        Appends one tuple: (uid[B], ds[B], values[B]) to _series_buf[target]
        """
        v   = series_vals.detach().cpu().numpy()     # (B,)
        uid = self._np(batch["unique_id"])           # (B,)
        ds  = self._np(batch["ds"])                  # (B,)
        self._series_buf[target].append((uid, ds, v))

    def _stash_bh_or_bk(self, batch, bh: torch.Tensor, target: str):
        """
        Vectorized stash of a (B,F) matrix (F = horizon H or slices K) into one tuple of arrays.
        Populates _crps_buf or _es_buf depending on target.
        """
        bh = bh.detach().cpu()                 # (B,H)
        B, H = bh.shape
        vals = bh.reshape(-1).numpy()          # (B*H,)
        row_mean = bh.mean(dim=1).repeat_interleave(H).numpy()

        uid = self._np(batch["unique_id"])     # (B,)
        ds  = self._np(batch["ds"])            # (B,)
        uid_flat = np.repeat(uid, H)
        ds_flat  = np.repeat(ds,  H)
        hidx     = np.tile(np.arange(H, dtype=np.int64), B)
        bidx     = np.repeat(np.arange(B, dtype=np.int64), H)
        gstep    = np.full(B*H, int(self.global_step), dtype=np.int64)

        buf = self._crps_buf if target == "crps" else self._es_buf
        buf.append((gstep, uid_flat, bidx, ds_flat, hidx, vals, row_mean))


    def configure_optimizers(self):
        ctx_params  = list(self.model.enc_proj.parameters()) \
                    + list(self.model.encoder.parameters()) \
                    + list(self.model.dec_proj.parameters()) \
                    + list(self.model.decoder.parameters()) \
                    + [self.model.pos_enc]
        flow_params = list(self.model.flow.parameters())
        for p in flow_params:
            p.requires_grad_(True)

        opt = torch.optim.AdamW(
            [
                {"params": ctx_params,  "lr": self.lr,       "weight_decay": 1e-4},
                {"params": flow_params, "lr": self.lr * 0.5, "weight_decay": 5e-6, "name": "flow"},
            ],
            lr=self.lr, eps=1e-7, betas=(0.9, 0.98),
        )

        total_epochs = getattr(self.trainer, "max_epochs", 100)
        warm = max(int(self.warmup_epochs), 0)
        remain = max(total_epochs - warm, 1)

        # pick a first cycle that allows a couple of restarts
        T_0 = max(min(10, remain // 2), 5)    # e.g., 3–10 epochs for the first cycle

        cosine_wr = CosineAnnealingWarmRestarts(
            opt,
            T_0=T_0,
            T_mult=2,
            eta_min=self.lr * 0.05,
        )

        if warm > 0:
            warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warm)
            sched = SequentialLR(opt, schedulers=[warmup, cosine_wr], milestones=[warm])
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        else:
            # no warmup; start CAWR immediately
            return {"optimizer": opt, "lr_scheduler": {"scheduler": cosine_wr, "interval": "epoch"}}
    

    # def configure_optimizers(self):
    #     ctx_params  = list(self.model.enc_proj.parameters()) \
    #                 + list(self.model.encoder.parameters()) \
    #                 + list(self.model.dec_proj.parameters()) \
    #                 + list(self.model.decoder.parameters()) \
    #                 + [self.model.pos_enc]
    #     flow_params = list(self.model.flow.parameters())

    #     for p in flow_params:
    #         p.requires_grad_(True)   # no freeze anymore

    #     # Single AdamW; set wd=0 for flow group
    #     opt = torch.optim.AdamW(
    #         [
    #             {"params": ctx_params,  "lr": self.lr,       "weight_decay": 1e-4},
    #             {"params": flow_params, "lr": self.lr * 0.5, "weight_decay": 5e-6, "name": "flow"},
    #         ],
    #         lr=self.lr, eps=1e-7, betas=(0.9, 0.98),
    #     )

    #     # Warmup -> cosine (same schedule applied to both groups)
    #     total_epochs = getattr(self.trainer, "max_epochs", 100) or 100
    #     warm = max(int(self.warmup_epochs), 1)
    #     cosT = max(total_epochs - warm, 1)

    #     from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    #     sch = SequentialLR(
    #         opt,
    #         schedulers=[
    #             LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warm),
    #             CosineAnnealingLR(opt, T_max=cosT, eta_min=self.lr * 0.05),
    #         ],
    #         milestones=[warm],
    #     )
    #     return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}