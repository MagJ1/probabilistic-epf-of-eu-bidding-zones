# src/models/nhits_qra/qra/qra_knobs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Literal
from omegaconf import OmegaConf

from utils.logging_utils import get_logger


@dataclass(frozen=True)
class QRAKnobs:
    n_samples: int
    add_mean_std: int
    use_pca: bool
    pca_var: Optional[float]
    n_comp: Optional[int]
    subsample_stride: int
    lambda_grid: List[float]
    taus_train: List[float]
    solver_loss: Literal["exact_pinball", "iterative_pinball", "iterative_calib"]
    mix_kappa: Optional[float]
    sample_dropout: bool
    verbose: bool
    target_quantiles_spec: Any
    n_samples_val: int
    sampling_backend: Literal["mc_dropout", "swag"]
    swag_scale: float
    swag_var_clamp: float

    @staticmethod
    def from_cfg(train_cfg: OmegaConf, test_cfg, log=None) -> "QRAKnobs":
        log = log or get_logger("qra_pipeline")
        if test_cfg:
            subsample_stride=int(test_cfg.test.subsample_stride)
        else:
            subsample_stride=int(train_cfg.train_qra.subsample_stride)

        swag_enabled = bool(getattr(train_cfg.train_nhits, "swag", {}).get("enabled", False))

        sampling_backend = "swag" if swag_enabled else "mc_dropout"
        swag_cfg = getattr(train_cfg.train_nhits, "swag", None)

        return QRAKnobs(
            n_samples=int(train_cfg.train_qra.mc_nhits_samples_qra),
            add_mean_std=int(train_cfg.train_qra.sample_k),
            use_pca=bool(train_cfg.train_qra.use_pca),
            pca_var=(float(train_cfg.train_qra.pca_var) if train_cfg.train_qra.pca_var is not None else None),
            n_comp=(int(train_cfg.train_qra.n_components) if train_cfg.train_qra.n_components is not None else None),
            subsample_stride=subsample_stride,
            lambda_grid=list(train_cfg.train_qra.lambda_grid),
            taus_train=list(train_cfg.train_qra.quantiles),
            solver_loss=str(train_cfg.train_qra.solver_loss),
            mix_kappa=float(train_cfg.train_qra.mix_kappa) if str(train_cfg.train_qra.solver_loss) == "iterative_calib" else None,
            sample_dropout=bool(train_cfg.train_nhits.sample_dropout),
            verbose=bool(getattr(train_cfg.train_qra, "verbose", False)),
            target_quantiles_spec=getattr(train_cfg.train_qra, "target_quantiles", None),
            n_samples_val=int(train_cfg.train_qra.n_samples_val),
            sampling_backend=sampling_backend,
            swag_scale=float(getattr(swag_cfg, "scale", 1.0)) if swag_cfg else 1.0,
            swag_var_clamp=float(getattr(swag_cfg, "var_clamp", 1e-30)) if swag_cfg else 1e-30,
        )