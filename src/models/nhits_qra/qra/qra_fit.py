# src/models/nhits_qra/qra/qra_fit.py
from __future__ import annotations
from omegaconf import OmegaConf
import numpy as np

from models.nhits_qra.qra.qra_exact_solver import fit_lasso_qr_grid, predict_quantiles_per_tau
from models.nhits_qra.qra.qra_pinball_iterative_solver import fit_lasso_qr_grid_torch, predict_quantiles_per_tau_torchlike
from models.nhits_qra.qra.qra_calib_iterative_solver import fit_lasso_pair_grid_torch, predict_quantiles_per_tau_pairlike
from models.nhits_qra.qra.qra_knobs import QRAKnobs

from models.nhits_qra.qra.qra_calib_chung_iterative_solver import (
    fit_chung_cali_grid_torch,
    predict_quantiles_chung,
)
from models.nhits_qra.qra.qra_calib_chung_loss import ChungLossArgs

from models.nhits_qra.qra.qra_calib_chung_pair_iterative_solver import (
    fit_chung_pair_grid_torch,
    predict_quantiles_per_tau_linear,
)

from models.nhits_qra.qra.qra_calib_chung_pair_loss import ChungLossPairsArgs

from models.nhits_qra.qra.qra_calib_chung_model import ChungModelCfg

import logging
log = logging.getLogger("fit_qra_per_h")

def fit_qra_per_h(Ztr, ytr, Zval, yval, taus, lambda_grid, knobs: QRAKnobs, device: str, cfg: OmegaConf):
    models_all, Qval_all, pinball_all = [], [], []
    for h, (Zt, yt, Zv, yv) in enumerate(zip(Ztr, ytr, Zval, yval)):
        log.info(f"Horizon: {h}")
        if Zt.size == 0 or Zt.shape[1] == 0:
            models_all.append({})
            Qval_all.append(None)
            pinball_all.append(None)
            continue

        if knobs.solver_loss == "exact_pinball":
            mh = fit_lasso_qr_grid(Zt, yt, Zv, yv, taus=taus, lambda_grid=lambda_grid)
            Qv = predict_quantiles_per_tau(Zv, mh, taus)
        elif knobs.solver_loss == "iterative_pinball":
            mh = fit_lasso_qr_grid_torch(
                Ztr=Zt, ytr=yt, Zval=Zv, yval=yv,
                taus=taus, lambda_grid=lambda_grid,
                device=device,
                max_epochs=int(cfg.train_qra.n_epochs),
                batch_size=int(cfg.train_qra.batch_size),
                lr=float(cfg.train_qra.lr),
                patience=int(cfg.train_qra.patience),
                seed=int(cfg.seed),
                verbose=knobs.verbose,
            )
            Qv = predict_quantiles_per_tau_torchlike(Zv, mh, taus)

        elif knobs.solver_loss == "iterative_calib":
            mh = fit_lasso_pair_grid_torch(
                Ztr=Zt, ytr=yt, Zval=Zv, yval=yv,
                taus=taus,
                lambda_grid=lambda_grid,
                mix_kappa=float(cfg.train_qra.mix_kappa),
                device=device,
                max_epochs=int(cfg.train_qra.n_epochs),
                batch_size=int(cfg.train_qra.batch_size),
                lr=float(cfg.train_qra.lr),
                patience=int(cfg.train_qra.patience),
                seed=int(cfg.seed),
                verbose=knobs.verbose,
            )
            Qv = predict_quantiles_per_tau_pairlike(Zv, mh, taus)

        elif knobs.solver_loss == "iterative_calib_chung_pair":
            loss_args = ChungLossPairsArgs(
                scale=bool(cfg.train_qra.chung_loss.scale),
                sharp_penalty=float(cfg.train_qra.chung_loss.sharp_penalty) if cfg.train_qra.chung_loss.sharp_penalty is float else None,  # mixing weight
                sharp_all=bool(cfg.train_qra.chung_loss.sharp_all),
            )

            mh = fit_chung_pair_grid_torch(
                Ztr=Zt, ytr=yt, Zval=Zv, yval=yv,
                taus=taus,
                lambda_grid=lambda_grid,
                loss_args=loss_args,
                device=device,
                max_epochs=int(cfg.train_qra.n_epochs),
                batch_size=int(cfg.train_qra.batch_size),
                lr=float(cfg.train_qra.lr),
                patience=int(cfg.train_qra.patience),
                seed=int(cfg.seed),
                verbose=knobs.verbose,
            )

            # ---- OPTIONAL: fit median (50th) with pinball and merge into mh ----
            # detect whether user requested 50 (accept 50 or 0.5)
            taus_u = np.asarray(taus, dtype=float)
            taus_u = taus_u / 100.0 if taus_u.max() > 1.0 else taus_u
            want_median = np.any(np.isclose(taus_u, 0.5, atol=1e-12))

            if want_median:
                mh50 = fit_lasso_qr_grid_torch(
                    Ztr=Zt, ytr=yt, Zval=Zv, yval=yv,
                    taus=[50.0],  # fit exactly the median
                    lambda_grid=lambda_grid,
                    device=device,
                    max_epochs=int(cfg.train_qra.n_epochs),
                    batch_size=int(cfg.train_qra.batch_size),
                    lr=float(cfg.train_qra.lr),
                    patience=int(cfg.train_qra.patience),
                    seed=int(cfg.seed),
                    verbose=False,
                )

                # mh50 should be keyed by 50.0, but be defensive
                k50 = 50.0
                if k50 not in mh50:
                    # fallback if solver keys differently
                    k50 = next(iter(mh50.keys()))

                m50 = mh50[k50]
                beta50 = np.asarray(m50["beta"], dtype=float).ravel()
                b50 = float(m50.get("beta0"))

                mh[50.0] = {
                    "beta": beta50,
                    "beta0": b50,                     # keep both names for compatibility
                    "lambda": float(m50.get("lambda", np.nan)),
                    "source": "pinball_median",
                }

            Qv = predict_quantiles_per_tau_linear(Zv, mh, taus)

        elif knobs.solver_loss == "iterative_calib_chung":
            loss_args = ChungLossArgs(
                scale=bool(cfg.train_qra.chung_loss.scale),
                sharp_penalty=float(cfg.train_qra.chung_loss.sharp_penalty),
                sharp_all=bool(cfg.train_qra.chung_loss.sharp_all),
            )
            model_cfg = ChungModelCfg(
                kind=str(cfg.train_qra.chung_model.kind),
                hidden=list(cfg.train_qra.chung_model.hidden_layers),
                dropout=float(cfg.train_qra.chung_model.dropout),
                act=str(cfg.train_qra.chung_model.activation_function),
            )

            mh = fit_chung_cali_grid_torch(
                Ztr=Zt, ytr=yt, Zval=Zv, yval=yv,
                taus=taus,
                lambda_grid=lambda_grid,
                loss_args=loss_args,
                model_cfg=model_cfg,
                device=device,
                max_epochs=int(cfg.train_qra.n_epochs),
                batch_size=int(cfg.train_qra.batch_size),
                lr=float(cfg.train_qra.lr),
                patience=int(cfg.train_qra.patience),
                seed=int(cfg.seed),
                verbose=knobs.verbose,
            )
            Qv = predict_quantiles_chung(Zv, mh, taus=taus, device=device)
        else:
            raise ValueError(f"Unknown solver_loss={knobs.solver_loss}")

        models_all.append(mh)
        Qval_all.append(Qv)

    return models_all, Qval_all