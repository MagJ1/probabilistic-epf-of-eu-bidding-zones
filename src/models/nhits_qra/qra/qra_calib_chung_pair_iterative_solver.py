# src/models/nhits_qra/qra/qra_calib_chung_pair_iterative_solver.py
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.nhits_qra.qra.qra_calib_chung_pair_loss import ChungLossPairsArgs, chung_pair_loss

class TwoLinear(nn.Module):
    """Two independent linear models on Z: lower quantile and upper quantile."""
    def __init__(self, F: int):
        super().__init__()
        self.lo = nn.Linear(F, 1, bias=True)
        self.hi = nn.Linear(F, 1, bias=True)

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lo(Z).view(-1), self.hi(Z).view(-1)

def _to_unit_taus(taus: List[float]) -> np.ndarray:
    t = np.asarray(taus, dtype=float)
    if t.max() > 1.0:
        t = t / 100.0
    return t

def _lower_tau_set(taus: List[float]) -> List[float]:
    taus_u = sorted(set(round(float(t), 12) for t in _to_unit_taus(taus).tolist()))
    lows = sorted(set(
        round(min(t, 1.0 - t), 12)
        for t in taus_u
        if abs(t - 0.5) > 1e-12
    ))
    return [t for t in lows if 0.0 < t < 0.5]

def _train_one_tau_pair(
    *,
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    tau: float,
    lambda_grid: List[float],
    loss_args: ChungLossPairsArgs,
    device: Optional[str],
    max_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
) -> Tuple[Dict, float]:
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )

    Ztr_t  = torch.from_numpy(Ztr).float().to(device)
    ytr_t  = torch.from_numpy(ytr).float().view(-1).to(device)
    Zval_t = torch.from_numpy(Zval).float().to(device)
    yval_t = torch.from_numpy(yval).float().view(-1).to(device)

    Ntr, F = Ztr_t.shape
    idx = torch.arange(Ntr, device=device)

    best_val = math.inf
    best_state = None
    best_lam = None

    for lam in lambda_grid:
        model = TwoLinear(F).to(device)
        opt = optim.AdamW(model.parameters(), lr=lr)

        cur_best = math.inf
        cur_state = None
        no_improve = 0

        for epoch in range(max_epochs):
            perm = idx[torch.randperm(Ntr, device=device)]
            for start in range(0, Ntr, batch_size):
                sel = perm[start:start + batch_size]
                Zb = Ztr_t[sel]
                yb = ytr_t[sel]

                q_lo_b, q_hi_b = model(Zb)
                loss_core = chung_pair_loss(y=yb, q_lo=q_lo_b, q_hi=q_hi_b, tau=tau, args=loss_args)

                # L1 on weights only (exclude biases)
                l1 = model.lo.weight.abs().sum() + model.hi.weight.abs().sum()
                loss = loss_core + float(lam) * (len(sel) / Ntr) * l1

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                q_lo_v, q_hi_v = model(Zval_t)
                val_loss = chung_pair_loss(y=yval_t, q_lo=q_lo_v, q_hi=q_hi_v, tau=tau, args=loss_args).item()

            if val_loss < cur_best - 1e-8:
                cur_best = val_loss
                cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # fallback if never improved (rare)
        if cur_state is None:
            cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if cur_best < best_val:
            best_val = cur_best
            best_state = cur_state
            best_lam = float(lam)

    if best_state is None:
        raise RuntimeError("No best_state found (empty lambda_grid?)")

    return best_state, float(best_lam)

def fit_chung_pair_grid_torch(
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    taus: List[float],
    lambda_grid: List[float],
    loss_args: ChungLossPairsArgs,
    device: Optional[str] = None,
    max_epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-2,
    patience: int = 20,
    seed: int = 123,
    verbose: bool = False,
) -> Dict[float, Dict[str, object]]:
    """
    Returns per-TAU models keyed by tau_pct:
      models[tau_pct] = {"beta": (F,), "bias": float, "lambda": float}
    We train on lower-tail taus; and we ALSO populate the symmetric upper tail (100 - tau_pct).
    """
    tau_lows = _lower_tau_set(taus)
    if len(tau_lows) == 0:
        raise ValueError("No lower-tail taus found (need something != 0.5)")

    Ntr, F = Ztr.shape
    out: Dict[float, Dict[str, object]] = {}

    for tau in tau_lows:
        best_state, best_lam = _train_one_tau_pair(
            Ztr=Ztr, ytr=ytr, Zval=Zval, yval=yval,
            tau=float(tau),
            lambda_grid=lambda_grid,
            loss_args=loss_args,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            seed=seed,
        )

        # export coefficients (two separate linear models)
        lo_w = best_state["lo.weight"].numpy().ravel()
        lo_b = float(best_state["lo.bias"].numpy().ravel()[0])
        hi_w = best_state["hi.weight"].numpy().ravel()
        hi_b = float(best_state["hi.bias"].numpy().ravel()[0])

        tau_lo_pct = round(100.0 * float(tau), 12)
        tau_hi_pct = round(100.0 * float(1.0 - tau), 12)

        out[tau_lo_pct] = {"beta": lo_w, "beta0": lo_b, "lambda": best_lam}
        out[tau_hi_pct] = {"beta": hi_w, "beta0": hi_b, "lambda": best_lam}

        if verbose:
            print(f"[chung-pair] tau={tau_lo_pct:.3f}/{tau_hi_pct:.3f} lam={best_lam:.2e}")

    return out

def predict_quantiles_per_tau_linear(
    Z: np.ndarray,
    models: Dict[float, Dict[str, object]],
    taus: List[float],
    ) -> np.ndarray:
    taus_u = _to_unit_taus(taus)
    Q_cols = []
    for tu in taus_u:
        key = round(float(tu) * 100.0, 12)
        m = models[key]
        beta = np.asarray(m["beta"], dtype=float).ravel()
        bias = float(m["beta0"])
        Q_cols.append(Z @ beta + bias)
    return np.stack(Q_cols, axis=1)