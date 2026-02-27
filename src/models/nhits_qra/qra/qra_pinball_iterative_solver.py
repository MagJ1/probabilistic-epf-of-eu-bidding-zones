# --- torch lasso-quantile QRA ---
import math
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import logging
log = logging.getLogger("qra_solver")

def _pinball_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float) -> torch.Tensor:
    # y_true, y_pred: (B,1)
    u = y_true - y_pred
    return torch.maximum(tau * u, (tau - 1.0) * u).mean()

def _train_lasso_qr_torch(
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    tau: float, lam: float,
    device: Optional[str] = None,
    max_epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-2,
    patience: int = 20,
    warm_start: Optional[Tuple[np.ndarray, float]] = None,  # (beta, beta0)
    seed: int = 123,
) -> Tuple[np.ndarray, float, float]:
    """
    Train a single (tau, lambda) model:
      min_{beta,bias} mean pinball_tau(y - (Z beta + bias)) + lam * ||beta||_1
    Returns: (beta (F,), bias (float), best_val_pinball)
    """
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    Ztr_t = torch.from_numpy(Ztr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().view(-1, 1).to(device)
    Zval_t = torch.from_numpy(Zval).float().to(device)
    yval_t = torch.from_numpy(yval).float().view(-1, 1).to(device)

    Ntr, F = Ztr_t.shape
    model = nn.Linear(F, 1, bias=True).to(device)

    # warm-start init if provided
    if warm_start is not None:
        beta_np, b0 = warm_start
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(beta_np).float().view(1, -1).to(device))
            model.bias.copy_(torch.tensor([b0], dtype=torch.float32, device=device))

    opt = optim.AdamW(model.parameters(), lr=lr)
    best_val = math.inf
    best_w = None
    best_b = None
    no_improve = 0

    # mini-batch indices
    idx = torch.arange(Ntr, device=device)

    for epoch in range(max_epochs):
        # shuffle each epoch
        perm = idx[torch.randperm(Ntr, device=device)]
        for start in range(0, Ntr, batch_size):
            sel = perm[start:start + batch_size]
            Zb = Ztr_t[sel]
            yb = ytr_t[sel]

            yhat = model(Zb)                        # (B,1)
            loss_q = _pinball_loss_torch(yb, yhat, tau)
            # L1 on weights only (no penalty on bias)
            l1 = model.weight.abs().sum()
            # scale L1 so objective matches full-data λ (not batch-size dependent)
            loss = loss_q + lam * (len(sel) / Ntr) * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # val
        with torch.no_grad():
            yhat_val = model(Zval_t)
            val_pinball = _pinball_loss_torch(yval_t, yhat_val, tau).item()

        if val_pinball < best_val - 1e-8:
            best_val = val_pinball
            with torch.no_grad():
                best_w = model.weight.detach().cpu().numpy().ravel()
                best_b = float(model.bias.detach().cpu().numpy().ravel()[0])
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # if never improved (unlikely), take current
    if best_w is None:
        with torch.no_grad():
            best_w = model.weight.detach().cpu().numpy().ravel()
            best_b = float(model.bias.detach().cpu().numpy().ravel()[0])

    return best_w, best_b, best_val

def fit_lasso_qr_grid_torch(
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    taus: List[float], lambda_grid: List[float],
    device: Optional[str] = None,
    max_epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-2,
    patience: int = 20,
    seed: int = 123,
    verbose: bool = False,
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Torch version: returns {tau: {"beta": beta, "beta0": beta0, "lambda": best_lambda}}
    Uses warm-start across lambdas for the same tau.
    """
    results: Dict[float, Dict[str, np.ndarray]] = {}
    for tau_pct in taus:
        tau = float(tau_pct) / 100.0
        best = {"val": math.inf}
        warm: Optional[Tuple[np.ndarray, float]] = None

        for lam in lambda_grid:
            beta, beta0, val_pin = _train_lasso_qr_torch(
                Ztr, ytr, Zval, yval, tau=tau, lam=lam, device=device,
                max_epochs=max_epochs, batch_size=batch_size, lr=lr, patience=patience,
                warm_start=warm, seed=seed
            )
            if verbose:
                log.info(
                    f"tau={tau_pct:>5.1f} "
                    f"lambda={lam:.1e} "
                    f"val_pinball={val_pin:.6f}"
                )
            warm = (beta, beta0)  # warm-start next lambda

            if val_pin < best["val"]:
                best = {"val": val_pin, "beta": beta, "beta0": beta0, "lambda": lam}

        results[float(tau_pct)] = {
            "beta": best["beta"],
            "beta0": best["beta0"],
            "lambda": best["lambda"],
        }
    return results

def predict_quantiles_per_tau_torchlike(Z: np.ndarray, models_for_taus: Dict[float, Dict[str, np.ndarray]], taus: List[float]) -> np.ndarray:
    preds = []
    for tau in taus:
        info = models_for_taus[float(tau)]
        preds.append(Z @ info["beta"] + info["beta0"])
    return np.stack(preds, axis=1)