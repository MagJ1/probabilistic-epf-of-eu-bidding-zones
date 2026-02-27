from tqdm import tqdm
import numpy as np
import torch
import cvxpy as cp
from typing import Dict, List, Tuple

def _solve_lasso_qr(Z: np.ndarray, y: np.ndarray, tau: float, lam: float) -> Tuple[np.ndarray, float]:
    """
    Solve: minimize  sum rho_tau(y - (Z beta + beta_0)) + lambda ||beta||_1
    Returns (beta, beta0).
    """
    N, F = Z.shape
    beta  = cp.Variable(F)
    beta0 = cp.Variable()   # intercept
    u = y - (Z @ beta + beta0)
    # pinball loss
    loss = cp.sum(cp.maximum(tau * u, (tau - 1.0) * u))
    # lasso
    obj = loss + lam * cp.norm1(beta)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        # try a fallback solver
        prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LASSO-QR did not converge (status={prob.status})")
    return np.asarray(beta.value).ravel(), float(beta0.value)

def fit_lasso_qr_grid(
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    taus: List[float], lambda_grid: List[float]
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    For each tau, search lambda in grid using val pinball loss.
    Returns: {tau: {"beta": beta, "beta0": beta0, "lambda": best_lambda}}
    """
    results: Dict[float, Dict[str, np.ndarray]] = {}
    for tau in taus:
        best = {"val_loss": np.inf}
        for lam in lambda_grid:
            beta, beta0 = _solve_lasso_qr(Ztr, ytr, tau=float(tau)/100.0, lam=lam)
            yhat_val = Zval @ beta + beta0
            vloss = pinball_loss(yval, yhat_val, tau=float(tau)/100.0)
            if vloss < best["val_loss"]:
                best = {"val_loss": vloss, "beta": beta, "beta0": beta0, "lambda": lam}
        results[float(tau)] = {"beta": best["beta"], "beta0": best["beta0"], "lambda": best["lambda"]}
    return results

def predict_quantiles_per_tau(Z: np.ndarray, models_for_taus: Dict[float, Dict[str, np.ndarray]], taus: List[float]) -> np.ndarray:
    """
    Z: (N,F). Returns Q: (N, T) in the order of `taus`.
    """
    preds = []
    for tau in taus:
        info = models_for_taus[float(tau)]
        preds.append(Z @ info["beta"] + info["beta0"])
    return np.stack(preds, axis=1)  # (N, T)