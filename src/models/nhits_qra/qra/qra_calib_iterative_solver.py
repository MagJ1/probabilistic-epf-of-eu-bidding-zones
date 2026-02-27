# src/models/nhits_qra/qra/qra_calib_iterative_solver.py
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.nhits_qra.qra.qra_calib_loss import combined_calib_sharp_loss_softplus

def softplus_np(x):
    # stable softplus: log(1+exp(x))
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

class LoGapLinear(nn.Module):
    def __init__(self, F: int):
        super().__init__()
        self.lo = nn.Linear(F, 1, bias=True)   # q_lo
        self.dg = nn.Linear(F, 1, bias=True)   # d_raw

    def forward(self, Z):
        q_lo = self.lo(Z)
        d_raw = self.dg(Z)
        return q_lo, d_raw

def _train_lasso_pair_torch(
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    tau: float,
    lam: float,
    mix_kappa: float,
    device: Optional[str] = None,
    max_epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-2,
    patience: int = 20,
    seed: int = 123,
):
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    Ztr_t = torch.from_numpy(Ztr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().view(-1, 1).to(device)
    Zval_t = torch.from_numpy(Zval).float().to(device)
    yval_t = torch.from_numpy(yval).float().view(-1, 1).to(device)

    Ntr, F = Ztr_t.shape
    model = LoGapLinear(F).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr)

    best_val = math.inf
    best_state = None
    no_improve = 0
    idx = torch.arange(Ntr, device=device)

    for epoch in range(max_epochs):
        perm = idx[torch.randperm(Ntr, device=device)]
        for start in range(0, Ntr, batch_size):
            sel = perm[start:start + batch_size]
            Zb = Ztr_t[sel]
            yb = ytr_t[sel].view(-1)   # (B,)

            q_lo_b, d_raw_b = model(Zb)
            q_lo_b = q_lo_b.view(-1)
            d_raw_b = d_raw_b.view(-1)

            loss_core = combined_calib_sharp_loss_softplus(
                y=yb, q_lo=q_lo_b, d_raw=d_raw_b, tau=tau, mix_kappa=mix_kappa
            )

            # L1 on weights only (no penalty on biases), scale similar to what we did
            l1 = model.lo.weight.abs().sum() + model.dg.weight.abs().sum()
            loss = loss_core + lam * (len(sel) / Ntr) * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # validation
        with torch.no_grad():
            q_lo_v, d_raw_v = model(Zval_t)
            val_loss = combined_calib_sharp_loss_softplus(
                y=yval_t.view(-1),
                q_lo=q_lo_v.view(-1),
                d_raw=d_raw_v.view(-1),
                tau=tau,
                mix_kappa=mix_kappa
            ).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # export coefficients
    with torch.no_grad():
        beta_lo = model.lo.weight.detach().cpu().numpy().ravel()
        b_lo = float(model.lo.bias.detach().cpu().numpy().ravel()[0])
        beta_d = model.dg.weight.detach().cpu().numpy().ravel()
        b_d = float(model.dg.bias.detach().cpu().numpy().ravel()[0])

    return (beta_lo, b_lo, beta_d, b_d, best_val)


def predict_interval_numpy(Z: np.ndarray, beta_lo, b_lo, beta_d, b_d):
    q_lo = Z @ beta_lo + b_lo
    d = Z @ beta_d + b_d
    gap = softplus_np(d)
    q_hi = q_lo + gap
    return q_lo, q_hi


def _to_unit_tau(tau):
    t = float(tau)
    return t / 100.0 if t > 1.0 else t

def _lower_tau_set(taus: List[float]) -> List[float]:
    # Unique lower-tail taus excluding 0.5
    taus_u = sorted(set(_to_unit_tau(t) for t in taus))
    lows = sorted(set(min(t, 1.0 - t) for t in taus_u if abs(t - 0.5) > 1e-12))
    return lows

def fit_lasso_pair_grid_torch(
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    taus: List[float],
    lambda_grid: List[float],
    mix_kappa: float,
    device: Optional[str] = None,
    max_epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-2,
    patience: int = 20,
    seed: int = 123,
    verbose: bool = False,
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Returns models keyed by LOWER-TAIL tau (in percent, matching old style keys):
      {tau_pct: {"beta_lo","b_lo","beta_d","b_d","lambda"}}
    """

    
    results: Dict[float, Dict[str, np.ndarray]] = {}

    tau_lows = _lower_tau_set(taus)
    assert len(tau_lows) > 0, "No lower-tail taus found (did you pass only 0.5/50?)"
    for tau in tau_lows:
        best_val = math.inf
        best_pack = None

        for lam in lambda_grid:
            beta_lo, b_lo, beta_d, b_d, val_loss = _train_lasso_pair_torch(
                Ztr, ytr, Zval, yval,
                tau=tau,
                lam=lam,
                mix_kappa=mix_kappa,
                device=device,
                max_epochs=max_epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                seed=seed,
            )
            if verbose:
                print(f"[pair] tau={tau:.3f} lam={lam:.2e} val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                best_pack = (beta_lo, b_lo, beta_d, b_d, lam)

        tau_pct = round(float(tau * 100.0), 12)
        beta_lo, b_lo, beta_d, b_d, lam = best_pack
        results[tau_pct] = {
            "beta_lo": beta_lo,
            "b_lo": float(b_lo),
            "beta_d": beta_d,
            "b_d": float(b_d),
            "lambda": float(lam),
        }

    return results


def predict_quantiles_per_tau_pairlike(Z: np.ndarray,
                                       models_pair: Dict[float, Dict],
                                       taus: List[float]) -> np.ndarray:
    """
    Build Q(N, len(taus)) in the same order as `taus`.
    models_pair is keyed by LOWER-TAIL tau in percent: e.g. 5.0, 10.0, ...
    """
    Q_cols = []
    for t in taus:
        tu = _to_unit_tau(t)  # 0..1
        # Handle symmetric mapping: model is trained for tau_low = min(t, 1-t)
        tau_low = min(tu, 1.0 - tu)

        # If tau==0.5, no natural "pair endpoint"
        if abs(tu - 0.5) < 1e-12:
            # raise and ensure 50 is not in taus_train for this solver
            raise ValueError("tau=0.5 not supported by pair solver; remove 50 from taus_train or train median separately.")


        key = round(tau_low * 100.0, 12)
        m = models_pair[key]
        beta_lo, b_lo = m["beta_lo"], m["b_lo"]
        beta_d,  b_d  = m["beta_d"],  m["b_d"]

        q_lo = Z @ beta_lo + b_lo
        gap  = softplus_np(Z @ beta_d + b_d)
        q_hi = q_lo + gap

        # If asked tau is the lower tail -> return q_lo, else return q_hi
        q = q_lo if (tu <= 0.5) else q_hi
        Q_cols.append(q)

    return np.stack(Q_cols, axis=1)  # (N, len(taus))