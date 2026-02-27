# src/models/nhits_qra/qra/qra_calib_chung_iterative_solver.py
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from models.nhits_qra.qra.qra_calib_chung_loss import batch_cali_loss, ChungLossArgs
from models.nhits_qra.qra.qra_calib_chung_model import ChungQModel, ChungModelCfg

def _to_unit_taus(taus: List[float]) -> np.ndarray:
    t = np.asarray(taus, dtype=float)
    if t.max() > 1.0:
        t = t / 100.0
    return t

def _pick_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def fit_chung_cali_grid_torch(
    Ztr: np.ndarray, ytr: np.ndarray,
    Zval: np.ndarray, yval: np.ndarray,
    taus: List[float],
    lambda_grid: List[float],
    loss_args: ChungLossArgs,
    model_cfg: Optional[ChungModelCfg] = None,
    device: Optional[str] = None,
    max_epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-2,
    patience: int = 20,
    seed: int = 123,
    verbose: bool = False,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    device = _pick_device(device)

    taus_u = _to_unit_taus(taus)
    q_list = torch.tensor(taus_u, dtype=torch.float32, device=device)  # (Q,)

    Ztr_t = torch.from_numpy(Ztr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().view(-1).to(device)
    Zval_t = torch.from_numpy(Zval).float().to(device)
    yval_t = torch.from_numpy(yval).float().view(-1).to(device)

    Ntr, F = Ztr_t.shape
    in_dim = F + 1

    idx = torch.arange(Ntr, device=device)

    best_val = math.inf
    best_state = None
    best_lam = None


    for lam in lambda_grid:
        model = ChungQModel(in_dim=in_dim, cfg=model_cfg).to(device)
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

                loss_core = batch_cali_loss(
                    model=model,
                    y=yb,
                    x=Zb,
                    q_list=q_list,
                    args=loss_args,
                )

                # L1 on all weights (excluding biases) — Chung-ish regularization
                l1 = 0.0
                for name, p in model.named_parameters():
                    if p.requires_grad and ("bias" not in name):
                        l1 = l1 + p.abs().sum()
                loss = loss_core + float(lam) * (len(sel) / Ntr) * l1

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                val_loss = batch_cali_loss(
                    model=model,
                    y=yval_t,
                    x=Zval_t,
                    q_list=q_list,
                    args=loss_args,
                ).item()

            if val_loss < cur_best - 1e-8:
                cur_best = val_loss
                cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if cur_state is None:
            # fallback (rare)
            cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"[chung] lam={lam:.2e} best_val={cur_best:.6f}")

        if cur_best < best_val:
            best_val = cur_best
            best_state = cur_state
            best_lam = float(lam)

    if best_state is None:
        raise RuntimeError("fit_chung_cali_grid_torch: no best_state found (empty lambda_grid?)")

    return {
        "state_dict": best_state,
        "lambda": float(best_lam),
        "taus_unit": taus_u,
        "in_dim": int(in_dim),
        "model_cfg": model_cfg,
    }

@torch.no_grad()
def predict_quantiles_chung(
    Z: np.ndarray,
    pack: Dict[str, object],
    taus: List[float],
    device: str = "cpu",
) -> np.ndarray:
    device = _pick_device(device)
    taus_u = _to_unit_taus(taus)

    Zt = torch.from_numpy(Z).float().to(device)
    N, F = Zt.shape
    in_dim = int(pack["in_dim"])

    model_cfg = pack.get("model_cfg", ChungModelCfg(kind="linear"))
    model = ChungQModel(in_dim=in_dim, cfg=model_cfg).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    q_list = torch.tensor(taus_u, dtype=torch.float32, device=device)  # (Q,)
    Q = q_list.numel()

    # vectorized: stack all taus in one forward
    # model_in: (Q*N, F+1)
    Z_rep = Zt.repeat(Q, 1)                        # (Q*N, F)
    q_rep = q_list.view(-1, 1).repeat(1, N).view(-1, 1)  # (Q*N, 1)
    model_in = torch.cat([Z_rep, q_rep], dim=1)

    pred = model(model_in).view(Q, N).transpose(0, 1)  # (N, Q)
    return pred.detach().cpu().numpy()



@torch.no_grad()
def q_sensitivity_chung(pack, Z: np.ndarray, q_grid, device="cpu") -> dict:
    q_grid = np.asarray(q_grid, dtype=float)
    if q_grid.max() > 1.0: q_grid = q_grid / 100.0

    # run prediction for all q_grid (N, Q)
    Q = predict_quantiles_chung(Z, pack, taus=q_grid.tolist(), device=device)

    # stats
    std_over_q = Q.std(axis=1)          # (N,)
    span = Q.max(axis=1) - Q.min(axis=1)

    return {
        "q_grid": q_grid.tolist(),
        "mean_std_over_q": float(std_over_q.mean()),
        "median_std_over_q": float(np.median(std_over_q)),
        "mean_span": float(span.mean()),
        "median_span": float(np.median(span)),
    }
