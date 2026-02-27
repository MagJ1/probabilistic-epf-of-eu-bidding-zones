# src/models/nhits_qra/qra/qra_calib_chung_pair_loss.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ChungLossPairsArgs:
    scale: bool = False
    sharp_penalty: float | None = None
    sharp_all: bool = False

    # ---- crossing control (optional) ----
    cross_penalty: float = 10.0        # 0.0 disables
    cross_gate_eps: float = 0.0       # only activate if mean crossing > eps
    cross_gate: str = "sigmoid"          # "hard" | "sigmoid" | "none"
    cross_sigmoid_k: float = 50.0     # steepness for sigmoid gate


def _chung_cali_term(y: torch.Tensor, q: torch.Tensor, target: float, scale: bool) -> torch.Tensor:
    y = y.view(-1)
    q = q.view(-1)

    idx_under = (y <= q)
    idx_over = ~idx_under
    coverage = idx_under.float().mean()
    cov_val = float(coverage.detach().item())

    if cov_val < float(target):
        term = (y[idx_over] - q[idx_over]).mean() if idx_over.any() else torch.zeros((), device=y.device)
    else:
        term = (q[idx_under] - y[idx_under]).mean() if idx_under.any() else torch.zeros((), device=y.device)

    if scale:
        term = (coverage.detach() - float(target)).abs() * term
    return term


def _crossing_penalty(q_lo: torch.Tensor, q_hi: torch.Tensor, args: ChungLossPairsArgs) -> torch.Tensor:
    """
    Penalize q_lo > q_hi. Gate it so it only "turns on" when crossings exist.
    Returns a scalar tensor.
    """
    if args.cross_penalty is None or float(args.cross_penalty) <= 0.0:
        return torch.zeros((), device=q_lo.device)

    # how much crossing (mean magnitude)
    cross = F.relu(q_lo - q_hi).mean()  # scalar

    gate_mode = str(getattr(args, "cross_gate", "hard")).lower()
    eps = float(getattr(args, "cross_gate_eps", 0.0))

    if gate_mode == "none":
        gate = 1.0
    elif gate_mode == "sigmoid":
        k = float(getattr(args, "cross_sigmoid_k", 50.0))
        # detach so gate doesn't create weird gradients; penalty gradients still come from `cross`
        gate = torch.sigmoid(k * (cross.detach() - eps))
    else:
        # "hard" gate (default)
        gate = (cross.detach() > eps).float()

    return float(args.cross_penalty) * gate * cross


def chung_pair_loss(
    *,
    y: torch.Tensor,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    tau: float,  # in (0, 0.5)
    args: ChungLossPairsArgs,
) -> torch.Tensor:
    y = y.view(-1)
    q_lo = q_lo.view(-1)
    q_hi = q_hi.view(-1)

    # ---- calibration for BOTH tails
    cal_lo = _chung_cali_term(y, q_lo, target=float(tau), scale=bool(args.scale))
    cal_hi = _chung_cali_term(y, q_hi, target=float(1.0 - tau), scale=bool(args.scale))
    calib = 0.5 * (cal_lo + cal_hi)

    # ---- add gated crossing penalty (OPTIONAL)
    calib = calib + _crossing_penalty(q_lo, q_hi, args)

    # ---- no sharpness mixing?
    if args.sharp_penalty is None:
        return calib

    alpha = float(args.sharp_penalty)

    # width (penalize only positive width like Chung)
    width = F.relu(q_hi - q_lo)  # (B,)

    if bool(args.sharp_all):
        sharp_term = width.mean()
        return (1 - alpha) * calib + alpha * sharp_term

    # gate sharpness only if interval overcovers
    lower = torch.minimum(q_lo, q_hi)
    upper = torch.maximum(q_lo, q_hi)
    within = ((lower <= y) & (y <= upper)).float().mean()
    exp_interval_props = 1.0 - 2.0 * float(tau)
    obs_over_exp = float(within.detach().item()) > exp_interval_props

    sharp_term = width.mean() if obs_over_exp else torch.zeros((), device=y.device)
    return (1 - alpha) * calib + alpha * sharp_term