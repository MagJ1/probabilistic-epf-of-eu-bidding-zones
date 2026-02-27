# src/models/nhits_qra/qra/qra_chung_loss.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChungLossArgs:
    # matches Chung toggles
    scale: bool = False                 # multiply by |coverage - q|
    sharp_penalty: Optional[float] = None  # mixing weight in [0,1], None disables
    sharp_all: bool = False             # if True: always penalize width>0; else only if overcoverage


@torch.no_grad()
def _expected_interval_props(q_list: torch.Tensor) -> torch.Tensor:
    # Chung uses exp_interval_props = abs((2*q)-1)
    return torch.abs((2.0 * q_list) - 1.0)


def batch_cali_loss(
    model: torch.nn.Module,
    y: torch.Tensor,               # (N,) or (N,1)
    x: Optional[torch.Tensor],     # (N,F) or None
    q_list: torch.Tensor,          # (Q,) in [0,1]
    args: ChungLossArgs,
) -> torch.Tensor:
    """
    Faithful re-implementation of Chung et al. batch_cali_loss:
    - calibration term: "pull from above" if under-covered, else "pull from below"
    - optional scaling by |coverage - q|
    - optional sharpness penalty based on opposite quantile (1-q)
    """
    device = y.device
    y = y.view(-1)                                # (N,)
    N = y.shape[0]
    Q = q_list.shape[0]

    # repeat q and y across quantiles
    q_rep = q_list.view(-1, 1).repeat(1, N).reshape(-1, 1).to(device)  # (Q*N,1)
    y_stacked = y.repeat(Q)                                              # (Q*N,)

    if x is None:
        model_in = q_rep
        x_stacked = None
    else:
        x = x.to(device)
        x_stacked = x.repeat(Q, 1)                                       # (Q*N,F)
        model_in = torch.cat([x_stacked, q_rep], dim=1)                  # (Q*N,F+1)

    pred_y = model(model_in).view(-1)                                    # (Q*N,)

    # coverage per quantile: P(y <= pred)
    idx_under = (y_stacked <= pred_y).view(Q, N)                         # (Q,N)
    idx_over  = (~idx_under)                                             # (Q,N)
    coverage  = idx_under.float().mean(dim=1)                            # (Q,)

    pred_y_mat = pred_y.view(Q, N)                                       # (Q,N)
    y_mat      = y_stacked.view(Q, N)                                    # (Q,N)
    diff_mat   = (y_mat - pred_y_mat)                                    # (Q,N)

    mean_diff_under = torch.mean((-1.0) * diff_mat * idx_under, dim=1)   # (Q,)
    mean_diff_over  = torch.mean(diff_mat * idx_over, dim=1)             # (Q,)

    cov_under = (coverage < q_list.to(device))                           # (Q,)
    loss_list = cov_under.float() * mean_diff_over + (~cov_under).float() * mean_diff_under  # (Q,)

    # scaling by |coverage - q|
    if args.scale:
        cov_diff = torch.abs(coverage - q_list.to(device))               # (Q,)
        loss_list = cov_diff * loss_list

    loss = loss_list.mean()

    # sharpness penalty
    if args.sharp_penalty is not None:
        sp = float(args.sharp_penalty)
        assert 0.0 <= sp <= 1.0

        # opposite-quantile predictions from the SAME model
        if x is None:
            opp_in = 1.0 - q_rep
        else:
            opp_in = torch.cat([x_stacked, (1.0 - q_rep)], dim=1)

        opp_pred = model(opp_in).view(-1)                                # (Q*N,)
        opp_pred_mat = opp_pred.view(Q, N)                               # (Q,N)

        # below_med = q <= 0.5, done elementwise like Chung
        below_med = (q_rep.view(Q, N) <= 0.5)                            # (Q,N)

        # width = opp - pred if below_med else pred - opp
        width = below_med.float() * (opp_pred_mat - pred_y_mat) + (~below_med).float() * (pred_y_mat - opp_pred_mat)  # (Q,N)
        width_pos = (width > 0.0)                                        # (Q,N)

        if args.sharp_all:
            sharp_term = width_pos.float() * width
        else:
            # penalize only when observed interval props > expected
            exp_props = _expected_interval_props(q_list.to(device))      # (Q,)
            # define interval lower/upper depending on side of median
            interval_lower = below_med.float() * pred_y_mat + (~below_med).float() * opp_pred_mat
            interval_upper = (~below_med).float() * pred_y_mat + below_med.float() * opp_pred_mat
            within = (interval_lower <= y_mat) & (y_mat <= interval_upper)
            obs_props = within.float().mean(dim=1)                       # (Q,)
            obs_over_exp = (obs_props > exp_props)                       # (Q,)

            sharp_term = obs_over_exp.view(Q, 1).float() * width_pos.float() * width  # (Q,N)

        loss = ((1.0 - sp) * loss) + (sp * sharp_term.mean())

    if not torch.isfinite(loss):
        raise RuntimeError("batch_cali_loss produced non-finite loss")
    return loss