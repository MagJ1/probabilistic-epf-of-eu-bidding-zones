# src/moels/nhits_qra/qra/qra_calib_loss.py
import torch
import torch.nn.functional as F

def combined_calib_sharp_loss_softplus(
    y: torch.Tensor,          # (B,)
    q_lo: torch.Tensor,       # (B,)  predicted lower quantile
    d_raw: torch.Tensor,      # (B,)  raw gap (can be any real)
    tau: float,
    mix_kappa: float,        # in [0,1]
):
    y = y.view(-1)
    q_lo = q_lo.view(-1)
    d_raw = d_raw.view(-1)

    # enforce non-crossing by construction
    gap = F.softplus(d_raw)          # >= 0
    q_hi = q_lo + gap                # always >= q_lo

    # observed coverage for q_lo (estimate p_hat = P(Y <= q_lo))
    p_hat = (y <= q_lo).float().mean()

    # Calibration objective (empirical version of Eq. 6):
    # if p_hat < tau => quantile too low => push it up via (y - q_lo)_+
    C_up = F.relu(y - q_lo).mean()
    # else => quantile too high => push it down via (q_lo - y)_+
    C_dn = F.relu(q_lo - y).mean()
    C = C_up if (p_hat.item() < tau) else C_dn

    # Sharpness: penalize width only if interval "overcovers"
    # observed coverage between q_lo and q_hi
    in_int = ((y >= q_lo) & (y <= q_hi)).float().mean()
    target_cov = 1.0 - 2.0 * tau     # expected coverage for [tau, 1-tau]
    width = (q_hi - q_lo).mean()     # = gap.mean()

    # gate sharpness penalty when observed coverage exceeds expected (paper’s idea)
    P = F.relu(in_int - target_cov) * width

    return (1.0 - mix_kappa) * C + mix_kappa * P