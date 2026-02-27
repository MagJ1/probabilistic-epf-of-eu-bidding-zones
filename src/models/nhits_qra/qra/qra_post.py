from __future__ import annotations
import numpy as np
from typing import List, Optional
import torch
from scipy.stats import norm
from models.nhits_qra.qra.qra_prep import make_quantile_grid

try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_ISO = True
except Exception:
    _HAS_ISO = False

try:
    from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _ensure_tau_array(taus):
    t = np.asarray(taus, dtype=float)
    if t.max() > 1.0:  # treat as percent
        t = t / 100.0
    return t

def repair_monotone_rows(Q: np.ndarray, method: str = "isotonic") -> np.ndarray:
    """Enforce non-decreasing quantiles per row."""
    if Q is None or Q.size == 0:
        return Q
    if method == "cmax":
        return np.maximum.accumulate(Q, axis=1)
    if method == "isotonic":
        if not _HAS_ISO:
            # fallback: cmax
            return np.maximum.accumulate(Q, axis=1)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        Qf = np.empty_like(Q)
        xs = np.arange(Q.shape[1], dtype=float)
        for i in range(Q.shape[0]):
            Qf[i] = iso.fit_transform(xs, Q[i])
        return Qf
    raise ValueError(f"Unknown monotonic repair method: {method}")

def interp_quantiles_rows(Q_src: np.ndarray, taus_src, taus_tgt, method: str = "linear") -> np.ndarray:
    """
    Row-wise interpolation from taus_src -> taus_tgt. No repair inside.
    method: 'linear' (np.interp), 'pchip' (monotone shape-preserving), 'akima' (smooth).
    """
    if Q_src is None or Q_src.size == 0:
        return Q_src
    t_src = _ensure_tau_array(taus_src)
    t_tgt = _ensure_tau_array(taus_tgt)
    # clamp targets to avoid extrapolation
    t_tgt_clip = np.clip(t_tgt, t_src.min(), t_src.max())
    out = np.empty((Q_src.shape[0], len(t_tgt)), dtype=Q_src.dtype)

    use_pchip = (method.lower() == "pchip") and _HAS_SCIPY
    use_akima = (method.lower() == "akima") and _HAS_SCIPY

    for i in range(Q_src.shape[0]):
        yi = Q_src[i]
        if use_pchip:
            f = PchipInterpolator(t_src, yi, extrapolate=False)
            out[i] = f(t_tgt_clip)
        elif use_akima:
            f = Akima1DInterpolator(t_src, yi)
            out[i] = f(t_tgt_clip)
        else:
            out[i] = np.interp(t_tgt_clip, t_src, yi)
    return out

def interpolate_and_enforce(Q_per_h: List[Optional[np.ndarray]],
                            taus_src,
                            taus_tgt,
                            repair_method: str = "cmax",
                            final_repair: bool = True,
                            interp: str = "linear",
                            enforce: bool = True) -> List[Optional[np.ndarray]]:
    """
    Per horizon:
      (optional) repair on source grid
      interpolate to target grid
      (optional) repair again on target grid
    """
    out = []
    for Q in Q_per_h:
        if Q is None:
            out.append(None); continue
        Q1 = repair_monotone_rows(Q, method=repair_method) if enforce else Q
        Q2 = interp_quantiles_rows(Q1, taus_src, taus_tgt, method=interp)
        if enforce and final_repair:
            Q2 = repair_monotone_rows(Q2, method=repair_method)
        out.append(Q2)
    return out

def _ensure_tau_array(taus):
    t = np.asarray(taus, dtype=float)
    if t.max() > 1.0:  # allow [1..99] style
        t = t / 100.0
    return t

def samples_from_quantiles(
    Q_per_h,          # list of arrays, each (N, T) for horizon h
    taus,             # list/array of length T
    n_samples: int,   # S
    device: str = "cpu",
):
    """
    Returns: samples_per_h = list of torch.Tensor, each (S, N)
    """
    t = _ensure_tau_array(taus)
    t_torch = torch.as_tensor(t, dtype=torch.float32, device=device)  # (T,)

    out = []
    for Q in Q_per_h:
        if Q is None or Q.size == 0:
            out.append(None); continue

        Q_t = torch.as_tensor(Q, dtype=torch.float32, device=device)  # (N, T)
        N, T = Q_t.shape

        # 1) uniforms to invert
        U = torch.rand((n_samples, N), device=device)  # (S, N)

        # 2) find right bucket (index of upper end j1)
        # searchsorted returns index in 0..T where U would be inserted in sorted taus
        j1 = torch.searchsorted(t_torch, U)  # (S, N)
        j1 = torch.clamp(j1, 1, T-1)         # keep inside [1, T-1]
        j0 = j1 - 1                          

        # 3) gather local taus and quantile values
        tau0 = t_torch[j0]                   # (S, N)
        tau1 = t_torch[j1]                   # (S, N)

        # need to index rows of Q per column j0/j1
        # expand row indices to (S,N) to pair with j0/j1
        row = torch.arange(N, device=device).view(1, N).expand(n_samples, N)  # (S,N)
        q0 = Q_t[row, j0]  # (S, N)
        q1 = Q_t[row, j1]  # (S, N)

        # 4) linear interpolation in value-space
        w = (U - tau0) / (tau1 - tau0 + 1e-12)
        X = q0 + w * (q1 - q0)               # (S, N)

        out.append(X)  # (S, N)

    return out  # list[(S, N)]


def pits_from_quantiles(
    y_per_h,   # list of arrays, each (N,)
    Q_per_h,   # list of arrays, each (N, T) monotone across T
    taus,      # length T (0..1 or 1..99)
    eps: float = 1e-6,
):
    """
    Returns: U_per_h (list of arrays), each (N,) PITs in (0,1).
    """
    t = _ensure_tau_array(taus)
    U_out = []
    for y, Q in zip(y_per_h, Q_per_h):
        if Q is None or Q.size == 0:
            U_out.append(None); continue
        y = np.asarray(y, dtype=np.float64)       # (N,)
        Q = np.asarray(Q, dtype=np.float64)       # (N, T)
        N, T = Q.shape
        U = np.empty(N, dtype=np.float64)
        rng = np.random.default_rng(1)
        for i in range(N):
            yi = y[i]
            Qi = Q[i]
            # left/right of support
            if yi <= Qi[0]:
                U[i] = rng.uniform(0.0, float(t[0]))
                continue
            if yi >= Qi[-1]:
                U[i] = rng.uniform(float(t[-1]), 1.0)
                continue
            # find j s.t. Qi[j-1] <= yi < Qi[j]
            j = np.searchsorted(Qi, yi, side="right")
            j = min(max(j, 1), T-1)
            q0, q1 = Qi[j-1], Qi[j]
            tau0, tau1 = t[j-1], t[j]
            w = (yi - q0) / (q1 - q0 + 1e-12)
            U[i] = tau0 + w * (tau1 - tau0)

        U = np.clip(U, eps, 1.0 - eps)
        U_out.append(U)  # (N,)

    return U_out  # list[(N,)]


def taus_to_unit(taus: List[float]) -> np.ndarray:
    t = np.asarray(taus, dtype=float)
    return t / 100.0 if t.max() > 1.0 else t


def postprocess_quantiles(Q_per_h, taus_src, knobs: QRAKnobs, cfg: OmegaConf):
    taus_tgt = taus_src if (knobs.target_quantiles_spec is None) else make_quantile_grid(knobs.target_quantiles_spec)
    if bool(cfg.train_qra.enforce_monotonicity):
        Q_fix = interpolate_and_enforce(
            Q_per_h=Q_per_h,
            taus_src=taus_src,
            taus_tgt=taus_tgt,
            repair_method=str(cfg.train_qra.repair_method),
            final_repair=True,
            interp=str(cfg.train_qra.interp_method),
            enforce=True,
        )
    else:
        Q_fix = Q_per_h
    return Q_fix, taus_tgt

def samples_from_q(Q_real_per_h, taus_tgt, n_samples: int, device: str):
    taus_unit = taus_to_unit(taus_tgt)
    return samples_from_quantiles(Q_real_per_h, taus_unit, n_samples=n_samples, device=device)