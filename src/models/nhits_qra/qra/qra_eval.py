# in src/models/nhits_qra/qra/qra_eval.py
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import json
import numpy as np
from scipy.stats import norm
from pathlib import Path
import torch

from utils.metrics import crps_terms_fast, sliced_energy_score, berkowitz_lr_test, fisher_method, harmonic_mean_pvalue
from models.nhits_qra.qra.qra_post import pits_from_quantiles, taus_to_unit


def berkowitz_from_pits(y_real_per_h, Q_real_per_h, taus_tgt, horizons="all") -> dict:
    """
    horizons: "all" | "first" | "last" | List[int]
      - "all": concatenate PITs across all horizons
      - "first": only horizon 0 -> T_used == #kept_origins, so only first step of forecast is used
      - "last":  only horizon H-1, so only last step of forecast is used
      - List[int]: explicit horizons subset (e.g., [0, 23]), specific steps of forecasts are used
    """
    U_per_h = pits_from_quantiles(y_real_per_h, Q_real_per_h, taus_tgt)

    H = len(U_per_h)
    if isinstance(horizons, str):
        if horizons == "first":
            hs = [0]
        elif horizons == "last":
            hs = [H - 1]
        else:
            hs = list(range(H))
    else:
        hs = [h for h in horizons if 0 <= h < H]

    chosen = [U_per_h[h] for h in hs if (U_per_h[h] is not None and U_per_h[h].size)]
    U = np.concatenate(chosen, axis=0) if chosen else np.array([])
    U = np.clip(U, 1e-6, 1 - 1e-6)
    z = norm.ppf(U) if U.size else np.array([])

    if z.size:
        res = berkowitz_lr_test(z)
        out = dict(
            LR=float(res.get("LR", np.nan)),
            df=int(res.get("df", 3)),
            p_value=float(res.get("p", np.nan)),
            T_used=int(res.get("T", 0)),
            mu_hat=float(res.get("mu", np.nan)),
            rho_hat=float(res.get("rho", np.nan)),
            sigma2_hat=float(res.get("sigma2", np.nan)),
            horizons_used=hs,
        )
    else:
        out = dict(
            LR=np.nan, df=3, p_value=np.nan, T_used=0,
            mu_hat=np.nan, rho_hat=np.nan, sigma2_hat=np.nan,
            horizons_used=hs,
        )

    # Always include arrays -> stable API forever
    out["U"] = U
    out["z"] = z
    return out


def berkowitz_suite(y_real_per_h, Q_real_per_h, taus_tgt) -> dict:
    hs_eff = [
        h for h, (y, Q) in enumerate(zip(y_real_per_h, Q_real_per_h))
        if isinstance(y, np.ndarray) and y.size and isinstance(Q, np.ndarray) and Q.size
    ]

    per_h = []
    pvals = []

    for h in hs_eff:
        res_h = berkowitz_from_pits(y_real_per_h, Q_real_per_h, taus_tgt, horizons=[h])
        p = res_h.get("p_value", np.nan)
        per_h.append({
            "h": h,
            "LR": res_h.get("LR"),
            "df": res_h.get("df", 3),
            "p":  p,
            "T_used": res_h.get("T_used"),
            "mu": res_h.get("mu_hat"),
            "rho": res_h.get("rho_hat"),
            "sigma2": res_h.get("sigma2_hat"),
        })
        pvals.append(p)

    fisher = fisher_method(pvals, min_count=1)
    fisher["hmp"] = harmonic_mean_pvalue(pvals)

    return {"per_h": per_h, "fisher": fisher, "hs_eff": hs_eff}


def write_berkowitz_files(metrics_dir: Path, prefix: str, berk: dict):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{prefix}_berkowitz_per_h.json").write_text(
        json.dumps({"per_h": berk.get("per_h", [])}, indent=2)
    )
    (metrics_dir / f"{prefix}_berkowitz_fisher.json").write_text(
        json.dumps({"fisher": berk.get("fisher", {})}, indent=2)
    )


def compute_crps_es(samples_per_h, y_real_per_h, device: str):
    """
    samples_per_h: list of tensors, each (N,S) or (S,N,1) or (N,S,1) or (S,N)
    y_real_per_h:  list of arrays, each (N,)
    Returns:
      mean_crps,
      crps_per_h (len=H, entries float or None),
      mean_es,
      es_per_h   (len=H, entries float or None),
      es_bh (optional, currently None)
    """
    H = len(y_real_per_h)
    crps_per_h: List[Optional[float]] = []
    es_per_h:   List[Optional[float]] = []

    # we'll collect per-horizon per-origin vectors here
    crps_cols = []
    es_cols   = []

    def _to_S_N_1(X: torch.Tensor, N: int) -> torch.Tensor:
        if X.dim() == 2:
            if X.shape[0] == N:          # (N,S) -> (S,N,1)
                return X.transpose(0, 1).unsqueeze(-1)
            if X.shape[1] == N:          # (S,N) -> (S,N,1)
                return X.unsqueeze(-1)
            raise RuntimeError(f"Cannot infer (S,N) from {tuple(X.shape)} with N={N}")
        if X.dim() == 3:
            if X.shape[-1] != 1:
                raise RuntimeError(f"Expected last dim=1, got {tuple(X.shape)}")
            if X.shape[0] == N:          # (N,S,1) -> (S,N,1)
                return X.transpose(0, 1)
            if X.shape[1] == N:          # (S,N,1)
                return X
            raise RuntimeError(f"Cannot align 3D samples {tuple(X.shape)} with N={N}")
        raise RuntimeError(f"Unexpected samples ndim={X.dim()}")

    N_ref = None  # enforce consistent N across horizons we keep

    for h, (X_sn, y_n) in enumerate(zip(samples_per_h, y_real_per_h)):

        if not (isinstance(y_n, np.ndarray) and y_n.size) or X_sn is None or (
            torch.is_tensor(X_sn) and X_sn.numel() == 0
        ):
            crps_per_h.append(None)
            es_per_h.append(None)
            continue

        N = int(len(y_n))
        if N_ref is None:
            N_ref = N
        elif N_ref != N:
            # this should not happen if masking is consistent
            raise RuntimeError(f"Inconsistent N across horizons: got {N} but expected {N_ref}")

        y = torch.as_tensor(y_n, device=device, dtype=torch.float32).unsqueeze(-1)  # (N,1)
        X = _to_S_N_1(X_sn, N)  # (S,N,1)

        fit_c, spread_c = crps_terms_fast(X, y)

        # per-origin per-horizon CRPS: keep vector shape (N,)
        crps_vec = (fit_c - spread_c).reshape(-1)  # (N,)
        crps_h = float(crps_vec.mean().item())
        crps_per_h.append(crps_h)

        # In 1D, ES == CRPS (beta=1) so we can reuse
        es_vec = crps_vec
        es_h = float(es_vec.mean().item())
        es_per_h.append(es_h)

        crps_cols.append(crps_vec.detach().cpu().numpy())
        es_cols.append(es_vec.detach().cpu().numpy())

    # aggregate across horizons
    crps_vals = [v for v in crps_per_h if v is not None]
    es_vals   = [v for v in es_per_h   if v is not None]
    mean_crps = float(np.mean(crps_vals)) if crps_vals else float("nan")
    mean_es   = float(np.mean(es_vals))   if es_vals   else float("nan")

    # stack per-origin arrays to (N, H_eff)
    crps_bh = None
    es_bh = None
    if crps_cols:
        crps_bh = np.stack(crps_cols, axis=1)  # (N, H_eff)
    if es_cols:
        es_bh = np.stack(es_cols, axis=1)      # (N, H_eff)

    extra = {"crps_bh": crps_bh, "es_bh": es_bh}
    return mean_crps, crps_per_h, mean_es, es_per_h, extra


def write_score_files(
    metrics_dir: Path,
    prefix: str,
    mean_crps,
    crps_list,
    mean_es,
    es_list,
    n_samples_val: int,
    ):
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / f"{prefix}_crps_per_h.json").write_text(
        json.dumps([None if v is None else float(v) for v in crps_list], indent=2)
    )
    (metrics_dir / f"{prefix}_crps_summary.json").write_text(
        json.dumps(
            {
                "mean_crps": float(mean_crps),
                "n_horizons_with_data": int(sum(v is not None for v in crps_list)),
                "n_samples_per_h": int(n_samples_val),
            },
            indent=2,
        )
    )

    (metrics_dir / f"{prefix}_es_per_h.json").write_text(
        json.dumps([None if v is None else float(v) for v in es_list], indent=2)
    )
    (metrics_dir / f"{prefix}_es_summary.json").write_text(
        json.dumps(
            {
                "mean_es": float(mean_es),
                "beta": 1.0,
                "n_horizons_with_data": int(sum(v is not None for v in es_list)),
                "n_samples_per_h": int(n_samples_val),
            },
            indent=2,
        )
    )


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Mean pinball loss for diagnostics/eval."""
    u = y_true - y_pred
    return float(np.mean(np.maximum(tau*u, (tau-1)*u)))

def compute_pinball_per_h(
    *,
    y_real_per_h: List[Optional[np.ndarray]],
    Q_real_per_h: List[Optional[np.ndarray]],
    taus: Sequence[float],
) -> Tuple[float, List[Optional[float]]]:
    """
    Computes a single pinball score per horizon:
      mean over taus of mean pinball over samples N.

    Returns:
      mean_pinball (over available horizons),
      pinball_per_h list aligned with horizons (None if missing).
    """
    taus_unit = taus_to_unit(taus)
    pinball_per_h: List[Optional[float]] = []

    for y, Q in zip(y_real_per_h, Q_real_per_h):
        if not (isinstance(y, np.ndarray) and y.size and isinstance(Q, np.ndarray) and Q.size):
            pinball_per_h.append(None)
            continue

        # Q: (N, n_tau)
        if Q.ndim != 2 or Q.shape[1] != len(taus_unit):
            raise ValueError(f"Expected Q shape (N,{len(taus_unit)}), got {Q.shape}")

        losses_tau = []
        for j, tau in enumerate(taus_unit):
            losses_tau.append(float(pinball_loss(y, Q[:, j], tau=float(tau))))
        pinball_per_h.append(float(np.mean(losses_tau)))

    vals = [v for v in pinball_per_h if v is not None]
    mean_pinball = float(np.mean(vals)) if vals else float("nan")
    return mean_pinball, pinball_per_h


def write_pinball_files(
    metrics_dir: Path,
    prefix: str,
    *,
    mean_pinball: float,
    pinball_per_h: List[Optional[float]],
    taus: Sequence[float],
) -> None:
    """
    Writes:
      <prefix>_pinball_per_h.json     (verbose)
      <prefix>_pinball_summary.json   (summary)
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / f"{prefix}_pinball_per_h.json").write_text(
        json.dumps([None if v is None else float(v) for v in pinball_per_h], indent=2)
    )
    (metrics_dir / f"{prefix}_pinball_summary.json").write_text(
        json.dumps(
            {
                "mean_pinball": float(mean_pinball),
                "n_horizons_with_data": int(sum(v is not None for v in pinball_per_h)),
                "n_taus": int(len(list(taus))),
                "taus": [float(t) for t in taus],
            },
            indent=2,
        )
    )


def write_ece_files(
    metrics_dir: Path,
    prefix: str,
    mean_ece: float,
    ece_per_h: List[Optional[float]],
    n_samples_per_h: int,
    n_taus: int,
):
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / f"{prefix}_ece_per_h.json").write_text(
        json.dumps([None if v is None else float(v) for v in ece_per_h], indent=2)
    )
    (metrics_dir / f"{prefix}_ece_summary.json").write_text(
        json.dumps(
            {
                "mean_ece": float(mean_ece),
                "n_horizons_with_data": int(sum(v is not None for v in ece_per_h)),
                "n_samples_per_h": int(n_samples_per_h),
                "n_taus": int(n_taus),
            },
            indent=2,
        )
    )
