# src/models/nhits_qra/qra/qra_diagnostics.py
from __future__ import annotations
from typing import List, Optional, Sequence, Dict, Any
import numpy as np
from pathlib import Path

def dump_and_log_nhits_point_forecasts(
    *,
    X_per_h: List[Optional[np.ndarray]],
    meta: Dict,
    out_dir: Path,
    split: str,
    log,
    n_points: int,
    preview_rows: int = 0,
    dump: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    uid = np.asarray(meta.get("unique_id", []))
    ods = np.asarray(meta.get("origin_ds", []))

    # ---- accumulators for ONE summary line ----
    agg = {
        "H_used": 0,
        "m_used": [],
        "mean_col_std": [],
        "mean_row_std": [],
        "mean_abs_corr_offdiag": [],
        "eff_rank": [],          # optional, robust summary
    }

    for h, X in enumerate(X_per_h):
        if not (isinstance(X, np.ndarray) and X.size):
            continue

        m = min(int(n_points), int(X.shape[1]))
        if m <= 0:
            continue

        PF = X[:, :m].astype(np.float64, copy=False)  # (N,m)

        # diversity stats
        col_std = PF.std(axis=0)            # (m,)
        row_std = PF.std(axis=1)            # (N,)
        C = np.corrcoef(PF, rowvar=False)   # (m,m)
        mean_abs_offdiag = float(np.abs(C - np.eye(m)).sum() / (m * m - m))

        # effective rank (0..m): high => diverse; low => redundant
        PF0 = PF - PF.mean(axis=0, keepdims=True)
        s = np.linalg.svd(PF0, full_matrices=False, compute_uv=False)
        p = (s**2) / np.maximum(1e-12, (s**2).sum())
        eff_rank = float(np.exp(-(p * np.log(np.maximum(p, 1e-12))).sum()))

        # accumulate
        agg["H_used"] += 1
        agg["m_used"].append(m)
        agg["mean_col_std"].append(float(col_std.mean()))
        agg["mean_row_std"].append(float(row_std.mean()))
        agg["mean_abs_corr_offdiag"].append(mean_abs_offdiag)
        agg["eff_rank"].append(eff_rank)

        # ---- optionally dump per-horizon matrices (unchanged) ----
        if dump:
            import pandas as pd
            df = pd.DataFrame(PF, columns=[f"pf_{j:03d}" for j in range(m)])
            if uid.size == len(df):
                df.insert(0, "unique_id", uid)
            if ods.size == len(df):
                df.insert(1, "origin_ds", ods.astype("datetime64[s]").astype(str))
            path = out_dir / f"{split}_nhits_point_forecasts_h{h:02d}.csv"
            df.to_csv(path, index=False)

            # keep preview off by default (it’s huge)
            if preview_rows > 0:
                log.info("NHITS-PF[%s] h=%d preview:\n%s", split, h, df.head(preview_rows).to_string(index=False))

    # ---- ONE summary log line ----
    if agg["H_used"] == 0:
        log.info("NHITS-PF[%s]: no horizons with point-forecast data.", split)
        return

    log.info(
        "NHITS-PF[%s] diversity summary: H_used=%d | m(median)=%d | "
        "mean_col_std=%.4g | mean_row_std=%.4g | mean_abs_corr_offdiag=%.4f | eff_rank(median)=%.2f",
        split,
        agg["H_used"],
        int(np.median(agg["m_used"])),
        float(np.mean(agg["mean_col_std"])),
        float(np.mean(agg["mean_row_std"])),
        float(np.mean(agg["mean_abs_corr_offdiag"])),
        float(np.median(agg["eff_rank"])),
    )



def _to_unit_taus(taus):
    t = np.asarray(taus, float)
    if t.size and t.max() > 1.0:
        t = t / 100.0
    return t

def summarize_design(Z: np.ndarray) -> dict:
    Z = np.asarray(Z)
    if Z.size == 0:
        return {"empty": True}
    # avoid huge cost: approximate rank via SVD on small subset
    n = Z.shape[0]
    m = min(n, 4096)
    idx = np.random.default_rng(0).choice(n, size=m, replace=False) if n > m else np.arange(n)
    Zs = Z[idx]
    # center columns for diagnostics
    Zc = Zs - Zs.mean(axis=0, keepdims=True)
    # singular values
    s = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
    s = np.asarray(s)
    # effective rank (entropy)
    p = (s**2) / (s**2).sum() if (s**2).sum() > 0 else np.zeros_like(s)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum())) if p.size else 0.0

    col_std = np.std(Zs, axis=0)
    return {
        "shape": tuple(Z.shape),
        "col_std_min": float(col_std.min()) if col_std.size else 0.0,
        "col_std_med": float(np.median(col_std)) if col_std.size else 0.0,
        "col_std_max": float(col_std.max()) if col_std.size else 0.0,
        "sv_first": float(s[0]) if s.size else 0.0,
        "sv_last": float(s[-1]) if s.size else 0.0,
        "cond_approx": float(s[0] / (s[-1] + 1e-12)) if s.size else np.inf,
        "eff_rank_approx": eff_rank,
    }

def summarize_quantiles(y: np.ndarray, Q: np.ndarray, taus) -> dict:
    y = np.asarray(y).reshape(-1)
    Q = np.asarray(Q)
    t = _to_unit_taus(taus)
    if Q.size == 0:
        return {"empty": True}

    # assume taus sorted ascending
    q_lo = Q[:, 0]
    q_hi = Q[:, -1]
    width = q_hi - q_lo

    # coverage checks
    cov_lo = float(np.mean(y <= q_lo))
    cov_hi = float(np.mean(y <= q_hi))
    outside = float(np.mean((y < q_lo) | (y > q_hi)))

    # variance checks
    return {
        "N": int(y.shape[0]),
        "y_std": float(np.std(y)),
        "q_lo_std": float(np.std(q_lo)),
        "q_hi_std": float(np.std(q_hi)),
        "width_mean": float(np.mean(width)),
        "width_std": float(np.std(width)),
        "cov_y_le_qlo": cov_lo,
        "cov_y_le_qhi": cov_hi,
        "outside_[qlo,qhi]": outside,
        "tau_lo": float(t[0]) if t.size else None,
        "tau_hi": float(t[-1]) if t.size else None,
    }


def summarize_coverage(y: np.ndarray, Q: np.ndarray, taus) -> dict:
    """
    y: (N,)
    Q: (N, T) with columns aligned with taus
    taus: list/array in [0,1] or [1..99] (percent)
    """
    y = np.asarray(y).reshape(-1)
    Q = np.asarray(Q)
    t = np.asarray(taus, dtype=float)
    if t.max() > 1.0:
        t = t / 100.0

    cov = (y[:, None] <= Q).mean(axis=0)  # (T,)
    err = cov - t
    return {
        "N": int(len(y)),
        "taus": t.tolist(),
        "cov": cov.tolist(),
        "err": err.tolist(),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }

def monotonicity_violations(Q: np.ndarray) -> float:
    # Q is (N, T) over increasing taus
    return float(np.mean(np.any(np.diff(Q, axis=1) < 0.0, axis=1)))


def collapse_stats(Q: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    """
    Q: (N, T) with taus increasing
    """
    d = np.diff(Q, axis=1)
    return {
        "mean_abs_step": float(np.mean(np.abs(d))),
        "frac_small_step": float(np.mean(np.abs(d) < eps)),
    }

def log_qra_quantile_diagnostics(
    *,
    log,
    split: str,
    hs: Sequence[int],
    # RAW: before postprocess / monotone fix
    Q_raw_per_h: Optional[List[Optional[np.ndarray]]] = None,
    taus_raw: Optional[Sequence[float]] = None,
    # FIX: after postprocess / monotone fix
    Q_fix_per_h: Optional[List[Optional[np.ndarray]]] = None,
    taus_fix: Optional[Sequence[float]] = None,
    # y/Q in "scaled space" (whatever model sees)
    y_scaled_per_h: Optional[List[Optional[np.ndarray]]] = None,
    Q_scaled_for_summary_per_h: Optional[List[Optional[np.ndarray]]] = None,
    # y/Q in real scale (after inverse scaling)
    y_real_per_h: Optional[List[Optional[np.ndarray]]] = None,
    Q_real_per_h: Optional[List[Optional[np.ndarray]]] = None,
    # knobs
    collapse_eps: float = 1e-6,
    log_cov_vec: bool = True,
    tag: str = "DBG",
) -> None:
    """
    One function to log:
      - monotonicity violations (RAW + FIX if present)
      - collapse stats (RAW + FIX if present)
      - summarize_quantiles (scaled) if y_scaled + Q_scaled provided
      - coverage metrics (real) if y_real + Q_real provided

    Designed to be called from both run_train_val and run_test.
    """
    def _get(per_h, h):
        if per_h is None:
            return None
        if h < 0 or h >= len(per_h):
            return None
        return per_h[h]

    # ------------- RAW -------------
    if Q_raw_per_h is not None:
        for h in hs:
            Qraw = _get(Q_raw_per_h, h)
            if isinstance(Qraw, np.ndarray) and Qraw.size:
                viol = monotonicity_violations(Qraw)
                log.info(f"[{tag}][{split}][mono][RAW][h={h}] viol_rate={viol:.4f}")

                cs = collapse_stats(Qraw, eps=collapse_eps)
                log.info(
                    f"[{tag}][{split}][collapse][RAW][h={h}] "
                    f"mean_abs_step={cs['mean_abs_step']:.3e} frac_|step|<{collapse_eps:g}={cs['frac_small_step']:.3f}"
                )

    # ------------- FIX -------------
    if Q_fix_per_h is not None:
        for h in hs:
            Qfix = _get(Q_fix_per_h, h)
            if isinstance(Qfix, np.ndarray) and Qfix.size:
                viol = monotonicity_violations(Qfix)
                log.info(f"[{tag}][{split}][mono][FIX][h={h}] viol_rate={viol:.4f}")

                cs = collapse_stats(Qfix, eps=collapse_eps)
                log.info(
                    f"[{tag}][{split}][collapse][FIX][h={h}] "
                    f"mean_abs_step={cs['mean_abs_step']:.3e} frac_|step|<{collapse_eps:g}={cs['frac_small_step']:.3f}"
                )

    # ------------- scaled quantile summaries (optional) -------------
    if (y_scaled_per_h is not None) and (Q_scaled_for_summary_per_h is not None) and (taus_fix is not None):
        for h in hs:
            y = _get(y_scaled_per_h, h)
            Q = _get(Q_scaled_for_summary_per_h, h)
            if isinstance(y, np.ndarray) and isinstance(Q, np.ndarray) and y.size and Q.size:
                log.info(f"[{tag}][{split}][scaled][h={h}] {summarize_quantiles(y, Q, list(taus_fix))}")

    # ------------- coverage on real scale (optional) -------------
    if (y_real_per_h is not None) and (Q_real_per_h is not None) and (taus_fix is not None):
        for h in hs:
            y = _get(y_real_per_h, h)
            Q = _get(Q_real_per_h, h)
            if isinstance(y, np.ndarray) and isinstance(Q, np.ndarray) and y.size and Q.size:
                cov_info = summarize_coverage(y, Q, list(taus_fix))
                log.info(f"[{tag}][{split}][COV][h={h}] mae={cov_info['mae']:.4f} rmse={cov_info['rmse']:.4f}")
                if log_cov_vec:
                    log.info(
                        f"[{tag}][{split}][COVVEC][h={h}] "
                        f"cov={np.round(np.array(cov_info['cov']), 4).tolist()} "
                        f"taus={np.round(np.array(cov_info['taus']), 4).tolist()}"
                    )