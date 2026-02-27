# tools/dm_compare.py
"""
Compare two per-origin forecast runs using Diebold-Mariano with Newey-West HAC.

Inputs
------
- Two CSV/Parquet tables produced by test phase:
  * For ES: columns include ['unique_id', 'origin_ds', 'es_mean']
  * For CRPS: columns include ['unique_id', 'origin_ds', 'crps_mean']

What it does
------------
1) Reads the two files, coerces origin_ds to datetime.
2) (Optional) Thins by --anchor-hour (keep only that clock hour) or --origin-step (keep every k-th origin).
3) Inner-joins on (unique_id, origin_ds).
4) Forms paired loss differences d_t = cand - base and runs DM with NW(HAC, Bartlett).

Usage
-----
python -m utils.dm_compare \
  --base /path/to/base_es_per_origin.parquet \
  --cand /path/to/cand_es_per_origin.parquet \
  --metric es \
  --anchor-hour 0 \
  --alpha 0.05 \
  --nw-lag-rule "T**0.25" \
  --out-csv diffs.csv \
  --out-json summary.json

Notes
-----
- If --metric is omitted, the script tries to infer ('es' if 'es_mean' present, else 'crps' if 'crps_mean' present).
- Newey-West lag defaults to floor(T**0.25). One can pass an integer or "T**(1/3)".
"""

from __future__ import annotations
import argparse
from math import floor
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import norm


def _read_any_table(p: Path) -> pd.DataFrame:
    p = Path(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {p.suffix} (path={p})")


def _filter_nonoverlapping_origins(
    df: pd.DataFrame, anchor_hour: Optional[int] = None, step: Optional[int] = None
) -> pd.DataFrame:
    """Keep only origins at a specific hour, or every k-th origin if step>1."""
    out = df.copy()
    if "origin_ds" not in out.columns:
        raise ValueError("Expected column 'origin_ds' in per-origin file.")
    out["origin_ds"] = pd.to_datetime(out["origin_ds"])

    if anchor_hour is not None:
        out = out[out["origin_ds"].dt.hour == int(anchor_hour)]
    elif step is not None and int(step) > 1:
        step = int(step)
        out = out.sort_values(["unique_id", "origin_ds"])
        out = (
            out.assign(_r=out.groupby("unique_id").cumcount())
               .query("_r % @step == 0")
               .drop(columns="_r")
        )
    return out


def _infer_metric(base_cols: pd.Index, cand_cols: pd.Index, explicit: Optional[str]) -> Tuple[str, str]:
    """
    Decide metric and return (metric_name, value_col) where value_col is 'es_mean' or 'crps_mean'.
    """
    if explicit:
        m = explicit.strip().lower()
        if m not in ("es", "crps"):
            raise ValueError("--metric must be 'es' or 'crps'")
        col = "es_mean" if m == "es" else "crps_mean"
        if col not in base_cols or col not in cand_cols:
            raise ValueError(f"Metric '{m}' selected but '{col}' not found in both files.")
        return m, col

    # Auto
    if "es_mean" in base_cols and "es_mean" in cand_cols:
        return "es", "es_mean"
    if "crps_mean" in base_cols and "crps_mean" in cand_cols:
        return "crps", "crps_mean"
    raise ValueError(
        "Could not infer metric. Pass --metric es|crps or ensure both files contain either 'es_mean' or 'crps_mean'."
    )


def _nw_lag(T: int, rule: str = "T**0.25") -> int:
    rule = str(rule).strip()
    if rule == "T**0.25":
        return max(0, floor(T ** 0.25))
    if rule in ("T**(1/3)", "T**0.3333"):
        return max(0, floor(T ** (1/3)))
    # integer fallback
    try:
        return max(0, int(rule))
    except Exception:
        return max(0, floor(T ** 0.25))


def _dm_newey_west(d: np.ndarray, lag: Optional[int] = None, nw_lag_rule: str = "T**0.25") -> dict:
    """
    Diebold-Mariano via intercept-only OLS with HAC(Newey-West, Bartlett).
    Input d is 1D array of paired loss diffs ordered in time.
    """
    d = np.asarray(d, dtype=float).ravel()
    T = d.size
    X = np.ones((T, 1))  # intercept only
    model = sm.OLS(d, X, hasconst=True)

    if lag is None:
        lag = _nw_lag(T, nw_lag_rule)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\..*")
        res = model.fit()
        try:
            V = cov_hac(res, nlags=int(lag))  # Bartlett kernel by default
        except Exception:
            V = np.array([[np.nan]])

    alpha_hat = float(res.params[0])
    se_alpha = float(np.sqrt(V[0, 0])) if np.isfinite(V[0, 0]) else np.nan
    if not np.isfinite(se_alpha) or se_alpha == 0.0:
        zstat = np.nan
        pval = np.nan
    else:
        zstat = alpha_hat / se_alpha
        pval = 2 * (1 - student_t.cdf(abs(zstat), df=max(T - 1, 1)))

    return {"T": int(T), "lag": int(lag), "mean": alpha_hat, "se": se_alpha,"stat": float(zstat), "p_value": float(pval)}


def compare_two_files(
    base_path: Path,
    cand_path: Path,
    metric: Optional[str] = None,          # 'es' or 'crps' or None to infer
    anchor_hour: Optional[int] = None,     # e.g., 0 for midnight only
    origin_step: int = 1,                  # keep every k-th origin if no anchor_hour
    nw_lag_rule: str = "T**0.25",
    alpha: float = 0.05,
) -> dict:
    """Core function: returns dict with summary, paired df, and decision."""
    b = _read_any_table(base_path)
    c = _read_any_table(cand_path)

    # pick metric and value column
    metric_name, val_col = _infer_metric(b.columns, c.columns, metric)

    # normalize types and timestamps
    for df in (b, c):
        if "origin_ds" not in df.columns:
            raise ValueError("Expected column 'origin_ds' in input files.")
        df["origin_ds"] = pd.to_datetime(df["origin_ds"])

    # coerce unique_id types if they differ
    if b["unique_id"].dtype != c["unique_id"].dtype:
        b["unique_id"] = b["unique_id"].astype(str)
        c["unique_id"] = c["unique_id"].astype(str)

    # optional thinning
    if anchor_hour is not None or (origin_step is not None and int(origin_step) > 1):
        b = _filter_nonoverlapping_origins(b, anchor_hour=anchor_hour, step=None if anchor_hour is not None else origin_step)
        c = _filter_nonoverlapping_origins(c, anchor_hour=anchor_hour, step=None if anchor_hour is not None else origin_step)

    # join & diffs
    df = (
        b.rename(columns={val_col: f"base_{val_col}"})
         .merge(c.rename(columns={val_col: f"cand_{val_col}"}), on=["unique_id", "origin_ds"], how="inner")
         .dropna(subset=[f"base_{val_col}", f"cand_{val_col}"])
         .sort_values(["unique_id", "origin_ds"], kind="mergesort")
         .reset_index(drop=True)
    )
    df["loss_diff"] = df[f"cand_{val_col}"] - df[f"base_{val_col}"]

    T = len(df)
    lag = _nw_lag(T, nw_lag_rule)
    dm = _dm_newey_west(df["loss_diff"].to_numpy(), lag=lag, nw_lag_rule=nw_lag_rule)

    mean_delta = float(df["loss_diff"].mean()) if T else float("nan")
    median_delta = float(df["loss_diff"].median()) if T else float("nan")

    if np.isfinite(dm.get("se", np.nan)) and np.isfinite(mean_delta):
        crit = float(norm.ppf(1 - alpha/2.0))
        ci_lo = mean_delta - crit * dm["se"]
        ci_hi = mean_delta + crit * dm["se"]
    else:
        ci_lo, ci_hi = float("nan"), float("nan")
    
    adopt = (dm["p_value"] < float(alpha)) and (mean_delta < 0.0)

    summary = {
        "metric": f"{metric_name}_mean",
        "T": int(T),
        "lag": int(dm["lag"]),
        "alpha": float(alpha),
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "dm_stat": float(dm["stat"]),
        "p_value": float(dm["p_value"]),
        "se_dm": float(dm.get("se", float("nan"))),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "decision": "ADOPT" if adopt else "REJECT",
        "rule": "adopt if p < alpha and mean_delta < 0",
    }

    return {"summary": summary, "paired": df, "dm": dm}


def main():
    ap = argparse.ArgumentParser(description="DM+NW comparison for per-origin ES/CRPS files.")
    ap.add_argument("--base", required=True, type=Path, help="Path to baseline per-origin file (.csv or .parquet).")
    ap.add_argument("--cand", required=True, type=Path, help="Path to candidate per-origin file (.csv or .parquet).")
    ap.add_argument("--metric", choices=["es", "crps"], default=None, help="Metric to use; if omitted, inferred.")
    ap.add_argument("--anchor-hour", type=int, default=None, help="Keep only origins at this hour (0..23).")
    ap.add_argument("--origin-step", type=int, default=1, help="If no anchor-hour, keep every k-th origin (per series).")
    ap.add_argument("--nw-lag-rule", default="T**0.25", help='NW lag rule: "T**0.25", "T**(1/3)", or an integer.')
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional path to write paired diffs CSV.")
    ap.add_argument("--out-json", type=Path, default=None, help="Optional path to write summary JSON.")

    ap.add_argument("--exp_super", type=str, default=None,
                help="Type of experiment category, i.e. either E0, E1, E2 or E3")
    ap.add_argument("--note", type=str, default=None,
                help="Optional note for this comparison (e.g., 'CX base: RE vs BASE').")
    ap.add_argument("--append-summary-csv", type=Path, default=None,
                help="If set, append a one-row summary to this CSV (create if missing).")

    args = ap.parse_args()
    res = compare_two_files(
        base_path=args.base,
        cand_path=args.cand,
        metric=args.metric,
        anchor_hour=args.anchor_hour,
        origin_step=args.origin_step,
        nw_lag_rule=args.nw_lag_rule,
        alpha=args.alpha,
    )

    s = res["summary"]
    print(
        f"[DM+NW] metric={s['metric']}  N={s['T']}  lag={s['lag']}  alpha={s['alpha']}\n"
        f"         meanΔ={s['mean_delta']:.6f}  medianΔ={s['median_delta']:.6f}\n"
        f"         DM z={s['dm_stat']:.4f}  p={s['p_value']:.4g}\n"
        f"         {(1 - s['alpha']):.0%} CI: [{s['ci_lo']:.6f}, {s['ci_hi']:.6f}]"
    )

    s_row = {
        "exp_super": args.exp_super or "",
        "note": args.note or "",
        "metric": s["metric"],
        "N": s["T"],
        "lag": s["lag"],
        "alpha": s["alpha"],
        "mean_delta": s["mean_delta"],
        "median_delta": s["median_delta"],
        "dm_z": s["dm_stat"],
        "p_value": s["p_value"],
        "ci_lo": s.get("ci_lo"),
        "ci_hi": s.get("ci_hi"),
        "decision": s["decision"],
    }

    if args.append_summary_csv:
        import csv, os
        args.append_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not os.path.exists(args.append_summary_csv)
        with open(args.append_summary_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(s_row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(s_row)

    if args.out_csv:
        res["paired"].to_csv(args.out_csv, index=False)
    if args.out_json:
        import json
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()