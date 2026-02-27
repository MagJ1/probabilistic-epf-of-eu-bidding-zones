# --- helpers ---
from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
def inverse_scale(arr, scaler):
    """
    Undo scaling.
    arr: np.ndarray or torch.Tensor
    scaler: tuple (kind, a, b) or None
      - kind="z": standardization (mean, std)
      - kind="r": robust (median, scale)
      - kind="m": minmax (min, range)
      - else: passthrough
    """
    if scaler is None:
        return arr

    kind, a, b = scaler
    if kind in ("z", "r", "m"):
        return arr * b + a
    else:
        return arr

def inverse_per_h(lst, scaler):
    """lst: List[np.ndarray or None]"""
    out = []
    for x in lst:
        if x is None:
            out.append(None)
        else:
            out.append(inverse_scale(x, scaler))
    return out


from pathlib import Path
import numpy as np
import joblib

def count_qra_params(qra_models_path: Path, nz_tol: float = 0.0) -> dict:
    """
    Supports per-tau entry formats:

      A) old dict:
         {tau: {"beta": np.ndarray[F], "beta0": float, "lambda": float}}

      B) old tuple:
         {tau: (beta: np.ndarray[F], beta0: float, lambda: float)}

      C) new pair-calib dict (Lo+Gap parameterization):
         {tau: {"beta_lo": np.ndarray[F], "b_lo": float,
                "beta_d":  np.ndarray[F], "b_d":  float,
                "lambda": float}}

    qra_models.pkl:
      list over horizons, each item is a dict tau->entry (can be empty dict).

    nz_tol:
      weights with |w| <= nz_tol are treated as zero for 'effective' count.
      (biases always counted as active parameters.)
    """

    def _is_chung_pack(obj) -> bool:
        return isinstance(obj, dict) and ("state_dict" in obj) and isinstance(obj["state_dict"], dict)

    def _state_dict_param_count(sd: dict, nz_tol: float) -> tuple[int, int]:
        """
        Returns (dense_params, effective_params) from a PyTorch-like state_dict.
        Accepts torch tensors or numpy arrays.
        """
        dense = 0
        eff = 0
        for _, v in sd.items():
            if v is None:
                continue
            if hasattr(v, "detach"):  # torch tensor
                a = v.detach().cpu().numpy().ravel()
            else:
                a = np.asarray(v).ravel()
            dense += int(a.size)
            eff += int((np.abs(a) > nz_tol).sum())
        return dense, eff

    qra_models = joblib.load(qra_models_path)

    # Allow a wrapped dict: {"version": ..., "models": [...]} or plain list
    if isinstance(qra_models, dict) and "models" in qra_models:
        qra_models = qra_models["models"]

    assert isinstance(qra_models, list), "qra_models must be a list (one dict per horizon)."

    per_h = []
    total_models = total_params = total_effective = 0
    F_set = set()

    def _nnz(w: np.ndarray) -> int:
        w = np.asarray(w, dtype=float).ravel()
        return int((np.abs(w) > nz_tol).sum())

    for h, mdict in enumerate(qra_models):
        if not mdict:
            per_h.append({"h": h, "n_models": 0, "F": 0, "params": 0, "effective_params": 0})
            continue

        # --- NEW: Chung-pack format (one model per horizon) ---
        if _is_chung_pack(mdict):
            sd = mdict["state_dict"]
            params_h, eff_h = _state_dict_param_count(sd, nz_tol=nz_tol)

            # optional: try to infer F (feature dim) from in_dim if present
            in_dim = int(mdict.get("in_dim", 0))
            F_h = max(in_dim - 1, 0) if in_dim else 0

            per_h.append({
                "h": h,
                "n_models": 1,
                "F": int(F_h),
                "params": int(params_h),
                "effective_params": int(eff_h),
            })
            total_models += 1
            total_params += params_h
            total_effective += eff_h
            F_set.add(F_h)
            continue
        # --- init counters for standard per-tau dicts ---
        n_models_h = 0
        params_h = 0
        eff_h = 0
        F_h = None
        for tau, entry in mdict.items():
            # -------- normalize each supported format --------
            if isinstance(entry, dict):
                # New pair-calib format?
                if ("beta_lo" in entry) or ("beta_d" in entry) or ("b_lo" in entry) or ("b_d" in entry):
                    beta_lo = np.asarray(entry["beta_lo"], dtype=float).ravel()
                    beta_d  = np.asarray(entry["beta_d"],  dtype=float).ravel()
                    # biases exist but we just count them
                    _ = float(entry.get("b_lo", entry.get("beta0", 0.0)))  # tolerate older naming if any
                    _ = float(entry.get("b_d",  0.0))
                    # lambda not needed for counts

                    F = int(beta_lo.size)
                    if beta_d.size != F:
                        raise ValueError(f"H{h} τ={tau}: beta_lo.size={F} != beta_d.size={beta_d.size}")

                    F_h = F if F_h is None else max(F_h, F)

                    # Dense: 2*F weights + 2 biases
                    params_h += (2 * F + 2)
                    # Effective: nnz(beta_lo)+nnz(beta_d) + 2 biases
                    eff_h += (_nnz(beta_lo) + _nnz(beta_d) + 2)
                    n_models_h += 1
                    continue

                # Old dict format
                beta  = np.asarray(entry["beta"], dtype=float).ravel()
                _ = float(entry["beta0"])
                F = int(beta.size)
                F_h = F if F_h is None else max(F_h, F)

                params_h += (F + 1)
                eff_h += (_nnz(beta) + 1)
                n_models_h += 1
                continue

            # Old tuple/list format
            elif isinstance(entry, (list, tuple)) and len(entry) == 3:
                b, b0, l = entry
                beta = np.asarray(b, dtype=float).ravel()
                _ = float(b0)
                F = int(beta.size)
                F_h = F if F_h is None else max(F_h, F)

                params_h += (F + 1)
                eff_h += (_nnz(beta) + 1)
                n_models_h += 1
                continue

            else:
                raise TypeError(
                    f"H{h} τ={tau}: unexpected model entry type {type(entry)} "
                    f"(expected dict or 3-tuple). Keys={list(entry.keys()) if isinstance(entry, dict) else 'n/a'}"
                )

        F_set.add(F_h or 0)
        per_h.append({
            "h": h,
            "n_models": int(n_models_h),
            "F": int(F_h or 0),
            "params": int(params_h),
            "effective_params": int(eff_h),
        })
        total_models += n_models_h
        total_params += params_h
        total_effective += eff_h

    return {
        "total_models": int(total_models),
        "total_params": int(total_params),
        "total_effective_params": int(total_effective),
        "feature_dims_present": sorted(int(x) for x in F_set),
        "per_horizon": per_h,
        "nz_tolerance": float(nz_tol),
    }


def normalize_tau_dict_keys_to_percent(mdict: dict, *, tol: float = 1e-9) -> dict:
    """
    Ensure keys are percent-quantiles: 1, 10, 50, 99 (floats ok).
    If keys look like unit (<=1), convert by *100.
    """
    if not mdict:
        return mdict

    keys = [float(k) for k in mdict.keys()]
    kmax = max(keys)

    # already percent-style
    if kmax > 1.0 + tol:
        return mdict

    # looks like unit-style: convert -> percent
    out = {}
    for k, v in mdict.items():
        kp = round(float(k) * 100.0, 12)
        out[kp] = v
    return out