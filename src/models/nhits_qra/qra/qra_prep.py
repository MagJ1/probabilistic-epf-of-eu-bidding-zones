# src/models/nhits_qra/qra_prep.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA, PCA
try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception:
    DictConfig = type(None)
    ListConfig = type(None)
    OmegaConf = None

from models.nhits_qra.nhits.swag import swag_sampling
from models.nhits_qra.qra.qra_knobs import QRAKnobs

@torch.no_grad()
def _enable_mc_dropout(module: torch.nn.Module, on: bool):
    for m in module.modules():
        if isinstance(m, (torch.nn.modules.dropout._DropoutNd, torch.nn.AlphaDropout)):
            m.train(on)

def collect_qra_design_for_split(dm, module, split: str, knobs: QRAKnobs):
    if split == "train":
        dl = dm.train_dataloader()
    elif split == "val":
        dl = dm.val_dataloader()
    else:
        dl = dm.test_dataloader()
    return collect_qra_design(module, dl, knobs)   # no override


@torch.no_grad()
def collect_qra_design(
    module,
    dataloader,
    knobs: "QRAKnobs",
    use_device: Optional[torch.device] = None,
    *,
    n_samples_override: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    model = module.model
    device = use_device or next(model.parameters()).device

    # ---- knobs ----
    subsample_stride = int(getattr(knobs, "subsample_stride", 1))
    add_mean_std = bool(getattr(knobs, "add_mean_std", False))
    sample_dropout = bool(getattr(knobs, "sample_dropout", True))
    n_samples = int(n_samples_override if n_samples_override is not None else getattr(knobs, "n_samples"))

    # backend (works even if haven't added it to QRAKnobs yet)
    sampling_backend = getattr(knobs, "sampling_backend", "mc_dropout")  # "mc_dropout" | "swag"
    swag_scale = float(getattr(knobs, "scale", 1.0))
    swag_var_clamp = float(getattr(knobs, "var_clamp", 1e-30))

    X_per_h: List[List[np.ndarray]] = []
    y_per_h: List[List[np.ndarray]] = []
    H: Optional[int] = None
    C: Optional[int] = None
    C_ck: int = 0
    S_static: int = 0

    uid_chunks: List[np.ndarray] = []
    ds_chunks: List[np.ndarray] = []

    w_idx = 0
    for batch in dataloader:
        if subsample_stride > 1 and (w_idx % subsample_stride != 0):
            w_idx += 1
            continue
        w_idx += 1

        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        windows, y_future = module._build_windows(batch)  # y_future: (B,H,1)
        B = int(y_future.shape[0])
        if B == 0:
            continue

        # infer H and C from windows the first time
        if H is None:
            H = int(y_future.shape[1])
            C = int(windows["insample_y"].shape[1])  # context length

            # futr_exog is (B, C+H, C_ck) in module
            futr_all = windows["futr_exog"]
            C_ck = int(futr_all.shape[-1]) if futr_all.ndim == 3 else 0

            stat = windows["stat_exog"]
            S_static = int(stat.shape[-1]) if stat.ndim == 2 else 0

            X_per_h = [[] for _ in range(H)]
            y_per_h = [[] for _ in range(H)]

        assert H is not None and C is not None

        # ----- slice CK future correctly: (B, H, C_ck) -----
        futr_all = windows["futr_exog"]  # (B, C+H, C_ck)
        if C_ck > 0:
            if futr_all.shape[1] != (C + H):
                raise RuntimeError(f"Expected futr_exog time length C+H={C+H}, got {futr_all.shape[1]}")
            ck_fut = futr_all[:, C:, :]  # (B, H, C_ck)
        else:
            ck_fut = None

        stat = windows["stat_exog"]  # (B, S_static)

        # ---- generate n_samples point forecasts: samples (B, S, H) ----
        preds: List[torch.Tensor] = []
        model.eval()

        if sampling_backend == "swag":
            swag = getattr(module, "swag", None)
            if swag is None:
                raise RuntimeError("sampling_backend='swag' but module.swag is missing?")
            if getattr(swag, "n_snapshots", 0) < 2:
                raise RuntimeError(f"SWAG needs >=2 snapshots, but have n_snapshots={getattr(swag, 'n_snapshots', 0)}")

            # (keep dropout OFF here; SWAG already provides diversity)
            with swag_sampling(model, swag, scale=swag_scale, var_clamp=swag_var_clamp) as sess:
                for _ in range(n_samples):
                    sess.sample_and_assign_()
                    y_hat = model(windows)              # (B,H,1)
                    preds.append(y_hat.squeeze(-1))     # (B,H)

        else:
            # MC Dropout path
            if sample_dropout:
                _enable_mc_dropout(model, True)
            try:
                for _ in range(n_samples):
                    y_hat = model(windows)              # (B,H,1)
                    preds.append(y_hat.squeeze(-1))     # (B,H)
            finally:
                if sample_dropout:
                    _enable_mc_dropout(model, False)

        samples = torch.stack(preds, dim=1)  # (B, S, H)

        # ---- move to numpy ----
        samples_np = samples.detach().cpu().numpy()              # (B,S,H)
        y_np = y_future.squeeze(-1).detach().cpu().numpy()       # (B,H)

        ck_np = ck_fut.detach().cpu().numpy() if (C_ck > 0 and ck_fut is not None) else None
        st_np = stat.detach().cpu().numpy() if S_static > 0 else None

        # ---- unique_id -> (B,) ----
        uid_b = batch.get("unique_id", None)
        if uid_b is None:
            raise KeyError("collect_qra_design expects batch['unique_id'].")
        if isinstance(uid_b, torch.Tensor):
            uid_b = uid_b.detach().cpu().numpy()
        else:
            uid_b = np.asarray(uid_b)
        uid_b = uid_b.reshape(-1)
        if uid_b.shape[0] != B:
            raise RuntimeError(f"unique_id len {uid_b.shape[0]} != batch size {B}")
        uid_chunks.append(uid_b)

        # ---- origin timestamp from batch['ds'] ----
        ds_obj = batch.get("ds", None)
        if ds_obj is None:
            raise KeyError("collect_qra_design expects batch['ds'].")

        # (yur existing ds parsing stays as-is)
        origins: np.ndarray
        if isinstance(ds_obj, (list, tuple)):
            first = ds_obj[0]
            if len(ds_obj) == B and isinstance(first, (list, tuple)):
                L0 = len(first)
                if any(len(r) != L0 for r in ds_obj):
                    raise RuntimeError("Inconsistent ds row lengths in row-major form.")
                C_len = L0 - H
                if C_len < 0:
                    raise RuntimeError(f"Computed C={C_len}<0 (L0={L0}, H={H}).")
                origins = np.array([ds_obj[i][C_len] for i in range(B)])
            elif isinstance(first, (list, tuple)) and len(first) == B:
                L0 = len(ds_obj)
                C_len = L0 - H
                if C_len < 0:
                    raise RuntimeError(f"Computed C={C_len}<0 (L0={L0}, H={H}).")
                origins = np.array([ds_obj[C_len][i] for i in range(B)])
            elif not isinstance(first, (list, tuple)) and len(ds_obj) == B:
                origins = np.asarray(ds_obj)
            else:
                raise RuntimeError(
                    f"collect_qra_design: unsupported ds layout. "
                    f"type(list) len={len(ds_obj)}, type(first)={type(first)}"
                )
        else:
            ds_arr = np.asarray(ds_obj)
            if ds_arr.ndim == 2:
                if ds_arr.shape[0] == B:
                    L0 = ds_arr.shape[1]
                    C_len = L0 - H
                    origins = ds_arr[:, C_len]
                elif ds_arr.shape[1] == B:
                    L0 = ds_arr.shape[0]
                    C_len = L0 - H
                    origins = ds_arr[C_len, :]
                else:
                    raise RuntimeError(f"Unexpected ds array shape {ds_arr.shape} (B={B}, H={H}).")
            elif ds_arr.ndim == 1 and ds_arr.shape[0] == B:
                origins = ds_arr
            else:
                raise RuntimeError(f"Unsupported ds array shape {ds_arr.shape}.")

        ods_b = pd.to_datetime(origins, errors="coerce")
        ods_b = pd.DatetimeIndex(ods_b)
        if getattr(ods_b, "tz", None) is not None:
            ods_b = ods_b.tz_localize(None)
        ods_b = ods_b.floor("h")
        if ods_b.isna().any():
            raise ValueError(f"collect_qra_design: {int(ods_b.isna().sum())} invalid timestamps in batch['ds'].")
        ds_chunks.append(ods_b.to_numpy())

        # ---- assemble per-horizon design ----
        for h in range(H):
            Y_h = y_np[:, h]            # (B,)
            S_h = samples_np[:, :, h]   # (B,S)

            parts = [S_h]
            if C_ck and ck_np is not None:
                parts.append(ck_np[:, h, :])      # (B,C_ck)
            if S_static and st_np is not None:
                parts.append(st_np)               # (B,S_static)
            if add_mean_std:
                mu = S_h.mean(axis=1, keepdims=True)
                sd = S_h.std(axis=1, keepdims=True)
                parts.extend([mu, sd])

            X_h = np.concatenate(parts, axis=1)
            X_per_h[h].append(X_h)
            y_per_h[h].append(Y_h)

    # stack
    X_per_h_out = [np.concatenate(ch, axis=0) if ch else np.zeros((0, 0)) for ch in X_per_h]
    y_per_h_out = [np.concatenate(ch, axis=0) if ch else np.zeros((0,)) for ch in y_per_h]

    uid_all = np.concatenate(uid_chunks, axis=0) if uid_chunks else np.asarray([], dtype=np.int32)
    ds_all = np.concatenate(ds_chunks, axis=0) if ds_chunks else np.asarray([], dtype="datetime64[ns]")

    # sanity
    N_index = len(uid_all)
    for h, X_h in enumerate(X_per_h_out):
        if X_h.shape[0] != N_index:
            raise RuntimeError(f"Row count mismatch at h={h}: X={X_h.shape[0]} vs index={N_index}")

    meta = dict(
        H=H or 0,
        C_ck=C_ck or 0,
        S_static=S_static or 0,
        S=n_samples,
        unique_id=uid_all,
        origin_ds=ds_all,
    )
    return X_per_h_out, y_per_h_out, meta


# ---------------------------
# PCA helpers (step 3)
# ---------------------------

def _choose_n_components_by_probe(X: np.ndarray, pca_var: float, max_probe_rows: int = 10000) -> int:
    """Use a small in-memory PCA on a random subset to choose n_components for IncrementalPCA."""
    n_rows = X.shape[0]
    if n_rows == 0:
        return 0
    take = min(n_rows, max_probe_rows)
    idx = np.random.choice(n_rows, take, replace=False)
    probe = X[idx]
    k_max = min(probe.shape[0], probe.shape[1])
    p = PCA(n_components=k_max, svd_solver="randomized", random_state=0).fit(probe)
    cum = np.cumsum(p.explained_variance_ratio_)
    k = int(np.searchsorted(cum, pca_var) + 1)
    return max(1, min(k, k_max))


def fit_incremental_pca_per_h(
    X_per_h: List[np.ndarray],
    n_samples: int,
    pca_var: float = 0.95,
    n_components: Optional[int] = None,
    chunk_size: int = 4096,
) -> Tuple[List[Optional[IncrementalPCA]], List[np.ndarray], List[int]]:
    """
    Returns:
      pcas: list of IncrementalPCA per horizon (or None)
      Z_per_h: transformed design matrices
      k_per_h: retained PCA dims per horizon (0 if no PCA)
    """
    pcas: List[Optional[IncrementalPCA]] = []
    Z_per_h: List[np.ndarray] = []
    k_per_h: List[int] = []

    for X in X_per_h:
        if X.size == 0 or X.shape[1] == 0:
            pcas.append(None)
            Z_per_h.append(X)
            k_per_h.append(0)
            continue

        F_total = X.shape[1]
        n_samples_eff = max(0, min(int(n_samples), F_total))
        X_samples = X[:, :n_samples_eff]
        X_det = X[:, n_samples_eff:]

        if n_samples_eff == 0:
            pcas.append(None)
            Z_per_h.append(X_det)
            k_per_h.append(0)
            continue

        if n_components is None:
            k = _choose_n_components_by_probe(X_samples, pca_var)
        else:
            k = int(n_components)
        k = max(1, min(k, n_samples_eff))

        ipca = IncrementalPCA(n_components=k)
        for start in range(0, X_samples.shape[0], chunk_size):
            ipca.partial_fit(X_samples[start:start + chunk_size])

        Z_chunks = []
        for start in range(0, X_samples.shape[0], chunk_size):
            Z_chunks.append(ipca.transform(X_samples[start:start + chunk_size]))
        Z_samples = np.concatenate(Z_chunks, axis=0)

        Z_total = np.concatenate([Z_samples, X_det], axis=1) if X_det.size > 0 else Z_samples

        pcas.append(ipca)
        Z_per_h.append(Z_total)
        k_per_h.append(int(getattr(ipca, "n_components_", k)))  # robust

    return pcas, Z_per_h, k_per_h


def transform_with_pcas(
    X_per_h: List[np.ndarray],
    pcas: List[Optional[IncrementalPCA]],
    n_samples: int,                 # number of NHITS MC sample columns at the front
    chunk_size: int = 4096,
) -> List[np.ndarray]:
    """
    Apply fitted per-horizon IncrementalPCAs to a new set of design matrices.
    Only the first `n_samples` columns (the MC sample block) are transformed;
    the remaining deterministic columns are passed through unchanged.

    If the corresponding fitted PCA is None (i.e., no sample cols during fit),
    this returns only the deterministic block, consistent with fitting.
    """
    out: List[np.ndarray] = []
    for X, ipca in zip(X_per_h, pcas):
        if X.size == 0:
            out.append(X)
            continue

        F_total = X.shape[1]
        n_samples_eff = max(0, min(n_samples, F_total))
        X_samples = X[:, :n_samples_eff]
        X_deterministic = X[:, n_samples_eff:]  # may be (N, 0)

        # No PCA was fitted for this horizon (no sample cols during fit)
        if ipca is None or n_samples_eff == 0:
            out.append(X_deterministic)  # mirror fit behavior
            continue

        # Transform the samples block in chunks
        Z_chunks = []
        for start in range(0, X_samples.shape[0], chunk_size):
            Z_chunks.append(ipca.transform(X_samples[start:start + chunk_size]))
        Z_samples = np.concatenate(Z_chunks, axis=0)  # (N, k)

        # Concatenate back deterministic block (if any)
        if X_deterministic.size > 0:
            Z_total = np.concatenate([Z_samples, X_deterministic], axis=1)
        else:
            Z_total = Z_samples

        out.append(Z_total)

    return out

def _to_plain(obj: Any):
    if OmegaConf is not None and isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj

def _ensure_tau_array(taus):
    t = np.asarray(taus, dtype=float)
    if t.size == 0:
        return t
    # accept percent or 0..1
    if t.max() > 1.0:
        t = t / 100.0
    # clip off exact 0/1 to avoid infinities later
    eps = 1e-6
    t = np.clip(t, eps, 1.0 - eps)
    # unique & sorted
    t = np.unique(np.sort(t))
    return t

def make_quantile_grid(spec: Union[Dict, List, int]) -> List[float]:
    """
    Accepts:
      - {'uniform': K}                           -> K evenly spaced in (0,1)
      - {'custom': [q1, q2, ...]}                -> use provided (percent or 0..1)
      - {'piecewise': [{'lo':a,'hi':b,'n':m}, ...]}   -> union of segments
    Returns a sorted, unique list in (0,1).
    """
    spec = _to_plain(spec)

    # case 1: plain list -> custom
    if isinstance(spec, list):
        return _ensure_tau_array(spec).tolist()

    # case 2: plain int -> uniform
    if isinstance(spec, int):
        K = int(spec)
        if K < 2:
            raise ValueError("uniform count must be >=2")
        t = np.linspace(0.0, 1.0, K + 2)[1:-1]  # exclude 0 and 1
        return _ensure_tau_array(t).tolist()

    # case 3: dict with a single mode
    if isinstance(spec, dict):
        # UNIFORM
        if "uniform" in spec:
            K = int(spec["uniform"])
            if K < 2:
                raise ValueError("uniform count must be >=2")
            t = np.linspace(0.0, 1.0, K + 2)[1:-1]
            return _ensure_tau_array(t).tolist()

        # CUSTOM
        if "custom" in spec:
            return _ensure_tau_array(spec["custom"]).tolist()

        # PIECEWISE
        if "piecewise" in spec:
            segs = _to_plain(spec["piecewise"])
            if not isinstance(segs, list) or len(segs) == 0:
                raise ValueError("piecewise must be a non-empty list of segments")
            buckets = []
            for seg in segs:
                lo, hi, n = float(seg["lo"]), float(seg["hi"]), int(seg["n"])
                # accept percent or 0..1
                if max(lo, hi) > 1.0:
                    lo, hi = lo/100.0, hi/100.0
                if not (0.0 <= lo < hi <= 1.0):
                    raise ValueError(f"bad segment bounds: {seg}")
                if n < 2:
                    # single point segment → put midpoint
                    buckets.append(np.array([(lo+hi)/2.0], dtype=float))
                    continue
                # create n points strictly inside [lo,hi]; avoid exact 0/1
                t = np.linspace(lo, hi, n)
                buckets.append(t)
            t_all = np.unique(np.concatenate(buckets))
            return _ensure_tau_array(t_all).tolist()

    raise ValueError(f"Unsupported quantile spec: {spec}")