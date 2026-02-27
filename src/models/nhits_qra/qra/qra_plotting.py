# src/models/nhits_qra/qra/qra_plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List
import numpy as np
import pandas as pd
import torch

from utils.plotting import (
    _get_test_dataset,
    _reconstruct_time_axes,
    _plot_fanchart_steps,
    _plot_fanchart_lines,
    _plot_pit_histogram,
)

def get_art_dir(dirs: Dict[str, Path], paths: Dict[str, Path], out_qra: Path) -> Path:
    for key in ("art", "artifacts", "fig", "figures", "plots"):
        p = (dirs.get(key) if dirs else None) or (paths.get(key) if paths else None)
        if p is not None:
            p = Path(p); p.mkdir(parents=True, exist_ok=True); return p
    p = out_qra / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_fan_charts(
    *,
    log,
    dm,
    test_cfg,
    dirs: Dict[str, Path],
    paths: Dict[str, Path],
    out_qra: Path,
    uid: np.ndarray,
    ods: pd.DatetimeIndex,
    Q_fix_real: List[Optional[np.ndarray]],              # per-h quantiles in real scale
    taus_tgt: Sequence[float],
    samples_per_h_for_plot: List[torch.Tensor],          # from samples_from_quantiles
):
    # Pull config
    fan_cfg = getattr(getattr(getattr(test_cfg, "test", None), "plotting", None), "fan_chart", None)
    if not (fan_cfg and bool(fan_cfg.enable_fan_plotting)):
        return

    origin_list    = list(fan_cfg.origin_ds)
    quantile_bands = [tuple(map(float, band)) for band in fan_cfg.quantile_ranges]
    history_hours  = int(fan_cfg.history_hours)
    model_name     = str(test_cfg.test.plotting.model_name)
    model_size     = str(test_cfg.test.plotting.size)
    line_type      = str(fan_cfg.type).lower()

    fan_dir = get_art_dir(dirs, paths, out_qra)

    ds_test = _get_test_dataset(dm)
    inv_target = ds_test.inverse_target

    # Map from (uid, origin) -> row index in filtered arrays
    kept_index = {(int(u), pd.Timestamp(o).value): i for i, (u, o) in enumerate(zip(uid, ods))}

    for origin_str in origin_list:
        n_plotted = 0
        target_origin = pd.to_datetime(origin_str).floor("h")

        for idx in range(len(ds_test)):
            sample = ds_test[idx]
            ds_list = sample["ds"]
            C = int(sample["y_past"].shape[0])
            sample_origin = pd.Timestamp(ds_list[C]).floor("h")
            if sample_origin != target_origin:
                continue

            uid_int = int(sample["unique_id"])
            key = (uid_int, int(target_origin.value))
            if key not in kept_index:
                continue
            k = kept_index[key]

            y_past_real = inv_target(sample["y_past"].detach().cpu().numpy())
            y_true_real = inv_target(sample["y_future"].detach().cpu().numpy())

            H = int(y_true_real.shape[0])
            CL = int(y_past_real.shape[0])
            origin_dt, past_ds, fut_ds = _reconstruct_time_axes(str(target_origin), context_length=CL, horizon=H)

            # build samples matrix (S, H_eff)
            samp_cols = []
            hs_eff_plot = [h for h in range(len(Q_fix_real)) if Q_fix_real[h] is not None]
            for h in hs_eff_plot:
                X_h = samples_per_h_for_plot[h]
                if X_h.dim() == 3:
                    xk = X_h[:, k, 0].detach().cpu().numpy()
                else:
                    xk = X_h[:, k].detach().cpu().numpy()
                samp_cols.append(xk)

            if not samp_cols:
                continue

            samples_matrix = np.stack(samp_cols, axis=1)
            y_true_plot = y_true_real[: samples_matrix.shape[1]]

            safe_model = model_name.replace(" ", "").lower().strip()
            safe_size  = model_size.replace(" ", "").lower().strip()
            bands_txt  = "_".join([str(int(round((hi - lo) * 100))) for (lo, hi) in quantile_bands])
            fname = (
                f"fan_{safe_model}_{safe_size}_"
                f"uid{uid_int}_"
                f"{pd.Timestamp(origin_dt).strftime('%Y-%m-%dT%H-%M')}_"
                f"S{samples_matrix.shape[0]}_bands{bands_txt}_{line_type}.png"
            )
            out_path = fan_dir / fname

            if line_type == "line":
                _plot_fanchart_lines(
                    past_ds=past_ds, past_y=y_past_real, origin=origin_dt,
                    fut_ds=fut_ds, samples_np=samples_matrix, y_true_future_np=y_true_plot,
                    quantile_bands=quantile_bands, show_mean=False, show_median=True,
                    history=history_hours, title=None, model_name=model_name, model_size=model_size,
                    out_path=out_path,
                )
            else:
                _plot_fanchart_steps(
                    past_ds=past_ds, past_y=y_past_real, origin=origin_dt,
                    fut_ds=None, samples_np=samples_matrix, y_true_future_np=y_true_plot,
                    quantile_bands=quantile_bands, show_mean=False, show_median=True,
                    history=history_hours, title=None, model_name=model_name, model_size=model_size,
                    out_path=out_path,
                )

            n_plotted += 1

        log.info("QRA[test] fan-plotting: origin=%s  plots_written=%d  → %s",
                 str(target_origin), n_plotted, str(fan_dir))


def plot_pit_histograms(
    *,
    log,
    test_cfg,
    dirs: Dict[str, Path],
    paths: Dict[str, Path],
    out_qra: Path,
    hs_eff: Sequence[int],
    U_per_h: List[Optional[np.ndarray]],
):
    pit_cfg = test_cfg.test.plotting.pit
    if not bool(pit_cfg.enable_pit_plotting):
        return

    if isinstance(pit_cfg.horizons, str) and pit_cfg.horizons.lower() == "all":
        hs_plot = list(hs_eff)
    else:
        hs_plot = [int(h) for h in list(pit_cfg.horizons)]
        invalid = [h for h in hs_plot if h not in set(hs_eff)]
        if invalid:
            raise IndexError(f"PIT horizons not available (no data at these h): {invalid}")

    bins = int(pit_cfg.bins)
    model_name = str(test_cfg.test.plotting.model_name)
    model_size = str(test_cfg.test.plotting.size)

    pit_dir = get_art_dir(dirs, paths, out_qra)

    for h in hs_plot:
        U_h = U_per_h[h]
        if U_h is None or (isinstance(U_h, np.ndarray) and U_h.size == 0):
            log.info("PIT[h=%d]: skipped (no data).", h)
            continue

        safe_model = model_name.replace(" ", "").lower().strip()
        safe_size  = model_size.replace(" ", "").lower().strip()
        fname = f"pit_{safe_model}_{safe_size}_h{h+1}_bins{bins}.png"
        out_path = pit_dir / fname

        title = f"{model_name} ({model_size}) — PIT histogram — h={h+1}"
        _plot_pit_histogram(U_h, bins=bins, title=title, out_path=out_path)

    log.info("QRA[test]: PIT histograms written → %s", str(pit_dir))