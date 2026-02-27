# src/utils/plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path
import matplotlib.dates as mdates

def _make_inverse_from_sklearn_scaler(sc):
    # StandardScaler or MinMaxScaler; else identity
    if hasattr(sc, "mean_") and hasattr(sc, "scale_"):
        mean  = np.asarray(sc.mean_).reshape(1, -1)
        scale = np.asarray(sc.scale_).reshape(1, -1)
        return lambda x: (x * scale[..., :x.shape[-1]]) + mean[..., :x.shape[-1]]
    if hasattr(sc, "data_min_") and hasattr(sc, "data_max_"):
        data_min = np.asarray(sc.data_min_).reshape(1, -1)
        data_max = np.asarray(sc.data_max_).reshape(1, -1)
        rng = np.maximum(data_max - data_min, 1e-12)
        return lambda x: (x * rng[..., :x.shape[-1]]) + data_min[..., :x.shape[-1]]
    return lambda x: x

def _reconstruct_time_axes(origin_str: str, context_length: int, horizon: int):
    origin = pd.Timestamp(origin_str)
    past_ds = pd.date_range(end=origin - pd.Timedelta(hours=1),
                            periods=context_length, freq="h").to_pydatetime()
    fut_ds  = np.array([origin + pd.Timedelta(hours=i) for i in range(horizon)], dtype="datetime64[ns]")
    return origin, past_ds, fut_ds


def _plot_fanchart_lines(
    *,
    past_ds,
    past_y,
    origin,
    fut_ds,
    samples_np,           # (S, H) numpy
    y_true_future_np,     # (H,) numpy or None
    quantile_bands,       # list[(low,high)]
    show_mean=True,
    show_median=True,
    history=168,
    title=None,
    model_name=None,
    model_size=None,
    out_path: Path,
):
    # history slice
    if len(past_y) > history:
        past_ds = past_ds[-history:]
        past_y  = past_y[-history:]

    # quantiles
    qs = sorted({q for a,b in quantile_bands for q in (a,b)})
    qvals = np.quantile(samples_np, qs, axis=0)  # (len(qs), H)
    qmap = {q: qvals[i] for i, q in enumerate(qs)}
    median = np.quantile(samples_np, 0.5, axis=0)
    mean   = samples_np.mean(axis=0)

    # draw
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(past_y) > 0:
        ax.plot(past_ds, past_y, label="history", linewidth=1.5, zorder=3)

    # shade widest first
    base_alpha, step_alpha = 0.20, 0.12 if len(quantile_bands) > 1 else 0.0
    for i, (ql, qh) in enumerate(quantile_bands):
        lower, upper = qmap[ql], qmap[qh]
        ax.fill_between(fut_ds, 
                        lower, upper, 
                        alpha=base_alpha + i*step_alpha, 
                        linewidth=0,
                        label=f"{round((qh-ql)*100)}% band",
                        color="C0")

    if show_median:
        ax.plot(fut_ds, median, label="median forecast", linewidth=1.5, zorder=4, color="red")
    if show_mean:
        ax.plot(fut_ds, mean,   label="mean forecast",   linewidth=1.5, linestyle="--", zorder=4, color="purple")

    if y_true_future_np is not None:
        ax.plot(fut_ds, y_true_future_np, label="actual", linewidth=1.5, zorder=5, color="green")

    ax.axvline(pd.Timestamp(origin), linestyle=":", linewidth=1.0, color="black")
    left = pd.Timestamp(origin) - pd.Timedelta(hours=max(history - 1, 1))
    right = pd.Timestamp(fut_ds[-1])
    ax.set_xlim(left, right)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # --- automatic title handling ---
    if title and len(str(title).strip()) > 0:
        ax.set_title(title)
    else:
        ts_str = pd.Timestamp(origin).strftime("%Y-%m-%d, %H:%M")
        name_str = f"{model_name.capitalize()} ({model_size}) — " if model_name else ""
        ax.set_title(f"{name_str}Forecast {ts_str}")

    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=3, color="black", width=0.6)
    ax.tick_params(axis="x", which="major", length=6, width=0.8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(ncol=4, frameon=False, loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def _get_test_dataset(dm):
    for attr in ("test_ds", "test_dataset", "test_set"):
        if hasattr(dm, attr):
            return getattr(dm, attr)
    raise AttributeError("DataModule has no test dataset attribute (expected one of: test_ds/test_dataset/test_set).")


def _plot_fanchart_steps(
    *,
    past_ds,
    past_y,
    origin,
    fut_ds,                 # ignored for x construction; we rebuild edges
    samples_np,             # (S, H)
    y_true_future_np,       # (H,) or None
    quantile_bands,
    show_mean=True,
    show_median=True,
    history=168,
    title=None,
    model_name=None,
    model_size=None,
    out_path: Path,
):
    # Slice history
    if len(past_y) > history:
        past_ds = pd.DatetimeIndex(past_ds[-history:])
        past_y  = past_y[-history:]
    else:
        past_ds = pd.DatetimeIndex(past_ds)

    origin = pd.Timestamp(origin).tz_localize(None)

    # Quantiles
    qs = sorted({q for a, b in quantile_bands for q in (a, b)})
    qvals = np.quantile(samples_np, qs, axis=0)  # (len(qs), H)
    qmap = {q: qvals[i] for i, q in enumerate(qs)}
    median = np.quantile(samples_np, 0.5, axis=0)
    mean   = samples_np.mean(axis=0)
    H = median.shape[0]

    # Step helpers: edges (H+1) and right-closed values
    x_hist_edges = pd.date_range(end=origin, periods=len(past_y) + 1, freq="h")
    y_hist_step  = np.append(past_y, past_y[-1])            # length L+1

    x_fut_edges  = pd.date_range(start=origin, periods=H + 1, freq="h")
    def _stepify(yH):  # (H,) -> (H+1,)
        return np.append(yH, yH[-1])

    # Draw
    fig, ax = plt.subplots(figsize=(11, 5))

    # History as step (constant over [t, t+1))
    if len(y_hist_step) > 0:
        ax.step(x_hist_edges, y_hist_step, where="post",
                label="history", linewidth=1.3, zorder=3)

    # Bands as stepped fills
    base_alpha = 0.25
    step_alpha = 0.12 if len(quantile_bands) > 1 else 0.0
    for i, (ql, qh) in enumerate(quantile_bands):
        lower, upper = _stepify(qmap[ql]), _stepify(qmap[qh])
        ax.fill_between(
            x_fut_edges, lower, upper,
            step="post",
            color="C0",
            alpha=base_alpha + i * step_alpha,
            linewidth=0,
            label=f"{(qh - ql) * 100:.0f}% band",
            zorder=4
        )

    # Median / Mean as steps
    if show_median:
        ax.step(x_fut_edges, _stepify(median), where="post",
                label="median forecast", color="red", linewidth=1.6, zorder=6)

    if show_mean:
        ax.step(x_fut_edges, _stepify(mean), where="post",
                label="mean forecast", color="purple", linestyle="--", linewidth=1.3, zorder=6)

    # Truth as steps
    if y_true_future_np is not None:
        ax.step(x_fut_edges, _stepify(y_true_future_np), where="post",
                label="actual", color="green", linewidth=1.6, zorder=7)

    # Cosmetics
    ax.axvline(origin, linestyle=":", linewidth=1.0, zorder=8, color="black")
    left  = origin - pd.Timedelta(hours=max(history, 1))
    right = x_fut_edges[-1]
    ax.set_xlim(left, right)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # Auto title
    if title and len(str(title).strip()) > 0:
        ax.set_title(title)
    else:
        ts_str = origin.strftime("%Y-%m-%d, %H:%M")
        name_str = f"{model_name} ({model_size}) — " if model_name else ""
        ax.set_title(f"{name_str}Forecast {ts_str}")

    # Minor hourly ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=3, color="black", width=0.6)
    ax.tick_params(axis="x", which="major", length=6, width=0.8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(ncol=4, frameon=False, loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --- new: plot fan chart directly from quantiles (no samples needed) ---
def _plot_fanchart_steps_from_quantiles(
    *,
    past_ds,                 # DatetimeIndex or array-like (L,)
    past_y,                  # (L,)
    origin,                  # pandas.Timestamp or str
    qmap_h: dict,            # {tau: (H_eff,)} arrays in real units; must include 0.5 for median
    y_true_future_np=None,   # (H_eff,) in real units (optional)
    quantile_bands=((0.05, 0.95), (0.25, 0.75)),
    history=168,
    title=None,
    model_name=None,
    model_size=None,
    out_path: Path = None,
):
    import matplotlib.dates as mdates
    origin = pd.Timestamp(origin).tz_localize(None)

    # slice history
    if len(past_y) > history:
        past_ds = pd.DatetimeIndex(past_ds[-history:])
        past_y  = past_y[-history:]
    else:
        past_ds = pd.DatetimeIndex(past_ds)

    # horizon length from any tau
    if not qmap_h:
        raise ValueError("qmap_h is empty; need at least median at tau=0.5.")
    if 0.5 not in qmap_h:
        raise ValueError("qmap_h must contain median at tau=0.5.")
    H = int(np.asarray(qmap_h[0.5]).shape[0])

    # step edges
    x_hist_edges = pd.date_range(end=origin, periods=len(past_y) + 1, freq="h")
    x_fut_edges  = pd.date_range(start=origin, periods=H + 1, freq="h")

    def _stepify(y):
        y = np.asarray(y, float)
        return np.append(y, y[-1])  # (H+1,)

    # draw
    fig, ax = plt.subplots(figsize=(11, 5))

    # history
    if len(past_y) > 0:
        ax.step(x_hist_edges, np.append(past_y, past_y[-1]), where="post",
                label="history", linewidth=1.3, zorder=3)

    # bands (widest first)
    base_alpha = 0.25
    step_alpha = 0.12 if len(quantile_bands) > 1 else 0.0
    for i, (ql, qh) in enumerate(quantile_bands):
        lower = _stepify(qmap_h[float(ql)])
        upper = _stepify(qmap_h[float(qh)])
        ax.fill_between(
            x_fut_edges, lower, upper,
            step="post", color="C0",
            alpha=base_alpha + i * step_alpha,
            linewidth=0, label=f"{(qh-ql)*100:.0f}% band", zorder=4
        )

    # median
    ax.step(x_fut_edges, _stepify(qmap_h[0.5]), where="post",
            label="median forecast", color="red", linewidth=1.6, zorder=6)

    # truth
    if y_true_future_np is not None:
        ax.step(x_fut_edges, _stepify(y_true_future_np), where="post",
                label="actual", color="green", linewidth=1.6, zorder=7)

    # cosmetics
    ax.axvline(origin, linestyle=":", linewidth=1.0, zorder=8, color="black")
    left  = origin - pd.Timedelta(hours=max(history, 1))
    right = x_fut_edges[-1]
    ax.set_xlim(left, right)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # auto title
    if title and len(str(title).strip()) > 0:
        ax.set_title(title)
    else:
        ts_str = origin.strftime("%Y-%m-%d, %H:%M")
        name_str = f"{model_name.capitalize()} ({model_size}) — " if model_name else ""
        ax.set_title(f"{name_str}Forecast {ts_str}")

    # minor hourly ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=3, width=0.6)
    ax.tick_params(axis="x", which="major", length=6, width=0.8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(ncol=4, frameon=False, loc="upper left")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_pit_histogram(U: np.ndarray, bins: int, title: str, out_path: Path):
    """
    Plot PIT histogram as density on [0,1] with reference line at y=1.
    U: 1D array of PIT values in [0,1]
    """
    if U is None or len(U) == 0:
        return
    U = np.asarray(U, dtype=float)
    U = U[np.isfinite(U)]
    if U.size == 0:
        return

    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, edges = np.histogram(U, bins=edges)
    n = counts.sum()
    if n == 0:
        return

    width = edges[1] - edges[0]  # = 1/bins
    density = counts / n / width  # so ideal calibrated = 1

    centers = (edges[:-1] + edges[1:]) * 0.5
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.bar(centers, density, width=width, edgecolor="none")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1.05 * density.max(), 1.5))
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)