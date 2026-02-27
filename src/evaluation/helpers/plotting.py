from __future__ import annotations
import numpy as np
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
from matplotlib.transforms import blended_transform_factory
from pandas.api.types import is_scalar

# --- plotting defaults (force light mode, seaborn-like) ---
import matplotlib as mpl
mpl.rcdefaults()

# ---- knobs you can tweak ----
FEATURES = ["GP","CBT","NRE","RE","LOAD","CO2","SP"]
MODEL_ORDER = ["NF","NQ","MO","CX"]
SIZE_RANK = {"tiny":0, "small":1, "base":2, "large":3}  # for ordering rows in each model
STRAT_ORDER = {"zero_shot": 0, "one_shot": 1, "few_shot": 2}

NOTE_PAT = re.compile(
    r"^\s*(?P<strategy>zero_shot|one_shot|few_shot)\s+"
    r"(?P<model>NF|NQ|MO|CX)\s+"
    r"(?P<size>\w+)\s+vs\.\s+"
    r"(?P<rhs>baseline_model|"
    r"(zero_shot|one_shot|few_shot)\s+(?P<rhs_model>NF|NQ|MO|CX)\s+(?P<rhs_size>\w+))\s*$"
)

def _parse_note_e3(note: str) -> dict:
    m = NOTE_PAT.match(str(note))
    if not m:
        return {"strategy": None, "model": None, "size": None, "cmp_type": "unknown",
                "rhs_strategy": None, "rhs_model": None, "rhs_size": None}
    g = m.groupdict()
    if g["rhs"] == "baseline_model":
        cmp_type = "baseline"
        rhs_strategy = rhs_model = rhs_size = None
    else:
        cmp_type = "strategy"
        rhs_strategy = g["rhs"].split()[0]
        rhs_model   = g.get("rhs_model")
        rhs_size    = g.get("rhs_size")
    return {
        "strategy": g["strategy"],
        "model": g["model"],
        "size": g["size"],
        "cmp_type": cmp_type,
        "rhs_strategy": rhs_strategy,
        "rhs_model": rhs_model,
        "rhs_size": rhs_size,
    }

def _adopt(mean_delta, p):
    # be robust to dtype issues
    md = pd.to_numeric(mean_delta, errors="coerce")
    pv = pd.to_numeric(p,          errors="coerce")

    # scalar path (e.g., single row or unit tests)
    if is_scalar(md) and is_scalar(pv):
        return (pv < 0.05) and (md < 0.0)

    # vectorized path
    out = (pv < 0.05) & (md < 0.0)
    return out.fillna(False)

def prepare_frames_e3(df_in: pd.DataFrame):
    df = df_in.copy()
    parsed = df["note"].apply(_parse_note_e3).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Baseline comparisons
    base = df[df["cmp_type"].eq("baseline")].copy()
    base["adopt"] = _adopt(base["mean_delta"], base["p_value"])
    base["model_key"] = base["model"].map({m:i for i,m in enumerate(MODEL_ORDER)})
    base["size_key"]  = base["size"].map(lambda s: SIZE_RANK.get(str(s), 999))
    base["strat_key"] = base["strategy"].map(lambda s: STRAT_ORDER.get(str(s), 999))
    base["label"] = base.apply(lambda r: f"{r['model']} {r['size']} — {r['strategy']}", axis=1)
    base = base.sort_values(["model_key","size_key","strat_key"]).reset_index(drop=True)

    # Head-to-head strategy comparisons (optional)
    h2h = df[df["cmp_type"].eq("strategy")].copy()
    h2h["adopt"] = _adopt(h2h["mean_delta"], h2h["p_value"])
    h2h["model_key"] = h2h["model"].map({m:i for i,m in enumerate(MODEL_ORDER)})
    h2h["size_key"]  = h2h["size"].map(lambda s: SIZE_RANK.get(str(s), 999))
    # label: "CX small — one_shot vs zero_shot"
    h2h["label"] = h2h.apply(lambda r: f"{r['model']} {r['size']} — {r['strategy']} vs {r['rhs_strategy']}", axis=1)
    h2h = h2h.sort_values(["model_key","size_key","strategy"]).reset_index(drop=True)

    return base, h2h


def plot_forest_e3(df: pd.DataFrame,
                   out_path: str | Path,
                   title: str,
                   label_col: str = "label"):
    d = df.copy()
    d = d.dropna(subset=["mean_delta", "ci_lo", "ci_hi", "p_value", label_col])

    # Adopt flag (vectorized, default False)
    adopt = (d["adopt"].astype(bool) if "adopt" in d.columns else pd.Series(False, index=d.index)).fillna(False).values

    y = np.arange(len(d))[::-1]
    xmin = float(np.nanmin(d["ci_lo"]))
    xmax = float(np.nanmax(d["ci_hi"]))
    span = xmax - xmin if np.isfinite(xmax - xmin) and (xmax - xmin) > 0 else 1.0
    pad  = 0.08 * span

    fig_h = 0.5 * len(d) + 1.5
    fig, ax = plt.subplots(figsize=(8.0, fig_h))

    # Zero line + CIs
    ax.axvline(0.0, color="0.2", lw=1.0, ls="--")
    ax.hlines(y, d["ci_lo"], d["ci_hi"], color="black", lw=1.2, zorder=1)

    # Points: larger markers for adopted rows
    md = d["mean_delta"].values
    ax.plot(md[~adopt], y[~adopt], "o", color="black", ms=5,  zorder=2)
    ax.plot(md[adopt],  y[adopt],  "o", color="black", ms=8,  zorder=3)

    # Plain labels (no bold)
    ax.set_yticks(y, d[label_col].tolist())

    ax.set_xlabel(r"$\Delta$CRPS (negative = improvement)")
    ax.set_title(title)
    ax.set_xlim(xmin - pad, xmax + pad)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def _parse_model_size_from_note_e2(s: str):
    if not isinstance(s, str):
        return (None, None)
    m = re.search(r'\b(NF|NQ|MO|CX)\b', s, re.IGNORECASE)
    z = re.search(r'\b(tiny|small|base|large)\b', s, re.IGNORECASE)
    model = m.group(1).upper() if m else None
    size  = z.group(1).lower() if z else None
    return (model, size)

def _ensure_clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # keep CRPS rows only
    if "metric" in d.columns:
        d["metric"] = d["metric"].astype(str).str.lower().str.strip()
        d = d[d["metric"].isin(["crps_mean", "crps"])]

    # make size_clean if missing
    if "size_clean" not in d.columns:
        import re as _re
        def _clean_size(x):
            s = str(x or "").strip().lower()
            return _re.sub(r"(_|-)?(fine)?tuned$", "", s)
        d["size_clean"] = d.get("size", "").map(_clean_size)

    # prefer adopted rows, then smallest p-value, per (model,size_clean,feature)
    for col in ("adopt", "p_value"):
        if col not in d.columns:
            d[col] = pd.NA
    d["adopt_rank"] = d["adopt"].astype(bool)
    d = (d.sort_values(["model","size_clean","feature","adopt_rank","p_value"],
                       ascending=[True, True, True, False, True])
           .drop_duplicates(subset=["model","size_clean","feature"], keep="first")
           .drop(columns=["adopt_rank"])
         )

    # ordering helpers
    d["__size_key"] = d["size_clean"].map(lambda s: SIZE_RANK.get(str(s), 999))
    return d


def plot_pvalue_heatmap(df: pd.DataFrame, out_path: str | Path = "figures/pval_heatmap.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path

    d = _ensure_clean(df)

    # row order
    row_labels = []
    for m in MODEL_ORDER:
        sub = d[d["model"].eq(m)]
        if sub.empty:
            continue
        sizes = sub["size_clean"].dropna().unique().tolist()
        sizes = sorted(sizes, key=lambda s: SIZE_RANK.get(str(s), 999))
        for s in sizes:
            row_labels.append((m, s))
    desired_rows = [f"{m} {s}" for (m, s) in row_labels]

    # pivots
    d["row_label"] = d["model"] + " " + d["size_clean"]
    pmat = d.pivot(index="row_label", columns="feature", values="p_value")
    adopt_mat = d.pivot(index="row_label", columns="feature", values="adopt")

    # align & fill
    pmat = pmat.reindex(index=desired_rows, columns=[f for f in FEATURES if f in pmat.columns])
    adopt_mat = adopt_mat.reindex(index=pmat.index, columns=pmat.columns).fillna(False)

    # color values = -log10(p); mask NaNs so they render blank
    vals = pmat.to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        heat = -np.log10(vals)
    finite_mask = np.isfinite(vals)
    if np.any(np.isfinite(heat)):
        vmax = np.nanpercentile(heat[finite_mask], 95)
        vmax = max(vmax, 1.0)
    else:
        vmax = 1.0
    heat_ma = np.ma.array(heat, mask=~finite_mask)  # masked where p is NaN

    # fig
    nrows, ncols = pmat.shape
    fig_w = max(6.0, 1.2 * ncols)
    fig_h = max(4.0, 0.6 * nrows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(heat_ma, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$-\log_{10}(p)$ (larger = more significant)")

    ax.set_xticks(np.arange(ncols), labels=list(pmat.columns), rotation=0, ha="center")
    ax.set_yticks(np.arange(nrows), labels=list(pmat.index))

    # minor grid
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotate (skip NaNs) and box adopted
    for i in range(nrows):
        for j in range(ncols):
            p = pmat.iat[i, j]
            if isinstance(p, (float, np.floating)) and np.isfinite(p):
                ax.text(j, i, f"{p:.3f}", ha="center", va="center", fontsize=8, color="black")
            if bool(adopt_mat.iat[i, j]):
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, linewidth=1.8, edgecolor="black"))

    ax.set_xlabel("Feature")
    ax.set_ylabel("Model & Size")
    ax.set_title("Diebold-Mariano p-values")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_lollipop_grid(df: pd.DataFrame, out_path: str | Path = "figures/lollipop_grid.png"):
    """
    4x3 lollipop grid. One subplot per (model, size). Stem at 0, marker at ΔCRPS,
    errorbar = CI [ci_lo, ci_hi]. Everything is black; adopted features use thicker lines
    and larger markers. Requires: _ensure_clean, MODEL_ORDER, FEATURES, SIZE_RANK.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    d = _ensure_clean(df)

    # --- global y-lims from CIs/means ---
    y_min, y_max = np.inf, -np.inf
    for _, r in d.iterrows():
        for v in (r.get("ci_lo"), r.get("ci_hi"), r.get("mean_delta")):
            if pd.notna(v):
                y_min = min(y_min, v)
                y_max = max(y_max, v)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0
    pad = 0.1 * max(abs(y_min), abs(y_max))
    ylim = (y_min - pad, y_max + pad)

    # --- figure (force white bg) ---
    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4*n_cols, 2.6*n_rows), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])

    # --- helper: single panel ---
    def plot_panel(ax, sub: pd.DataFrame, title: str):
        ax.set_facecolor("white")
        # all black theme
        AX_KW = dict(color="black")
        STEM_LW = 1.2
        STEM_LW_ADOPT = 1.2
        MS = 4.0
        MS_ADOPT = 7.5
        CAPSIZE = 3

        feats = [f for f in FEATURES if f in sub["feature"].unique()]
        x = np.arange(len(feats))
        sub = sub.set_index("feature")

        for i, f in enumerate(feats):
            if f not in sub.index:
                continue
            md = sub.at[f, "mean_delta"]
            lo = sub.at[f, "ci_lo"]
            hi = sub.at[f, "ci_hi"]
            ad = bool(sub.at[f, "adopt"])
            if pd.isna(md):
                continue

            lw = STEM_LW_ADOPT if ad else STEM_LW
            ms = MS_ADOPT if ad else MS

            # stem (from 0 to md)
            ax.vlines(i, 0, md, linewidth=lw, **AX_KW)

            # point
            ax.plot(i, md, marker="o", markersize=ms, mfc="black", mec="black", linestyle="None")

            # error bar if CI present
            if pd.notna(lo) and pd.notna(hi):
                yerr = [[md - lo], [hi - md]]
                ax.errorbar(i, md, yerr=yerr, fmt="none", ecolor="black",
                            elinewidth=lw*0.5, capsize=CAPSIZE, capthick=lw*0.5)

        # zero line & cosmetics
        ax.axhline(0.0, **AX_KW, linewidth=1.0, linestyle="--")
        ax.set_xticks(x, feats)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=10, color="black")
        # simple black spines
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_color("black")
        ax.tick_params(colors="black")

    # --- grid fill ---
    for r, m in enumerate(MODEL_ORDER):
        sub_m = d[d["model"].eq(m)]
        sizes = sorted(sub_m["size_clean"].dropna().unique(), key=lambda s: SIZE_RANK.get(str(s), 999))
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(sizes):
                s = sizes[c]
                sub = sub_m[sub_m["size_clean"].eq(s)]
                plot_panel(ax, sub, f"{m} {s}")
            else:
                ax.set_facecolor("white")
                ax.axis("off")

    fig.supylabel(r"$\Delta$CRPS (negative = improvement)", color="black")
    fig.supxlabel("Feature", color="black")
    fig.suptitle("Feature effects by model-size (95% CI)", y=0.98, color="black")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def plot_forest_e2(df: pd.DataFrame,
                   out_path: str | Path = "figures/forest_dm.png",
                   title: str = "Baseline comparisons (ΔCRPS; 95% CI)"):
    df = df.copy()

    # Parse model/size if missing
    if "model" not in df.columns:
        df[["model","size"]] = df["note"].apply(_parse_model_size_from_note_e2).apply(pd.Series)
    if "size" not in df.columns:
        df["size"] = df["size"]

    # Keep essentials
    cols_needed = ["model","size","mean_delta","ci_lo","ci_hi","p_value"]
    df = df[[c for c in cols_needed if c in df.columns] + (["decision"] if "decision" in df.columns else [])]
    df = df.dropna(subset=["mean_delta","ci_lo","ci_hi","p_value","model","size"])

    # One row per (model,size): keep smallest p
    df = (df.sort_values(["model","size","p_value"])
            .drop_duplicates(["model","size"], keep="first"))

    # Adopt flag
    if "decision" in df.columns:
        df["adopt"] = df["decision"].astype(str).str.upper().eq("ADOPT")
    else:
        df["adopt"] = (df["p_value"] < 0.05) & (df["mean_delta"] < 0.0)

    # Order
    df["model_key"] = df["model"].map({m:i for i,m in enumerate(MODEL_ORDER)})
    df["size_key"]  = df["size"].map(lambda s: SIZE_RANK.get(str(s), 999))
    df = df.sort_values(["model_key","size_key"]).reset_index(drop=True)

    # Labels & positions
    labels = df.apply(lambda r: f"{r['model']} {r['size']}", axis=1)
    y = np.arange(len(df))[::-1]

    # Axis limits with padding
    xmin = float(np.nanmin(df["ci_lo"]))
    xmax = float(np.nanmax(df["ci_hi"]))
    span = xmax - xmin if np.isfinite(xmax - xmin) and (xmax - xmin) > 0 else 1.0
    pad  = 0.08 * span

    # Figure
    fig_h = 0.5 * len(df) + 1.5
    fig, ax = plt.subplots(figsize=(8.0, fig_h))

    # Zero line + CIs
    ax.axvline(0.0, color="0.2", lw=1.0, ls="--")
    ax.hlines(y, df["ci_lo"], df["ci_hi"], color="black", lw=1.2, zorder=1)

    # Points (bigger if adopted)
    md = df["mean_delta"].values
    adopt = df["adopt"].fillna(False).to_numpy(dtype=bool)
    ax.plot(md[~adopt], y[~adopt], "o", color="black", ms=5, zorder=2)
    ax.plot(md[adopt],  y[adopt],  "o", color="black", ms=8, zorder=3)

    # Plain labels (no bold)
    ax.set_yticks(y, labels)

    ax.set_xlabel(r"$\Delta$CRPS (negative = improvement)")
    ax.set_title(title)
    ax.set_xlim(xmin - pad, xmax + pad)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_forest_e1(df: pd.DataFrame,
                   out_path: str | Path = "figures/forest_dm.png",
                   title: str = "Baseline comparisons (ΔCRPS; 95% CI)"):
    df = df.copy()

    # Parse model/size if missing
    if "model" not in df.columns:
        df[["model","size"]] = df["note"].apply(_parse_model_size_from_note_e2).apply(pd.Series)
    if "size" not in df.columns:
        df["size"] = df["size"]

    # Keep essentials
    cols_needed = ["model","size","mean_delta","ci_lo","ci_hi","p_value"]
    df = df[[c for c in cols_needed if c in df.columns] + (["decision"] if "decision" in df.columns else [])]
    df = df.dropna(subset=["mean_delta","ci_lo","ci_hi","p_value","model","size"])

    # One row per (model,size): keep smallest p
    df = (df.sort_values(["model","size","p_value"])
            .drop_duplicates(["model","size"], keep="first"))

    # Adopt flag
    if "decision" in df.columns:
        df["adopt"] = df["decision"].astype(str).str.upper().eq("ADOPT")
    else:
        df["adopt"] = (df["p_value"] < 0.05) & (df["mean_delta"] < 0.0)

    # Order
    df["model_key"] = df["model"].map({m:i for i,m in enumerate(MODEL_ORDER)})
    df["size_key"]  = df["size"].map(lambda s: SIZE_RANK.get(str(s), 999))
    df = df.sort_values(["model_key","size_key"]).reset_index(drop=True)

    # Labels & positions
    labels = df.apply(lambda r: f"{r['model']} {r['size']}", axis=1)
    y = np.arange(len(df))[::-1]

    # Axis limits with padding
    xmin = float(np.nanmin(df["ci_lo"]))
    xmax = float(np.nanmax(df["ci_hi"]))
    span = xmax - xmin if np.isfinite(xmax - xmin) and (xmax - xmin) > 0 else 1.0
    pad  = 0.08 * span

    # Figure
    fig_h = 0.5 * len(df) + 1.5
    fig, ax = plt.subplots(figsize=(8.0, fig_h))

    # Zero line + CIs
    ax.axvline(0.0, color="0.2", lw=1.0, ls="--")
    ax.hlines(y, df["ci_lo"], df["ci_hi"], color="black", lw=1.2, zorder=1)

    # Points (bigger if adopted)
    md = df["mean_delta"].values
    adopt = df["adopt"].fillna(False).to_numpy(dtype=bool)
    ax.plot(md[~adopt], y[~adopt], "o", color="black", ms=5, zorder=2)
    ax.plot(md[adopt],  y[adopt],  "o", color="black", ms=8, zorder=3)

    # Plain labels (no bold)
    ax.set_yticks(y, labels)

    ax.set_xlabel(r"$\Delta$CRPS (negative = improvement)")
    ax.set_title(title)
    ax.set_xlim(xmin - pad, xmax + pad)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------- example usage -------------------
if __name__ == "__main__":
    # Suppose your combined dataframe is in 'summary_tables_w_ci.parquet'
    # df = pd.read_parquet("summary_tables_w_ci.parquet")
    # Or if it's already in memory as 'summary_tables_w_ci', just pass it in:
    pass