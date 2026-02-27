from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np

BASE_DIR = Path("outputs")

MODEL_TAGS = {
    "chronosx": "CX",
    "moirai": "MO",
    "normalizing_flows": "NF",
    "nhits_qra": "NQ",
}

FEATURE_MAP = {
    "gas_price": ("GP", "Gas price"),
    "cross_border_trading": ("CBT", "Cross-border trading"),
    "non_renewable": ("NRE", "Non-renewables"),
    "renewable": ("RE", "Renewables"),
    "load": ("LOAD", "Load"),
    "co2_emission_allowances": ("CO2", "CO₂ emission allowances"),
    "synthetic_price": ("SP", "Synthetic price"),
}

PATH_RE = re.compile(
    r".*/outputs/(?P<model>[^/]+)/(?P<size>[^/]+)/(?P<exp>[^/]+)/super_runs/(?P<trial_id>[^/]+)/summary\.csv$"
)

# NEW: normalize size + tuned flag
def _split_size(size: str) -> tuple[str, str, bool]:
    s = str(size)
    s_l = s.lower()
    # strip trailing separators + 'tuned' (e.g., tiny_tuned, tiny-tuned, tiny.tuned)
    size_clean = re.sub(r'(?i)[_\-\.]?tuned$', '', s_l).strip()
    is_tuned = (size_clean != s_l) or ("tuned" in s_l)
    return s, size_clean, is_tuned

def _read_summary_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=";", engine="python")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "pvalue": "p_value",
        "p-val": "p_value",
        "dm": "dm_stat",
        "dm_statistic": "dm_stat",
        "t_stat": "t",
    }
    df = df.rename(columns=rename_map)

    for col in ["p_value", "dm_stat", "t", "lag", "mean_delta", "median_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "adopt" in df.columns:
        df["adopt"] = df["adopt"].astype(str).str.strip().str.lower().isin({"true","1","yes","y"})
    else:
        df["adopt"] = False

    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.lower().str.strip().replace({"es_mean": "crps_mean"})
    else:
        df["metric"] = "crps_mean"

    needed = ["trial_id","metric","p_value","dm_stat","t","lag","mean_delta","median_delta","adopt"]
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def _parse_feature_from_trial_id(trial_id: str) -> tuple[str, str, str]:
    if not isinstance(trial_id, str):
        return ("unknown", "UNK", "Unknown")
    m = re.match(r"R\d+_(.+)", trial_id)
    slug = (m.group(1) if m else trial_id).strip().lower()
    slug = slug.replace("-", "_").replace(" ", "_")
    short, pretty = FEATURE_MAP.get(slug, (slug.upper(), slug.replace("_", " ").title()))
    return slug, short, pretty

def _parse_model_size_trial(path: Path) -> tuple[str, str, str, str]:
    m = PATH_RE.match(str(path))
    if not m:
        raise ValueError(f"Unexpected path layout: {path}")
    model_folder = m.group("model")
    size = m.group("size")
    trial_id = m.group("trial_id")
    model_tag = MODEL_TAGS.get(model_folder, model_folder.upper())
    return model_folder, model_tag, size, trial_id

def load_all_summaries(base_dir: Path | str = BASE_DIR) -> pd.DataFrame:
    outputs_root = Path(base_dir) / "outputs"
    if not outputs_root.exists():
        return pd.DataFrame()  # or raise FileNotFoundError(f"{outputs_root} not found")
    files = sorted(outputs_root.rglob("summary.csv"))

    rows = []
    for f in files:
        try:
            model_folder, model_tag, size_path, trial_id = _parse_model_size_trial(f)
        except ValueError:
            continue

        df = _read_summary_csv(f)

        # Parse/attach size variants
        size_raw, size_clean, is_tuned = _split_size(size_path)

        # Attach path-derived columns
        df["model_folder"] = model_folder
        df["model"] = model_tag
        df["size"] = size_raw              # keep original (e.g., 'tiny_tuned')
        df["size_clean"] = size_clean      # normalized (e.g., 'tiny')
        df["is_tuned"] = bool(is_tuned)    # True/False
        df["trial_id"] = df["trial_id"].astype(str).fillna(trial_id)
        df["source_path"] = str(f)

        # Feature columns from trial_id
        feats = df["trial_id"].apply(_parse_feature_from_trial_id)
        df["feature_slug"] = feats.map(lambda x: x[0])
        df["feature"] = feats.map(lambda x: x[1])
        df["feature_name"] = feats.map(lambda x: x[2])

        rows.append(df)

    if not rows:
        cols = [
            "model","model_folder","size","size_clean","is_tuned","trial_id",
            "feature","feature_slug","feature_name",
            "metric","mean_delta","median_delta","p_value","dm_stat","t","lag",
            "adopt","source_path",
        ]
        return pd.DataFrame(columns=cols)

    out = pd.concat(rows, ignore_index=True)

    # Column order + fill
    col_order = [
        "model","model_folder","size","size_clean","is_tuned","trial_id",
        "feature","feature_slug","feature_name",
        "metric","mean_delta","median_delta","p_value","dm_stat","t","lag",
        "adopt","source_path",
    ]
    for c in col_order:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[col_order]

    # Sort: model -> size_clean (tiny < small < base < large) -> feature order
    feat_order = list(FEATURE_MAP.keys())
    out["__feat_rank"] = out["feature_slug"].map({k:i for i,k in enumerate(feat_order)}).fillna(999).astype(int)

    size_rank_map = {"tiny":0, "small":1, "base":2, "large":3}
    out["__size_rank"] = out["size_clean"].map(size_rank_map).fillna(999).astype(int)

    out = (out
           .sort_values(["model","__size_rank","__feat_rank","trial_id"])
           .drop(columns=["__size_rank","__feat_rank"])
           .reset_index(drop=True))

    return out


def add_confidence_intervals(
    df: pd.DataFrame,
    alpha: float = 0.05,
    use_student_t_when_T: bool = False,
    horizon_from_lag: bool = True,
) -> pd.DataFrame:
    """
    Add SE and (1-alpha) confidence intervals for mean_delta using the DM test.
    
    Assumptions:
      - `mean_delta` is the sample mean of the loss differential d_t (e.g., CRPS_modelB - CRPS_modelA).
      - `dm_stat` is DM (or HLN-corrected DM) computed with HAC/NeweyWest variance.
      - If `dm_stat` is missing but `p_value` is present, we invert the two-sided p-value.
      - If T is available (column 't'), we can use Student-t critical values; otherwise normal.
      - If `assume_hln_correction=True`, we *apply* the HLN small-sample adjustment to DM
        (only do this if stored `dm_stat` is the *raw* DM; if `dm_stat` is
        already HLN-adjusted, leave this False).

    Columns expected if available: ['mean_delta', 'dm_stat', 'p_value', 't', 'lag'].
    Returns a copy with: ['se_dm', 'ci_level', 'ci_lo', 'ci_hi', 'ci_excludes_zero'] appended.
    """
    out = df.copy()

    # Safe imports for critical values
    try:
        from scipy.stats import norm, t as t_dist
        have_scipy = True
    except Exception:
        have_scipy = False

    def _crit_val(T):
        if use_student_t_when_T and pd.notna(T) and T and T > 2 and have_scipy:
            return t_dist.ppf(1 - alpha / 2.0, df=int(T) - 1)
        # fallback: normal 1.96-ish
        if have_scipy:
            from scipy.stats import norm
            return norm.ppf(1 - alpha / 2.0)
        return 1.959963984540054  # hard-coded z_0.975

    def _z_from_p(p):
        if not have_scipy or pd.isna(p) or not (0 < p < 1):
            return np.nan
        from scipy.stats import norm
        return norm.isf(p / 2.0)  # two-sided

    se_list, lo_list, hi_list, excl_zero = [], [], [], []

    for _, r in out.iterrows():
        md = r.get("mean_delta", np.nan)
        dm = r.get("dm_stat", np.nan)
        pv = r.get("p_value", np.nan)
        T  = r.get("t", np.nan)
        lag = r.get("lag", np.nan)


        # Derive SE from DM or from p-value
        if pd.notna(dm) and dm != 0 and pd.notna(md):
            se = abs(float(md)) / abs(float(dm))
        elif pd.notna(pv) and pd.notna(md):
            z = _z_from_p(float(pv))
            se = abs(float(md)) / z if (z is not None and z > 0) else np.nan
        else:
            se = np.nan

        crit = _crit_val(T)
        if pd.notna(md) and pd.notna(se) and np.isfinite(se):
            lo = float(md) - crit * se
            hi = float(md) + crit * se
        else:
            lo, hi = np.nan, np.nan

        se_list.append(se)
        lo_list.append(lo)
        hi_list.append(hi)
        excl_zero.append( (lo > 0) or (hi < 0) )

    out["se_dm"] = se_list
    out["ci_level"] = 1 - alpha
    out["ci_lo"] = lo_list
    out["ci_hi"] = hi_list
    out["ci_excludes_zero"] = excl_zero
    return out



def make_brief_table(
    df: pd.DataFrame,
    *,
    feature_order = ("GP","CBT","NRE","RE","LOAD","CO2","SP"),
    model_order   = ("NF","NQ","MO","CX"),
    size_order_map = {"tiny": 0, "small": 1, "base": 2, "large": 3},
    decimals: int = 2,
    include_ci: bool = True,
    adopted_sign: str = "",          # no glyph; we bold instead
    bold_adopted: bool = True,
    flatten_index: bool = False,     # used only when returning a DataFrame
    line_break_ci: bool = True,      # used in 'stacked' layout
    use_makecell: bool = True,       # used in 'stacked' layout
    as_latex: bool = False,          # set True to emit LaTeX string
    layout: str = "split_rows",      # "split_rows" (Δ row + CI row) or "stacked"
    add_vertical_after_size: bool = True,
    stacked_col_width_cm: float = 2.2,  # width per feature col in 'stacked' mode
) -> Union[pd.DataFrame, str]:
    import re as _re
    d = df.copy()

    # --- ensure size_clean exists ---
    if "size_clean" not in d.columns:
        def _clean_size(x):
            s = str(x or "").strip().lower()
            return _re.sub(r"(_|-)?(fine)?tuned$", "", s)
        d["size_clean"] = d.get("size", "").map(_clean_size)

    # --- keep CRPS rows only ---
    if "metric" in d.columns:
        d["metric"] = d["metric"].astype(str).str.lower().str.strip()
        d = d[d["metric"].isin(["crps_mean", "crps"])]

    # --- de-dup per (model,size,feature): prefer adopted=True, then smallest p-value ---
    for col in ("adopt", "p_value"):
        if col not in d.columns:
            d[col] = pd.NA
    d["adopt_rank"] = d["adopt"].astype(bool)
    d = (
        d.sort_values(["model","size_clean","feature","adopt_rank","p_value"],
                      ascending=[True, True, True, False, True])
         .drop_duplicates(subset=["model","size_clean","feature"], keep="first")
         .drop(columns=["adopt_rank"])
    )

    # If caller wants a tidy wide DataFrame, return that
    if not as_latex:
        def _fmt(md, lo, hi, ad):
            if pd.isna(md):
                return ""
            s = f"{md:+.{decimals}f}"
            if include_ci and pd.notna(lo) and pd.notna(hi):
                s += f" [{lo:+.{decimals}f}, {hi:+.{decimals}f}]"
            if bold_adopted and bool(ad):
                s = rf"\textbf{{{s}}}"
            return s

        d["cell"] = d.apply(lambda r: _fmt(r.get("mean_delta"), r.get("ci_lo"), r.get("ci_hi"), r.get("adopt")), axis=1)
        wide = d.pivot_table(index=["model","size_clean"], columns="feature", values="cell", aggfunc="first")

        # order columns and rows
        cols = [f for f in feature_order if f in wide.columns]
        other = [c for c in wide.columns if c not in cols]
        wide = wide.reindex(columns=cols + other)

        desired_idx = []
        for m in model_order:
            if m not in wide.index.get_level_values("model"):
                continue
            sub_sizes = sorted(
                wide.loc[m].index.unique().tolist(),
                key=lambda s: size_order_map.get(str(s), 999)
            )
            desired_idx.extend((m, s) for s in sub_sizes if (m, s) in wide.index)
        if desired_idx:
            wide = wide.reindex(desired_idx)

        if flatten_index:
            wide.index = [f"{m} {s}" for m, s in wide.index]
        else:
            wide.index.set_names(["Model","Size"], inplace=True)
            wide.columns.name = "Feature"
        return wide.fillna("")

    # ---------------- LaTeX emission ----------------
    # Prepare feature order present
    feat_list = [f for f in feature_order if f in d["feature"].unique()]
    feat_idx = {f:i for i,f in enumerate(feat_list)}
    def _size_sort_key(s): return size_order_map.get(str(s), 999)

    if layout == "split_rows":
        # Build nested dict with separate Δ, CI, stat, and p-value cells, bold if adopted
        table_data = {}
        for m in model_order:
            sub_m = d[d["model"] == m]
            if sub_m.empty:
                continue
            table_data[m] = {}
            for s in sorted(sub_m["size_clean"].unique(), key=_size_sort_key):
                sub_ms = sub_m[sub_m["size_clean"] == s]
                delta_cells = [""] * len(feat_list)
                ci_cells    = [""] * len(feat_list)
                stat_cells  = [""] * len(feat_list)
                p_cells     = [""] * len(feat_list)

                for _, r in sub_ms.iterrows():
                    f = r["feature"]
                    if f not in feat_idx:
                        continue
                    i  = feat_idx[f]
                    ad = bool(r.get("adopt"))
                    md = r.get("mean_delta")
                    lo, hi = r.get("ci_lo"), r.get("ci_hi")
                    p  = r.get("p_value")
                    z  = r.get("dm_stat")   

                    # Δ row
                    if pd.notna(md):
                        delta_txt = f"{float(md):+.{decimals}f}"
                        if ad and bold_adopted:
                            delta_txt = rf"\textbf{{{delta_txt}}}"
                        delta_cells[i] = delta_txt

                    # CI row
                    if include_ci and pd.notna(lo) and pd.notna(hi):
                        ci_txt = f"[{float(lo):+.{decimals}f}, {float(hi):+.{decimals}f}]"
                        if ad and bold_adopted:
                            ci_txt = rf"\textbf{{{ci_txt}}}"
                        ci_cells[i] = ci_txt

                    # test statistic row (e.g. DM t/z)
                    if pd.notna(z):
                        stat_txt = f"{float(z):+.2f}"
                        if ad and bold_adopted:
                            stat_txt = rf"\textbf{{{stat_txt}}}"
                        stat_cells[i] = stat_txt

                    # p-value row
                    if pd.notna(p):
                        p = float(p)
                        if p < 0.001:
                            p_txt = r"$< .001$"
                        else:
                            p_txt = f"{p:.3f}"
                            if p_txt.startswith("0"):
                                p_txt = p_txt[1:]
                            p_txt = f"${p_txt}$"
                        if ad and bold_adopted:
                            p_txt = rf"\textbf{{{p_txt}}}"
                        p_cells[i] = p_txt

                table_data[m][s] = {
                    "delta": delta_cells,
                    "ci":    ci_cells,
                    "stat":  stat_cells,
                    "p":     p_cells,
                }

        nF = len(feat_list)
        feature_head = " & ".join(feat_list)
        col_spec = ("l l|" if add_vertical_after_size else "l l ") + ("c"*nF)

        lines = []
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        lines.append(
            r"\multirow{2}{*}{Model} & \multirow{2}{*}{Size} & \multicolumn{"
            + str(nF) + r"}{c}{Feature} \\"
        )
        lines.append(r"\cmidrule(lr){3-" + str(2 + nF) + r"}")
        lines.append(r" &  & " + feature_head + r" \\")
        lines.append(r"\midrule")

        for m in model_order:
            if m not in table_data:
                continue
            sizes = list(table_data[m].keys())
            if not sizes:
                continue
            # Δ row + CI row + stat row + p row
            block_rows = 4 * len(sizes)
            first = True
            for s in sizes:
                deltas = table_data[m][s]["delta"]
                cis    = table_data[m][s]["ci"]
                stats  = table_data[m][s]["stat"]
                ps     = table_data[m][s]["p"]

                # Δ row
                if first:
                    lines.append(
                        rf"\multirow{{{block_rows}}}{{*}}{{{m}}} & {s} & "
                        + " & ".join([x if x else r"--" for x in deltas]) + r" \\"
                    )
                    first = False
                else:
                    lines.append(
                        rf" & {s} & "
                        + " & ".join([x if x else r"--" for x in deltas]) + r" \\"
                    )

                # CI row
                lines.append(
                    rf" &  & " + " & ".join([c if c else r"--" for c in cis]) + r" \\"
                )

                # test statistic row
                lines.append(
                    rf" &  & " + " & ".join([z if z else r"--" for z in stats]) + r" \\"
                )

                # p-value row
                lines.append(
                    rf" &  & " + " & ".join([p if p else r"--" for p in ps]) + r" \\"
                )

            lines.append(r"\addlinespace")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    elif layout == "stacked":
        # Single row per size: Δ on first line, CI on second line inside each cell.
        # Requires: \usepackage{booktabs,makecell,array}
        # We use m{<width>} and \centering for nicer wrapping.
        def fmt_cell(md, lo, hi, ad):
            if pd.isna(md):
                return ""
            main = f"{float(md):+.{decimals}f}"
            ci   = f"[{float(lo):+.{decimals}f}, {float(hi):+.{decimals}f}]" if (include_ci and pd.notna(lo) and pd.notna(hi)) else ""
            body = main if not ci or not line_break_ci else rf"{main} \\ {ci}"
            if use_makecell and line_break_ci and ci:
                body = rf"\makecell{{{body}}}"
            if ad and bold_adopted:
                body = rf"\textbf{{{body}}}"
            return body

        # Build row strings
        rows = []
        for m in model_order:
            sub_m = d[d["model"] == m]
            if sub_m.empty:
                continue
            for s in sorted(sub_m["size_clean"].unique(), key=_size_sort_key):
                sub_ms = sub_m[sub_m["size_clean"] == s]
                cells = []
                for f in feat_list:
                    r = sub_ms[sub_ms["feature"] == f]
                    if r.empty:
                        cells.append("--")
                    else:
                        r = r.iloc[0]
                        cells.append(fmt_cell(r.get("mean_delta"), r.get("ci_lo"), r.get("ci_hi"), bool(r.get("adopt"))))
                rows.append((m, s, cells))

        nF = len(feat_list)
        feature_head = " & ".join(feat_list)
        width = f"{stacked_col_width_cm:.2f}cm"
        feat_cols = " ".join([rf">{{\centering\arraybackslash}}m{{{width}}}" for _ in range(nF)])
        col_spec = ("l l|" if add_vertical_after_size else "l l ") + feat_cols

        lines = []
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        lines.append(r"Model & Size & " + feature_head + r" \\")
        lines.append(r"\midrule")
        # multirow per model (number of sizes for that model)
        i = 0
        while i < len(rows):
            m = rows[i][0]
            # count how many consecutive rows share the same model
            j = i
            count = 0
            while j < len(rows) and rows[j][0] == m:
                count += 1; j += 1
            # print those 'count' rows with multirow
            for k in range(i, j):
                _, s, cells = rows[k]
                if k == i:
                    lines.append(rf"\multirow{{{count}}}{{*}}{{{m}}} & {s} & " + " & ".join(cells) + r" \\")
                else:
                    lines.append(rf" & {s} & " + " & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
            i = j

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    else:
        raise ValueError("layout must be 'split_rows' or 'stacked'")