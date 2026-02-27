# models/nhits_qra/helpers.py
from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np

def write_model_io(meta_dir: Path, *,
                   ck_cols: List[str],
                   cu_cols: List[str],
                   static_cols: List[str],
                   add_flag: bool,
                   hist_exog_list: List[str],
                   futr_exog_list: List[str],
                   stat_exog_list: List[str]) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "ck_cols": list(ck_cols),
        "cu_cols": list(cu_cols),
        "static_cols": list(static_cols),
        "add_flag": bool(add_flag),
        "hist_exog_list": list(hist_exog_list),
        "futr_exog_list": list(futr_exog_list),
        "stat_exog_list": list(stat_exog_list),
    }
    p = meta_dir / "model_io.json"
    p.write_text(json.dumps(out, indent=2))
    return p

def read_model_io(train_run_dir: Path) -> Dict[str, Any]:
    p = train_run_dir / "meta" / "model_io.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing model_io.json at {p}")
    return json.loads(p.read_text())


def extract_pl_metrics(trainer) -> dict:
    out = {}
    for k, v in trainer.callback_metrics.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def build_test_metrics(*, trainer, qra_summary, train_cfg, test_cfg) -> dict:
    metrics = {}

    # 1) Always: pull PL callback metrics (if any)
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass

    # Convenience: horizon + samples if present
    try:
        metrics["H"] = int(train_cfg.data.forecast_horizon)
    except Exception:
        pass
    try:
        metrics["S"] = int(test_cfg.test.n_samples_eval)
    except Exception:
        pass

    # 2) If no QRA summary, we're done
    if qra_summary is None:
        return metrics

    # 3) QRA summary → standardized keys
    try:
        metrics["es_mean_over_all_origins"] = float(qra_summary["metrics"]["ES_mean"])
    except Exception:
        if "test_mean_es" in qra_summary:
            metrics["es_mean_over_all_origins"] = float(qra_summary["test_mean_es"])

    try:
        metrics["crps_mean_over_all_origins"] = float(qra_summary["metrics"]["CRPS_mean"])
    except Exception:
        if "test_mean_crps" in qra_summary:
            metrics["crps_mean_over_all_origins"] = float(qra_summary["test_mean_crps"])

    try:
        metrics["windows"] = int(qra_summary["data"]["N_origins"])
    except Exception:
        if "N_origins" in qra_summary:
            metrics["windows"] = int(qra_summary["N_origins"])

    # Berkowitz summary
    berk = {}
    try:
        berk = qra_summary["calibration"]["berkowitz"]
    except Exception:
        pass

    try:
        fisher = berk.get("fisher", {})
        if fisher:
            if fisher.get("p", None) is not None:
                metrics["berkowitz_fisher_p"] = float(fisher["p"])
            if fisher.get("stat", None) is not None:
                metrics["berkowitz_fisher_stat"] = float(fisher["stat"])
            metrics["berkowitz_fisher_df"] = int(fisher.get("df", 0))
            metrics["berkowitz_fisher_k"] = int(fisher.get("k", 0))
            if fisher.get("hmp", None) is not None:
                metrics["berkowitz_hmp_p"] = float(fisher["hmp"])
    except Exception:
        pass

    # Per-horizon rollups (optional)
    try:
        per_h = berk.get("per_h", [])
        pvals = [r.get("p") for r in per_h if r.get("p") is not None and np.isfinite(r.get("p"))]
        if pvals:
            pvals = np.asarray(pvals, float)
            metrics["berkowitz_per_h_min_p"] = float(np.min(pvals))
            metrics["berkowitz_per_h_median_p"] = float(np.median(pvals))
            metrics["berkowitz_per_h_max_p"] = float(np.max(pvals))
            metrics["berkowitz_per_h_num"] = int(len(pvals))
    except Exception:
        pass

    return metrics