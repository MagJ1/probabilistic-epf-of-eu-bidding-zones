from pathlib import Path
import numpy as np

def _pp_path(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)

def _from_anchor(p: Path, anchor: str = "outputs") -> str:
    p = Path(p).resolve()
    parts = p.parts
    try:
        i = parts.index(anchor)
        return str(Path(*parts[i:]))   # "outputs/..."
    except ValueError:
        return str(p)                  # fallback: full path
    
def log_train_run_header(log, *, cfg, run_id: str, run_dir: Path, dirs: dict, monitor_key: str):
    log.info("=" * 88)
    log.info("NHITS TRAIN RUN")
    log.info("  Complete Path    : %s", _from_anchor(cfg.runs_root, "outputs"))
    log.info("  run_id           : %s", str(run_id))
    log.info("  runs_root        : %s", _pp_path(Path(cfg.runs_root)))
    log.info("  this_run_dir     : %s", _pp_path(run_dir))
    log.info("  seed             : %s", str(cfg.seed))
    log.info("  monitor_key      : %s", str(monitor_key))
    log.info("  ckpt_dir         : %s", _pp_path(dirs["ckpt"] / "nhits"))
    log.info("  metrics_dir      : %s", _pp_path(dirs["metrics"]))
    log.info("  meta_dir         : %s", _pp_path(dirs["meta"]))
    log.info("  data_dir         : %s", _pp_path(dirs["data"]))
    log.info("=" * 88)

def _brief_train_summary(metrics: dict, *, monitor_key: str, ckpt_cb) -> dict:
    def _gf(k):
        v = metrics.get(k, None)
        try:
            return float(v)
        except Exception:
            return float("nan")

    best_path = getattr(ckpt_cb, "best_model_path", "") or ""
    best_score = getattr(ckpt_cb, "best_model_score", None)
    try:
        best_score = float(best_score) if best_score is not None else float("nan")
    except Exception:
        best_score = float("nan")

    return {
        "best_ckpt": Path(best_path).name if best_path else "",
        "best_score": round(best_score, 6) if np.isfinite(best_score) else float("nan"),
        "train_loss": round(_gf("train_loss"), 6) if np.isfinite(_gf("train_loss")) else float("nan"),
        monitor_key:  round(_gf(monitor_key), 6) if np.isfinite(_gf(monitor_key)) else float("nan"),
    }

def log_test_run_header(log, *, cfg, train_run_dir: Path, test_run_dir: Path, dirs: dict, ckpt_path: Path):
    log.info("=" * 88)
    log.info("NHITS+QRA TEST RUN")
    log.info("  Complete Path    : %s", _from_anchor(cfg.out_dir, "outputs"))
    log.info("  tag              : %s", str(getattr(cfg, "tag", "test")))
    log.info("  source_run_id    : %s", str(cfg.test.source_run_id))
    log.info("  source_run_dir   : %s", _pp_path(train_run_dir))
    log.info("  this_test_outdir : %s", test_run_dir)
    log.info("  ckpt_used        : %s", ckpt_path.name)
    log.info("  metrics_dir      : %s", _pp_path(dirs["metrics"]))
    log.info("  artifacts_dir    : %s", _pp_path(dirs["art"]))
    log.info("  preds_dir        : %s", _pp_path(dirs["pred"]))
    log.info("=" * 88)

def brief_qra_summary(s):
    fisher = s["calibration"]["berkowitz"]["fisher"]
    p = fisher.get("p", None)
    hmp = fisher.get("hmp", None)
    return {
        "CRPS": round(float(s["metrics"]["CRPS_mean"]), 4),
        "ES":   round(float(s["metrics"]["ES_mean"]), 4),
        "N":    int(s["data"]["N_origins"]),
        "H":    int(s["data"]["n_horizons"]),
        "fisher_p": float(p) if p is not None else float("nan"),
        "hmp_p":    float(hmp) if hmp is not None else float("nan"),
    }