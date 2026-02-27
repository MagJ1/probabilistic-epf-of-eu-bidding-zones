# utils/super_runner/super_runner_common.py
from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Dict, Any, List
import os, shlex, subprocess, time
from dataclasses import dataclass
import warnings

import copy
import pandas as pd
import numpy as np
import inspect

from math import floor
from scipy.stats import t as student_t
from utils.super_runner.trial_items import (
    _queue_head_matches_item, 
    apply_item_to_features, 
    parse_trial_item, 
    trial_item_label, 
    trial_item_to_dict, 
    split_queue_entry, 
    TrialPolicy)
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from utils.feature_hash import feature_hash
from utils.super_runner.base_backend import any_ckpt_exists

def _has_checkpoint_backend(run_dir: Path, backend) -> bool:
    # legacy fallback
    return any_ckpt_exists(backend.train_checkpoint_globs(run_dir))

def _is_train_complete_backend(run_dir: Path, backend) -> bool:
    # Prefer the backend’s explicit readiness check
    if hasattr(backend, "train_is_complete"):
        return backend.train_is_complete(run_dir)
    # Fallback to the old behavior
    return run_dir.exists() and _has_checkpoint_backend(run_dir, backend)

def _is_test_complete_backend(run_dir: Path, test_tag: str, backend) -> bool:
    return backend.test_is_complete(run_dir, test_tag)


################
### STATE IO ###
################

def _atomic_write_json(obj: Dict[str, Any], dst: Path) -> None:
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(dst)

def _atomic_read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_state(state_path: Path) -> Dict[str, Any] | None:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return None


def _init_state(cfg: DictConfig, super_dir: Path) -> Dict[str, Any]:
    """Create initial state.json from cfg (first run of the campaign)."""
    # Coerce safely; some models don't define static_cols
    def _as_list(x): 
        try: 
            return list(x) if x is not None else []
        except Exception:
            return []

    base_features = {
        "ck_cols": _as_list(getattr(cfg.features, "ck_cols", [])),
        "cu_cols": _as_list(getattr(cfg.features, "cu_cols", [])),
        # Always present to keep a uniform schema across backends
        "static_cols": _as_list(getattr(cfg.features, "static_cols", [])),
    }

    baseline = {
    "features": base_features,
    "run_dir": None,
    "run_id": None,
    "test_tag": "BASELINE",
    "test_dir": None,
    "metrics": {},
    }

    state = {
        "version": 2,
        "type": cfg.model.model_type,
        "size": cfg.model.size,
        "experiment": cfg.experiment,
        "alpha": float(cfg.decision.alpha),
        "nw_lag_rule": str(cfg.decision.nw_lag_rule),
        "seed_policy": OmegaConf.to_container(cfg.seed_policy, resolve=True),
        "current_round": 1,
        # frozen
        "baseline": baseline,
        # moving reference (starts identical)
        "working_baseline": copy.deepcopy(baseline),
        "selected_features": [],
        "candidates_queue": list(OmegaConf.to_container(cfg.candidates.candidates_queue, resolve=True)),
        "trials": {},
    }
    _atomic_write_json(state, super_dir / "state.json")
    return state


def _upgrade_state_inplace(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible state upgrade:
      - Introduce working_baseline if missing
      - Ensure schema keys exist
    """
    if state is None:
        return state

    # If older runs don't have working_baseline, initialize it from baseline.
    if "working_baseline" not in state:
        state["working_baseline"] = copy.deepcopy(state.get("baseline", {}))

    # Ensure required keys exist
    for k in ("features", "run_dir", "test_tag", "test_dir", "metrics", "run_id"):
        state.setdefault("baseline", {}).setdefault(k, None if k != "features" else {"ck_cols": [], "cu_cols": [], "static_cols": []})
        state.setdefault("working_baseline", {}).setdefault(k, None if k != "features" else {"ck_cols": [], "cu_cols": [], "static_cols": []})

    state["version"] = int(state.get("version", 1))

    return state

#########################
### subprocess runner ###
#########################

def _maybe_run_pre_steps(backend, cfg, run_id, feat_t, runs_root, project_root, super_dir, log):
    if hasattr(backend, "make_pre_train_cmds"):
        cmds = backend.make_pre_train_cmds(cfg, run_id, feat_t, runs_root)
        for i, cmd in enumerate(cmds):
            log.info("[pre] step %d: %s", i+1, " ".join(str(x) for x in cmd))
            rc = _run_cmd(cmd, cwd=project_root, log_path=super_dir / "logs" / f"{run_id}_pre_{i+1}.log")
            if rc != 0:
                raise RuntimeError(f"Pre-train step {i+1} failed with rc={rc}")
            
def _train_or_finetune(backend, cfg, run_id, feat_t, runs_root, project_root, super_dir, log,
                       abs_train: str = "", abs_test: str = "") -> int:
    """
    Prefer finetune if backend provides it and experiment != zero_shot.
    Otherwise fall back to legacy make_train_cmd (NF etc.).
    """
    exp = str(getattr(cfg, "experiment", "zero_shot")).lower()
    if exp == "zero_shot":
        log.info("[train] zero-shot: skipping train/finetune for %s", run_id)
        return 0

    # If backend implements finetune:
    if hasattr(backend, "make_finetune_cmd"):
        cmd = backend.make_finetune_cmd(cfg, run_id, feat_t, runs_root, abs_train, abs_test)
    else:
        # Legacy train path (e.g., NFBackend)
        if not hasattr(backend, "make_train_cmd"):
            raise NotImplementedError("Backend defines neither make_finetune_cmd nor make_train_cmd.")
        cmd = backend.make_train_cmd(cfg, run_id, feat_t, runs_root, abs_train, abs_test)

    log.info("[train] %s", " ".join(str(x) for x in cmd))
    return _run_cmd(cmd, cwd=project_root, log_path=super_dir / "logs" / f"{run_id}_train.log")

def _make_test_cmd_safely(backend, cfg, train_run_dir, test_tag, feat_t):
    fn = getattr(backend, "make_test_cmd")
    sig = inspect.signature(fn)
    if len(sig.parameters) >= 4:
        # backends like MoiraiBackend: (cfg, run_dir, tag, feat_t)
        return fn(cfg, train_run_dir, test_tag, feat_t=feat_t)
    # legacy backends (NF, NHITS+QRA): (cfg, run_dir, tag)
    return fn(cfg, train_run_dir, test_tag)

def _run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> int:
    """
    Run `cmd` in `cwd`, capture stdout+stderr to `log_path`.
    No console streaming, just file logging. Returns process returncode.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Make Python subprocess unbuffered so logs are written promptly
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # If call like ["python", "-m", ...], switch to -u for unbuffered
    if len(cmd) >= 2 and cmd[0] == "python" and cmd[1] == "-m":
        cmd = ["python", "-u"] + cmd[1:]

    with open(log_path, "a", buffering=1) as lf:
        start = time.time()
        # Header for reproducibility
        lf.write("$ " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n")
        lf.write(f"# CWD={cwd.resolve()}\n")
        lf.write(f"# START={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"# PYTHON={shlex.quote(os.sys.executable)}\n")
        lf.flush()

        try:
            # Run and send BOTH stdout+stderr to the same log file
            rc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,          # we return rc ourselves
                env=env,
            ).returncode
        except subprocess.TimeoutExpired as e:
            lf.write(f"[timeout] {e}\n")
            lf.flush()
            return 124
        except Exception as e:
            lf.write(f"[exception] {type(e).__name__}: {e}\n")
            lf.flush()
            return 1

        duration = time.time() - start
        lf.write(f"[exit {rc}] duration={duration:.2f}s\n")
        lf.flush()
        return rc
    

#######################
### feature helpers ###
#######################

def _make_run_id(ck_cols: List[str], cu_cols: List[str], suffix: str | None = None) -> str:
    h = feature_hash(ck_cols, cu_cols)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}-f{h}{('-' + suffix) if suffix else ''}"


def _candidate_to_features(baseline, cand):
    ck = list(baseline["features"]["ck_cols"])
    cu = list(baseline["features"]["cu_cols"])
    static_cols = list(baseline["features"].get("static_cols", []))
    if cand.type == "ck":
        if cand.name not in ck:
            ck.append(cand.name)
    else:
        if cand.name not in cu:
            cu.append(cand.name)
    return {"ck_cols": ck, "cu_cols": cu, "static_cols": static_cols}

def _queue_item_to_features(baseline, item: dict) -> dict:
    ck = list(baseline["features"]["ck_cols"])
    cu = list(baseline["features"]["cu_cols"])
    static_cols = list(baseline["features"].get("static_cols", []))

    def add_one(name: str, typ: str):
        nonlocal ck, cu
        if typ == "ck":
            if name not in ck: ck.append(name)
        else:
            if name not in cu: cu.append(name)

    if "group" in item:
        for sub in item["group"]:
            add_one(sub["name"], sub["type"])
    else:
        add_one(item["name"], item["type"])

    return {"ck_cols": ck, "cu_cols": cu, "static_cols": static_cols}

def _item_label(item: dict) -> str:
    if "group" in item:
        names = [x["name"] for x in item["group"]]
        return "GROUP_" + "+".join(sorted(names))
    return item["name"]


def _apply_item_to_baseline_features(state, item):
    if "group" in item:
        for sub in item["group"]:
            _apply_item_to_baseline_features(state, sub)
        return
    typ, name = item["type"], item["name"]
    if typ == "ck":
        if name not in state["baseline"]["features"]["ck_cols"]:
            state["baseline"]["features"]["ck_cols"].append(name)
    else:
        if name not in state["baseline"]["features"]["cu_cols"]:
            state["baseline"]["features"]["cu_cols"].append(name)

##################
### evaluation ###
##################


def _ensure_baseline(state: Dict[str, Any], 
                     cfg: DictConfig, 
                     runs_root: Path, 
                     log, 
                     backend) -> Dict[str, Any]:
    """
    If baseline hasn't been trained/tested yet, do it now and update state.
    """
    # --- normalize baseline features for the backend (ck, cu, static) ---
    base = state["baseline"]["features"]
    norm_feat = backend.normalize_features_for_backend(
        {
            "ck_cols": list(base.get("ck_cols", [])),
            "cu_cols": list(base.get("cu_cols", [])),
            "static_cols": list(base.get("static_cols", [])),
        },
        cfg,
    )

    # --- ensure baseline run_id ---
    run_id = state["baseline"].get("run_id")
    if not run_id:
        run_id = _make_run_id(norm_feat["ck_cols"], norm_feat["cu_cols"], suffix="BASE")
        state["baseline"]["run_id"] = run_id
        _atomic_write_json(state, Path.cwd() / "state.json")

    # --- stable pointers ---
    train_run_dir = Path(state["baseline"].get("run_dir") or (runs_root / run_id))
    test_tag = state["baseline"].get("test_tag") or "BASELINE"
    test_dir = backend.test_dir(train_run_dir, test_tag)

    changed = False
    if state["baseline"].get("run_dir") != str(train_run_dir):
        state["baseline"]["run_dir"] = str(train_run_dir); changed = True
    if state["baseline"].get("test_tag") != test_tag:
        state["baseline"]["test_tag"] = test_tag; changed = True
    if state["baseline"].get("test_dir") != str(test_dir):
        state["baseline"]["test_dir"] = str(test_dir); changed = True
    if changed:
        _atomic_write_json(state, Path.cwd() / "state.json")

    super_dir = Path.cwd()
    project_root = Path(to_absolute_path("."))

    # --- TRAIN ---
    if not _is_train_complete_backend(train_run_dir, backend):

        log.info("[baseline] training %s", run_id)
        _maybe_run_pre_steps(backend, cfg, run_id, norm_feat, runs_root, project_root, super_dir, log)

        # TRAIN or FINETUNE (depending on backend + experiment)
        rc = _train_or_finetune(
            backend, 
            cfg, 
            run_id, 
            norm_feat, 
            runs_root, 
            project_root, 
            super_dir, 
            log,
            abs_train=str(Path(to_absolute_path(cfg.data.train_csv_path))),abs_test=str(Path(to_absolute_path(cfg.data.test_csv_path))))
        if rc != 0 or not _is_train_complete_backend(train_run_dir, backend):
            _atomic_write_json(state, Path.cwd() / "state.json")
            raise RuntimeError(f"Baseline training/finetune failed (rc={rc}) or not complete in {train_run_dir}")
    else:
        log.info("[baseline] training already complete for %s; skipping.", run_id)

    # --- TEST ---
    if not _is_train_complete_backend(train_run_dir, backend):
        _atomic_write_json(state, Path.cwd() / "state.json")
        raise RuntimeError(f"[baseline] Refusing to test: training not complete in {train_run_dir}")

    if not _is_test_complete_backend(train_run_dir, test_tag, backend):
        test_cmd = _make_test_cmd_safely(backend, cfg, train_run_dir, test_tag, norm_feat)
        log.info("[baseline] testing %s", run_id)
        rc = _run_cmd(test_cmd, cwd=project_root, log_path=super_dir / "logs" / f"{run_id}_test.log")
        if rc != 0 or not _is_test_complete_backend(train_run_dir, test_tag, backend):
            _atomic_write_json(state, Path.cwd() / "state.json")
            raise RuntimeError(f"Baseline testing failed (rc={rc})")
    else:
        log.info("[baseline] testing already complete for %s; skipping.", run_id)

    # --- METRICS + PERSIST ---
    metrics_path = backend.metrics_json_path(train_run_dir, test_tag)
    metrics = _atomic_read_json(metrics_path) or {}
    state["baseline"]["metrics"]  = metrics
    state["baseline"]["run_dir"]  = str(train_run_dir)
    state["baseline"]["test_dir"] = str(test_dir)
    _atomic_write_json(state, Path.cwd() / "state.json")

    if not state.get("working_baseline", {}).get("run_dir"):
        state["working_baseline"] = copy.deepcopy(state["baseline"])
        _atomic_write_json(state, Path.cwd() / "state.json")
    return state

def _read_any_table(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file: {p}")

def _filter_nonoverlapping_origins(df: pd.DataFrame, anchor_hour: int | None = None,
                                   step: int | None = None) -> pd.DataFrame:
    """Keep only origins at a specific hour, or every k-th origin if step>1."""
    out = df.copy()
    out["origin_ds"] = pd.to_datetime(out["origin_ds"])
    if anchor_hour is not None:
        out = out[out["origin_ds"].dt.hour == int(anchor_hour)]
    elif step is not None and step > 1:
        out = out.sort_values(["unique_id","origin_ds"])
        out = (out.assign(_r=out.groupby("unique_id").cumcount())
                 .query("_r % @step == 0").drop(columns="_r"))
    return out


def _pair_energy_from_per_origin_paths(base_po_p: Path, cand_po_p: Path, log,
                                       anchor_hour: int | None = None, step: int | None = None) -> pd.DataFrame:
    b = _read_any_table(base_po_p).rename(columns={"es_mean":"base_es"})
    c = _read_any_table(cand_po_p).rename(columns={"es_mean":"cand_es"})
    for df in (b, c):
        if "origin_ds" not in df.columns:
            raise ValueError("Expected 'origin_ds' column in ES per-origin file.")
        df["origin_ds"] = pd.to_datetime(df["origin_ds"])

        if b["unique_id"].dtype != c["unique_id"].dtype:
            df["unique_id"] = df["unique_id"].astype(str)

    # optional filtering
    if anchor_hour is not None or (step is not None and step > 1):
        b = _filter_nonoverlapping_origins(b, anchor_hour=anchor_hour, step=step)
        c = _filter_nonoverlapping_origins(c, anchor_hour=anchor_hour, step=step)

    df = (b.merge(c, on=["unique_id","origin_ds"], how="inner")
            .dropna(subset=["base_es","cand_es"])
            .assign(loss_diff=lambda x: x.cand_es - x.base_es)
            .sort_values(["unique_id","origin_ds"], kind="mergesort")
            .reset_index(drop=True))
    return df


def _pair_crps_from_per_origin_paths(base_po_p: Path, cand_po_p: Path,
                                     anchor_hour: int | None = None, step: int | None = None) -> pd.DataFrame:
    b = _read_any_table(base_po_p).rename(columns={"crps_mean":"base_crps"})
    c = _read_any_table(cand_po_p).rename(columns={"crps_mean":"cand_crps"})
    for df in (b, c):
        if "origin_ds" not in df.columns:
            raise ValueError("Expected 'origin_ds' column in CRPS per-origin file.")
        df["origin_ds"] = pd.to_datetime(df["origin_ds"])

        if b["unique_id"].dtype != c["unique_id"].dtype:
            df["unique_id"] = df["unique_id"].astype(str)

    if anchor_hour is not None or (step is not None and step > 1):
        b = _filter_nonoverlapping_origins(b, anchor_hour=anchor_hour, step=step)
        c = _filter_nonoverlapping_origins(c, anchor_hour=anchor_hour, step=step)

    df = (b.merge(c, on=["unique_id","origin_ds"], how="inner")
            .dropna(subset=["base_crps","cand_crps"])
            .assign(loss_diff=lambda x: x.cand_crps - x.base_crps)
            .sort_values(["unique_id","origin_ds"], kind="mergesort")
            .reset_index(drop=True))
    return df


def _nw_lag(T: int, nw_lag_rule: str = "T**0.25") -> int:
    nw_lag_rule = str(nw_lag_rule).strip()
    if nw_lag_rule == "T**0.25":
        return max(0, floor(T ** 0.25))
    if nw_lag_rule == "T**(1/3)":
        return max(0, floor(T ** (1/3)))
    # fallback: integer or 0
    try:
        return max(0, int(nw_lag_rule))
    except Exception:
        return max(0, floor(T ** 0.25))

def _dm_newey_west(d: np.ndarray, lag: int | None = None, nw_lag_rule: str = "T**0.25") -> dict:
    """
    Diebold–Mariano via intercept-only OLS with HAC (Newey–West).
    d must be a 1D array of loss differentials ordered in time.
    """
    d = np.asarray(d, dtype=float).ravel()
    T = d.size
    X = np.ones((T, 1))                     # intercept only
    model = sm.OLS(d, X, hasconst=True)
    res = model.fit()

    if lag is None:
        lag = _nw_lag(T, nw_lag_rule)  

    # HAC covariance for parameters (1x1 here)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), \
        warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\..*")
        res = model.fit()
        try:
            V = cov_hac(res, nlags=int(lag))  # Bartlett by default
        except Exception:
            V = np.array([[np.nan]])

    alpha_hat = float(res.params[0])
    se_alpha  = float(np.sqrt(V[0, 0]))
    if not np.isfinite(se_alpha) or se_alpha == 0.0:
        tstat = np.nan
        pval  = np.nan
    else:
        tstat = alpha_hat / se_alpha
        # two-sided p-value
        pval  = 2 * (1 - student_t.cdf(abs(tstat), df=max(T - 1, 1)))
    
    return {
        "T": T,
        "lag": int(lag),
        "mean": float(alpha_hat),    
        "stat": float(tstat),
        "p_value": float(pval),
    }
def _evaluate_and_decide(
    state: dict,
    cfg: DictConfig,
    log,                                   # <- positional
    *,                                     
    backend,
    cand_train_run_dir: Path,
    cand_test_tag: str,
    anchor_hour: int | None = None,
    origin_step: int = 1,
) -> dict:
    baseline_run_dir = Path(state["working_baseline"]["run_dir"])
    baseline_test_tag = state["working_baseline"]["test_tag"]

    metric = str(getattr(cfg.decision, "metric", "crps_mean")).lower()

    log.info(
    "frozen_baseline run_dir=%s tag=%s | working_baseline run_dir=%s tag=%s | metric=%s",
    state["baseline"].get("run_dir"),
    state["baseline"].get("test_tag"),
    state["working_baseline"].get("run_dir"),
    state["working_baseline"].get("test_tag"),
    metric,
    )

    # allow overrides via cfg, but args win if passed
    ah = getattr(cfg.decision, "anchor_hour", None) if anchor_hour is None else anchor_hour
    anchor_hour = None if ah is None else int(ah)
    origin_step = int(getattr(cfg.decision, "origin_step", origin_step))

    cand_run_dir = Path(cand_train_run_dir)
    cand_tag     = cand_test_tag

    base_paths = backend.pred_paths_for_metric(baseline_run_dir, baseline_test_tag, metric)
    cand_paths = backend.pred_paths_for_metric(cand_run_dir, cand_tag, metric)

    if metric == "es_mean":
        b_po = base_paths.get("es_per_origin"); 
        c_po = cand_paths.get("es_per_origin")
        if not b_po:
            raise FileNotFoundError("Missing ES file from baseline/ current state.")
        if not c_po:
            raise FileNotFoundError("Missing ES file from candidate.")
            
        df = _pair_energy_from_per_origin_paths(
            b_po, c_po, log,
            anchor_hour=anchor_hour,
            step=None if anchor_hour is not None else origin_step,
        )
    elif metric == "crps_mean":
        b_po = base_paths.get("crps_per_origin"); 
        c_po = cand_paths.get("crps_per_origin")
        if not b_po:
            raise FileNotFoundError(f"Missing CRPS file from baseline/ current state. Currently looking in directory: {b_po}")
        if not c_po:
            raise FileNotFoundError(f"Missing CRPS file from candidate. Currently looking in directory: {c_po}")
        df = _pair_crps_from_per_origin_paths(
            b_po, c_po,
            anchor_hour=anchor_hour,
            step=None if anchor_hour is not None else origin_step,
        )
    else:
        raise ValueError(f"Unsupported decision.metric='{metric}'")

    T   = len(df)
    lag = _nw_lag(T, state.get("nw_lag_rule", cfg.decision.nw_lag_rule))
    dm  = _dm_newey_west(df["loss_diff"].to_numpy(), lag=lag,
                         nw_lag_rule=state.get("nw_lag_rule", cfg.decision.nw_lag_rule))

    delta = {
        "mean":   float(df["loss_diff"].mean()) if T else float("nan"),
        "median": float(df["loss_diff"].median()) if T else float("nan"),
        "T":      int(T),
        "metric": metric,
    }
    adopt = (dm["p_value"] < float(state["alpha"])) and (delta["mean"] < 0.0)
    log.info("[eval] metric=%s | N_origins=%d | NW lag=%d | mean Δ=%.5f | p=%.4g",
             metric, T, lag, delta["mean"], dm["p_value"])
    return {"dm": dm, "delta": delta, "adopt": adopt}


def _append_summary_row(
    tid: str,
    eval_res: dict,
    summary_path: Path,
    *,
    base_metrics: dict | None = None,
    cand_metrics: dict | None = None,
    frozen_base_metrics: dict | None = None,  
) -> None:
    row = {
        "trial_id": tid,
        "metric": eval_res["delta"]["metric"],
        "p_value": eval_res["dm"]["p_value"],
        "dm_stat": eval_res["dm"]["stat"],
        "T": eval_res["dm"]["T"],
        "lag": eval_res["dm"]["lag"],
        "mean_delta": eval_res["delta"]["mean"],
        "median_delta": eval_res["delta"]["median"],
        "adopt": eval_res["adopt"],
    }

    if frozen_base_metrics is not None:
        row.update({f"frozen_base__{k}": v for k, v in _flatten_metrics(frozen_base_metrics).items()})
    if base_metrics is not None:
        row.update({f"base__{k}": v for k, v in _flatten_metrics(base_metrics).items()})
    if cand_metrics is not None:
        row.update({f"cand__{k}": v for k, v in _flatten_metrics(cand_metrics).items()})

    pd.DataFrame([row]).to_csv(
        summary_path,
        mode="a",
        header=not summary_path.exists(),
        index=False,
    )

def _flatten_metrics(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict-like metrics into one level:
      {"a": {"b": 1}} -> {"a.b": 1}
    Lists get indexed:
      {"x": [10, 20]} -> {"x.0": 10, "x.1": 20}
    Non-JSON scalars pass through.
    """
    out: Dict[str, Any] = {}

    def rec(x: Any, p: str):
        if isinstance(x, dict):
            for k, v in x.items():
                rec(v, f"{p}{sep}{k}" if p else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, f"{p}{sep}{i}" if p else str(i))
        else:
            out[p] = x

    rec(obj, prefix)
    return out



def _trial_phase_from_fs(trial: dict, backend) -> str:
    run_dir = Path(trial["train_run_dir"])
    if not _is_train_complete_backend(run_dir, backend):
        return "needs_train"
    if not backend.test_is_complete(run_dir, trial["test_tag"]):
        return "needs_test"
    if trial.get("dm") is None or trial.get("delta") is None:
        return "needs_eval"
    return "done"

def _resume_or_start_next(super_dir: Path, 
                          project_root: Path, 
                          state_path: Path, 
                          state: dict, cfg: DictConfig, 
                          runs_root: Path, 
                          log, 
                          backend):
    while True:
        # 1) Is there an unfinished trial to resume?
        unfinished_id = None
        for tid, tr in state["trials"].items():
            phase = _trial_phase_from_fs(tr, backend)
            if phase != "done":
                unfinished_id = tid
                break

        if unfinished_id:
            tid = unfinished_id
            tr = state["trials"][tid]

            train_run_dir  = Path(tr["train_run_dir"])
            test_dir = backend.test_dir(train_run_dir, tr["test_tag"])

            phase = _trial_phase_from_fs(tr, backend)
            log.info("[resume] trial %s phase=%s", tid, phase)

            if phase == "needs_train":
                feat_ck = ast.literal_eval(tr.get("features_ck", "[]")) if "features_ck" in tr else None
                feat_cu = ast.literal_eval(tr.get("features_cu", "[]")) if "features_cu" in tr else None

                if feat_ck is None or feat_cu is None:
                    # backward-compat: rebuild from stored trial item
                    item = parse_trial_item(tr.get("item"))
                    feat_t = apply_item_to_features(state["working_baseline"]["features"], item)
                else:
                    static_cols = list(state["baseline"]["features"].get("static_cols", []))
                    feat_t = {"ck_cols": feat_ck, "cu_cols": feat_cu, "static_cols": static_cols}

                feat_t = backend.normalize_features_for_backend(feat_t, cfg)

                run_id = train_run_dir.name
                _maybe_run_pre_steps(backend, cfg, run_id, feat_t, runs_root, project_root, super_dir, log)
                rc = _train_or_finetune(backend, cfg, run_id, feat_t, runs_root, project_root, super_dir, log,
                        abs_train=str(Path(to_absolute_path(cfg.data.train_csv_path))),
                        abs_test=str(Path(to_absolute_path(cfg.data.test_csv_path))))
                if rc != 0 or not _is_train_complete_backend(train_run_dir, backend):
                    tr["status"] = "failed_train"; _atomic_write_json(state, state_path)
                    raise RuntimeError(f"Training failed or produced no checkpoint in {train_run_dir}. Exit ({rc})")

                tr["train_run_dir"] = str(train_run_dir)
                _atomic_write_json(state, state_path)

            if phase in ("needs_test", "needs_eval"):
                feat_ck = ast.literal_eval(tr.get("features_ck", "[]"))
                feat_cu = ast.literal_eval(tr.get("features_cu", "[]"))
                static_cols = list(state["baseline"]["features"].get("static_cols", []))
                feat_t = backend.normalize_features_for_backend(
                    {"ck_cols": feat_ck, "cu_cols": feat_cu, "static_cols": static_cols}, cfg
                )
                # ensure test (idempotent)
                if not _is_train_complete_backend(train_run_dir, backend):
                    tr["status"] = "failed_train"; _atomic_write_json(state, state_path)
                    raise RuntimeError(f"Refusing to test: no checkpoint in {train_run_dir}")
                
                if not backend.test_is_complete(train_run_dir, tr["test_tag"]):
                    test_cmd = _make_test_cmd_safely(backend, cfg, train_run_dir, tr["test_tag"], feat_t)
                    rc = _run_cmd(test_cmd, cwd=project_root, log_path=super_dir / "logs" / f"{train_run_dir.name}_test.log")
                    if rc != 0 or not backend.test_is_complete(train_run_dir, tr["test_tag"]):
                        tr["status"] = "failed_test"; _atomic_write_json(state, state_path)
                        raise RuntimeError(f"Testing failed. Exit ({rc})")

                # evaluate if not yet done
                if tr.get("dm") is None or tr.get("delta") is None:
                    metrics_p = backend.metrics_json_path(train_run_dir, tr["test_tag"])
                    cand_metrics = _atomic_read_json(metrics_p) or {}
                    tr["candidate_metrics"] = cand_metrics
                    tr["test_dir"] = str(test_dir)
                    tr["status"]   = "tested"
                    eval_res = _evaluate_and_decide(
                        state, cfg, log,
                        backend=backend,
                        cand_train_run_dir=train_run_dir,
                        cand_test_tag=tr["test_tag"],
                    )
                    state = _finalize_trial(state, tid, train_run_dir, test_dir, eval_res, log)
                    _atomic_write_json(state, state_path)

            # loop back to see if more unfinished work remains
            continue

        # 2) No unfinished trial — start next from queue
        if not state.get("candidates_queue"):
            log.info("Queue empty. All trials done.")
            break

        entry_raw = state["candidates_queue"][0]
        item_dict, policy = split_queue_entry(entry_raw)
        item = parse_trial_item(item_dict)

        feat_t = apply_item_to_features(state["working_baseline"]["features"], item)
        feat_t = backend.normalize_features_for_backend(feat_t, cfg)

        round_idx = int(state["current_round"])
        run_id = _make_run_id(feat_t["ck_cols"], feat_t["cu_cols"], suffix=f"R{round_idx}")
        train_run_dir = runs_root / run_id
        test_tag = f"{cfg.test.tag_prefix}"
        tid = f"R{round_idx}_{trial_item_label(item)}"

        state["trials"][tid] = {
            "item": trial_item_to_dict(item),     
            "status": "running",
            "train_run_dir": str(train_run_dir),
            "test_tag": test_tag,
            "test_dir": None,
            "delta": None,
            "dm": None,
            "policy": policy.to_dict(),
            "decision": None,
            "features_ck": str(feat_t["ck_cols"]),
            "features_cu": str(feat_t["cu_cols"]),
        }
        _atomic_write_json(state, state_path)

        # ---- TRAIN ----

        log.info("[trial %s] training %s", tid, run_id)
        _maybe_run_pre_steps(backend, cfg, run_id, feat_t, runs_root, project_root, super_dir, log)
        rc = _train_or_finetune(backend, cfg, run_id, feat_t, runs_root, project_root, super_dir, log,
                        abs_train=str(Path(to_absolute_path(cfg.data.train_csv_path))),
                        abs_test=str(Path(to_absolute_path(cfg.data.test_csv_path))))
        if rc != 0 or not _is_train_complete_backend(train_run_dir, backend):
            state["trials"][tid]["status"] = "failed_train"; _atomic_write_json(state, state_path)
            raise RuntimeError(f"Training failed with rc={rc} or no checkpoint in {train_run_dir}")

        state["trials"][tid]["train_run_dir"] = str(train_run_dir)
        _atomic_write_json(state, state_path)

        # ---- TEST ----
        test_dir = backend.test_dir(train_run_dir, test_tag)

        # Guard: training must have produced a checkpoint
        if not _is_train_complete_backend(train_run_dir, backend):
            state["trials"][tid]["status"] = "failed_train"; _atomic_write_json(state, state_path)
            raise RuntimeError(f"Refusing to test: no checkpoint in {train_run_dir}")
        
        if not backend.test_is_complete(train_run_dir, test_tag):
            test_cmd = _make_test_cmd_safely(backend, cfg, train_run_dir, test_tag, feat_t)
            log.info("[trial %s] testing %s", tid, run_id)
            rc = _run_cmd(test_cmd, cwd=project_root, log_path=super_dir / "logs" / f"{run_id}_test.log")
            if rc != 0 or not backend.test_is_complete(train_run_dir, test_tag):
                state["trials"][tid]["status"] = "failed_test"; _atomic_write_json(state, state_path)
                raise RuntimeError("Testing failed")
            
        # ---- EVAL ----
        state["trials"][tid]["test_dir"] = str(test_dir)
        state["trials"][tid]["status"] = "tested"
        metrics_p = backend.metrics_json_path(train_run_dir, test_tag)
        cand_metrics = _atomic_read_json(metrics_p) or {}
        state["trials"][tid]["candidate_metrics"] = cand_metrics

        eval_res = _evaluate_and_decide(
            state, cfg, log,
            backend=backend,
            cand_train_run_dir=train_run_dir,
            cand_test_tag=test_tag,
        )
        state = _finalize_trial(state, tid, train_run_dir, test_dir, eval_res, log)
        _atomic_write_json(state, state_path)
        log.info("[trial %s] decision persisted. Next round=%d, queue_len=%d",
                 tid, state["current_round"], len(state["candidates_queue"]))


def _finalize_trial(state: dict, trial_id: str, cand_run_dir: Path, cand_test_dir: Path, eval_res: dict, log) -> dict:
    tr = state["trials"][trial_id]
    tr["dm"] = eval_res["dm"]
    tr["delta"] = eval_res["delta"]

    # --- policy (legacy-safe) ---
    from utils.super_runner.trial_items import TrialPolicy  # local import to avoid cycles
    pol = TrialPolicy.from_obj(tr.get("policy", None))

    dm_adopt = bool(eval_res["adopt"])  # DM says "improves significantly"
    if pol.adopt == "adopt_if_improves":
        final_adopt = dm_adopt
    elif pol.adopt == "adopt_always":
        final_adopt = True
    elif pol.adopt == "adopt_never":
        final_adopt = False
    elif pol.adopt == "adopt_if_worse":
        final_adopt = (not dm_adopt)
    else:
        # safety fallback
        final_adopt = dm_adopt

    tr["status"] = "accepted" if final_adopt else "rejected"

    tr["decision"] = {
        "dm_adopt": dm_adopt,
        "final_adopt": final_adopt,
        "policy": pol.to_dict(),
        "metric": eval_res["delta"]["metric"],
        "p_value": float(eval_res["dm"]["p_value"]),
        "mean_delta": float(eval_res["delta"]["mean"]),
        "criteria": {
            "alpha": float(state["alpha"]),
            "rule": "dm_adopt iff p < alpha and mean_delta < 0; final_adopt per policy",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_csv = Path.cwd() / "summary.csv"
    if not tr.get("_summary_written", False):
        _append_summary_row(trial_id, eval_res, summary_csv)
        tr["_summary_written"] = True

    metric = eval_res["delta"].get("metric", "loss")

    # reconstruct TrialItem from stored dict
    item = parse_trial_item(tr["item"])

    # --- apply baseline update according to FINAL decision ---
    if final_adopt:
        log.info(
            "[trial %s] ADOPTED_BY_POLICY (dm_adopt=%s, p=%.4g, mean diff_%s=%.4f, policy=%s)",
            trial_id, dm_adopt, tr["dm"]["p_value"], metric, tr["delta"]["mean"], pol.adopt
        )

        state["working_baseline"]["features"] = apply_item_to_features(state["working_baseline"]["features"], item)
        state["working_baseline"]["run_dir"]  = str(cand_run_dir)
        state["working_baseline"]["test_dir"] = str(cand_test_dir)
        state["working_baseline"]["test_tag"] = Path(cand_test_dir).name
        state["working_baseline"]["metrics"]  = tr.get("candidate_metrics", {})

        state["selected_features"].append(trial_item_to_dict(item))
    else:
        log.info(
            "[trial %s] NOT_ADOPTED_BY_POLICY (dm_adopt=%s, p=%.4g, mean diff_%s=%.4f, policy=%s)",
            trial_id, dm_adopt, tr["dm"]["p_value"], metric, tr["delta"]["mean"], pol.adopt
        )

    # --- queue handling (for now only pop_after_run) ---
    if pol.queue == "pop_after_run":
        if _queue_head_matches_item(state.get("candidates_queue", []), item):
            state["candidates_queue"].pop(0)

    state["current_round"] = int(state.get("current_round", 1)) + 1
    return state