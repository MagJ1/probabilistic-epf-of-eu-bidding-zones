# src/models/nhits_qra/qra/qra_runner.py
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
from models.nhits_qra.qra.qra_diagnostics import dump_and_log_nhits_point_forecasts
import numpy as np
import pandas as pd
import joblib
from omegaconf import OmegaConf


from models.nhits_qra.qra.qra_fit import fit_qra_per_h
from models.nhits_qra.nhits.nhits_module import NHITSForecasterModule
from models.nhits_qra.datamodule import NHITSQRADataModule
from models.nhits_qra.qra.qra_prep import (
    fit_incremental_pca_per_h, 
    transform_with_pcas, 
    collect_qra_design_for_split
)
from models.nhits_qra.qra.qra_io import emit_per_origin_files
from models.nhits_qra.qra.qra_io import (
    QRAPaths, 
    save_design, 
    save_pred_bundles, 
    first_nonempty)
from models.nhits_qra.qra.qra_exact_solver import predict_quantiles_per_tau

from models.nhits_qra.qra.qra_post import pits_from_quantiles
from models.nhits_qra.qra.qra_help import inverse_per_h, count_qra_params, normalize_tau_dict_keys_to_percent
from models.nhits_qra.qra.qra_calib_iterative_solver import predict_quantiles_per_tau_pairlike
from models.nhits_qra.qra.qra_summary import pack_summary

from models.nhits_qra.qra.qra_calib_chung_iterative_solver import predict_quantiles_chung, q_sensitivity_chung

from models.nhits_qra.qra.qra_context import QRAContext
from models.nhits_qra.qra.qra_knobs import QRAKnobs
from models.nhits_qra.qra.qra_time import _build_origin_mask_anchor_stride, ensure_dtindex

from models.nhits_qra.qra.qra_plotting import (
    plot_fan_charts, 
    plot_pit_histograms, 
    get_art_dir)

from models.nhits_qra.qra.qra_post import samples_from_q, postprocess_quantiles
from models.nhits_qra.qra.qra_eval_runner import run_qra_eval

from models.nhits_qra.qra.qra_diagnostics import summarize_design, log_qra_quantile_diagnostics, summarize_coverage



#######################
### RUN TRAIN & VAL ###
#######################

def run_train_val(
    *,
    ctx: QRAContext,
    train_cfg: OmegaConf,
    module: NHITSForecasterModule,
    dm: NHITSQRADataModule,
    knobs: QRAKnobs,
    paths: Dict[str, Path],
    p: QRAPaths,
) -> Dict:
    log = ctx.log
    device = ctx.torch_device

    # ---- TRAIN design
    Xtr_per_h, ytr_per_h, meta_tr = collect_qra_design_for_split(dm, module, "train", knobs)

    dump_and_log_nhits_point_forecasts(
    X_per_h=Xtr_per_h,
    meta=meta_tr,
    out_dir=p.metrics_dir / "nhits_point_forecasts",
    split="train",
    log=log,
    n_points=meta_tr["S"],     # assumes these are the “points”
    preview_rows=0,
    dump=True,
    )

    # ---- PCA
    if knobs.use_pca and knobs.n_samples > 0:
        pcas, Ztr_per_h, k_train = fit_incremental_pca_per_h(
            X_per_h=Xtr_per_h,
            n_samples=knobs.n_samples,
            pca_var=knobs.pca_var,
            n_components=knobs.n_comp,
        )
        log.info("QRA[train]: PCA retained per-horizon: %s", k_train)
        log.info("QRA[train]: PCA retained summary: min=%d median=%d max=%d",
                min(k_train), int(np.median(k_train)), max(k_train))
    else:
        pcas, Ztr_per_h = [None] * len(Xtr_per_h), Xtr_per_h

    # ---- VAL design
    Xval_per_h, yval_per_h, meta_val = collect_qra_design_for_split(dm, module, "val", knobs)

    for h in [0, 1, 12, 23]:
        Xh = Xtr_per_h[h]
        if isinstance(Xh, np.ndarray) and Xh.size:
            log.info(f"[DBG][train][X][h={h}] {summarize_design(Xh)}")

    Zval_per_h = (
        transform_with_pcas(Xval_per_h, pcas, knobs.n_samples)
        if (knobs.use_pca and knobs.n_samples > 0)
        else Xval_per_h
    )

    for h in [0, 1, 12, 23]:
        Zh = Ztr_per_h[h]
        if isinstance(Zh, np.ndarray) and Zh.size:
            log.info(f"[DBG][train][Z][h={h}] {summarize_design(Zh)}")


    dump_and_log_nhits_point_forecasts(
    X_per_h=Xval_per_h,
    meta=meta_val,
    out_dir=p.metrics_dir / "nhits_point_forecasts",
    split="val",
    log=log,
    n_points=meta_val["S"],
    preview_rows=0,
    dump=True,
    )

    # ---- persist meta/pcas/designs
    joblib.dump(
        dict(
            n_samples=knobs.n_samples,
            add_mean_std=knobs.add_mean_std,
            use_pca=knobs.use_pca,
            pca_var=knobs.pca_var,
            n_components=knobs.n_comp,
            subsample_stride=knobs.subsample_stride,
            taus_train=knobs.taus_train,
            lambda_grid=knobs.lambda_grid,
            solver_loss=knobs.solver_loss,
            mix_kappa=knobs.mix_kappa,
        ),
        p.out_qra / "qra_meta.pkl",
    )
    joblib.dump(pcas, p.out_qra / "pcas.pkl")
    save_design(p.design_dir, Ztr_per_h, ytr_per_h, Zval_per_h, yval_per_h)

    # ---- fit QRA per horizon
    qra_models, Qval_per_h = fit_qra_per_h(
        Ztr_per_h,
        ytr_per_h,
        Zval_per_h,
        yval_per_h,
        taus=knobs.taus_train,
        lambda_grid=knobs.lambda_grid,
        knobs=knobs,
        device=device,
        cfg=train_cfg,
    )

    if str(getattr(knobs, "solver_loss", "")) in ("iterative_calib_chung", "chung"):
        for h in [0, 1, 12, 23]:
            Zdbg = Zval_per_h[h]
            mh = qra_models[h]
            if isinstance(Zdbg, np.ndarray) and Zdbg.size and isinstance(mh, dict) and ("state_dict" in mh):
                Zsmall = Zdbg[:512]
                sens = q_sensitivity_chung(
                    pack=mh,
                    Z=Zsmall,
                    q_grid=[0.1, 0.3, 0.5, 0.7, 0.9],
                    device=str(device),
                )
                log.info(f"[DBG][val][q-sens][h={h}] {sens}")

    # ---- postprocess
    Qval_fix, taus_tgt = postprocess_quantiles(Qval_per_h, knobs.taus_train, knobs, train_cfg)

    # ---- save preds
    save_pred_bundles(
        prefix_raw=p.pred_raw_dir,
        prefix_fix=p.pred_fix_dir,
        Q_raw_per_h=Qval_per_h,
        Q_fix_per_h=Qval_fix,
        idx_df=None,
        taus=taus_tgt,
    )

    scaler = ctx.target_scaler if ctx.target_scaler is not None else dm.train_dataset.get_target_scaler()
    Q_fix_real = inverse_per_h(Qval_fix, scaler)
    y_val_real = inverse_per_h(yval_per_h, scaler)

    hs_dbg = [0, 1, 12, 23]
    log_qra_quantile_diagnostics(
        log=log,
        split="val",
        hs=hs_dbg,
        Q_raw_per_h=Qval_per_h,
        taus_raw=list(knobs.taus_train),
        Q_fix_per_h=Qval_fix,
        taus_fix=list(taus_tgt),
        y_scaled_per_h=yval_per_h,
        Q_scaled_for_summary_per_h=Qval_fix,
        y_real_per_h=y_val_real,
        Q_real_per_h=Q_fix_real,
        collapse_eps=1e-6,
        tag="DBG",
    )


    # ---- metrics on VAL

    eval_cfg = OmegaConf.select(train_cfg, "train_qra.qra_eval", throw_on_missing=True)

    eval_res = run_qra_eval(
        eval_cfg=eval_cfg,
        metrics_dir=p.metrics_dir,
        log=log,
        device=device,
        y_real_per_h=y_val_real,
        Q_fix_real_per_h=Q_fix_real,
        taus_tgt=taus_tgt,
        tb_dir=ctx.dirs["logs_tb"]
        
    )

    # For pack_summary / logs:
    mean_crps = eval_res.mean_crps
    mean_es   = eval_res.mean_es
    mean_ece  = eval_res.mean_ece
    berk      = eval_res.berk or {"per_h": [], "fisher": {}, "hs_eff": []}

    per_h_results = berk.get("per_h", [])
    fisher_res = berk.get("fisher", {})

    # ---- checkpoint qra models
    p.qra_ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(qra_models, p.qra_ckpt_dir / "qra_models.pkl")
    summary_params = count_qra_params(p.qra_ckpt_dir / "qra_models.pkl", nz_tol=0.0)

    # ---- N/H_eff bookkeeping
    N = 0
    for _y in yval_per_h:
        if isinstance(_y, np.ndarray) and _y.size:
            N = int(len(_y))
            break
    hs_eff = [h for h, q in enumerate(Qval_fix) if isinstance(q, np.ndarray) and q.size > 0]
    H_eff = len(hs_eff)

    anchor_hour = None
    origin_stride = int(knobs.subsample_stride)
    taus_train_used = list(knobs.taus_train)

    log.info(
        "QRA[%s]: N=%d H_eff=%d | stride=%d",
        ctx.split,
        int(N),
        int(H_eff),
        int(origin_stride),
    )

    summary = pack_summary(
        split=ctx.split,
        device=device,
        Qte_per_h=Qval_per_h,
        mean_crps=mean_crps,
        mean_es=mean_es,
        mean_ece=mean_ece,
        berk={"per_h": per_h_results, "fisher": fisher_res},
        taus_train_used=taus_train_used,
        taus_tgt=taus_tgt,
        summary_params=summary_params,
        N=N,
        anchor_hour=anchor_hour,
        origin_stride=origin_stride,
    )
    return summary

################
### RUN TEST ###
################

def run_test(
    *,
    ctx: QRAContext,
    train_cfg: OmegaConf,
    module: NHITSForecasterModule,
    dm: NHITSQRADataModule,
    knobs: QRAKnobs,
    paths: Dict[str, Path],
    p: QRAPaths,
    test_cfg,
) -> Dict:
    log = ctx.log
    device = ctx.torch_device

    assert ctx.load_from_train_run is not None, "load_from_train_run must be set for split='test'"
    train_run = Path(ctx.load_from_train_run)

    # ---- load artifacts
    pcas = joblib.load(train_run / "data" / "qra" / "pcas.pkl")
    qra_meta = joblib.load(train_run / "data" / "qra" / "qra_meta.pkl")
    qra_models: List[Dict] = joblib.load(train_run / "train" / "checkpoints" / "qra" / "qra_models.pkl")
    solver_loss = str(qra_meta.get("solver_loss", "iterative_pinball"))


    taus_train_used = list(qra_meta.get("taus_train", []))
    if not taus_train_used:
        first = first_nonempty(qra_models)
        if first is None:
            raise RuntimeError("Loaded QRA models are empty; cannot infer taus.")
        taus_train_used = sorted([float(k) for k in first.keys()])

    knobs_test = replace(
        knobs,
        n_samples=int(qra_meta.get("n_samples", knobs.n_samples)),
        add_mean_std=int(qra_meta.get("add_mean_std", knobs.add_mean_std)),
        subsample_stride=1,
    )

    # ---- collect design (unfiltered)
    Xte_per_h, yte_per_h, meta_te = collect_qra_design_for_split(dm, module, "test", knobs_test)

    uid_all = np.asarray(meta_te["unique_id"])
    ods_all = ensure_dtindex(np.asarray(meta_te["origin_ds"]))

    # ---- filter (single pass)
    anchor_hour = getattr(train_cfg.data, "origin_anchor_hour_test", None)
    origin_stride = int(getattr(train_cfg.data, "origin_stride_test", 24))
    mask = _build_origin_mask_anchor_stride(uid_all, ods_all, anchor_hour, origin_stride)
    N0, N = len(uid_all), int(mask.sum())
    log.info("QRA[test]: anchor_hour=%s origin_stride=%d | kept %d/%d origins", str(anchor_hour), origin_stride, N, N0)

    uid = uid_all[mask]
    ods = ods_all[mask]
    Xte_per_h = [X[mask] if (isinstance(X, np.ndarray) and X.size) else X for X in Xte_per_h]
    yte_per_h = [y[mask] if (isinstance(y, np.ndarray) and y.size) else y for y in yte_per_h]

    dump_and_log_nhits_point_forecasts(
    X_per_h=Xte_per_h,
    meta={"unique_id": uid, "origin_ds": ods},
    out_dir=p.metrics_dir / "nhits_point_forecasts",
    split="test",
    log=log,
    n_points=meta_te["S"],
    preview_rows=0,
    dump=True,
    )

    # ---- PCA transform (train-time PCs)
    use_pca_eff = bool(qra_meta.get("use_pca", knobs.use_pca)) and int(qra_meta.get("n_samples", knobs.n_samples)) > 0
    Zte_per_h = (
        transform_with_pcas(Xte_per_h, pcas, int(qra_meta.get("n_samples", knobs.n_samples)))
        if use_pca_eff
        else Xte_per_h
    )

    k_fixed = [0 if p is None else int(getattr(p, "n_components_", p.components_.shape[0])) for p in pcas]
    log.info("QRA[test]: PCA retained per-horizon (from train PCA): %s", k_fixed)

    # ---- predict per horizon
    Qte_per_h: List[Optional[np.ndarray]] = []
    for h, Zte in enumerate(Zte_per_h):
        if not (isinstance(Zte, np.ndarray) and Zte.size):
            Qte_per_h.append(None)
            continue
        mh = qra_models[h]
        if not mh:
            Qte_per_h.append(None)
            continue
        if solver_loss == "iterative_calib":
            Qte_per_h.append(predict_quantiles_per_tau_pairlike(Zte, mh, taus_train_used))
        elif solver_loss == "iterative_calib_chung":
            Qte_per_h.append(predict_quantiles_chung(Zte, mh, taus_train_used, device=str(device)))
        else:
            Qte_per_h.append(predict_quantiles_per_tau(Zte, mh, taus_train_used))

    # ---- postprocess to target taus
    Q_fix_per_h, taus_tgt = postprocess_quantiles(Qte_per_h, taus_train_used, knobs_test, train_cfg)

    # ---- save test predictions (+ index)
    idx_df = pd.DataFrame({"unique_id": uid, "origin_ds": ods.strftime("%Y-%m-%d %H:%M:%S")})
    save_pred_bundles(p.pred_raw_dir, p.pred_fix_dir, Qte_per_h, Q_fix_per_h, idx_df=idx_df, taus=taus_tgt)

    # ---- metrics on TEST
    scaler = ctx.target_scaler if ctx.target_scaler is not None else dm.train_dataset.get_target_scaler()
    Q_fix_real = inverse_per_h(Q_fix_per_h, scaler)
    y_test_real = inverse_per_h(yte_per_h, scaler)

    hs_dbg = [0, 1, 12, 23]
    log_qra_quantile_diagnostics(
        log=log,
        split="test",
        hs=hs_dbg,
        Q_raw_per_h=Qte_per_h,
        taus_raw=list(taus_train_used),
        Q_fix_per_h=Q_fix_per_h,
        taus_fix=list(taus_tgt),
        # scaled summary in test is optional; usually caring about real scale
        y_real_per_h=y_test_real,
        Q_real_per_h=Q_fix_real,
        collapse_eps=1e-6,
        tag="DBG",
    )



    eval_cfg = OmegaConf.select(test_cfg, "test.qra_eval", throw_on_missing=True)

    eval_res = run_qra_eval(
        eval_cfg=eval_cfg,            # source of truth for what to compute/export/log
        metrics_dir=p.metrics_dir,
        log=log,
        device=device,
        y_real_per_h=y_test_real,
        Q_fix_real_per_h=Q_fix_real,
        taus_tgt=taus_tgt,
        tb_dir=ctx.dirs["logs_tb"]
    )

    emit_per_origin_files(
        pred_dir=p.general_pred_dir, 
        uid=uid,
        origin_ds=ods,
        crps_bh=eval_res.crps_bh,      
        es_bh=eval_res.es_bh,
        write="parquet",               # or "csv"
    )

    mean_crps = eval_res.mean_crps
    mean_es   = eval_res.mean_es
    mean_ece = eval_res.mean_ece
    berk      = eval_res.berk or {"per_h": [], "fisher": {}, "hs_eff": []}

    per_h_results = berk.get("per_h", [])
    fisher_res = berk.get("fisher", {})
    hs_eff = berk.get("hs_eff", [])

    summary_params = count_qra_params(train_run / "train" / "checkpoints" / "qra" / "qra_models.pkl", nz_tol=0.0)

    # ---- infer N/H_eff
    def _infer_N_from_y(y_per_h):
        if isinstance(y_per_h, (list, tuple)):
            for arr in y_per_h:
                if isinstance(arr, np.ndarray) and arr.size:
                    return int(len(arr))
            return 0
        if isinstance(y_per_h, np.ndarray):
            return int(y_per_h.shape[0])
        return 0

    N = _infer_N_from_y(y_test_real)
    hs_eff = [h for h, q in enumerate(Q_fix_per_h) if isinstance(q, np.ndarray) and q.size > 0]
    H_eff = len(hs_eff)

    log.info(
        "QRA[test]: N=%d H_eff=%d | stride=%d",
        int(N),
        int(H_eff),
        int(origin_stride),
    )

    # -------------------- plotting (TEST) --------------------
    # fan-charts: only compute samples if enabled
    try:
        fan_cfg = getattr(getattr(getattr(test_cfg, "test", None), "plotting", None), "fan_chart", None)
    except Exception:
        fan_cfg = None

    samples_per_h_for_plot = None
    if fan_cfg and bool(getattr(fan_cfg, "enable_fan_plotting", False)):
        S = int(getattr(fan_cfg, "n_samples_for_plotting", 0))
        if S <= 0:
            raise ValueError("n_samples_for_plotting must be > 0 for sampling fan-charts.")
        log.info("QRA[test]: fan-chart plotting enabled (sampling path).")
        samples_per_h_for_plot = samples_from_q(Q_fix_real, taus_tgt, n_samples=S, device=device)

    plot_fan_charts(
        log=log,
        dm=dm,
        test_cfg=test_cfg,
        dirs=ctx.dirs,
        paths=paths,
        out_qra=p.out_qra,
        uid=uid,
        ods=ods,
        Q_fix_real=Q_fix_real,
        taus_tgt=taus_tgt,
        samples_per_h_for_plot=samples_per_h_for_plot,
    )

    # PIT hist
    pit_cfg = getattr(getattr(getattr(test_cfg, "test", None), "plotting", None), "pit", None)
    U_per_h = None
    if pit_cfg and bool(getattr(pit_cfg, "enable_pit_plotting", False)):
        log.info("QRA[test]: PIT histogram plotting enabled.")
        U_per_h = pits_from_quantiles(y_test_real, Q_fix_real, taus_tgt)

    for h in [0, 1, 12, 23]:
        y = y_test_real[h]
        Q = Q_fix_real[h]
        if isinstance(y, np.ndarray) and isinstance(Q, np.ndarray) and y.size and Q.size:
            cov_info = summarize_coverage(y, Q, taus_tgt)
            log.info(f"[DBG][test][COV][h={h}] mae={cov_info['mae']:.4f} rmse={cov_info['rmse']:.4f}")
            # optionally dump full vector for one horizon
            log.info(f"[DBG][test][COVVEC][h={h}] cov={np.round(np.array(cov_info['cov']), 4).tolist()} "
                    f"taus={np.round(np.array(cov_info['taus']), 4).tolist()}")

    plot_pit_histograms(
        log=log,
        test_cfg=test_cfg,
        dirs=ctx.dirs,
        paths=paths,
        out_qra=p.out_qra,
        hs_eff=hs_eff,
        U_per_h=U_per_h,
    )

    # ---- summary
    summary = pack_summary(
        split="test",
        device=device,
        Qte_per_h=Qte_per_h,
        mean_crps=mean_crps,
        mean_es=mean_es,
        mean_ece=mean_ece,
        berk={"per_h": per_h_results, "fisher": fisher_res},
        taus_train_used=taus_train_used,
        taus_tgt=taus_tgt,
        summary_params=summary_params,
        N=N,
        anchor_hour=anchor_hour,
        origin_stride=origin_stride,
    )
    (p.metrics_dir / "qra_summary_test.json").write_text(json.dumps(summary, indent=2))

    log.info("-" * 88)
    log.info("QRA[%s] RESULTS", ctx.split)
    log.info("  CRPS_mean        : %.4f", float(mean_crps))
    log.info("  ES_mean          : %.4f", float(mean_es))
    log.info("  N_origins        : %d (kept after filter)", int(N))
    log.info("  H_eff            : %d", int(H_eff))
    log.info("  filter           : anchor_hour=%s stride=%d", str(anchor_hour), int(origin_stride))
    log.info("  calib (Fisher p) : %.3g  (X=%.2f, df=%d, k=%d)",
             float(fisher_res.get("p", np.nan)),
             float(fisher_res.get("stat", np.nan)),
             int(fisher_res.get("df", 0)),
             int(fisher_res.get("k", 0)))
    log.info("  calib (HMP p)    : %.3g", float(fisher_res.get("hmp", np.nan)))
    log.info("  artifacts        : %s", str(get_art_dir(ctx.dirs, paths, p.out_qra)))
    log.info("-" * 88)

    return summary