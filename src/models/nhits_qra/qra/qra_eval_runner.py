# src/models/nhits_qra/qra/qra_eval_runner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from models.nhits_qra.qra.qra_io import safe_get
from models.nhits_qra.qra.qra_eval import (
    compute_crps_es,
    berkowitz_suite,
    compute_pinball_per_h,
    write_pinball_files,
    write_score_files,
    write_berkowitz_files,
    write_ece_files,
)
from models.nhits_qra.qra.qra_post import samples_from_q
from utils.metrics import compute_ece_per_h


@dataclass(frozen=True)
class QRAEvalResult:
    prefix: str
    mean_pinball: float
    pinball_per_h: List[Optional[float]]

    mean_crps: float
    crps_per_h: List[float]
    mean_es: float
    es_per_h: List[float]

    mean_ece: float
    ece_per_h: List[Optional[float]]

    berk: Optional[Dict[str, Any]]  # berkowitz_suite output
    crps_bh: Optional[np.ndarray] = None  # shape (N, H_eff)
    es_bh: Optional[np.ndarray] = None    # shape (N, H_eff)


def _fmt(x: Any, nd: int = 6) -> str:
    try:
        xf = float(x)
    except Exception:
        return "nan"
    if np.isnan(xf):
        return "nan"
    return f"{xf:.{nd}f}"


def _log_metric_summary(
    *,
    log: Any,
    prefix: str,
    compute_set: set,
    mean_pinball: float,
    mean_crps: float,
    mean_es: float,
    berk: Optional[Dict[str, Any]],
    mean_ece: float | None = None,
) -> None:
    # What was requested (and thus safe to talk about)
    if compute_set:
        log.info("QRA[%s]: compute=%s", prefix, sorted(list(compute_set)))

    parts: List[str] = []

    if "pinball" in compute_set:
        parts.append(f"pinball={_fmt(mean_pinball, 6)}")

    if "crps" in compute_set:
        parts.append(f"CRPS={_fmt(mean_crps, 6)}")

    if "es" in compute_set:
        parts.append(f"ES={_fmt(mean_es, 6)}")

    if "ece" in compute_set and mean_ece is not None:
        parts.append(f"ECE={_fmt(mean_ece, 6)}")

    if "berkowitz" in compute_set and isinstance(berk, dict):
        fisher = berk.get("fisher", {}) if isinstance(berk.get("fisher", {}), dict) else {}
        p = fisher.get("p", np.nan)
        hmp = fisher.get("hmp", np.nan)
        stat = fisher.get("stat", np.nan)
        df = fisher.get("df", None)
        k = fisher.get("k", None)

        # Keep it compact and robust
        if df is not None and k is not None:
            parts.append(f"Fisher_p={_fmt(p, 3)} (X={_fmt(stat, 2)}, df={int(df)}, k={int(k)})")
        else:
            parts.append(f"Fisher_p={_fmt(p, 3)}")
        parts.append(f"HMP_p={_fmt(hmp, 3)}")

    if parts:
        log.info("QRA[%s]: %s", prefix, " | ".join(parts))


def run_qra_eval(
    *,
    eval_cfg: Any,
    metrics_dir: Path,
    log: Any,
    device: str,
    y_real_per_h: List[Optional[np.ndarray]],
    Q_fix_real_per_h: List[Optional[np.ndarray]],
    taus_tgt: Sequence[float],
    tb_dir: Path,
) -> QRAEvalResult:
    """
    Hydra-controlled evaluation for QRA.

    eval_cfg schema:
      compute: [pinball, crps, es, berkowitz, (ece)]
      export:  [pinball, crps, es, berkowitz, (ece)]
      prefix:  val|test
      resources:
        n_samples_eval: int
      tensorboard:
        enable: bool
        log_per_h: bool
        # log_dir is ignored here because passing tb_dir explicitly
    """
    compute_set = set(list(safe_get(eval_cfg, "compute", [])))
    export_set = set(list(safe_get(eval_cfg, "export", [])))
    prefix = str(safe_get(eval_cfg, "prefix", "eval"))

    # --- tensorboard ---
    tb_enable = bool(safe_get(eval_cfg, "tensorboard.enable", False))
    tb_log_per_h = bool(safe_get(eval_cfg, "tensorboard.log_per_h", True))

    writer: Optional[SummaryWriter] = None
    if tb_enable:
        # Keep TB logs separated per prefix
        tb_dir_eff = Path(tb_dir) / "qra" / prefix
        tb_dir_eff.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir_eff))
        log.info("QRA[%s]: TensorBoard enabled -> %s", prefix, str(tb_dir_eff))

    # ---------------------------
    # pinball (on FIXED quantiles)
    # ---------------------------
    mean_pinball = float("nan")
    pinball_per_h: List[Optional[float]] = []
    if "pinball" in compute_set:
        mean_pinball, pinball_per_h = compute_pinball_per_h(
            y_real_per_h=y_real_per_h,
            Q_real_per_h=Q_fix_real_per_h,
            taus=taus_tgt,
        )

    # ---------------------------
    # CRPS / ES (sampling)
    # ---------------------------
    mean_crps = float("nan")
    mean_es = float("nan")
    crps_per_h: List[float] = []
    es_per_h: List[float] = []

    if ("crps" in compute_set) or ("es" in compute_set):
        n_samples_eval = int(safe_get(eval_cfg, "resources.n_samples_eval", 0))
        if n_samples_eval <= 0:
            raise ValueError("qra_eval.resources.n_samples_eval must be > 0 if CRPS/ES is requested.")
        samples_per_h = samples_from_q(Q_fix_real_per_h, taus_tgt, n_samples=n_samples_eval, device=device)
        mean_crps, crps_per_h, mean_es, es_per_h, extra = compute_crps_es(samples_per_h, y_real_per_h, device)
        crps_bh = extra.get("crps_bh", None) if isinstance(extra, dict) else None
        es_bh   = extra.get("es_bh", None) if isinstance(extra, dict) else None

    # ---------------------------
    # Berkowitz suite
    # ---------------------------
    berk: Optional[Dict[str, Any]] = None
    if "berkowitz" in compute_set:
        berk = berkowitz_suite(y_real_per_h, Q_fix_real_per_h, taus_tgt)

    # ---------------------------
    # ECE
    # ---------------------------
    mean_ece = float("nan")
    ece_per_h: List[Optional[float]] = []
    n_used = 0
    if "ece" in compute_set:
        mean_ece, ece_per_h, n_used = compute_ece_per_h(
            y_real_per_h=y_real_per_h,
            Q_real_per_h=Q_fix_real_per_h,
            taus=taus_tgt,
        )

    # ---------------------------
    # exports
    # ---------------------------
    if "pinball" in export_set and "pinball" in compute_set:
        write_pinball_files(
            metrics_dir,
            prefix=prefix,
            mean_pinball=mean_pinball,
            pinball_per_h=pinball_per_h,
            taus=taus_tgt,
        )

    if (("crps" in export_set) or ("es" in export_set)) and (("crps" in compute_set) or ("es" in compute_set)):
        n_samples_eval = int(safe_get(eval_cfg, "resources.n_samples_eval", 0))
        write_score_files(
            metrics_dir,
            prefix=prefix,
            mean_crps=mean_crps,
            crps_list=crps_per_h,
            mean_es=mean_es,
            es_list=es_per_h,
            n_samples_val=n_samples_eval,
        )

    if "berkowitz" in export_set and ("berkowitz" in compute_set) and berk is not None:
        write_berkowitz_files(metrics_dir, prefix=prefix, berk=berk)

    if "ece" in export_set and "ece" in compute_set:
        write_ece_files(
            metrics_dir=metrics_dir,
            prefix=prefix,
            mean_ece=mean_ece,
            ece_per_h=ece_per_h,
            n_samples_per_h=n_used,
            n_taus=len(list(taus_tgt)),
        )

    # ---------------------------
    # metric-only logging (SAFE)
    # ---------------------------
    _log_metric_summary(
        log=log,
        prefix=prefix,
        compute_set=compute_set,
        mean_pinball=mean_pinball,
        mean_crps=mean_crps,
        mean_es=mean_es,
        berk=berk,
        mean_ece=mean_ece,
    )

    # ---------------------------
    # tensorboard logging
    # ---------------------------
    if writer is not None:
        step = 0
        if "pinball" in compute_set:
            writer.add_scalar(f"{prefix}/pinball_mean", mean_pinball, step)
            if tb_log_per_h:
                for h, v in enumerate(pinball_per_h):
                    if v is not None:
                        writer.add_scalar(f"{prefix}/pinball_h{h:02d}", float(v), step)

        if ("crps" in compute_set) or ("es" in compute_set):
            if "crps" in compute_set:
                writer.add_scalar(f"{prefix}/crps_mean", float(mean_crps), step)
                if tb_log_per_h:
                    for h, v in enumerate(crps_per_h):
                        writer.add_scalar(f"{prefix}/crps_h{h:02d}", float(v), step)

            if "es" in compute_set:
                writer.add_scalar(f"{prefix}/es_mean", float(mean_es), step)
                if tb_log_per_h:
                    for h, v in enumerate(es_per_h):
                        writer.add_scalar(f"{prefix}/es_h{h:02d}", float(v), step)

        if "berkowitz" in compute_set and berk is not None:
            fisher = berk.get("fisher", {}) if isinstance(berk, dict) else {}
            p = fisher.get("p", np.nan)
            hmp = fisher.get("hmp", np.nan)
            writer.add_scalar(f"{prefix}/berkowitz_fisher_p", float(p) if p is not None else float("nan"), step)
            writer.add_scalar(f"{prefix}/berkowitz_hmp_p", float(hmp) if hmp is not None else float("nan"), step)

        if "ece" in compute_set:
            writer.add_scalar(f"{prefix}/ece_mean", float(mean_ece), step)
            if tb_log_per_h:
                for h, v in enumerate(ece_per_h):
                    if v is not None:
                        writer.add_scalar(f"{prefix}/ece_h{h:02d}", float(v), step)

        writer.flush()
        writer.close()

    return QRAEvalResult(
        prefix=prefix,
        mean_pinball=mean_pinball,
        pinball_per_h=pinball_per_h,
        mean_crps=float(mean_crps),
        crps_per_h=list(crps_per_h),
        mean_es=float(mean_es),
        es_per_h=list(es_per_h),
        mean_ece=float(mean_ece),
        ece_per_h=ece_per_h,
        berk=berk,
        crps_bh=crps_bh,
        es_bh=es_bh,
    )