# models/common/backends/nhits_qra_backend.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import json
import joblib
from hydra.core.hydra_config import HydraConfig

from utils.super_runner.base_backend import Backend  # ABC

def _hydra_cli_list(xs: List[str]) -> str:
    q = lambda s: '"' + str(s).replace('"', '\\"') + '"'
    return "[" + ",".join(q(x) for x in xs) + "]"

class NHITSQRABackend(Backend):
    """
    Adapter for NHITS+QRA pipeline.

    Layout:
      train_run/
        train/checkpoints/nhits/*.ckpt
        train/checkpoints/qra/qra_models.pkl
        data/qra/{qra_meta.pkl, pcas.pkl?}
        meta/params.json
      test/<tag>/{metrics,predictions,meta,...}
    """

    # ---------------- locations ----------------
    def test_dir(self, train_run_dir: Path, test_tag: str) -> Path:
        return train_run_dir / "test" / test_tag

    def metrics_json_path(self, train_run_dir: Path, test_tag: str) -> Path:
        return self.test_dir(train_run_dir, test_tag) / "metrics" / "test_metrics.json"

    # ---------------- commands ----------------
    def make_finetune_cmd(
        self,
        cfg,
        run_id: str,
        feat_t: Dict[str, list],
        runs_root: Path,
        abs_train: str,
        abs_test: str,
    ) -> List[str]:
        ck_cli = _hydra_cli_list(feat_t.get("ck_cols", []))
        cu_cli = _hydra_cli_list(feat_t.get("cu_cols", []))
        st_cli = _hydra_cli_list(feat_t.get("static_cols", []))

        eb = "true" if getattr(cfg.train_nhits, "enable_progress_bar", True) else "false"
        pw = "true" if getattr(cfg.train_nhits, "persistent_workers", True) else "false"

        data_choice = HydraConfig.get().runtime.choices.get("data")
        nhits_model_choice = HydraConfig.get().runtime.choices.get("model")
        train_nhits_choice = HydraConfig.get().runtime.choices.get("train_nhits")
        train_qra_choice = HydraConfig.get().runtime.choices.get("train_qra")


        return [
            "python", "-u", "-m", "models.nhits_qra.cli.train_nhits_qra",
            f"data={data_choice}",
            f"model={nhits_model_choice}",
            f"train_nhits={train_nhits_choice}",
            f"train_qra={train_qra_choice}",
            f"data.train_csv_path={abs_train}",
            f"data.test_csv_path={abs_test}",
            f"features.active.ck={ck_cli}",
            f"features.active.cu={cu_cli}",
            f"features.active.static={st_cli}",
            f"train.run_id={run_id}",
            f"runs_root={str(runs_root)}",
            f"experiment={cfg.experiment}",
            f"train_nhits.enable_progress_bar={eb}",
            f"train_nhits.n_epochs={int(getattr(cfg.train_nhits, 'n_epochs', 1))}",
            f"train_nhits.num_workers={int(cfg.train_nhits.num_workers)}",
            f"train_nhits.persistent_workers={pw}",
        ]

    def make_test_cmd(self, cfg, train_run_dir: Path, test_tag: str) -> List[str]:
        out_dir = self.test_dir(train_run_dir, test_tag)
        eb = "true" if getattr(cfg.test, "enable_progress_bar", True) else "false"
        nse = int(getattr(cfg.test, "n_samples_eval", 200))
        nhits_model_choice = HydraConfig.get().runtime.choices.get("model")
        data_choice = HydraConfig.get().runtime.choices.get("data")
        experiment_choice = HydraConfig.get().runtime.choices.get("experiment")

        return [
            "python", "-u", "-m", "models.nhits_qra.cli.test_nhits_qra",
            f"experiment={experiment_choice}",
            f"model={nhits_model_choice}",
            f"data={data_choice}",
            f"test.source_run_dir={str(train_run_dir.resolve())}",
            f"test.source_run_id={str(train_run_dir.name)}",
            f"out_dir={str(out_dir)}",
            f"tag={test_tag}",
            f"test.qra_eval.resources.n_samples_eval={nse}",
            f"test.enable_progress_bar={eb}",
        ]

    # --------- artifacts / completeness ----------
    def train_checkpoint_globs(self, train_run_dir: Path) -> List[Path]:
        # Only NHITS ckpts here; QRA checked in train_is_complete()
        return [train_run_dir / "train" / "checkpoints" / "nhits" / "*.ckpt"]

    def train_is_complete(self, train_run_dir: Path) -> bool:
        nhits_dir = train_run_dir / "train" / "checkpoints" / "nhits"
        if not any(nhits_dir.glob("*.ckpt")):
            return False

        # Is QRA enabled?
        qra_enabled = True
        cfg_path = train_run_dir / "meta" / "params.json"
        try:
            if cfg_path.exists():
                with cfg_path.open("r") as f:
                    cfg = json.load(f)
                qra_enabled = bool(cfg.get("train_qra", {}).get("enabled", True))
        except Exception:
            qra_enabled = True  # be conservative

        if not qra_enabled:
            return True

        qra_models = train_run_dir / "train" / "checkpoints" / "qra" / "qra_models.pkl"
        meta_path  = train_run_dir / "data" / "qra" / "qra_meta.pkl"
        if not (qra_models.exists() and meta_path.exists()):
            return False

        # Require PCAs if used
        try:
            meta = joblib.load(meta_path)
            use_pca   = bool(meta.get("use_pca", False))
            n_samples = int(meta.get("n_samples", 0))
        except Exception:
            use_pca, n_samples = True, 1

        if use_pca and n_samples > 0:
            pcas = train_run_dir / "data" / "qra" / "pcas.pkl"
            if not pcas.exists():
                return False

        return True

    def test_is_complete(self, train_run_dir: Path, test_tag: str) -> bool:
        # Accept either sentinel or metrics file
        root = self.test_dir(train_run_dir, test_tag)
        sentinel = root / f"_done.test.{test_tag}"
        metrics  = self.metrics_json_path(train_run_dir, test_tag)
        return sentinel.exists() or metrics.exists()

    # ------------- metric files for decision -------------
    def pred_paths_for_metric(
        self,
        train_run_dir: Path,
        test_tag: str,
        metric: str,
    ) -> Dict[str, Optional[Path]]:
        pred_dir = self.test_dir(train_run_dir, test_tag) / "predictions"

        def prefer_parquet_csv(stem: str) -> Optional[Path]:
            pqt = pred_dir / f"{stem}.parquet"
            if pqt.exists():
                return pqt
            csv = pred_dir / f"{stem}.csv"
            return csv if csv.exists() else None

        # Map canonical keys to your current filenames (“*_per_origin.*”)
        if metric.lower() == "es_mean":
            return {
                "es_per_origin": prefer_parquet_csv("es_per_origin"),
            }
        else:
            return {
                "crps_per_origin": prefer_parquet_csv("crps_per_origin"),
            }

    # ------------- features normalization -------------
    def normalize_features_for_backend(self, feat_t: Dict[str, list], cfg) -> Dict[str, list]:
        return {
            "ck_cols": list(feat_t.get("ck_cols", [])),
            "cu_cols": list(feat_t.get("cu_cols", [])),
            "static_cols": list(feat_t.get("static_cols", [])),
        }