# models/common/backends/nf_backend.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from hydra.core.hydra_config import HydraConfig

from utils.super_runner.base_backend import Backend, any_ckpt_exists


class NFBackend(Backend):
    """Adapter for Normalizing Flows runner (single-stage)."""

    # ---------- locations ----------
    def test_dir(self, train_run_dir: Path, test_tag: str) -> Path:
        return train_run_dir / "test" / test_tag

    def metrics_json_path(self, train_run_dir: Path, test_tag: str) -> Path:
        return self.test_dir(train_run_dir, test_tag) / "metrics" / "test_metrics.json"

    # ---------- command construction ----------
    def make_finetune_cmd(
        self,
        cfg,
        run_id: str,
        feat_t: Dict[str, list],
        runs_root: Path,
        abs_train: str,
        abs_test: str,
    ) -> List[str]:
        model_choice = HydraConfig.get().runtime.choices.get("model")
        train_choice = HydraConfig.get().runtime.choices.get("train")
        data_choice = HydraConfig.get().runtime.choices.get("data")
        return [
            "python", "-u", "-m", "models.normalizing_flows.cli.train_flows",
            f"model={model_choice}",
            f"train={train_choice}",
            f"data={data_choice}",
            f"features.ck_cols={feat_t.get('ck_cols', [])}",
            f"features.cu_cols={feat_t.get('cu_cols', [])}",
            f"train.run_id={run_id}",
            f"experiment={cfg.experiment}",
            f"data.train_csv_path={abs_train}",
            f"data.test_csv_path={abs_test}",
            f"runs_root={str(runs_root)}",
            f"train.enable_progress_bar={cfg.train.enable_progress_bar}",
            f"train.n_epochs={int(cfg.train.n_epochs)}",
            f"train.num_workers={int(cfg.train.num_workers)}",
            f"train.persistent_workers={bool(cfg.train.persistent_workers)}",
            f"data.origin_stride_train={cfg.data.origin_stride_train}",
            f"data.origin_stride_val={cfg.data.origin_stride_val}",
        ]

    def make_test_cmd(self, cfg, train_run_dir: Path, test_tag: str) -> List[str]:
        model_choice = HydraConfig.get().runtime.choices.get("model")
        data_choice = HydraConfig.get().runtime.choices.get("data")
        test_choice = HydraConfig.get().runtime.choices.get("test")
        return [
            "python", "-u", "-m", "models.normalizing_flows.cli.test_flows",
            f"model={model_choice}",
            f"data={data_choice}",
            f"test={test_choice}",
            f"test.source_run_id={str(train_run_dir.name)}",
            f"test.source_run_dir={str(train_run_dir.resolve())}",
            f"tag={test_tag}",
            f"data.test_csv_path={cfg.data.test_csv_path}",
            f"test.n_samples_eval={cfg.test.n_samples_eval}",
            f"test.enable_progress_bar={cfg.test.enable_progress_bar}",
            f"data.origin_stride_test={cfg.data.origin_stride_test}",
            f"data.origin_anchor_hour_test={cfg.data.origin_anchor_hour_test}",
        ]

    # ---------- artifact discovery / completeness ----------
    def train_checkpoint_globs(self, train_run_dir: Path) -> List[Path]:
        # support both new and legacy layouts
        return [
            train_run_dir / "train" / "checkpoints" / "*.ckpt",
            train_run_dir / "checkpoints" / "*.ckpt",
        ]

    def train_is_complete(self, train_run_dir: Path) -> bool:
        # NF has no extra heads to check; "complete" == any train checkpoint exists
        return any_ckpt_exists(self.train_checkpoint_globs(train_run_dir))

    def test_is_complete(self, train_run_dir: Path, test_tag: str) -> bool:
        return self.metrics_json_path(train_run_dir, test_tag).exists()

    # ---------- metric files used by decision logic ----------
    def pred_paths_for_metric(
        self,
        train_run_dir: Path,
        test_tag: str,
        metric: str,
    ) -> Dict[str, Optional[Path]]:
        pred_dir = self.test_dir(train_run_dir, test_tag) / "predictions"

        def pick(stem: str) -> Optional[Path]:
            p = pred_dir / f"{stem}.parquet"
            if p.exists():
                return p
            q = pred_dir / f"{stem}.csv"
            return q if q.exists() else None

        if metric.lower() == "es_mean":
            return {"es_per_origin": pick("es_per_origin")}
        else:  # "crps"
            return {"crps_per_origin": pick("crps_per_origin")}

    # ---------- features normalization (optional) ----------
    def normalize_features_for_backend(self, feat_t: Dict[str, list], cfg) -> Dict[str, list]:
        # identity mapping for NF
        return {
            "ck_cols": list(feat_t.get("ck_cols", [])),
            "cu_cols": list(feat_t.get("cu_cols", [])),
        }