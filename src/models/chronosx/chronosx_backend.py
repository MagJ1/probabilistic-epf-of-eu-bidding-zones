# src/utils/backends/chronosx_backend.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from hydra.core.hydra_config import HydraConfig

from utils.super_runner.base_backend import Backend, any_ckpt_exists


class ChronosXBackend(Backend):
    """
    Adapter for ChronosX (finetune + test) driven by the super-runner.
    Assumes train runner is:   models.chronosx.train_chronosx
    and test runner is:        models.chronosx.test_chronosx

    Produced folder structure:
      <run_dir>/
        train/checkpoints/{last_state_dict.pt,best_state_dict.pt,*.ckpt}
        test/<tag>/
          predictions/{es_per_origin.parquet|csv, crps_per_origin.parquet|csv}
          metrics/test_metrics.json
    """

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
        """
        ChronosX training overrides — minimal and explicit.
        Train runner derives features_order & past_only_names from ck/cu and persists them.
        """
        ck = feat_t.get("ck_cols", [])
        cu = feat_t.get("cu_cols", [])
        model_choice = HydraConfig.get().runtime.choices.get("model")
        data_choice = HydraConfig.get().runtime.choices.get("data")
        train_choice = HydraConfig.get().runtime.choices.get("train")
        return [
            "python", "-u", "-m", "models.chronosx.cli.train_chronosx",
            f"model={model_choice}",
            f"data={data_choice}",
            f"train={train_choice}",
            f"experiment={cfg.experiment}",
            f"train.run_id={run_id}",
            f"runs_root={str(runs_root)}",
            f"data.train_csv_path={abs_train}",
            f"data.val_csv_path={cfg.data.val_csv_path}",
            f"features.ck_cols={ck}",
            f"features.cu_cols={cu}",
        ]

    def make_test_cmd(self, cfg, train_run_dir: Path, test_tag: str) -> List[str]:
        """
        ChronosX test runner. Ask for both ES & CRPS per-origin for the decision stage.
        """
        model_choice = HydraConfig.get().runtime.choices.get("model")
        data_choice = HydraConfig.get().runtime.choices.get("data")
        experiment_choice = HydraConfig.get().runtime.choices.get("experiment")
        return [
            "python", "-u", "-m", "models.chronosx.cli.test_chronosx",
            f"test.source_run_dir={str(train_run_dir.resolve())}",
            f"tag={test_tag}",
            # generation / batching
            f"data={data_choice}",
            f"model={model_choice}",
            f"experiment={experiment_choice}",
            f"data.test_csv_path={str(cfg.data.test_csv_path)}",
            f"test.num_samples={int(cfg.test.num_samples)}",
            f"test.batch_size={int(cfg.test.batch_size)}",
            f"model.device_map={cfg.model.device_map}",
            # windowing
            f"data.origin_stride_test={int(cfg.data.origin_stride_test)}",
            f"data.origin_anchor_hour_test={cfg.data.origin_anchor_hour_test}",
            # force both per-origin files
            f"output.metric={cfg.output.metric}",
            # progress bar
            f"progress_bar.enabled={bool(cfg.test.enable_progress_bar)}",
        ]

    # ---------- artifact discovery / completeness ----------
    def train_checkpoint_globs(self, train_run_dir: Path) -> List[Path]:
        return [
            train_run_dir / "train" / "checkpoints" / "*.pt",
            train_run_dir / "train" / "checkpoints" / "*.ckpt",
        ]

    def train_is_complete(self, train_run_dir: Path) -> bool:
        return any_ckpt_exists(self.train_checkpoint_globs(train_run_dir))

    def test_is_complete(self, train_run_dir: Path, test_tag: str) -> bool:
        return self.metrics_json_path(train_run_dir, test_tag).exists()

    # ---------- metric files used by the decision logic ----------
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

        metric = str(metric).lower()
        if metric == "es_mean":
            return {"es_per_origin": pick("es_per_origin")}
        elif metric == "crps_mean":
            return {"crps_per_origin": pick("crps_per_origin")}
        return {}

    # ---------- features normalization (identity) ----------
    def normalize_features_for_backend(self, feat_t: Dict[str, list], cfg) -> Dict[str, list]:
        return {
            "ck_cols": list(feat_t.get("ck_cols", [])),
            "cu_cols": list(feat_t.get("cu_cols", [])),
        }