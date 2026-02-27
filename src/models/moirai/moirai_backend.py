# models/moirai/moirai_backend.py
from __future__ import annotations
from pathlib import Path
import re
import shlex
from typing import Dict, List, Optional
from hydra.utils import to_absolute_path

from utils.super_runner.base_backend import Backend, any_ckpt_exists
from models.moirai.moirai_finetune_lagmask import MoiraiFinetuneLagMask

def _hydra_list(val_list) -> str:
    # For CLI list overrides: output.metric=[es,crps]
    items = [str(x) for x in val_list]
    return "[" + ",".join(items) + "]"

def _resolve_hub_id(base_hub_id: str) -> str:
    # allow local HF export dir (contains config.json + model.safetensors)
    p = Path(base_hub_id)
    return str(p.resolve()) if p.exists() else base_hub_id

def _pick_arch_hub(train_run_dir: Path, cfg) -> str:
    """Prefer this-run HF export, else fallback to base_hub_id."""
    run_export = train_run_dir / "train" / "HF_checkpoints" / "last"
    if run_export.joinpath("config.json").exists():
        return str(run_export.resolve())
    base = getattr(cfg.model, "base_hub_id", getattr(cfg.model, "hub_id", ""))
    return _resolve_hub_id(base)

class MoiraiBackend(Backend):
    """
    Adapter for Moirai. Supports:
      - zero_shot:   no train, just test_moirai
      - one_shot / few_shot/ finetune: finetune via Hydra CLI, then test_moirai with ckpt
    """

    def __init__(self, cfg):
        # we persist what we need for train/test decisions/commands
        self.cfg = cfg
        self.experiment = str(cfg.experiment).lower()

    # ---------- locations ----------
    def test_dir(self, train_run_dir: Path, test_tag: str) -> Path:
        return train_run_dir / "test" / test_tag

    def metrics_json_path(self, train_run_dir: Path, test_tag: str) -> Path:
        return self.test_dir(train_run_dir, test_tag) / "metrics" / "test_metrics.json"
    
    # ---------- optional pre-train steps: build datasets ----------
    def make_pre_train_cmds(
        self,
        cfg,
        run_id: str,
        feat_t: Dict[str, list],
        runs_root: Path,
    ) -> List[List[str]]:
        """Return *two* builder commands (train + val) when finetune is enabled,
        else [] for zero-shot.
        Requires CUSTOM_DATA_PATH to be set in the environment"""
        if self.experiment == "zero_shot":
            return []

        b = cfg.moirai.builder
        id_col   = str(cfg.data.id_col)
        time_col = str(cfg.data.date_col)
        tgt_col  = str(cfg.data.target_col)

        ck_cols = list(feat_t.get("ck_cols", []))
        cu_cols = list(feat_t.get("cu_cols", []))

        run_dir  = runs_root / run_id
        data_dir = run_dir / "train" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Per-run dataset names to avoid clashes between trials
        ds_train_name = f"SR_{run_id}_train"

        def _build_cmd(ds_name: str, csv_path: str) -> List[str]:
            cmd = [
                "python", "-u", "-m", "models.moirai.build_moirai_dataset",
                ds_name, str(Path(to_absolute_path(csv_path)).resolve()),
                "--dataset_type", str(getattr(b, "dataset_type")),
                "--id_col", id_col,
                "--time_col", time_col,
                "--target_col", tgt_col,
            ]
            if ck_cols:
                cmd += ["--ck_cols"] + ck_cols
            if cu_cols:
                cmd += ["--cu_cols"] + cu_cols
            if bool(getattr(b, "normalize", True)):
                cmd += ["--normalize"]
            date_offset = getattr(b, "date_offset", None)
            if date_offset:
                cmd += ["--date_offset", str(date_offset)]
            return cmd
        
        build_tokens = _build_cmd(ds_train_name, b.train_csv_path)
        build_cmd = [
            "bash", "-lc",
            f'CUSTOM_DATA_PATH="{data_dir}" ' + " ".join(shlex.quote(str(x)) for x in build_tokens)
        ]

        conf_root = Path(to_absolute_path(cfg.moirai.conf_fine_dir)).resolve()
        out_dir_train = conf_root / "data"
        out_dir_val   = conf_root / "val_data"

        windows = cfg.windows
        fine    = cfg.moirai.finetune

        yaml_cmd = [
            "python", "-u", "-m", "models.moirai.tools.make_ft_data_yaml",
            "--dataset-name", ds_train_name,
            "--csv", str(Path(to_absolute_path(b.train_csv_path)).resolve()),
            "--id-col", id_col,
            "--time-col", time_col,
            "--target-col", tgt_col,
            "--context-length", str(int(windows.context_length)),
            "--prediction-length", str(int(windows.prediction_length)),
            "--patch-size", str(int(fine.patch_size)),
            "--mode", str(fine.mode),
            "--distance", "1",
            "--make-eval",
            "--eval_date_offset", str(b.date_offset),
            "--eval-distance", str(windows.distance),
            "--offset", str(windows.offset),
            "--out-dir-train", str(out_dir_train),
            "--out-dir-val",   str(out_dir_val),
        ]

        print(f"DATE_OFFSET: {str(b.date_offset)}")

        return [build_cmd, yaml_cmd]


    # ---------- command construction ----------
    def make_finetune_cmd(
        self,
        cfg,
        run_id: str,
        feat_t: Dict[str, list],
        runs_root: Path,
        abs_train: str,     # kept for interface symmetry; unused here
        abs_test: str,      # same
    ) -> List[str]:
        """
        For zero_shot we won't be called (train_is_complete==True).
        For one_shot/few_shot we call the uni2ts finetune CLI and force Hydra to write into <run_dir>/train.
        """
        if self.experiment == "zero_shot":
            return ["python", "-c", "print('zero_shot: skipping train')"]

        run_dir  = runs_root / run_id
        data_dir = run_dir / "train" / "data"
        # Where this run should live
        hydra_run_dir = str((run_dir / "train").resolve())

        # Required config knobs for the user's finetune CLI
        fine = cfg.moirai.finetune
        # NOTE: MUST set this in super cfg to point to:
        # src/models/moirai/uni2ts/cli/conf/finetune
        conf_fine_dir = str(Path(to_absolute_path(cfg.moirai.conf_fine_dir)).resolve())
        
        ds_train_name = f"SR_{run_id}_train"
        ds_train_val = f"{ds_train_name}_eval"

        # already prepared datasets externally (e.g., GER_train / GER_train_eval)
        cmd= [
            "python", "-u", "-m", "models.moirai.cli.train_moirai",
            "-cp", conf_fine_dir,
            "-cn", "config_finetune",
            f"exp_name={cfg.experiment}",
            f"run_name={run_id}",
            f"model={fine.model}",
            f"model.patch_size={int(fine.patch_size)}",
            f"model.context_length={int(cfg.windows.context_length)}",
            f"model.prediction_length={int(cfg.windows.prediction_length)}",
            f"data={ds_train_name}",
            f"data.patch_size={int(fine.patch_size)}",
            f"data.context_length={int(cfg.windows.context_length)}",
            f"data.prediction_length={int(cfg.windows.prediction_length)}",
            f"data.mode={str(fine.mode)}",
            f"val_data={ds_train_val}",
            f"val_data.patch_size={int(fine.patch_size)}",
            f"val_data.context_length={int(cfg.windows.context_length)}",
            f"val_data.prediction_length={int(cfg.windows.prediction_length)}",
            f"trainer.enable_progress_bar={bool(cfg.moirai.finetune.enable_progress_bar)}",
            f"trainer.precision={int(cfg.moirai.finetune.trainer_precision)}",
            f"train_dataloader.num_workers={int(cfg.moirai.finetune.train_data_workers)}",
            # Force Hydra output into our controlled run_dir layout
            f"hydra.run.dir={hydra_run_dir}",
        ]
        joined = " ".join(shlex.quote(str(x)) for x in cmd)
        return ["bash", "-lc", f'CUSTOM_DATA_PATH="{data_dir}" {joined}']

    def make_test_cmd(self, cfg, train_run_dir: Path, test_tag: str, feat_t: Optional[Dict[str, list]] = None,) -> List[str]:
        data = cfg.data
        windows = cfg.windows
        model = cfg.model
        out = cfg.output

        ck = list((feat_t or {}).get("ck_cols", getattr(cfg.features, "ck_cols", [])))
        cu = list((feat_t or {}).get("cu_cols", getattr(cfg.features, "cu_cols", [])))

        # pick checkpoint if we trained
        ckpt_path: Optional[str] = None
        if self.experiment in ("one_shot", "few_shot", "finetune", "dev_finetune", "e3_grouped_features"):
            ckpt_path = self._pick_best_ckpt(train_run_dir / "train" / "checkpoints")

        base_hub_id = _pick_arch_hub(train_run_dir, cfg)

        out_dir = str(self.test_dir(train_run_dir, test_tag).resolve())
        metrics_list = list(out.metric)
        metric_cli = f"output.metric={_hydra_list(metrics_list)}"

        cmd = [
            "python", "-u", "-m", "models.moirai.cli.test_moirai",
            f"out_dir={out_dir}",
            f"tag={test_tag}",
            # data
            f"model_size={str(cfg.model.size)}",
            f"data.test_csv_path={str(Path(to_absolute_path(data.test_csv_path)).resolve())}",
            f"data.id_col={data.id_col}",
            f"data.date_col={data.date_col}",
            f"data.target_col={data.target_col}",
            f"data.feat_dynamic_real={_hydra_list(ck)}",
            f"data.past_feat_dynamic_real={_hydra_list(cu)}",
            f"data.fillna_forward={bool(getattr(data,'fillna_forward', True))}",
            # windows
            f"windows.context_length={int(windows.context_length)}",
            f"windows.prediction_length={int(windows.prediction_length)}",
            f"windows.distance={int(windows.distance)}",
            # model
            f"model.base_hub_id={base_hub_id}",
            f"model.num_samples={int(model.num_samples)}",
            f"model.batch_size={int(model.batch_size)}",
            metric_cli,
            f"output.format={out.format}",
            f"logging.log_level={cfg.logging.log_level}",
        ]
        # Only pass ckpt in few/one-shot
        if ckpt_path:
            cmd.append(f"model.ckpt_path={ckpt_path}")

        return cmd

    # ---------- artifact discovery / completeness ----------
    def train_checkpoint_globs(self, train_run_dir: Path) -> List[Path]:
        return [train_run_dir / "train" / "checkpoints" / "*.ckpt"]

    def train_is_complete(self, train_run_dir: Path) -> bool:
        # zero-shot: pretend "trained"
        if self.experiment == "zero_shot":
            return True
        return any_ckpt_exists(self.train_checkpoint_globs(train_run_dir))

    def test_is_complete(self, train_run_dir: Path, test_tag: str) -> bool:
        return self.metrics_json_path(train_run_dir, test_tag).exists()

    # ---------- metric files for decision ----------
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
        else:
            return {"crps_per_origin": pick("crps_per_origin")}

    # ---------- feature normalization ----------
    def normalize_features_for_backend(self, feat_t: Dict[str, list], cfg) -> Dict[str, list]:
        # Map to Moirai names (test script consumes these lists directly)
        return {
            "ck_cols": list(feat_t.get("ck_cols", [])),  # -> feat_dynamic_real
            "cu_cols": list(feat_t.get("cu_cols", [])),  # -> past_feat_dynamic_real
            "static_cols": list(feat_t.get("static_cols", [])),
        }
    
    # ---------- helper: best ckpt with fallback ----------
    def _pick_best_ckpt(self, ckpt_dir: Path) -> Optional[str]:
        if not ckpt_dir.exists():
            return None

        # 1) try explicit best (val metric in filename)
        # Examples: epoch=3-step=123-val_PackedNLLLoss=3.330.ckpt  OR
        #           epoch=2-...-val%2FPackedNLLLoss=3.540.ckpt
        best = None
        best_score = None
        pat = re.compile(r"(val[^=]*)=([0-9.]+)")
        for p in sorted(ckpt_dir.glob("*.ckpt")):
            m = pat.search(p.name.replace("%2F", "_").replace("/", "_"))
            if m:
                try:
                    score = float(m.group(2))
                    if (best_score is None) or (score < best_score):
                        best = p
                        best_score = score
                except Exception:
                    pass

        if best:
            return str(best.resolve())

        # 2) fallback to last.ckpt
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return str(last.resolve())

        # 3) final fallback: any ckpt
        any_c = sorted(ckpt_dir.glob("*.ckpt"))
        return str(any_c[0].resolve()) if any_c else None