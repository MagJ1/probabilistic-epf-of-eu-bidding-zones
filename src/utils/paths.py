# src/utils/paths.py
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

def prepare_train_run_dirs(base_dir: Path):
    """
    Prepare run directory structure and return all key subpaths.
    base_dir is expected to already include timestamped run folder
    (e.g. checkpoints/normalizing_flows/base/2025-08-13_12-34-56).
    """
    dirs = {
        "root": base_dir,

        "train":   base_dir / "train",
        "ckpt":    base_dir / "train" / "checkpoints",
        "py":      base_dir / "train" / "logs_py",
        "metrics": base_dir / "train" / "metrics",

        "data": base_dir / "data",
        "meta": base_dir / "meta",
        "art":  base_dir / "artifacts",

        # add this too for consistency (even if unused in train/val)
        "pred": base_dir / "predictions",

        "logs_tb": base_dir / "logs_tb",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

def prepare_test_run_dirs(base_dir: Path):
    """
    Prepare run directory structure and return all key subpaths.
    base_dir is expected to already include timestamped run folder
    (e.g. checkpoints/normalizing_flows/base/2025-08-13_12-34-56).
    """
    dirs = {
        "root": base_dir,
        "metrics":  base_dir / "metrics",
        "py": base_dir / "logs_py",
        "pred": base_dir / "predictions",
        "meta": base_dir / "meta",
        "art": base_dir / "artifacts",
        "logs_tb": base_dir / "logs_tb"
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

def hydra_path_helper(cfg):
    hc = HydraConfig.get()
    if hc.job.chdir:               # hydra moved us already
        base_dir = Path.cwd()      # == ${out_root}
    else:
        base_dir = Path(to_absolute_path(cfg.out_root))
    return base_dir