# Minimal Step 1 scaffold: init state, compute run_id, dry-run print
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels(\.|$)")

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from utils.super_runner.factory import get_backend

from utils.logging_utils import init_logging, get_logger
from utils.super_runner.super_runner_common import _load_state, _init_state, _ensure_baseline, _resume_or_start_next

# -----------------
# Super-runner main
# -----------------

@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="config_super",
)
def main(cfg: DictConfig):
    super_dir = Path.cwd()  # hydra chdir here: outputs/.../super_runs/<timestamp>
    (super_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Logging for the super-runner itself
    init_logging(
        log_dir=str(super_dir / "logs"),
        run_id="super_runner",
        level=str(cfg.logging.log_level),
    )
    log = get_logger("super_runner")
    log.info("Super-run started in %s", str(super_dir))

    backend = get_backend(cfg)
    
    # Prepare roots to pass to train/test scripts
    runs_root = Path(to_absolute_path(cfg.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    state_path = super_dir / "state.json"
    state = _load_state(state_path)
    if state is None:
        log.info("No state.json found. Initializing new campaign state.")
        state = _init_state(cfg, super_dir)
    else:
        log.info("Loaded existing state.json. Resuming campaign.")

    project_root = Path(to_absolute_path("."))
    state = _ensure_baseline(state, cfg, runs_root, log, backend=backend)
    _resume_or_start_next(super_dir, project_root, state_path, state, cfg, runs_root, log, backend)
    # Sanity: candidates present?
    if not state["candidates_queue"]:
        log.info("No candidates in queue. Nothing to do. Exiting.")
        return
    


if __name__ == "__main__":
    main()