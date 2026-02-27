# src/utils/optuna_helper.y
import math
from pathlib import Path
import pytorch_lightning as pl
import optuna
from omegaconf import DictConfig, ListConfig
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.pruners import MedianPruner, NopPruner
from omegaconf import DictConfig

class SimpleOptunaPruner(pl.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str, ignore_first_vals: int = 2):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.ignore_first_vals = int(ignore_first_vals)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # skip Lightning’s sanity validation
        if getattr(trainer, "sanity_checking", False):
            return

        epoch = int(trainer.current_epoch or 0)
        if epoch < self.ignore_first_vals:
            return

        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        try:
            value = float(metrics[self.monitor])
        except Exception:
            return
        if not math.isfinite(value):
            return

        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} with {self.monitor}={value:.5f}")
        

# ----------------------------- Optuna helpers -------------------------------- #

def _make_study(cfg: DictConfig) -> optuna.study.Study:
    metric_cfg = cfg.tune.metric
    is_multi = isinstance(metric_cfg, (list, tuple, ListConfig))

    # ---- sampler ----
    sampler_name = str(getattr(cfg.tune, "sampler", "tpe")).lower()

    if not is_multi:
        # single-objective
        if sampler_name in ("tpe", "motpe"):   # allow "motpe" alias
            sampler = TPESampler(seed=int(cfg.seed))
        elif sampler_name in ("nsgaii", "nsga2", "nsga-ii"):
            sampler = NSGAIISampler(seed=int(cfg.seed))
        else:
            sampler = TPESampler(seed=int(cfg.seed))
    else:
        # multi-objective
        if sampler_name in ("tpe", "motpe"):
            sampler = TPESampler(seed=int(cfg.seed))
        elif sampler_name in ("nsgaii", "nsga2", "nsga-ii"):
            sampler = NSGAIISampler(seed=int(cfg.seed))
        else:
            # safe default for Pareto search
            sampler = NSGAIISampler(seed=int(cfg.seed))

    # ---- pruner ---- (usually disable for multi-objective)
    pruner_map = {
        "none": NopPruner(),
        "median": MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    }
    pruner = NopPruner() if is_multi else pruner_map.get(str(cfg.tune.pruner).lower(), NopPruner())

    storage = cfg.tune.storage if cfg.tune.storage else None

    if is_multi:
        if hasattr(cfg.tune, "directions") and cfg.tune.directions is not None:
            directions = [str(x) for x in cfg.tune.directions]
            assert len(directions) == len(list(metric_cfg))
        else:
            directions = [str(cfg.tune.direction)] * len(list(metric_cfg))
        return optuna.create_study(
            study_name=str(cfg.tune.study_name),
            directions=directions,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
    else:
        return optuna.create_study(
            study_name=str(cfg.tune.study_name),
            direction=str(cfg.tune.direction),
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )


def _trial_run_dir(base_dir: Path, trial: optuna.Trial) -> Path:
    d = base_dir / f"trial_{trial.number:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d