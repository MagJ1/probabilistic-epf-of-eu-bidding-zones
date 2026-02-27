# qra_pipeline.py (refactored)
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

from models.nhits_qra.nhits.nhits_module import NHITSForecasterModule
from models.nhits_qra.datamodule import NHITSQRADataModule
from models.nhits_qra.qra.qra_io import prepare_paths
from models.nhits_qra.qra.qra_context import QRAContext
from models.nhits_qra.qra.qra_knobs import QRAKnobs
from models.nhits_qra.qra.qra_runner import run_train_val, run_test

# --------------------------- main entry ---------------------------

def run_qra(
    train_cfg: OmegaConf,
    module: NHITSForecasterModule,
    dm: NHITSQRADataModule,
    dirs: dict,
    split: str,
    load_from_train_run: Optional[Path] = None,
    target_scaler: Optional[Tuple] = None,
    log=None,
    test_cfg=None,
    use_all_gpus: bool = False,
) -> Dict:
    """
    Entry point into QRA pipeline:
    (1) Creates a QRAContext object that groups the required dependencies and runtime related stuff.
    (2) Creates a QRAKnobs object, which stores hyperparameters regarding the algorithm.
    (3)
    
    :param train_cfg: Description
    :type train_cfg: OmegaConf
    :param module: Description
    :type module: NHITSForecasterModule
    :param dm: Description
    :type dm: NHITSQRADataModule
    :param dirs: Description
    :type dirs: dict
    :param split: Description
    :type split: str
    :param load_from_train_run: Description
    :type load_from_train_run: Optional[Path]
    :param target_scaler: Description
    :type target_scaler: Optional[Tuple]
    :param log: Description
    :param test_cfg: Description
    :param use_all_gpus: Description
    :type use_all_gpus: bool
    :return: Description
    :rtype: Dict
    """

    ctx = QRAContext.build(
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        module=module,
        dm=dm,
        dirs=dirs,
        split=split,
        load_from_train_run=load_from_train_run,
        target_scaler=target_scaler,
        log=log,
        use_all_gpus=use_all_gpus,
    )
    assert ctx.split in {"train", "val", "test"}, f"Unsupported split: {ctx.split}"

    log = ctx.log
    knobs = QRAKnobs.from_cfg(train_cfg, test_cfg, log=log)
    log.info(
        "QRA[%s]: solver_loss=%s | torch_device=%s | accelerator=%s devices=%s precision=%s | lambda_grid=%s",
        ctx.split, knobs.solver_loss, ctx.torch_device, ctx.accelerator, ctx.devices, str(ctx.precision), knobs.lambda_grid
    )

    paths, p = prepare_paths(ctx, ctx.dirs)

    if ctx.split in {"train", "val"}:
        return run_train_val(
            ctx=ctx, train_cfg=train_cfg, module=module, dm=dm,
            knobs=knobs, paths=paths, p=p
        )

    return run_test(
        ctx=ctx, train_cfg=train_cfg, module=module, dm=dm,
        knobs=knobs, paths=paths, p=p, test_cfg=test_cfg
    )