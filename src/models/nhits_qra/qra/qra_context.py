# src/models/nhits_qra/qra/qra_context.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Literal
from omegaconf import OmegaConf

from utils.logging_utils import get_logger
from utils.device import pick_accelerator_and_devices, accelerator_to_torch_device


@dataclass(frozen=True)
class QRAContext:
    train_cfg: Any
    test_cfg: Any
    module: Any
    dm: Any
    dirs: Dict[str, Path]
    split: str
    load_from_train_run: Optional[Path]
    target_scaler: Optional[Tuple]
    log: Any
    # runtime choices
    accelerator: str
    devices: int
    precision: Any
    map_location: str
    torch_device: str

    @staticmethod
    def build(
        *,
        train_cfg: Any,
        test_cfg: Any,
        module: Any,
        dm: Any,
        dirs: Dict[str, Any],
        split: str,
        load_from_train_run: Optional[Path] = None,
        target_scaler: Optional[Tuple] = None,
        log=None,
        use_all_gpus: bool = False,
    ) -> "QRAContext":
        assert split in {"train", "val", "test"}

        log = log or get_logger("qra_pipeline")

        dirs_norm = {k: Path(v).expanduser().resolve() for k, v in (dirs or {}).items() if v is not None}
        ltr = Path(load_from_train_run).expanduser().resolve() if load_from_train_run is not None else None

        accel, devices, precision, map_location = pick_accelerator_and_devices(use_all_gpus=use_all_gpus)
        torch_device = accelerator_to_torch_device(accel)

        return QRAContext(
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            module=module,
            dm=dm,
            dirs=dirs_norm,
            split=split,
            load_from_train_run=ltr,
            target_scaler=target_scaler,
            log=log,
            accelerator=accel,
            devices=int(devices),
            precision=precision,
            map_location=map_location,
            torch_device=torch_device,
        )
    