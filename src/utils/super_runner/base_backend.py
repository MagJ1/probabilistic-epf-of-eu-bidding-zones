# models/common/backends/base_backend.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

class Backend(ABC):
    """Abstract adapter for a forecasting backend (NF, NHITS+QRA, …)."""

    # ---- locations ----------------------------------------------------------
    @abstractmethod
    def test_dir(self, train_run_dir: Path, test_tag: str) -> Path:
        ...

    @abstractmethod
    def metrics_json_path(self, train_run_dir: Path, test_tag: str) -> Path:
        ...

    # ---- train / test command construction ---------------------------------
    @abstractmethod
    def make_finetune_cmd(
        self,
        cfg,
        run_id: str,
        feat_t: Dict[str, list],
        runs_root: Path,
        abs_train: str,
        abs_test: str,
    ) -> List[str]:
        ...

    @abstractmethod
    def make_test_cmd(
        self,
        cfg,
        train_run_dir: Path,
        test_tag: str,
    ) -> List[str]:
        ...

    # ---- artifact discovery / completeness checks --------------------------
    @abstractmethod
    def train_checkpoint_globs(self, train_run_dir: Path) -> List[Path]:
        """Return GLOB **PATTERNS** (can include wildcards)."""
        ...

    @abstractmethod
    def test_is_complete(self, train_run_dir: Path, test_tag: str) -> bool:
        """Return True if test artifacts for decision are present."""
        ...

    @abstractmethod
    def train_is_complete(self, train_run_dir: Path) -> bool:
        """Return True when a train run is 'ready for testing'."""
        ...

    # ---- metric files used by the decision logic ---------------------------
    @abstractmethod
    def pred_paths_for_metric(
        self,
        train_run_dir: Path,
        test_tag: str,
        metric: str,
    ) -> Dict[str, Optional[Path]]:
        """
        Return concrete files for the metric:
        - for CRPS: {"crps_detailed": Path}
        - for ES:   {"es_detailed": Path, "fallback_crps": Optional[Path]}
        """
        ...

    # ---- feature normalization (optional) ----------------------------------
    def normalize_features_for_backend(self, feat_t: Dict[str, list], cfg) -> Dict[str, list]:
        """Default: identity. Override if backend expects different keys."""
        return feat_t


def any_ckpt_exists(globs: List[Path]) -> bool:
    for pat in globs:
        for _ in pat.parent.glob(pat.name):
            return True
    return False