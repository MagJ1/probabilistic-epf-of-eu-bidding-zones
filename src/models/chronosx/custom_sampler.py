from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd

from pydantic.v1 import PrivateAttr
from gluonts.transform.sampler import InstanceSampler


class AnchoredStrideInstanceSampler(InstanceSampler):
    # Pydantic fields (validated / stored)
    context_length: int
    prediction_length: int
    stride_hours: int = 24
    anchor_hour: Optional[int] = None
    seed: Optional[int] = None

    # private state (NOT a pydantic field)
    _rng: np.random.Generator = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        # init RNG
        self._rng = np.random.default_rng(None if self.seed is None else int(self.seed))

        # normalize stride semantics
        if self.anchor_hour is not None:
            sh = int(self.stride_hours)
            self.stride_hours = 24 if sh < 24 else sh
        else:
            self.stride_hours = max(1, int(self.stride_hours))

        if self.anchor_hour is not None and not (0 <= int(self.anchor_hour) <= 23):
            raise ValueError("anchor_hour must be in 0..23 or None")

    def _get_hour(self, start, t: int) -> int:
        try:
            return int((start + t).hour)  # pd.Period + int hours
        except Exception:
            ts0 = start.to_timestamp() if hasattr(start, "to_timestamp") else pd.Timestamp(start)
            return int((ts0 + pd.Timedelta(hours=t)).hour)

    def _first_anchored_t(self, start, t_min: int, t_max: int, anchor_hour: int) -> Optional[int]:
        for t in range(t_min, min(t_min + 24, t_max + 1)):
            if self._get_hour(start, t) == int(anchor_hour):
                return t
        return None

    def _valid_indices(self, start, T: int) -> List[int]:
        t_min = int(self.context_length)
        t_max = int(T - self.prediction_length)
        if t_max < t_min:
            return []

        if self.anchor_hour is None:
            first_t = t_min
            step = int(self.stride_hours)
        else:
            first_t = self._first_anchored_t(start, t_min, t_max, int(self.anchor_hour))
            if first_t is None:
                return []
            step = int(self.stride_hours)

        return list(range(first_t, t_max + 1, step))

    def __call__(self, ts: np.ndarray, start: Optional[object] = None, *args, **kwargs) -> List[int]:
        if start is None:
            start = kwargs.get("start", None)

        # if anchoring requested but we have no timestamp -> no samples
        if start is None and self.anchor_hour is not None:
            print("Sampler kwargs keys:", list(kwargs.keys()))
            return []

        T = int(len(ts))
        valid = self._valid_indices(start if start is not None else 0, T)
        if not valid:
            return []

        return [int(self._rng.choice(valid))]