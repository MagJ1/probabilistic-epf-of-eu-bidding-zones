from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

def ensure_dtindex(x) -> pd.DatetimeIndex:
    # robust, tz-naive, hourly floor
    odt = pd.to_datetime(x, errors="coerce")
    odt = pd.DatetimeIndex(odt)
    if getattr(odt, "tz", None) is not None:
        odt = odt.tz_localize(None)
    return odt.floor("h")

def _build_origin_mask_anchor_stride(
    uid_all: np.ndarray,
    ods_all,
    anchor_hour: Optional[int],
    origin_stride_hours: int,
) -> np.ndarray:
    """Per-series keep mask. Without anchor: stride in hours. With anchor: daily grid at hour=h, stride in days."""
    uid_all = np.asarray(uid_all)
    ods = ensure_dtindex(ods_all)
    N = len(uid_all)
    keep = np.zeros(N, dtype=bool)
    stride_h = int(max(1, origin_stride_hours))

    for u in np.unique(uid_all):
        idx = np.flatnonzero(uid_all == u)       # keep original order
        if anchor_hour is not None:
            ah = int(anchor_hour)
            idx = idx[ods[idx].hour.values == ah]
            step = max(1, stride_h // 24)        # stride in *days*
        else:
            step = stride_h                      # stride in *hours*
        if idx.size:
            keep[idx[::step]] = True
    return keep