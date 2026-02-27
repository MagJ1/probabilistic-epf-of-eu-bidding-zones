from __future__ import annotations
from typing import Iterable, Optional, Sequence
import numpy as np
from gluonts.dataset.common import DataEntry
from gluonts.transform import MapTransformation


class MaskPastCovariates(MapTransformation):
    """
    Mask the last `mask_len` time steps of past-only channels in past_feat_dynamic_real.
    """
    def __init__(
        self,
        mask_len: int = 14,
        mask_value: float = np.nan,
        indices: Optional[Sequence[int]] = None,   # channels to mask; None -> all
    ) -> None:
        super().__init__()
        self.mask_len = int(mask_len)
        self.mask_value = mask_value
        self.indices = None if indices is None else tuple(int(i) for i in indices)

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        key = "past_feat_dynamic_real"
        arr = data.get(key, None)
        if arr is None:
            return data

        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        T, C = arr.shape
        if self.mask_len <= 0 or T == 0:
            data[key] = arr
            return data

        m = min(self.mask_len, T)
        if self.indices is None:
            arr[-m:, :] = self.mask_value
        else:
            arr[-m:, self.indices] = self.mask_value

        data[key] = arr
        return data


class MaskFutureUnknownChannels(MapTransformation):
    """
    Mask *all* time steps of future-only channels in future_feat_dynamic_real.
    Use for channels not available beyond the forecast origin.
    """
    def __init__(
        self,
        mask_value: float = np.nan,
        indices: Optional[Sequence[int]] = None,  # channels to mask; None -> all
    ) -> None:
        super().__init__()
        self.mask_value = mask_value
        self.indices = None if indices is None else tuple(int(i) for i in indices)

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        key = "future_feat_dynamic_real"
        arr = data.get(key, None)
        if arr is None:
            return data

        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if self.indices is None:
            arr[:, :] = self.mask_value
        else:
            arr[:, self.indices] = self.mask_value

        data[key] = arr
        return data