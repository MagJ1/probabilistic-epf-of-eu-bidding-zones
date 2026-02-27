# models/moirai/custom_transforms.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from uni2ts.transform import Transformation as uni2tsTransformation
from gluonts.transform import MapTransformation
from gluonts.dataset.common import DataEntry

class MaskTailOptional(uni2tsTransformation):
    """
    Set the last `steps` raw time steps of an optional feature field (e.g.
    'past_feat_dynamic_real') to NaN so AddObservedMask will mark them
    unobserved. Place BEFORE AddObservedMask.
    """
    def __init__(self, steps: int, optional_field: str = "past_feat_dynamic_real"):
        self.steps = int(steps)
        self.optional_field = optional_field

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.steps <= 0 or self.optional_field not in data:
            return data
        x = data[self.optional_field]  # shape (F, T) pre-Patchify
        k = min(self.steps, x.shape[-1])
        x[..., -k:] = np.nan
        data[self.optional_field] = x
        return data

class MaskContextTailPostSplit(MapTransformation):
    """
    After InstanceSplitter (post-split), mask the last `steps` rows of the
    *context segment* for a past-only feature by:
      • setting values to `fill_value`
      • flipping the observed flag to False

    Expects post-split shapes (NTC=True):
      past_feat_dynamic_real: (past_length, past_feat_dim)
      past_observed_feat_dynamic_real: (past_length, past_feat_dim)
    """

    def __init__(
        self,
        steps: int,
        *,
        field: str = "past_feat_dynamic_real",
        observed_field: str = "past_observed_feat_dynamic_real",
        context_length: Optional[int] = None,  # informative only
        fill_value: float = 0.0,
    ):
        self.steps = int(steps)
        self.field = field
        self.observed_field = observed_field
        self.context_length = None if context_length is None else int(context_length)
        self.fill_value = float(fill_value)

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        # GluonTS passes each split here (dict-like "DataEntry")
        if self.steps <= 0:
            return data

        x = data.get(self.field, None)
        m = data.get(self.observed_field, None)
        if x is None or m is None:
            return data

        # Ensure we don't mutate shared buffers
        x = np.array(x, copy=True)
        m = np.array(m, copy=True)

        T = int(x.shape[0])
        if T <= 0:
            return data

        k = min(self.steps, T)
        x[-k:, ...] = self.fill_value
        m[-k:, ...] = False

        data[self.field] = x
        data[self.observed_field] = m
        return data
