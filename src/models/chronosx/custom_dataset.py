# src/models/chronosx/custom_dataset.py
from __future__ import annotations
from typing import Iterable, List, Optional

from torch.utils.data import get_worker_info
from models.chronosx.custom_sampler import AnchoredStrideInstanceSampler
import numpy as np
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic
from gluonts.transform import (
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    FilterTransformation,
    InstanceSplitter,
    MissingValueImputation,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)

# Import upstream dataset
from chronosx.utils.chronos_dataset import ChronosDataset  # their code
from models.chronosx.custom_transforms import MaskPastCovariates, MaskFutureUnknownChannels  # ourextension


class ChronosDatasetExt(ChronosDataset):
    """
    Drop-in replacement for ChronosDataset with two changes:

    1) Removes random DropValues(target) -> no random NaNs on target anywhere.
    2) Adds a post-split MaskPastCovariates(transform) that hides the last K steps
       of `past_feat_dynamic_real` for the selected splits (train/validation/test).

    All other behavior (tokenization, label shift, seq2seq/causal handling)
    remains upstream-identical.
    """

    def __init__(
        self,
        *args,
        realistic_mode_enabled: bool = True,
        realistic_mask_len: int = 14,
        realistic_apply_in: Optional[List[str]] = None,   # ["training","validation","test"]
        realistic_mask_value: float = np.nan,
        realistic_mask_future: bool = True,
        realistic_mask_channel_indices: Optional[List[int]] = None,  # past only channels
        disable_target_drop: bool = True,
        origin_enabled: bool = False,
        origin_stride_hours: int = 24,
	    origin_anchor_hour: Optional[int] = 0,
        origin_seed: Optional[int] = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._rm_enabled = bool(realistic_mode_enabled)
        self._rm_mask_len = int(realistic_mask_len)
        self._rm_apply_in = set((realistic_apply_in or ["test"]))
        self._rm_mask_value = realistic_mask_value
        self._rm_mask_future = bool(realistic_mask_future)
        self._rm_mask_idx = None if realistic_mask_channel_indices is None else list(realistic_mask_channel_indices)
        self._disable_target_drop = bool(disable_target_drop)

        self._origin_enabled = origin_enabled
        self._origin_stride_hours = origin_stride_hours
        self._origin_anchor_hour = origin_anchor_hour
        self._origin_seed = origin_seed

        if self._origin_enabled:
            if self._origin_anchor_hour is not None and not (0 <= int(self._origin_anchor_hour) <= 23):
                raise ValueError("origin_anchor_hour must be in 0..23 or None")
            if int(self._origin_stride_hours) <= 0:
                raise ValueError("origin_stride_hours must be positive")

    # --------- override: no DropValues(target) ----------
    def create_transformation(self) -> Transformation:
        """
        Upstream adds: SelectFields -> DropValues(target) -> AddObservedValuesIndicator.
        We omit DropValues so no random missing targets.
        """
        fields = [FieldName.START, FieldName.TARGET]
        if self.include_covariates:
            fields.append(FieldName.FEAT_DYNAMIC_REAL)

        # SelectFields + ObservedValues only
        return (
            SelectFields(fields, allow_missing=True)
            + AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            )
        )

    # --------- keep the same splitting, but optionally append MaskPastCovariates ---------
    def _create_instance_splitter(self, mode: str, worker_id: int = 0) -> InstanceSplitter:
        assert mode in ("training", "validation", "test")

        if self._origin_enabled:
            # stable per-worker seed
            seed = int(self._origin_seed) + int(worker_id)
            # optionally also separate train/val/test streams
            if mode != "training":
                seed += 1

            instance_sampler = AnchoredStrideInstanceSampler(
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                stride_hours=self._origin_stride_hours,
                anchor_hour=self._origin_anchor_hour,
                seed=seed,
            )
        else:
            instance_sampler = {
                "training": ExpectedNumInstanceSampler(
                    num_instances=1.0,
                    min_instances=1,
                    min_past=self.min_past,
                    min_future=self.prediction_length,
                ),
                "validation": ValidationSplitSampler(min_future=self.prediction_length),
                "test": TestSplitSampler(),
            }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=self.time_series_fields,
            dummy_value=np.nan,
        )

    def _maybe_mask(self, it: Iterable, split_name: str) -> Iterable:
        if (
            self._rm_enabled
            and split_name in self._rm_apply_in
            and self.include_covariates
        ):
            tfs = []
            # 1) tail of past window (K steps) for past-only channels
            tfs.append(
                MaskPastCovariates(
                    mask_len=self._rm_mask_len,
                    mask_value=self._rm_mask_value,
                    indices=self._rm_mask_idx,
                )
            )
            # 2) entire future window for those channels
            if self._rm_mask_future:
                tfs.append(
                    MaskFutureUnknownChannels(
                        mask_value=self._rm_mask_value,
                        indices=self._rm_mask_idx,
                    )
                )
            return Chain(tfs).apply(it, is_train=(split_name == "training"))
        return it

    def create_training_data(self, data):
        data = Cyclic(data)
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        split = self._create_instance_splitter("training", worker_id=wid)
        pipe = split + FilterTransformation(
            condition=lambda e: (~np.isnan(e["past_target"])).sum() > 0
        )
        it = pipe.apply(data, is_train=True)
        return self._maybe_mask(it, "training")

    def create_validation_data(self, data):
        split = self._create_instance_splitter("validation")
        it = split.apply(data, is_train=False)
        return self._maybe_mask(it, "validation")

    def create_test_data(self, data):
        split = self._create_instance_splitter("test")
        it = split.apply(data, is_train=False)
        return self._maybe_mask(it, "test")