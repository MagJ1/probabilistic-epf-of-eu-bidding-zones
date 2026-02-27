from collections import defaultdict
from typing import Callable, Dict
import math
from uni2ts.model.moirai.finetune import MoiraiFinetune
from uni2ts.transform import (
    GetPatchSize, FinetunePatchCrop, PackFields, EvalPad, AddObservedMask,
    ImputeTimeSeries, Patchify, AddVariateIndex, AddTimeIndex, AddSampleIndex,
    EvalMaskedPrediction, ExtendMask, FlatPackCollection, FlatPackFields, EvalCrop,
    SequencifyField, SelectFields, FixedPatchSizeConstraints, Transformation, DummyValueImputation
)
from models.moirai.custom_transforms import MaskTailOptional

class MoiraiFinetuneLagMask(MoiraiFinetune):
    """
    Same as MoiraiFinetune, but masks the last K steps of past-only features
    (e.g. past_feat_dynamic_real) while keeping target fully observed.
    You can use different K for train/val.
    """

    def __init__(
        self,
        *args,
        train_lag_steps: int = 14,   
        val_lag_steps: int = 14,    # realistic eval: hide last 14 for exogenous
        fill_value: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._train_lag_steps = int(train_lag_steps)
        self._val_lag_steps = int(val_lag_steps)
        self._fill_value = float(fill_value)

    @property
    def train_transform_map(self) -> Dict[str | type, Callable[..., Transformation]]:
        def default_train_transform(distance: int, prediction_length: int, context_length: int, patch_size: int):
            # ---- copy of MoiraiFinetune.train builder, with ONE extra line ----
            return (
                GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                    offset=True,
                )
                + FinetunePatchCrop(
                    distance, 
                    prediction_length, 
                    context_length,
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                )
                + PackFields(
                    output_field="target", 
                    fields=("target",),)
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(), 
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="feat_dynamic_real",
                    fields=tuple(), 
                    optional_fields=("feat_dynamic_real",),
                )
                + EvalPad(
                    prediction_pad=-prediction_length % patch_size,
                    context_pad=-context_length % patch_size,
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real",),
                )
                # >>> INSERTED: mask last K steps of past-only features <<<
                + MaskTailOptional(
                    steps=self._train_lag_steps,
                    optional_field="past_feat_dynamic_real",
                )
                # ---------------------------------------------------------------
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                )
                + AddVariateIndex(
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    variate_id_field="variate_id", 
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim, 
                    randomize=False, 
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    time_id_field="time_id", 
                    expected_ndim=3, 
                    collection_type=dict,
                )
                + AddSampleIndex(
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    sample_id_field="sample_id", 
                    expected_ndim=3, 
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(
                        "variate_id",
                        "time_id",
                        "observed_mask",
                        "sample_id"),
                    optional_truncate_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    prediction_mask_field="prediction_mask", 
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    mask_field="prediction_mask", 
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id", 
                    feat=False,)
                + FlatPackCollection(
                    field="time_id", 
                    feat=False,)
                + FlatPackCollection(
                    field="sample_id", 
                    feat=False,)
                + FlatPackCollection(
                    field="prediction_mask", 
                    feat=False,)
                + FlatPackCollection(
                    field="observed_mask", 
                    feat=True,)
                + FlatPackFields(
                    output_field="target",
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"), 
                    feat=True,
                )
                + SequencifyField(
                    field="patch_size", 
                    target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )
        return defaultdict(lambda: default_train_transform)

    @property
    def val_transform_map(
        self
        ) -> Dict[str | type, Callable[..., Transformation]]:
        def default_val_transform(
            offset: int, 
            distance: int, 
            prediction_length: int, 
            context_length: int, 
            patch_size: int):
            # ---- copy of MoiraiFinetune.val builder, with ONE extra line ----
            return (
                GetPatchSize(
                    min_time_patches=2,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                    offset=True,
                )
                + EvalCrop(
                    offset,
                    distance,
                    prediction_length,
                    context_length,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                )
                + PackFields(
                    output_field="target", 
                    fields=("target",),)
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(), 
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="feat_dynamic_real",
                    fields=tuple(), 
                    optional_fields=("feat_dynamic_real",),
                )
                + EvalPad(
                    prediction_pad=-prediction_length % patch_size,
                    context_pad=-context_length % patch_size,
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                )
                # >>> INSERTED: mask last K steps of past-only features for EVAL <<<
                + MaskTailOptional(
                    steps=self._val_lag_steps,
                    optional_field="past_feat_dynamic_real",
                )
                # ---------------------------------------------------------------
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                )
                + AddVariateIndex(
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    variate_id_field="variate_id", 
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim, 
                    randomize=False, 
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    time_id_field="time_id", 
                    expected_ndim=3, 
                    collection_type=dict,
                )
                + AddSampleIndex(
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    sample_id_field="sample_id", 
                    expected_ndim=3, 
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(
                        "variate_id",
                        "time_id",
                        "observed_mask",
                        "sample_id"
                        ),
                    optional_truncate_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    prediction_mask_field="prediction_mask", 
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    mask_field="prediction_mask", 
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id", 
                    feat=False,)
                + FlatPackCollection(
                    field="time_id", 
                    feat=False,)
                + FlatPackCollection(
                    field="sample_id", 
                    feat=False,)
                + FlatPackCollection(
                    field="prediction_mask", 
                    feat=False,)
                + FlatPackCollection(
                    field="observed_mask", 
                    feat=True,)
                + FlatPackFields(
                    output_field="target",
                    fields=("target",), 
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"), 
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )
        return defaultdict(lambda: default_val_transform)