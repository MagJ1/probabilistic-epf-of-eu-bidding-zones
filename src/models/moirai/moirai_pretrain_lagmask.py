from collections import defaultdict
from typing import Callable, Dict
from uni2ts.transform import (
    SampleDimension, GetPatchSize, PatchCrop, PackFields, AddObservedMask,
    ImputeTimeSeries, DummyValueImputation, Patchify, AddVariateIndex, AddTimeIndex,
    MaskedPrediction, ExtendMask, FlatPackCollection, FlatPackFields,
    SequencifyField, SelectFields, DefaultPatchSizeConstraints, Transformation
)
from models.moirai.custom_transforms import MaskTailOptional
from uni2ts.model.moirai.pretrain import MoiraiPretrain

class MoiraiPretrainLagMask(MoiraiPretrain):
    def __init__(self, *args, lag_mask_steps: int = 14, **kwargs):
        super().__init__(*args, **kwargs)
        self._lag_mask_steps = int(lag_mask_steps)

    @property
    def train_transform_map(self,
                            ) -> Dict[str | type, Callable[..., Transformation]]:
        def default_train_transform(distance=None, prediction_length=None, context_length=None, patch_size=None):
            return (
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=DefaultPatchSizeConstraints(),
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                )
                + PackFields(output_field="target", fields=("target",), feat=False)
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=False,
                )
                + PackFields(
                    output_field="feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("feat_dynamic_real",),
                    feat=False,
                )
                # <<< INSERT: make the last K steps of past-only features unknown >>>
                + MaskTailOptional(
                    steps=self._lag_mask_steps,
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
                    randomize=True,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "observed_mask"),
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
                + FlatPackCollection(field="variate_id", feat=False)
                + FlatPackCollection(field="time_id", feat=False)
                + FlatPackCollection(field="prediction_mask", feat=False)
                + FlatPackCollection(field="observed_mask", feat=True)
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real","feat_dynamic_real"),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )
        return defaultdict(lambda: default_train_transform)