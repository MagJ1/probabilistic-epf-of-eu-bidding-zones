# tests/test_features_in_pipeline.py
import numpy as np
from uni2ts.transform import (
    PackFields, AddObservedMask, ImputeTimeSeries, DummyValueImputation
)

def test_feat_dynamic_real_included_and_shaped():
    T = 32
    # 1 target dim, 3 future feats, 2 past-only feats
    entry = {
        "target": np.ones((1, T), dtype=np.float32),
        "feat_dynamic_real": np.stack(
            [np.full(T, 10.0), np.full(T, 20.0), np.full(T, 30.0)]
        ).astype(np.float32),
        "past_feat_dynamic_real": np.stack(
            [np.full(T, 1.0), np.full(T, 2.0)]
        ).astype(np.float32),
    }

    # Pack into separate fields (as pipeline does)
    entry = PackFields(output_field="target", fields=("target",), feat=False)(entry)
    entry = PackFields(
        output_field="feat_dynamic_real",
        fields=tuple(),
        optional_fields=("feat_dynamic_real",),
        feat=False,
    )(entry)
    entry = PackFields(
        output_field="past_feat_dynamic_real",
        fields=tuple(),
        optional_fields=("past_feat_dynamic_real",),
        feat=False,
    )(entry)

    # Observed mask is just to ensure nothing crashes here
    entry = AddObservedMask(
        fields=("target",),
        optional_fields=("feat_dynamic_real", "past_feat_dynamic_real"),
        observed_mask_field="observed_mask",
        collection_type=dict,
    )(entry)

    # After packing, shapes should be (dim, time)
    assert entry["target"].shape == (1, T)
    assert entry["feat_dynamic_real"].shape == (3, T)
    assert entry["past_feat_dynamic_real"].shape == (2, T)

    # Optional: imputation pass shouldn’t change non-NaN data
    entry = ImputeTimeSeries(
        fields=("target",),
        optional_fields=("feat_dynamic_real", "past_feat_dynamic_real"),
        imputation_method=DummyValueImputation(value=0.0),
    )(entry)
    assert np.all(entry["feat_dynamic_real"] == np.array([[10]*T,[20]*T,[30]*T],dtype=np.float32))