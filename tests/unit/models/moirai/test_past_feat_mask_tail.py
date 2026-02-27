# tests/test_past_feat_mask_tail.py
import numpy as np
from models.moirai.custom_transforms import MaskTailOptional  # or MaskTailByNaNOptional
from uni2ts.transform import AddObservedMask, ImputeTimeSeries, DummyValueImputation

def test_tail_mask_applied_and_propagates_to_mask_and_impute():
    T, K = 32, 5
    entry = {
        "target": np.ones((1, T), dtype=np.float32),
        "past_feat_dynamic_real": np.ones((2, T), dtype=np.float32),
    }

    # 1) Mask tail by inserting NaNs on the past-only feats
    entry = MaskTailOptional(steps=K, optional_field="past_feat_dynamic_real")(entry)

    # 2) Observed mask must be built AFTER the NaNs are in place
    entry = AddObservedMask(
        fields=("target",),
        optional_fields=("past_feat_dynamic_real",),
        observed_mask_field="observed_mask",
        collection_type=dict,
    )(entry)

    m = entry["observed_mask"]["past_feat_dynamic_real"]  # shape (2,T), dtype=bool
    # last K are unobserved (False), earlier are observed (True)
    assert m[:, :-K].all()
    assert (~m[:, -K:]).all()

    # 3) Imputation should fill those NaNs with the chosen dummy value (e.g., 0.0)
    entry = ImputeTimeSeries(
        fields=("target",),
        optional_fields=("past_feat_dynamic_real",),
        imputation_method=DummyValueImputation(value=0.0),
    )(entry)

    # Confirm data was imputed at the tail and target stayed intact
    assert np.all(entry["past_feat_dynamic_real"][:, -K:] == 0.0)
    assert np.all(entry["target"] == 1.0)