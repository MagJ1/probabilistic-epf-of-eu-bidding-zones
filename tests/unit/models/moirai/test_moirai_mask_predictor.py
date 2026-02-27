# tests/test_moirai_mask_predictor.py
import numpy as np
import pandas as pd
import torch
import pytest

from models.moirai.moirai_forecast_lagmask import MoiraiForecastLagMask
from uni2ts.model.moirai import MoiraiModule

def make_raw_entry(T=256, F_past=2):
    # one series, hourly
    start = pd.Timestamp("2024-01-01 00:00:00").to_period("H")  
    target = np.linspace(0.0, 1.0, T).astype("float32")
    past_feat = np.stack([np.arange(T), np.arange(T)[::-1]], 0).astype("float32")[:F_past]
    return {
        "start": start,
        "target": target,
        "past_feat_dynamic_real": past_feat,
    }


@pytest.mark.parametrize("patch_size", [32, "auto"])
def test_post_split_mask_application(patch_size):
    ctx = 128
    H   = 24
    k_mask = 14
    fill = 0.0

    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
    model = MoiraiForecastLagMask(
        module=base,
        prediction_length=H,
        context_length=ctx,
        patch_size=patch_size,
        num_samples=8,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=2,
        lag_mask_steps=k_mask,
        lag_mask_value=fill,
    )

    predictor = model.create_predictor(batch_size=2)
    tf = predictor.input_transform

    # With auto patching, the InstanceSplitter needs past_length=(ctx+H) and future_length=H
    raw = make_raw_entry(T=ctx + 2*H + 64, F_past=2)

    one = next(tf(iter([raw]), is_train=False))

    X = one["past_feat_dynamic_real"]          # (past_len, F_past)
    M = one["past_observed_feat_dynamic_real"] # (past_len, F_past)

    assert X.ndim == 2 and M.ndim == 2, (X.shape, M.shape)
    past_len = X.shape[0]

    expected_past_len = (ctx + H) if patch_size == "auto" else ctx
    assert past_len == expected_past_len, (past_len, expected_past_len)

    # The "context segment" used for prediction is the *last ctx* steps of past_* in the auto case too.
    ctx_start = max(0, past_len - ctx)
    ctx_end = past_len

    # We want to mask the last k steps of the context tail (i.e., last k rows of X within [ctx_start:ctx_end])
    mask_start = max(ctx_start, ctx_end - k_mask)

    # masked tail
    assert np.allclose(X[mask_start:ctx_end], fill), "context tail values not set to fill_value"
    assert np.all(~M[mask_start:ctx_end]),          "context tail observed flags not set to False"

    # unmasked context head
    assert np.all(M[ctx_start:mask_start]), "context head should remain observed=True"
    assert not np.allclose(X[ctx_start:mask_start], fill), "context head should not be all fill_value"


def test_end_to_end_prediction_runs():
    ctx, H = 128, 24
    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")

    model = MoiraiForecastLagMask(
        module=base,
        prediction_length=H,
        context_length=ctx,
        patch_size=32,
        num_samples=4,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=2,
        lag_mask_steps=14,
        lag_mask_value=0.0,
    )
    predictor = model.create_predictor(batch_size=2)

    # Make a tiny "dataset" of two entries
    raws = [make_raw_entry(T=ctx + H + 32, F_past=2) for _ in range(2)]
    fcsts = list(predictor.predict(raws))
    assert len(fcsts) == 2
    s = fcsts[0].samples  # (S, H) since target_dim=1
    assert s.shape[1] == H
    print("Prediction runs. Samples shape:", s.shape)


def test_toggle_mask_changes_tail():
    ctx, H = 128, 24
    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")

    def get_tail(flags_on: bool):
        steps = 14 if flags_on else 0
        m = MoiraiForecastLagMask(
            module=base,
            prediction_length=H,
            context_length=ctx,
            patch_size=32,
            num_samples=4,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=2,
            lag_mask_steps=steps,
            lag_mask_value=0.0,
        )
        pred = m.create_predictor(batch_size=1)
        tf = pred.input_transform
        one = next(tf(iter([make_raw_entry(T=ctx + H + 16, F_past=2)]), is_train=False))
        return one["past_feat_dynamic_real"].copy(), one["past_observed_feat_dynamic_real"].copy()

    X_off, M_off = get_tail(False)
    X_on,  M_on  = get_tail(True)

    # their shapes equal
    assert X_off.shape == X_on.shape == M_off.shape == M_on.shape

    # When mask is on, last 14 rows differ (values 0, observed False)
    k = 14
    tail = slice(-k, None)
    assert not np.allclose(X_off[tail], X_on[tail]) or np.any(M_off[tail] != M_on[tail])
    assert np.all(~M_on[tail]) and not np.all(~M_off[tail])
    print("Toggling mask alters tail as expected.")