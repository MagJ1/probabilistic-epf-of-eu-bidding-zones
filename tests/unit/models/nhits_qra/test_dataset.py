# test_nhits_qra_dataset.py
from __future__ import annotations
import numpy as np
import torch
import pandas as pd
import pytest

from src.models.nhits_qra.dataset import NHITSQRAForecasterDataset  # adjust path if needed

def _build_datasets(csv_path, params, split_dates, scale_data=True):
    # train (fits scalers in base)
    train_ds = NHITSQRAForecasterDataset(
        csv_path=csv_path,
        split_type="train",
        scale_data=scale_data,
        **params,
        **split_dates,
    )
    # val (reuse target scaler from train)
    val_ds = NHITSQRAForecasterDataset(
        csv_path=csv_path,
        split_type="val",
        scale_data=scale_data,
        **params,
        **split_dates,
        target_scaler_state=train_ds.get_target_scaler(),
    )
    # share feature scalers too (if any were fit)
    val_ds.set_scalers(train_ds.get_scalers())
    return train_ds, val_ds

def test_len_positive(synthetic_csv_file, dataset_params, split_dates):
    train_ds, val_ds = _build_datasets(synthetic_csv_file, dataset_params, split_dates)
    assert len(train_ds) > 0
    assert len(val_ds) > 0

def test_batch_shapes_and_types(synthetic_csv_file, dataset_params, split_dates):
    C = dataset_params["context_length"]
    H = dataset_params["forecast_horizon"]
    train_ds, _ = _build_datasets(synthetic_csv_file, dataset_params, split_dates)

    batch = train_ds[0]
    # ds is list[str] covering context+future: length C+H
    assert isinstance(batch["ds"], list)
    assert len(batch["ds"]) == C + H
    # tensors + shapes
    assert isinstance(batch["y_past"], torch.Tensor) and batch["y_past"].shape == (C,)
    assert isinstance(batch["y_future"], torch.Tensor) and batch["y_future"].shape == (H,)

    C_ck = len(dataset_params["ck_cols"])
    C_cu = len(dataset_params["cu_cols"])

    assert batch["c_ctx_future_known"].shape == (C, C_ck)
    assert batch["c_fct_future_known"].shape == (H, C_ck)
    assert batch["c_ctx_future_unknown"].shape == (C, C_cu)
    assert batch["flags_ctx_cu_known"].shape == (C, C_cu)

    uid_idx = batch["unique_id"]
    assert isinstance(uid_idx, (int, np.integer))
    # static can be empty or not depending on registry; here we expect unique_id included
    static = batch["static"]
    assert isinstance(static, torch.Tensor)
    # since registry marks unique_id as static, expect at least length 1
    assert static.ndim == 1 and static.shape[0] >= 1
    assert static.dtype == torch.float32
    assert np.isclose(static[0].item(), float(uid_idx))


def test_realistic_masking_applied(synthetic_csv_file, dataset_params, split_dates):
    C = dataset_params["context_length"]
    cutoff = dataset_params["past_unknown_cov_cutoff"]
    train_ds, _ = _build_datasets(synthetic_csv_file, dataset_params, split_dates)

    # pick a sample safely away from edges
    idx = min(2, len(train_ds)-1)
    b = train_ds[idx]

    flags = b["flags_ctx_cu_known"].numpy()
    cu_ctx = b["c_ctx_future_unknown"].numpy()

    if flags.shape[1] == 0:
        pytest.skip("No cu features to test masking against.")

    # last 'cutoff' rows should be 0; before that 1
    assert flags.shape[0] == C
    if cutoff > 0:
        assert np.all(flags[:-cutoff, :] == 1.0)
        assert np.all(flags[-cutoff:, :] == 0.0)
        # where flags==0, cu should be neutralized to zero (after scaling)
        masked = cu_ctx[-cutoff:, :]
        assert np.allclose(masked, 0.0, atol=1e-6)

def test_scaling_toggle_changes_values(synthetic_csv_file, dataset_params, split_dates):
    # Build two datasets: one scaled, one raw
    ds_scaled, _ = _build_datasets(synthetic_csv_file, dataset_params, split_dates, scale_data=True)
    ds_raw, _    = _build_datasets(synthetic_csv_file, dataset_params, split_dates, scale_data=False)

    b_scaled = ds_scaled[0]
    b_raw    = ds_raw[0]

    # y_past should differ (z-scored vs raw)
    ys = b_scaled["y_past"].numpy()
    yr = b_raw["y_past"].numpy()
    assert not np.allclose(ys, yr)

    # same for a cu channel (if exists)
    cu_s = b_scaled["c_ctx_future_unknown"].numpy()
    cu_r = b_raw["c_ctx_future_unknown"].numpy()
    if cu_s.shape[1] > 0:
        # compare only on the “available” part (avoid masked zeros influencing)
        cutoff = dataset_params["past_unknown_cov_cutoff"]
        available_rows = slice(0, cu_s.shape[0]-cutoff) if cutoff > 0 else slice(None)
        if (cu_s.shape[0]-cutoff) > 0:
            assert not np.allclose(cu_s[available_rows], cu_r[available_rows])

def test_ds_window_alignment(synthetic_csv_file, dataset_params, split_dates):
    C = dataset_params["context_length"]
    H = dataset_params["forecast_horizon"]
    train_ds, _ = _build_datasets(synthetic_csv_file, dataset_params, split_dates)

    b0 = train_ds[0]
    # ds contains context+future timestamps → length C+H
    assert len(b0["ds"]) == C + H

    # check that the split between context and future is at index C
    # convert back to pd.Timestamp for comparison
    ds_all = list(map(pd.Timestamp, b0["ds"]))
    ctx_end = ds_all[C-1]
    fut_start = ds_all[C]
    # ensure future starts one hour after context end
    assert fut_start == ctx_end + pd.Timedelta(hours=1)

def test_static_contains_uid_int(synthetic_csv_file, dataset_params, split_dates):
    train_ds, _ = _build_datasets(synthetic_csv_file, dataset_params, split_dates)
    b = train_ds[0]

    static = b["static"].cpu().numpy()
    uid_int = float(b["unique_id"])
    if train_ds.id_col in train_ds.static_cols:
        pos = train_ds.static_cols.index(train_ds.id_col)
        assert np.isclose(static[pos], uid_int)