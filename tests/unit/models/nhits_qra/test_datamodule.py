# tests/test_datamodule_nhits_qra.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.nhits_qra.datamodule import NHITSQRADataModule


def _build_dm(csv_path: str, dataset_params: dict, split_dates: dict, **extra_dm_kwargs):
    """Helper to construct the DM with fixtures + any overrides."""
    return NHITSQRADataModule(
        train_csv_path=csv_path,
        test_csv_path=csv_path,
        batch_size=4,
        num_workers=0,
        scale_data=True,           # apply transforms in the child dataset
        **dataset_params,          # context_length, forecast_horizon, cols, registry, etc.
        **split_dates,             # val_split_date
        **extra_dm_kwargs,         # share_scalers / toggles per-test
    )


def test_setup_fit_builds_and_shares(synthetic_csv_file, dataset_params, split_dates):
    dm = _build_dm(synthetic_csv_file, dataset_params, split_dates, share_scalers=True)
    dm.setup(stage="fit")

    # datasets exist
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None

    # target scaler shared train -> val
    t_scaler_train = dm.train_dataset.get_target_scaler()
    t_scaler_val   = dm.val_dataset.get_target_scaler()
    assert t_scaler_train is not None
    assert t_scaler_val == t_scaler_train

    # feature scalers shared train -> val (dict equality by keys and values)
    f_scalers_train = dm.train_dataset.get_scalers()
    f_scalers_val   = dm.val_dataset.get_scalers()
    assert isinstance(f_scalers_train, dict) and isinstance(f_scalers_val, dict)
    assert f_scalers_train.keys() == f_scalers_val.keys()
    for k in f_scalers_train.keys():
        assert f_scalers_val[k] == f_scalers_train[k]


def test_dataloaders_and_single_sample_contract(synthetic_csv_file, dataset_params, split_dates):
    C, H = dataset_params["context_length"], dataset_params["forecast_horizon"]
    dm = _build_dm(synthetic_csv_file, dataset_params, split_dates, share_scalers=True)
    dm.setup(stage="fit")

    b = dm.train_dataset[0]

    assert isinstance(b["ds"], list) and len(b["ds"]) == C + H
    ds_all = list(map(pd.Timestamp, b["ds"]))
    assert ds_all[C] == ds_all[C - 1] + pd.Timedelta(hours=1)

    assert b["y_past"].shape == (C,)
    assert b["y_future"].shape == (H,)

    C_ck = b["c_ctx_future_known"].shape[1]
    C_cu = b["c_ctx_future_unknown"].shape[1]
    assert b["c_ctx_future_known"].shape == (C, C_ck)
    assert b["c_fct_future_known"].shape == (H, C_ck)
    assert b["c_ctx_future_unknown"].shape == (C, C_cu)
    assert b["flags_ctx_cu_known"].shape == (C, C_cu)

    # unique_id must be an integer index; static[0] matches it (float)
    uid_idx = b["unique_id"]
    assert isinstance(uid_idx, (int, np.integer))

    static = b["static"]
    assert static.ndim == 1 and static.shape[0] >= 1
    assert static.dtype == torch.float32
    assert np.isclose(static[0].item(), float(uid_idx))

    # dataloader yields batches
    batch = next(iter(dm.train_dataloader()))
    for key in ["y_past", "y_future", "c_ctx_future_known", "c_ctx_future_unknown", "flags_ctx_cu_known", "static"]:
        assert key in batch


def test_realistic_masking_applied_in_context(synthetic_csv_file, dataset_params, split_dates):
    C = dataset_params["context_length"]
    cutoff = dataset_params["past_unknown_cov_cutoff"]

    dm = _build_dm(synthetic_csv_file, dataset_params, split_dates, share_scalers=True)
    dm.setup(stage="fit")
    b = dm.train_dataset[1]

    flags = b["flags_ctx_cu_known"].numpy()
    cu    = b["c_ctx_future_unknown"].numpy()

    if flags.shape[1] == 0:
        pytest.skip("No cu features in this setup.")

    assert flags.shape[0] == C
    if cutoff > 0:
        # available part = 1, masked tail = 0
        assert np.all(flags[:-cutoff, :] == 1.0)
        assert np.all(flags[-cutoff:, :] == 0.0)
        # masked cu rows neutralized to 0 (after scaling)
        assert np.allclose(cu[-cutoff:, :], 0.0, atol=1e-6)


def test_setup_test_before_fit_still_populates_test(synthetic_csv_file, dataset_params, split_dates):
    dm = _build_dm(synthetic_csv_file, dataset_params, split_dates, share_scalers=True)
    # Call test first; Base DM will build a temp-train to get scaler states
    dm.setup(stage="test")
    assert dm.test_dataset is not None

    # test loader yields at least one batch
    _ = next(iter(dm.test_dataloader()))


def test_disable_share_scalers_means_no_val_states(synthetic_csv_file, dataset_params, split_dates):
    # With share_scalers=False, val should not receive train's scaler states
    dm = _build_dm(synthetic_csv_file, dataset_params, split_dates, share_scalers=False)
    dm.setup(stage="fit")

    # train may have fitted scalers internally (target/features) …
    t_scaler_train = dm.train_dataset.get_target_scaler()
    f_scalers_train = dm.train_dataset.get_scalers()

    # …but val should *not* have them copied over by the DM
    t_scaler_val = dm.val_dataset.get_target_scaler()
    f_scalers_val = dm.val_dataset.get_scalers()

    # Target scaler: train may be non-None; val should be None if not shared
    # (dataset itself does not fit on val when fit_on="train")
    assert t_scaler_val is None

    # Feature scalers also should not be copied
    assert (f_scalers_val or {}) == {}

    # Sanity: train has *some* states (not strictly required but likely true)
    assert t_scaler_train is not None
    assert isinstance(f_scalers_train, dict)