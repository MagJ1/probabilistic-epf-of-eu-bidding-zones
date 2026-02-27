import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas import Timestamp

def test_flow_forecaster_dataset_length(params, FlowForecasterDatasetFixture, synthetic_csv_file):
    """Test correct length of dataset"""
    dataset = FlowForecasterDatasetFixture(csv_path=synthetic_csv_file, context_length=params["Tctx"], forecast_horizon=params["Tpred"])

    assert len(dataset) == params["dataset_lenght"]

def test_flow_forecaster_dataset_getitem(params, FlowForecasterDatasetFixture, synthetic_csv_file):
    """Test whether correct data is returned"""

    df = pd.read_csv(synthetic_csv_file, parse_dates=["ds"])
    df.sort_values(["unique_id", "ds"], inplace=True)
    dataset = FlowForecasterDatasetFixture(csv_path=synthetic_csv_file, context_length=params["Tctx"], forecast_horizon=params["Tpred"])

    rand_index = np.random.randint(0, params["dataset_lenght"])

    sample = dataset[rand_index]
    sidx, t0 = dataset.indices[rand_index]
    time_series = dataset.ts_collection[sidx]

    expected_y_past = time_series["y"][t0 - dataset.context_length : t0]
    expected_y_fut  = time_series["y"][t0 : t0 + dataset.forecast_horizon]
    expected_ck_past = time_series["ck"][t0 - dataset.context_length : t0]
    expected_cu_past = time_series["cu"][t0 - dataset.context_length : t0]
    expected_ck_fut  = time_series["ck"][t0 : t0 + dataset.forecast_horizon]

    assert torch.allclose(sample["y_past"], torch.tensor(expected_y_past))
    assert torch.allclose(sample["y_past"], torch.tensor(expected_y_past))
    assert torch.allclose(sample["y_future"], torch.tensor(expected_y_fut))
    assert torch.allclose(sample["c_ctx_future_known"], torch.tensor(expected_ck_past))
    assert torch.allclose(sample["c_ctx_future_unknown"], torch.tensor(expected_cu_past))
    assert torch.allclose(sample["c_fct_future_known"], torch.tensor(expected_ck_fut))

def test_dataset_scaler(params, col_features, FlowForecasterDatasetFixture, synthetic_csv_file):
    """Test whether correct data is returned"""

    df = pd.read_csv(synthetic_csv_file, parse_dates=["ds"])
    df.sort_values(["unique_id", "ds"], inplace=True)

    y = [col_features["y"]]
    ck = [col_features["is_weekend"], col_features["is_holiday"]]
    cu = [col_features["load"], col_features["renewable"]]
    static = [col_features["unique_id"]]
    cols = y + ck + cu + static
    scalers = {}

    dataset_unscaled = FlowForecasterDatasetFixture(
        csv_path=synthetic_csv_file, 
        context_length=params["Tctx"], 
        forecast_horizon=params["Tpred"], 
        ck_cols=ck, 
        cu_cols=cu,
        static_cols=static, 
        scale_data=False, 
        scalers=None, 
        scale_cols=None)

    for col in cu:
        v = dataset_unscaled.get_col(col).reshape(-1, 1)   # (N,1)
        v = v[np.isfinite(v[:, 0])]                # (N_masked,1)
        scalers[col] = StandardScaler().fit(v)

    dataset_scaled = FlowForecasterDatasetFixture(
        csv_path=synthetic_csv_file, 
        context_length=params["Tctx"], 
        forecast_horizon=params["Tpred"], 
        ck_cols=ck, 
        cu_cols=cu, 
        static_cols=static,
        scale_data=True, 
        scalers=scalers, 
        scale_cols=cu)


    rand_index = np.random.randint(0, len(dataset_scaled))
    sample = dataset_scaled[rand_index]

    sidx, t0 = dataset_scaled.indices[rand_index]
    time_series = dataset_scaled.ts_collection[sidx]

    expected = {}
    cu_past = time_series["cu"][t0-dataset_scaled.context_length:t0].copy()
    for col in cols:
        if col in cu+y:
            _, j = dataset_scaled.feature_map[col]
            expected[col] = scalers[col].transform(cu_past[:, j:j+1])
        else:
            expected[col] = dataset_scaled.get_col(col)
        
    # TODO: Make assert. Probably best to change NF model, away from combining features to certain keys, like cu, ck and so on. 
    for col in cols:
        assert torch.allclose(sample[col], torch.tensor(expected[col], dtype=torch.float32).squeeze(), atol=1e-5)


def test_dataset_train_val_split(params, FlowForecasterDatasetFixture, synthetic_csv_file):
    split_date = pd.Timestamp("2022-01-01 03:00:00")

    train_ds = FlowForecasterDatasetFixture(
        csv_path=synthetic_csv_file,
        context_length=params["Tctx"],
        forecast_horizon=params["Tpred"],
        val_split_date=split_date,
        split="train"
    )

    val_ds = FlowForecasterDatasetFixture(
        csv_path=synthetic_csv_file,
        context_length=params["Tctx"],
        forecast_horizon=params["Tpred"],
        val_split_date=split_date,
        split="val"
    )
    # Check all training samples are before the split date
    for sidx, t0 in train_ds.indices:
        ts = train_ds.ts_collection[sidx]
        assert ts["ds"][t0] < split_date, f"Train sample at or after split: {ts['ds'][t0]}"

    # Check all validation samples are on or after the split date
    for sidx, t0 in val_ds.indices:
        ts = val_ds.ts_collection[sidx]
        assert ts["ds"][t0] >= split_date, f"Val sample before split: {ts['ds'][t0]}"







    
