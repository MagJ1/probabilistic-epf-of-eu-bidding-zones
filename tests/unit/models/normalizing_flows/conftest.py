# tests/conftest.py
import torch
import pytest
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile

from models.normalizing_flows.FlowForecaster import FlowForecaster
from models.normalizing_flows.dataset import FlowForecasterDataset

# tests/helpers.py  (or inline in the test file)
# ---------- global constants for tiny sanity tests ------------
B = 4
Cu, Ck, flag, static   = 1, 2, 1, 1
Tctx     = 8      
Tpred    = 4
tf_in_size  = 16
nf_hidden_dim = 32
dataset_length = 100
dataset_unique_ids = ["A", "B"]
tf_dropout = 0.1
enc_unknown_cutoff = 14
dec_known_past_injection_horizon = 12
realistic_mode = True
lr = 5e-3

# ---------- helper to make parameters variable
@pytest.fixture
def params():
    return(dict(
    B = B,
    Cu = Cu, 
    Ck = Ck,
    flag = flag,
    static = static,
    Tctx = Tctx,      
    Tpred = Tpred,
    tf_in_size = tf_in_size,
    nf_hidden_dim = nf_hidden_dim,
    dataset_lenght = (dataset_length-Tctx-Tpred) * len(dataset_unique_ids),
    dataset_unique_ids = dataset_unique_ids,
    tf_dropout = tf_dropout, 
    enc_unknown_cutoff = enc_unknown_cutoff, 
    realistic_mode = realistic_mode,
    lr = lr
    ))

@pytest.fixture
def col_features():
    return (dict(
        ds = "ds",
        unique_id = "unique_id", 
        y = "y", 
        day_of_week = "day_of_week", 
        month = "month",
        hour = "hour", 
        is_weekend = "is_weekend", 
        is_holiday = "is_holiday", 
        month_sin = "month_sin", 
        month_cos = "month_cos", 
        day_of_week_sin = "day_of_week_sin", 
        day_of_week_cos = "day_of_week_cos", 
        hour_sin = "hour_sin", 
        hour_cos = "hour_cos", 
        gas_price = "gas_price", 
        cross_border_trading = "cross_border_trading", 
        non_renewable = "non_renewable", 
        renewable = "renewable", 
        load = "load", 
        co2_emission_allowances = "co2_emission_allowances", 
        synthetic_price = "synthetic_price"
    ))


class SingleSampleDS(torch.utils.data.Dataset):
    """
    Wraps a pre-batched dict (shape: (B,…)) and returns one sample per __getitem__.
    """
    def __init__(self, batched_dict, repeat=32):
        self.data   = batched_dict
        self.batch_size = next(iter(batched_dict.values())).shape[0]
        self.repeat = repeat

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        i = idx % self.batch_size                     # recycle indices
        return {k: v[i] for k, v in self.data.items()}
    
@pytest.fixture
def SingleSampleDSFixture():
    return SingleSampleDS


# ---------- reusable fake batch fixture -----------------------
@pytest.fixture
def fake_batch():
    return {
        "y_past": torch.randn(B, Tctx),
        "c_ctx_future_unknown": torch.randn(B, Tctx, Cu),
        "c_ctx_future_known": torch.randn(B, Tctx, Ck),
        "c_fct_future_known": torch.randn(B, Tpred, Ck),
        "y_future": torch.randn(B, Tpred),
        "flag": torch.randn(B, flag),
        "static": torch.randn(B, static)
    }

# ---------- Lightning model fixture ---------------------------
@pytest.fixture
def ff_model():
    return FlowForecaster(
        tf_in_size        = tf_in_size,
        nf_hidden_dim = nf_hidden_dim,
        n_layers       = 2,
        n_heads        = 2,
        n_flow_layers    = 2,
        tf_dropout = tf_dropout,
        c_future_unknown   = Cu,
        c_future_known = Ck,
        context_length = Tctx,
        forecast_horizon = Tpred,
        enc_unknown_cutoff= enc_unknown_cutoff,
        dec_known_past_injection_horizon=dec_known_past_injection_horizon,
        lr = lr
    )


@pytest.fixture
def FlowForecasterDatasetFixture():
    return FlowForecasterDataset


@pytest.fixture
def synthetic_csv_file(col_features):
    """Generates a minimal synthetic CSV file emulating realistic time series data."""
    rows = []

    for uid in dataset_unique_ids:
        for t in range(dataset_length):
            ds = pd.Timestamp("2022-01-01") + pd.Timedelta(hours=t)
            rows.append({
                col_features["ds"]: ds,
                col_features["unique_id"]: uid,
                col_features["y"]: 10 + t if uid == "A" else 20 + t,
                col_features["day_of_week"]: ds.dayofweek,
                col_features["month"]: ds.month,
                col_features["hour"]: ds.hour,
                col_features["is_weekend"]: int(ds.weekday() >= 5),
                col_features["is_holiday"]: 0,  # no holidays in this example
                col_features["month_sin"]: np.sin(2 * np.pi * ds.month / 12),
                col_features["month_cos"]: np.cos(2 * np.pi * ds.month / 12),
                col_features["day_of_week_sin"]: np.sin(2 * np.pi * ds.dayofweek / 7),
                col_features["day_of_week_cos"]: np.cos(2 * np.pi * ds.dayofweek / 7),
                col_features["hour_sin"]: np.sin(2 * np.pi * ds.hour / 24),
                col_features["hour_cos"]: np.cos(2 * np.pi * ds.hour / 24),
                col_features["gas_price"]: 1.0 + 0.1 * t,
                col_features["cross_border_trading"]: 0.5,
                col_features["non_renewable"]: 0.6,
                col_features["renewable"]: 0.3,
                col_features["load"]: 100 + t,
                col_features["co2_emission_allowances"]: 80 + t,
                col_features["synthetic_price"]: 60 + t,
            })

    df = pd.DataFrame(rows)

    with NamedTemporaryFile(delete=False, suffix=".csv", mode='w+') as tmp:
        df.to_csv(tmp.name, index=False)
        yield tmp.name