# conftest.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# ---------------------------
# Global test parameters
# ---------------------------

@pytest.fixture
def dataset_params():
    # small, fast config
    return dict(
        context_length=8,
        forecast_horizon=4,
        past_unknown_cov_cutoff=3,   # last 3 hours in context are "not yet available"
        date_col="ds",
        id_col="unique_id",
        y_col="y",
        ck_cols=[
            "is_weekend", "is_holiday",
            "hour_sin", "hour_cos"
        ],
        cu_cols=[
            "load", "co2_emission_allowances"
        ],
        static_cols = ["unique_id"],
        # realistic masking on by default for tests
        realistic_mode=True,
        # registry enables per-feature scaling for ck/cu; static for uid
        registry={
            "is_weekend": {"role":"ck","standardize": False},
            "is_holiday": {"role":"ck","standardize": False},
            "hour_sin": {"role":"ck","standardize": False},
            "hour_cos": {"role":"ck","standardize": False},

            "load": {"role":"cu","standardize": True, "scaler":"zscore"},
            "co2_emission_allowances":{"role":"cu","standardize": True, "scaler":"zscore"},

            "unique_id": {"role":"static","standardize": False}
        },
        enable_registry=True,
        fit_on="train",       
        scale_target=True,
    )

@pytest.fixture
def synthetic_csv_file(tmp_path: Path, dataset_params) -> str:
    # build two short series with simple patterns
    ids = ["A", "B"]
    T_total = 40  # enough for several windows
    rows = []
    for uid in ids:
        for t in range(T_total):
            ds = pd.Timestamp("2022-01-01") + pd.Timedelta(hours=t)
            # simple periodic ck
            hour = ds.hour
            rows.append({
                "ds": ds,
                "unique_id": uid,
                "y": (10 if uid=="A" else 20) + 0.5*t,   # linear trend so scaling is well-defined
                "is_weekend": int(ds.weekday()>=5),
                "is_holiday": 0,
                "hour_sin": np.sin(2*np.pi*hour/24),
                "hour_cos": np.cos(2*np.pi*hour/24),
                "load": 100 + (5 if uid=="A" else -3)*t,
                "co2_emission_allowances": 80 + (2 if uid=="A" else 4)*t,
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "synthetic.csv"
    df.to_csv(p, index=False)
    return str(p)

@pytest.fixture
def split_dates():
    # put validation split later in series to keep both train/val non-empty
    return dict(val_split_date=pd.Timestamp("2022-01-01 18:00:00"))