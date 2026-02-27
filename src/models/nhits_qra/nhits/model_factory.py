# models/nhits_qra/model_factory.py
from __future__ import annotations
import torch.nn as nn
from neuralforecast.models.nhits import NHITS

def build_nhits_model(
    input_size: int,
    h: int,
    n_blocks,
    mlp_units,
    dropout_prob_theta: float = 0.1,
    # Lists are for NHITS BaseModel to compute exog sizes
    hist_exog_list=None,   # ck + cu (history exog)
    futr_exog_list=None,   # ck (future-known)
    stat_exog_list=None,   # static (e.g., unique_id)
    **kwargs
) -> nn.Module:
    L = len(n_blocks)
    assert len(mlp_units) == L, "len(mlp_units) must equal len(n_blocks)"
    if "n_pool_kernel_size" in kwargs:
        assert len(kwargs["n_pool_kernel_size"]) == L
    if "n_freq_downsample" in kwargs:
        assert len(kwargs["n_freq_downsample"]) == L
    # Pass the *lists*, not counts
    model = NHITS(
        input_size=input_size,
        h=h,
        stack_types=["identity"] * L,
        n_stacks=L,
        n_blocks=n_blocks,
        mlp_units=mlp_units,
        dropout_prob_theta=dropout_prob_theta,
        hist_exog_list=list(hist_exog_list or []),
        futr_exog_list=list(futr_exog_list or []),
        stat_exog_list=list(stat_exog_list or []),
        **kwargs,
    )

    return model