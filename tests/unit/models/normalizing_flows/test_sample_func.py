import torch
import pytest

def test_sample_inference(ff_model, fake_batch, params):
    """
    Verifies that FlowForecaster.sample
      • returns the correct tensor shape
      • produces finite numbers
      • yields non-deterministic trajectories
    """
    ff_model.eval()                               # no dropout / norm updates
    n_samples = 4

    with torch.no_grad():
        y_hat, log_q = ff_model.sample(fake_batch, n_per_series=n_samples)

    # ---------- 1 : shape check ------------------------------------------
    assert y_hat.shape == (n_samples,
                           params["B"],          # batch size
                           params["Tpred"])      # forecast horizon

    # ---------- 2 : numeric sanity --------------------------------------
    assert torch.isfinite(y_hat).all()

    # ---------- 3 : stochastic sanity -----------------------------------
    # At least two trajectories should differ; this is a cheap proxy that
    # the sampling really draws fresh noise.
    assert not torch.allclose(y_hat[0], y_hat[1])