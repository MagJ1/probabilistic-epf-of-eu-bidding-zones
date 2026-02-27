def test_context_shape(ff_model, fake_batch, params):
    ctx = ff_model.make_ctx(fake_batch, teacher_force=False)
    B, forecast_horizon, d = ctx.shape
    assert B == params["B"] and forecast_horizon == params["Tpred"] and d == params["tf_in_size"]