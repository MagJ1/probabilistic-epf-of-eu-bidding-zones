import torch

def test_gradcheck(ff_model, fake_batch):
    ff_model.double()
    batch = {k: v.double() for k, v in fake_batch.items()}
    batch["y_future"].requires_grad_(True)

    # full ctx: (B, forecast_horizon, tf_in_size)
    ctx_full = ff_model.make_ctx(batch, teacher_force=True)

    ctx = ctx_full[:, -1, :] #(B,tf_in_size)

    func = lambda x: ff_model.flow.log_prob(x, context=ctx)
    torch.autograd.gradcheck(func, (batch["y_future"],), eps=1e-4, atol=1e-3)