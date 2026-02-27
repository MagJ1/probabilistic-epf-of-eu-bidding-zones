import torch
import torch.testing as tt

def test_flow_roundtrip(ff_model, fake_batch, params):
    ctx_full = ff_model.make_ctx(fake_batch, teacher_force=False)
    ctx      = ctx_full[:, -1, :] 
    z  = torch.randn(params["B"], params["Tpred"])            
    x   = ff_model.flow.forward(z, ctx)  # z -> price
    z2  = ff_model.flow.inverse(x, ctx)  # price -> z
    tt.assert_close(z, z2, atol=1e-5, rtol=1e-5)