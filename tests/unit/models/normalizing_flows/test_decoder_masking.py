
def test_zero_mask_decoder_input_teacher_force_false(ff_model, fake_batch):
    # (B,forecast_horizon,Din_dec)
    dec_in = ff_model._build_decoder_input(fake_batch, teacher_force=False)
    Ck = ff_model.c_future_known      # future-known channels
    Cu = ff_model.c_future_unknown    # future-unknown channels

    # --- 1. target column (index 0) must be zero --------------------------
    target_column = dec_in[..., 0]            # shape (B, H)
    assert (target_column == 0).all()

    # --- 2. future-unknown covariate block must be zero -------------------
    unknown_block = dec_in[..., 1 + Ck : 1 + Ck + Cu]   # shape (B, H, Cu)
    assert (unknown_block == 0).all()

def test_zero_mask_decoder_input_teacher_force_true(ff_model, fake_batch):
    # (B,forecast_horizon,Din_dec)
    dec_in = ff_model._build_decoder_input(fake_batch, teacher_force=True)
    Ck = ff_model.c_future_known      # future-known channels
    Cu = ff_model.c_future_unknown    # future-unknown channels

    # --- 1. target column (index 0) must be zero --------------------------
    target_column = dec_in[..., 0]            # shape (B, H)
    assert (target_column != 0).any()

    # --- 2. future-unknown covariate block must be zero -------------------
    unknown_block = dec_in[..., 1 + Ck : 1 + Ck + Cu]   # shape (B, H, Cu)
    assert (unknown_block == 0).all()