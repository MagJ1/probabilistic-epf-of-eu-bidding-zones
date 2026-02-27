# tests/unit/tokenizer/test_chronos_tokenizer.py

import torch
import pytest


def test_create_tokenizer(chronos_config):
    tokenizer = chronos_config.create_tokenizer()
    
    assert tokenizer is not None
    assert hasattr(tokenizer, "context_input_transform")

    # dummy batch (1 sequence of 5 steps, 2 nans for padding)
    x = torch.tensor([[1.0, 2.0, 3.0, float('nan'), float('nan')]])

    token_ids, attention_mask, scale = tokenizer.context_input_transform(x)

    assert token_ids.shape == attention_mask.shape
    assert scale.shape[0] == x.shape[0]  # one scale per batch element