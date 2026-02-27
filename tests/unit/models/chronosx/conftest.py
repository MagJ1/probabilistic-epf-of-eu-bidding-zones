import pytest
import torch.nn as nn
from chronos import ChronosConfig


@pytest.fixture()
def test_params():
    return {
        "batch_size": 4,
        "context_size": 512,
        "horizon": 64,
        "emb_dim": 512,
        "cov_dim": 8,
        "hidden_dim": 512,
        "low_limit": -15.0,
        "high_limit": 15.0,
        "n_tokens": 4096,
        "n_special_tokens": 2,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "use_eos_token": True,
        "num_samples": 20,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "csv_path": "tests/data/train_with_all_min_example.csv",
    }

@pytest.fixture(params=[True, False], ids=["eos_on", "eos_off"])
def chronos_config(request):
    use_eos_token = request.param
    return ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        context_length=512,
        prediction_length=64,
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=use_eos_token,
        model_type="seq2seq",
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )