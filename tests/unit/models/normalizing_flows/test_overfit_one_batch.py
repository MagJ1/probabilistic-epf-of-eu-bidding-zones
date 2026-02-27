import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader

def test_overfit_one_batch(ff_model, fake_batch, SingleSampleDSFixture):
    dataset = SingleSampleDSFixture(fake_batch, repeat=32)
    loader = DataLoader(dataset, batch_size=4,shuffle=False)
    trainer = pl.Trainer(max_epochs=200,
                         logger=False,
                         enable_checkpointing=False,
                         enable_progress_bar=False,
                         overfit_batches=1,
                         accelerator="cpu")
    trainer.fit(ff_model, loader, loader)
    final_nll = trainer.logged_metrics["train_nll"].item()
    assert final_nll < 0.1