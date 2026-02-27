import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict

class ChronosXLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 5e-5):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, input_ids, decoder_input_ids, past_covs, future_covs, labels=None, attention_mask=None):
        return self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            past_covs=past_covs,
            future_covs=future_covs,
            labels=labels,
            attention_mask=attention_mask,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self(**batch)
        self.log("train_loss", output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self(**batch)
        self.log("val_loss", output.loss, prog_bar=True)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)