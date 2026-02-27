# src/models/nhits_qra/qra/qra_calib_chung_model.py
from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

@dataclass
class ChungModelCfg:
    kind: str = "linear"          # "linear" | "mlp"
    hidden: List[int] = None      # e.g. [64, 64]
    dropout: float = 0.0          # e.g. 0.0 or 0.1
    act: str = "relu"             # "relu" | "gelu"

def _act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

class ChungQModel(nn.Module):
    """Model takes [Z, q] and outputs predicted quantile value."""
    def __init__(self, in_dim: int, cfg: ChungModelCfg):
        super().__init__()
        cfg = cfg or ChungModelCfg()
        self.cfg = cfg

        if cfg.kind == "linear":
            self.net = nn.Linear(in_dim, 1, bias=True)
        elif cfg.kind == "mlp":
            hidden = cfg.hidden
            layers = []
            d = in_dim
            for h in hidden:
                layers.append(nn.Linear(d, h))
                layers.append(_act(cfg.act))
                if cfg.dropout and cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown Chung model kind: {cfg.kind}")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.net(inp)