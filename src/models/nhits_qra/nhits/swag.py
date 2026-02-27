# src/models/nhits_qra/nhits/swag.py
from __future__ import annotations
from typing import List
from dataclasses import dataclass, field
import torch
import pytorch_lightning as pl
from contextlib import contextmanager
from utils.logging_utils import get_logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neuralforecast.models.nhits import NHITS

log = get_logger(__name__)

def _trainable_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]

@torch.no_grad()
def flatten_params(model) -> torch.Tensor:
    ps = _trainable_params(model)
    return torch.cat([p.detach().reshape(-1) for p in ps], dim=0)

@torch.no_grad()
def assign_flat_params_(model, vec: torch.Tensor) -> None:
    ps = _trainable_params(model)
    offset = 0
    for p in ps:
        n = p.numel()
        p.copy_(vec[offset:offset+n].view_as(p))
        offset += n
    if offset != vec.numel():
        raise RuntimeError(f"assign_flat_params_: vec has {vec.numel()} elems but used {offset}")

@dataclass
class SWAGState:
    max_rank: int = 10
    n_snapshots: int = 0
    mean: torch.Tensor = None      # (P,)
    sq_mean: torch.Tensor = None   # (P,)
    deviations: List[torch.Tensor] = field(default_factory=list)  # list of (P,)

    @torch.no_grad()
    def update(self, w: torch.Tensor):

        w = w.detach().cpu()

        if self.mean is None:
            self.mean = w.clone()
            self.sq_mean = (w * w).clone()
            self.n_snapshots = 1
            return

        n = self.n_snapshots
        # running mean / second moment
        self.mean = (n * self.mean + w) / (n + 1)
        self.sq_mean = (n * self.sq_mean + w * w) / (n + 1)
        self.n_snapshots += 1

        # low-rank deviation cache (full SWAG)
        dev = (w - self.mean)
        self.deviations.append(dev)
        if len(self.deviations) > self.max_rank:
            self.deviations.pop(0)

    @torch.no_grad()
    def sample(
        self,
        *,
        scale: float = 1.0,
        var_clamp: float = 1e-30,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if self.mean is None or self.n_snapshots < 2:
            raise RuntimeError("SWAGState.sample: need >=2 snapshots.")

        # choose device (default: CPU, but we can pass GPU)
        if device is None:
            device = self.mean.device  # typically CPU since storing CPU snapshots

        mean = self.mean.to(device)
        var = (self.sq_mean.to(device) - mean * mean).clamp_min(var_clamp)
        std = var.sqrt()

        eps = torch.randn_like(mean)  # now on chosen device
        w = mean + std * eps          # diag part

        K = len(self.deviations)
        if K >= 2:
            D = torch.stack(self.deviations, dim=0).to(device)  # (K,P) on device
            z = torch.randn(K, device=device)                   # on device
            w = w + (D.t() @ z) / (K - 1) ** 0.5

        return mean + scale * (w - mean)
    

class SWAGCallback(pl.Callback):
    """
    Callback function for PL trainer, that let's an NHITS model gather weights for SWAG (Maddox et al, 2019)
    """
    def __init__(self, 
    *, 
    start_epoch: int, 
    collect_every: int, 
    max_rank: int):
        
        self.start_epoch = int(start_epoch)
        self.collect_every = int(collect_every)
        self.max_rank = int(max_rank)


    def on_fit_start(self, trainer, pl_module):
        if getattr(pl_module, "swag", None) is None:
            pl_module.swag = SWAGState(max_rank=self.max_rank)

    @torch.no_grad()
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        ep = int(trainer.current_epoch)
        if ep < self.start_epoch:
            return
        if (ep - self.start_epoch) % self.collect_every != 0:
            return

        w = flatten_params(pl_module.model)
        pl_module.swag.update(w)
        log.info("SWAG: collected snapshot %d (rank_cache=%d)",
                 pl_module.swag.n_snapshots, len(pl_module.swag.deviations))
        

@dataclass
class SWAGSession:
    """
    Creates a SWAG Session, i.e. backup model weights, sample weights and assigns them to the model.
    """
    model: torch.nn.Module
    swag: SWAGState
    var_clamp: float = 1e-30
    scale: float = 1.0


    def __post_init__(self):
        self.ps = _trainable_params(self.model)
        self.device = next(self.model.parameters()).device
        self.w0 = torch.cat([p.detach().reshape(-1) for p in self.ps], dim=0).to("cpu")  # keep backup on CPU

    @torch.no_grad()
    def _assign_(self, vec: torch.Tensor) -> None:
        vec = vec.to(self.device, non_blocking=True)
        offset = 0
        for p in self.ps:
            n = p.numel()
            p.copy_(vec[offset:offset+n].view_as(p))
            offset += n
        if offset != vec.numel():
            raise RuntimeError(f"assign: vec has {vec.numel()} elems but used {offset}")

    @torch.no_grad()
    def sample_and_assign_(self) -> None:
        wS = self.swag.sample(scale=self.scale, var_clamp=self.var_clamp, device=self.device)   # (P,) on CPU
        self._assign_(wS)

    @torch.no_grad()
    def restore_(self) -> None:
        self._assign_(self.w0)

@contextmanager
def swag_sampling(model: NHITS, swag: SWAGState, *, var_clamp: float = 1e-30, scale: float=1.0):
    """
    Docstring for swag_sampling
    
    :param model: NHITS model
    :type model: NHITS
    :param swag: Swagstate Object
    :type swag: SWAGState
    :param scale: Scale factor to 
    :type scale: float
    """
    sess = SWAGSession(model=model, swag=swag, scale=scale, var_clamp=var_clamp)
    try:
        yield sess
    finally:
        sess.restore_()