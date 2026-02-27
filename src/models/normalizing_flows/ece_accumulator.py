from dataclasses import dataclass
import torch

def _normalize_taus(taus: torch.Tensor) -> torch.Tensor:
    """
    Accept taus as either probabilities in [0,1] or percentages in [0,100].
    Returns float64 probabilities in [0,1] on CPU.
    """
    if not torch.is_tensor(taus):
        taus = torch.tensor(taus)

    taus = taus.detach().to(device="cpu", dtype=torch.float64).flatten()

    if taus.numel() == 0:
        raise ValueError("taus must not be empty")

    # Heuristic:
    # - if any value > 1, we interpret as percent scale (0..100)
    # - else probability scale (0..1)
    if (taus > 1.0).any():
        taus = taus / 100.0

    # Now enforce range with a helpful message
    if (taus < 0.0).any() or (taus > 1.0).any():
        bad = taus[(taus < 0.0) | (taus > 1.0)]
        raise ValueError(
            f"taus must be probabilities in [0,1] (or percents in [0,100]). "
            f"After normalization, found out-of-range values: {bad.tolist()[:10]}"
        )

    return taus


@dataclass
class ECEAccumulator:
    taus: torch.Tensor   # (T,) on CPU float64
    H: int

    def __post_init__(self):
        self.taus = _normalize_taus(self.taus)  # CPU float64 probs in [0,1]
        T = int(self.taus.numel())
        self.hits = torch.zeros((T, self.H), dtype=torch.float64)  # (T,H) on CPU
        self.ns   = torch.zeros((self.H,), dtype=torch.float64)    # (H,) on CPU

    @torch.no_grad()
    def update(self, samples: torch.Tensor, y_true: torch.Tensor):
        """
        samples: (S,B,H) float32/float16 on device
        y_true:  (B,H)    float32 on device
        """
        if samples.dim() != 3:
            raise ValueError(f"ECE update expects samples (S,B,H), got {tuple(samples.shape)}")
        if y_true.dim() != 2:
            raise ValueError(f"ECE update expects y_true (B,H), got {tuple(y_true.shape)}")
        S, B, H = samples.shape
        if H != self.H:
            raise ValueError(f"ECEAccumulator initialized for H={self.H}, got H={H}")

        taus_dev = self.taus.to(device=samples.device, dtype=samples.dtype)  # (T,)

        # q: (T,B,H) quantiles across S
        q = torch.quantile(samples, taus_dev, dim=0)

        mask = torch.isfinite(y_true)  # (B,H)
        y = y_true.unsqueeze(0)        # (1,B,H)

        # hits: (T,H) = sum_B 1{y <= q} over valid y
        hits_th = ((y <= q) & mask.unsqueeze(0)).sum(dim=1).to(torch.float64)   # (T,H)
        ns_h    = mask.sum(dim=0).to(torch.float64)                             # (H,)

        self.hits += hits_th.detach().cpu()
        self.ns   += ns_h.detach().cpu()

    def compute(self):
        """
        Returns:
          mean_ece: float
          ece_per_h: list[float|None] length H
          cover_th: (T,H) torch float64 coverage (NaN where ns==0)
        """
        T, H = self.hits.shape
        taus = self.taus.to(torch.float64).view(T, 1)  # (T,1)

        cover = torch.full((T, H), float("nan"), dtype=torch.float64)
        valid_h = self.ns > 0
        cover[:, valid_h] = self.hits[:, valid_h] / self.ns[valid_h].view(1, -1)

        ece_per_h = [None] * H
        for h in range(H):
            if self.ns[h] <= 0:
                continue
            ece_h = torch.mean(torch.abs(cover[:, h] - taus[:, 0]))
            ece_per_h[h] = float(ece_h.item())

        vals = [v for v in ece_per_h if v is not None]
        mean_ece = float(sum(vals) / len(vals)) if vals else float("nan")
        return mean_ece, ece_per_h, cover