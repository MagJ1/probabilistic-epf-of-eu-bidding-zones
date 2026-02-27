import torch.nn as nn
import normflows as nf

# This assumes the ActNorm showed (subclassing AffineConstFlow) is imported as ActNorm
class ActNormNoCtx(nf.flows.Flow):
    """Wrapper so ActNorm works inside Conditional flows that pass `context`."""
    def __init__(self, shape, scale=True, shift=True):
        super().__init__()
        self.act = nf.flows.ActNorm(shape=shape, scale=scale, shift=shift)

    def forward(self, z, context=None):
        # ActNorm ignores context by design
        return self.act.forward(z)

    def inverse(self, z, context=None):
        return self.act.inverse(z)
    
class GaussianMixtureNoCtx(nf.distributions.GaussianMixture):
    """Wraps GaussianMixture so it accepts an unused `context` kwarg."""
    def forward(self, z, context=None):
        # Most NF libs define forward() on distributions to return log_prob.
        return super().forward(z)

    def log_prob(self, z, context=None):
        return super().log_prob(z)

    def sample(self, num_samples, context=None):
        return super().sample(num_samples)