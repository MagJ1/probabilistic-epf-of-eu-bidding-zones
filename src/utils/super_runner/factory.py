# utils/backends/factory.py
from __future__ import annotations
from .base_backend import Backend

def get_backend(cfg) -> Backend:
    model_type = str(cfg.model.model_type).lower()

    if model_type in ("nf", "normalizing_flows", "normalising_flows"):
        from models.normalizing_flows.nf_backend import NFBackend
        return NFBackend()

    if model_type in ("nhits_qra", "nhits+qra", "nhitsqra", "nq"):
        from models.nhits_qra.nhits_qra_backend import NHITSQRABackend
        return NHITSQRABackend()

    if model_type in ("moirai", "mo"):
        from models.moirai.moirai_backend import MoiraiBackend
        return MoiraiBackend(cfg)

    if model_type in ("chronosx", "chronos", "chronos-x", "cx"):
        from models.chronosx.chronosx_backend import ChronosXBackend
        return ChronosXBackend()

    raise ValueError(f"Unknown model_type for backend: {model_type}")