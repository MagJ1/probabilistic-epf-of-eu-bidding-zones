import os, random
import numpy as np
import torch
import pytorch_lightning as pl

def seed_everything_hard(seed: int, deterministic: bool = False) -> None:

    seed = int(seed)

    # 1) Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # 2) NumPy + PL
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)

    # 3) PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4) Make CuDNN / algorithms deterministic (CUDA only)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Will error if hit a non-deterministic op (good for debugging)
        torch.use_deterministic_algorithms(True)

        # Required for determinism for some CUDA ops (matmul etc.)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Optional: reduce CPU thread nondeterminism
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

def seed_worker(worker_id: int) -> None:
    # torch initial seed is already set per-worker by DataLoader;
    # we reuse it to seed numpy/random deterministically.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)