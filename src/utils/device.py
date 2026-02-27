# utils/helpers.py
import torch

def pick_accelerator_and_devices(use_all_gpus: bool = False):
    """
    Decide runtime device for Trainer and always return map_location='cpu'
    so checkpoints load safely everywhere. Returns:
      (accelerator, devices, precision, map_location)
    """
    # Default: safe universal load target
    map_location = "cpu"

    # CUDA first (HPC)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count() if use_all_gpus else 1
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        precision = "bf16-mixed" if supports_bf16 else "16-mixed"
        return accelerator, devices, precision, map_location

    # Apple Silicon (MPS)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        precision = 32  # keep fp32 on MPS
        # Optional perf hint on newer torch
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
        return accelerator, devices, precision, map_location

    raise ValueError(f"No GPU found!")
    # CPU fallback
    return "cpu", 1, 32, map_location


def accelerator_to_torch_device(accelerator: str) -> str:
    """
    Convert Lightning accelerator naming to torch device naming.
    """
    acc = str(accelerator).lower()
    if acc in {"gpu", "cuda"}:
        return "cuda"
    if acc == "mps":
        return "mps"
    return "cpu"