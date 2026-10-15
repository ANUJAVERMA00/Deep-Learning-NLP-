import os
import random
import numpy as np
import torch

# Let unsupported ops fall back to CPU instead of crashing on Apple GPUs
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
