from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """Set seed for random, numpy, torch (if available), and CUDA (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # CPU VM: torch not required for preprocessing


def make_rng(seed: int) -> np.random.Generator:
    """Return an isolated numpy Generator. Does not affect global numpy state."""
    return np.random.default_rng(seed)
