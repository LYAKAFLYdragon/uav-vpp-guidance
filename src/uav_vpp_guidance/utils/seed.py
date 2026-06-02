"""
Random seed setting utility.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: make CUDA operations deterministic (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
