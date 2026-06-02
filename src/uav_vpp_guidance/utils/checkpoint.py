"""
Checkpoint saving and loading utilities.
"""

import os
import torch


def save_checkpoint(state_dict, path, metadata=None):
    """
    Save a checkpoint dictionary.

    Args:
        state_dict (dict): Model and training state.
        path (str): Save path.
        metadata (dict, optional): Additional metadata to store.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {"state_dict": state_dict}
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    """
    Load a checkpoint dictionary.

    Args:
        path (str): Checkpoint path.
        device (str): Device to map tensors to.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)
