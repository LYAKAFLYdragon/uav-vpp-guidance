"""
PyTorch device resolution utilities.

Provides safe CPU/CUDA device selection with explicit fallback or strict-raise
semantics, so that checkpoint loading and inference never silently land on
the wrong device.
"""

import warnings

import torch


def resolve_torch_device(device_str: str, allow_fallback: bool = True):
    """Resolve a device string to a torch.device.

    Args:
        device_str (str): Requested device, e.g. "cpu", "cuda", "cuda:0".
        allow_fallback (bool): If True and CUDA is unavailable, warn and
            return CPU. If False and CUDA is unavailable, raise RuntimeError.

    Returns:
        torch.device

    Raises:
        RuntimeError: If allow_fallback=False and the requested CUDA device
            is not available.
    """
    requested = torch.device(device_str)

    if requested.type == "cpu":
        return requested

    if requested.type == "cuda":
        if not torch.cuda.is_available():
            if allow_fallback:
                warnings.warn(
                    "CUDA requested but not available. "
                    "Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return torch.device("cpu")
            raise RuntimeError(
                f"CUDA device '{device_str}' requested but torch.cuda.is_available() "
                f"is False. Set allow_fallback=True to fall back to CPU, or "
                f"install a CUDA-capable PyTorch build."
            )
        return requested

    # Unknown device type (e.g. 'mps', 'xpu') – let PyTorch validate it
    return requested


def load_checkpoint_to_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    device_str: str = "cpu",
    allow_device_fallback: bool = True,
    strict: bool = True,
) -> None:
    """Load a checkpoint into a model with safe device handling.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the .pt or .pth checkpoint.
        device_str (str): Target device for loading.
        allow_device_fallback (bool): If True, silently fall back to CPU when
            CUDA is unavailable.
        strict (bool): Passed to model.load_state_dict().

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        RuntimeError: If allow_device_fallback=False and CUDA unavailable.
        RuntimeError: If state_dict keys mismatch (when strict=True).
    """
    if not checkpoint_path or not isinstance(checkpoint_path, str):
        raise ValueError(f"Invalid checkpoint_path: {checkpoint_path}")

    device = resolve_torch_device(device_str, allow_fallback=allow_device_fallback)

    try:
        # weights_only=True is preferred for security but only available
        # in PyTorch >= 2.0. Fall back gracefully for older versions.
        try:
            state_dict = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
        except TypeError:
            # Older PyTorch without weights_only support
            state_dict = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Please train the model first or correct the path."
        ) from None

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
