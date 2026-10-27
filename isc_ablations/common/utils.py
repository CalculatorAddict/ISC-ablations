import torch


def set_torch_device() -> torch.device:
    """
    Pick a torch device, preferring GPU backends when available.

    Returns
    -------
    torch.device
        ``cuda`` if available, otherwise ``mps`` on Apple Silicon, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
