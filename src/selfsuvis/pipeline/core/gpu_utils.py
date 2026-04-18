"""GPU / device utility functions shared across models and vision pipeline stages.

Centralises three patterns that were previously copy-pasted into every model file:

- :func:`is_cuda_oom` — detect CUDA out-of-memory exceptions
- :func:`resolve_device` — map ``settings.DEVICE`` to ``"cuda" | "mps" | "cpu"``
- :func:`pipeline_device_arg` — convert a device string to the integer HuggingFace
  ``pipeline()`` expects (``-1`` for CPU, ``0`` for CUDA/MPS)
"""

def is_cuda_oom(exc: Exception) -> bool:
    """Return True if *exc* is a CUDA out-of-memory error.

    Works for both PyTorch's ``torch.cuda.OutOfMemoryError`` and the older
    ``RuntimeError`` messages emitted before PyTorch introduced the dedicated
    exception subclass.
    """
    msg = str(exc).lower()
    return type(exc).__name__ == "OutOfMemoryError" or "cuda out of memory" in msg


def resolve_device(device_cfg: str | None = None) -> str:
    """Return the concrete device string to use for model loading.

    Resolution order:
    1. *device_cfg* parameter (falls back to ``settings.DEVICE`` when ``None``)
    2. ``"auto"`` → probe for CUDA, then MPS, then fall back to CPU
    3. Explicit ``"cuda"`` / ``"mps"`` / ``"cpu"`` validated against availability

    Returns one of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if device_cfg is None:
        from selfsuvis.pipeline.core.config import settings
        device_cfg = settings.DEVICE

    cfg = device_cfg.lower()
    try:
        import torch  # type: ignore[import-untyped]

        if cfg == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if cfg == "cuda" and torch.cuda.is_available():
            return "cuda"
        if cfg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def pipeline_device_arg(device: str) -> int:
    """Convert a device string to the integer HuggingFace ``pipeline()`` expects.

    Returns ``-1`` for ``"cpu"`` and ``0`` for all accelerator devices
    (``"cuda"``, ``"mps"``, etc.).
    """
    return -1 if device == "cpu" else 0
