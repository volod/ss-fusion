"""Shared lifecycle helpers for HuggingFace pipeline-backed vision models.

Both DepthModel and DetectionModel follow the same acquire/release pattern:
  - _release_pipe(): move model to CPU, delete pipeline ref, flush CUDA cache
  - _fallback_to_cpu(): release then reload on CPU (called on OOM)
  - release(): quick public release (no cpu() move, no synchronize)

Subclasses must implement _get_pipe(force_device=None) and expose _pipe,
_device, and _load_failed instance attributes.
"""

import gc


class _HFPipeMixin:
    """Mixin that provides pipe lifecycle methods for HuggingFace pipeline wrappers."""

    def release(self) -> None:
        """Delete the pipeline and flush CUDA cache."""
        self._pipe = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _release_pipe(self) -> None:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if self._pipe is not None:
            model = getattr(self._pipe, "model", None)
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
            del self._pipe
            self._pipe = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

    def _fallback_to_cpu(self):
        self._release_pipe()
        self._load_failed = False
        return self._get_pipe(force_device="cpu")
