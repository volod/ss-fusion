"""Depth estimation model wrapper — DepthAnything-V2, DPT, Apple DepthPro.

Produces a compact depth representation stored in ``frame_facts_json["depth"]``
as ``{"percentiles": [p10, p25, p50, p75, p90], "model": "..."}`` — a 5-bucket
summary that captures relative depth distribution without storing a full map.

Disabled by default (``DEPTH_ENABLED=false``).  Enable with:

    DEPTH_ENABLED=true DEPTH_MODEL=auto python worker/main.py

Top-10 depth models (small → large, override with ``DEPTH_MODEL``):

  1. depth-anything/Depth-Anything-V2-Small-hf   25 M  ~0.05 GB  fastest
  2. depth-anything/Depth-Anything-V2-Base-hf    97 M  ~0.2  GB  good outdoor
  3. vinvino02/glpn-kitti                        85 M  ~0.2  GB  outdoor/KITTI
  4. Intel/dpt-large                            307 M  ~0.6  GB  DPT, solid
  5. depth-anything/Depth-Anything-V2-Large-hf  335 M  ~0.7  GB  best V2 quality
  6. LiheYoung/depth-anything-large-hf          335 M  ~0.7  GB  V1 Large
  7. Intel/zoedepth-nk                          345 M  ~0.7  GB  metric depth
  8. prs-eth/marigold-lcm-v1-0                  859 M  ~1.7  GB  diffusion-based
  9. apple/DepthPro-hf                          1.1 B  ~2.2  GB  metric + focal
 10. geovision-research/DPT-DINOv2-L-384       307 M  ~0.6  GB  DINOv2 backbone

CLI override::

    DEPTH_ENABLED=true DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

from PIL import Image

from selfsuvis.pipeline.core import get_logger, is_cuda_oom, pipeline_device_arg, resolve_device, settings

from .registry import resolve_model_id

logger = get_logger(__name__)


def _resolve_model_id() -> str:
    return resolve_model_id(settings.DEPTH_MODEL, "depth", "depth-anything/Depth-Anything-V2-Small-hf")


class DepthModel:
    """Monocular depth estimation.

    Returns depth percentiles [p10, p25, p50, p75, p90] normalised to [0, 1]
    where 0 = closest and 1 = farthest relative to the frame.
    """

    def __init__(self) -> None:
        self._pipe = None
        self._model_id: Optional[str] = None
        self._load_failed: bool = False
        self._device: Optional[str] = None

    def is_enabled(self) -> bool:
        return settings.DEPTH_ENABLED

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = _resolve_model_id()
        return self._model_id

    def estimate_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Return depth summary dicts for a list of images."""
        if not self.is_enabled():
            return [{"depth_disabled": True}] * len(images)
        pipe = self._get_pipe()
        if pipe is None:
            return [{"depth_unavailable": True}] * len(images)
        return self._estimate_many(images, pipe)

    def estimate(self, image: Image.Image) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"depth_disabled": True}
        pipe = self._get_pipe()
        if pipe is None:
            return {"depth_unavailable": True}
        return self._estimate_one(image, pipe)

    def release(self) -> None:
        """Delete the pipeline and flush CUDA cache."""
        import gc
        self._pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _estimate_one(self, image: Image.Image, pipe) -> Dict[str, Any]:
        try:
            return self._normalise_depth_output(pipe(image))
        except Exception as exc:
            if is_cuda_oom(exc) and self._device == "cuda":
                logger.warning(
                    "Depth CUDA OOM for %s — retrying on CPU for remaining frames.",
                    self.model_id,
                )
                cpu_pipe = self._fallback_to_cpu()
                if cpu_pipe is not None:
                    try:
                        return self._estimate_one(image, cpu_pipe)
                    except Exception:
                        logger.warning("Depth CPU retry failed", exc_info=True)
            logger.warning("Depth estimation failed", exc_info=True)
            return {"depth_error": True}

    def _estimate_many(self, images: List[Image.Image], pipe) -> List[Dict[str, Any]]:
        try:
            raw_outputs = pipe(images)
            if isinstance(raw_outputs, list) and len(raw_outputs) == len(images):
                return [self._normalise_depth_output(output) for output in raw_outputs]
        except Exception as exc:
            if is_cuda_oom(exc) and self._device == "cuda":
                logger.warning(
                    "Depth CUDA OOM for %s — retrying on CPU for remaining frames.",
                    self.model_id,
                )
                cpu_pipe = self._fallback_to_cpu()
                if cpu_pipe is not None:
                    return self._estimate_many(images, cpu_pipe)
            logger.debug("Depth batch inference failed; falling back to per-image processing", exc_info=True)
        return [self._estimate_one(image, pipe) for image in images]

    def _normalise_depth_output(self, output: Any) -> Dict[str, Any]:
        import numpy as np

        # HuggingFace depth-estimation pipeline returns {"depth": PIL.Image, ...}
        depth_img = output.get("depth") if isinstance(output, dict) else output
        if depth_img is None:
            return {"depth_unavailable": True}
        depth_arr = np.array(depth_img).astype(np.float32)
        dmin, dmax = float(depth_arr.min()), float(depth_arr.max())
        if dmax > dmin:
            depth_arr = (depth_arr - dmin) / (dmax - dmin)
        pcts = np.percentile(depth_arr, [10, 25, 50, 75, 90]).tolist()
        return {
            "depth": {
                "percentiles": [round(p, 4) for p in pcts],
                "model": self.model_id,
            }
        }

    def _get_pipe(self, force_device: Optional[str] = None):
        target_device = force_device or self._device or _get_device()
        if self._pipe is not None and self._device == target_device:
            return self._pipe
        if self._load_failed:
            return None
        logger.info("Loading depth model: %s", self.model_id)
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            torch_dtype = torch.float16 if settings.USE_FP16 and target_device != "cpu" else torch.float32
            self._release_pipe()
            self._pipe = hf_pipeline(
                "depth-estimation",
                model=self.model_id,
                device=pipeline_device_arg(target_device),
                torch_dtype=torch_dtype,
            )
            self._device = target_device
            logger.info("Depth model loaded: %s on %s", self.model_id, target_device)
        except Exception as exc:
            if is_cuda_oom(exc) and target_device == "cuda" and force_device != "cpu":
                logger.warning(
                    "Depth model %s failed to load on CUDA due to OOM — retrying on CPU.",
                    self.model_id,
                )
                self._release_pipe()
                self._load_failed = False
                return self._get_pipe(force_device="cpu")
            logger.warning(
                "Failed to load depth model %s — run: python scripts/prepare_models.py --depth",
                self.model_id, exc_info=True,
            )
            self._pipe = None
            self._device = None
            self._load_failed = True
        return self._pipe

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


def _get_device() -> str:
    return resolve_device()
