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

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import get_logger, is_cuda_oom, pipeline_device_arg, resolve_device, settings
from selfsuvis.pipeline.vision._quiet import suppress_runtime_noise

from ._pipe_mixin import _HFPipeMixin
from .registry import resolve_model_id

logger = get_logger(__name__)


def _resolve_model_id() -> str:
    cfg = (settings.DEPTH_MODEL or "").strip()
    if cfg and cfg.lower() != "auto":
        return cfg
    if getattr(settings, "DEPTH_AUTO_PROFILE", "fast").lower() == "fast":
        # Local pipeline stores only 5-bucket normalized depth percentiles, so
        # a compact model is a better default than DepthPro's higher-cost metric depth.
        return "depth-anything/Depth-Anything-V2-Base-hf"
    return resolve_model_id(settings.DEPTH_MODEL, "depth", "depth-anything/Depth-Anything-V2-Small-hf")


def _prepare_depth_image(image: Image.Image) -> Image.Image:
    """Downscale oversized frames before depth inference.

    Percentile summaries are robust to moderate resizing, and reducing spatial
    resolution materially improves throughput for local depth estimation.
    """
    max_side = max(0, int(getattr(settings, "DEPTH_IMAGE_MAX_SIDE", 0) or 0))
    if not max_side or max(image.size) <= max_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


class DepthModel(_HFPipeMixin):
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
        images = [_prepare_depth_image(img) for img in images]
        pipe = self._get_pipe()
        if pipe is None:
            return [{"depth_unavailable": True}] * len(images)
        return self._estimate_many(images, pipe)

    def estimate(self, image: Image.Image) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"depth_disabled": True}
        image = _prepare_depth_image(image)
        pipe = self._get_pipe()
        if pipe is None:
            return {"depth_unavailable": True}
        return self._estimate_one(image, pipe)

    def estimate_dense(self, image: Image.Image) -> Dict[str, Any]:
        """Return a dense depth map payload when enabled.

        For pipelines that only expose image-space relative depth, the returned
        values are normalized to [0, 1]. Confidence is a placeholder uniform map
        until a model-specific uncertainty head is available.
        """
        if not self.is_enabled():
            return {"depth_disabled": True}
        image = _prepare_depth_image(image)
        pipe = self._get_pipe()
        if pipe is None:
            return {"depth_unavailable": True}
        try:
            with suppress_runtime_noise(
                r"You seem to be using the pipelines sequentially on GPU.*",
                logger_levels={
                    "transformers": logging.ERROR,
                    "transformers.pipelines.base": logging.ERROR,
                },
            ):
                raw = pipe(image)
            return self._dense_depth_output(raw)
        except Exception as exc:
            if is_cuda_oom(exc) and self._device == "cuda":
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner

                log_oom_banner(logger, f"Depth/{self.model_id}", "dense depth OOM — falling back to CPU")
                cpu_pipe = self._fallback_to_cpu()
                if cpu_pipe is not None:
                    return self.estimate_dense(image)
            logger.warning("Dense depth estimation failed", exc_info=True)
            return {"depth_error": True}

    def _estimate_one(self, image: Image.Image, pipe) -> Dict[str, Any]:
        try:
            with suppress_runtime_noise(
                r"You seem to be using the pipelines sequentially on GPU.*",
                logger_levels={
                    "transformers": logging.ERROR,
                    "transformers.pipelines.base": logging.ERROR,
                },
            ):
                return self._normalise_depth_output(pipe(image))
        except Exception as exc:
            if is_cuda_oom(exc) and self._device == "cuda":
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner
                log_oom_banner(logger, f"Depth/{self.model_id}", "falling back to CPU for remaining frames")
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
            with suppress_runtime_noise(
                r"You seem to be using the pipelines sequentially on GPU.*",
                logger_levels={
                    "transformers": logging.ERROR,
                    "transformers.pipelines.base": logging.ERROR,
                },
            ):
                raw_outputs = pipe(images)
            if isinstance(raw_outputs, list) and len(raw_outputs) == len(images):
                return [self._normalise_depth_output(output) for output in raw_outputs]
        except Exception as exc:
            if is_cuda_oom(exc) and self._device == "cuda":
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner
                log_oom_banner(logger, f"Depth/{self.model_id}", "batch OOM — falling back to CPU")
                cpu_pipe = self._fallback_to_cpu()
                if cpu_pipe is not None:
                    return self._estimate_many(images, cpu_pipe)
            logger.debug("Depth batch inference failed; falling back to per-image processing", exc_info=True)
        return [self._estimate_one(image, pipe) for image in images]

    def _normalise_depth_output(self, output: Any) -> Dict[str, Any]:
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

    def _dense_depth_output(self, output: Any) -> Dict[str, Any]:
        depth_img = output.get("depth") if isinstance(output, dict) else output
        if depth_img is None:
            return {"depth_unavailable": True}
        depth_arr = np.array(depth_img).astype(np.float32)
        dmin, dmax = float(depth_arr.min()), float(depth_arr.max())
        if dmax > dmin:
            depth_arr = (depth_arr - dmin) / (dmax - dmin)
        confidence = np.ones_like(depth_arr, dtype=np.float32)
        return {
            "depth_dense": {
                "map": depth_arr,
                "confidence": confidence,
                "model": self.model_id,
            },
            "depth": {
                "percentiles": [round(p, 4) for p in np.percentile(depth_arr, [10, 25, 50, 75, 90]).tolist()],
                "model": self.model_id,
            },
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
            with suppress_runtime_noise(
                r".*Loading weights.*",
                logger_levels={
                    "transformers": logging.ERROR,
                    "transformers.pipelines.base": logging.ERROR,
                    "huggingface_hub": logging.ERROR,
                },
            ):
                self._pipe = hf_pipeline(
                    "depth-estimation",
                    model=self.model_id,
                    device=pipeline_device_arg(target_device),
                    dtype=torch_dtype,
                )
            self._device = target_device
            logger.info("Depth model loaded: %s on %s", self.model_id, target_device)
        except Exception as exc:
            if is_cuda_oom(exc) and target_device == "cuda" and force_device != "cpu":
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner
                log_oom_banner(logger, f"Depth/{self.model_id}", "load OOM on CUDA — retrying on CPU")
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

def _get_device() -> str:
    return resolve_device()
