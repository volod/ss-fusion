"""Object detection model wrapper — RT-DETR, Grounding DINO, OmDet-Turbo.

Detects objects in frame images and stores the result in
``frame_facts_json["detections"]`` as a list of
``{"label": str, "confidence": float, "bbox_norm": [x1, y1, x2, y2]}`` dicts.
Bounding boxes are normalised to [0, 1] relative to image width/height.

Disabled by default (``DETECTION_ENABLED=false``).  Enable with:

    DETECTION_ENABLED=true DETECTION_MODEL=auto python worker/main.py

Top-10 detection models (small → large, override with ``DETECTION_MODEL``):

  1. facebook/detr-resnet-50              41 M  ~0.1 GB  classic COCO, fast
  2. PekingU/rtdetr_r50vd                 42 M  ~0.1 GB  RT-DETR, 108 FPS on T4
  3. PekingU/rtdetr_r101vd                76 M  ~0.2 GB  RT-DETR larger backbone
  4. omlab/omdet-turbo-swin-tiny-hf      108 M  ~0.3 GB  open-vocabulary
  5. omlab/omdet-turbo-swin-large-hf     218 M  ~0.5 GB  open-vocab, best speed
  6. IDEA-Research/grounding-dino-tiny   173 M  ~0.4 GB  text-guided, zero-shot
  7. IDEA-Research/grounding-dino-base   341 M  ~0.7 GB  text-guided, stronger
  8. microsoft/conditional-detr-resnet-101 62 M ~0.2 GB  faster convergence
  9. jozhang97/deta-swin-large           218 M  ~0.5 GB  63.5 COCO AP
 10. SenseTime/deformable-detr            40 M  ~0.1 GB  sparse attention DETR

CLI override::

    DETECTION_ENABLED=true DETECTION_MODEL=IDEA-Research/grounding-dino-tiny
    DETECTION_ENABLED=true DETECTION_LABELS="vehicle,person,weapon,infrastructure"
"""

import contextlib
import io
import warnings
from typing import Any, Dict, List, Optional

from PIL import Image

from selfsuvis.pipeline.core import get_logger, is_cuda_oom, pipeline_device_arg, resolve_device, settings

from ._pipe_mixin import _HFPipeMixin
from .registry import resolve_model_id

logger = get_logger(__name__)


def _resolve_model_id() -> str:
    return resolve_model_id(settings.DETECTION_MODEL, "detection", "PekingU/rtdetr_r50vd")


class DetectionModel(_HFPipeMixin):
    """Object detector wrapping HuggingFace object-detection pipelines.

    For open-vocabulary models (Grounding DINO, OmDet), also accepts a
    ``candidate_labels`` list for zero-shot detection.
    """

    def __init__(self) -> None:
        self._pipe = None
        self._model_id: Optional[str] = None
        self._load_failed: bool = False
        self._device: Optional[str] = None

    def is_enabled(self) -> bool:
        return settings.DETECTION_ENABLED

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = _resolve_model_id()
        return self._model_id

    def detect_batch(
        self,
        images: List[Image.Image],
        candidate_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run detection on a batch of images. Returns one result dict per image."""
        if not self.is_enabled():
            return [{"detection_disabled": True}] * len(images)
        pipe = self._get_pipe()
        if pipe is None:
            return [{"detection_unavailable": True}] * len(images)
        return self._detect_many(images, pipe, candidate_labels)

    def detect(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"detection_disabled": True}
        pipe = self._get_pipe()
        if pipe is None:
            return {"detection_unavailable": True}
        return self._detect_one(image, pipe, candidate_labels)

    def _detect_one(
        self,
        image: Image.Image,
        pipe,
        candidate_labels: Optional[List[str]],
    ) -> Dict[str, Any]:
        kwargs = self._detect_kwargs(candidate_labels)
        try:
            if hasattr(pipe, "call_count"):
                pipe.call_count = 0
            return self._normalise_detection_output(pipe(image, **kwargs), image)
        except Exception as exc:
            if is_cuda_oom(exc) and self._device == "cuda":
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner
                log_oom_banner(logger, f"Detection/{self.model_id}", "falling back to CPU for remaining frames")
                cpu_pipe = self._fallback_to_cpu()
                if cpu_pipe is not None:
                    try:
                        return self._detect_one(image, cpu_pipe, candidate_labels)
                    except Exception:
                        logger.warning("Detection CPU retry failed", exc_info=True)
            logger.warning("Detection failed", exc_info=True)
            return {"detection_error": True}

    def _detect_many(
        self,
        images: List[Image.Image],
        pipe,
        candidate_labels: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        kwargs = self._detect_kwargs(candidate_labels)
        try:
            if hasattr(pipe, "call_count"):
                pipe.call_count = 0
            raw_outputs = pipe(images, **kwargs)
            if isinstance(raw_outputs, list) and len(raw_outputs) == len(images):
                return [
                    self._normalise_detection_output(raw, image)
                    for raw, image in zip(raw_outputs, images)
                ]
        except Exception as exc:
            if is_cuda_oom(exc) and self._device == "cuda":
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner
                log_oom_banner(logger, f"Detection/{self.model_id}", "batch OOM — falling back to CPU")
                cpu_pipe = self._fallback_to_cpu()
                if cpu_pipe is not None:
                    return self._detect_many(images, cpu_pipe, candidate_labels)
            logger.debug(
                "Detection batch inference failed; falling back to per-image processing",
                exc_info=True,
            )
        return [self._detect_one(image, pipe, candidate_labels) for image in images]

    def _detect_kwargs(self, candidate_labels: Optional[List[str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"threshold": settings.DETECTION_CONFIDENCE}
        if candidate_labels is not None:
            kwargs["candidate_labels"] = candidate_labels
        elif settings.DETECTION_LABELS:
            kwargs["candidate_labels"] = [
                lbl.strip() for lbl in settings.DETECTION_LABELS.split(",") if lbl.strip()
            ]
        return kwargs

    def _normalise_detection_output(self, raw: Any, image: Image.Image) -> Dict[str, Any]:
        if not isinstance(raw, list):
            raw = []
        w, h = image.size
        detections = []
        for det in raw:
            box = det.get("box", {})
            x1 = box.get("xmin", 0) / w
            y1 = box.get("ymin", 0) / h
            x2 = box.get("xmax", w) / w
            y2 = box.get("ymax", h) / h
            detections.append({
                "label": det.get("label", ""),
                "confidence": round(float(det.get("score", 0.0)), 4),
                "bbox_norm": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            })
        return {"detections": detections, "detection_model": self.model_id}

    def _get_pipe(self, force_device: Optional[str] = None):
        target_device = force_device or self._device or _get_device()
        if self._pipe is not None and self._device == target_device:
            return self._pipe
        if self._load_failed:
            return None
        logger.info("Loading detection model: %s", self.model_id)
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            torch_dtype = torch.float16 if settings.USE_FP16 and target_device != "cpu" else torch.float32
            self._release_pipe()
            with _suppress_hf_detection_noise():
                self._pipe = hf_pipeline(
                    "object-detection",
                    model=self.model_id,
                    device=pipeline_device_arg(target_device),
                    torch_dtype=torch_dtype,
                    use_fast=True,
                    model_kwargs={"low_cpu_mem_usage": False},
                )
            self._device = target_device
            if hasattr(self._pipe, "call_count"):
                self._pipe.call_count = 0
            logger.info("Detection model loaded: %s on %s", self.model_id, target_device)
        except Exception as exc:
            if is_cuda_oom(exc) and target_device == "cuda" and force_device != "cpu":
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner
                log_oom_banner(logger, f"Detection/{self.model_id}", "load OOM on CUDA — retrying on CPU")
                self._release_pipe()
                self._load_failed = False
                return self._get_pipe(force_device="cpu")
            logger.warning(
                "Failed to load detection model %s — run: python scripts/prepare_models.py --detection",
                self.model_id, exc_info=True,
            )
            self._pipe = None
            self._device = None
            self._load_failed = True
        return self._pipe

def _get_device() -> str:
    return resolve_device()


@contextlib.contextmanager
def _suppress_hf_detection_noise():
    transformers_logging = None
    hf_hub_logging = None
    transformers_verbosity = None
    hf_hub_verbosity = None
    try:
        from transformers.utils import logging as transformers_logging  # type: ignore

        transformers_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
    except Exception:
        transformers_logging = None
    try:
        from huggingface_hub.utils import logging as hf_hub_logging  # type: ignore

        hf_hub_verbosity = hf_hub_logging.get_verbosity()
        hf_hub_logging.set_verbosity_error()
    except Exception:
        hf_hub_logging = None

    sink = io.StringIO()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            warnings.filterwarnings(
                "ignore",
                message=".*Some weights of the model checkpoint at .* were not used when initializing.*",
            )
            warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
            yield
    finally:
        if transformers_logging is not None and transformers_verbosity is not None:
            transformers_logging.set_verbosity(transformers_verbosity)
        if hf_hub_logging is not None and hf_hub_verbosity is not None:
            hf_hub_logging.set_verbosity(hf_hub_verbosity)
