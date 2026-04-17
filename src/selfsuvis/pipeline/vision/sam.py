"""SAM2 / SAM3 segmentation wrapper.

Supports Meta's Segment Anything Model family:

  - SAM3 (pip: sam3) — latest, preferred
  - SAM2 (pip: sam2) — fallback
  - Original SAM (pip: segment-anything) — last resort

Usage in the local full-analysis pipeline:

    predictor = SAMPredictor()
    if predictor.is_available():
        masks = predictor.predict_boxes(image_pil, bboxes_norm=[(x1,y1,x2,y2), ...])

``predict_boxes`` takes YOLO normalised bbox tuples and returns one mask dict
per bbox with:

    {
        "mask":      np.ndarray bool (H, W),
        "score":     float,
        "area_norm": float,   # fraction of image area covered
    }

Disabled gracefully (returns empty list) when:
- ``SAM_ENABLED=false`` (default)
- No SAM package is installed
- ``SAM_CHECKPOINT`` file is missing for the original SAM backend
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import get_logger, resolve_device, settings

logger = get_logger(__name__)

# ── Backend resolution ────────────────────────────────────────────────────────

_BACKEND_SAM3 = "sam3"
_BACKEND_SAM2 = "sam2"
_BACKEND_SAM1 = "sam1"
_BACKEND_NONE = "none"

# HuggingFace model IDs for auto-download (SAM2 / SAM3)
_SAM2_HF_MODELS = [
    "facebook/sam2-hiera-large",
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2-hiera-small",
    "facebook/sam2-hiera-tiny",
]
_SAM3_HF_MODELS = [
    "facebook/sam3",
]


def _detect_backend() -> str:
    cfg = settings.SAM_MODEL.strip().lower()
    # Explicit override
    if cfg == "sam3":
        return _BACKEND_SAM3
    if cfg in ("sam2", "sam-2"):
        return _BACKEND_SAM2
    if cfg in ("sam1", "sam", "segment-anything"):
        return _BACKEND_SAM1
    # Auto-detect by import availability (sam3 preferred, sam2 fallback)
    try:
        import sam3  # type: ignore[import]  # noqa: F401
        return _BACKEND_SAM3
    except ImportError:
        pass
    try:
        import sam2  # type: ignore[import]  # noqa: F401
        return _BACKEND_SAM2
    except ImportError:
        pass
    try:
        import segment_anything  # type: ignore[import]  # noqa: F401
        # SAM1 requires a manually downloaded checkpoint file.  Without it the
        # backend cannot load, so treat an unconfigured SAM1 the same as "not
        # installed" rather than advertising it and then logging a WARNING.
        if settings.SAM_CHECKPOINT:
            return _BACKEND_SAM1
        logger.debug(
            "segment_anything installed but SAM_CHECKPOINT is not set — "
            "SAM disabled. Set SAM_CHECKPOINT to a .pth file to enable SAM1, "
            "or install the sam2/sam3 package for checkpoint-free loading."
        )
    except ImportError:
        pass
    return _BACKEND_NONE


def _get_device() -> str:
    return resolve_device()


# ── Main predictor class ──────────────────────────────────────────────────────

class SAMPredictor:
    """Unified SAM2/SAM3 predictor for box-prompted instance segmentation.

    Loads the best available SAM backend and exposes ``predict_boxes`` which
    takes normalised bounding boxes (from YOLO) and returns binary masks.
    """

    def __init__(self) -> None:
        self._backend: Optional[str] = None
        self._predictor = None
        self._load_failed = False
        self._amg = None  # cached SAM2/SAM3 AutomaticMaskGenerator

    def is_available(self) -> bool:
        """Return True when SAM is enabled and a backend is installed."""
        if not settings.SAM_ENABLED:
            return False
        return self._get_predictor() is not None

    def predict_boxes(
        self,
        image: Image.Image,
        bboxes_norm: List[Tuple[float, float, float, float]],
    ) -> List[Dict[str, Any]]:
        """Segment objects specified by normalised bounding boxes.

        Args:
            image: Input PIL image (RGB).
            bboxes_norm: List of (x1, y1, x2, y2) in [0, 1] coordinates,
                one per detected object (typically from YOLO output).

        Returns:
            List of mask dicts (same length as bboxes_norm). Each dict:
                mask (np.ndarray bool), score (float), area_norm (float).
            Returns empty list when no predictor is available.
        """
        if not self.is_available() or not bboxes_norm:
            return []
        predictor = self._get_predictor()
        if predictor is None:
            return []
        w, h = image.size
        # Convert normalised → pixel coordinates
        boxes_px = np.array([
            [x1 * w, y1 * h, x2 * w, y2 * h]
            for x1, y1, x2, y2 in bboxes_norm
        ], dtype=np.float32)
        try:
            return self._predict_with_backend(image, boxes_px, w, h)
        except Exception as exc:
            logger.warning("SAM prediction failed: %s", exc)
            return []

    def get_auto_mask_generator(self, points_per_side: int = 8):
        """Return a cached automatic mask generator for this SAM backend.

        Uses ``points_per_side=8`` (64 prompts) by default — a 4× reduction
        vs the 16 (256 prompts) that was causing ~25 min/frame when combined
        with serial CLIP filtering.  Caches across frames so the generator
        is not re-instantiated on every call.

        Returns None if no SAM2/SAM3 backend is available.
        """
        if self._amg is not None:
            return self._amg
        predictor = self._get_predictor()
        if predictor is None:
            return None
        backend_tag, predictor_obj = predictor
        try:
            if backend_tag == "sam3":
                import sam3.automatic_mask_generator as _sam3_amg  # type: ignore
                amg_cls = getattr(_sam3_amg, "SAM3AutomaticMaskGenerator", None)
                if amg_cls is None:
                    amg_cls = getattr(_sam3_amg, "SAMAutomaticMaskGenerator", None)
                if amg_cls is not None:
                    self._amg = amg_cls(
                        predictor_obj.model,
                        points_per_side=points_per_side,
                        pred_iou_thresh=0.7,
                        stability_score_thresh=0.85,
                        min_mask_region_area=100,
                    )
            elif backend_tag == "sam2":
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
                self._amg = SAM2AutomaticMaskGenerator(
                    predictor_obj.model,
                    points_per_side=points_per_side,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.85,
                    min_mask_region_area=100,
                )
        except Exception as exc:
            logger.debug("SAM AMG init failed: %s", exc)
            self._amg = None
        return self._amg

    def release(self) -> None:
        """Free GPU memory held by the SAM predictor."""
        self._predictor = None
        self._amg = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor
        if self._load_failed:
            return None
        backend = _detect_backend()
        self._backend = backend
        if backend == _BACKEND_NONE:
            logger.info(
                "No SAM backend found. Install one with:\n"
                "  pip install sam3           # SAM3 (preferred)\n"
                "  pip install sam2           # SAM2 fallback"
            )
            self._load_failed = True
            return None
        try:
            self._predictor = self._load_predictor(backend)
            logger.info("SAM predictor ready (backend=%s)", self._backend)
        except Exception as exc:
            logger.warning("SAM load failed (backend=%s): %s", backend, exc)
            self._load_failed = True
            self._predictor = None
        return self._predictor

    def _load_predictor(self, backend: str):
        device = _get_device()
        if backend == _BACKEND_SAM3:
            return self._load_sam3(device)
        if backend == _BACKEND_SAM2:
            return self._load_sam2(device)
        return self._load_sam1(device)

    def _load_sam3(self, device: str):
        """Load SAM3 predictor (https://github.com/facebookresearch/sam3)."""
        try:
            from sam3.sam3_image_predictor import SAM3ImagePredictor  # type: ignore[import]
            model_id = _SAM3_HF_MODELS[0]
            predictor = SAM3ImagePredictor.from_pretrained(model_id)
            if device == "cuda":
                predictor.model.to("cuda")
            logger.info("  SAM3 loaded from %s on %s", model_id, device)
            return ("sam3", predictor)
        except Exception as exc:
            logger.debug("SAM3 load failed: %s — falling back to SAM2", exc)
            self._backend = _BACKEND_SAM2
            return self._load_sam2(device)

    def _load_sam2(self, device: str):
        """Load SAM2 predictor (pip: sam2)."""
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]
            model_id = _SAM2_HF_MODELS[0]
            predictor = SAM2ImagePredictor.from_pretrained(model_id)
            if device == "cuda":
                predictor.model.to("cuda")
            logger.info("  SAM2 loaded from %s on %s", model_id, device)
            return ("sam2", predictor)
        except Exception as exc:
            logger.debug("SAM2 load failed: %s — falling back to SAM1", exc)
            self._backend = _BACKEND_SAM1
            return self._load_sam1(device)

    def _load_sam1(self, device: str):
        """Load original SAM predictor (pip: segment-anything)."""
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore[import]
        checkpoint = settings.SAM_CHECKPOINT
        if not checkpoint:
            raise RuntimeError(
                "SAM1 requires SAM_CHECKPOINT to be set. "
                "Download from https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        model_type = settings.SAM_MODEL_TYPE or "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        if device == "cuda":
            sam.to(device="cuda")
        predictor = SamPredictor(sam)
        logger.info("  SAM1 loaded (type=%s) on %s", model_type, device)
        return ("sam1", predictor)

    def _predict_with_backend(
        self,
        image: Image.Image,
        boxes_px: np.ndarray,
        w: int,
        h: int,
    ) -> List[Dict[str, Any]]:
        backend_tag, predictor = self._predictor
        img_np = np.array(image)  # RGB uint8 (H, W, 3)
        results: List[Dict[str, Any]] = []

        if backend_tag in ("sam2", "sam3"):
            try:
                import torch
                with torch.inference_mode():
                    predictor.set_image(img_np)
                    for box in boxes_px:
                        box_t = torch.from_numpy(box[None]).float()
                        masks, scores, _ = predictor.predict(
                            box=box_t,
                            multimask_output=False,
                        )
                        mask = masks[0].astype(bool)
                        score = float(scores[0])
                        area_norm = float(mask.sum()) / (w * h)
                        results.append({
                            "mask": mask,
                            "score": round(score, 4),
                            "area_norm": round(area_norm, 6),
                        })
            except Exception as exc:
                logger.debug("SAM2/3 predict_boxes failed: %s", exc)
                results = []

        elif backend_tag == "sam1":
            predictor.set_image(img_np)
            for box in boxes_px:
                try:
                    masks, scores, _ = predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
                    mask = masks[0].astype(bool)
                    score = float(scores[0])
                    area_norm = float(mask.sum()) / (w * h)
                    results.append({
                        "mask": mask,
                        "score": round(score, 4),
                        "area_norm": round(area_norm, 6),
                    })
                except Exception:
                    results.append({"mask": None, "score": 0.0, "area_norm": 0.0})

        return results
