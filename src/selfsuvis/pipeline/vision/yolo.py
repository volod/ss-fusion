"""YOLO11 object detector wrapper via the ultralytics package.

Uses ultralytics YOLO11 for real-time instance detection with priority-aware
output ordering:

    Priority 1 — human      : person, pedestrian
    Priority 2 — vehicle    : bicycle, car, motorcycle, bus, train, truck, boat, airplane
    Priority 3 — artificial : all other manufactured objects (signs, traffic lights, …)
    Priority 4 — other      : natural / unclassified

Priority labels are attached to every detection dict so downstream consumers
(VideoKnowledge, comparison artifacts) can sort and filter consistently.

Model tiers (smallest → largest — set with YOLO_MODEL):

  yolo11n   2.6 M   ~6 MB    real-time edge, 39.5 COCO mAP50-95
  yolo11s   9.4 M  ~18 MB    balanced speed / accuracy, 47.0
  yolo11m  20.1 M  ~38 MB    medium quality, 51.5
  yolo11l  25.3 M  ~48 MB    large, 53.4
  yolo11x  56.9 M ~109 MB    max quality, 54.7

Disabled by default (``YOLO_ENABLED=false``).  Enable with:

    YOLO_ENABLED=true YOLO_MODEL=auto python main.py --mode local

Output schema per detection:

    {
        "label":    str,           # class name (e.g. "person")
        "confidence": float,       # 0–1
        "bbox_norm": [x1,y1,x2,y2],  # normalised [0,1]
        "priority": int,           # 1=human 2=vehicle 3=artificial 4=other
        "priority_label": str,     # "human" | "vehicle" | "artificial" | "other"
        "mask_area_norm": float | None,  # fraction of image area covered by SAM mask (if available)
    }
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from selfsuvis.pipeline.core import get_logger, resolve_device, settings

logger = get_logger(__name__)

# ── Priority taxonomy ─────────────────────────────────────────────────────────

_HUMAN_LABELS = frozenset({"person", "pedestrian", "rider", "child"})

_VEHICLE_LABELS = frozenset({
    "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "van", "vehicle", "motorbike", "bike",
})

# Anything manufactured but not a vehicle or person.
_ARTIFICIAL_LABELS = frozenset({
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "traffic cone", "barrier",
    "sign", "pole", "fence", "building", "wall", "column",
    "streetlight", "street light", "lamppost", "bollard",
    "mailbox", "fire escape", "awning", "bridge", "column", "pillar",
})

PRIORITY_HUMAN = 1
PRIORITY_VEHICLE = 2
PRIORITY_ARTIFICIAL = 3
PRIORITY_OTHER = 4

_PRIORITY_LABEL = {
    PRIORITY_HUMAN: "human",
    PRIORITY_VEHICLE: "vehicle",
    PRIORITY_ARTIFICIAL: "artificial",
    PRIORITY_OTHER: "other",
}


def classify_label_priority(label: str) -> int:
    """Return the detection priority for a class label.

    Args:
        label: Raw class name from the detector (case-insensitive).

    Returns:
        1 for human, 2 for vehicle, 3 for artificial, 4 for other.
    """
    norm = label.lower().strip()
    if norm in _HUMAN_LABELS or "person" in norm or "pedestrian" in norm:
        return PRIORITY_HUMAN
    if norm in _VEHICLE_LABELS or any(v in norm for v in ("car", "truck", "bus", "van", "moto", "bike", "cycle", "boat", "plane", "train", "vehicle")):
        return PRIORITY_VEHICLE
    if norm in _ARTIFICIAL_LABELS:
        return PRIORITY_ARTIFICIAL
    return PRIORITY_OTHER


def sort_detections_by_priority(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort detections: human first, then vehicle, then artificial, then other.

    Within each priority tier, sort by confidence descending.
    """
    return sorted(
        detections,
        key=lambda d: (d.get("priority", PRIORITY_OTHER), -float(d.get("confidence", 0.0))),
    )


# ── Model resolution ──────────────────────────────────────────────────────────

_YOLO_AUTO_TIERS = [
    ("yolo11n.pt", 6),
    ("yolo11s.pt", 18),
    ("yolo11m.pt", 38),
    ("yolo11l.pt", 48),
    ("yolo11x.pt", 109),
]


def _resolve_yolo_model() -> str:
    cfg = settings.YOLO_MODEL.strip()
    if cfg and cfg.lower() != "auto":
        return cfg
    # Auto-select based on available VRAM
    try:
        from selfsuvis.pipeline.vision.registry import detect_resources
        res = detect_resources()
        vram_mb = res.get("vram_gb", 0.0) * 1024
        for name, size_mb in _YOLO_AUTO_TIERS:
            if vram_mb >= size_mb * 2:  # 2× headroom
                chosen = name
            else:
                break
        else:
            chosen = _YOLO_AUTO_TIERS[-1][0]
        return chosen
    except Exception:
        return "yolo11l.pt"


# ── Detector class ────────────────────────────────────────────────────────────

class YOLODetector:
    """YOLO11 object detector using the ultralytics package.

    Produces priority-sorted detections compatible with the existing
    ``frame_facts_json["detections"]`` schema, with an extra ``priority``
    and ``priority_label`` field per detection.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_id: Optional[str] = None
        self._load_failed = False

    def is_enabled(self) -> bool:
        return settings.YOLO_ENABLED

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = _resolve_yolo_model()
        return self._model_id

    def detect_batch(
        self,
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """Run YOLO detection on a batch of PIL images.

        Returns one result dict per image with keys:
            detections (list), detection_model (str), yolo_model (str)
        """
        if not self.is_enabled():
            return [{"detection_disabled": True}] * len(images)
        model = self._get_model()
        if model is None:
            return [{"detection_unavailable": True}] * len(images)
        return [self._detect_one(img, model) for img in images]

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """Run YOLO detection on a single PIL image."""
        if not self.is_enabled():
            return {"detection_disabled": True}
        model = self._get_model()
        if model is None:
            return {"detection_unavailable": True}
        return self._detect_one(image, model)

    def release(self) -> None:
        """Free the YOLO model and flush CUDA cache."""
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _detect_one(self, image: Image.Image, model) -> Dict[str, Any]:
        w, h = image.size
        try:
            results = model(
                image,
                conf=settings.YOLO_CONFIDENCE,
                verbose=False,
                stream=False,
            )
            detections = self._parse_results(results, w, h)
            detections = sort_detections_by_priority(detections)
            return {
                "detections": detections,
                "detection_model": "yolo",
                "yolo_model": self.model_id,
            }
        except Exception as exc:
            logger.warning("YOLO detection failed: %s", exc)
            return {"detection_error": True, "error": str(exc)}

    def _parse_results(self, results, w: int, h: int) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                    label = result.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    if conf < settings.YOLO_CONFIDENCE:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    priority = classify_label_priority(label)
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox_norm": [
                            round(x1 / w, 4),
                            round(y1 / h, 4),
                            round(x2 / w, 4),
                            round(y2 / h, 4),
                        ],
                        "priority": priority,
                        "priority_label": _PRIORITY_LABEL[priority],
                        "mask_area_norm": None,
                    })
                except Exception:
                    continue
        return detections

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError:
            logger.warning(
                "ultralytics not installed — install with: pip install ultralytics"
            )
            self._load_failed = True
            return None
        try:
            device = _get_device()
            logger.info("Loading YOLO model: %s on %s", self.model_id, device)
            # Prefer the canonical cache location so the file is never pulled
            # from (or downloaded to) the current working directory.
            model_file = self.model_id if self.model_id.endswith(".pt") else f"{self.model_id}.pt"
            cached = Path.home() / ".cache" / "ultralytics" / model_file
            model_arg = str(cached) if cached.exists() else model_file
            self._model = YOLO(model_arg)
            if device == "cuda":
                self._model.to("cuda")
            logger.info("YOLO model ready: %s", self.model_id)
        except Exception as exc:
            logger.warning(
                "Failed to load YOLO model %s (%s) — "
                "run: pip install ultralytics && python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"",
                self.model_id, exc,
            )
            self._model = None
            self._load_failed = True
        return self._model


def _get_device() -> str:
    return resolve_device()
