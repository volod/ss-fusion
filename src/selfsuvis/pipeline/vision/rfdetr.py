"""RF-DETR object detection and tracking wrapper.

Wraps Roboflow's RF-DETR model (``pip install rfdetr``) for single-frame detection
and lightweight multi-frame tracking via greedy IoU assignment.

Model tiers:
  base   — RFDETRBase   smaller, faster (~25 M params)
  large  — RFDETRLarge  higher accuracy

Output schema per detection:
    {
        "label":          str,              # class name (e.g. "person")
        "confidence":     float,            # 0–1
        "bbox_norm":      [x1,y1,x2,y2],   # normalised [0,1]
        "track_id":       int,              # persistent ID across frames (0 = unassigned)
        "priority":       int,              # 1=human 2=vehicle 3=artificial 4=other
        "priority_label": str,
    }

Tracking IDs are assigned by greedy IoU matching across frames (threshold 0.45).
IDs reset to 1 for each new video/sequence.

Disabled gracefully when ``rfdetr`` is not installed or ``RFDETR_ENABLED=false``.
"""

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.vision._quiet import suppress_runtime_noise

logger = get_logger(__name__)
logging.getLogger("rfdetr").setLevel(logging.ERROR)
logging.getLogger("rfdetr.main").setLevel(logging.ERROR)
logging.getLogger("rf-detr").setLevel(logging.ERROR)

# ── Priority taxonomy (mirrors pipeline/vision/yolo.py) ───────────────────────

_HUMAN_LABELS = frozenset({"person", "pedestrian", "rider", "child", "people"})

_VEHICLE_LABELS = frozenset(
    {
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "van",
        "vehicle",
        "motorbike",
        "bike",
    }
)

# COCO 80-class index → name; used as fallback when supervision doesn't return class_name
_COCO_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

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

_TARGET_LABEL_ALIASES: dict[str, frozenset[str]] = {
    "person": frozenset({"person", "pedestrian", "people", "human", "worker", "rider", "child"}),
    "vehicle": frozenset(
        {
            "vehicle",
            "car",
            "truck",
            "bus",
            "van",
            "pickup",
            "pickup truck",
            "motorcycle",
            "motorbike",
            "bike",
            "bicycle",
            "train",
            "boat",
            "airplane",
        }
    ),
    "building": frozenset({"building", "house", "shed", "warehouse", "garage"}),
    "road": frozenset({"road", "street"}),
    "sign": frozenset({"sign", "stop sign", "traffic sign"}),
}


def _normalise_target_label(label: str) -> str:
    return " ".join(label.lower().replace("-", " ").replace("_", " ").split())


def _expand_target_labels(target_labels: list[str]) -> list[str]:
    """Expand Gemma-style abstract labels into detector-matchable synonyms."""
    expanded: list[str] = []
    seen: set[str] = set()
    for raw in target_labels:
        norm = _normalise_target_label(raw)
        if not norm:
            continue
        candidates = {norm}
        for family, aliases in _TARGET_LABEL_ALIASES.items():
            if norm == family or norm in aliases or family in norm:
                candidates.update(aliases)
                candidates.add(family)
        for token in norm.split():
            if token in _TARGET_LABEL_ALIASES:
                candidates.update(_TARGET_LABEL_ALIASES[token])
                candidates.add(token)
        for candidate in candidates:
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
    return expanded


def _rfdetr_weights_path(variant: str) -> str:
    """Return the shared RF-DETR checkpoint path outside the repo root.

    Also pre-downloads the weights when missing.  rfdetr's
    ``download_pretrain_weights`` only matches by bare filename, so passing an
    absolute path to it silently no-ops.  We bypass that by calling the
    underlying ``_download_file`` directly with our absolute destination path.
    """
    name = "rf-detr-large.pth" if variant == "large" else "rf-detr-base.pth"
    cache_dir = Path(settings.DATA_DIR).resolve() / "models" / "rfdetr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / name

    # Backward-compat migration: older rfdetr versions download into the current
    # working directory using a bare filename. Move that shared weight once.
    legacy = Path.cwd() / name
    if not dst.exists() and legacy.exists():
        try:
            legacy.replace(dst)
            logger.info("RFDETRTracker: moved legacy checkpoint %s → %s", legacy, dst)
        except Exception:
            pass

    # Pre-download if still missing.  rfdetr.download_pretrain_weights() does an
    # exact-match lookup by bare filename, so it silently skips absolute paths.
    # We look up the asset ourselves and call _download_file with our target path.
    if not dst.exists():
        try:
            from rfdetr.assets.model_weights import ModelWeights  # type: ignore[import]
            from rfdetr.util.files import _download_file  # type: ignore[import]

            asset = ModelWeights.from_filename(name)
            if asset is not None:
                logger.info("RFDETRTracker: downloading %s → %s", name, dst)
                _download_file(url=asset.url, filename=str(dst), expected_md5=asset.md5_hash)
            else:
                logger.warning(
                    "RFDETRTracker: no asset entry for %s — will let rfdetr attempt download", name
                )
        except Exception as exc:
            logger.warning(
                "RFDETRTracker: pre-download failed (%s) — rfdetr will retry on load", exc
            )

    return str(dst)


def _classify_priority(label: str) -> int:
    lbl = label.lower()
    if lbl in _HUMAN_LABELS:
        return PRIORITY_HUMAN
    if lbl in _VEHICLE_LABELS:
        return PRIORITY_VEHICLE
    # Anything manufactured but not person or vehicle → artificial
    if lbl not in {
        "sky",
        "tree",
        "grass",
        "road",
        "water",
        "ground",
        "rock",
        "soil",
        "sand",
        "cloud",
        "mountain",
        "hill",
        "forest",
        "river",
        "lake",
        "sea",
        "ocean",
    }:
        return PRIORITY_ARTIFICIAL
    return PRIORITY_OTHER


# ── IoU helper ─────────────────────────────────────────────────────────────────


def _iou_norm(a: list[float], b: list[float]) -> float:
    """Compute IoU for two normalised [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _track_label_family(label: str) -> str:
    norm = _normalise_target_label(label)
    if norm in _HUMAN_LABELS:
        return "human"
    if norm in _VEHICLE_LABELS or norm == "vehicle":
        return "vehicle"
    return norm


def _track_match_score(track: dict[str, Any], det: dict[str, Any]) -> float:
    """Return a continuity score for matching *det* to *track*."""
    track_label = _track_label_family(str(track.get("label", "")))
    det_label = _track_label_family(str(det.get("label", "")))
    if track_label and det_label and track_label != det_label:
        return 0.0

    track_bbox = track["bbox_norm"]
    det_bbox = det["bbox_norm"]
    iou = _iou_norm(track_bbox, det_bbox)
    if iou > 0.0:
        return iou

    tcx, tcy = _bbox_center(track_bbox)
    dcx, dcy = _bbox_center(det_bbox)
    center_dist = float(np.hypot(tcx - dcx, tcy - dcy))
    track_area = _bbox_area(track_bbox)
    det_area = _bbox_area(det_bbox)
    area_ratio = det_area / max(track_area, 1e-6)
    if 0.4 <= area_ratio <= 2.5 and center_dist <= 0.08:
        # Small positive fallback score: enough to preserve identity across
        # sparse aerial frames, but still below any real-overlap match.
        return 0.10 + (0.08 - center_dist) * 0.5
    return 0.0


# ── Main tracker class ─────────────────────────────────────────────────────────


class RFDETRTracker:
    """Unified RF-DETR detector with lightweight IoU-based multi-frame tracking.

    Loads the RF-DETR model on first use (lazy) and exposes:
    - ``detect_frame``: single-frame detection, returns list of detection dicts
    - ``track_sequence``: detection + persistent track ID assignment across frames

    Track IDs reset when ``reset_tracking()`` is called or a new ``track_sequence``
    begins (call signature explicitly resets state).
    """

    # Tracks not matched for this many consecutive frames are retired.
    # 8 frames = 4 s at 2 fps; keeps identities alive through a full-second
    # occlusion (e.g. vehicle passing under a tree canopy in aerial footage).
    _MAX_MISS_FRAMES = 8
    # IoU threshold for matching a detection to an existing track.
    # Aerial drone footage at 2 fps: a vehicle travelling 60 km/h moves
    # ~8 m between frames; at typical drone altitude that is 30-50 % of the
    # box width, leaving very little overlap.  0.10 keeps tracks alive without
    # merging clearly-separate vehicles (min separable IoU ≈ 0.0 for non-
    # overlapping boxes, so 0.10 still requires at least a 10 % overlap).
    _IOU_THRESHOLD = 0.10

    def __init__(self) -> None:
        self._model = None
        self._load_failed = False
        self._model_variant: str | None = None
        self._reset_tracking_state()

    # ── Public interface ──────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return settings.RFDETR_ENABLED

    @property
    def model_id(self) -> str:
        variant = settings.RFDETR_MODEL.strip().lower()
        return f"rfdetr_{variant}" if variant in ("base", "large") else "rfdetr_base"

    def detect_frame(
        self,
        image: Image.Image,
        target_labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect objects in *image*.

        Args:
            image: RGB PIL image.
            target_labels: When provided, only detections whose label appears in
                this list (case-insensitive substring match) are returned.

        Returns:
            List of detection dicts (may be empty).
        """
        model = self._get_model()
        if model is None:
            return []
        expanded_targets = (
            _expand_target_labels(target_labels) if target_labels is not None else None
        )
        w, h = image.size
        try:
            dets = self._run_inference(model, image)
            results: list[dict[str, Any]] = []
            for label, conf, (x1, y1, x2, y2) in dets:
                bbox_norm = [x1 / w, y1 / h, x2 / w, y2 / h]
                if expanded_targets is not None and not _label_matches_any(label, expanded_targets):
                    continue
                priority = _classify_priority(label)
                results.append(
                    {
                        "label": label,
                        "confidence": round(float(conf), 4),
                        "bbox_norm": [round(v, 6) for v in bbox_norm],
                        "track_id": 0,
                        "priority": priority,
                        "priority_label": _PRIORITY_LABEL[priority],
                    }
                )
            return results
        except Exception as exc:
            logger.warning("RFDETRTracker.detect_frame failed: %s", exc)
            return []

    def track_sequence(
        self,
        frame_items: list[tuple[str, float]],
        target_labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect and track objects across a sequence of frames.

        Args:
            frame_items: List of (frame_path, t_sec) pairs.
            target_labels: When provided, filters detections to matching labels.

        Returns:
            List of per-frame result dicts:
                {
                    "frame_path": str,
                    "t_sec":      float,
                    "detections": List[detection_dict],   # with persistent track_id
                }
        """
        self._reset_tracking_state()
        results: list[dict[str, Any]] = []
        model = self._get_model()
        if model is None:
            return [{"frame_path": fp, "t_sec": t, "detections": []} for fp, t in frame_items]

        for fp, t_sec in frame_items:
            try:
                img = Image.open(fp).convert("RGB")
                dets = self.detect_frame(img, target_labels=target_labels)
                dets = self._assign_track_ids(dets)
                results.append({"frame_path": fp, "t_sec": t_sec, "detections": dets})
            except Exception as exc:
                logger.debug("RFDETRTracker.track_sequence frame %s failed: %s", fp, exc)
                results.append({"frame_path": fp, "t_sec": t_sec, "detections": []})

        return results

    def reset_tracking(self) -> None:
        """Reset tracking state (call between videos/sequences)."""
        self._reset_tracking_state()

    def release(self) -> None:
        """Free GPU memory and model references."""
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_tracking_state(self) -> None:
        """Active tracks: {track_id: {"bbox_norm": list, "label": str, "miss": int}}."""
        self._active_tracks: dict[int, dict[str, Any]] = {}
        self._next_id = 1

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            self._model = self._load_model()
        except Exception as exc:
            logger.warning("RFDETRTracker: failed to load model: %s", exc)
            self._load_failed = True
            return None
        return self._model

    def _load_model(self):
        variant = settings.RFDETR_MODEL.strip().lower()
        weights_path = _rfdetr_weights_path(variant)
        try:
            if variant == "large":
                from rfdetr import RFDETRLarge  # type: ignore[import]

                with suppress_runtime_noise(
                    r".*loss_type=None.*",
                    logger_levels={
                        "rfdetr": logging.ERROR,
                        "rfdetr.main": logging.ERROR,
                        "rf-detr": logging.ERROR,
                        "transformers": logging.ERROR,
                    },
                ):
                    model = RFDETRLarge(pretrain_weights=weights_path)
                self._model_variant = "large"
            else:
                from rfdetr import RFDETRBase  # type: ignore[import]

                with suppress_runtime_noise(
                    r".*loss_type=None.*",
                    logger_levels={
                        "rfdetr": logging.ERROR,
                        "rfdetr.main": logging.ERROR,
                        "rf-detr": logging.ERROR,
                        "transformers": logging.ERROR,
                    },
                ):
                    model = RFDETRBase(pretrain_weights=weights_path)
                self._model_variant = "base"
            if hasattr(model, "optimize_for_inference"):
                with suppress_runtime_noise(
                    r".*loss_type=None.*",
                    logger_levels={
                        "rfdetr": logging.ERROR,
                        "rfdetr.main": logging.ERROR,
                        "rf-detr": logging.ERROR,
                        "transformers": logging.ERROR,
                    },
                ):
                    model.optimize_for_inference()
            logger.info(
                "RFDETRTracker: loaded rfdetr_%s (weights=%s)", self._model_variant, weights_path
            )
            return model
        except ImportError as exc:
            raise ImportError(
                "rfdetr package not installed. Install with:\n"
                "  pip install rfdetr\n"
                "or set RFDETR_ENABLED=false to disable."
            ) from exc

    def _run_inference(
        self, model, image: Image.Image
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        """Run RF-DETR inference and return (label, confidence, (x1,y1,x2,y2)) tuples
        in pixel coordinates."""
        conf_threshold = max(0.05, float(settings.RFDETR_CONFIDENCE))
        try:
            # RF-DETR predict returns a supervision.Detections object
            with suppress_runtime_noise(
                r".*loss_type=None.*",
                logger_levels={
                    "rfdetr": logging.ERROR,
                    "rfdetr.main": logging.ERROR,
                    "rf-detr": logging.ERROR,
                    "transformers": logging.ERROR,
                },
            ):
                sv_dets = model.predict(image, threshold=conf_threshold)
        except Exception as exc:
            logger.debug("RF-DETR inference error: %s", exc)
            return []

        results = []
        try:
            xyxy = sv_dets.xyxy  # (N, 4) float pixel coords
            confs = sv_dets.confidence  # (N,) float
            # class names — sv stores them in data["class_name"] for RF-DETR
            class_names = None
            if hasattr(sv_dets, "data") and sv_dets.data:
                class_names = sv_dets.data.get("class_name")
            if class_names is None and sv_dets.class_id is not None:
                # Fallback: map COCO class IDs to names; keeps label-filter logic working
                class_names = [_COCO_CLASSES.get(int(c), str(int(c))) for c in sv_dets.class_id]
            if class_names is None:
                return []

            for i, (box, conf, label) in enumerate(zip(xyxy, confs, class_names)):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                lbl = str(label).lower() if label else "unknown"
                results.append((lbl, float(conf), (x1, y1, x2, y2)))
        except Exception as exc:
            logger.debug("RF-DETR result parsing error: %s", exc)
        return results

    def _assign_track_ids(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign persistent track IDs to detections using greedy IoU matching.

        Updates ``self._active_tracks`` in-place:
        - Matched tracks keep their ID and reset miss counter.
        - Unmatched detections spawn new IDs.
        - Tracks unmatched for > _MAX_MISS_FRAMES are retired.
        """
        if not detections:
            # Increment miss counters for all active tracks
            to_remove = []
            for tid, track in self._active_tracks.items():
                track["miss"] += 1
                if track["miss"] > self._MAX_MISS_FRAMES:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._active_tracks[tid]
            return detections

        # Build IoU matrix: rows = active tracks, cols = new detections
        track_ids = list(self._active_tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for ti, tid in enumerate(track_ids):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = _track_match_score(self._active_tracks[tid], det)

        # Greedy assignment: highest IoU first
        assigned_track: dict[int, int] = {}  # det_idx → track_id
        assigned_det: set = set()
        assigned_track_set: set = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_val = iou_matrix.max()
            if max_val < self._IOU_THRESHOLD:
                break
            ti, di = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            ti, di = int(ti), int(di)
            if ti not in assigned_track_set and di not in assigned_det:
                track_id = track_ids[ti]
                assigned_track[di] = track_id
                assigned_det.add(di)
                assigned_track_set.add(ti)
            iou_matrix[ti, :] = -1.0
            iou_matrix[:, di] = -1.0

        # Retire unmatched active tracks (increment miss)
        to_remove = []
        for ti, tid in enumerate(track_ids):
            if ti not in assigned_track_set:
                self._active_tracks[tid]["miss"] += 1
                if self._active_tracks[tid]["miss"] > self._MAX_MISS_FRAMES:
                    to_remove.append(tid)
        for tid in to_remove:
            del self._active_tracks[tid]

        # Assign IDs to detections
        for di, det in enumerate(detections):
            if di in assigned_track:
                tid = assigned_track[di]
                det["track_id"] = tid
                self._active_tracks[tid]["bbox_norm"] = det["bbox_norm"]
                self._active_tracks[tid]["label"] = det["label"]
                self._active_tracks[tid]["priority"] = det.get("priority", PRIORITY_OTHER)
                self._active_tracks[tid]["miss"] = 0
            else:
                # New track
                new_id = self._next_id
                self._next_id += 1
                det["track_id"] = new_id
                self._active_tracks[new_id] = {
                    "bbox_norm": det["bbox_norm"],
                    "label": det["label"],
                    "priority": det.get("priority", PRIORITY_OTHER),
                    "miss": 0,
                }

        return detections


# ── Label matching helper ──────────────────────────────────────────────────────


def _label_matches_any(label: str, target_labels: list[str]) -> bool:
    """Return True when *label* appears in or overlaps with any target label."""
    label_lower = _normalise_target_label(label)
    for target in target_labels:
        target_lower = _normalise_target_label(target)
        if target_lower in label_lower or label_lower in target_lower:
            return True
    return False
