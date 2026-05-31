import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import ensure_dir, get_logger, now_iso, settings
from selfsuvis.pipeline.core.optional_deps import require_cv2

if TYPE_CHECKING:
    pass


@dataclass
class Segment:
    segment_id: str
    label: str
    bbox: tuple[int, int, int, int]
    mean_color: tuple[int, int, int]
    area: int


@dataclass
class FrameResult:
    description: str
    segments: list[Segment]
    entities: list[dict[str, Any]]
    tracks: list[dict[str, Any]]
    warnings: list[str]


def _dominant_color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


def _scene_description(frame_bgr: np.ndarray) -> str:
    cv2 = require_cv2()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    brightness = "bright" if mean_val > 150 else "dim" if mean_val < 80 else "balanced"
    texture = "detailed" if edge_density > 0.08 else "smooth"
    return f"A {brightness}, {texture} frame."


def _segment_kmeans(frame_bgr: np.ndarray, k: int = 4) -> list[Segment]:
    cv2 = require_cv2()
    h, w = frame_bgr.shape[:2]
    data = frame_bgr.reshape((-1, 3)).astype(np.float32)
    if data.shape[0] < k:
        k = max(1, data.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    centers = centers.astype(np.uint8)

    segments: list[Segment] = []
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            continue
        ys, xs = np.where(mask.reshape((h, w)))
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        area = int(mask.sum())
        mean_color = tuple(int(c) for c in centers[i].tolist())
        seg = Segment(
            segment_id=f"region_{i}",
            label=f"region_{i}",
            bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            mean_color=mean_color,
            area=area,
        )
        segments.append(seg)
    return segments


def _bbox_iou(a: Segment, b: Segment) -> float:
    """Intersection-over-union for two bounding boxes."""
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(aw * ah + bw * bh - inter, 1)


def image_to_text_agent(
    frame_bgr: np.ndarray,
    tagger: Any | None = None,
    segmenter: Any | None = None,
) -> tuple[str, list[Segment]]:
    segments: list[Segment] = []
    if segmenter and tagger:
        try:
            from selfsuvis.pipeline.vision.factory import mask_to_segments

            masks = segmenter.segment(frame_bgr)
            sam_segments = mask_to_segments(frame_bgr, masks, tagger)
            segments = [
                Segment(
                    segment_id=s.segment_id,
                    label=s.label,
                    bbox=s.bbox,
                    mean_color=s.mean_color,
                    area=s.area,
                )
                for s in sam_segments
            ]
        except Exception:
            segments = _segment_kmeans(frame_bgr, k=4)
    else:
        segments = _segment_kmeans(frame_bgr, k=4)

    description = _scene_description(frame_bgr)
    if tagger is not None:
        info = tagger.describe_image(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        if info.get("labels"):
            labels = ", ".join([item["label"] for item in info["labels"]])
            description = f"{description} Likely: {labels}."

    if segments:
        dominant = max(segments, key=lambda s: s.area)
        color = _dominant_color_name(dominant.mean_color[::-1])
        description = f"{description} Dominant color: {color}."
    return description, segments


def recognition_correctness_agent(description: str, segments: list[Segment]) -> list[str]:
    warnings = []
    if not segments:
        warnings.append("no_segments_detected")
    if "Dominant color" not in description:
        warnings.append("missing_color_summary")
    return warnings


def ontology_agent(
    ontology: dict[str, Any],
    segments: list[Segment],
    frame_index: int,
    t_sec: float,
) -> dict[str, Any]:
    entities = ontology.setdefault("entities", {})
    for seg in segments:
        ent = entities.setdefault(
            seg.label, {"count": 0, "first_seen": frame_index, "last_seen": frame_index}
        )
        ent["count"] += 1
        ent["last_seen"] = frame_index
        ent.setdefault("colors", {})
        color_name = _dominant_color_name(seg.mean_color[::-1])
        ent["colors"][color_name] = ent["colors"].get(color_name, 0) + 1
    ontology.setdefault("timeline", []).append({"frame_index": frame_index, "t_sec": t_sec})
    return ontology


def matching_agent(
    segments: list[Segment],
    prev_segments: list[Segment] | None,
    prev_tracks: dict[str, int],
    next_track_id: int,
    iou_threshold: float = 0.2,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    tracks: list[dict[str, Any]] = []
    updated_tracks: dict[str, int] = {}

    for seg in segments:
        track_id = None
        if prev_segments:
            best_score = 0.0
            best_seg = None
            for prev in prev_segments:
                score = _bbox_iou(seg, prev)
                if score > best_score:
                    best_score = score
                    best_seg = prev
            if best_seg and best_score >= iou_threshold:
                track_id = prev_tracks.get(best_seg.segment_id)
        if track_id is None:
            track_id = next_track_id
            next_track_id += 1
        updated_tracks[seg.segment_id] = track_id
        tracks.append(
            {
                "track_id": track_id,
                "segment_id": seg.segment_id,
                "label": seg.label,
                "bbox": seg.bbox,
            }
        )
    return tracks, updated_tracks, next_track_id


def _build_ontology_entities(ontology: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ontology entities dict into a list for the frame record."""
    result = []
    for name, ent in ontology.get("entities", {}).items():
        colors = ent.get("colors", {})
        dominant_color = max(colors, key=colors.get) if colors else None
        result.append(
            {
                "name": name,
                "count": ent.get("count", 0),
                "first_seen": ent.get("first_seen"),
                "last_seen": ent.get("last_seen"),
                "dominant_color": dominant_color,
            }
        )
    return result


def _process_frame_to_record(
    rec: Any,
    video_name: str,
    tagger: Any | None,
    segmenter: Any | None,
    ontology: dict[str, Any],
    prev_segments: list[Segment] | None,
    prev_tracks: dict[str, int],
    next_track_id: int,
    base_metadata: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], list[Segment] | None, dict[str, int], int]:
    """Process one frame through agents. Returns (record_dict, ontology, prev_segments, prev_tracks, next_track_id).
    record_dict is None if the frame could not be read."""
    frame = cv2.imread(rec.path)
    if frame is None:
        return (None, ontology, prev_segments, prev_tracks, next_track_id)

    description, segments = image_to_text_agent(frame, tagger=tagger, segmenter=segmenter)
    warnings = recognition_correctness_agent(description, segments)
    ontology = ontology_agent(ontology, segments, rec.index, rec.t_sec)
    tracks, prev_tracks, next_track_id = matching_agent(
        segments, prev_segments, prev_tracks, next_track_id
    )

    entities = [
        {
            "name": seg.label,
            "type": "region",
            "bbox": seg.bbox,
            "dominant_color": _dominant_color_name(seg.mean_color[::-1]),
        }
        for seg in segments
    ]

    record = formatter_agent(
        video_name=video_name,
        frame_index=rec.index,
        t_sec=rec.t_sec,
        width=rec.width,
        height=rec.height,
        description=description,
        segments=segments,
        entities=entities,
        tracks=tracks,
        warnings=warnings,
        frame_path=rec.path,
        ontology_entities=_build_ontology_entities(ontology),
        metadata=base_metadata,
    )
    return (record, ontology, segments, prev_tracks, next_track_id)


def formatter_agent(
    video_name: str,
    frame_index: int,
    t_sec: float,
    width: int,
    height: int,
    description: str,
    segments: list[Segment],
    entities: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    warnings: list[str],
    frame_path: str,
    ontology_entities: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "video_name": video_name,
        "frame_index": frame_index,
        "t_sec": t_sec,
        "width": width,
        "height": height,
        "frame_path": frame_path,
        "description": description,
        "segments": [
            {
                "segment_id": seg.segment_id,
                "label": seg.label,
                "bbox": seg.bbox,
                "mean_color": seg.mean_color,
                "area": seg.area,
            }
            for seg in segments
        ],
        "entities": entities,
        "tracks": tracks,
        "warnings": warnings,
        "ontology_entities": ontology_entities,
        "metadata": metadata,
    }


def process_frames(
    video_name: str,
    frame_records: list[Any],
    output_dir: str,
    run_metadata: dict[str, Any] | None = None,
    model_type: str = "openclip_sam",
    sam_checkpoint: str | None = None,
    sam_model_type: str | None = None,
    labels_file: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    logger = get_logger(__name__)
    ensure_dir(output_dir)
    jsonl_path = os.path.join(output_dir, f"{video_name}.jsonl")
    ontology_path = os.path.join(output_dir, f"{video_name}.ontology.json")

    ontology: dict[str, Any] = {"video_name": video_name, "entities": {}, "timeline": []}
    base_metadata: dict[str, Any] = {
        "created_at": now_iso(),
        "output_dir": output_dir,
        "model_type": model_type,
    }
    if run_metadata:
        base_metadata.update(run_metadata)
    prev_segments: list[Segment] | None = None
    prev_tracks: dict[str, int] = {}
    next_track_id = 1

    tagger: Any | None = None
    segmenter: Any | None = None
    if model_type == "openclip_sam" and not (sam_checkpoint or settings.SAM_CHECKPOINT):
        logger.warning("SAM checkpoint missing; falling back to openclip_only")
        model_type = "openclip_only"
        base_metadata["model_type"] = model_type

    if model_type in {"openclip_sam", "openclip_only"}:
        from selfsuvis.pipeline.vision.factory import OpenCLIPTagger

        tagger = OpenCLIPTagger(labels_file=labels_file)
    if model_type == "openclip_sam":
        from selfsuvis.pipeline.vision.factory import SAMSegmenter

        segmenter = SAMSegmenter(model_type=sam_model_type, checkpoint=sam_checkpoint)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in frame_records:
            record, ontology, prev_segments, prev_tracks, next_track_id = _process_frame_to_record(
                rec,
                video_name,
                tagger,
                segmenter,
                ontology,
                prev_segments,
                prev_tracks,
                next_track_id,
                base_metadata,
            )
            if record is None:
                continue
            if verbose:
                logger.info(
                    "frame=%s t=%.3fs size=%sx%s desc=%s",
                    rec.path,
                    rec.t_sec,
                    rec.width,
                    rec.height,
                    record["description"],
                )
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    with open(ontology_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=True, indent=2)

    logger.info("Wrote jsonl=%s ontology=%s", jsonl_path, ontology_path)
    return {"jsonl_path": jsonl_path, "ontology_path": ontology_path}
