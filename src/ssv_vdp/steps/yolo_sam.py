"""Step 09: YOLO11 detection + SAM2/3 segmentation with priority ordering.

Runs after the HuggingFace object-detection step (P) and produces:

  yolo_sam/
    frame_{t:.3f}_annotated.jpg     annotated frame (boxes + masks + priority color)
  yolo_sam_results.json             per-frame detections with priority labels
  detection_comparison.md           comparison: YOLO vs HF detector vs SAM masks

Priority color coding (RGB):
  Human      (priority 1) → red   (#E53935)
  Vehicle    (priority 2) → blue  (#1E88E5)
  Artificial (priority 3) → green (#43A047)
  Other      (priority 4) → grey  (#9E9E9E)
"""

import logging as _logging_mod
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.vision.yolo import (
    PRIORITY_ARTIFICIAL,
    PRIORITY_HUMAN,
    PRIORITY_VEHICLE,
    YOLODetector,
    classify_label_priority,
)

from .common import _open_frame_image, write_json_artifact, write_markdown_artifact

_log = _logging_mod.getLogger("pipeline.local.yolo_sam")

# -- Priority → display color (RGB) -------------------------------------------

_PRIORITY_COLOR: dict[int, tuple[int, int, int]] = {
    PRIORITY_HUMAN: (229, 57, 53),  # red
    PRIORITY_VEHICLE: (30, 136, 229),  # blue
    PRIORITY_ARTIFICIAL: (67, 160, 71),  # green
    4: (158, 158, 158),  # grey
}

_PRIORITY_LABEL = {
    PRIORITY_HUMAN: "human",
    PRIORITY_VEHICLE: "vehicle",
    PRIORITY_ARTIFICIAL: "artificial",
    4: "other",
}

# Sample at most this many frames for YOLO+SAM to keep the step fast
_MAX_YOLO_FRAMES = 60

# Render bounding boxes at this line width relative to image width
_BOX_WIDTH_RATIO = 0.003


def _draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    sam_masks: list[Any | None] | None = None,
) -> Image.Image:
    """Draw bounding boxes (and optional SAM masks) on *image*.

    Box color and label badge follow the priority color scheme.
    SAM masks are rendered as a semi-transparent overlay when available.
    """
    import numpy as np

    w, h = image.size
    result = image.copy().convert("RGBA")

    # Draw SAM mask overlays (semi-transparent filled regions)
    if sam_masks:
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for det, mask_info in zip(detections, sam_masks):
            if mask_info is None:
                continue
            mask = mask_info.get("mask")
            if mask is None:
                continue
            priority = det.get("priority", 4)
            color = _PRIORITY_COLOR.get(priority, (158, 158, 158))
            mask_img = Image.fromarray((mask * 80).astype(np.uint8), mode="L")
            mask_img = mask_img.resize((w, h), Image.NEAREST)
            fill = Image.new("RGBA", (w, h), color + (0,))
            fill.putalpha(mask_img)
            overlay = Image.alpha_composite(overlay, fill)
        result = Image.alpha_composite(result, overlay)

    draw = ImageDraw.Draw(result)
    lw = max(2, int(w * _BOX_WIDTH_RATIO))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        priority = det.get("priority", 4)
        color = _PRIORITY_COLOR.get(priority, (158, 158, 158))
        x1n, y1n, x2n, y2n = det.get("bbox_norm", [0, 0, 1, 1])
        x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)

        # Bounding box
        for offset in range(lw):
            draw.rectangle(
                [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                outline=color,
            )

        # Label badge
        label = det.get("label", "?")
        conf = det.get("confidence", 0.0)
        prio_name = _PRIORITY_LABEL.get(priority, "other")
        badge = f"[{prio_name[0].upper()}] {label} {conf:.2f}"

        if font is not None:
            try:
                bbox = draw.textbbox((x1, y1 - 16), badge, font=font)
                draw.rectangle(bbox, fill=color + (220,))
                draw.text((x1, y1 - 16), badge, fill=(255, 255, 255), font=font)
            except Exception:
                pass

    return result.convert("RGB")


def step_yolo_sam_detection(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    det_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 09: YOLO11 detection + optional SAM2/3 segmentation.

    Args:
        frame_list: List of (frame_path, t_sec) from the extraction step.
        video_name: Human-readable video identifier.
        video_dir:  Per-video output directory.
        device:     Torch device string ("cpu" or "cuda").
        det_result: Optional output from step 08 (HF detector) for comparison.

    Returns:
        Dict with keys:
            skipped (bool), detection_results (list), comparison_md (str),
            n_frames (int), total_objects (int), elapsed_sec (float),
            human_count (int), vehicle_count (int), artificial_count (int).
    """
    result: dict[str, Any] = {"skipped": True, "detection_results": [], "total_objects": 0}

    detector = YOLODetector()
    if not detector.is_enabled():
        result["reason"] = "YOLO_ENABLED=false"
        return result

    # Try to load SAM predictor (optional; silently skip if unavailable)
    sam_available = False
    sam_predictor = None
    if settings.SAM_ENABLED:
        try:
            from selfsuvis.pipeline.vision.sam import SAMPredictor

            sam_predictor = SAMPredictor()
            sam_available = sam_predictor.is_available()
            if not sam_available:
                _log.info("SAM not available — running detection only")
        except Exception as exc:
            _log.debug("SAM import failed: %s", exc)

    t0 = time.time()

    # Sample frames evenly; do not exceed _MAX_YOLO_FRAMES
    n_avail = len(frame_list)
    step = max(1, n_avail // _MAX_YOLO_FRAMES)
    sampled = frame_list[::step][:_MAX_YOLO_FRAMES]
    n = len(sampled)
    _log.info(
        "Running YOLO11 (%s) on %d/%d frames%s",
        detector.model_id,
        n,
        n_avail,
        " + SAM2/3 segmentation" if sam_available else "",
    )

    out_dir = video_dir / "yolo_sam"
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_results: list[dict[str, Any]] = []
    human_count = vehicle_count = artificial_count = other_count = 0
    total_objects = 0
    annotated_paths: list[str] = []

    for idx, (fp, t_sec) in enumerate(sampled):
        try:
            img = _open_frame_image(fp)
            det = detector.detect(img)
            detections = det.get("detections", [])
            total_objects += len(detections)

            # SAM segmentation: produce masks for all detected bboxes
            sam_masks: list | None = None
            if sam_available and sam_predictor is not None and detections:
                bboxes_norm = [tuple(d["bbox_norm"]) for d in detections]
                raw_masks = sam_predictor.predict_boxes(img, bboxes_norm)
                # Attach mask area to each detection
                sam_masks = []
                for det_item, mask_info in zip(detections, raw_masks):
                    area = mask_info.get("area_norm", 0.0) if mask_info else 0.0
                    det_item["mask_area_norm"] = round(area, 6)
                    sam_masks.append(mask_info)

            # Count by priority
            for d in detections:
                p = d.get("priority", 4)
                if p == PRIORITY_HUMAN:
                    human_count += 1
                elif p == PRIORITY_VEHICLE:
                    vehicle_count += 1
                elif p == PRIORITY_ARTIFICIAL:
                    artificial_count += 1
                else:
                    other_count += 1

            # Render annotated frame
            annotated = _draw_detections(img, detections, sam_masks)
            ann_path = out_dir / f"frame_{t_sec:.3f}_annotated.jpg"
            annotated.save(ann_path, quality=88)
            annotated_paths.append(str(ann_path))

            detection_results.append(
                {
                    "frame_path": fp,
                    "t_sec": t_sec,
                    "detections": detections,
                    "sam_available": sam_available,
                    "detection_model": det.get("yolo_model", detector.model_id),
                }
            )

            if (idx + 1) % 10 == 0:
                _log.info(
                    "    ... %d/%d frames processed (objects so far: %d)",
                    idx + 1,
                    n,
                    total_objects,
                )

        except Exception as exc:
            _log.debug("frame %s failed: %s", fp, exc)
            detection_results.append(
                {
                    "frame_path": fp,
                    "t_sec": t_sec,
                    "detections": [],
                    "error": str(exc),
                }
            )

    elapsed = time.time() - t0
    _log.info(
        "Done: %d objects in %d frames (human=%d vehicle=%d artificial=%d other=%d) in %.1fs",
        total_objects,
        n,
        human_count,
        vehicle_count,
        artificial_count,
        other_count,
        elapsed,
    )

    # Save JSON results
    results_json = {
        "model": detector.model_id,
        "sam_enabled": sam_available,
        "n_frames": n,
        "total_objects": total_objects,
        "by_priority": {
            "human": human_count,
            "vehicle": vehicle_count,
            "artificial": artificial_count,
            "other": other_count,
        },
        "frames": [
            {k: v for k, v in r.items() if k != "detections"}
            | {"n_detections": len(r.get("detections", []))}
            for r in detection_results
        ],
        "elapsed_sec": round(elapsed, 2),
    }
    results_path = video_dir / "yolo_sam_results.json"
    write_json_artifact(results_path, results_json, ensure_ascii=False)
    _log.info("  [ok] YOLO+SAM results → %s", results_path)

    # Write comparison markdown
    comparison_md = _write_detection_comparison_md(
        video_dir,
        video_name,
        yolo_results=results_json,
        hf_det_result=det_result,
        n_frames=n,
        elapsed_sec=elapsed,
        sam_available=sam_available,
        annotated_paths=annotated_paths[:8],
    )

    # Release models to free VRAM
    detector.release()
    if sam_predictor is not None:
        sam_predictor.release()

    result.update(
        {
            "skipped": False,
            "detection_results": detection_results,
            "n_frames": n,
            "total_objects": total_objects,
            "human_count": human_count,
            "vehicle_count": vehicle_count,
            "artificial_count": artificial_count,
            "other_count": other_count,
            "elapsed_sec": elapsed,
            "results_json_path": str(results_path),
            "comparison_md_path": comparison_md,
            "sam_enabled": sam_available,
            "annotated_count": len(annotated_paths),
        }
    )
    return result


def _write_detection_comparison_md(
    video_dir: Path,
    video_name: str,
    yolo_results: dict[str, Any],
    hf_det_result: dict[str, Any] | None,
    n_frames: int,
    elapsed_sec: float,
    sam_available: bool,
    annotated_paths: list[str],
) -> str:
    """Write detection_comparison.md comparing YOLO vs HF detector results.

    Returns the path to the written file.
    """
    by_p = yolo_results.get("by_priority", {})
    total_yolo = yolo_results.get("total_objects", 0)
    yolo_model = yolo_results.get("model", "yolo11n.pt")

    # Gather HF detector stats when available
    hf_total = 0
    hf_model = "n/a"
    hf_label_counts: dict[str, int] = {}
    if hf_det_result and not hf_det_result.get("skipped"):
        hf_model = "HF detector"
        for frame_r in hf_det_result.get("detection_results", []):
            for d in frame_r.get("detections", []):
                lbl = d.get("label", "?")
                hf_label_counts[lbl] = hf_label_counts.get(lbl, 0) + 1
                hf_total += 1
        # Map HF labels to priority buckets
        hf_by_priority: dict[str, int] = {
            "human": 0,
            "vehicle": 0,
            "artificial": 0,
            "other": 0,
        }
        for lbl, cnt in hf_label_counts.items():
            p = classify_label_priority(lbl)
            bucket = {
                PRIORITY_HUMAN: "human",
                PRIORITY_VEHICLE: "vehicle",
                PRIORITY_ARTIFICIAL: "artificial",
            }.get(p, "other")
            hf_by_priority[bucket] += cnt

    fps_yolo = n_frames / max(elapsed_sec, 0.01)

    lines = [
        f"# Detection Comparison — {video_name}",
        "",
        f"Generated by `steps_yolo_sam.py` | {n_frames} frames analysed",
        "",
        "## Model Summary",
        "",
        "| Model | Backend | Frames | Total objects | ms/frame |",
        "|-------|---------|--------|--------------|---------|",
        f"| **YOLO11** | `{yolo_model}` | {n_frames} | {total_yolo} | {1000 / fps_yolo:.1f} |",
    ]
    if hf_total > 0:
        lines.append(f"| **HF Detector** | `{hf_model}` | {n_frames} | {hf_total} | — |")
    lines += ["", "## Priority Breakdown", ""]
    lines += [
        "| Priority | Label | YOLO count | HF count |",
        "|----------|-------|-----------|---------|",
        f"| 1 [high] | **Human** | {by_p.get('human', 0)} | {hf_by_priority.get('human', '—') if hf_total else '—'} |",
        f"| 2 🔵 | **Vehicle** | {by_p.get('vehicle', 0)} | {hf_by_priority.get('vehicle', '—') if hf_total else '—'} |",
        f"| 3 🟢 | **Artificial** | {by_p.get('artificial', 0)} | {hf_by_priority.get('artificial', '—') if hf_total else '—'} |",
        f"| 4 ⚫ | **Other** | {by_p.get('other', 0)} | {hf_by_priority.get('other', '—') if hf_total else '—'} |",
    ]
    lines += ["", "## Segmentation", ""]
    if sam_available:
        lines += [
            "SAM2/3 masks generated for each YOLO detection.",
            "Mask area (fraction of image) is stored in `yolo_sam_results.json` per detection.",
        ]
    else:
        lines += [
            "SAM segmentation **not enabled**. Install the project extras with `make venv` or add a SAM backend manually:",
            "```",
            "pip install sam3    # preferred",
            "pip install sam2    # fallback",
            "```",
            "Then re-run with `--yolo` flag (SAM is auto-enabled when `SAM_ENABLED=true`).",
        ]
    lines += ["", "## Priority Color Legend", ""]
    lines += [
        "| Priority | Color | Meaning |",
        "|----------|-------|---------|",
        "| 1 | [high] Red | Human — highest safety priority |",
        "| 2 | 🔵 Blue | Vehicle — dynamic scene actor |",
        "| 3 | 🟢 Green | Artificial object — infrastructure/equipment |",
        "| 4 | ⚫ Grey | Other — natural / uncategorized |",
        "",
    ]
    if annotated_paths:
        lines += ["## Sample Annotated Frames", ""]
        for p in annotated_paths[:6]:
            rel = Path(p).relative_to(video_dir)
            lines.append(f"- `{rel}`")

    lines += [
        "",
        "## Artifacts",
        "",
        "- `yolo_sam_results.json` — full per-frame detection JSON",
        f"- `yolo_sam/frame_*.jpg` — annotated frames ({len(annotated_paths)} written)",
        "",
        f"*Elapsed: {elapsed_sec:.1f}s for {n_frames} frames ({fps_yolo:.1f} fps)*",
    ]

    out_path = video_dir / "detection_comparison.md"
    write_markdown_artifact(out_path, lines)
    _log.info("  [ok] Detection comparison → %s", out_path)
    return str(out_path)
