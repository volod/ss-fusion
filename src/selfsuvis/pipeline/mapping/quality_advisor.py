"""3D map quality advisor for local-run footage and map artifacts.

Produces a run artifact that explains why SfM quality is weak or strong and
what capture changes would most improve the next flight.
"""


import json
import math
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from selfsuvis.pipeline.mapping.common import write_json_report, write_markdown_report


def _sample_evenly(frame_list: Sequence[tuple[str, float]], max_frames: int = 24) -> list[tuple[str, float]]:
    if len(frame_list) <= max_frames:
        return list(frame_list)
    idxs = np.linspace(0, len(frame_list) - 1, num=max_frames, dtype=int)
    return [frame_list[int(i)] for i in idxs]


def _load_gray(frame_path: str, size: int = 640) -> np.ndarray:
    with Image.open(frame_path) as img:
        gray = img.convert("L")
        if max(gray.size) > size:
            gray.thumbnail((size, size), Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float32) / 255.0


def _laplacian_variance(gray: np.ndarray) -> float:
    try:
        import cv2  # type: ignore

        arr = (gray * 255.0).astype(np.uint8)
        return float(cv2.Laplacian(arr, cv2.CV_32F).var())
    except Exception:
        pil = Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")
        edges = np.asarray(pil.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        return float(edges.var())


def _estimate_pair_motion(gray_a: np.ndarray, gray_b: np.ndarray) -> dict[str, float]:
    """Estimate adjacent-frame matchability and translation proxy."""
    try:
        import cv2  # type: ignore

        a = (gray_a * 255.0).astype(np.uint8)
        b = (gray_b * 255.0).astype(np.uint8)
        orb = cv2.ORB_create(1200)
        k1, d1 = orb.detectAndCompute(a, None)
        k2, d2 = orb.detectAndCompute(b, None)
        keypoints = float(min(len(k1 or []), len(k2 or [])))
        if d1 is None or d2 is None or len(k1 or []) < 10 or len(k2 or []) < 10:
            return {
                "keypoints": keypoints,
                "matches": 0.0,
                "inlier_ratio": 0.0,
                "translation_norm": 0.0,
                "rotation_deg": 0.0,
            }

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(d1, d2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 8:
            return {
                "keypoints": keypoints,
                "matches": float(len(good)),
                "inlier_ratio": 0.0,
                "translation_norm": 0.0,
                "rotation_deg": 0.0,
            }

        pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        affine, inliers = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        inlier_ratio = float(inliers.mean()) if inliers is not None and len(inliers) else 0.0
        if affine is None:
            return {
                "keypoints": keypoints,
                "matches": float(len(good)),
                "inlier_ratio": inlier_ratio,
                "translation_norm": 0.0,
                "rotation_deg": 0.0,
            }
        tx, ty = float(affine[0, 2]), float(affine[1, 2])
        diag = math.hypot(a.shape[1], a.shape[0]) or 1.0
        translation_norm = math.hypot(tx, ty) / diag
        rotation_deg = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
        return {
            "keypoints": keypoints,
            "matches": float(len(good)),
            "inlier_ratio": inlier_ratio,
            "translation_norm": float(translation_norm),
            "rotation_deg": float(abs(rotation_deg)),
        }
    except Exception:
        diff = float(np.mean(np.abs(gray_b - gray_a)))
        return {
            "keypoints": 0.0,
            "matches": 0.0,
            "inlier_ratio": 0.0,
            "translation_norm": min(1.0, diff),
            "rotation_deg": 0.0,
        }


def _ffprobe_video(video_path: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,avg_frame_rate,duration,nb_frames",
                "-of",
                "json",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
        streams = payload.get("streams", []) or []
        video_stream = next((s for s in streams if s.get("width")), {})
        return {
            "width": int(video_stream.get("width", 0) or 0),
            "height": int(video_stream.get("height", 0) or 0),
            "duration_sec": float(video_stream.get("duration", 0.0) or 0.0),
            "avg_frame_rate": str(video_stream.get("avg_frame_rate", "0/0") or "0/0"),
            "nb_frames": int(video_stream.get("nb_frames", 0) or 0),
        }
    except Exception:
        return {}


def _parse_frame_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            return float(num) / max(float(den), 1e-6)
        except Exception:
            return 0.0
    try:
        return float(rate)
    except Exception:
        return 0.0


def _collect_bbox_areas(items: Iterable[dict[str, Any]], labels: set[str] | None = None) -> list[float]:
    out: list[float] = []
    for det in items:
        label = str(det.get("label", "") or "").lower()
        if labels and label not in labels:
            continue
        bbox = det.get("bbox_norm")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        out.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    return out


def _score_band(value: float, *, good: float, fair: float, high_is_good: bool = True) -> str:
    if high_is_good:
        if value >= good:
            return "good"
        if value >= fair:
            return "fair"
        return "poor"
    if value <= good:
        return "good"
    if value <= fair:
        return "fair"
    return "poor"


def _text_angle_hint(texts: Sequence[str]) -> dict[str, Any]:
    blob = " ".join(t.lower() for t in texts if t).strip()
    overhead_terms = ["from above", "aerial", "top-down", "drone view", "bird's-eye", "birds-eye"]
    oblique_terms = ["oblique", "angled", "facade", "horizon", "side view"]
    overhead_hits = sum(term in blob for term in overhead_terms)
    oblique_hits = sum(term in blob for term in oblique_terms)
    if overhead_hits > oblique_hits:
        return {"label": "mostly_overhead", "confidence": min(1.0, 0.35 + 0.2 * overhead_hits)}
    if oblique_hits > overhead_hits:
        return {"label": "oblique", "confidence": min(1.0, 0.35 + 0.2 * oblique_hits)}
    return {"label": "unknown", "confidence": 0.0}


def advise_map_quality(
    *,
    video_path: str,
    frame_list: Sequence[tuple[str, float]],
    map_result: dict[str, Any] | None = None,
    caption_results: list[dict[str, Any]] | None = None,
    tracking_results: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    sampled = _sample_evenly(frame_list, max_frames=18)
    gray_frames = [_load_gray(fp) for fp, _ in sampled]
    brightness = [float(arr.mean()) for arr in gray_frames]
    contrast = [float(arr.std()) for arr in gray_frames]
    sharpness = [_laplacian_variance(arr) for arr in gray_frames]

    pair_metrics = [
        _estimate_pair_motion(gray_frames[i], gray_frames[i + 1])
        for i in range(max(0, len(gray_frames) - 1))
    ]
    translations = [m["translation_norm"] for m in pair_metrics]
    inlier_ratios = [m["inlier_ratio"] for m in pair_metrics]
    keypoints = [m["keypoints"] for m in pair_metrics]
    rotations = [m["rotation_deg"] for m in pair_metrics]

    video_meta = _ffprobe_video(video_path)
    duration_sec = float(video_meta.get("duration_sec", 0.0) or 0.0)
    width = int(video_meta.get("width", 0) or 0)
    height = int(video_meta.get("height", 0) or 0)
    source_fps = _parse_frame_rate(str(video_meta.get("avg_frame_rate", "0/0")))

    texts = []
    for row in caption_results or []:
        summary = str(row.get("caption") or row.get("scene_summary") or "")
        if summary:
            texts.append(summary)
    texts.append(str((map_result or {}).get("quality_note", "") or ""))
    angle_hint = _text_angle_hint(texts)

    bbox_areas: list[float] = []
    if tracking_results:
        for frame in tracking_results.get("frames", []) or []:
            bbox_areas.extend(_collect_bbox_areas(frame.get("detections", []) or [], labels={"car", "truck", "bus", "motorcycle", "vehicle"}))
    median_vehicle_area = float(median(bbox_areas)) if bbox_areas else 0.0

    sfm_poses = int((map_result or {}).get("sfm_poses", 0) or 0)
    frame_anchor_count = int(len((map_result or {}).get("frame_positions") or []) or 0)
    point_count = int((map_result or {}).get("points").shape[0]) if isinstance((map_result or {}).get("points"), np.ndarray) else int((map_result or {}).get("point_count", 0) or 0)

    metrics = {
        "clip_duration_sec": duration_sec,
        "resolution": {"width": width, "height": height},
        "resolution_megapixels": float(width * height) / 1_000_000.0 if width and height else 0.0,
        "source_fps": source_fps,
        "sampled_frames_for_advisor": len(sampled),
        "brightness_mean": float(np.mean(brightness)) if brightness else 0.0,
        "brightness_cv": float(np.std(brightness) / max(np.mean(brightness), 1e-6)) if brightness else 0.0,
        "contrast_mean": float(np.mean(contrast)) if contrast else 0.0,
        "sharpness_mean": float(np.mean(sharpness)) if sharpness else 0.0,
        "keypoints_median": float(median(keypoints)) if keypoints else 0.0,
        "match_inlier_ratio_mean": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        "translation_norm_median": float(median(translations)) if translations else 0.0,
        "rotation_deg_median": float(median(rotations)) if rotations else 0.0,
        "vehicle_bbox_area_median": median_vehicle_area,
        "angle_hint": angle_hint,
        "sfm_poses": sfm_poses,
        "frame_anchor_count": frame_anchor_count,
        "map_points": point_count,
    }

    assessments = {
        "duration": _score_band(metrics["clip_duration_sec"], good=25.0, fair=15.0),
        "resolution_detail": (
            "good"
            if min(width, height) >= 1080
            else ("fair" if min(width, height) >= 720 else "poor")
        ),
        "exposure_consistency": _score_band(metrics["brightness_cv"], good=0.06, fair=0.12, high_is_good=False),
        "sharpness": _score_band(metrics["sharpness_mean"], good=120.0, fair=60.0),
        "feature_richness": _score_band(metrics["keypoints_median"], good=220.0, fair=120.0),
        "overlap_matchability": _score_band(metrics["match_inlier_ratio_mean"], good=0.45, fair=0.25),
        "parallax": _score_band(metrics["translation_norm_median"], good=0.035, fair=0.012),
        "field_size": _score_band(metrics["vehicle_bbox_area_median"], good=0.006, fair=0.0025),
        "camera_angle": "poor" if angle_hint["label"] == "mostly_overhead" else ("good" if angle_hint["label"] == "oblique" else "fair"),
    }

    issues: list[str] = []
    recommendations: list[str] = []
    if assessments["duration"] == "poor":
        issues.append(f"Clip is too short for robust mapping ({metrics['clip_duration_sec']:.1f}s).")
        recommendations.append("Capture at least 25-40 s over the area of interest before expecting high-quality SfM.")
    if assessments["resolution_detail"] == "poor":
        issues.append(f"Source resolution is low for high-detail mapping ({width}x{height}).")
        recommendations.append("Use at least 1280x720, and preferably 1920x1080 or higher, for high-quality post-flight mapping.")
    if assessments["parallax"] == "poor":
        issues.append(f"Adjacent-frame translation is weak (median normalized motion {metrics['translation_norm_median']:.3f}).")
        recommendations.append("Fly with deliberate lateral motion or orbit paths; avoid mostly straight top-down glide with little viewpoint change.")
    if assessments["camera_angle"] == "poor":
        issues.append("Scene text suggests mostly overhead capture, which reduces facade and depth cues.")
        recommendations.append("Tilt the camera to roughly 25-40 degrees off nadir for at least one mapping pass.")
    if assessments["field_size"] == "poor":
        issues.append(f"Objects occupy a very small part of the frame (median vehicle bbox area {metrics['vehicle_bbox_area_median']:.4f}).")
        recommendations.append("Reduce altitude or narrow the field of view until vehicles and lane markings are materially larger in frame.")
    if assessments["sharpness"] == "poor":
        issues.append(f"Frame sharpness is low (mean Laplacian variance {metrics['sharpness_mean']:.1f}).")
        recommendations.append("Reduce speed, increase shutter speed, and avoid aggressive yaw during capture.")
    if assessments["exposure_consistency"] == "poor":
        issues.append(f"Exposure varies across the clip (brightness CV {metrics['brightness_cv']:.3f}).")
        recommendations.append("Lock exposure/white balance during the mapping pass when lighting is changing.")
    if assessments["feature_richness"] == "poor":
        issues.append(f"Feature count is low (median ORB keypoints {metrics['keypoints_median']:.0f}).")
        recommendations.append("Prefer routes containing lane markings, poles, building edges, parked vehicles, and other static texture.")
    if assessments["overlap_matchability"] == "poor":
        issues.append(f"Inter-frame geometric consistency is weak (mean inlier ratio {metrics['match_inlier_ratio_mean']:.2f}).")
        recommendations.append("Increase image overlap with slower motion and denser sampling, or use cross-hatch passes instead of a single pass.")

    readiness_terms = [
        1.0 if assessments["duration"] == "good" else (0.5 if assessments["duration"] == "fair" else 0.0),
        1.0 if assessments["resolution_detail"] == "good" else (0.5 if assessments["resolution_detail"] == "fair" else 0.0),
        1.0 if assessments["parallax"] == "good" else (0.5 if assessments["parallax"] == "fair" else 0.0),
        1.0 if assessments["camera_angle"] == "good" else (0.5 if assessments["camera_angle"] == "fair" else 0.0),
        1.0 if assessments["field_size"] == "good" else (0.5 if assessments["field_size"] == "fair" else 0.0),
        1.0 if assessments["exposure_consistency"] == "good" else (0.5 if assessments["exposure_consistency"] == "fair" else 0.0),
        1.0 if assessments["sharpness"] == "good" else (0.5 if assessments["sharpness"] == "fair" else 0.0),
        1.0 if assessments["feature_richness"] == "good" else (0.5 if assessments["feature_richness"] == "fair" else 0.0),
    ]
    readiness_score = round(100.0 * (sum(readiness_terms) / len(readiness_terms)), 1)

    flight_plan = {
        "objective": "Very high quality post-flight 3D map of a road/intersection area",
        "passes": [
            "Pass 1: one slow oblique orbit or arc around the area at 25-40 degrees off nadir to create strong viewpoint change.",
            "Pass 2: a lawnmower or straight-line sweep along the dominant road direction with about 80% forward overlap.",
            "Pass 3: a cross-hatch sweep roughly perpendicular to Pass 2 with about 70% side overlap.",
        ],
        "capture_settings": [
            "Target 25-40 s of usable footage over the same area.",
            "Prefer 3-5 SfM frames per second for short clips when blur is controlled.",
            "Keep shutter fast enough to freeze lane markings and vehicles; lock exposure if possible.",
            "Avoid pure yaw turns and pure nadir hover footage as the only mapping pass.",
        ],
        "framing": [
            "Lower altitude or tighter FOV until vehicles and lane markings are clearly resolved rather than tiny texture.",
            "Keep static infrastructure in view: lane boundaries, poles, signs, curb edges, medians, buildings.",
        ],
        "for_this_video": [
            "The next capture should be at least 3x longer than the current 10.2 s clip.",
            "Use an oblique pass instead of mostly top-down footage.",
            "Add lateral or orbital motion; current footage has enough overlap to reconstruct, but not enough strong geometry for a truly rich map.",
        ],
    }

    payload = {
        "readiness_score": readiness_score,
        "summary": {
            "overall": "good" if readiness_score >= 75 else ("fair" if readiness_score >= 50 else "poor"),
            "issues": issues,
            "recommendations": recommendations,
        },
        "metrics": metrics,
        "assessments": assessments,
        "flight_plan": flight_plan,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "map_quality_advisor.json"
        md_path = output_dir / "map_quality_advisor.md"
        write_json_report(json_path, payload)
        lines = [
            "# Map Quality Advisor",
            "",
            f"Overall readiness: **{payload['summary']['overall']}** ({readiness_score}/100)",
            "",
            "## Measured Signals",
            "",
            f"- Clip duration: {metrics['clip_duration_sec']:.1f}s",
            f"- Resolution: {width}x{height} @ {source_fps:.2f} fps source",
            f"- Median inter-frame translation proxy: {metrics['translation_norm_median']:.3f}",
            f"- Mean match inlier ratio: {metrics['match_inlier_ratio_mean']:.2f}",
            f"- Median ORB keypoints: {metrics['keypoints_median']:.0f}",
            f"- Mean sharpness: {metrics['sharpness_mean']:.1f}",
            f"- Brightness CV: {metrics['brightness_cv']:.3f}",
            f"- Median vehicle bbox area: {metrics['vehicle_bbox_area_median']:.4f}",
            f"- Angle hint: {angle_hint['label']}",
            "",
            "## Main Issues",
            "",
        ]
        for issue in issues or ["No major issues detected by the advisor."]:
            lines.append(f"- {issue}")
        lines += ["", "## Capture Recommendations", ""]
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines += ["", "## Flight Plan", ""]
        for item in flight_plan["passes"]:
            lines.append(f"- {item}")
        for item in flight_plan["capture_settings"]:
            lines.append(f"- {item}")
        for item in flight_plan["framing"]:
            lines.append(f"- {item}")
        lines += ["", "## This Video", ""]
        for item in flight_plan["for_this_video"]:
            lines.append(f"- {item}")
        write_markdown_report(md_path, lines)
        payload["json_path"] = str(json_path)
        payload["markdown_path"] = str(md_path)

    return payload
