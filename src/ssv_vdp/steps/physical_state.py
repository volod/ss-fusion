"""Physical scene layer — aggregate depth, object-state fusion, and pose into a
clip-level physical state summary.

Reads from already-computed results (full_fusion_result, depth_result,
gemma_tracking_result) and writes a compact summary that subsequent steps
(SSL fine-tuning, report generation, threat primitives) can consume without
re-running any models.

Output schema (physical_state_summary.json):
    platform_pose_confidence    : float [0, 1] — 1 = tightly constrained pose
    near_field_occupancy_density: float [0, 1] — mean fraction of central image area
                                  occupied by tracked bboxes across all frames
    tracked_object_velocities   : {mean, max, by_label} — in normalised coords/frame
    free_space_estimate         : float [0, 1] — conservative lower bound on clear space
    confirmed_tracks            : int
    mean_bbox_uncertainty       : float — mean bbox Kalman std (normalised coords)
    depth_near_ratio_mean       : float — mean fraction of "near" depth pixels
    skipped                     : bool
"""

import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.fusion.object_state import summarize_object_frame_dicts

from .common import _log, write_json_artifact

# -- Platform confidence -------------------------------------------------------


def _platform_pose_confidence(
    platform_status: str,
    smoothed_trajectory: list[dict[str, Any]],
) -> float:
    """Derive a [0, 1] pose confidence from the Kalman covariance trace.

    Scale is calibrated against _sample_quality thresholds in fusion/summaries.py:
      cov_trace ≤ 10  → "good"       (confidence ≈ 0.5–1.0)
      cov_trace ≤ 40  → "degraded"   (confidence ≈ 0.2–0.5)
      cov_trace  > 40 → "uncertain"  (confidence < 0.2)

    Formula: confidence = 1 / (1 + mean_cov_trace / 10)
    """
    if platform_status != "ok" or not smoothed_trajectory:
        return 0.0
    cov_values = [float(s.get("cov_trace", 0.0)) for s in smoothed_trajectory]
    if not cov_values:
        return 0.0
    mean_cov = sum(cov_values) / len(cov_values)
    return 1.0 / (1.0 + mean_cov / 10.0)


# -- Depth near-ratio ----------------------------------------------------------


def _mean_depth_near_ratio(depth_results: list[dict[str, Any]]) -> float:
    """Average fraction of near pixels across all valid depth frames."""
    ratios = []
    for r in depth_results:
        if r.get("depth_error") or r.get("depth_unavailable") or r.get("depth_disabled"):
            continue
        nr = r.get("near_ratio", r.get("near_frac", None))
        if nr is not None:
            ratios.append(float(nr))
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


# -- Free space estimate -------------------------------------------------------


def _free_space_estimate(near_field_density: float, depth_near_ratio: float) -> float:
    """Conservative free-space lower bound combining object occupancy and depth.

    Uses the larger of the two occupancy signals (pessimistic), discounts the
    depth near_ratio by 0.4 because it includes non-object near geometry
    (ground, walls) that does not block the platform path.
    """
    effective_occupancy = max(near_field_density, depth_near_ratio * 0.4)
    return max(0.0, 1.0 - effective_occupancy)


def _yolo_near_field_density(yolo_sam_result: dict[str, Any]) -> float:
    """Estimate central occupancy from YOLO/SAM detections when fusion is sparse.

    Prefers SAM mask area when available; otherwise falls back to bbox area.
    """
    detection_results = yolo_sam_result.get("detection_results") or []
    if not detection_results:
        return 0.0

    _LO, _HI = 0.3, 0.7
    frame_areas: list[float] = []
    for frame_result in detection_results:
        area = 0.0
        for det in frame_result.get("detections", []):
            bbox = det.get("bbox_norm") or []
            if len(bbox) != 4:
                continue
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5
            if not (_LO <= cx <= _HI and _LO <= cy <= _HI):
                continue
            mask_area = det.get("mask_area_norm")
            if mask_area is not None:
                area += max(0.0, float(mask_area))
            else:
                area += max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        frame_areas.append(min(1.0, area))
    return float(sum(frame_areas) / len(frame_areas)) if frame_areas else 0.0


# -- Main step -----------------------------------------------------------------


def step_physical_state(
    full_fusion_result: dict[str, Any],
    depth_result: dict[str, Any],
    gemma_tracking_result: dict[str, Any],
    yolo_sam_result: dict[str, Any],
    frame_list: list[tuple[str, float]],
    video_dir: Path,
    video_name: str,
) -> dict[str, Any]:
    """Aggregate fusion outputs into an explicit local physical state summary.

    This step is deliberately lightweight — it reads from already-computed
    dicts produced by full_state_fusion, depth estimation, and tracking, and
    does no model inference of its own.

    Args:
        full_fusion_result:      Output of step_full_state_fusion().
        depth_result:            Output of step_depth_estimation().
        gemma_tracking_result:   Output of step_gemma_directed_tracking().
        yolo_sam_result:         Output of step_yolo_sam_detection().
        frame_list:              [(frame_path, t_sec), ...] for the video.
        video_dir:               Per-video output directory.
        video_name:              Human-readable video name (for logging).

    Returns:
        Dict with fields described in module docstring plus "skipped".
    """
    t0 = time.time()

    # Graceful degradation: if full fusion was skipped we still produce a
    # skeleton result so downstream steps do not need to gate on this.
    fusion_skipped = full_fusion_result.get("skipped", True)
    depth_skipped = depth_result.get("skipped", True)
    tracking_skipped = gemma_tracking_result.get("skipped", True)
    yolo_skipped = yolo_sam_result.get("skipped", True)

    if fusion_skipped and depth_skipped and tracking_skipped and yolo_skipped:
        _log.info("  [physical state] all upstream inputs skipped — producing empty summary")
        result: dict[str, Any] = _empty_summary()
        result["skipped"] = True
        _write_json(result, video_dir)
        return result

    # -- Object-state summary --------------------------------------------------
    per_frame_dicts: list[list[dict[str, Any]]] = (
        full_fusion_result.get("per_frame_object_states") or [] if not fusion_skipped else []
    )
    obj_summary = summarize_object_frame_dicts(per_frame_dicts)

    # -- Platform pose confidence ----------------------------------------------
    smoothed_traj = (
        full_fusion_result.get("smoothed_trajectory") or [] if not fusion_skipped else []
    )
    platform_status = (
        full_fusion_result.get("platform_status", "skipped") if not fusion_skipped else "skipped"
    )
    pose_confidence = _platform_pose_confidence(platform_status, smoothed_traj)

    # -- Depth near-ratio -----------------------------------------------------
    depth_near_ratio = (
        _mean_depth_near_ratio(depth_result.get("depth_results") or [])
        if not depth_skipped
        else 0.0
    )

    # -- Detection occupancy fallback / complement ----------------------------
    yolo_near_density = 0.0 if yolo_skipped else _yolo_near_field_density(yolo_sam_result)
    effective_occ = max(obj_summary["near_field_density"], yolo_near_density)

    # -- Free space -----------------------------------------------------------
    free_space = _free_space_estimate(effective_occ, depth_near_ratio)

    # -- Assemble result -------------------------------------------------------
    n_frames = len(frame_list)
    result = {
        "skipped": False,
        "platform_pose_confidence": round(pose_confidence, 4),
        "near_field_occupancy_density": round(effective_occ, 4),
        "tracked_object_velocities": {
            "mean": round(obj_summary["mean_velocity_norm"], 6),
            "max": round(obj_summary["max_velocity_norm"], 6),
            "by_label": {lbl: round(v, 6) for lbl, v in obj_summary["velocity_by_label"].items()},
        },
        "free_space_estimate": round(free_space, 4),
        "confirmed_tracks": obj_summary["confirmed_track_count"],
        "mean_bbox_uncertainty": round(obj_summary["mean_bbox_uncertainty"], 6),
        "depth_near_ratio_mean": round(depth_near_ratio, 4),
        "yolo_near_field_density": round(yolo_near_density, 4),
        "platform_status": platform_status,
        "n_frames": n_frames,
        "elapsed_sec": round(time.time() - t0, 3),
        # provenance flags for the audit step
        "fusion_used": not fusion_skipped,
        "depth_used": not depth_skipped,
        "tracking_used": not tracking_skipped,
        "yolo_used": not yolo_skipped,
    }

    _write_json(result, video_dir)

    _log.info(
        "  [ok] Physical state: pose_conf=%.2f  occ=%.2f  free=%.2f  tracks=%d  depth_near=%.2f",
        pose_confidence,
        obj_summary["near_field_density"],
        free_space,
        obj_summary["confirmed_track_count"],
        depth_near_ratio,
    )
    return result


def _empty_summary() -> dict[str, Any]:
    return {
        "platform_pose_confidence": 0.0,
        "near_field_occupancy_density": 0.0,
        "tracked_object_velocities": {"mean": 0.0, "max": 0.0, "by_label": {}},
        "free_space_estimate": 1.0,
        "confirmed_tracks": 0,
        "mean_bbox_uncertainty": 0.0,
        "depth_near_ratio_mean": 0.0,
        "yolo_near_field_density": 0.0,
        "platform_status": "skipped",
        "n_frames": 0,
        "elapsed_sec": 0.0,
        "fusion_used": False,
        "depth_used": False,
        "tracking_used": False,
        "yolo_used": False,
    }


def _write_json(result: dict[str, Any], video_dir: Path) -> None:
    out = video_dir / "physical_state_summary.json"
    try:
        write_json_artifact(out, result)
    except Exception as exc:
        _log.warning("physical_state: could not write JSON: %s", exc)
