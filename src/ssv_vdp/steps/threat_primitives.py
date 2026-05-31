"""Threat primitive layer — structured, evidence-gated threat signals.

Produces structured primitive types from physical state, field state, and
upstream fusion outputs.
A primitive is only emitted when at least two independent evidence sources agree,
preventing single-sensor false alarms.

Each primitive schema:
    type                : str   — "collision_risk" | "visibility_degradation" |
                                   "track_anomaly" | "pose_uncertain" |
                                   "rf_anomaly"
    score               : float [0, 1] — severity (1 = worst)
    uncertainty         : float [0, 1] — confidence bounds on the score
    spatial_support     : list[str]  — frame paths where the condition holds
    temporal_persistence: int  — max consecutive frames the condition persists
    evidence_sources    : list[str]  — independent signals that agree (≥ 2 to emit)

Output artifact: threat_primitives.json
"""

import json
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger

from .threat_contradictions import contradiction_signals_for_threat, summarize_contradictions

_log = get_logger("pipeline.local")

# -- Evidence thresholds -------------------------------------------------------

_COLL_OCC_THRESH = 0.15  # near-field occupancy fraction to flag
_COLL_VEL_THRESH = 0.02  # mean normalised velocity/frame to flag
_COLL_FREE_THRESH = 0.70  # free space < this flags constraint
_VIS_DEPTH_FAIL_THRESH = 0.15  # fraction of depth frames that fail
_VIS_SSIM_QUALITY_THRESH = 0.20  # mean SSIM-diff below this implies low visual novelty / blur
_VIS_CAPTION_THRESH = 0.55  # mean caption confidence below this
_TRACK_BREAK_THRESH = 0.08  # track-break rate (breaks / total consecutive pairs)
_TRACK_IOU_DROP_THRESH = 0.25  # same-track IoU below this implies unstable geometry
_TRACK_SHORTLEN_THRESH = 5.0  # mean track length in frames
_POSE_KALMAN_THRESH = 0.40  # Kalman pose confidence below this
_POSE_SFM_FAIL_THRESH = 0.30  # fraction of frames without an SfM pose


# -- Helpers -------------------------------------------------------------------


def _max_run_length(flags: list[bool]) -> int:
    """Return the longest consecutive run of True in *flags*."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _per_frame_near_field_occ(
    per_frame_object_states: list[list[dict[str, Any]]],
    tracking_results: list[dict[str, Any]],
) -> list[tuple[str, float]]:
    """Return (frame_path, near_field_area) per tracked frame."""
    _LO, _HI = 0.3, 0.7
    _ACTIVE = {"confirmed", "smoothed"}
    out: list[tuple[str, float]] = []
    for frame_states, tr in zip(per_frame_object_states, tracking_results):
        fp = tr.get("frame_path", "")
        area = 0.0
        for s in frame_states:
            if s.get("track_state") not in _ACTIVE:
                continue
            bbox = s.get("bbox_norm") or []
            if len(bbox) != 4:
                continue
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5
            if _LO <= cx <= _HI and _LO <= cy <= _HI:
                area += max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        out.append((fp, min(1.0, area)))
    return out


def _track_break_stats(
    tracking_results: list[dict[str, Any]],
) -> tuple[float, float, list[int]]:
    """Return (break_rate, mean_track_length, gap_frame_indices).

    break_rate     = n_breaks / max(1, total_consecutive_pairs)
    mean_track_len = mean number of frames each track_id appears in
    gap_frame_indices = frame indices that fall inside a continuity gap
    """
    appearances: dict[int, list[int]] = {}
    for fi, fr in enumerate(tracking_results):
        for det in fr.get("detections", []):
            tid = int(det.get("track_id", 0) or 0)
            if tid > 0:
                appearances.setdefault(tid, []).append(fi)

    n_breaks = 0
    total_pairs = 0
    gap_indices: set = set()
    for tid, frames in appearances.items():
        frames.sort()
        for i in range(1, len(frames)):
            total_pairs += 1
            gap = frames[i] - frames[i - 1]
            if gap > 1:
                n_breaks += 1
                for gi in range(frames[i - 1] + 1, frames[i]):
                    gap_indices.add(gi)

    break_rate = n_breaks / max(1, total_pairs)
    lengths = [len(v) for v in appearances.values()]
    mean_len = float(sum(lengths) / len(lengths)) if lengths else 0.0
    return break_rate, mean_len, sorted(gap_indices)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _track_iou_drop_stats(
    tracking_results: list[dict[str, Any]],
) -> tuple[float, list[int]]:
    """Return same-track IoU-drop rate and frame indices where drops occur."""
    appearances: dict[int, list[tuple[int, list[float]]]] = {}
    for fi, frame_result in enumerate(tracking_results):
        for det in frame_result.get("detections", []):
            tid = int(det.get("track_id", 0) or 0)
            bbox = det.get("bbox_norm") or []
            if tid > 0 and len(bbox) == 4:
                appearances.setdefault(tid, []).append((fi, list(bbox)))

    drops = 0
    total = 0
    drop_indices: set[int] = set()
    for track_samples in appearances.values():
        track_samples.sort(key=lambda item: item[0])
        for (prev_fi, prev_bbox), (curr_fi, curr_bbox) in zip(track_samples, track_samples[1:]):
            total += 1
            if _bbox_iou(prev_bbox, curr_bbox) < _TRACK_IOU_DROP_THRESH:
                drops += 1
                drop_indices.add(curr_fi)
                drop_indices.add(prev_fi)
    return (drops / total if total else 0.0), sorted(drop_indices)


def _depth_failure_rate(depth_results: list[dict[str, Any]]) -> float:
    """Fraction of depth frames that are errored / unavailable / disabled."""
    if not depth_results:
        return 0.0
    failed = sum(
        1
        for r in depth_results
        if r.get("depth_error") or r.get("depth_unavailable") or r.get("depth_disabled")
    )
    return failed / len(depth_results)


def _caption_confidence_stats(
    caption_results: list[dict[str, Any]],
) -> tuple[float, float, list[str]]:
    """Return (mean_conf, std_conf, low_conf_frame_paths)."""
    confs: list[float] = []
    low_frames: list[str] = []
    for r in caption_results:
        conf = float(r.get("caption_confidence", 0.75) or 0.75)
        fp = str(r.get("frame_path", ""))
        confs.append(conf)
        if conf < _VIS_CAPTION_THRESH:
            low_frames.append(fp)
    if not confs:
        return 0.75, 0.0, []
    mean_c = sum(confs) / len(confs)
    var = sum((c - mean_c) ** 2 for c in confs) / len(confs)
    std_c = var**0.5
    return mean_c, std_c, low_frames


def _ssim_quality_stats(frame_list: list[tuple[str, float]]) -> tuple[float, list[str], int]:
    """Estimate keyframe quality from consecutive SSIM-diff values.

    Low mean SSIM-diff means frames are visually too similar, often due to blur,
    low motion, or poor keyframe separation. This is an inference from the
    existing frame-heuristics signal, not a calibrated vision confidence model.
    """
    if len(frame_list) < 2:
        return 1.0, [], 0

    try:
        import numpy as np
        from PIL import Image

        from selfsuvis.pipeline.media.heuristics import downsample_gray, ssim_diff
    except Exception:
        return 1.0, [], 0

    diffs: list[float] = []
    low_paths: list[str] = []
    low_flags: list[bool] = []
    prev_small = None
    for frame_path, _t_sec in frame_list:
        try:
            img = Image.open(frame_path).convert("RGB")
            arr = np.asarray(img)
            small = downsample_gray(arr, width=64)
        except Exception:
            continue
        if prev_small is None:
            prev_small = small
            continue
        diff = float(ssim_diff(prev_small, small))
        diffs.append(diff)
        is_low = diff < _VIS_SSIM_QUALITY_THRESH
        low_flags.append(is_low)
        if is_low:
            low_paths.append(frame_path)
        prev_small = small
    if not diffs:
        return 1.0, [], 0
    return float(sum(diffs) / len(diffs)), low_paths, _max_run_length(low_flags)


# -- Primitive builders --------------------------------------------------------


def _build_collision_risk(
    physical_state: dict[str, Any],
    per_frame_occ: list[tuple[str, float]],
    frame_list: list[tuple[str, float]],
) -> dict[str, Any] | None:
    occ = float(physical_state.get("near_field_occupancy_density", 0.0))
    vel_m = float(physical_state.get("tracked_object_velocities", {}).get("mean", 0.0))
    free = float(physical_state.get("free_space_estimate", 1.0))
    bbox_u = float(physical_state.get("mean_bbox_uncertainty", 0.0))

    sources: list[str] = []
    if occ > _COLL_OCC_THRESH:
        sources.append("near_field_occupancy")
    if vel_m > _COLL_VEL_THRESH:
        sources.append("object_velocity")
    if free < _COLL_FREE_THRESH:
        sources.append("free_space_estimate")

    if len(sources) < 2:
        return None

    score = min(1.0, occ * 0.50 + min(1.0, vel_m / 0.05) * 0.30 + (1.0 - free) * 0.20)
    uncertainty = min(0.50, bbox_u * 10.0)

    if per_frame_occ:
        threshold = _COLL_OCC_THRESH
        flags = [a > threshold for _, a in per_frame_occ]
        frames = [fp for (fp, a) in per_frame_occ if a > threshold]
        persist = _max_run_length(flags)
    else:
        # No per-frame data: clip-level signal spans the whole video
        frames = [fp for fp, _ in frame_list]
        persist = len(frame_list)

    return {
        "type": "collision_risk",
        "score": round(score, 4),
        "uncertainty": round(uncertainty, 4),
        "spatial_support": frames,
        "temporal_persistence": persist,
        "evidence_sources": sources,
    }


def _build_visibility_degradation(
    depth_results: list[dict[str, Any]],
    caption_results: list[dict[str, Any]],
    frame_list: list[tuple[str, float]],
    field_state: dict[str, Any],
) -> dict[str, Any] | None:
    fail_rate = _depth_failure_rate(depth_results)
    mean_conf, std_conf, low_cap_frames = _caption_confidence_stats(caption_results)
    mean_ssim_quality, low_ssim_frames, ssim_persist = _ssim_quality_stats(frame_list)
    visibility_field = (field_state.get("clip_level_fields") or {}).get("visibility") or {}
    field_mean = float(visibility_field.get("mean", 0.0) or 0.0)
    field_trend = str(visibility_field.get("trend", "stable") or "stable")
    field_support = list(visibility_field.get("support_frames") or [])

    sources: list[str] = []
    if fail_rate > _VIS_DEPTH_FAIL_THRESH:
        sources.append("depth_failure_rate")
    if mean_ssim_quality < _VIS_SSIM_QUALITY_THRESH:
        sources.append("ssim_keyframe_quality")
    if mean_conf < _VIS_CAPTION_THRESH:
        sources.append("caption_confidence")
    if field_mean > 0.25:
        sources.append("visibility_field_intensity")
    if field_trend == "worsening":
        sources.append("visibility_field_trend")

    if len(sources) < 2:
        return None

    score = min(
        1.0,
        fail_rate * 0.30
        + (1.0 - mean_ssim_quality) * 0.20
        + (1.0 - mean_conf) * 0.25
        + field_mean * 0.25,
    )
    uncertainty = min(0.40, std_conf + max(0.0, 0.15 - fail_rate))

    # Spatial support: frames where depth failed
    depth_fail_frames: list[str] = [
        str(r.get("frame_path", ""))
        for r in depth_results
        if r.get("depth_error") or r.get("depth_unavailable") or r.get("depth_disabled")
    ]
    support = list(
        dict.fromkeys(depth_fail_frames + low_cap_frames + low_ssim_frames + field_support)
    )

    # Temporal persistence across depth results
    depth_flags = [
        bool(r.get("depth_error") or r.get("depth_unavailable") or r.get("depth_disabled"))
        for r in depth_results
    ]
    persist = max(
        _max_run_length(depth_flags) if depth_flags else 0,
        ssim_persist,
    ) or len(frame_list)

    return {
        "type": "visibility_degradation",
        "score": round(score, 4),
        "uncertainty": round(uncertainty, 4),
        "spatial_support": support,
        "temporal_persistence": persist,
        "evidence_sources": sources,
    }


def _build_rf_anomaly(
    field_state: dict[str, Any],
    frame_list: list[tuple[str, float]],
) -> dict[str, Any] | None:
    rf_field = (field_state.get("clip_level_fields") or {}).get("rf_interference") or {}
    if not rf_field:
        return None
    score = float(rf_field.get("mean", 0.0) or 0.0)
    sources = list(rf_field.get("evidence_sources") or [])
    if str(rf_field.get("trend", "stable")) == "worsening":
        sources.append("rf_field_trend")
    # Enforce the same two-source rule.
    sources = list(dict.fromkeys([s for s in sources if s]))
    if score < 0.30 or len(sources) < 2:
        return None
    support = list(rf_field.get("support_frames") or [fp for fp, _ in frame_list])
    uncertainty = float(rf_field.get("uncertainty", 0.45) or 0.45)
    persistence = max(1, len(support))
    return {
        "type": "rf_anomaly",
        "score": round(score, 4),
        "uncertainty": round(min(0.55, uncertainty), 4),
        "spatial_support": support,
        "temporal_persistence": persistence,
        "evidence_sources": sources,
    }


def _build_track_anomaly(
    tracking_results: list[dict[str, Any]],
    physical_state: dict[str, Any],
) -> dict[str, Any] | None:
    if not tracking_results:
        return None

    break_rate, mean_len, gap_indices = _track_break_stats(tracking_results)
    iou_drop_rate, iou_drop_indices = _track_iou_drop_stats(tracking_results)

    sources: list[str] = []
    if break_rate > _TRACK_BREAK_THRESH:
        sources.append("track_breaks")
    if iou_drop_rate > _TRACK_BREAK_THRESH:
        sources.append("iou_drop_events")
    if mean_len < _TRACK_SHORTLEN_THRESH and mean_len > 0:
        sources.append("short_track_length")

    if len(sources) < 2:
        return None

    score = min(
        1.0,
        min(1.0, break_rate / 0.30) * 0.60
        + min(1.0, iou_drop_rate / 0.40) * 0.25
        + min(1.0, max(0.0, _TRACK_SHORTLEN_THRESH - mean_len) / _TRACK_SHORTLEN_THRESH) * 0.15,
    )
    # Uncertainty: if tracking was used as evidence in physical state, lower uncertainty
    tracking_used = bool(physical_state.get("tracking_used", False))
    uncertainty = 0.20 if tracking_used else 0.40

    support_indices = sorted(set(gap_indices) | set(iou_drop_indices))
    gap_paths = [
        tr.get("frame_path", "")
        for i, tr in enumerate(tracking_results)
        if i in set(support_indices)
    ]
    support_index_set = set(support_indices)
    flags = [i in support_index_set for i in range(len(tracking_results))]
    persist = _max_run_length(flags)

    return {
        "type": "track_anomaly",
        "score": round(score, 4),
        "uncertainty": round(uncertainty, 4),
        "spatial_support": [p for p in gap_paths if p],
        "temporal_persistence": persist,
        "evidence_sources": sources,
    }


def _build_pose_uncertain(
    physical_state: dict[str, Any],
    sfm_poses: int,
    map_degraded: bool,
    frame_list: list[tuple[str, float]],
) -> dict[str, Any] | None:
    pose_conf = float(physical_state.get("platform_pose_confidence", 0.0))
    n_frames = len(frame_list)
    sfm_fail_rate = 1.0 - (sfm_poses / max(1, n_frames))

    sources: list[str] = []
    if pose_conf < _POSE_KALMAN_THRESH:
        sources.append("kalman_pose_confidence")
    if map_degraded:
        sources.append("sfm_quality_degraded")
    if sfm_fail_rate > _POSE_SFM_FAIL_THRESH:
        sources.append("sfm_failure_rate")

    if len(sources) < 2:
        return None

    score = min(1.0, (1.0 - pose_conf) * 0.60 + min(1.0, sfm_fail_rate / 0.60) * 0.40)
    # Uncertainty decreases as more independent sources agree
    uncertainty = max(0.10, 0.40 - 0.10 * len(sources))

    return {
        "type": "pose_uncertain",
        "score": round(score, 4),
        "uncertainty": round(uncertainty, 4),
        "spatial_support": [fp for fp, _ in frame_list],
        "temporal_persistence": n_frames,
        "evidence_sources": sources,
    }


# -- Overall threat level ------------------------------------------------------


def _threat_level(primitives: list[dict[str, Any]]) -> str:
    if not primitives:
        return "none"
    max_score = max(p["score"] for p in primitives)
    if max_score >= 0.70:
        return "high"
    if max_score >= 0.50:
        return "medium"
    if max_score >= 0.25:
        return "low"
    return "none"


# -- Main step -----------------------------------------------------------------


def step_threat_primitives(
    physical_state_result: dict[str, Any],
    field_state_result: dict[str, Any],
    depth_result: dict[str, Any],
    caption_results: list[dict[str, Any]],
    unidrive_result: dict[str, Any],
    gemma_tracking_result: dict[str, Any],
    full_fusion_result: dict[str, Any],
    frame_list: list[tuple[str, float]],
    sfm_poses: int,
    map_degraded: bool,
    video_dir: Path,
    video_name: str,
) -> dict[str, Any]:
    """Compute structured threat primitives from physical state and upstream fusion outputs.

    Each primitive is only emitted when at least two independent evidence sources
    agree.  Writes threat_primitives.json to video_dir.

    Args:
        physical_state_result:  Output of step_physical_state().
        field_state_result:     Output of step_field_state().
        depth_result:           Output of step_depth_estimation().
        caption_results:        Per-frame Florence caption list.
        unidrive_result:        Output of step_unidrive_analysis().
        gemma_tracking_result:  Output of step_gemma_directed_tracking().
        full_fusion_result:     Output of step_full_state_fusion().
        frame_list:             [(frame_path, t_sec), ...] for the video.
        sfm_poses:              Number of SfM-registered poses (from stats).
        map_degraded:           True when 3D map quality is degraded.
        video_dir:              Per-video output directory.
        video_name:             Human-readable video name (for logging).

    Returns:
        Dict with fields: primitives, summary, skipped, elapsed_sec.
    """
    t0 = time.time()

    phys_skipped = physical_state_result.get("skipped", True)
    depth_skipped = depth_result.get("skipped", True)
    tracking_skipped = gemma_tracking_result.get("skipped", True)
    fusion_skipped = full_fusion_result.get("skipped", True)

    if phys_skipped and depth_skipped and tracking_skipped and fusion_skipped:
        _log.info("  [threat primitives] all upstream inputs skipped — no primitives")
        result: dict[str, Any] = _empty_result()
        result["skipped"] = True
        _write_json(result, video_dir)
        return result

    physical_state = physical_state_result if not phys_skipped else {}
    field_state = field_state_result if not field_state_result.get("skipped", True) else {}
    unidrive_rows = [
        row
        for row in (unidrive_result.get("results") or [])
        if not row.get("service_unavailable") and not row.get("parse_error")
    ]

    # Per-frame object occupancy (requires both tracking and fusion)
    tracking_results: list[dict[str, Any]] = (
        gemma_tracking_result.get("tracking_results") or [] if not tracking_skipped else []
    )
    per_frame_obj = (
        full_fusion_result.get("per_frame_object_states") or [] if not fusion_skipped else []
    )
    # Align lengths for safe zip
    _min_len = min(len(per_frame_obj), len(tracking_results))
    per_frame_occ = _per_frame_near_field_occ(per_frame_obj[:_min_len], tracking_results[:_min_len])

    depth_results_list: list[dict[str, Any]] = (
        depth_result.get("depth_results") or [] if not depth_skipped else []
    )

    primitives: list[dict[str, Any]] = []

    # -- Collision risk --------------------------------------------------------
    p = _build_collision_risk(physical_state, per_frame_occ, frame_list)
    if p:
        primitives.append(p)

    # -- Visibility degradation ------------------------------------------------
    p = _build_visibility_degradation(
        depth_results_list, caption_results or [], frame_list, field_state
    )
    if p:
        primitives.append(p)

    # -- RF anomaly -----------------------------------------------------------
    p = _build_rf_anomaly(field_state, frame_list)
    if p:
        primitives.append(p)

    # -- Track anomaly ---------------------------------------------------------
    p = _build_track_anomaly(tracking_results, physical_state)
    if p:
        primitives.append(p)

    # -- Pose uncertain --------------------------------------------------------
    p = _build_pose_uncertain(physical_state, sfm_poses, map_degraded, frame_list)
    if p:
        primitives.append(p)

    contradiction_signals: list[dict[str, Any]] = []
    for primitive in primitives:
        contradiction_signals.extend(
            contradiction_signals_for_threat(
                str(primitive.get("type", "")),
                primitive,
                unidrive_rows,
                physical_state,
            )
        )

    if any(p.get("type") == "visibility_degradation" for p in primitives):
        fail_rate = _depth_failure_rate(depth_results_list)
        mean_conf, _std_conf, _low_frames = _caption_confidence_stats(caption_results or [])
        if fail_rate > _VIS_DEPTH_FAIL_THRESH and mean_conf >= 0.70:
            contradiction_signals.append(
                {
                    "pattern": "caption_confidence_vs_depth_failure",
                    "description": "caption confidence stays high while depth failures persist",
                    "source_a": "Florence-2 captioning",
                    "source_b": "depth estimation",
                    "frame_id": Path(str(frame_list[0][0])).name if frame_list else None,
                    "severity": 0.26,
                }
            )

    if any(p.get("type") == "track_anomaly" for p in primitives):
        iou_drop_rate, drop_indices = _track_iou_drop_stats(tracking_results)
        confirmed_tracks = int(physical_state.get("confirmed_tracks", 0) or 0)
        tracking_used = bool(physical_state.get("tracking_used", False))
        if iou_drop_rate > _TRACK_BREAK_THRESH and (tracking_used or confirmed_tracks >= 3):
            frame_id = None
            if drop_indices:
                idx = drop_indices[0]
                if 0 <= idx < len(tracking_results):
                    frame_id = Path(str(tracking_results[idx].get("frame_path", ""))).name or None
            contradiction_signals.append(
                {
                    "pattern": "tracking_persistence_vs_iou_break",
                    "description": "stable tracked-object persistence coexists with repeated same-track IoU breaks",
                    "source_a": "object-state persistence",
                    "source_b": "RF-DETR IoU continuity",
                    "frame_id": frame_id,
                    "severity": 0.28,
                }
            )

    n = len(primitives)
    types = sorted({p["type"] for p in primitives})
    level = _threat_level(primitives)
    contradiction_summary = summarize_contradictions(contradiction_signals=contradiction_signals)

    result = {
        "skipped": False,
        "primitives": primitives,
        "contradiction_signals": contradiction_signals,
        "summary": {
            "n_primitives": n,
            "types_detected": types,
            "overall_threat_level": level,
            "disagreement_count": int(contradiction_summary.get("disagreement_count", 0)),
            "disagreement_rate": float(contradiction_summary.get("disagreement_rate", 0.0)),
            "source_pair_conflicts": contradiction_summary.get("source_pair_conflicts", []),
            "trust_penalty": float(contradiction_summary.get("trust_penalty", 0.0)),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }

    _write_json(result, video_dir)

    _log.info(
        "  [ok] Threat primitives: %d emitted  level=%s  types=%s",
        n,
        level,
        types or "none",
    )
    return result


def _empty_result() -> dict[str, Any]:
    return {
        "primitives": [],
        "contradiction_signals": [],
        "summary": {
            "n_primitives": 0,
            "types_detected": [],
            "overall_threat_level": "none",
            "disagreement_count": 0,
            "disagreement_rate": 0.0,
            "source_pair_conflicts": [],
            "trust_penalty": 0.0,
        },
        "elapsed_sec": 0.0,
    }


def _write_json(result: dict[str, Any], video_dir: Path) -> None:
    out = video_dir / "threat_primitives.json"
    try:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        _log.warning("threat_primitives: could not write JSON: %s", exc)
