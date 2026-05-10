"""Probabilistic object-state fusion and association.

Extends the basic IoU-based RF-DETR tracker with:
  1. Per-track Kalman filter (constant-velocity in image space)
  2. Mahalanobis-distance gating (chi² 4-DOF, p=0.99) for robust association
  3. Hungarian optimal assignment (scipy.optimize.linear_sum_assignment)
  4. Track lifecycle: tentative → confirmed → deleted
  5. RTS backward smoother over confirmed track histories
  6. Per-label speed prior clamping from semantic prior

Input schema (RF-DETR tracking_results):
    [
        {
            "frame_path": str,
            "t_sec": float,
            "detections": [{"label": str, "confidence": float,
                             "bbox_norm": [x1, y1, x2, y2], ...}],
        },
        ...
    ]

Output: ObjectFusionResult with per-frame smoothed track states.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.fusion.filters.object_filter import (
    ObjectFilterHistory,
    ObjectKalmanFilter,
    _state_to_bbox,
)
from selfsuvis.pipeline.fusion.filters.rts_smoother import (
    FilteredStep,
    rts_smooth,
)
from selfsuvis.pipeline.fusion.semantic_priors import SemanticPrior

logger = get_logger(__name__)

# Track lifecycle thresholds
_CONFIRM_HITS = 3  # frames needed to become confirmed
_MAX_MISS_FRAMES = 5  # consecutive misses before deletion
# Observation noise (normalised bbox coords)
_OBS_NOISE = 0.005
# Process noise scale for object RTS smoother (normalised coords / frame)
_OBJ_PROC_POS_STD = 0.02
_OBJ_PROC_VEL_STD = 0.10
# Cost matrix fill value for infeasible assignments
_INF_COST = 1e9


@dataclass
class ObjectStateSample:
    """Smoothed state estimate for one object at one frame."""

    track_id: int
    label: str
    t_sec: float
    bbox_norm: list[float]  # [x1, y1, x2, y2] smoothed
    velocity_norm: list[float]  # [vcx, vcy] normalised/frame
    bbox_std: list[float]  # [std_cx, std_cy, std_w, std_h]
    confidence: float
    track_state: str  # "tentative" | "confirmed" | "smoothed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "t_sec": self.t_sec,
            "bbox_norm": self.bbox_norm,
            "velocity_norm": self.velocity_norm,
            "bbox_std": self.bbox_std,
            "confidence": self.confidence,
            "track_state": self.track_state,
        }


@dataclass
class ObjectFusionResult:
    enabled: bool
    status: str
    reason: str = ""
    source: str = "object_kalman_mahalanobis_v1"
    track_count: int = 0
    confirmed_tracks: int = 0
    per_frame: list[list[ObjectStateSample]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
            "track_count": self.track_count,
            "confirmed_tracks": self.confirmed_tracks,
            "diagnostics": self.diagnostics,
            "per_frame": [[s.to_dict() for s in frame_samples] for frame_samples in self.per_frame],
        }

    def to_local_state_summary(self) -> dict[str, Any]:
        """Collapse per-frame object states into a clip-level physical belief.

        Returns a dict with:
            confirmed_track_count  — number of Kalman-confirmed tracks
            mean_velocity_norm     — mean speed across all confirmed/smoothed samples
            max_velocity_norm      — peak speed observed in the clip
            velocity_by_label      — {label: mean_speed} for each object class
            mean_bbox_uncertainty  — mean of bbox_std entries (position uncertainty)
            near_field_density     — mean fraction of central image area (cx,cy ∈ [0.3,0.7])
                                     occupied by tracked bboxes
        """
        frame_dicts = [[s.to_dict() for s in frame_samples] for frame_samples in self.per_frame]
        return summarize_object_frame_dicts(frame_dicts)


def _build_cost_matrix(
    tracks: list[ObjectKalmanFilter],
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Build cost matrix [n_tracks × n_dets] using Mahalanobis distance².
    Cells exceeding the gate threshold are set to _INF_COST (infeasible).
    """
    n_t = len(tracks)
    n_d = len(detections)
    cost = np.full((n_t, n_d), _INF_COST, dtype=np.float64)
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            bbox = det.get("bbox_norm", [])
            if len(bbox) < 4:
                continue
            d2 = trk.mahalanobis_distance_sq(bbox)
            if trk.is_gated(bbox):
                cost[i, j] = d2
    return cost


def _apply_speed_prior(
    kf: ObjectKalmanFilter,
    speed_priors: dict[str, float],
    dt: float,
) -> None:
    """Clamp the KF velocity estimate to the per-label speed cap."""
    label = kf.label.lower()
    max_speed_px_frame = speed_priors.get(label, speed_priors.get("vehicle", 40.0))
    # Convert m/s to normalised-frame units is hard without knowing image scale,
    # so we apply a heuristic: cap at max_speed_norm_per_frame = 0.15 (image width)
    # scaled by the label-specific speed ratio relative to "car" (40 m/s).
    car_ref = 40.0
    norm_cap = 0.15 * (max_speed_px_frame / max(car_ref, 1.0))
    v = kf.x[4:6]
    speed = float(np.linalg.norm(v))
    if speed > norm_cap and speed > 1e-6:
        kf.x[4:6] = v * (norm_cap / speed)


def run_object_state_fusion(
    tracking_results: Sequence[dict[str, Any]],
    prior: SemanticPrior | None = None,
    obs_noise: float = _OBS_NOISE,
    confirm_hits: int = _CONFIRM_HITS,
    max_miss_frames: int = _MAX_MISS_FRAMES,
) -> ObjectFusionResult:
    """Run probabilistic object-state fusion over RF-DETR tracking results.

    Args:
        tracking_results: Per-frame list of detection dicts (RF-DETR output).
        prior:            SemanticPrior for speed clamping and noise adaptation.
        obs_noise:        Observation noise for bbox KF (normalised coords).
        confirm_hits:     Frames before a track becomes confirmed.
        max_miss_frames:  Consecutive misses before track deletion.

    Returns:
        ObjectFusionResult with per-frame smoothed object states.
    """
    if not tracking_results:
        return ObjectFusionResult(enabled=True, status="skipped", reason="no tracking results")

    speed_priors = prior.object_speed_priors if prior else {}
    active_tracks: dict[int, ObjectKalmanFilter] = {}
    next_id = 1
    per_frame_raw: list[list[dict[str, Any]]] = []  # for RTS

    # -- Forward pass ---------------------------------------------------------
    for frame_result in tracking_results:
        t_sec = float(frame_result.get("t_sec", 0.0))
        dets = [d for d in frame_result.get("detections", []) if len(d.get("bbox_norm", [])) == 4]

        # Predict all active tracks to current time
        for trk in active_tracks.values():
            trk.predict(t_sec)

        confirmed_tracks = [
            trk for trk in active_tracks.values() if trk.state in ("tentative", "confirmed")
        ]

        # Build cost matrix and solve assignment
        if confirmed_tracks and dets:
            cost = _build_cost_matrix(confirmed_tracks, dets)
            row_idx, col_idx = linear_sum_assignment(cost)
            matched_trks: set = set()
            matched_dets: set = set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < _INF_COST:
                    trk = confirmed_tracks[r]
                    trk.update(dets[c]["bbox_norm"], t_sec)
                    if trk.hits >= confirm_hits:
                        trk.state = "confirmed"
                    if speed_priors:
                        dt = max(
                            0.02,
                            t_sec - (trk.history[-2].t_sec if len(trk.history) >= 2 else t_sec),
                        )
                        _apply_speed_prior(trk, speed_priors, dt)
                    matched_trks.add(r)
                    matched_dets.add(c)
        else:
            matched_trks = set()
            matched_dets = set()

        # Mark unmatched tracks
        for i, trk in enumerate(confirmed_tracks):
            if i not in matched_trks:
                trk.mark_missed()
                if trk.misses >= max_miss_frames:
                    trk.state = "deleted"

        # Spawn new tracks for unmatched detections
        for j, det in enumerate(dets):
            if j not in matched_dets:
                new_trk = ObjectKalmanFilter(
                    track_id=next_id,
                    label=det.get("label", "unknown"),
                    initial_bbox=det["bbox_norm"],
                    t_sec=t_sec,
                    obs_noise=obs_noise,
                )
                active_tracks[next_id] = new_trk
                next_id += 1

        # Snapshot for this frame
        frame_snapshot: list[dict[str, Any]] = []
        for trk in active_tracks.values():
            if trk.state in ("tentative", "confirmed"):
                frame_snapshot.append(
                    {
                        "track_id": trk.track_id,
                        "label": trk.label,
                        "t_sec": t_sec,
                        "x": trk.x.copy(),
                        "P": trk.P.copy(),
                        "hits": trk.hits,
                        "state": trk.state,
                    }
                )
        per_frame_raw.append(frame_snapshot)

    # -- RTS backward smoother per confirmed track --------------------------
    # Collect complete forward history for each confirmed track
    confirmed_filter_history: dict[int, list[ObjectFilterHistory]] = {}
    for trk in active_tracks.values():
        if trk.state == "confirmed" and len(trk.history) >= 2:
            confirmed_filter_history[trk.track_id] = trk.history

    # Run RTS over each confirmed track
    smoothed_by_track: dict[int, dict[float, np.ndarray]] = {}
    for tid, hist in confirmed_filter_history.items():
        steps = [FilteredStep(t_sec=h.t_sec, x=h.x, P=h.P) for h in hist]
        rts_result = rts_smooth(steps, _OBJ_PROC_POS_STD, _OBJ_PROC_VEL_STD)
        smoothed_by_track[tid] = {s.t_sec: s.x for s in rts_result}

    # -- Assemble output ----------------------------------------------------
    per_frame_out: list[list[ObjectStateSample]] = []
    for frame_idx, frame_snapshot in enumerate(per_frame_raw):
        frame_out: list[ObjectStateSample] = []
        for snap in frame_snapshot:
            tid = snap["track_id"]
            t = snap["t_sec"]

            # Use smoothed state if available, else filtered
            if tid in smoothed_by_track and t in smoothed_by_track[tid]:
                x_out = smoothed_by_track[tid][t]
                track_state = "smoothed"
            else:
                x_out = snap["x"]
                track_state = snap["state"]

            P_diag = np.sqrt(np.diag(snap["P"]))
            frame_out.append(
                ObjectStateSample(
                    track_id=tid,
                    label=snap["label"],
                    t_sec=t,
                    bbox_norm=_state_to_bbox(x_out),
                    velocity_norm=[float(x_out[4]), float(x_out[5])],
                    bbox_std=[
                        float(P_diag[0]),
                        float(P_diag[1]),
                        float(P_diag[2]),
                        float(P_diag[3]),
                    ],
                    confidence=float(snap["hits"] / max(snap["hits"] + snap.get("misses", 0), 1)),
                    track_state=track_state,
                )
            )
        per_frame_out.append(frame_out)

    n_confirmed = sum(1 for trk in active_tracks.values() if trk.state == "confirmed")
    logger.info(
        "Object fusion: %d tracks total, %d confirmed, %d RTS-smoothed",
        len(active_tracks),
        n_confirmed,
        len(smoothed_by_track),
    )
    return ObjectFusionResult(
        enabled=True,
        status="ok",
        source="object_kalman_mahalanobis_v1",
        track_count=len(active_tracks),
        confirmed_tracks=n_confirmed,
        per_frame=per_frame_out,
        diagnostics={
            "total_tracks": len(active_tracks),
            "confirmed_tracks": n_confirmed,
            "rts_smoothed_tracks": len(smoothed_by_track),
            "frames_processed": len(tracking_results),
        },
    )


# -- Clip-level summary helper -------------------------------------------------


def summarize_object_frame_dicts(
    per_frame_dicts: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Collapse serialised per-frame object states into a clip-level belief dict.

    Works on the output of ``ObjectFusionResult.to_dict()["per_frame"]`` so it
    can be called from pipeline steps that have the JSON representation rather
    than the in-memory dataclass.

    Returns:
        confirmed_track_count : int   — smoothed/confirmed tracks seen in this clip
        mean_velocity_norm    : float — mean speed (normalised coords/frame) across
                                        all confirmed samples
        max_velocity_norm     : float — peak speed observed
        velocity_by_label     : dict  — {label: mean_speed} per object class
        mean_bbox_uncertainty : float — mean of bbox_std entries across all samples
        near_field_density    : float — mean per-frame fraction of the central
                                        image region ([0.3,0.7]² in normalised
                                        coords) occupied by tracked bboxes
    """
    _NEAR_LO, _NEAR_HI = 0.3, 0.7  # central region boundaries
    _ACTIVE = {"confirmed", "smoothed"}

    vel_by_label: dict[str, list[float]] = {}
    all_uncertainties: list[float] = []
    frame_near_densities: list[float] = []
    confirmed_track_ids: set[int] = set()

    for frame_samples in per_frame_dicts:
        frame_area = 0.0
        for s in frame_samples:
            if s.get("track_state") not in _ACTIVE:
                continue
            tid = s.get("track_id")
            if isinstance(tid, int):
                confirmed_track_ids.add(tid)
            vx, vy = (s.get("velocity_norm") or [0.0, 0.0])[:2]
            speed = (vx**2 + vy**2) ** 0.5
            label = (s.get("label") or "unknown").lower()
            vel_by_label.setdefault(label, []).append(speed)

            std = s.get("bbox_std") or []
            if std:
                all_uncertainties.append(sum(std) / len(std))

            bbox = s.get("bbox_norm") or []
            if len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) * 0.5
                cy = (bbox[1] + bbox[3]) * 0.5
                if _NEAR_LO <= cx <= _NEAR_HI and _NEAR_LO <= cy <= _NEAR_HI:
                    w = max(0.0, bbox[2] - bbox[0])
                    h = max(0.0, bbox[3] - bbox[1])
                    frame_area += w * h

        frame_near_densities.append(min(1.0, frame_area))

    all_speeds = [s for speeds in vel_by_label.values() for s in speeds]

    return {
        "confirmed_track_count": len(confirmed_track_ids),
        "mean_velocity_norm": float(sum(all_speeds) / len(all_speeds)) if all_speeds else 0.0,
        "max_velocity_norm": float(max(all_speeds)) if all_speeds else 0.0,
        "velocity_by_label": {lbl: float(sum(sp) / len(sp)) for lbl, sp in vel_by_label.items()},
        "mean_bbox_uncertainty": float(sum(all_uncertainties) / len(all_uncertainties))
        if all_uncertainties
        else 0.0,
        "near_field_density": float(sum(frame_near_densities) / len(frame_near_densities))
        if frame_near_densities
        else 0.0,
    }
