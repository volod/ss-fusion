"""Shared contradiction and trust-calibration helpers for threat outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_CONFLICT_SEVERITY: dict[str, float] = {
    "occupancy_vs_unidrive_clear": 0.30,
    "collision_vs_unidrive_low_risk": 0.24,
    "collision_vs_unidrive_continue": 0.24,
    "visibility_vs_unidrive_low_risk": 0.22,
    "visibility_vs_unidrive_clear": 0.22,
    "visibility_vs_unidrive_continue": 0.20,
    "pose_vs_unidrive_low_risk": 0.18,
    "pose_vs_unidrive_clear": 0.18,
    "pose_vs_unidrive_continue": 0.16,
    "track_vs_unidrive_clear": 0.22,
    "track_vs_unidrive_low_risk": 0.18,
    "tracking_persistence_vs_iou_break": 0.28,
    "caption_confidence_vs_depth_failure": 0.26,
    "unidrive_moe_disagreement": 0.14,
}

_EVIDENCE_SOURCE_SENSOR_MAP: dict[str, list[str]] = {
    "near_field_occupancy": ["object-state fusion", "occupancy aggregation"],
    "object_velocity": ["RF-DETR tracking", "object Kalman fusion"],
    "free_space_estimate": ["depth estimation", "physical scene aggregation"],
    "depth_failure_rate": ["depth estimation"],
    "ssim_keyframe_quality": ["frame SSIM heuristics"],
    "caption_confidence": ["Florence-2 captioning"],
    "visibility_field_intensity": ["field-state aggregation", "depth estimation"],
    "visibility_field_trend": ["field-state temporal trend"],
    "rf_spectral_flatness": ["RF signal analysis"],
    "rf_occupied_bandwidth": ["RF signal analysis"],
    "rf_low_snr": ["RF signal analysis"],
    "rf_field_trend": ["field-state temporal trend"],
    "track_breaks": ["RF-DETR tracking"],
    "iou_drop_events": ["RF-DETR tracking"],
    "short_track_length": ["object-state fusion"],
    "kalman_pose_confidence": ["platform/map Kalman fusion"],
    "sfm_quality_degraded": ["pycolmap SfM"],
    "sfm_failure_rate": ["pycolmap SfM"],
}


def sensor_sources_from_evidence(evidence_sources: list[str]) -> list[str]:
    sensors: list[str] = []
    for source in evidence_sources:
        for label in _EVIDENCE_SOURCE_SENSOR_MAP.get(str(source), [str(source)]):
            if label not in sensors:
                sensors.append(label)
    return sensors


def support_frame_names(paths: list[str], limit: int = 4) -> list[str]:
    names: list[str] = []
    for path in paths:
        if not path:
            continue
        name = Path(path).name
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def matching_unidrive_rows(
    support_paths: list[str],
    unidrive_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not support_paths or not unidrive_rows:
        return []
    support_set = {str(p) for p in support_paths if p}
    support_names = {Path(p).name for p in support_paths if p}
    return [
        row
        for row in unidrive_rows
        if str(row.get("frame_path", "")) in support_set
        or Path(str(row.get("frame_path", ""))).name in support_names
    ]


def contradiction_signals_for_threat(
    threat_type: str,
    primitive: dict[str, Any],
    unidrive_rows: list[dict[str, Any]],
    physical_state: dict[str, Any],
) -> list[dict[str, Any]]:
    support_paths = list(primitive.get("spatial_support") or [])
    rows = matching_unidrive_rows(support_paths, unidrive_rows) or list(unidrive_rows[:6])
    disagreements: list[dict[str, Any]] = []
    occupancy_dense = float(physical_state.get("near_field_occupancy_density", 0.0) or 0.0) > 0.15

    for row in rows:
        understanding = row.get("understanding", {}) or {}
        perception = row.get("perception", {}) or {}
        planning = row.get("planning", {}) or {}
        moe = row.get("mixture_of_experts", {}) or {}

        risk_level = str(understanding.get("risk_level", "unknown") or "unknown").lower()
        drivable = str(perception.get("drivable_area", "unknown") or "unknown").lower()
        action = str(planning.get("recommended_action", "") or "").lower()
        frame_id = Path(str(row.get("frame_path", ""))).name or None

        if threat_type == "collision_risk":
            if drivable == "clear" and occupancy_dense:
                _add_signal(
                    disagreements,
                    pattern="occupancy_vs_unidrive_clear",
                    description="near-field occupancy is dense while UniDrive marks drivable area clear",
                    source_a="occupancy aggregation",
                    source_b="UniDriveVLA perception",
                    frame_id=frame_id,
                )
            if risk_level == "low":
                _add_signal(
                    disagreements,
                    pattern="collision_vs_unidrive_low_risk",
                    description="collision primitive is active while UniDrive reports low risk",
                    source_a="local collision primitive",
                    source_b="UniDriveVLA understanding",
                    frame_id=frame_id,
                )
            if any(token in action for token in ("continue", "maintain", "keep", "proceed")):
                _add_signal(
                    disagreements,
                    pattern="collision_vs_unidrive_continue",
                    description="collision primitive is active while UniDrive recommends continuing",
                    source_a="local collision primitive",
                    source_b="UniDriveVLA planning",
                    frame_id=frame_id,
                )
        elif threat_type == "visibility_degradation":
            if risk_level == "low":
                _add_signal(
                    disagreements,
                    pattern="visibility_vs_unidrive_low_risk",
                    description="visibility degradation is active while UniDrive reports low risk",
                    source_a="local visibility primitive",
                    source_b="UniDriveVLA understanding",
                    frame_id=frame_id,
                )
            if drivable == "clear":
                _add_signal(
                    disagreements,
                    pattern="visibility_vs_unidrive_clear",
                    description="visibility degradation is active while UniDrive sees clear drivable area",
                    source_a="local visibility primitive",
                    source_b="UniDriveVLA perception",
                    frame_id=frame_id,
                )
            if any(token in action for token in ("continue", "maintain", "keep", "proceed")):
                _add_signal(
                    disagreements,
                    pattern="visibility_vs_unidrive_continue",
                    description="visibility degradation is active while UniDrive recommends continuing",
                    source_a="local visibility primitive",
                    source_b="UniDriveVLA planning",
                    frame_id=frame_id,
                )
        elif threat_type == "pose_uncertain":
            if risk_level == "low":
                _add_signal(
                    disagreements,
                    pattern="pose_vs_unidrive_low_risk",
                    description="pose uncertainty is active while UniDrive reports low risk",
                    source_a="local pose primitive",
                    source_b="UniDriveVLA understanding",
                    frame_id=frame_id,
                )
            if drivable == "clear":
                _add_signal(
                    disagreements,
                    pattern="pose_vs_unidrive_clear",
                    description="pose uncertainty is active while UniDrive sees clear drivable area",
                    source_a="local pose primitive",
                    source_b="UniDriveVLA perception",
                    frame_id=frame_id,
                )
            if any(token in action for token in ("continue", "maintain", "keep", "proceed")):
                _add_signal(
                    disagreements,
                    pattern="pose_vs_unidrive_continue",
                    description="pose uncertainty is active while UniDrive recommends continuing",
                    source_a="local pose primitive",
                    source_b="UniDriveVLA planning",
                    frame_id=frame_id,
                )
        elif threat_type == "track_anomaly":
            if str(moe.get("expert_agreement", "unknown") or "unknown").lower() == "high":
                _add_signal(
                    disagreements,
                    pattern="track_vs_unidrive_clear",
                    description="track anomaly is active while UniDrive experts strongly agree on a clear interpretation",
                    source_a="RF-DETR tracking",
                    source_b="UniDriveVLA mixture-of-experts",
                    frame_id=frame_id,
                )
            if risk_level == "low":
                _add_signal(
                    disagreements,
                    pattern="track_vs_unidrive_low_risk",
                    description="track anomaly is active while UniDrive reports low risk",
                    source_a="RF-DETR tracking",
                    source_b="UniDriveVLA understanding",
                    frame_id=frame_id,
                )

        for point in list(moe.get("disagreement_points") or [])[:2]:
            _add_signal(
                disagreements,
                pattern="unidrive_moe_disagreement",
                description=str(point)[:120],
                source_a="UniDriveVLA expert mixture",
                source_b="UniDriveVLA consensus layer",
                frame_id=frame_id,
            )

    return disagreements


def summarize_contradictions(
    threat_rows: list[dict[str, Any]] | None = None,
    contradiction_signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = list(threat_rows or [])
    signals = list(contradiction_signals or [])
    for row in rows:
        for signal in row.get("contradiction_signals") or []:
            signals.append(dict(signal))

    comparison_budget = sum(max(1, len(row.get("sensor_sources") or [])) for row in rows)
    comparison_budget = max(comparison_budget, len(signals), 1)
    disagreement_count = len(signals)
    disagreement_rate = min(1.0, disagreement_count / comparison_budget)

    grouped: dict[str, dict[str, Any]] = {}
    for signal in signals:
        pattern = str(signal.get("pattern", "unknown") or "unknown")
        severity = float(signal.get("severity", _CONFLICT_SEVERITY.get(pattern, 0.15)) or 0.15)
        entry = grouped.setdefault(
            pattern,
            {
                "pattern": pattern,
                "count": 0,
                "severity": 0.0,
                "source_pair": [
                    signal.get("source_a", "unknown"),
                    signal.get("source_b", "unknown"),
                ],
                "frames": [],
                "description": str(signal.get("description", "") or ""),
            },
        )
        entry["count"] += 1
        entry["severity"] = max(float(entry["severity"]), severity)
        frame_id = str(signal.get("frame_id", "") or "")
        if frame_id and frame_id not in entry["frames"]:
            entry["frames"].append(frame_id)
    source_pair_conflicts = sorted(
        grouped.values(),
        key=lambda row: (-float(row["severity"]), -int(row["count"]), row["pattern"]),
    )

    if source_pair_conflicts:
        raw_penalty = sum(
            float(item["severity"]) * min(1.0, 0.5 + 0.25 * int(item["count"]))
            for item in source_pair_conflicts
        )
    else:
        raw_penalty = 0.0
    trust_penalty = min(0.65, raw_penalty * max(0.35, disagreement_rate))

    return {
        "disagreement_count": disagreement_count,
        "disagreement_rate": round(disagreement_rate, 4),
        "source_pair_conflicts": source_pair_conflicts,
        "trust_penalty": round(trust_penalty, 4),
    }


def _add_signal(
    items: list[dict[str, Any]],
    *,
    pattern: str,
    description: str,
    source_a: str,
    source_b: str,
    frame_id: str | None = None,
) -> None:
    key = (pattern, frame_id or "", source_a, source_b)
    for item in items:
        if (
            str(item.get("pattern", "")) == key[0]
            and str(item.get("frame_id", "")) == key[1]
            and str(item.get("source_a", "")) == key[2]
            and str(item.get("source_b", "")) == key[3]
        ):
            return
    items.append(
        {
            "pattern": pattern,
            "description": description,
            "source_a": source_a,
            "source_b": source_b,
            "frame_id": frame_id,
            "severity": round(_CONFLICT_SEVERITY.get(pattern, 0.15), 4),
        }
    )
