"""File-based mission threat memory and contradiction-history summaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sectors import build_route_segment_id, sectorize_global_positions, unique_sector_sequence


@dataclass
class ThreatMemoryRecord:
    mission_id: str
    video_id: str
    sector_ids: list[str] = field(default_factory=list)
    route_id: str = ""
    local_threat_score: float = 0.0
    automation_confidence: float = 1.0
    trust_penalty: float = 0.0
    top_threats: list[dict[str, Any]] = field(default_factory=list)
    contradiction_signals: list[dict[str, Any]] = field(default_factory=list)
    source_pair_conflicts: list[dict[str, Any]] = field(default_factory=list)
    platform_metadata: dict[str, Any] = field(default_factory=dict)
    sensor_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "video_id": self.video_id,
            "sector_ids": list(self.sector_ids),
            "route_id": self.route_id,
            "local_threat_score": float(self.local_threat_score),
            "automation_confidence": float(self.automation_confidence),
            "trust_penalty": float(self.trust_penalty),
            "top_threats": list(self.top_threats),
            "contradiction_signals": list(self.contradiction_signals),
            "source_pair_conflicts": list(self.source_pair_conflicts),
            "platform_metadata": dict(self.platform_metadata),
            "sensor_metadata": dict(self.sensor_metadata),
        }


def persist_threat_memory(
    output_dir: Path,
    per_video_stats: Sequence[dict[str, Any]],
    global_threat_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mission_id = output_dir.name or "mission"
    records = [
        record
        for record in (
            _collect_memory_record(Path(str(stats.get("video_dir", ""))), mission_id)
            for stats in per_video_stats
            if stats.get("video_dir")
        )
        if record is not None
    ]

    memory_dir = output_dir / "threat_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mission_id": mission_id,
        "record_count": len(records),
        "records": [record.to_dict() for record in records],
        "history_summary": summarize_threat_memory(records, global_threat_result or {}),
    }
    out_path = memory_dir / f"mission_{mission_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_threat_memory(
    records: Sequence[ThreatMemoryRecord],
    global_threat_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global_threat_result = global_threat_result or {}
    pattern_counts: dict[str, int] = {}
    route_penalties: dict[str, list[float]] = {}
    subsystem_conflicts: dict[str, int] = {
        "unidrive": 0,
        "tracking": 0,
        "depth": 0,
    }

    for record in records:
        route_penalties.setdefault(record.route_id or "unknown", []).append(
            float(record.trust_penalty)
        )
        for conflict in record.source_pair_conflicts:
            pattern = str(conflict.get("pattern", "unknown") or "unknown")
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + int(
                conflict.get("count", 0) or 0
            )
            if "unidrive" in pattern:
                subsystem_conflicts["unidrive"] += int(conflict.get("count", 0) or 0)
            if "track" in pattern or "iou" in pattern:
                subsystem_conflicts["tracking"] += int(conflict.get("count", 0) or 0)
            if "depth" in pattern or "caption" in pattern or "visibility" in pattern:
                subsystem_conflicts["depth"] += int(conflict.get("count", 0) or 0)

    repeated_patterns = [
        {"pattern": pattern, "count": count}
        for pattern, count in sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ]
    sensor_health_flags = []
    if pattern_counts.get("occupancy_vs_unidrive_clear", 0) >= 2:
        sensor_health_flags.append("unidrive_perception_consistency_check")
    if pattern_counts.get("tracking_persistence_vs_iou_break", 0) >= 2:
        sensor_health_flags.append("tracking_geometry_health_check")
    if pattern_counts.get("caption_confidence_vs_depth_failure", 0) >= 2:
        sensor_health_flags.append("depth_caption_crosscheck_required")

    subsystem_reliability_trends = []
    for subsystem, count in subsystem_conflicts.items():
        if count <= 0:
            continue
        trend = "degrading" if count >= 3 else "watch"
        subsystem_reliability_trends.append(
            {"subsystem": subsystem, "trend": trend, "conflict_count": count}
        )

    route_level_trust_warnings = []
    for route_id, penalties in sorted(route_penalties.items()):
        mean_penalty = sum(penalties) / max(1, len(penalties))
        if mean_penalty < 0.20:
            continue
        route_level_trust_warnings.append(
            {
                "route_id": route_id,
                "mean_trust_penalty": round(mean_penalty, 4),
                "warning_level": "high" if mean_penalty >= 0.35 else "medium",
                "n_videos": len(penalties),
            }
        )

    return {
        "repeated_conflict_patterns": repeated_patterns,
        "sensor_health_flags": sensor_health_flags,
        "subsystem_reliability_trends": subsystem_reliability_trends,
        "route_level_trust_warnings": route_level_trust_warnings,
        "global_routes": list(global_threat_result.get("route_advisories") or []),
    }


def _collect_memory_record(video_dir: Path, mission_id: str) -> ThreatMemoryRecord | None:
    if not video_dir.exists():
        return None
    local_threat = _load_json(video_dir / "local_threat_assessment.json")
    if not local_threat:
        return None
    full_fusion = _load_json(video_dir / "full_state_fusion.json")
    physical_state = _load_json(video_dir / "physical_state_summary.json")
    primitives = _load_json(video_dir / "threat_primitives.json")
    sector_ids, route_id = _extract_sector_index(full_fusion, video_dir.name)
    return ThreatMemoryRecord(
        mission_id=mission_id,
        video_id=video_dir.name,
        sector_ids=sector_ids,
        route_id=route_id,
        local_threat_score=float(local_threat.get("local_threat_score", 0.0) or 0.0),
        automation_confidence=float(local_threat.get("automation_confidence", 1.0) or 1.0),
        trust_penalty=float(local_threat.get("trust_penalty", 0.0) or 0.0),
        top_threats=list(local_threat.get("top_threats") or []),
        contradiction_signals=list(primitives.get("contradiction_signals") or []),
        source_pair_conflicts=list(local_threat.get("source_pair_conflicts") or []),
        platform_metadata={
            "time_range_sec": _time_range_from_fusion(full_fusion),
            "origin_lla": dict((full_fusion.get("platform") or {}).get("origin_lla") or {}),
            "platform_pose_confidence": float(
                physical_state.get("platform_pose_confidence", 0.0) or 0.0
            ),
        },
        sensor_metadata=_sensor_metadata(local_threat, primitives, physical_state),
    )


def _extract_sector_index(full_fusion: dict[str, Any], video_name: str) -> tuple[list[str], str]:
    platform = full_fusion.get("platform") or {}
    origin = platform.get("origin_lla") or {}
    smoothed = (full_fusion.get("map_state") or {}).get("smoothed_samples") or []
    positions = [
        dict(row.get("position_enu_m") or {}) for row in smoothed if row.get("position_enu_m")
    ]
    if not origin or not positions:
        return [], build_route_segment_id(video_name, [])
    sector_samples = sectorize_global_positions(origin, positions, tile_size_m=50.0)
    sector_ids = unique_sector_sequence(sector_samples)
    return sector_ids, build_route_segment_id(video_name, sector_ids)


def _sensor_metadata(
    local_threat: dict[str, Any],
    primitives: dict[str, Any],
    physical_state: dict[str, Any],
) -> dict[str, Any]:
    source_labels = set()
    for threat in local_threat.get("top_threats") or []:
        for source in (threat.get("evidence") or {}).get("evidence_sources") or []:
            source_labels.add(str(source))
    for primitive in primitives.get("primitives") or []:
        for source in primitive.get("evidence_sources") or []:
            source_labels.add(str(source))
    return {
        "tracking_used": bool(physical_state.get("tracking_used", False)),
        "confirmed_tracks": int(physical_state.get("confirmed_tracks", 0) or 0),
        "available_signal_sources": sorted(source_labels),
    }


def _time_range_from_fusion(full_fusion: dict[str, Any]) -> list[float]:
    smoothed = (full_fusion.get("map_state") or {}).get("smoothed_samples") or []
    if not smoothed:
        return [0.0, 0.0]
    return [
        float(smoothed[0].get("t_sec", 0.0) or 0.0),
        float(smoothed[-1].get("t_sec", 0.0) or 0.0),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
