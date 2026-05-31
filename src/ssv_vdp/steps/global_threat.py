"""Batch global threat aggregation over completed local-run artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.fusion import (
    GlobalThreatResult,
    PersistentAnomaly,
    RouteAdvisory,
    SectorThreatState,
)
from selfsuvis.pipeline.fusion.sectors import (
    build_route_segment_id,
    build_sector_adjacency,
    sectorize_global_positions,
    unique_sector_sequence,
)

_log = get_logger("pipeline.local")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.25:
        return "low"
    return "none"


def _video_dirs_from_stats(
    output_dir: Path, per_video_stats: Sequence[dict[str, Any]]
) -> list[Path]:
    out: list[Path] = []
    for stats in per_video_stats:
        vdir = stats.get("video_dir")
        if vdir:
            out.append(Path(str(vdir)))
            continue
        name = str(stats.get("name", "") or "")
        if name:
            out.append(output_dir / name)
    return [p for p in out if p.exists()]


def _extract_sector_sequence(
    full_fusion_payload: dict[str, Any], video_name: str
) -> tuple[list[str], list[dict[str, Any]]]:
    platform = full_fusion_payload.get("platform") or {}
    origin = platform.get("origin_lla") or {}
    map_state = full_fusion_payload.get("map_state") or {}
    smoothed = map_state.get("smoothed_samples") or []
    positions = [
        dict(row.get("position_enu_m") or {}) for row in smoothed if row.get("position_enu_m")
    ]
    if not origin or not positions:
        return [], []
    sector_samples = sectorize_global_positions(origin, positions, tile_size_m=50.0)
    return unique_sector_sequence(sector_samples), sector_samples


def _collect_video_record(video_dir: Path) -> dict[str, Any] | None:
    local_threat = _load_json(video_dir / "local_threat_assessment.json")
    policy = _load_json(video_dir / "policy_decision.json")
    primitives = _load_json(video_dir / "threat_primitives.json")
    physical = _load_json(video_dir / "physical_state_summary.json")
    full_fusion = _load_json(video_dir / "full_state_fusion.json")
    if not local_threat:
        return None
    video_name = video_dir.name
    sector_sequence, sector_samples = _extract_sector_sequence(full_fusion, video_name)
    route_id = build_route_segment_id(video_name, sector_sequence)
    return {
        "video_name": video_name,
        "video_dir": str(video_dir),
        "local_threat": local_threat,
        "primitives": list(primitives.get("primitives") or []),
        "physical_state": physical,
        "full_fusion": full_fusion,
        "sector_sequence": sector_sequence,
        "sector_samples": sector_samples,
        "route_id": route_id,
        "recommended_action": str(
            policy.get("recommended_action", local_threat.get("recommended_action", "continue"))
        ),
        "time_range_sec": _time_range_from_fusion(full_fusion),
    }


def _time_range_from_fusion(full_fusion_payload: dict[str, Any]) -> list[float]:
    smoothed = (full_fusion_payload.get("map_state") or {}).get("smoothed_samples") or []
    if not smoothed:
        return [0.0, 0.0]
    first = float(smoothed[0].get("t_sec", 0.0) or 0.0)
    last = float(smoothed[-1].get("t_sec", 0.0) or 0.0)
    return [first, last]


def _aggregate_sector_states(records: Sequence[dict[str, Any]]) -> list[SectorThreatState]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for sector_id in record.get("sector_sequence") or []:
            grouped[str(sector_id)].append(record)

    out: list[SectorThreatState] = []
    for sector_id, sector_records in sorted(grouped.items()):
        scores = [
            float((r.get("local_threat") or {}).get("local_threat_score", 0.0) or 0.0)
            for r in sector_records
        ]
        uncertainties: list[float] = []
        primitive_types: list[str] = []
        evidence_sources: list[str] = []
        route_ids: list[str] = []
        supporting_videos: list[str] = []
        for record in sector_records:
            if record["video_name"] not in supporting_videos:
                supporting_videos.append(record["video_name"])
            if record["route_id"] not in route_ids:
                route_ids.append(record["route_id"])
            for primitive in record.get("primitives") or []:
                ptype = str(primitive.get("type", ""))
                if ptype and ptype not in primitive_types:
                    primitive_types.append(ptype)
                uncertainties.append(float(primitive.get("uncertainty", 0.0) or 0.0))
                for source in primitive.get("evidence_sources") or []:
                    if source not in evidence_sources:
                        evidence_sources.append(source)
        independent_support = len(supporting_videos)
        remaining = 1.0
        for score in scores:
            remaining *= 1.0 - max(0.0, min(1.0, score))
        base_score = 1.0 - remaining
        support_bonus = min(0.15, 0.05 * max(0, independent_support - 1))
        threat_score = min(1.0, base_score + support_bonus)
        out.append(
            SectorThreatState(
                sector_id=sector_id,
                threat_score=round(threat_score, 4),
                risk_level=_risk_level(threat_score),
                supporting_videos=supporting_videos,
                route_ids=route_ids,
                primitive_types=primitive_types,
                observation_count=len(scores),
                mean_uncertainty=round(sum(uncertainties) / len(uncertainties), 4)
                if uncertainties
                else 0.0,
                evidence_sources=evidence_sources,
                metadata={"independent_support_count": independent_support},
            )
        )
    return out


def _aggregate_route_advisories(records: Sequence[dict[str, Any]]) -> list[RouteAdvisory]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["route_id"])].append(record)
    out: list[RouteAdvisory] = []
    for route_id, route_records in sorted(grouped.items()):
        scores = [
            float((r.get("local_threat") or {}).get("local_threat_score", 0.0) or 0.0)
            for r in route_records
        ]
        sector_ids: list[str] = []
        evidence_sources: list[str] = []
        actions: list[str] = []
        videos: list[str] = []
        for record in route_records:
            videos.append(record["video_name"])
            actions.append(record["recommended_action"])
            for sector_id in record.get("sector_sequence") or []:
                if sector_id not in sector_ids:
                    sector_ids.append(sector_id)
            for primitive in record.get("primitives") or []:
                for source in primitive.get("evidence_sources") or []:
                    if source not in evidence_sources:
                        evidence_sources.append(source)
        score = max(scores) if scores else 0.0
        if "abort" in actions:
            action = "abort"
        elif "reroute" in actions:
            action = "reroute"
        elif "inspect_sensor" in actions:
            action = "inspect_sensor"
        elif "reduce_speed" in actions:
            action = "reduce_speed"
        else:
            action = "continue"
        out.append(
            RouteAdvisory(
                route_id=route_id,
                advisory_score=round(score, 4),
                recommended_action=action,
                sector_ids=sector_ids,
                supporting_videos=videos,
                evidence_sources=evidence_sources,
                metadata={"n_videos": len(route_records)},
            )
        )
    return out


def _aggregate_persistent_anomalies(records: Sequence[dict[str, Any]]) -> list[PersistentAnomaly]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        sectors = record.get("sector_sequence") or []
        if not sectors:
            continue
        for primitive in record.get("primitives") or []:
            ptype = str(primitive.get("type", ""))
            if not ptype:
                continue
            for sector_id in sectors:
                grouped[(ptype, str(sector_id))].append((record, primitive))

    out: list[PersistentAnomaly] = []
    for (ptype, sector_id), rows in sorted(grouped.items()):
        videos = []
        evidence_sources: list[str] = []
        scores: list[float] = []
        for record, primitive in rows:
            if record["video_name"] not in videos:
                videos.append(record["video_name"])
            scores.append(float(primitive.get("score", 0.0) or 0.0))
            for source in primitive.get("evidence_sources") or []:
                if source not in evidence_sources:
                    evidence_sources.append(source)
        if len(videos) < 2 and len(rows) < 2:
            continue
        out.append(
            PersistentAnomaly(
                anomaly_id=f"{ptype}:{sector_id}",
                anomaly_type=ptype,
                sector_ids=[sector_id],
                threat_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
                persistence_count=len(rows),
                supporting_videos=videos,
                evidence_sources=evidence_sources,
                metadata={"independent_videos": len(videos)},
            )
        )
    return out


def _aggregate_corridor_graph(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        score = float((record.get("local_threat") or {}).get("local_threat_score", 0.0) or 0.0)
        for edge in build_sector_adjacency(record.get("sector_sequence") or []):
            key = (edge["from_sector"], edge["to_sector"])
            current = edge_map.setdefault(
                key,
                {
                    "from_sector": edge["from_sector"],
                    "to_sector": edge["to_sector"],
                    "weight": 0.0,
                    "supporting_videos": [],
                    "max_threat_score": 0.0,
                },
            )
            current["weight"] += 1.0
            current["max_threat_score"] = max(float(current["max_threat_score"]), score)
            if record["video_name"] not in current["supporting_videos"]:
                current["supporting_videos"].append(record["video_name"])
    return sorted(
        edge_map.values(),
        key=lambda row: (-float(row["weight"]), row["from_sector"], row["to_sector"]),
    )


def step_global_threat(
    output_dir: Path,
    per_video_stats: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-video local threat artifacts into one mission-level summary."""
    video_dirs = _video_dirs_from_stats(output_dir, per_video_stats)
    records = [record for record in (_collect_video_record(vdir) for vdir in video_dirs) if record]
    if not records:
        result = GlobalThreatResult(
            enabled=True,
            status="skipped",
            reason="no local threat artifacts available",
            diagnostics={"video_dirs_checked": [str(v) for v in video_dirs]},
        )
        payload = result.to_dict()
        payload["skipped"] = True
        _write_json(output_dir / "global_threat_summary.json", payload)
        _log.info("Global threat aggregation skipped: no local threat artifacts found")
        return payload

    sector_states = _aggregate_sector_states(records)
    route_advisories = _aggregate_route_advisories(records)
    anomalies = _aggregate_persistent_anomalies(records)
    corridor_graph = _aggregate_corridor_graph(records)

    global_map = [
        {
            "sector_id": state.sector_id,
            "threat_score": state.threat_score,
            "risk_level": state.risk_level,
            "supporting_videos": list(state.supporting_videos),
            "route_ids": list(state.route_ids),
        }
        for state in sector_states
    ]

    result = GlobalThreatResult(
        enabled=True,
        status="ok",
        global_threat_map=global_map,
        sector_risk_levels=sector_states,
        persistent_anomalies=anomalies,
        route_advisories=route_advisories,
        threat_corridor_graph=corridor_graph,
        diagnostics={
            "n_videos": len(records),
            "video_names": [record["video_name"] for record in records],
            "n_unique_sectors": len({state.sector_id for state in sector_states}),
        },
    )
    payload = result.to_dict()
    payload["skipped"] = False
    _write_json(output_dir / "global_threat_summary.json", payload)
    _log.info(
        "[ok] Global threat aggregation: sectors=%d routes=%d anomalies=%d",
        len(sector_states),
        len(route_advisories),
        len(anomalies),
    )
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
