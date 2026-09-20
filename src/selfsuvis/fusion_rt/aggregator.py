"""One-process realtime aggregator boundary for sector/global threat updates."""

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.freshness import downweight_score, expire_event
from selfsuvis.pipeline.fusion.utils import probability_union

from .degraded_mode import apply_degraded_mode_to_threat, evaluate_degraded_mode
from .event_access import (
    event_freshness_sec,
    event_kind,
    event_node_id,
    event_payload,
    event_sector_id,
    event_sensor_type,
    payload_float,
    payload_text,
)


class RealtimeThreatAggregator:
    """Consume replayed or live event envelopes and emit operator snapshots."""

    def __init__(self) -> None:
        self._sensor_events: list[dict[str, Any]] = []
        self._threat_events: list[dict[str, Any]] = []
        self._node_health_events: list[dict[str, Any]] = []

    def consume(self, event: dict[str, Any]) -> None:
        kind = event_kind(event)
        if kind == "sensor":
            self._sensor_events.append(dict(event))
        elif kind == "threat":
            self._threat_events.append(dict(event))
        elif kind == "node_health":
            self._node_health_events.append(dict(event))
        else:
            raise ValueError(f"unsupported event_kind: {kind or '<empty>'}")

    def consume_all(self, events: Iterable[dict[str, Any]]) -> None:
        for event in events:
            self.consume(event)

    def snapshot(self) -> dict[str, Any]:
        sector_rows = self._sector_rows()
        health = evaluate_degraded_mode(
            self._sensor_events,
            self._node_health_events,
            base_automation_confidence=self._base_automation_confidence(),
        )
        route_rows = self._route_rows(
            sector_rows, automation_confidence=float(health["automation_confidence"])
        )
        return {
            "global_threat_map": [
                {
                    "sector_id": row["sector_id"],
                    "threat_score": row["threat_score"],
                    "risk_level": row["risk_level"],
                }
                for row in sector_rows
            ],
            "sector_risk_levels": sector_rows,
            "persistent_anomalies": self._persistent_anomalies(),
            "route_advisories": route_rows,
            "threat_corridor_graph": self._corridor_graph(),
            "automation_confidence": health["automation_confidence"],
            "degraded": health["degraded"],
            "health_warnings": health["health_warnings"],
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    def write_snapshot(self, output_path: Path) -> dict[str, Any]:
        snapshot = self.snapshot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return snapshot

    def _sector_rows(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in self._threat_events:
            if expire_event(event, hard_expiry_sec=120.0):
                continue
            grouped[event_sector_id(event)].append(event)

        rows: list[dict[str, Any]] = []
        for sector_id, events in sorted(grouped.items()):
            score_terms: list[float] = []
            nodes: list[str] = []
            route_ids: list[str] = []
            primitive_types: list[str] = []
            for event in events:
                payload = event_payload(event)
                sensor_type = event_sensor_type(event)
                if sensor_type == "local_threat":
                    score = payload_float(payload, "local_threat_score")
                else:
                    score = payload_float(payload, "score")
                score_terms.append(downweight_score(score, event_freshness_sec(event)))
                node_id = event_node_id(event)
                if node_id not in nodes:
                    nodes.append(node_id)
                route_id = payload_text(payload, "route_id")
                if route_id and route_id not in route_ids:
                    route_ids.append(route_id)
                threat_type = payload_text(payload, "threat_type", default=sensor_type)
                if threat_type and threat_type not in primitive_types:
                    primitive_types.append(threat_type)

            aggregated = probability_union([max(0.0, min(1.0, s)) for s in score_terms])
            support_bonus = min(0.15, 0.05 * max(0, len(nodes) - 1))
            threat_score = min(1.0, aggregated + support_bonus)
            rows.append(
                {
                    "sector_id": sector_id,
                    "threat_score": round(threat_score, 4),
                    "risk_level": _risk_level(threat_score),
                    "supporting_nodes": nodes,
                    "route_ids": route_ids,
                    "primitive_types": primitive_types,
                    "observation_count": len(events),
                }
            )
        return rows

    def _route_rows(
        self, sector_rows: Sequence[dict[str, Any]], *, automation_confidence: float
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sector_rows:
            for route_id in row.get("route_ids") or []:
                grouped[str(route_id)].append(row)
        out: list[dict[str, Any]] = []
        for route_id, rows in sorted(grouped.items()):
            max_score = max(float(row.get("threat_score", 0.0) or 0.0) for row in rows)
            health = apply_degraded_mode_to_threat(
                max_score,
                automation_confidence,
            )
            if max_score >= 0.75:
                action = "abort"
            elif health["automation_confidence"] < 0.50:
                action = "inspect_sensor"
            elif max_score >= 0.60:
                action = "reroute"
            elif max_score >= 0.35:
                action = "reduce_speed"
            else:
                action = "continue"
            out.append(
                {
                    "route_id": route_id,
                    "advisory_score": round(max_score, 4),
                    "recommended_action": action,
                    "sector_ids": [event_sector_id(row) for row in rows],
                    "automation_confidence": health["automation_confidence"],
                }
            )
        return out

    def _persistent_anomalies(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event in self._threat_events:
            sensor_type = event_sensor_type(event)
            if sensor_type == "local_threat":
                continue
            payload = event_payload(event)
            threat_type = payload_text(payload, "threat_type", default=sensor_type)
            grouped[(threat_type, event_sector_id(event))].append(event)
        rows: list[dict[str, Any]] = []
        for (threat_type, sector_id), events in sorted(grouped.items()):
            if len(events) < 2:
                continue
            rows.append(
                {
                    "anomaly_id": f"{threat_type}:{sector_id}",
                    "anomaly_type": threat_type,
                    "sector_ids": [sector_id],
                    "persistence_count": len(events),
                    "supporting_nodes": sorted({event_node_id(event) for event in events}),
                }
            )
        return rows

    def _corridor_graph(self) -> list[dict[str, Any]]:
        edge_weights: dict[tuple[str, str], float] = defaultdict(float)
        route_sequences: dict[str, list[str]] = defaultdict(list)
        for row in self._sector_rows():
            for route_id in row.get("route_ids") or []:
                route_sequences[route_id].append(event_sector_id(row))
        for sectors in route_sequences.values():
            for left, right in zip(sectors, sectors[1:]):
                if not left or not right or left == right:
                    continue
                edge_weights[(left, right)] += 1.0
        return [
            {"from_sector": left, "to_sector": right, "weight": weight}
            for (left, right), weight in sorted(
                edge_weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
            )
        ]

    def _base_automation_confidence(self) -> float:
        if not self._node_health_events:
            return 1.0
        return max(
            0.0,
            min(
                1.0,
                sum(
                    payload_float(event_payload(event), "automation_confidence", 1.0)
                    for event in self._node_health_events
                )
                / len(self._node_health_events),
            ),
        )


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.25:
        return "low"
    return "none"
