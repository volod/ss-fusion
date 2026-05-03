"""Schema for mission-level global threat aggregation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SectorThreatState:
    sector_id: str
    threat_score: float
    risk_level: str
    supporting_videos: list[str] = field(default_factory=list)
    route_ids: list[str] = field(default_factory=list)
    primitive_types: list[str] = field(default_factory=list)
    observation_count: int = 0
    mean_uncertainty: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_id": self.sector_id,
            "threat_score": float(self.threat_score),
            "risk_level": self.risk_level,
            "supporting_videos": list(self.supporting_videos),
            "route_ids": list(self.route_ids),
            "primitive_types": list(self.primitive_types),
            "observation_count": int(self.observation_count),
            "mean_uncertainty": float(self.mean_uncertainty),
            "evidence_sources": list(self.evidence_sources),
            "metadata": dict(self.metadata),
        }


@dataclass
class RouteAdvisory:
    route_id: str
    advisory_score: float
    recommended_action: str
    sector_ids: list[str] = field(default_factory=list)
    supporting_videos: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "advisory_score": float(self.advisory_score),
            "recommended_action": self.recommended_action,
            "sector_ids": list(self.sector_ids),
            "supporting_videos": list(self.supporting_videos),
            "evidence_sources": list(self.evidence_sources),
            "metadata": dict(self.metadata),
        }


@dataclass
class PersistentAnomaly:
    anomaly_id: str
    anomaly_type: str
    sector_ids: list[str] = field(default_factory=list)
    threat_score: float = 0.0
    persistence_count: int = 0
    supporting_videos: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "sector_ids": list(self.sector_ids),
            "threat_score": float(self.threat_score),
            "persistence_count": int(self.persistence_count),
            "supporting_videos": list(self.supporting_videos),
            "evidence_sources": list(self.evidence_sources),
            "metadata": dict(self.metadata),
        }


@dataclass
class GlobalThreatResult:
    enabled: bool
    status: str
    reason: str = ""
    global_threat_map: list[dict[str, Any]] = field(default_factory=list)
    sector_risk_levels: list[SectorThreatState] = field(default_factory=list)
    persistent_anomalies: list[PersistentAnomaly] = field(default_factory=list)
    route_advisories: list[RouteAdvisory] = field(default_factory=list)
    threat_corridor_graph: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "global_threat_map": list(self.global_threat_map),
            "sector_risk_levels": [row.to_dict() for row in self.sector_risk_levels],
            "persistent_anomalies": [row.to_dict() for row in self.persistent_anomalies],
            "route_advisories": [row.to_dict() for row in self.route_advisories],
            "threat_corridor_graph": list(self.threat_corridor_graph),
            "diagnostics": dict(self.diagnostics),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summary()
