"""Schema for mission-level global threat aggregation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SectorThreatState:
    sector_id: str
    threat_score: float
    risk_level: str
    supporting_videos: List[str] = field(default_factory=list)
    route_ids: List[str] = field(default_factory=list)
    primitive_types: List[str] = field(default_factory=list)
    observation_count: int = 0
    mean_uncertainty: float = 0.0
    evidence_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    sector_ids: List[str] = field(default_factory=list)
    supporting_videos: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    sector_ids: List[str] = field(default_factory=list)
    threat_score: float = 0.0
    persistence_count: int = 0
    supporting_videos: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    global_threat_map: List[Dict[str, Any]] = field(default_factory=list)
    sector_risk_levels: List[SectorThreatState] = field(default_factory=list)
    persistent_anomalies: List[PersistentAnomaly] = field(default_factory=list)
    route_advisories: List[RouteAdvisory] = field(default_factory=list)
    threat_corridor_graph: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()
