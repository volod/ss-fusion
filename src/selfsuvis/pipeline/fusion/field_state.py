"""Field-state schema for continuous hazard layers.

The first implementation is intentionally coarse: one or more field cells are
estimated from local video evidence and aggregated into clip-level summaries.
This provides a stable schema for downstream threat logic before dense spatial
field reconstruction exists.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FieldObservation:
    field_type: str
    cell_id: str
    frame_path: str
    t_sec: float
    intensity: float
    uncertainty: float
    evidence_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_type": self.field_type,
            "cell_id": self.cell_id,
            "frame_path": self.frame_path,
            "t_sec": float(self.t_sec),
            "intensity": float(self.intensity),
            "uncertainty": float(self.uncertainty),
            "evidence_sources": list(self.evidence_sources),
            "metadata": dict(self.metadata),
        }


@dataclass
class FieldCellEstimate:
    cell_id: str
    field_type: str
    intensity_mean: float
    intensity_uncertainty: float
    temporal_gradient: float
    source_count: int
    evidence_sources: List[str] = field(default_factory=list)
    support_frames: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "field_type": self.field_type,
            "intensity_mean": float(self.intensity_mean),
            "intensity_uncertainty": float(self.intensity_uncertainty),
            "temporal_gradient": float(self.temporal_gradient),
            "source_count": int(self.source_count),
            "evidence_sources": list(self.evidence_sources),
            "support_frames": list(self.support_frames),
            "metadata": dict(self.metadata),
        }


@dataclass
class FieldStateResult:
    enabled: bool
    status: str
    reason: str = ""
    field_types: List[str] = field(default_factory=list)
    cells: List[FieldCellEstimate] = field(default_factory=list)
    clip_level_fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    observations: List[FieldObservation] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "field_types": list(self.field_types),
            "cell_count": len(self.cells),
            "observation_count": len(self.observations),
            "clip_level_fields": dict(self.clip_level_fields),
            "diagnostics": dict(self.diagnostics),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.summary()
        payload["cells"] = [cell.to_dict() for cell in self.cells]
        payload["observations"] = [obs.to_dict() for obs in self.observations]
        return payload
