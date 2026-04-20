from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from selfsuvis.pipeline.fusion.object_state import ObjectFusionResult
    from selfsuvis.pipeline.fusion.map_state import MapFusionResult
    from selfsuvis.pipeline.fusion.semantic_priors import SemanticPrior


@dataclass
class PlatformPosteriorSample:
    t_sec: float
    position_enu_m: Dict[str, float]
    velocity_enu_mps: Dict[str, float]
    covariance_trace: float
    quality: str
    measurement_kinds: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_sec": self.t_sec,
            "position_enu_m": self.position_enu_m,
            "velocity_enu_mps": self.velocity_enu_mps,
            "covariance_trace": self.covariance_trace,
            "quality": self.quality,
            "measurement_kinds": list(self.measurement_kinds),
        }


@dataclass
class PlatformFusionResult:
    enabled: bool
    status: str
    reason: str = ""
    source: str = "platform_kalman_v1"
    origin_lla: Optional[Dict[str, float]] = None
    telemetry_sources: List[str] = field(default_factory=list)
    measurement_counts: Dict[str, int] = field(default_factory=dict)
    posterior_samples: List[PlatformPosteriorSample] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        final_sample = self.posterior_samples[-1].to_dict() if self.posterior_samples else None
        cov_values = [s.covariance_trace for s in self.posterior_samples]
        return {
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
            "origin_lla": self.origin_lla,
            "telemetry_sources": list(self.telemetry_sources),
            "measurement_counts": dict(self.measurement_counts),
            "frame_count": len(self.posterior_samples),
            "mean_covariance_trace": (sum(cov_values) / len(cov_values)) if cov_values else None,
            "final_covariance_trace": cov_values[-1] if cov_values else None,
            "final_state": final_sample,
            "diagnostics": dict(self.diagnostics),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.summary()
        payload["posterior_samples"] = [sample.to_dict() for sample in self.posterior_samples]
        return payload


@dataclass
class FullFusionResult:
    """Composite result from all four fusion layers."""
    platform: PlatformFusionResult
    object_state: "ObjectFusionResult"
    map_state: "MapFusionResult"
    semantic_prior: "SemanticPrior"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.to_dict(),
            "object_state": self.object_state.to_dict(),
            "map_state": self.map_state.to_dict(),
            "semantic_prior": {
                "scene_type": self.semantic_prior.scene_type,
                "process_noise_scale": self.semantic_prior.process_noise_scale,
                "gps_noise_scale": self.semantic_prior.gps_noise_scale,
                "temporal_noise_scale": self.semantic_prior.temporal_noise_scale,
                "urban_canyon_detected": self.semantic_prior.urban_canyon_detected,
                "dominant_labels": self.semantic_prior.dominant_labels,
            },
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "platform_status": self.platform.status,
            "object_tracks": self.object_state.track_count,
            "confirmed_tracks": self.object_state.confirmed_tracks,
            "map_smoother": self.map_state.diagnostics.get("smoother_applied", False),
            "map_sfm_measurements": self.map_state.diagnostics.get("sfm_measurements", 0),
            "scene_type": self.semantic_prior.scene_type,
            "process_noise_scale": self.semantic_prior.process_noise_scale,
        }
