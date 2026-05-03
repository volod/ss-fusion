"""Map-state fusion and trajectory smoothing.

Extends the platform Kalman filter with two additional capabilities:

  1. **Visual-pose constraints**: SfM camera centres are Sim(3)-aligned to
     GPS-ENU and injected as additional position measurements (kind="sfm_position"),
     tightening the estimate between GPS readings.

  2. **RTS trajectory smoothing**: After the forward Kalman pass, a Rauch-
     Tung-Striebel backward smoother is run over the per-frame posterior
     states to reduce lag and uncertainty throughout the trajectory.

  3. **Semantic noise adaptation**: process and GPS noise are scaled by the
     SemanticPrior before the forward pass.

Returns a MapFusionResult containing the smoothed per-frame states alongside
the alignment diagnostics.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.fusion.filters.platform import PlatformStateFilter
from selfsuvis.pipeline.fusion.filters.rts_smoother import FilteredStep, rts_smooth
from selfsuvis.pipeline.fusion.measurements import PlatformMeasurement
from selfsuvis.pipeline.fusion.semantic_priors import SemanticPrior
from selfsuvis.pipeline.fusion.state import PlatformPosteriorSample
from selfsuvis.pipeline.fusion.summaries import (
    _build_baro_measurements,
    _build_imu_measurements,
    _Event,
    _measurement_covariance,
    _recent_measurement_kinds,
    _sample_quality,
)
from selfsuvis.pipeline.fusion.visual_pose import align_sfm_to_enu
from selfsuvis.pipeline.mapping.gps_registration import gps_to_enu

logger = get_logger(__name__)


@dataclass
class MapStateSample:
    """Smoothed platform state estimate at one frame timestamp."""
    t_sec: float
    position_enu_m: dict[str, float]
    velocity_enu_mps: dict[str, float]
    covariance_diag: list[float]   # diagonal of 6×6 P (6 values)
    cov_trace: float
    quality: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_sec": self.t_sec,
            "position_enu_m": self.position_enu_m,
            "velocity_enu_mps": self.velocity_enu_mps,
            "covariance_diag": self.covariance_diag,
            "cov_trace": self.cov_trace,
            "quality": self.quality,
        }


@dataclass
class MapFusionResult:
    enabled: bool
    status: str
    reason: str = ""
    source: str = "map_rts_v1"
    sfm_alignment: dict[str, Any] | None = None
    smoothed_samples: list[MapStateSample] = field(default_factory=list)
    raw_samples: list[PlatformPosteriorSample] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
            "sfm_alignment": self.sfm_alignment,
            "frame_count": len(self.smoothed_samples),
            "diagnostics": self.diagnostics,
            "smoothed_samples": [s.to_dict() for s in self.smoothed_samples],
        }


def run_map_state_fusion(
    *,
    video_path: str,
    frame_times_sec: Sequence[float],
    gps_samples: Sequence[dict[str, float] | None],
    sfm_frame_positions: Sequence[dict[str, Any]] | None = None,
    prior: SemanticPrior | None = None,
    enable_smoother: bool = True,
) -> MapFusionResult:
    """Run map-state fusion with visual pose constraints and RTS smoothing.

    Args:
        video_path:         Path to source video (for IMU/baro sidecar lookup).
        frame_times_sec:    Frame timestamps in seconds.
        gps_samples:        Per-frame GPS dicts (lat/lon/alt), None for missing.
        sfm_frame_positions: Per-frame SfM camera centres from build_sparse_map()
                             (list of {"t_sec": float, "position": {x,y,z}}).
        prior:              SemanticPrior for adaptive noise scaling.
        enable_smoother:    Whether to run RTS backward pass.

    Returns:
        MapFusionResult with smoothed trajectory and alignment diagnostics.
    """
    if not settings.STATE_FUSION_ENABLED:
        return MapFusionResult(
            enabled=False, status="skipped", reason="STATE_FUSION_ENABLED=false"
        )
    if not frame_times_sec:
        return MapFusionResult(enabled=True, status="skipped", reason="no frame timestamps")

    # ── 1. Build GPS measurements ────────────────────────────────────────────
    noise_scale = prior.process_noise_scale if prior else 1.0
    gps_scale = prior.gps_noise_scale if prior else 1.0
    proc_pos_std = float(settings.STATE_FUSION_PROCESS_POS_STD_M) * noise_scale
    proc_vel_std = float(settings.STATE_FUSION_PROCESS_VEL_STD_MPS) * noise_scale
    gps_pos_std = float(settings.STATE_FUSION_GPS_POS_STD_M) * gps_scale

    origin_lla, gps_measurements = _build_gps_measurements_scaled(
        frame_times_sec, gps_samples, gps_pos_std
    )
    if origin_lla is None or not gps_measurements:
        return MapFusionResult(
            enabled=True, status="skipped",
            reason="no GPS measurements aligned to frames"
        )

    imu_measurements = _build_imu_measurements(video_path)
    baro_measurements = _build_baro_measurements(video_path, origin_lla)

    # ── 2. Visual pose constraints (SfM → ENU alignment) ────────────────────
    sfm_alignment_info: dict[str, Any] | None = None
    sfm_measurements: list[PlatformMeasurement] = []
    if sfm_frame_positions:
        gps_enu_by_t: dict[float, tuple[float, float, float]] = {}
        for t_sec, gps in zip(frame_times_sec, gps_samples):
            if gps is None:
                continue
            tx, ty, tz = gps_to_enu(
                float(gps["lat"]), float(gps["lon"]), float(gps.get("alt", 0.0)),
                origin_lla["lat"], origin_lla["lon"], origin_lla["alt"],
            )
            gps_enu_by_t[float(t_sec)] = (tx, ty, tz)

        sfm_alignment_info, sfm_measurements = align_sfm_to_enu(
            sfm_positions=sfm_frame_positions,
            gps_enu_by_t=gps_enu_by_t,
            gps_std_m=gps_pos_std,
            sfm_base_std_m=float(getattr(settings, "STATE_FUSION_SFM_POS_STD_M", 2.0)),
        )
        if sfm_measurements:
            logger.info(
                "Visual-pose: %d SfM measurements added (alignment RMSE=%.2f m)",
                len(sfm_measurements),
                sfm_alignment_info.get("rmse_m", float("nan")),
            )

    # ── 3. Merge all measurements and build event timeline ───────────────────
    all_measurements = (
        list(gps_measurements)
        + list(imu_measurements)
        + list(baro_measurements)
        + sfm_measurements
    )
    _kind_priority = {
        "imu_accel": 0, "gps_position": 1, "sfm_position": 1,
        "barometer_altitude": 2,
    }
    all_measurements.sort(
        key=lambda m: (m.t_sec, _kind_priority.get(m.kind, 9))
    )

    events: list[_Event] = []
    for m in all_measurements:
        events.append(_Event(
            t_sec=m.t_sec,
            priority=_kind_priority.get(m.kind, 9),
            event_type="measurement",
            payload=m,
        ))
    for t in frame_times_sec:
        events.append(_Event(
            t_sec=float(t), priority=3,
            event_type="frame_sample", payload=float(t),
        ))
    events.sort(key=lambda e: (e.t_sec, e.priority))

    # ── 4. Forward Kalman pass ───────────────────────────────────────────────
    flt = PlatformStateFilter(
        process_pos_std_m=proc_pos_std,
        process_vel_std_mps=proc_vel_std,
        init_vel_std_mps=float(settings.STATE_FUSION_INIT_VEL_STD_MPS),
    )

    raw_samples: list[PlatformPosteriorSample] = []
    filtered_history: list[FilteredStep] = []
    context_gap = float(settings.STATE_FUSION_CONTEXT_GAP_SEC)

    for event in events:
        flt.predict(event.t_sec)
        if event.event_type == "measurement":
            m = event.payload
            if m.kind == "imu_accel":
                flt.set_acceleration(m)
            elif m.kind in ("gps_position", "sfm_position"):
                flt.update_position(m)
            elif m.kind == "barometer_altitude":
                flt.update_altitude(m)
            continue

        if not flt.is_initialized():
            continue
        assert flt.x is not None and flt.P is not None

        cov_trace = float(np.trace(flt.P))
        raw_samples.append(PlatformPosteriorSample(
            t_sec=float(event.payload),
            position_enu_m={"x": float(flt.x[0]), "y": float(flt.x[1]), "z": float(flt.x[2])},
            velocity_enu_mps={"x": float(flt.x[3]), "y": float(flt.x[4]), "z": float(flt.x[5])},
            covariance_trace=cov_trace,
            quality=_sample_quality(cov_trace),
            measurement_kinds=_recent_measurement_kinds(all_measurements, float(event.payload), context_gap),
        ))
        filtered_history.append(
            FilteredStep(t_sec=float(event.payload), x=flt.x.copy(), P=flt.P.copy())
        )

    if not raw_samples:
        return MapFusionResult(
            enabled=True, status="skipped",
            reason="filter never initialized",
            sfm_alignment=sfm_alignment_info,
        )

    # ── 5. RTS backward smoother ─────────────────────────────────────────────
    if enable_smoother and len(filtered_history) >= 2:
        rts_result = rts_smooth(filtered_history, proc_pos_std, proc_vel_std)
    else:
        rts_result = None

    smoothed_samples: list[MapStateSample] = []
    for i, raw in enumerate(raw_samples):
        if rts_result is not None:
            x_s = rts_result[i].x
            P_s = rts_result[i].P
            cov_trace_s = rts_result[i].cov_trace
        else:
            x_s = filtered_history[i].x
            P_s = filtered_history[i].P
            cov_trace_s = float(np.trace(P_s))

        smoothed_samples.append(MapStateSample(
            t_sec=raw.t_sec,
            position_enu_m={"x": float(x_s[0]), "y": float(x_s[1]), "z": float(x_s[2])},
            velocity_enu_mps={"x": float(x_s[3]), "y": float(x_s[4]), "z": float(x_s[5])},
            covariance_diag=[float(np.sqrt(abs(P_s[k, k]))) for k in range(6)],
            cov_trace=cov_trace_s,
            quality=_sample_quality(cov_trace_s),
        ))

    cov_values = [s.cov_trace for s in smoothed_samples]
    raw_cov_values = [s.covariance_trace for s in raw_samples]
    innovation_values = list(flt.innovation_norms)

    logger.info(
        "Map-state fusion: %d frames | smoother=%s | "
        "mean cov trace: raw=%.2f → smoothed=%.2f",
        len(smoothed_samples),
        "rts" if rts_result is not None else "off",
        (sum(raw_cov_values) / len(raw_cov_values)) if raw_cov_values else 0.0,
        (sum(cov_values) / len(cov_values)) if cov_values else 0.0,
    )

    return MapFusionResult(
        enabled=True,
        status="ok",
        source="map_rts_v1",
        sfm_alignment=sfm_alignment_info,
        smoothed_samples=smoothed_samples,
        raw_samples=raw_samples,
        diagnostics={
            "frames": len(smoothed_samples),
            "smoother_applied": rts_result is not None,
            "sfm_measurements": len(sfm_measurements),
            "gps_measurements": len(gps_measurements),
            "imu_measurements": len(imu_measurements),
            "baro_measurements": len(baro_measurements),
            "mean_cov_trace_raw": (
                sum(raw_cov_values) / len(raw_cov_values)
            ) if raw_cov_values else None,
            "mean_cov_trace_smoothed": (
                sum(cov_values) / len(cov_values)
            ) if cov_values else None,
            "mean_innovation_norm": (
                sum(innovation_values) / len(innovation_values)
            ) if innovation_values else None,
            "process_noise_scale": round(noise_scale, 3),
            "gps_noise_scale": round(gps_scale, 3),
        },
    )


# ── Private helpers (extend summaries.py helpers with noise scaling) ─────────

def _build_gps_measurements_scaled(
    frame_times_sec: Sequence[float],
    gps_samples: Sequence[dict[str, float] | None],
    gps_pos_std: float,
) -> tuple[dict[str, float] | None, list[PlatformMeasurement]]:
    """Like summaries._build_gps_measurements but with explicit noise std."""
    origin = next((g for g in gps_samples if g is not None), None)
    if origin is None:
        return None, []
    origin_lla = {
        "lat": float(origin["lat"]),
        "lon": float(origin["lon"]),
        "alt": float(origin.get("alt", 0.0)),
    }
    cov = _measurement_covariance([gps_pos_std ** 2] * 3)
    measurements: list[PlatformMeasurement] = []
    for t_sec, sample in zip(frame_times_sec, gps_samples):
        if sample is None:
            continue
        tx, ty, tz = gps_to_enu(
            float(sample["lat"]), float(sample["lon"]), float(sample.get("alt", 0.0)),
            origin_lla["lat"], origin_lla["lon"], origin_lla["alt"],
        )
        measurements.append(PlatformMeasurement(
            kind="gps_position",
            t_sec=float(t_sec),
            values=(float(tx), float(ty), float(tz)),
            covariance=cov,
            source="video_gps",
            frame="enu",
        ))
    return origin_lla, measurements
