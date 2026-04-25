
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.fusion.filters import PlatformStateFilter
from selfsuvis.pipeline.fusion.measurements import PlatformMeasurement
from selfsuvis.pipeline.fusion.sidecars import load_baro_sidecar, load_imu_sidecar, normalize_baro_rows
from selfsuvis.pipeline.fusion.state import PlatformFusionResult, PlatformPosteriorSample
from selfsuvis.pipeline.mapping.gps_registration import gps_to_enu

logger = logging.getLogger(__name__)


def _measurement_covariance(diag: Sequence[float]) -> Tuple[Tuple[float, ...], ...]:
    mat = np.diag(diag).astype(np.float64)
    return tuple(tuple(float(v) for v in row) for row in mat)


def _build_gps_measurements(
    frame_times_sec: Sequence[float],
    gps_samples: Sequence[Optional[Dict[str, float]]],
) -> Tuple[Optional[Dict[str, float]], List[PlatformMeasurement]]:
    origin = next((g for g in gps_samples if g is not None), None)
    if origin is None:
        return None, []

    origin_lla = {"lat": float(origin["lat"]), "lon": float(origin["lon"]), "alt": float(origin.get("alt", 0.0))}
    pos_std = float(settings.STATE_FUSION_GPS_POS_STD_M)
    gps_cov = _measurement_covariance([pos_std ** 2, pos_std ** 2, pos_std ** 2])
    measurements: List[PlatformMeasurement] = []
    for t_sec, sample in zip(frame_times_sec, gps_samples):
        if sample is None:
            continue
        tx, ty, tz = gps_to_enu(
            float(sample["lat"]),
            float(sample["lon"]),
            float(sample.get("alt", 0.0)),
            origin_lla["lat"],
            origin_lla["lon"],
            origin_lla["alt"],
        )
        measurements.append(
            PlatformMeasurement(
                kind="gps_position",
                t_sec=float(t_sec),
                values=(float(tx), float(ty), float(tz)),
                covariance=gps_cov,
                source="video_gps",
                frame="enu",
            )
        )
    return origin_lla, measurements


def _build_imu_measurements(video_path: str) -> List[PlatformMeasurement]:
    accel_std = float(settings.STATE_FUSION_IMU_ACCEL_STD_MPS2)
    cov = _measurement_covariance([accel_std ** 2, accel_std ** 2, accel_std ** 2])
    rows = load_imu_sidecar(video_path)
    measurements: List[PlatformMeasurement] = []
    for row in rows:
        t_sec = float(row.get("t", row.get("timestamp", 0.0)) or 0.0)
        ax = float(row.get("ax", 0.0))
        ay = float(row.get("ay", 0.0))
        az = float(row.get("az", 0.0)) + 9.81
        measurements.append(
            PlatformMeasurement(
                kind="imu_accel",
                t_sec=t_sec,
                values=(ax, ay, az),
                covariance=cov,
                source="imu_sidecar",
                frame="enu_assumed",
                quality="approx_world_frame",
            )
        )
    return measurements


def _build_baro_measurements(video_path: str, origin_lla: Dict[str, float]) -> List[PlatformMeasurement]:
    alt_std = float(settings.STATE_FUSION_BARO_ALT_STD_M)
    cov = _measurement_covariance([alt_std ** 2])
    rows = normalize_baro_rows(load_baro_sidecar(video_path), origin_lla["alt"])
    measurements: List[PlatformMeasurement] = []
    for row in rows:
        measurements.append(
            PlatformMeasurement(
                kind="barometer_altitude",
                t_sec=float(row["t_sec"]),
                values=(float(row["alt_enu_m"]),),
                covariance=cov,
                source="baro_sidecar",
                frame="enu",
            )
        )
    return measurements


@dataclass(frozen=True)
class _Event:
    t_sec: float
    priority: int
    event_type: str
    payload: Any


def _sample_quality(cov_trace: float) -> str:
    if cov_trace <= 10.0:
        return "good"
    if cov_trace <= 40.0:
        return "degraded"
    return "uncertain"


def _recent_measurement_kinds(measurements: Iterable[PlatformMeasurement], t_sec: float, max_gap_sec: float) -> List[str]:
    kinds = {
        measurement.kind
        for measurement in measurements
        if abs(measurement.t_sec - t_sec) <= max_gap_sec
    }
    return sorted(kinds)


def run_platform_state_fusion(
    *,
    video_path: str,
    frame_times_sec: Sequence[float],
    gps_samples: Sequence[Optional[Dict[str, float]]],
) -> PlatformFusionResult:
    """Run the platform-state fusion MVP on frame times and telemetry sidecars."""

    if not settings.STATE_FUSION_ENABLED:
        return PlatformFusionResult(enabled=False, status="skipped", reason="STATE_FUSION_ENABLED=false")

    if not frame_times_sec:
        return PlatformFusionResult(enabled=True, status="skipped", reason="no frame timestamps")

    origin_lla, gps_measurements = _build_gps_measurements(frame_times_sec, gps_samples)
    if origin_lla is None or not gps_measurements:
        return PlatformFusionResult(enabled=True, status="skipped", reason="no GPS measurements aligned to frames")

    imu_measurements = _build_imu_measurements(video_path)
    baro_measurements = _build_baro_measurements(video_path, origin_lla)

    all_measurements = list(gps_measurements) + list(imu_measurements) + list(baro_measurements)
    all_measurements.sort(key=lambda measurement: (measurement.t_sec, {"imu_accel": 0, "gps_position": 1, "barometer_altitude": 2}.get(measurement.kind, 9)))

    filter_ = PlatformStateFilter(
        process_pos_std_m=float(settings.STATE_FUSION_PROCESS_POS_STD_M),
        process_vel_std_mps=float(settings.STATE_FUSION_PROCESS_VEL_STD_MPS),
        init_vel_std_mps=float(settings.STATE_FUSION_INIT_VEL_STD_MPS),
    )

    events: List[_Event] = []
    for measurement in all_measurements:
        events.append(_Event(t_sec=measurement.t_sec, priority={"imu_accel": 0, "gps_position": 1, "barometer_altitude": 2}[measurement.kind], event_type="measurement", payload=measurement))
    for t_sec in frame_times_sec:
        events.append(_Event(t_sec=float(t_sec), priority=3, event_type="frame_sample", payload=float(t_sec)))
    events.sort(key=lambda event: (event.t_sec, event.priority))

    posterior_samples: List[PlatformPosteriorSample] = []
    for event in events:
        filter_.predict(event.t_sec)
        if event.event_type == "measurement":
            measurement = event.payload
            if measurement.kind == "imu_accel":
                filter_.set_acceleration(measurement)
            elif measurement.kind == "gps_position":
                filter_.update_position(measurement)
            elif measurement.kind == "barometer_altitude":
                filter_.update_altitude(measurement)
            continue

        if not filter_.is_initialized():
            continue
        assert filter_.x is not None and filter_.P is not None
        cov_trace = float(np.trace(filter_.P))
        posterior_samples.append(
            PlatformPosteriorSample(
                t_sec=float(event.payload),
                position_enu_m={
                    "x": float(filter_.x[0]),
                    "y": float(filter_.x[1]),
                    "z": float(filter_.x[2]),
                },
                velocity_enu_mps={
                    "x": float(filter_.x[3]),
                    "y": float(filter_.x[4]),
                    "z": float(filter_.x[5]),
                },
                covariance_trace=cov_trace,
                quality=_sample_quality(cov_trace),
                measurement_kinds=_recent_measurement_kinds(all_measurements, float(event.payload), float(settings.STATE_FUSION_CONTEXT_GAP_SEC)),
            )
        )

    telemetry_sources = ["gps"]
    if imu_measurements:
        telemetry_sources.append("imu")
    if baro_measurements:
        telemetry_sources.append("barometer")

    innovation_values = list(filter_.innovation_norms)
    diagnostics = {
        "mean_innovation_norm": (sum(innovation_values) / len(innovation_values)) if innovation_values else None,
        "max_innovation_norm": max(innovation_values) if innovation_values else None,
        "predict_steps": sum(1 for event in events if event.event_type == "frame_sample"),
        "measurement_total": len(all_measurements),
    }
    result = PlatformFusionResult(
        enabled=True,
        status="ok" if posterior_samples else "skipped",
        reason="" if posterior_samples else "filter never initialized",
        origin_lla=origin_lla,
        telemetry_sources=telemetry_sources,
        measurement_counts=dict(Counter(measurement.kind for measurement in all_measurements)),
        posterior_samples=posterior_samples,
        diagnostics=diagnostics,
    )
    return result


def run_full_state_fusion(
    *,
    video_path: str,
    frame_times_sec: Sequence[float],
    gps_samples: Sequence[Optional[Dict[str, float]]],
    sfm_frame_positions: Optional[Sequence[Dict[str, Any]]] = None,
    tracking_results: Optional[Sequence[Dict[str, Any]]] = None,
    gemma_analysis: Optional[Dict[str, Any]] = None,
    qwen_captions: Optional[Sequence[Dict[str, Any]]] = None,
    rssm_surprise_mean: Optional[float] = None,
) -> "FullFusionResult":
    """Orchestrate all four fusion layers into a single FullFusionResult.

    Layers:
      1. Semantic priors  — VLM-grounded noise adaptation
      2. Platform fusion  — GPS-only Kalman (backward-compatible)
      3. Map-state fusion — GPS + SfM visual constraints + RTS smoother
      4. Object fusion    — per-track Kalman + Mahalanobis + RTS smoother

    Args:
        video_path:          Path to source video (for IMU/baro sidecar lookup).
        frame_times_sec:     Frame timestamps in seconds.
        gps_samples:         Per-frame GPS dicts (lat/lon/alt), None = missing.
        sfm_frame_positions: Per-frame SfM camera centres from build_sparse_map().
        tracking_results:    RF-DETR per-frame detection/tracking output.
        gemma_analysis:      Structured Gemma scene-analysis output.
        qwen_captions:       Per-frame Qwen structured captions.
        rssm_surprise_mean:  Mean RSSM temporal-surprise score [0, 1].

    Returns:
        FullFusionResult combining all four subsystem results.
    """
    from selfsuvis.pipeline.fusion.map_state import run_map_state_fusion
    from selfsuvis.pipeline.fusion.object_state import run_object_state_fusion
    from selfsuvis.pipeline.fusion.semantic_priors import build_semantic_prior
    from selfsuvis.pipeline.fusion.state import FullFusionResult

    # 1 — Semantic priors
    prior = build_semantic_prior(
        gemma_analysis=gemma_analysis,
        qwen_captions=qwen_captions,
        rssm_surprise_mean=rssm_surprise_mean,
    )

    # 2 — Platform fusion (baseline GPS-only, unchanged from v1)
    platform = run_platform_state_fusion(
        video_path=video_path,
        frame_times_sec=frame_times_sec,
        gps_samples=gps_samples,
    )

    # 3 — Map-state fusion (GPS + SfM visual poses + RTS smoother)
    map_result = run_map_state_fusion(
        video_path=video_path,
        frame_times_sec=frame_times_sec,
        gps_samples=gps_samples,
        sfm_frame_positions=sfm_frame_positions,
        prior=prior,
        enable_smoother=True,
    )

    # 4 — Object-state fusion (Mahalanobis tracking + RTS smoother)
    object_result = run_object_state_fusion(
        tracking_results=tracking_results or [],
        prior=prior,
    )

    return FullFusionResult(
        platform=platform,
        object_state=object_result,
        map_state=map_result,
        semantic_prior=prior,
    )
