"""Environmental field-state estimation for local runs."""

import json
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.fusion import FieldCellEstimate, FieldObservation, FieldStateResult
from selfsuvis.pipeline.vision.rf_analyzer import RFSignalAnalyzer, _find_iq_sidecar

_log = get_logger("pipeline.local")

_VISIBILITY_TERMS = (
    "fog",
    "smoke",
    "dust",
    "haze",
    "rain",
    "blur",
    "low visibility",
    "mist",
)
_THERMAL_TERMS = (
    "fire",
    "flame",
    "burn",
    "hot",
    "overheat",
    "thermal",
    "glow",
    "smolder",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _trend_label(gradient: float) -> str:
    if gradient > 0.03:
        return "worsening"
    if gradient < -0.03:
        return "improving"
    return "stable"


def _linear_gradient(observations: list[FieldObservation]) -> float:
    if len(observations) < 2:
        return 0.0
    xs = [float(obs.t_sec) for obs in observations]
    ys = [float(obs.intensity) for obs in observations]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-6:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return numer / denom


def _field_cell_from_observations(
    field_type: str,
    cell_id: str,
    observations: list[FieldObservation],
) -> FieldCellEstimate | None:
    if not observations:
        return None
    intensities = [obs.intensity for obs in observations]
    uncertainties = [obs.uncertainty for obs in observations]
    support_frames: list[str] = []
    evidence_sources: list[str] = []
    for obs in observations:
        if obs.frame_path and obs.frame_path not in support_frames:
            support_frames.append(obs.frame_path)
        for source in obs.evidence_sources:
            if source not in evidence_sources:
                evidence_sources.append(source)
    return FieldCellEstimate(
        cell_id=cell_id,
        field_type=field_type,
        intensity_mean=_clamp01(sum(intensities) / len(intensities)),
        intensity_uncertainty=_clamp01(sum(uncertainties) / len(uncertainties)),
        temporal_gradient=float(_linear_gradient(observations)),
        source_count=len(evidence_sources),
        evidence_sources=evidence_sources,
        support_frames=support_frames,
        metadata={"n_observations": len(observations)},
    )


def _build_visibility_observations(
    frame_list: list[tuple[str, float]],
    depth_result: dict[str, Any],
    physical_state_result: dict[str, Any],
    caption_results: list[dict[str, Any]],
    unidrive_result: dict[str, Any],
) -> list[FieldObservation]:
    depth_rows = {
        str(r.get("frame_path", "")): r
        for r in (depth_result.get("depth_results") or [])
        if r.get("frame_path")
    }
    captions_by_path = {
        str(r.get("frame_path", "")): r for r in (caption_results or []) if r.get("frame_path")
    }
    unidrive_rows = {
        str(r.get("frame_path", "")): r
        for r in (unidrive_result.get("results") or [])
        if r.get("frame_path")
    }
    occupancy = _safe_float(physical_state_result.get("near_field_occupancy_density", 0.0), 0.0)

    observations: list[FieldObservation] = []
    for frame_path, t_sec in frame_list:
        fp = str(frame_path)
        depth_row = depth_rows.get(fp, {})
        caption_row = captions_by_path.get(fp, {})
        unidrive_row = unidrive_rows.get(fp, {})

        sources: list[str] = []
        depth_term = 0.0
        if depth_row:
            if (
                depth_row.get("depth_error")
                or depth_row.get("depth_unavailable")
                or depth_row.get("depth_disabled")
            ):
                depth_term = 1.0
                sources.append("depth_confidence_drop")
            else:
                depth_conf = _safe_float(depth_row.get("depth_confidence", 1.0), 1.0)
                near_ratio = _safe_float(
                    depth_row.get("near_ratio", depth_row.get("near_frac", 0.0)),
                    0.0,
                )
                depth_term = _clamp01((1.0 - depth_conf) * 0.75 + near_ratio * 0.25)
                if depth_conf < 0.65:
                    sources.append("depth_confidence_drop")

        semantic_blob = " ".join(
            [
                str(caption_row.get("caption", "") or caption_row.get("description", "") or ""),
                str((unidrive_row.get("understanding") or {}).get("scene_summary", "") or ""),
                str((unidrive_row.get("perception") or {}).get("environment", "") or ""),
            ]
        ).lower()
        semantic_term = 0.0
        if any(term in semantic_blob for term in _VISIBILITY_TERMS):
            semantic_term = 0.7
            sources.append("unidrive_visibility_semantics")

        intensity = _clamp01(depth_term * 0.55 + semantic_term * 0.25 + occupancy * 0.20)
        if intensity <= 0.05:
            continue
        uncertainty = _clamp01(0.55 - min(0.25, 0.10 * len(sources)))
        observations.append(
            FieldObservation(
                field_type="visibility",
                cell_id="clip_local",
                frame_path=fp,
                t_sec=float(t_sec),
                intensity=intensity,
                uncertainty=uncertainty,
                evidence_sources=sources or ["occupancy_density"],
                metadata={
                    "depth_term": round(depth_term, 4),
                    "semantic_term": round(semantic_term, 4),
                    "occupancy_density": round(occupancy, 4),
                },
            )
        )
    return observations


def _build_rf_observations(
    video_path: Path,
    frame_list: list[tuple[str, float]],
) -> list[FieldObservation]:
    iq_path, _source = _find_iq_sidecar(str(video_path))
    if not iq_path:
        return []
    analyzer = RFSignalAnalyzer()
    rf_rows = analyzer.analyze_video(
        str(video_path),
        [float(t_sec) for _fp, t_sec in frame_list],
        audio_wav_path=None,
    )
    observations: list[FieldObservation] = []
    for (frame_path, t_sec), row in zip(frame_list, rf_rows):
        signal = row.get("rf_signal") or {}
        if (
            not signal
            or signal.get("rf_insufficient_samples")
            or signal.get("rf_spectrogram_error")
        ):
            continue
        flatness = _safe_float(signal.get("spectral_flatness", 0.0), 0.0)
        occupied = _safe_float(signal.get("occupied_bw_ratio", 0.0), 0.0)
        snr_db = _safe_float(signal.get("snr_db", 0.0), 0.0)
        low_snr = _clamp01((12.0 - snr_db) / 20.0)
        intensity = _clamp01(flatness * 0.45 + occupied * 0.35 + low_snr * 0.20)
        if intensity <= 0.10:
            continue
        sources: list[str] = []
        if flatness > 0.55:
            sources.append("rf_spectral_flatness")
        if occupied > 0.40:
            sources.append("rf_occupied_bandwidth")
        if low_snr > 0.25:
            sources.append("rf_low_snr")
        uncertainty = _clamp01(0.45 - min(0.20, 0.08 * len(sources)))
        observations.append(
            FieldObservation(
                field_type="rf_interference",
                cell_id="clip_local",
                frame_path=str(frame_path),
                t_sec=float(t_sec),
                intensity=intensity,
                uncertainty=uncertainty,
                evidence_sources=sources or ["rf_signal_sidecar"],
                metadata={
                    "snr_db": round(snr_db, 2),
                    "spectral_flatness": round(flatness, 4),
                    "occupied_bw_ratio": round(occupied, 4),
                },
            )
        )
    return observations


def _build_thermal_observations(
    frame_list: list[tuple[str, float]],
    caption_results: list[dict[str, Any]],
    unidrive_result: dict[str, Any],
) -> list[FieldObservation]:
    captions_by_path = {
        str(r.get("frame_path", "")): r for r in (caption_results or []) if r.get("frame_path")
    }
    unidrive_rows = {
        str(r.get("frame_path", "")): r
        for r in (unidrive_result.get("results") or [])
        if r.get("frame_path")
    }
    observations: list[FieldObservation] = []
    for frame_path, t_sec in frame_list:
        fp = str(frame_path)
        caption_row = captions_by_path.get(fp, {})
        unidrive_row = unidrive_rows.get(fp, {})
        blob = " ".join(
            [
                str(caption_row.get("caption", "") or caption_row.get("description", "") or ""),
                str((unidrive_row.get("understanding") or {}).get("scene_summary", "") or ""),
                str((unidrive_row.get("perception") or {}).get("scene_elements", "") or ""),
            ]
        ).lower()
        matches = [term for term in _THERMAL_TERMS if term in blob]
        if not matches:
            continue
        intensity = _clamp01(0.35 + 0.12 * len(matches))
        observations.append(
            FieldObservation(
                field_type="thermal_anomaly",
                cell_id="clip_local",
                frame_path=fp,
                t_sec=float(t_sec),
                intensity=intensity,
                uncertainty=0.65,
                evidence_sources=["thermal_semantic_cues", "caption_semantics"],
                metadata={"matched_terms": matches[:4]},
            )
        )
    return observations


def step_field_state(
    video_path: Path,
    video_dir: Path,
    video_name: str,
    frame_list: list[tuple[str, float]],
    depth_result: dict[str, Any],
    physical_state_result: dict[str, Any],
    caption_results: list[dict[str, Any]],
    unidrive_result: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate coarse continuous-field summaries from local evidence."""
    t0 = time.time()

    visibility_obs = _build_visibility_observations(
        frame_list,
        depth_result,
        physical_state_result,
        caption_results,
        unidrive_result,
    )
    rf_obs = _build_rf_observations(video_path, frame_list)
    thermal_obs = _build_thermal_observations(frame_list, caption_results, unidrive_result)

    all_observations = visibility_obs + rf_obs + thermal_obs
    if not all_observations:
        result = FieldStateResult(
            enabled=True,
            status="skipped",
            reason="no field evidence available",
            diagnostics={"video_name": video_name},
        )
        payload = result.to_dict()
        payload["skipped"] = True
        payload["elapsed_sec"] = round(time.time() - t0, 3)
        _write_json(video_dir, payload)
        _log.info("  [field state] no field evidence available — skipped")
        return payload

    cells: list[FieldCellEstimate] = []
    clip_level_fields: dict[str, dict[str, Any]] = {}
    field_types: list[str] = []
    for field_type, observations in (
        ("visibility", visibility_obs),
        ("rf_interference", rf_obs),
        ("thermal_anomaly", thermal_obs),
    ):
        cell = _field_cell_from_observations(field_type, "clip_local", observations)
        if cell is None:
            continue
        cells.append(cell)
        field_types.append(field_type)
        clip_level_fields[field_type] = {
            "mean": round(cell.intensity_mean, 4),
            "uncertainty": round(cell.intensity_uncertainty, 4),
            "trend": _trend_label(cell.temporal_gradient),
            "temporal_gradient": round(cell.temporal_gradient, 4),
            "source_count": cell.source_count,
            "evidence_sources": list(cell.evidence_sources),
            "support_frames": list(cell.support_frames[:8]),
        }

    result = FieldStateResult(
        enabled=True,
        status="ok",
        field_types=field_types,
        cells=cells,
        clip_level_fields=clip_level_fields,
        observations=all_observations,
        diagnostics={
            "video_name": video_name,
            "rf_sidecar_present": bool(_find_iq_sidecar(str(video_path))[0]),
            "observation_count_by_type": {
                "visibility": len(visibility_obs),
                "rf_interference": len(rf_obs),
                "thermal_anomaly": len(thermal_obs),
            },
        },
    )
    payload = result.to_dict()
    payload["skipped"] = False
    payload["elapsed_sec"] = round(time.time() - t0, 3)
    _write_json(video_dir, payload)
    _log.info(
        "  [ok] Field state: fields=%s observations=%d",
        field_types or ["none"],
        len(all_observations),
    )
    return payload


def _write_json(video_dir: Path, payload: dict[str, Any]) -> None:
    out = video_dir / "field_state_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
