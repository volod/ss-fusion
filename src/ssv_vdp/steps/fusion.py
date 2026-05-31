"""Local-run platform-state fusion helpers."""

from pathlib import Path
from typing import Any

from selfsuvis.pipeline.fusion import run_full_state_fusion, run_platform_state_fusion
from selfsuvis.pipeline.media.gps import extract_gps

from .common import _log, write_json_artifact
from .report import write_state_fusion_md


def step_platform_state_fusion(
    video_path: Path,
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> dict[str, Any]:
    frame_times_sec = [t_sec for _frame_path, t_sec in frame_list]
    gps_samples = extract_gps(str(video_path), [t_sec * 1000.0 for t_sec in frame_times_sec])
    fusion_result = run_platform_state_fusion(
        video_path=str(video_path),
        frame_times_sec=frame_times_sec,
        gps_samples=gps_samples,
    )
    out_json = video_dir / "state_fusion.json"
    out_md = video_dir / "state_fusion.md"
    write_json_artifact(out_json, fusion_result.to_dict())
    write_state_fusion_md(out_md, video_name, fusion_result)
    summary = fusion_result.summary()
    if fusion_result.status == "ok":
        _log.info(
            "  [ok] Platform-state fusion: %d posterior samples (%s)",
            len(fusion_result.posterior_samples),
            ", ".join(fusion_result.telemetry_sources) or "no telemetry sources",
        )
    else:
        _log.info(
            "  -> Platform-state fusion skipped: %s", fusion_result.reason or fusion_result.status
        )
    return {
        "skipped": fusion_result.status != "ok",
        "summary": summary,
        "posterior_samples": fusion_result.posterior_samples,
        "json_path": str(out_json),
        "md_path": str(out_md),
    }


def step_full_state_fusion(
    video_path: Path,
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    sfm_frame_positions: list[dict[str, Any]] | None = None,
    tracking_results: list[dict[str, Any]] | None = None,
    gemma_analysis: dict[str, Any] | None = None,
    qwen_captions: list[dict[str, Any]] | None = None,
    rssm_surprise_mean: float | None = None,
) -> dict[str, Any]:
    """Run full four-layer probabilistic state fusion.

    Layers:
      1. Semantic priors from Gemma/Qwen/RSSM
      2. Platform Kalman (GPS baseline)
      3. Map-state: GPS + SfM visual-pose constraints + RTS smoother
      4. Object-state: per-track Kalman with Mahalanobis gating + RTS smoother

    Writes full_state_fusion.json to video_dir.
    """
    frame_times_sec = [t_sec for _fp, t_sec in frame_list]
    gps_samples = extract_gps(str(video_path), [t * 1000.0 for t in frame_times_sec])

    result = run_full_state_fusion(
        video_path=str(video_path),
        frame_times_sec=frame_times_sec,
        gps_samples=gps_samples,
        sfm_frame_positions=sfm_frame_positions,
        tracking_results=tracking_results,
        gemma_analysis=gemma_analysis,
        qwen_captions=qwen_captions,
        rssm_surprise_mean=rssm_surprise_mean,
    )

    out_json = video_dir / "full_state_fusion.json"
    write_json_artifact(out_json, result.to_dict())

    summary = result.summary()
    _log.info(
        "  [ok] Full state fusion: platform=%s | tracks=%d (confirmed=%d) | "
        "map smoother=%s | SfM constraints=%d | scene=%s",
        summary.get("platform_status", "n/a"),
        summary.get("object_tracks", 0),
        summary.get("confirmed_tracks", 0),
        summary.get("map_smoother", False),
        summary.get("map_sfm_measurements", 0),
        summary.get("scene_type", "unknown"),
    )

    return {
        "skipped": False,
        "summary": summary,
        "json_path": str(out_json),
        "platform_status": result.platform.status,
        "track_count": result.object_state.track_count,
        "confirmed_tracks": result.object_state.confirmed_tracks,
        "map_smoother": result.map_state.diagnostics.get("smoother_applied", False),
        "sfm_measurements": result.map_state.diagnostics.get("sfm_measurements", 0),
        "scene_type": result.semantic_prior.scene_type,
        "per_frame_object_states": [
            [s.to_dict() for s in frame_samples] for frame_samples in result.object_state.per_frame
        ],
        "smoothed_trajectory": [s.to_dict() for s in result.map_state.smoothed_samples],
    }
