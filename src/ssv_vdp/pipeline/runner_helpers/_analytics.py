"""Local-run analytics payload builder and emitter."""

import json
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger(__name__)


def _build_local_run_analytics_payload(summary: Any) -> dict[str, Any]:
    diagnostics = getattr(summary, "diagnostics", None)

    def _diag_float(name: str, default: float = 0.0) -> float:
        return float(getattr(diagnostics, name, default) or 0.0)

    payload: dict[str, Any] = {
        "video_name": summary.video_name,
        "n_frames": int(summary.n_frames),
        "duration_sec": float(summary.duration_sec),
        "fps": float(summary.fps),
        "domain": summary.domain,
        "top_category": summary.top_category,
        "scene_complexity": summary.scene_complexity,
        "n_scene_clusters": int(summary.n_scene_clusters),
        "artifact_count": int(summary.artifact_inventory.total_files),
        "artifact_bytes": int(summary.artifact_inventory.total_bytes),
        "has_3d_map": bool(summary.has_3d_map),
        "has_edge_model": bool(summary.has_edge_model),
        "diagnostics": {
            "modality_completeness": _diag_float("modality_completeness"),
            "quality_score": _diag_float("quality_score"),
            "detection_density_per_frame": _diag_float("detection_density_per_frame"),
            "detection_count_cv": _diag_float("detection_count_cv"),
            "detection_entropy_norm": _diag_float("detection_entropy_norm"),
            "tracking_fragmentation": _diag_float("tracking_fragmentation"),
            "track_persistence": _diag_float("track_persistence"),
            "surprise_std": _diag_float("surprise_std"),
            "surprise_peak_rate": _diag_float("surprise_peak_rate"),
            "surprise_detection_overlap": _diag_float("surprise_detection_overlap"),
            "map_points_per_pose": _diag_float("map_points_per_pose"),
            "map_pose_coverage": _diag_float("map_pose_coverage"),
            "adaptation_efficiency": _diag_float("adaptation_efficiency"),
            "artifact_density_per_frame": _diag_float("artifact_density_per_frame"),
            "artifact_mb_per_min": _diag_float("artifact_mb_per_min"),
        },
        "run_health": {
            "florence_caption_coverage": float(summary.run_health.florence_caption_coverage),
            "qwen_caption_coverage": float(summary.run_health.qwen_caption_coverage),
            "qwen_parse_error_count": int(summary.run_health.qwen_parse_error_count),
            "asr_coverage": float(summary.run_health.asr_coverage),
            "ocr_coverage": float(summary.run_health.ocr_coverage),
            "world_model_ok": bool(summary.run_health.world_model_ok),
            "tracking_ok": bool(summary.run_health.tracking_ok),
            "tracking_filter_fallback_used": bool(summary.run_health.tracking_filter_fallback_used),
            "florence_runtime_mode": summary.run_health.florence_runtime_mode,
            "restore_failures": int(summary.run_health.restore_failures),
            "vram_wait_time_sec": float(summary.run_health.vram_wait_time_sec),
            "warnings": list(summary.run_health.warnings),
        },
    }

    if summary.detection_stats:
        payload["detection_stats"] = {
            "model": summary.detection_stats.model,
            "total_objects": int(summary.detection_stats.total_objects),
            "mean_per_frame": float(summary.detection_stats.mean_per_frame),
            "max_per_frame": int(summary.detection_stats.max_per_frame),
            "top_classes": sorted(
                summary.detection_stats.by_class.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5],
        }
    if summary.temporal_stats:
        payload["temporal_stats"] = {
            "method": summary.temporal_stats.method,
            "mean_surprise": float(summary.temporal_stats.mean_surprise),
            "peak_frames": list(summary.temporal_stats.peak_frames[:10]),
        }
    if summary.training_stats:
        payload["training_stats"] = {
            "ssl_best_loss": float(summary.training_stats.ssl_best_loss),
            "distill_best_loss": float(summary.training_stats.distill_best_loss),
            "distill_best_r1": float(summary.training_stats.distill_best_r1),
            "distill_compression": float(summary.training_stats.distill_compression),
        }
    if summary.tracking_stats:
        payload["tracking_stats"] = {
            "model": summary.tracking_stats.model,
            "scene_type": summary.tracking_stats.gemma_scene_type,
            "priority": list(summary.tracking_stats.tracking_priority),
            "targets_effective": list(summary.tracking_stats.tracking_targets_effective),
            "unique_track_ids": int(summary.tracking_stats.unique_track_ids),
            "total_detections": int(summary.tracking_stats.total_detections),
            "sam_masks_total": int(summary.tracking_stats.sam_masks_total),
        }
    if summary.embedding_stats:
        payload["embedding_stats"] = {
            "n_embeddings": int(summary.embedding_stats.n_embeddings),
            "embedding_dim": int(summary.embedding_stats.embedding_dim),
            "mean_neighbour_similarity": float(summary.embedding_stats.mean_neighbour_similarity),
        }
    if summary.map_stats:
        payload["map_stats"] = {
            "method": summary.map_stats.method,
            "points": int(summary.map_stats.points),
            "poses": int(summary.map_stats.poses),
            "sfm_poses": int(summary.map_stats.sfm_poses),
            "frame_anchor_count": int(summary.map_stats.frame_anchor_count),
            "degraded": bool(summary.map_stats.degraded),
            "quality_note": summary.map_stats.quality_note,
        }

    return payload


def _emit_local_run_analytics(video_dir: Path) -> dict[str, Any] | None:
    try:
        from selfsuvis.analytics import LocalRunLoader

        summary = LocalRunLoader(video_dir).load()
    except Exception as exc:
        _log.warning("Local analytics skipped for %s (%s)", video_dir.name, exc)
        return None

    payload = _build_local_run_analytics_payload(summary)
    summary_path = video_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _log.info("Analytics summary → %s", summary_path)
    _log.info(
        "  Analytics: %s | %d frames | %.1fs @ %.1f fps | %d artifacts",
        summary.video_name,
        summary.n_frames,
        summary.duration_sec,
        summary.fps,
        summary.artifact_inventory.total_files,
    )
    _log.info(
        "  Coverage: Florence %.0f%% | Qwen %.0f%% | ASR %.0f%% | OCR %.0f%% | world=%s",
        100.0 * summary.run_health.florence_caption_coverage,
        100.0 * summary.run_health.qwen_caption_coverage,
        100.0 * summary.run_health.asr_coverage,
        100.0 * summary.run_health.ocr_coverage,
        "ok" if summary.run_health.world_model_ok else "degraded",
    )
    _log.info(
        "  Diagnostics: quality=%.1f/100 | modality=%.0f%% | track_frag=%.3f | map_pose=%.0f%% | adapt_eff=%.3f",
        summary.diagnostics.quality_score,
        100.0 * summary.diagnostics.modality_completeness,
        summary.diagnostics.tracking_fragmentation,
        100.0 * summary.diagnostics.map_pose_coverage,
        summary.diagnostics.adaptation_efficiency,
    )

    if summary.detection_stats:
        top_classes = (
            ", ".join(
                f"{label}:{count}"
                for label, count in sorted(
                    summary.detection_stats.by_class.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
            )
            or "none"
        )
        _log.info(
            "  Detections: %d total | mean %.1f/frame | max %d/frame | top=%s",
            summary.detection_stats.total_objects,
            summary.detection_stats.mean_per_frame,
            summary.detection_stats.max_per_frame,
            top_classes,
        )
    if summary.temporal_stats:
        _log.info(
            "  Temporal: %s mean_surprise=%.3f | peak_frames=%d",
            summary.temporal_stats.method or "unknown",
            summary.temporal_stats.mean_surprise,
            len(summary.temporal_stats.peak_frames),
        )
    if summary.tracking_stats:
        _log.info(
            "  Tracking: %s | tracks=%d | detections=%d | SAM masks=%d | scene=%s",
            summary.tracking_stats.model or "unknown",
            summary.tracking_stats.unique_track_ids,
            summary.tracking_stats.total_detections,
            summary.tracking_stats.sam_masks_total,
            summary.tracking_stats.gemma_scene_type or "unknown",
        )
    if summary.embedding_stats:
        _log.info(
            "  Embeddings: %d x %d | mean NN sim=%.3f",
            summary.embedding_stats.n_embeddings,
            summary.embedding_stats.embedding_dim,
            summary.embedding_stats.mean_neighbour_similarity,
        )
    if summary.map_stats:
        _log.info(
            "  3D map: %s | points=%d | poses=%d | sfm_poses=%d | quality=%s",
            summary.map_stats.method or "unknown",
            summary.map_stats.points,
            summary.map_stats.poses,
            summary.map_stats.sfm_poses,
            "degraded" if summary.map_stats.degraded else "ok",
        )
    if summary.training_stats and summary.training_stats.ssl_epochs:
        _log.info(
            "  Training: SSL best=%.4f | distill R@1=%.3f | compression=%.1fx",
            summary.training_stats.ssl_best_loss,
            summary.training_stats.distill_best_r1,
            summary.training_stats.distill_compression,
        )
    if summary.run_health.warnings:
        _log.warning("  Analytics warnings: %s", ", ".join(summary.run_health.warnings))

    return payload
