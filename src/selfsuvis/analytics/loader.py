"""Load and aggregate all artifacts produced by a local pipeline run."""


import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

from .embeddings import load_gallery, nearest_neighbour_recall
from .models import (
    ArtifactInventory,
    ArtifactRecord,
    AnalyticsDiagnostics,
    DetectionStats,
    EmbeddingStats,
    FrameRecord,
    MapStats,
    RunHealth,
    RunSummary,
    TemporalStats,
    TrackingStats,
    TrainingStats,
)

logger = logging.getLogger(__name__)


class LocalRunLoader:
    """Parse every artifact in a local-run output directory and return a RunSummary."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def load(self) -> RunSummary:
        frames_meta = self._load_json("frames_metadata.json")
        yolo_data = self._load_json("yolo_sam_results.json")
        rssm_data = self._load_json("rssm_temporal.json")
        ontology_data = self._load_json("video_ontology.json")
        tracking_data = self._load_json("gemma_tracking_results.json")
        map_data = self._load_json("3d_map/map_stats.json")
        runtime_metrics = self._load_json("runtime_metrics.json")

        scene_captions = self._parse_scene_captions()
        qwen_captions = self._parse_qwen_captions()
        asr_map = self._parse_asr_segments(frames_meta)

        frames = self._build_frames(
            meta=frames_meta,
            yolo=yolo_data,
            rssm=rssm_data,
            scene_captions=scene_captions,
            qwen_captions=qwen_captions,
            asr_map=asr_map,
            tracking=tracking_data,
        )
        detection_stats = self._build_detection_stats(yolo_data)
        temporal_stats = self._build_temporal_stats(rssm_data)
        training_stats = self._build_training_stats()
        tracking_stats = self._build_tracking_stats(tracking_data)
        embedding_stats = self._build_embedding_stats()
        map_stats = self._build_map_stats(map_data)
        artifact_inventory = self._build_artifact_inventory()

        video_name = frames_meta.get("video_id", self.run_dir.name) if frames_meta else self.run_dir.name
        n_frames = frames_meta.get("frame_count", len(frames)) if frames_meta else len(frames)
        duration = frames_meta.get("duration_sec", 0.0) if frames_meta else 0.0
        fps = frames_meta.get("fps", 0.0) if frames_meta else 0.0

        run_health = self._build_run_health(frames, n_frames, tracking_stats, map_stats, runtime_metrics)
        diagnostics = self._build_diagnostics(
            frames=frames,
            n_frames=n_frames,
            duration_sec=duration,
            detection_stats=detection_stats,
            temporal_stats=temporal_stats,
            training_stats=training_stats,
            tracking_stats=tracking_stats,
            map_stats=map_stats,
            artifact_inventory=artifact_inventory,
            run_health=run_health,
            has_edge_model=(self.run_dir / "edge_models" / "dino_local.onnx").exists(),
        )

        ontology_data = ontology_data or {}
        return RunSummary(
            run_dir=str(self.run_dir),
            video_name=video_name,
            n_frames=n_frames,
            duration_sec=duration,
            fps=fps,
            frames=frames,
            detection_stats=detection_stats,
            temporal_stats=temporal_stats,
            training_stats=training_stats,
            tracking_stats=tracking_stats,
            embedding_stats=embedding_stats,
            map_stats=map_stats,
            artifact_inventory=artifact_inventory,
            run_health=run_health,
            diagnostics=diagnostics,
            domain=ontology_data.get("domain", "") or "",
            top_category=self._parse_top_category(),
            scene_complexity=ontology_data.get("scene_complexity", "") or "",
            n_scene_clusters=self._parse_scene_clusters(),
            has_3d_map=(self.run_dir / "3d_map" / "gaussian_splat.ply").exists(),
            has_edge_model=(self.run_dir / "edge_models" / "dino_local.onnx").exists(),
        )

    def _build_diagnostics(
        self,
        *,
        frames: List[FrameRecord],
        n_frames: int,
        duration_sec: float,
        detection_stats: Optional[DetectionStats],
        temporal_stats: Optional[TemporalStats],
        training_stats: Optional[TrainingStats],
        tracking_stats: Optional[TrackingStats],
        map_stats: Optional[MapStats],
        artifact_inventory: ArtifactInventory,
        run_health: RunHealth,
        has_edge_model: bool,
    ) -> AnalyticsDiagnostics:
        denom = max(n_frames, 1)
        qwen_coverage_score = _targeted_coverage_score(
            run_health.qwen_caption_coverage,
            n_frames=n_frames,
            target_frames=20,
        )
        coverage_terms = [
            run_health.florence_caption_coverage,
            qwen_coverage_score,
            run_health.asr_coverage,
            run_health.ocr_coverage,
            1.0 if run_health.world_model_ok else 0.0,
            1.0 if run_health.tracking_ok else 0.0,
            0.0 if (map_stats and map_stats.degraded) else (1.0 if map_stats else 0.0),
            1.0 if has_edge_model else 0.0,
        ]
        modality_completeness = _clamp01(sum(coverage_terms) / len(coverage_terms))

        detection_density = float(detection_stats.mean_per_frame) if detection_stats else 0.0
        detection_cv = _coefficient_of_variation(detection_stats.per_frame_counts) if detection_stats else 0.0
        detection_entropy = (
            _normalised_entropy(detection_stats.by_class)
            if detection_stats and detection_stats.by_class else 0.0
        )

        surprise_scores = temporal_stats.surprise_scores if temporal_stats else []
        surprise_std = _stddev(surprise_scores)
        surprise_peak_rate = (
            len(temporal_stats.peak_frames) / max(temporal_stats.n_frames, 1)
            if temporal_stats else 0.0
        )
        surprise_detection_overlap = 0.0
        if temporal_stats and temporal_stats.peak_frames:
            hits = 0
            total = 0
            for idx in temporal_stats.peak_frames:
                if 0 <= idx < len(frames):
                    total += 1
                    if frames[idx].n_detections > 0 or frames[idx].tracking_detections > 0:
                        hits += 1
            surprise_detection_overlap = hits / max(total, 1)

        tracking_fragmentation = 0.0
        track_persistence = 0.0
        if tracking_stats and tracking_stats.total_detections > 0:
            tracking_fragmentation = tracking_stats.unique_track_ids / max(tracking_stats.total_detections, 1)
            track_persistence = _clamp01(tracking_stats.mean_track_length_frames / denom)

        map_points_per_pose = 0.0
        map_pose_coverage = 0.0
        if map_stats:
            if map_stats.sfm_poses > 0:
                map_points_per_pose = map_stats.points / max(map_stats.sfm_poses, 1)
            map_pose_coverage = _clamp01(map_stats.sfm_poses / denom)

        adaptation_efficiency = 0.0
        if training_stats and training_stats.distill_best_r1 > 0:
            # Retained retrieval quality per compression factor; higher is better.
            adaptation_efficiency = training_stats.distill_best_r1 / max(training_stats.distill_compression, 1.0)

        artifact_density = artifact_inventory.total_files / denom
        artifact_mb_per_min = (
            (artifact_inventory.total_bytes / (1024 * 1024)) / max(duration_sec / 60.0, 1e-6)
        )

        map_quality = 0.0
        if map_stats:
            point_score = _clamp01(map_stats.points / 200.0)
            # Only reconstructed SfM poses count as true 3D map pose coverage.
            # PCA fallback anchors are still useful for visualisation, but they
            # should not inflate mapping quality in analytics summaries.
            pose_score = _clamp01(map_stats.sfm_poses / max(denom, 1))
            anchor_score = 0.0
            if map_stats.sfm_poses == 0 and map_stats.frame_anchor_count > 0:
                anchor_score = 0.25 * _clamp01(map_stats.frame_anchor_count / max(denom, 1))
            map_quality = 0.6 * point_score + 0.3 * pose_score + 0.1 * anchor_score
        tracking_quality = 0.0
        if tracking_stats and tracking_stats.total_detections > 0:
            tracking_quality = _clamp01(0.7 * track_persistence + 0.3 * (1.0 - tracking_fragmentation))
        training_quality = 1.0 if has_edge_model else 0.0
        if training_stats and training_stats.distill_best_r1 > 0:
            training_quality = _clamp01(training_stats.distill_best_r1)
        warning_penalty = min(0.25, 0.05 * len(run_health.warnings))
        quality_score = 100.0 * _clamp01(
            0.35 * modality_completeness
            + 0.20 * tracking_quality
            + 0.15 * map_quality
            + 0.15 * training_quality
            + 0.15 * (1.0 if run_health.world_model_ok else 0.0)
            - warning_penalty
        )

        return AnalyticsDiagnostics(
            modality_completeness=modality_completeness,
            quality_score=quality_score,
            detection_density_per_frame=detection_density,
            detection_count_cv=detection_cv,
            detection_entropy_norm=detection_entropy,
            tracking_fragmentation=tracking_fragmentation,
            track_persistence=track_persistence,
            surprise_std=surprise_std,
            surprise_peak_rate=surprise_peak_rate,
            surprise_detection_overlap=surprise_detection_overlap,
            map_points_per_pose=map_points_per_pose,
            map_pose_coverage=map_pose_coverage,
            adaptation_efficiency=adaptation_efficiency,
            artifact_density_per_frame=artifact_density,
            artifact_mb_per_min=artifact_mb_per_min,
        )

    def _load_json(self, filename: str) -> Optional[dict]:
        path = self.run_dir / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to load %s: %s", filename, exc)
            return None

    def _build_frames(
        self,
        *,
        meta: Optional[dict],
        yolo: Optional[dict],
        rssm: Optional[dict],
        scene_captions: Dict[float, Dict[str, object]],
        qwen_captions: Dict[float, str],
        asr_map: Dict[float, str],
        tracking: Optional[dict],
    ) -> List[FrameRecord]:
        if not meta:
            return []

        raw_frames = meta.get("frames", [])
        surprise = (rssm or {}).get("surprise_scores", [])
        yolo_frames = {f.get("t_sec", -1): f for f in (yolo or {}).get("frames", [])}
        tracking_frames = {f.get("t_sec", -1): f for f in (tracking or {}).get("frames", [])}

        records: List[FrameRecord] = []
        for i, frame in enumerate(raw_frames):
            t_sec = float(frame.get("t_sec", i * (1.0 / max(meta.get("fps", 2.0), 0.001))))
            yolo_frame = yolo_frames.get(t_sec, {})
            scene_frame = scene_captions.get(t_sec, {})
            tracking_frame = tracking_frames.get(t_sec, {})
            records.append(
                FrameRecord(
                    idx=i,
                    t_sec=t_sec,
                    path=frame.get("path", ""),
                    n_detections=int(yolo_frame.get("n_detections", 0) or 0),
                    surprise_score=surprise[i] if i < len(surprise) else 0.0,
                    caption=str(scene_frame.get("caption", "")),
                    asr_text=asr_map.get(t_sec, ""),
                    qwen_caption=qwen_captions.get(t_sec, ""),
                    florence_confidence=float(scene_frame.get("confidence", 0.0) or 0.0),
                    tracking_masks=len(tracking_frame.get("sam_masks", [])),
                    tracking_detections=int(tracking_frame.get("n_detections", 0) or 0),
                )
            )
        return records

    def _build_detection_stats(self, yolo: Optional[dict]) -> Optional[DetectionStats]:
        if not yolo:
            return None
        per_frame = [f.get("n_detections", 0) for f in yolo.get("frames", [])]
        return DetectionStats(
            total_objects=int(yolo.get("total_objects", 0) or 0),
            n_frames=int(yolo.get("n_frames", len(per_frame)) or len(per_frame)),
            by_class=dict(yolo.get("by_priority", {}) or {}),
            per_frame_counts=per_frame,
            model=yolo.get("model", ""),
        )

    def _build_temporal_stats(self, rssm: Optional[dict]) -> Optional[TemporalStats]:
        if not rssm:
            return None
        scores = list(rssm.get("surprise_scores", []) or [])
        mean_s = sum(scores) / len(scores) if scores else 0.0
        peaks = []
        if scores:
            threshold = sorted(scores)[max(0, int(len(scores) * 0.9))]
            peaks = [i for i, score in enumerate(scores) if score >= threshold]
        return TemporalStats(
            method=rssm.get("method", ""),
            n_frames=int(rssm.get("n_frames", len(scores)) or len(scores)),
            surprise_scores=scores,
            mean_surprise=mean_s,
            peak_frames=peaks,
        )

    def _build_training_stats(self) -> Optional[TrainingStats]:
        finetune_md = self.run_dir / "finetune_stats.md"
        distill_md = self.run_dir / "distill_stats.md"
        if not finetune_md.exists() and not distill_md.exists():
            return None

        stats = TrainingStats()
        if finetune_md.exists():
            text = finetune_md.read_text()
            match = re.search(r"\|\s*Best loss\s*\|\s*([\d.]+)", text)
            if match:
                stats.ssl_best_loss = float(match.group(1))
            epoch_losses = re.findall(r"^\|\s*\d+\s*\|\s*([\d.]+)", text, re.MULTILINE)
            if epoch_losses:
                stats.ssl_losses = [float(value) for value in epoch_losses]
                stats.ssl_epochs = len(stats.ssl_losses)
            match = re.search(r"\|\s*Checkpoint size\s*\|\s*([\d.]+)\s*MB", text)
            if match:
                stats.teacher_mb = float(match.group(1))

        if distill_md.exists():
            text = distill_md.read_text()
            match = re.search(r"\|\s*Best total loss\s*\|\s*([\d.]+)", text)
            if match:
                stats.distill_best_loss = float(match.group(1))
            match = re.search(r"\|\s*Best Recall@1[^|]*\|\s*([\d.]+)", text)
            if match:
                stats.distill_best_r1 = float(match.group(1))
            match = re.search(r"\|\s*Compression ratio\s*\|\s*([\d.]+)", text)
            if match:
                stats.distill_compression = float(match.group(1))
            combined = text + (finetune_md.read_text() if finetune_md.exists() else "")
            match = re.search(r"ONNX.*?([\d.]+)\s*MB", combined, re.I)
            if match:
                stats.onnx_mb = float(match.group(1))
            match = re.search(r"\|\s*Student.*?\|\s*([\d.]+)\s*MB", text)
            if match:
                stats.student_mb = float(match.group(1))
        return stats

    def _build_tracking_stats(self, tracking: Optional[dict]) -> Optional[TrackingStats]:
        if not tracking:
            return None
        return TrackingStats(
            model=str(tracking.get("model", "") or ""),
            gemma_scene_type=str(tracking.get("gemma_scene_type", "") or ""),
            tracking_priority=list(tracking.get("tracking_priority", []) or []),
            tracking_targets_effective=list(tracking.get("tracking_targets_effective", []) or []),
            filter_retry_mode=str(tracking.get("tracking_filter_retry_mode", "none") or "none"),
            total_detections=int(tracking.get("total_detections", 0) or 0),
            unique_track_ids=int(tracking.get("n_unique_track_ids", 0) or 0),
            sam_masks_total=int(tracking.get("sam_masks_total", 0) or 0),
            mean_track_length_frames=float(tracking.get("mean_track_length_frames", 0.0) or 0.0),
            median_track_length_frames=float(tracking.get("median_track_length_frames", 0.0) or 0.0),
            elapsed_sec=float(tracking.get("elapsed_sec", 0.0) or 0.0),
        )

    def _build_embedding_stats(self) -> Optional[EmbeddingStats]:
        gallery = load_gallery(self.run_dir)
        if gallery is None or len(gallery) == 0:
            return None
        mean_nn, _ = nearest_neighbour_recall(gallery)
        return EmbeddingStats(
            n_embeddings=int(gallery.shape[0]),
            embedding_dim=int(gallery.shape[1]) if gallery.ndim == 2 else 0,
            mean_neighbour_similarity=float(mean_nn),
        )

    def _build_map_stats(self, map_data: Optional[dict]) -> Optional[MapStats]:
        if not map_data:
            return None
        points = int(
            map_data.get("points", map_data.get("point_count", 0)) or 0
        )
        method = str(map_data.get("method", "") or "")
        sfm_poses = int(map_data.get("sfm_poses", map_data.get("poses", 0)) or 0)
        frame_anchor_count = int(map_data.get("frame_anchor_count", sfm_poses) or sfm_poses)
        poses = frame_anchor_count if "pca" in method else sfm_poses
        degraded = bool(map_data.get("quality_degraded", False)) or bool(
            points < 50 or sfm_poses < 20 or str(method).startswith("sfm_sparse+")
        )
        return MapStats(
            method=method,
            points=points,
            poses=poses,
            sfm_poses=sfm_poses,
            frame_anchor_count=frame_anchor_count,
            degraded=degraded,
            quality_note=str(map_data.get("quality_note", "") or ""),
        )

    def _build_artifact_inventory(self) -> ArtifactInventory:
        files: List[ArtifactRecord] = []
        by_suffix: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for path in sorted(self.run_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.run_dir)
            suffix = path.suffix.lower() or "<no_ext>"
            category = rel.parts[0] if len(rel.parts) > 1 else "root"
            files.append(
                ArtifactRecord(
                    path=str(rel),
                    size_bytes=path.stat().st_size,
                    suffix=suffix,
                    category=category,
                )
            )
            by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
        return ArtifactInventory(files=files, by_suffix=by_suffix, by_category=by_category)

    def _build_run_health(
        self,
        frames: List[FrameRecord],
        n_frames: int,
        tracking_stats: Optional[TrackingStats],
        map_stats: Optional[MapStats],
        runtime_metrics: Optional[dict],
    ) -> RunHealth:
        denom = max(n_frames, 1)
        florence_count = sum(1 for frame in frames if frame.caption.strip())
        qwen_count = sum(1 for frame in frames if frame.qwen_caption.strip())
        qwen_parse_error_count = self._count_qwen_parse_errors()
        asr_count = sum(1 for frame in frames if frame.asr_text.strip())

        warnings: List[str] = []
        if florence_count == 0 and n_frames:
            warnings.append("Florence captions are empty for all frames")
        if qwen_parse_error_count and qwen_count == 0:
            warnings.append(
                f"Qwen structured captioning returned parse errors for all sampled frames ({qwen_parse_error_count})"
            )
        elif qwen_parse_error_count:
            warnings.append(
                f"Qwen structured captioning returned {qwen_parse_error_count} parse error(s)"
            )
        if tracking_stats and tracking_stats.total_detections == 0:
            warnings.append("Gemma-directed RF-DETR produced zero tracked detections")
        if tracking_stats and tracking_stats.filter_retry_mode != "none":
            warnings.append(
                f"Gemma-directed RF-DETR required {tracking_stats.filter_retry_mode} label-filter fallback"
            )
        if (
            tracking_stats
            and tracking_stats.unique_track_ids > 0
            and n_frames > 0
            and tracking_stats.unique_track_ids > (2 * n_frames)
            and tracking_stats.mean_track_length_frames < 3.0
        ):
            warnings.append("Gemma-directed tracking appears highly fragmented (too many unique tracks)")
        if self._world_model_failed():
            warnings.append("World model inference failed or produced no usable embeddings")
        if map_stats and map_stats.degraded:
            if map_stats.frame_anchor_count > map_stats.sfm_poses:
                warnings.append(
                    "3D map quality is degraded "
                    f"({map_stats.points} points, {map_stats.sfm_poses} SfM poses, "
                    f"{map_stats.frame_anchor_count} frame anchors)"
                )
            else:
                warnings.append(
                    f"3D map quality is degraded ({map_stats.points} points, {map_stats.sfm_poses} SfM poses)"
                )
        elif map_stats and "pca" in map_stats.method.lower():
            warnings.append(
                f"3D map used PCA fallback geometry ({map_stats.sfm_poses} SfM poses, {map_stats.frame_anchor_count} frame anchors)"
            )
        restore_failures = int((runtime_metrics or {}).get("restore_failures", 0) or 0)
        vram_wait_time = float((runtime_metrics or {}).get("vram_wait_time_sec", 0.0) or 0.0)
        if restore_failures > 0:
            warnings.append(f"Model restore encountered {restore_failures} failure(s)")
        if vram_wait_time >= 30.0:
            warnings.append(f"VRAM waits consumed {vram_wait_time:.1f}s")
        florence_runtime_mode = self._parse_florence_runtime_mode()

        return RunHealth(
            florence_caption_coverage=florence_count / denom,
            qwen_caption_coverage=qwen_count / denom,
            qwen_parse_error_count=qwen_parse_error_count,
            asr_coverage=asr_count / denom,
            ocr_coverage=self._estimate_ocr_coverage(),
            world_model_ok=not self._world_model_failed(),
            tracking_ok=bool(tracking_stats and tracking_stats.total_detections > 0),
            tracking_filter_fallback_used=bool(tracking_stats and tracking_stats.filter_retry_mode != "none"),
            florence_runtime_mode=florence_runtime_mode,
            restore_failures=restore_failures,
            vram_wait_time_sec=vram_wait_time,
            warnings=warnings,
        )

    def _parse_florence_runtime_mode(self) -> str:
        path = self.run_dir / "scene_captions.md"
        if not path.exists():
            return ""
        for line in path.read_text().splitlines():
            if line.startswith("Runtime mode:"):
                return line.split(":", 1)[1].strip()
        return ""

    def _parse_scene_captions(self) -> Dict[float, Dict[str, object]]:
        path = self.run_dir / "scene_captions.md"
        if not path.exists():
            return {}
        mapping: Dict[float, Dict[str, object]] = {}
        pattern = re.compile(
            r"^\|\s*`[^`]+`\s*\|\s*([\d.]+)\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|\s*(.*?)\s*\|$"
        )
        for line in path.read_text().splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            caption = match.group(3).replace("*same scene*", "").strip()
            mapping[float(match.group(1))] = {
                "caption": caption,
                "confidence": float(match.group(2)),
            }
        return mapping

    def _parse_qwen_captions(self) -> Dict[float, str]:
        path = self.run_dir / "detailed_captions.md"
        if not path.exists():
            return {}
        mapping: Dict[float, str] = {}
        pattern = re.compile(
            r"^\|\s*`[^`]+`\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|.*?\|\s*(.*?)\s*\|\s*.*?\|$"
        )
        for line in path.read_text().splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            caption = match.group(2).strip()
            normalized = caption.lower().replace("*", "").strip()
            if (
                "parse_error: true" in normalized
                or normalized.startswith("parse error")
                or normalized.startswith("sidecar unavailable")
                or normalized.startswith("skipped")
            ):
                continue
            mapping[float(match.group(1))] = caption
        return mapping

    def _count_qwen_parse_errors(self) -> int:
        path = self.run_dir / "detailed_captions.md"
        if not path.exists():
            return 0
        text = path.read_text()
        return len(re.findall(r"parse_error:\s*True|\*parse error\*", text, re.IGNORECASE))

    def _parse_asr_segments(self, frames_meta: Optional[dict]) -> Dict[float, str]:
        path = self.run_dir / "asr_subtitles.md"
        if not path.exists() or not frames_meta:
            return {}

        segments: List[tuple[float, float, str]] = []
        pattern = re.compile(r"^\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(.*?)\s*\|$")
        for line in path.read_text().splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            segments.append((float(match.group(1)), float(match.group(2)), match.group(3).strip()))

        frame_map: Dict[float, str] = {}
        for frame in frames_meta.get("frames", []):
            t_sec = float(frame.get("t_sec", 0.0) or 0.0)
            for start, end, text in segments:
                if start <= t_sec <= end:
                    frame_map[t_sec] = text
                    break
        return frame_map

    def _world_model_failed(self) -> bool:
        path = self.run_dir / "multimodal_features.md"
        if not path.exists():
            return False
        text = path.read_text().lower()
        return (
            "world model unavailable" in text
            or "world_model_error" in text
            or "0 clips processed" in text
        )

    def _estimate_ocr_coverage(self) -> float:
        path = self.run_dir / "multimodal_features.md"
        if not path.exists():
            return 0.0
        text = path.read_text()
        match = re.search(r"OCR:\s*(\d+)/(\d+)\s*frames have text", text, re.I)
        if not match:
            match = re.search(
                r"^\|\s*OCR\s*\|\s*[^|]+\|\s*(\d+)\s+frames with text\s*\|$",
                text,
                re.I | re.M,
            )
        if not match:
            return 0.0
        present = int(match.group(1))
        if match.lastindex and match.lastindex >= 2:
            total = int(match.group(2))
        else:
            # Table format only captures the count; derive denominator from
            # frames_metadata.json (authoritative) or text fallback.
            total = 0
            frames_meta_path = self.run_dir / "frames_metadata.json"
            if frames_meta_path.exists():
                try:
                    import json as _json
                    meta = _json.loads(frames_meta_path.read_text())
                    total = int(
                        meta.get("frame_count", 0)
                        or len(meta.get("frames", []))
                        or 0
                    )
                except Exception:
                    pass
            if not total:
                total_match = re.search(r"Total frames\s*:\s*(\d+)", text, re.I)
                total = int(total_match.group(1)) if total_match else 0
        total = max(total, 1)
        return present / total

    def _parse_top_category(self) -> str:
        path = self.run_dir / "gemma_analysis.md"
        if not path.exists():
            return ""
        text = path.read_text()
        match = re.search(r"top category[:\s]+(.+)", text, re.I)
        if match:
            return match.group(1).strip()
        match = re.search(r"^\|\s*([^|]+)\s*\|\s*\d+\s*\|", text, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _parse_scene_clusters(self) -> int:
        path = self.run_dir / "gemma_analysis.md"
        if not path.exists():
            return 0
        text = path.read_text()
        match = re.search(r"(\d+)\s+semantic\s+cluster", text, re.I)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s+cluster", text, re.I)
        return int(match.group(1)) if match else 0


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _targeted_coverage_score(raw_coverage: float, *, n_frames: int, target_frames: int) -> float:
    """Normalise sparse expert-pass coverage against its intended sample budget."""
    expected_coverage = min(1.0, float(target_frames) / max(int(n_frames), 1))
    if expected_coverage <= 0.0:
        return 0.0
    return _clamp01(raw_coverage / expected_coverage)


def _stddev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(float(v) for v in values) / len(values)
    var = sum((float(v) - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _coefficient_of_variation(values: List[int]) -> float:
    if not values:
        return 0.0
    mean = sum(float(v) for v in values) / len(values)
    if abs(mean) < 1e-9:
        return 0.0
    return _stddev([float(v) for v in values]) / mean


def _normalised_entropy(counts: Dict[str, int]) -> float:
    total = sum(max(0, int(v)) for v in counts.values())
    n_classes = sum(1 for v in counts.values() if int(v) > 0)
    if total <= 0 or n_classes <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        c = max(0, int(count))
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log(p)
    return _clamp01(entropy / math.log(n_classes))
