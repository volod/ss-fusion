"""Dataclasses for local-run analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FrameRecord:
    idx: int
    t_sec: float
    path: str
    n_detections: int = 0
    surprise_score: float = 0.0
    caption: str = ""
    asr_text: str = ""
    has_ocr: bool = False
    depth_available: bool = False
    qwen_caption: str = ""
    florence_confidence: float = 0.0
    tracking_masks: int = 0
    tracking_detections: int = 0


@dataclass
class ArtifactRecord:
    path: str
    size_bytes: int
    suffix: str
    category: str


@dataclass
class ArtifactInventory:
    files: List[ArtifactRecord] = field(default_factory=list)
    by_suffix: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


@dataclass
class DetectionStats:
    total_objects: int
    n_frames: int
    by_class: Dict[str, int]
    per_frame_counts: List[int]
    model: str = ""

    @property
    def mean_per_frame(self) -> float:
        if not self.per_frame_counts:
            return 0.0
        return sum(self.per_frame_counts) / len(self.per_frame_counts)

    @property
    def max_per_frame(self) -> int:
        return max(self.per_frame_counts, default=0)


@dataclass
class TemporalStats:
    method: str
    n_frames: int
    surprise_scores: List[float]
    mean_surprise: float
    peak_frames: List[int]          # indices where surprise is in top 10 %

    @property
    def surprise_timeline(self) -> List[float]:
        return self.surprise_scores


@dataclass
class TrainingStats:
    ssl_best_loss: float = 0.0
    ssl_epochs: int = 0
    ssl_losses: List[float] = field(default_factory=list)
    distill_best_loss: float = 0.0
    distill_best_r1: float = 0.0
    distill_compression: float = 0.0
    teacher_mb: float = 0.0
    student_mb: float = 0.0
    onnx_mb: float = 0.0


@dataclass
class TrackingStats:
    model: str = ""
    gemma_scene_type: str = ""
    tracking_priority: List[str] = field(default_factory=list)
    tracking_targets_effective: List[str] = field(default_factory=list)
    filter_retry_mode: str = "none"
    total_detections: int = 0
    unique_track_ids: int = 0
    sam_masks_total: int = 0
    mean_track_length_frames: float = 0.0
    median_track_length_frames: float = 0.0
    elapsed_sec: float = 0.0


@dataclass
class EmbeddingStats:
    n_embeddings: int = 0
    embedding_dim: int = 0
    mean_neighbour_similarity: float = 0.0


@dataclass
class MapStats:
    method: str = ""
    points: int = 0
    poses: int = 0
    degraded: bool = False


@dataclass
class RunHealth:
    florence_caption_coverage: float = 0.0
    qwen_caption_coverage: float = 0.0
    asr_coverage: float = 0.0
    ocr_coverage: float = 0.0
    world_model_ok: bool = False
    tracking_ok: bool = False
    tracking_filter_fallback_used: bool = False
    florence_runtime_mode: str = ""
    restore_failures: int = 0
    vram_wait_time_sec: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class AnalyticsDiagnostics:
    modality_completeness: float = 0.0
    quality_score: float = 0.0
    detection_density_per_frame: float = 0.0
    detection_count_cv: float = 0.0
    detection_entropy_norm: float = 0.0
    tracking_fragmentation: float = 0.0
    track_persistence: float = 0.0
    surprise_std: float = 0.0
    surprise_peak_rate: float = 0.0
    surprise_detection_overlap: float = 0.0
    map_points_per_pose: float = 0.0
    map_pose_coverage: float = 0.0
    adaptation_efficiency: float = 0.0
    artifact_density_per_frame: float = 0.0
    artifact_mb_per_min: float = 0.0


@dataclass
class RunSummary:
    run_dir: str
    video_name: str
    n_frames: int
    duration_sec: float
    fps: float
    frames: List[FrameRecord]
    detection_stats: Optional[DetectionStats]
    temporal_stats: Optional[TemporalStats]
    training_stats: Optional[TrainingStats]
    tracking_stats: Optional[TrackingStats] = None
    embedding_stats: Optional[EmbeddingStats] = None
    map_stats: Optional[MapStats] = None
    artifact_inventory: ArtifactInventory = field(default_factory=ArtifactInventory)
    run_health: RunHealth = field(default_factory=RunHealth)
    diagnostics: AnalyticsDiagnostics = field(default_factory=AnalyticsDiagnostics)
    domain: str = ""
    top_category: str = ""
    scene_complexity: str = ""
    n_scene_clusters: int = 0
    has_3d_map: bool = False
    has_edge_model: bool = False
