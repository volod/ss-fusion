"""Analytics subpackage — parse and summarise local-run artifacts.

Typical usage::

    from selfsuvis.analytics import LocalRunLoader, RunSummary

    loader = LocalRunLoader("data/local_runs/drone_mission")
    summary = loader.load()
    print(summary.detection_stats)
"""

from .loader import LocalRunLoader
from .models import (
    AnalyticsDiagnostics,
    ArtifactInventory,
    ArtifactRecord,
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

__all__ = [
    "LocalRunLoader",
    "ArtifactInventory",
    "ArtifactRecord",
    "AnalyticsDiagnostics",
    "FrameRecord",
    "DetectionStats",
    "EmbeddingStats",
    "MapStats",
    "TemporalStats",
    "TrackingStats",
    "TrainingStats",
    "RunHealth",
    "RunSummary",
]
