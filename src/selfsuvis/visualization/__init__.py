"""Visualization subpackage — produce charts and HTML reports from RunSummary.

Typical usage::

    from selfsuvis.analytics import LocalRunLoader
    from selfsuvis.visualization import plot_timeline, plot_detections, generate_report

    summary = LocalRunLoader(".data/local_runs/drone_mission").load()
    plot_timeline(summary, out_path="timeline.png")
    plot_detections(summary, out_path="detections.png")
    generate_report(summary, out_dir=".data/local_runs/drone_mission")
"""

from .detections import plot_detections
from .embeddings import plot_embedding_pca, plot_similarity_matrix
from .report import generate_report
from .timeline import plot_timeline
from .training import plot_training_curves

__all__ = [
    "plot_timeline",
    "plot_detections",
    "plot_embedding_pca",
    "plot_similarity_matrix",
    "plot_training_curves",
    "generate_report",
]
