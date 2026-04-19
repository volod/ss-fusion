"""Detection charts: class distribution and per-frame heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from selfsuvis.analytics.models import RunSummary


def plot_detections(
    summary: RunSummary,
    out_path: Optional[str | Path] = None,
    show: bool = False,
) -> "matplotlib.figure.Figure":
    """Two-panel figure: pie chart of object classes + bar chart by frame.

    Args:
        summary: RunSummary from LocalRunLoader.load().
        out_path: If set, saves the figure to this path.
        show: If True, calls plt.show().

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = summary.detection_stats
    if ds is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No detection data", ha="center", va="center")
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Object Detections — {summary.video_name} ({ds.model})")

    # Pie chart
    labels = [k for k, v in ds.by_class.items() if v > 0]
    values = [ds.by_class[k] for k in labels]
    if values:
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
        ax1.pie(values, labels=labels, autopct="%1.0f%%",
                colors=colors[:len(labels)], startangle=140)
    ax1.set_title(f"Class distribution (total={ds.total_objects})")

    # Per-frame bar chart
    frames = summary.frames
    ts = [f.t_sec for f in frames]
    counts = ds.per_frame_counts
    width = (ts[1] - ts[0]) * 0.8 if len(ts) > 1 else 0.1
    ax2.bar(ts[:len(counts)], counts[:len(ts)], width=width,
            color="#3498db", alpha=0.85)
    ax2.axhline(ds.mean_per_frame, color="#e74c3c", linestyle="--",
                linewidth=1.2, label=f"Mean={ds.mean_per_frame:.1f}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Objects per frame")
    ax2.set_title("Detections over time")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
