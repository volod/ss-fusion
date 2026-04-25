"""Timeline chart: per-frame metrics over time."""


from pathlib import Path
from typing import Optional

from selfsuvis.analytics.models import RunSummary


def plot_timeline(
    summary: RunSummary,
    out_path: Optional[str | Path] = None,
    show: bool = False,
) -> "matplotlib.figure.Figure":
    """Two-panel timeline: RSSM surprise and detection count per frame.

    Args:
        summary: RunSummary from LocalRunLoader.load().
        out_path: If set, saves the figure to this path.
        show: If True, calls plt.show() (useful in notebooks).

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = summary.frames
    if not frames:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No frame data", ha="center", va="center")
        return fig

    ts = [f.t_sec for f in frames]
    surprise = [f.surprise_score for f in frames]
    detections = [f.n_detections for f in frames]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Timeline — {summary.video_name}")

    ax1.plot(ts, surprise, color="#e67e22", linewidth=1.5, label="Surprise")
    ax1.axhline(sum(surprise) / max(len(surprise), 1), color="#e67e22", linestyle="--",
                alpha=0.5, linewidth=0.8, label="Mean")
    # Shade peak frames
    if summary.temporal_stats:
        for pi in summary.temporal_stats.peak_frames:
            if pi < len(ts):
                ax1.axvline(ts[pi], color="#c0392b", alpha=0.3, linewidth=0.7)
    ax1.set_ylabel("RSSM Surprise")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.bar(ts, detections, width=max(ts[1] - ts[0], 0.1) if len(ts) > 1 else 0.1,
            color="#2980b9", alpha=0.8, label="Objects")
    ax2.set_ylabel("Detected objects")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
