"""Training curves: SSL fine-tuning loss and distillation metrics."""


from pathlib import Path
from typing import Optional

from selfsuvis.analytics.models import RunSummary


def plot_training_curves(
    summary: RunSummary,
    out_path: Optional[str | Path] = None,
    show: bool = False,
) -> "matplotlib.figure.Figure":
    """Bar-per-epoch SSL loss curve + distillation summary table.

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

    ts = summary.training_stats
    if ts is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No training data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Metrics — {summary.video_name}")

    # SSL loss per epoch
    ax1 = axes[0]
    if ts.ssl_losses:
        epochs = list(range(1, len(ts.ssl_losses) + 1))
        ax1.plot(epochs, ts.ssl_losses, "o-", color="#e67e22", linewidth=2, markersize=6)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("SSL contrastive loss")
        ax1.set_title(f"DINOv3 SSL fine-tuning  (best={ts.ssl_best_loss:.4f})")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No SSL loss data", ha="center", va="center")

    # Distillation summary
    ax2 = axes[1]
    ax2.axis("off")
    rows = [
        ["Metric", "Value"],
        ["Distill best loss", f"{ts.distill_best_loss:.4f}"],
        ["Distill R@1", f"{ts.distill_best_r1:.3f}"],
        ["Compression", f"{ts.distill_compression:.1f}×"],
        ["Teacher (MB)", f"{ts.teacher_mb:.1f}"],
        ["Student (MB)", f"{ts.student_mb:.1f}"],
        ["ONNX (MB)", f"{ts.onnx_mb:.1f}"],
    ]
    table = ax2.table(cellText=rows[1:], colLabels=rows[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title("Distillation summary")

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
