"""Embedding visualizations: PCA scatter and cosine-similarity heatmap."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from selfsuvis.analytics.embeddings import (
    cosine_similarity_matrix,
    load_gallery,
    pca_project,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_embedding_pca(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    show: bool = False,
) -> "Figure":
    """2-D PCA scatter of frame embeddings from edge_models/gallery.npz.

    Points are coloured by temporal index (early=blue, late=red).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embeddings = load_gallery(run_dir)
    if embeddings is None or len(embeddings) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "gallery.npz not found", ha="center", va="center")
        return fig

    proj = pca_project(embeddings, n_components=2)
    colors = np.linspace(0, 1, len(proj))

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="coolwarm",
                    s=40, alpha=0.85, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Temporal index (early → late)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Frame Embedding PCA  (n={len(proj)}, dim={embeddings.shape[1]})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_similarity_matrix(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    show: bool = False,
) -> "Figure":
    """Cosine-similarity heatmap of all frame-pair embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embeddings = load_gallery(run_dir)
    if embeddings is None or len(embeddings) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "gallery.npz not found", ha="center", va="center")
        return fig

    sim = cosine_similarity_matrix(embeddings)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, vmin=0.5, vmax=1.0, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    ax.set_title(f"Frame Similarity Matrix  (n={len(embeddings)})")

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
