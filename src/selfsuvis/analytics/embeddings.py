"""Embedding-space analysis from edge_models/gallery.npz."""


from pathlib import Path

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)


def load_gallery(run_dir: str | Path) -> np.ndarray | None:
    """Return (N, dim) float32 array from gallery.npz, or None if unavailable."""
    path = Path(run_dir) / "edge_models" / "gallery.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        key = "embeddings" if "embeddings" in data else list(data.keys())[0]
        return data[key].astype(np.float32)
    except Exception as exc:
        logger.warning("Failed to load gallery: %s", exc)
        return None


def pca_project(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Return (N, n_components) PCA projection."""
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine similarity matrix for normalised embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-8, None)
    return (normed @ normed.T).astype(np.float32)


def nearest_neighbour_recall(
    embeddings: np.ndarray, k: int = 5
) -> tuple[float, np.ndarray]:
    """Mean cosine similarity to top-k neighbours (excluding self).

    Returns (mean_mnn_score, per_frame_scores).
    """
    sim = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(sim, -1.0)
    top_k = np.sort(sim, axis=1)[:, -k:]
    per_frame = top_k.mean(axis=1)
    return float(per_frame.mean()), per_frame
