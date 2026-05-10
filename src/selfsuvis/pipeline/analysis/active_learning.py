"""Active learning scoring and frame tagging.

Computes per-frame uncertainty scores and assigns al_tags for the annotation queue.

Score formula (without RSSM):
    al_score = 0.6 * dino_dist + 0.4 * (1 - caption_confidence)

Score formula (with RSSM temporal surprise, DREAMER_ENABLED=true):
    al_score = 0.35 * dino_dist + 0.25 * (1 - caption_confidence) + 0.40 * rssm_surprise

The RSSM surprise signal is derived from the DreamerV3 world model architecture
(Romero et al., ICRA 2026, "Dream to Fly"). It measures how unexpected a frame
was given the RSSM's prediction from prior context — capturing temporal novelty
that pure per-frame distance metrics miss.

Tags:
    needs_annotation — top-K uncertain frames per mission (high combined score)
    novel            — frames with very high dino_dist beyond top-K (out-of-distribution)
    none             — all other frames

Clustering:
    fit_kmeans(embeddings, n_clusters) automatically selects KMeans vs MiniBatchKMeans
    based on settings.KMEANS_BATCH_THRESHOLD. Switch threshold is tunable via the
    KMEANS_BATCH_THRESHOLD env var (default 25_000 frames).
"""

from typing import Any

import numpy as np

from selfsuvis.pipeline.core.config import settings

_DINO_WEIGHT = 0.6
_CAPTION_WEIGHT = 0.4
# Weights when RSSM surprise signal is available (must sum to 1.0)
_DINO_WEIGHT_RSSM = 0.35
_CAPTION_WEIGHT_RSSM = 0.25
_RSSM_WEIGHT = 0.40
_DEFAULT_NOVEL_THRESHOLD = 0.7  # dino_dist above this (outside top-K) → novel


def compute_al_score(
    dino_dist: float,
    caption_confidence: float,
    rssm_surprise: float | None = None,
) -> float:
    """Compute active learning score for a single frame.

    Higher score means more uncertain / informative for annotation.
    All inputs are expected in [0, 1].

    When *rssm_surprise* is provided, the score uses the three-signal formula
    that weights temporal surprise (DreamerV3 RSSM) more heavily:
        score = 0.35 * dino_dist + 0.25 * (1 - caption_confidence) + 0.40 * rssm_surprise

    Without *rssm_surprise*, falls back to the original two-signal formula:
        score = 0.6 * dino_dist + 0.4 * (1 - caption_confidence)
    """
    if rssm_surprise is not None:
        return (
            _DINO_WEIGHT_RSSM * float(dino_dist)
            + _CAPTION_WEIGHT_RSSM * (1.0 - float(caption_confidence))
            + _RSSM_WEIGHT * float(rssm_surprise)
        )
    return _DINO_WEIGHT * float(dino_dist) + _CAPTION_WEIGHT * (1.0 - float(caption_confidence))


def assign_al_tags(
    dino_dists: list[float],
    caption_confidences: list[float],
    top_k: int | None = None,
    novel_threshold: float = _DEFAULT_NOVEL_THRESHOLD,
    rssm_surprises: list[float] | None = None,
) -> tuple[list[float], list[str]]:
    """Compute al_scores and assign al_tags for a batch of frames from one mission.

    Args:
        dino_dists: DINOv3 nearest-neighbour distances for each frame (0 = seen before, 1 = novel).
        caption_confidences: Florence-2 caption confidence per frame (0–1).
        top_k: Number of frames tagged 'needs_annotation'. Defaults to settings.AL_TAG_K.
        novel_threshold: Minimum dino_dist to tag a frame as 'novel' (beyond top-K).
        rssm_surprises: Optional per-frame RSSM temporal surprise scores (0–1).
            When provided, uses the three-signal formula (DreamerV3-enhanced AL).
            Must be the same length as dino_dists, or None.

    Returns:
        (al_scores, al_tags) — same length as inputs.
    """
    if top_k is None:
        top_k = settings.AL_TAG_K

    n = len(dino_dists)
    if n == 0:
        return [], []

    if rssm_surprises is not None and len(rssm_surprises) == n:
        scores = [
            compute_al_score(d, c, s)
            for d, c, s in zip(dino_dists, caption_confidences, rssm_surprises)
        ]
    else:
        scores = [compute_al_score(d, c) for d, c in zip(dino_dists, caption_confidences)]
    tags = ["none"] * n

    # Sort indices by score descending; top-K → needs_annotation
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(order):
        if rank < top_k:
            tags[idx] = "needs_annotation"
        elif float(dino_dists[idx]) >= novel_threshold:
            # High DINOv3 distance outside top-K → genuinely out-of-distribution
            tags[idx] = "novel"

    return scores, tags


def fit_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 20,
    batch_threshold: int | None = None,
    random_state: int = 42,
) -> Any:
    """Fit k-means clustering on embeddings, auto-selecting KMeans vs MiniBatchKMeans.

    When the number of embeddings exceeds `batch_threshold`, uses
    `sklearn.cluster.MiniBatchKMeans` to avoid the O(n²) memory cost of full KMeans.
    Below the threshold, uses standard `KMeans` for higher-quality centroids.

    Args:
        embeddings: 2D float array of shape (n_frames, embed_dim). Assumed L2-normalised.
        n_clusters: Number of k-means clusters (default 20).
        batch_threshold: Switch threshold. Defaults to settings.KMEANS_BATCH_THRESHOLD (25_000).
        random_state: Random seed for reproducibility.

    Returns:
        A fitted sklearn KMeans or MiniBatchKMeans object. Centroids are at `.cluster_centers_`.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans  # type: ignore

    if batch_threshold is None:
        batch_threshold = settings.KMEANS_BATCH_THRESHOLD

    n_samples = embeddings.shape[0]
    # Can't have more clusters than samples
    k = min(n_clusters, n_samples)

    if n_samples >= batch_threshold:
        # MiniBatchKMeans: O(batch_size × k) per step — scales to millions of embeddings
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=min(1024, n_samples),
            n_init=3,
        )
    else:
        # Full KMeans: better centroid quality for small-to-medium datasets
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

    model.fit(embeddings)
    return model


def dino_distances_from_centroids(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute each embedding's distance to its nearest centroid.

    Uses cosine distance (1 − cosine_similarity) for L2-normalised embeddings;
    falls back to Euclidean distance if embeddings are not normalised.

    Args:
        embeddings: (n_frames, dim) float array.
        centroids:  (n_clusters, dim) float array (from model.cluster_centers_).

    Returns:
        (n_frames,) float array of distances in [0, 1].
    """
    # Dot-product similarity matrix: (n_frames, n_clusters)
    norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_c = np.linalg.norm(centroids, axis=1, keepdims=True)

    # Safe normalisation — avoid division by zero
    eps = 1e-9
    e_norm = embeddings / np.maximum(norms_e, eps)
    c_norm = centroids / np.maximum(norms_c, eps)

    similarities = e_norm @ c_norm.T  # (n_frames, n_clusters)
    max_sim = similarities.max(axis=1)  # nearest centroid similarity
    return np.clip(1.0 - max_sim, 0.0, 1.0)
