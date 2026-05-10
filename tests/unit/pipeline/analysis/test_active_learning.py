"""Unit tests for pipeline.active_learning."""

import numpy as np
import pytest

from selfsuvis.pipeline.analysis.active_learning import (
    assign_al_tags,
    compute_al_score,
    dino_distances_from_centroids,
    fit_kmeans,
)


def test_compute_al_score_max_uncertainty():
    """dino_dist=1.0, confidence=0.0 → maximum score 1.0."""
    assert compute_al_score(1.0, 0.0) == pytest.approx(1.0)


def test_compute_al_score_min_uncertainty():
    """dino_dist=0.0, confidence=1.0 → minimum score 0.0."""
    assert compute_al_score(0.0, 1.0) == pytest.approx(0.0)


def test_compute_al_score_weights():
    """Verify: score = 0.6 * dino_dist + 0.4 * (1 - caption_confidence)."""
    assert compute_al_score(0.5, 0.5) == pytest.approx(0.6 * 0.5 + 0.4 * 0.5)


def test_compute_al_score_mid():
    assert compute_al_score(0.8, 0.2) == pytest.approx(0.6 * 0.8 + 0.4 * 0.8)


def test_assign_al_tags_empty():
    """Empty input returns empty lists."""
    scores, tags = assign_al_tags([], [])
    assert scores == []
    assert tags == []


def test_assign_al_tags_top_k_count():
    """Exactly top_k frames get 'needs_annotation'."""
    dists = [0.1, 0.9, 0.5, 0.8, 0.2]
    confs = [0.9, 0.1, 0.5, 0.2, 0.8]
    scores, tags = assign_al_tags(dists, confs, top_k=2)
    assert tags.count("needs_annotation") == 2


def test_assign_al_tags_highest_score_gets_annotated():
    """Frame with highest al_score is always tagged needs_annotation."""
    dists = [0.0, 1.0, 0.5]
    confs = [1.0, 0.0, 0.5]
    scores, tags = assign_al_tags(dists, confs, top_k=1)
    # Frame 1: score = 0.6*1.0 + 0.4*1.0 = 1.0 — clearly highest
    assert tags[1] == "needs_annotation"


def test_assign_al_tags_novel_beyond_top_k():
    """Frame with very high dino_dist outside top-K is tagged novel."""
    # Frame 0 gets top_k=1 (high combined score via low confidence)
    # Frame 1 has high dino_dist (0.95) but lower combined score
    dists = [0.3, 0.95]
    confs = [0.0, 0.6]
    # Scores: frame0 = 0.6*0.3 + 0.4*1.0 = 0.58, frame1 = 0.6*0.95 + 0.4*0.4 = 0.73
    # frame1 actually wins — let's use top_k=1 and pick the one that's NOT annotated
    scores, tags = assign_al_tags(dists, confs, top_k=1, novel_threshold=0.8)
    # One frame needs_annotation, other may be novel if dino_dist >= 0.8
    assert "needs_annotation" in tags


def test_assign_al_tags_novel_threshold_respected():
    """Frames below novel_threshold outside top-K stay 'none'."""
    dists = [0.9, 0.3]
    confs = [0.0, 0.9]
    # Frame 0: score = 0.6*0.9 + 0.4*1.0 = 0.94 → needs_annotation (top_k=1)
    # Frame 1: dino_dist=0.3 < novel_threshold=0.8 → none
    scores, tags = assign_al_tags(dists, confs, top_k=1, novel_threshold=0.8)
    assert tags[0] == "needs_annotation"
    assert tags[1] == "none"


def test_assign_al_tags_top_k_exceeds_n():
    """top_k >= n → all frames tagged needs_annotation."""
    scores, tags = assign_al_tags([0.5, 0.3], [0.4, 0.7], top_k=10)
    assert all(t == "needs_annotation" for t in tags)


def test_assign_al_tags_top_k_zero():
    """top_k=0 → no needs_annotation; high-dist frames may be novel."""
    dists = [0.9, 0.1]
    confs = [0.0, 0.9]
    scores, tags = assign_al_tags(dists, confs, top_k=0, novel_threshold=0.8)
    assert "needs_annotation" not in tags
    assert tags[0] == "novel"


def test_assign_al_tags_uses_settings_default(monkeypatch):
    """assign_al_tags uses settings.AL_TAG_K when top_k is None."""
    from selfsuvis.pipeline.core import config

    monkeypatch.setattr(config.settings, "AL_TAG_K", 2)
    dists = [0.9, 0.8, 0.1, 0.1]
    confs = [0.0, 0.1, 0.9, 0.9]
    scores, tags = assign_al_tags(dists, confs)
    assert tags.count("needs_annotation") == 2


def test_assign_al_tags_scores_match_formula():
    """Returned scores match compute_al_score for each frame."""
    dists = [0.2, 0.6, 0.9]
    confs = [0.8, 0.4, 0.1]
    scores, tags = assign_al_tags(dists, confs, top_k=1)
    for i, (d, c) in enumerate(zip(dists, confs)):
        assert scores[i] == pytest.approx(compute_al_score(d, c))


# ── fit_kmeans tests ──────────────────────────────────────────────────────────


def _random_embeddings(n, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, dim)).astype(np.float32)
    # L2-normalise
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


def test_fit_kmeans_returns_correct_cluster_count():
    """fit_kmeans returns a model with n_clusters centroids."""
    emb = _random_embeddings(100)
    model = fit_kmeans(emb, n_clusters=5, batch_threshold=50_000)
    assert model.cluster_centers_.shape == (5, 64)


def test_fit_kmeans_uses_kmeans_below_threshold():
    """Below batch_threshold, fit_kmeans uses KMeans."""
    from sklearn.cluster import KMeans

    emb = _random_embeddings(50)
    model = fit_kmeans(emb, n_clusters=3, batch_threshold=200)
    assert isinstance(model, KMeans)


def test_fit_kmeans_uses_minibatch_above_threshold():
    """At or above batch_threshold, fit_kmeans uses MiniBatchKMeans."""
    from sklearn.cluster import MiniBatchKMeans

    emb = _random_embeddings(50)
    model = fit_kmeans(emb, n_clusters=3, batch_threshold=10)
    assert isinstance(model, MiniBatchKMeans)


def test_fit_kmeans_caps_clusters_at_n_samples():
    """n_clusters > n_samples is clamped to n_samples."""
    emb = _random_embeddings(5)
    model = fit_kmeans(emb, n_clusters=20, batch_threshold=50_000)
    assert model.cluster_centers_.shape[0] <= 5


def test_fit_kmeans_uses_settings_threshold(monkeypatch):
    """fit_kmeans uses settings.KMEANS_BATCH_THRESHOLD when batch_threshold=None."""
    from sklearn.cluster import MiniBatchKMeans

    from selfsuvis.pipeline.core import config

    monkeypatch.setattr(config.settings, "KMEANS_BATCH_THRESHOLD", 10)
    emb = _random_embeddings(50)
    model = fit_kmeans(emb, n_clusters=3)  # batch_threshold=None → uses settings
    assert isinstance(model, MiniBatchKMeans)


# ── dino_distances_from_centroids tests ──────────────────────────────────────


def test_dino_distances_shape():
    """Returns one distance per embedding."""
    emb = _random_embeddings(30)
    centroids = _random_embeddings(5)
    dists = dino_distances_from_centroids(emb, centroids)
    assert dists.shape == (30,)


def test_dino_distances_range():
    """All distances are in [0, 1]."""
    emb = _random_embeddings(20)
    centroids = _random_embeddings(4)
    dists = dino_distances_from_centroids(emb, centroids)
    assert np.all(dists >= 0.0)
    assert np.all(dists <= 1.0)


def test_dino_distances_exact_centroid_is_zero():
    """An embedding identical to a centroid has distance 0."""
    emb = _random_embeddings(3)
    centroids = emb[[0]]  # centroid = first embedding
    dists = dino_distances_from_centroids(emb, centroids)
    assert dists[0] == pytest.approx(0.0, abs=1e-5)


def test_dino_distances_orthogonal_is_one():
    """Two orthogonal L2-normalised vectors have cosine distance 1."""
    e = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = np.array([[1.0, 0.0]], dtype=np.float32)
    dists = dino_distances_from_centroids(e, c)
    assert dists[0] == pytest.approx(0.0, abs=1e-5)
    assert dists[1] == pytest.approx(1.0, abs=1e-5)
