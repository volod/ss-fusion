"""Unit tests for pipeline.recent_index."""

from unittest.mock import patch

import numpy as np

from selfsuvis.pipeline.storage.recent_index import RecentEmbeddingIndex

DIM = 4  # small dimension for fast tests


def _unit(values):
    """Create a unit-normalised numpy vector."""
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# --- empty index ---


def test_max_cosine_empty_returns_minus_one():
    """max_cosine on an empty index returns -1.0."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=60.0)
    q = _unit([1.0, 0.0, 0.0, 0.0])
    assert idx.max_cosine(q) == -1.0


# --- basic add + max_cosine ---


def test_max_cosine_identical_vector_returns_one():
    """A query identical to the stored vector returns cosine ~1.0."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=60.0)
    v = _unit([1.0, 2.0, 3.0, 4.0])
    idx.add(v.reshape(1, -1))
    sim = idx.max_cosine(v)
    assert abs(sim - 1.0) < 1e-5


def test_max_cosine_orthogonal_returns_zero():
    """Orthogonal vectors have cosine similarity ~0."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=60.0)
    a = _unit([1.0, 0.0, 0.0, 0.0])
    b = _unit([0.0, 1.0, 0.0, 0.0])
    idx.add(a.reshape(1, -1))
    sim = idx.max_cosine(b)
    assert abs(sim) < 1e-5


def test_max_cosine_opposite_vector_returns_minus_one():
    """Opposite-direction vector returns cosine ~-1.0."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=60.0)
    a = _unit([1.0, 0.0, 0.0, 0.0])
    idx.add(a.reshape(1, -1))
    sim = idx.max_cosine(-a)
    assert abs(sim + 1.0) < 1e-5


def test_max_cosine_returns_max_among_multiple():
    """max_cosine returns the highest similarity among stored vectors."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=60.0)
    far = _unit([0.0, 1.0, 0.0, 0.0])
    near = _unit([1.0, 0.1, 0.0, 0.0])
    idx.add(far.reshape(1, -1))
    idx.add(near.reshape(1, -1))
    q = _unit([1.0, 0.0, 0.0, 0.0])
    sim = idx.max_cosine(q)
    # near is more similar to q than far
    assert sim > 0.9


def test_add_multiple_vectors_at_once():
    """add accepts a 2-D array of multiple vectors."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=60.0)
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    idx.add(vecs)
    assert len(idx.vectors) == 2


# --- TTL pruning ---


def test_ttl_expired_vectors_pruned():
    """Vectors older than ttl_sec are pruned and max_cosine returns -1.0."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=5.0)
    v = _unit([1.0, 0.0, 0.0, 0.0])

    t0 = 1000.0
    with patch("selfsuvis.pipeline.storage.recent_index.time") as mock_time:
        mock_time.time.return_value = t0
        idx.add(v.reshape(1, -1))

    # Advance time past TTL
    with patch("selfsuvis.pipeline.storage.recent_index.time") as mock_time:
        mock_time.time.return_value = t0 + 6.0
        sim = idx.max_cosine(v)

    assert sim == -1.0


def test_ttl_within_window_vectors_kept():
    """Vectors within ttl_sec window are not pruned."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=100, ttl_sec=10.0)
    v = _unit([1.0, 0.0, 0.0, 0.0])

    t0 = 1000.0
    with patch("selfsuvis.pipeline.storage.recent_index.time") as mock_time:
        mock_time.time.return_value = t0
        idx.add(v.reshape(1, -1))

    # Advance time within TTL
    with patch("selfsuvis.pipeline.storage.recent_index.time") as mock_time:
        mock_time.time.return_value = t0 + 5.0
        sim = idx.max_cosine(v)

    assert abs(sim - 1.0) < 1e-5


# --- max_size eviction ---


def test_max_size_evicts_oldest_vectors():
    """When more than max_size vectors are added, oldest are evicted."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=2, ttl_sec=3600.0)
    v1 = _unit([1.0, 0.0, 0.0, 0.0])
    v2 = _unit([0.0, 1.0, 0.0, 0.0])
    v3 = _unit([0.0, 0.0, 1.0, 0.0])
    idx.add(v1.reshape(1, -1))
    idx.add(v2.reshape(1, -1))
    idx.add(v3.reshape(1, -1))
    # Only max_size=2 vectors kept; v1 should be gone
    assert len(idx.vectors) == 2
    # v1 (oldest) evicted; similarity to v1 should be low/zero
    sim = idx.max_cosine(v1)
    assert abs(sim) < 1e-5  # neither v2 nor v3 has overlap with v1


def test_max_size_one_keeps_latest():
    """max_size=1 always keeps only the most recent vector."""
    idx = RecentEmbeddingIndex(dim=DIM, max_size=1, ttl_sec=3600.0)
    old = _unit([1.0, 0.0, 0.0, 0.0])
    new = _unit([0.0, 1.0, 0.0, 0.0])
    idx.add(old.reshape(1, -1))
    idx.add(new.reshape(1, -1))
    assert len(idx.vectors) == 1
    sim = idx.max_cosine(new)
    assert abs(sim - 1.0) < 1e-5
