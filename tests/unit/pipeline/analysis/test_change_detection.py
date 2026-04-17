"""Unit tests for pipeline.change_detection."""
import numpy as np
import pytest

from selfsuvis.pipeline.analysis.change_detection import (
    cosine_distance,
    detect_changes,
    latlon_bbox,
    threshold_for_model,
)


# ── latlon_bbox ───────────────────────────────────────────────────────────────

def test_latlon_bbox_center_inside():
    min_lat, max_lat, min_lon, max_lon = latlon_bbox(47.0, 8.0, 100.0)
    assert min_lat < 47.0 < max_lat
    assert min_lon < 8.0 < max_lon


def test_latlon_bbox_lat_symmetry():
    min_lat, max_lat, min_lon, max_lon = latlon_bbox(47.0, 8.0, 100.0)
    assert (max_lat - 47.0) == pytest.approx(47.0 - min_lat, rel=1e-5)


def test_latlon_bbox_lon_symmetry():
    min_lat, max_lat, min_lon, max_lon = latlon_bbox(47.0, 8.0, 100.0)
    assert (max_lon - 8.0) == pytest.approx(8.0 - min_lon, rel=1e-5)


def test_latlon_bbox_zero_radius():
    """Zero radius collapses to a point."""
    min_lat, max_lat, min_lon, max_lon = latlon_bbox(10.0, 20.0, 0.0)
    assert min_lat == pytest.approx(10.0)
    assert max_lat == pytest.approx(10.0)
    assert min_lon == pytest.approx(20.0)
    assert max_lon == pytest.approx(20.0)


def test_latlon_bbox_larger_radius_wider():
    """Larger radius produces wider bounding box."""
    bbox_50 = latlon_bbox(47.0, 8.0, 50.0)
    bbox_200 = latlon_bbox(47.0, 8.0, 200.0)
    # max_lat - min_lat should be larger for 200 m
    assert (bbox_200[1] - bbox_200[0]) > (bbox_50[1] - bbox_50[0])


# ── cosine_distance ───────────────────────────────────────────────────────────

def test_cosine_distance_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(v, v) == pytest.approx(0.0)


def test_cosine_distance_identical_unnormalized():
    a = np.array([3.0, 4.0])
    assert cosine_distance(a, a) == pytest.approx(0.0)


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_distance(a, b) == pytest.approx(1.0)


def test_cosine_distance_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_distance(a, b) == pytest.approx(2.0)


def test_cosine_distance_zero_vector():
    """Zero vector returns 1.0 (graceful fallback)."""
    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(a, b) == pytest.approx(1.0)
    assert cosine_distance(b, a) == pytest.approx(1.0)


# ── threshold_for_model ───────────────────────────────────────────────────────

def test_threshold_for_model_dino(monkeypatch):
    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "MODEL_NAME", "dinov3")
    monkeypatch.setattr(config.settings, "CHANGE_DETECTION_THRESHOLD_DINO", 0.25)
    assert threshold_for_model() == pytest.approx(0.25)


def test_threshold_for_model_dinov2(monkeypatch):
    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "MODEL_NAME", "dinov2")
    monkeypatch.setattr(config.settings, "CHANGE_DETECTION_THRESHOLD_DINO", 0.25)
    assert threshold_for_model() == pytest.approx(0.25)


def test_threshold_for_model_clip(monkeypatch):
    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "MODEL_NAME", "openclip")
    monkeypatch.setattr(config.settings, "CHANGE_DETECTION_THRESHOLD_CLIP", 0.35)
    assert threshold_for_model() == pytest.approx(0.35)


# ── detect_changes ────────────────────────────────────────────────────────────

def _no_candidates(embedding, bbox):
    return []


def test_detect_changes_empty_frames():
    assert detect_changes([], _no_candidates) == []


def test_detect_changes_frame_without_gps():
    frames = [
        {"frame_id": "f1", "mission_id": "m1", "embedding": [1.0, 0.0], "gps": None}
    ]
    assert detect_changes(frames, _no_candidates) == []


def test_detect_changes_frame_with_null_lat():
    frames = [
        {
            "frame_id": "f1",
            "mission_id": "m1",
            "embedding": [1.0, 0.0],
            "gps": {"lat": None, "lon": 8.0},
        }
    ]
    assert detect_changes(frames, _no_candidates) == []


def test_detect_changes_no_candidates():
    frames = [
        {
            "frame_id": "f1",
            "mission_id": "m1",
            "embedding": [1.0, 0.0],
            "gps": {"lat": 47.0, "lon": 8.0},
        }
    ]
    assert detect_changes(frames, _no_candidates) == []


def test_detect_changes_above_threshold():
    """Orthogonal embeddings (dist=1.0) above threshold → change detected."""
    emb_new = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb_ref = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    frames = [
        {
            "frame_id": "new_f1",
            "mission_id": "m2",
            "embedding": emb_new.tolist(),
            "gps": {"lat": 47.0, "lon": 8.0},
        }
    ]

    def mock_query(embedding, bbox):
        return [{"frame_id": "ref_f1", "mission_id": "m1", "embedding": emb_ref.tolist()}]

    changes = detect_changes(frames, mock_query, threshold=0.3)
    assert len(changes) == 1
    assert changes[0]["frame_id"] == "new_f1"
    assert changes[0]["ref_frame_id"] == "ref_f1"
    assert changes[0]["ref_mission_id"] == "m1"
    assert changes[0]["change_score"] == pytest.approx(1.0, abs=0.01)
    assert changes[0]["threshold"] == pytest.approx(0.3)


def test_detect_changes_below_threshold():
    """Identical embeddings (dist=0) below threshold → no change."""
    emb = np.array([1.0, 0.0], dtype=np.float32)
    frames = [
        {
            "frame_id": "f1",
            "mission_id": "m2",
            "embedding": emb.tolist(),
            "gps": {"lat": 47.0, "lon": 8.0},
        }
    ]

    def mock_query(embedding, bbox):
        return [{"frame_id": "ref_f1", "mission_id": "m1", "embedding": emb.tolist()}]

    assert detect_changes(frames, mock_query, threshold=0.3) == []


def test_detect_changes_skips_same_mission():
    """Candidates from the same mission are skipped."""
    emb_a = np.array([1.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0], dtype=np.float32)
    frames = [
        {
            "frame_id": "f1",
            "mission_id": "m1",
            "embedding": emb_a.tolist(),
            "gps": {"lat": 47.0, "lon": 8.0},
        }
    ]

    # All candidates are same-mission
    def mock_query(embedding, bbox):
        return [{"frame_id": "ref_f1", "mission_id": "m1", "embedding": emb_b.tolist()}]

    assert detect_changes(frames, mock_query, threshold=0.1) == []


def test_detect_changes_picks_closest_candidate():
    """When multiple candidates exist, the one with smallest distance wins."""
    emb_new = np.array([1.0, 0.0], dtype=np.float32)
    # close candidate: dist ≈ 0.03
    emb_close = np.array([0.99, 0.14], dtype=np.float32)
    # far candidate: orthogonal → dist = 1.0
    emb_far = np.array([0.0, 1.0], dtype=np.float32)

    frames = [
        {
            "frame_id": "f1",
            "mission_id": "m2",
            "embedding": emb_new.tolist(),
            "gps": {"lat": 47.0, "lon": 8.0},
        }
    ]

    def mock_query(embedding, bbox):
        return [
            {"frame_id": "far", "mission_id": "m1", "embedding": emb_far.tolist()},
            {"frame_id": "close", "mission_id": "m1", "embedding": emb_close.tolist()},
        ]

    # threshold = 0 so even a close match triggers if best_dist >= 0
    # close dist ≈ 0.03 → below default threshold 0.35, no change reported
    changes = detect_changes(frames, mock_query, threshold=0.0)
    assert len(changes) == 1
    assert changes[0]["ref_frame_id"] == "close"
