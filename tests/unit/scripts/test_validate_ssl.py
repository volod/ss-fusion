"""Unit tests for scripts/validate_ssl_improvement.py.

Tests the pure utility functions — recall_at_1, _collect_frames, gate logic.
No model loading or GPU required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── recall_at_1 ───────────────────────────────────────────────────────────────

def _import_r1():
    from selfsuvis.scripts.validate_ssl_improvement import recall_at_1
    return recall_at_1


def test_recall_at_1_single_frame():
    """Single frame → can't compute R@1; returns 0.0."""
    recall_at_1 = _import_r1()
    embs = np.random.randn(1, 8).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= norms
    assert recall_at_1(embs) == 0.0


def test_recall_at_1_empty():
    """Zero frames → 0.0."""
    recall_at_1 = _import_r1()
    embs = np.zeros((0, 8), dtype=np.float32)
    assert recall_at_1(embs) == 0.0


def test_recall_at_1_perfect_temporal_ordering():
    """Smooth trajectory embeddings → R@1 should be well above chance (≥ 0.5)."""
    recall_at_1 = _import_r1()
    np.random.seed(7)
    # Create a smooth trajectory: each frame is a small step from the previous.
    # This guarantees adjacent frames are genuinely more similar than distant ones.
    n = 20
    steps = np.random.randn(n, 16).astype(np.float32) * 0.1
    embs = np.cumsum(steps, axis=0)  # cumulative sum → smooth walk
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms < 1e-8, 1.0, norms)

    r1 = recall_at_1(embs, window=5)
    # Nearest neighbours of each frame should mostly be temporally adjacent
    assert r1 >= 0.5, f"Expected R@1 ≥ 0.5 for smooth trajectory frames, got {r1:.3f}"


def test_recall_at_1_random_embeddings_below_1():
    """Random orthogonal embeddings → R@1 should be in [0, 1]."""
    recall_at_1 = _import_r1()
    np.random.seed(0)
    embs = np.random.randn(30, 64).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= np.where(norms < 1e-8, 1.0, norms)
    r1 = recall_at_1(embs, window=2)
    assert 0.0 <= r1 <= 1.0


def test_recall_at_1_output_is_float():
    recall_at_1 = _import_r1()
    embs = np.random.randn(10, 8).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= norms
    r1 = recall_at_1(embs)
    assert isinstance(r1, float)


# ── _collect_frames ───────────────────────────────────────────────────────────

def _import_collect():
    from selfsuvis.scripts.validate_ssl_improvement import _collect_frames
    return _collect_frames


def test_collect_frames_finds_jpg(tmp_path):
    _collect_frames = _import_collect()
    (tmp_path / "frame_001.jpg").write_bytes(b"fake")
    (tmp_path / "frame_002.jpg").write_bytes(b"fake")
    frames = _collect_frames(tmp_path)
    assert len(frames) == 2


def test_collect_frames_finds_png(tmp_path):
    _collect_frames = _import_collect()
    (tmp_path / "a.png").write_bytes(b"fake")
    frames = _collect_frames(tmp_path)
    assert len(frames) == 1


def test_collect_frames_ignores_non_image_files(tmp_path):
    _collect_frames = _import_collect()
    (tmp_path / "frame.jpg").write_bytes(b"fake")
    (tmp_path / "metadata.json").write_bytes(b"{}")
    (tmp_path / "video.mp4").write_bytes(b"fake")
    frames = _collect_frames(tmp_path)
    assert len(frames) == 1


def test_collect_frames_recursive(tmp_path):
    _collect_frames = _import_collect()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "frame.jpg").write_bytes(b"fake")
    (tmp_path / "root.png").write_bytes(b"fake")
    frames = _collect_frames(tmp_path)
    assert len(frames) == 2


def test_collect_frames_empty_dir(tmp_path):
    _collect_frames = _import_collect()
    frames = _collect_frames(tmp_path)
    assert frames == []


def test_collect_frames_sorted(tmp_path):
    _collect_frames = _import_collect()
    for name in ["frame_003.jpg", "frame_001.jpg", "frame_002.jpg"]:
        (tmp_path / name).write_bytes(b"fake")
    frames = _collect_frames(tmp_path)
    names = [f.name for f in frames]
    assert names == sorted(names)


# ── Gate logic ────────────────────────────────────────────────────────────────

def test_gate_passes_when_two_of_three_pass():
    """Gate passes when ΔR@1 > 0.02 on ≥2/3 videos."""
    from selfsuvis.scripts.validate_ssl_improvement import _GATE_VIDEOS, _GATE_DELTA

    video_results = {
        "video_a": {"delta_median": 0.05, "gate_passed": True},
        "video_b": {"delta_median": 0.03, "gate_passed": True},
        "video_c": {"delta_median": -0.01, "gate_passed": False},
    }
    n_passed = sum(1 for r in video_results.values() if r["gate_passed"])
    assert n_passed >= _GATE_VIDEOS


def test_gate_fails_when_only_one_passes():
    from selfsuvis.scripts.validate_ssl_improvement import _GATE_VIDEOS

    video_results = {
        "video_a": {"delta_median": 0.05, "gate_passed": True},
        "video_b": {"delta_median": -0.01, "gate_passed": False},
        "video_c": {"delta_median": 0.01, "gate_passed": False},
    }
    n_passed = sum(1 for r in video_results.values() if r["gate_passed"])
    assert n_passed < _GATE_VIDEOS


def test_gate_threshold():
    """delta exactly at the boundary does NOT pass the gate (strict >)."""
    from selfsuvis.scripts.validate_ssl_improvement import _GATE_DELTA
    assert _GATE_DELTA == pytest.approx(0.02)
    delta = 0.02  # equal, not strictly greater
    gate_passed = delta > _GATE_DELTA
    assert gate_passed is False
