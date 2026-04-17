"""Unit tests for pipeline.heuristics."""

import numpy as np
import pytest

pytest.importorskip("cv2", reason="cv2 required for heuristics")
pytest.importorskip("skimage.metrics", reason="skimage required for heuristics")

from selfsuvis.pipeline.media.heuristics import (
    downsample_gray,
    edge_density,
    histogram_diff,
    mean_abs_diff,
    mean_intensity,
    tile_entropy,
    tile_std,
)


def test_downsample_gray():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    out = downsample_gray(img, size=32)
    assert out.shape == (32, 32)
    assert out.dtype == np.uint8


def test_mean_intensity():
    gray = np.ones((10, 10), dtype=np.uint8) * 128
    assert mean_intensity(gray) == 128.0


def test_histogram_diff_same():
    a = np.ones((64, 64), dtype=np.uint8) * 100
    b = np.ones((64, 64), dtype=np.uint8) * 100
    assert histogram_diff(a, b) < 0.01


def test_histogram_diff_different():
    a = np.zeros((64, 64), dtype=np.uint8)
    b = np.ones((64, 64), dtype=np.uint8) * 255
    assert histogram_diff(a, b) > 0.5


def test_mean_abs_diff():
    a = np.zeros((64, 64), dtype=np.uint8)
    b = np.ones((64, 64), dtype=np.uint8) * 255
    diff = mean_abs_diff(a, b)
    assert 0.9 < diff <= 1.0


def test_mean_abs_diff_same():
    a = np.ones((64, 64), dtype=np.uint8) * 100
    assert mean_abs_diff(a, a) == 0.0


def test_edge_density():
    gray = np.zeros((64, 64), dtype=np.uint8)
    assert edge_density(gray) == 0.0
    gray[32, :] = 255
    assert edge_density(gray) > 0


def test_tile_std():
    gray = np.ones((64, 64), dtype=np.uint8) * 128
    assert tile_std(gray) < 1.0
    gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    assert tile_std(gray) > 0


def test_tile_entropy():
    gray = np.ones((64, 64), dtype=np.uint8) * 128
    assert tile_entropy(gray) < 1.0
    gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    assert tile_entropy(gray) > 0
