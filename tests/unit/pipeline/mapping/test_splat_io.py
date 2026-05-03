"""Unit tests for pipeline.splat_io and scripts.generate_test_splat.

Tests cover:
  - write/read round-trip for structured arrays
  - write_splat_from_arrays convenience wrapper
  - splat_positions extracts correct shape
  - splat_count without full load
  - is_splat_ply detection
  - companion metadata read/write
  - generator produces valid PLY files
  - test assets (scene_a/b/c) are present and valid
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from selfsuvis.pipeline.mapping.splat_io import (
    _SPLAT_DTYPE,
    ALL_PROPERTIES,
    is_splat_ply,
    read_splat,
    read_splat_metadata,
    splat_count,
    splat_positions,
    write_splat,
    write_splat_from_arrays,
    write_splat_metadata,
)

_ASSETS = Path(__file__).resolve().parents[3] / "assets" / "splats"

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_splat(n: int = 10, seed: int = 0) -> np.ndarray:
    """Create a minimal structured array with all 59 3DGS properties."""
    rng = np.random.default_rng(seed)
    data = np.zeros(n, dtype=_SPLAT_DTYPE)
    data["x"] = rng.uniform(-5, 5, n).astype(np.float32)
    data["y"] = rng.uniform(-5, 5, n).astype(np.float32)
    data["z"] = rng.uniform(-1, 1, n).astype(np.float32)
    data["opacity"] = rng.uniform(-2, 2, n).astype(np.float32)
    for s in ("scale_0", "scale_1", "scale_2"):
        data[s] = rng.uniform(-3, 0, n).astype(np.float32)
    data["rot_0"] = 1.0   # identity quaternion (w=1, x=y=z=0)
    return data


# ── write / read round-trip ───────────────────────────────────────────────────

def test_write_read_roundtrip():
    original = _make_splat(20, seed=1)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat(path, original)
        loaded = read_splat(path)
        assert len(loaded) == 20
        np.testing.assert_allclose(loaded["x"], original["x"], rtol=1e-5)
        np.testing.assert_allclose(loaded["opacity"], original["opacity"], rtol=1e-5)
    finally:
        os.unlink(path)


def test_all_59_properties_present_after_roundtrip():
    data = _make_splat(5)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat(path, data)
        loaded = read_splat(path)
        for prop in ALL_PROPERTIES:
            assert prop in loaded.dtype.names, f"Missing property: {prop}"
    finally:
        os.unlink(path)


def test_write_requires_all_properties():
    incomplete = np.zeros(5, dtype=[("x", "f4"), ("y", "f4")])
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="missing properties"):
            write_splat(path, incomplete)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_read_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        read_splat("/nonexistent/path/splat.ply")


# ── write_splat_from_arrays ───────────────────────────────────────────────────

def test_write_splat_from_arrays_roundtrip():
    n = 15
    rng = np.random.default_rng(7)
    positions  = rng.uniform(-10, 10, (n, 3)).astype(np.float32)
    opacities  = rng.uniform(-2, 2, n).astype(np.float32)
    scales     = rng.uniform(-3, 0, (n, 3)).astype(np.float32)
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rotations = q

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat_from_arrays(path, positions, opacities, scales, rotations)
        data = read_splat(path)
        assert len(data) == n
        np.testing.assert_allclose(data["x"], positions[:, 0], rtol=1e-5)
        np.testing.assert_allclose(data["y"], positions[:, 1], rtol=1e-5)
        np.testing.assert_allclose(data["z"], positions[:, 2], rtol=1e-5)
        np.testing.assert_allclose(data["opacity"], opacities, rtol=1e-5)
    finally:
        os.unlink(path)


def test_write_splat_from_arrays_sh_dc():
    n = 5
    rng = np.random.default_rng(8)
    positions = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    sh_dc = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    q = np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat_from_arrays(
            path, positions,
            opacities=np.zeros(n, dtype=np.float32),
            scales=np.zeros((n, 3), dtype=np.float32),
            rotations=q,
            sh_dc=sh_dc,
        )
        data = read_splat(path)
        np.testing.assert_allclose(data["f_dc_0"], sh_dc[:, 0], rtol=1e-5)
        np.testing.assert_allclose(data["f_dc_1"], sh_dc[:, 1], rtol=1e-5)
        np.testing.assert_allclose(data["f_dc_2"], sh_dc[:, 2], rtol=1e-5)
    finally:
        os.unlink(path)


# ── splat_positions ───────────────────────────────────────────────────────────

def test_splat_positions_shape():
    data = _make_splat(30)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat(path, data)
        pts = splat_positions(path)
        assert pts.shape == (30, 3)
        assert pts.dtype == np.float32
    finally:
        os.unlink(path)


def test_splat_positions_values_match():
    data = _make_splat(10, seed=3)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat(path, data)
        pts = splat_positions(path)
        np.testing.assert_allclose(pts[:, 0], data["x"], rtol=1e-5)
        np.testing.assert_allclose(pts[:, 1], data["y"], rtol=1e-5)
        np.testing.assert_allclose(pts[:, 2], data["z"], rtol=1e-5)
    finally:
        os.unlink(path)


# ── splat_count ───────────────────────────────────────────────────────────────

def test_splat_count():
    data = _make_splat(47)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat(path, data)
        assert splat_count(path) == 47
    finally:
        os.unlink(path)


# ── is_splat_ply ──────────────────────────────────────────────────────────────

def test_is_splat_ply_true_for_3dgs():
    data = _make_splat(5)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat(path, data)
        assert is_splat_ply(path) is True
    finally:
        os.unlink(path)


def test_is_splat_ply_false_for_non_splat():
    from plyfile import PlyData, PlyElement
    pts = np.array([(1.0, 2.0, 3.0)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(pts, "vertex")
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        PlyData([el]).write(path)
        assert is_splat_ply(path) is False
    finally:
        os.unlink(path)


def test_is_splat_ply_false_for_missing_file():
    assert is_splat_ply("/nonexistent/file.ply") is False


# ── metadata ──────────────────────────────────────────────────────────────────

def test_metadata_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = os.path.join(tmpdir, "splat.ply")
        write_splat(ply_path, _make_splat(3))
        write_splat_metadata(ply_path, 48.123, 11.456, 200.0)
        meta = read_splat_metadata(ply_path)
        assert meta["origin_lat"] == pytest.approx(48.123)
        assert meta["origin_lon"] == pytest.approx(11.456)
        assert meta["origin_alt"] == pytest.approx(200.0)


def test_metadata_returns_none_if_missing():
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        write_splat(path, _make_splat(3))
        assert read_splat_metadata(path) is None
    finally:
        os.unlink(path)


def test_metadata_extra_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = os.path.join(tmpdir, "splat.ply")
        write_splat(ply_path, _make_splat(3))
        write_splat_metadata(ply_path, 48.0, 11.0, 100.0, extra={"mission_id": "m1", "scene": 0})
        meta = read_splat_metadata(ply_path)
        assert meta["mission_id"] == "m1"
        assert meta["scene"] == 0


# ── generator ─────────────────────────────────────────────────────────────────

def test_generator_produces_valid_ply():
    from selfsuvis.scripts.generate_test_splat import generate_splat
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.ply")
        generate_splat(path, n_gaussians=50, origin_lat=48.0, origin_lon=11.0,
                       origin_alt=100.0, radius_m=10.0, seed=0)
        assert os.path.isfile(path)
        assert splat_count(path) == 50
        assert is_splat_ply(path)


def test_generator_metadata_matches_args():
    from selfsuvis.scripts.generate_test_splat import generate_splat
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.ply")
        generate_splat(path, n_gaussians=30, origin_lat=52.5, origin_lon=13.4,
                       origin_alt=50.0, seed=1)
        meta = read_splat_metadata(path)
        assert meta["origin_lat"] == pytest.approx(52.5)
        assert meta["origin_lon"] == pytest.approx(13.4)
        assert meta["n_gaussians"] == 30


def test_generator_positions_within_radius():
    from selfsuvis.scripts.generate_test_splat import generate_splat
    radius = 8.0
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.ply")
        generate_splat(path, n_gaussians=200, radius_m=radius, seed=42)
        pts = splat_positions(path)
        # All points should be within 2×radius of origin (generous bound for the scatter shape)
        dists = np.linalg.norm(pts[:, :2], axis=1)  # horizontal distance
        assert np.all(dists <= radius * 2.0), f"max dist={dists.max():.2f} > {radius*2}"


def test_generator_reproducible_with_seed():
    from selfsuvis.scripts.generate_test_splat import generate_splat
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "a.ply")
        p2 = os.path.join(tmpdir, "b.ply")
        generate_splat(p1, n_gaussians=20, seed=99)
        generate_splat(p2, n_gaussians=20, seed=99)
        d1 = read_splat(p1)
        d2 = read_splat(p2)
        np.testing.assert_array_equal(d1["x"], d2["x"])


# ── test assets ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scene", ["scene_a", "scene_b", "scene_c"])
def test_asset_exists(scene):
    path = _ASSETS / f"{scene}.ply"
    assert path.is_file(), f"Missing test asset: {path}"


@pytest.mark.parametrize("scene", ["scene_a", "scene_b", "scene_c"])
def test_asset_is_valid_splat(scene):
    path = str(_ASSETS / f"{scene}.ply")
    assert is_splat_ply(path)
    assert splat_count(path) == 200


@pytest.mark.parametrize("scene", ["scene_a", "scene_b", "scene_c"])
def test_asset_has_metadata(scene):
    path = str(_ASSETS / f"{scene}.ply")
    meta = read_splat_metadata(path)
    assert meta is not None
    assert "origin_lat" in meta
    assert "origin_lon" in meta
    assert "origin_alt" in meta


def test_scene_a_b_overlap():
    """scene_a and scene_b are ~7m apart — positions should overlap within combined radius."""
    pts_a = splat_positions(str(_ASSETS / "scene_a.ply"))
    pts_b = splat_positions(str(_ASSETS / "scene_b.ply"))
    # Both centered near origin in ENU, radius 10m each — centroids are close
    centroid_a = pts_a[:, :2].mean(axis=0)
    centroid_b = pts_b[:, :2].mean(axis=0)
    dist = np.linalg.norm(centroid_a - centroid_b)
    # In ENU, scene_b is at ~(5.5m E, 5.5m N) from scene_a — within combined radius
    assert dist < 20.0, f"Centroids too far apart for overlap test: {dist:.1f}m"


def test_scene_c_not_overlapping_with_a():
    """scene_c is ~6km from scene_a — ENU offset should be large."""
    meta_a = read_splat_metadata(str(_ASSETS / "scene_a.ply"))
    meta_c = read_splat_metadata(str(_ASSETS / "scene_c.ply"))
    # Approximate ENU distance from GPS difference
    dlat = (meta_c["origin_lat"] - meta_a["origin_lat"]) * 111_320.0
    dlon = (meta_c["origin_lon"] - meta_a["origin_lon"]) * 111_320.0 * 0.73  # cos(48°)
    dist = (dlat ** 2 + dlon ** 2) ** 0.5
    assert dist > 5_000.0, f"scene_c should be >5km from scene_a, got {dist:.0f}m"
