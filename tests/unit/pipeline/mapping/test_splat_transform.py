"""Unit tests for splat_io.apply_transform_to_splat and merge_splats.

Tests run against the synthetic test assets in tests/assets/splats/.
No open3d or GPU required.
"""
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ASSETS = Path(__file__).resolve().parents[3] / "assets" / "splats"


# ── helpers ───────────────────────────────────────────────────────────────────

def _identity_4x4():
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _translation_4x4(tx=0.0, ty=0.0, tz=0.0):
    return [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]


def _rotation_z_4x4(angle_deg: float):
    """Pure rotation around Z axis."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1],
    ]


def _rotation_y_4x4(angle_deg: float):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ]


# ── _rot_matrix_to_quat_wxyz ─────────────────────────────────────────────────

class TestRotToQuat:
    def test_identity_gives_w1(self):
        from selfsuvis.pipeline.mapping.splat_io import _rot_matrix_to_quat_wxyz
        R = np.eye(3)
        q = _rot_matrix_to_quat_wxyz(R)
        assert abs(q[0] - 1.0) < 1e-5   # w ≈ 1
        assert abs(q[1]) < 1e-5          # x ≈ 0
        assert abs(q[2]) < 1e-5
        assert abs(q[3]) < 1e-5

    def test_unit_norm(self):
        from selfsuvis.pipeline.mapping.splat_io import _rot_matrix_to_quat_wxyz
        for angle in [0, 30, 90, 180, 270]:
            a = math.radians(angle)
            R = np.array([[math.cos(a), -math.sin(a), 0],
                          [math.sin(a),  math.cos(a), 0],
                          [0, 0, 1]], dtype=np.float64)
            q = _rot_matrix_to_quat_wxyz(R)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-5

    def test_180_degree_rotation_around_z(self):
        from selfsuvis.pipeline.mapping.splat_io import _rot_matrix_to_quat_wxyz
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        q = _rot_matrix_to_quat_wxyz(R)
        # 180° around Z: w≈0, z≈1 (or -1)
        assert abs(q[0]) < 1e-4
        assert abs(abs(q[3]) - 1.0) < 1e-4


# ── _quat_multiply_wxyz ───────────────────────────────────────────────────────

class TestQuatMultiply:
    def test_identity_times_identity(self):
        from selfsuvis.pipeline.mapping.splat_io import _quat_multiply_wxyz
        q_id = np.array([1, 0, 0, 0], dtype=np.float32)
        q_batch = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        result = _quat_multiply_wxyz(q_id, q_batch)
        np.testing.assert_allclose(result, q_batch, atol=1e-6)

    def test_identity_times_arbitrary(self):
        from selfsuvis.pipeline.mapping.splat_io import _quat_multiply_wxyz
        q_id = np.array([1, 0, 0, 0], dtype=np.float32)
        q = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        result = _quat_multiply_wxyz(q_id, q)
        np.testing.assert_allclose(result, q, atol=1e-6)

    def test_result_unit_norm(self):
        from selfsuvis.pipeline.mapping.splat_io import _quat_multiply_wxyz
        rng = np.random.default_rng(42)
        q1 = rng.standard_normal(4).astype(np.float32)
        q1 /= np.linalg.norm(q1)
        q2 = rng.standard_normal((10, 4)).astype(np.float32)
        norms = np.linalg.norm(q2, axis=1, keepdims=True)
        q2 /= norms
        result = _quat_multiply_wxyz(q1, q2)
        norms_out = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms_out, np.ones(10), atol=1e-5)

    def test_180_twice_is_identity(self):
        """Rotating 180° around Z twice should give identity."""
        from selfsuvis.pipeline.mapping.splat_io import (
            _quat_multiply_wxyz,
            _rot_matrix_to_quat_wxyz,
        )
        R_180z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        q_180 = _rot_matrix_to_quat_wxyz(R_180z)
        q_id = np.array([[1, 0, 0, 0]], dtype=np.float32)
        step1 = _quat_multiply_wxyz(q_180, q_id)
        step2 = _quat_multiply_wxyz(q_180, step1)
        # q² = ±identity (sign can flip but represents same rotation)
        assert abs(abs(step2[0, 0]) - 1.0) < 1e-4   # w ≈ ±1
        np.testing.assert_allclose(step2[0, 1:], [0, 0, 0], atol=1e-4)


# ── apply_transform_to_splat ──────────────────────────────────────────────────

class TestApplyTransform:
    def test_identity_transform_leaves_positions_unchanged(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, splat_positions
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            apply_transform_to_splat(src, _identity_4x4(), out)
            orig = splat_positions(src)
            result = splat_positions(out)
            np.testing.assert_allclose(result, orig, atol=1e-5)
        finally:
            os.unlink(out)

    def test_pure_translation_shifts_positions(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, splat_positions
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            T = _translation_4x4(tx=10.0, ty=5.0, tz=-3.0)
            apply_transform_to_splat(src, T, out)
            orig = splat_positions(src)
            result = splat_positions(out)
            delta = result - orig                          # (N, 3)
            np.testing.assert_allclose(delta[:, 0], 10.0, atol=1e-4)
            np.testing.assert_allclose(delta[:, 1],  5.0, atol=1e-4)
            np.testing.assert_allclose(delta[:, 2], -3.0, atol=1e-4)
        finally:
            os.unlink(out)

    def test_returns_gaussian_count(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, splat_count
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            n = apply_transform_to_splat(src, _identity_4x4(), out)
            assert n == splat_count(src)
        finally:
            os.unlink(out)

    def test_rotation_rotates_positions(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, splat_positions
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            T = _rotation_z_4x4(90.0)
            apply_transform_to_splat(src, T, out)
            orig = splat_positions(src)
            result = splat_positions(out)
            # 90° around Z: x' ≈ -y, y' ≈ x
            np.testing.assert_allclose(result[:, 0], -orig[:, 1], atol=1e-4)
            np.testing.assert_allclose(result[:, 1],  orig[:, 0], atol=1e-4)
            np.testing.assert_allclose(result[:, 2],  orig[:, 2], atol=1e-4)
        finally:
            os.unlink(out)

    def test_rotation_updates_quaternions(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, read_splat
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            T = _rotation_y_4x4(45.0)
            apply_transform_to_splat(src, T, out)
            orig = read_splat(src)
            result = read_splat(out)
            # Quaternions should be unit-normalised
            quats = np.column_stack([result["rot_0"], result["rot_1"],
                                     result["rot_2"], result["rot_3"]])
            norms = np.linalg.norm(quats, axis=1)
            np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)
            # And they should differ from the originals (rotation was applied)
            orig_quats = np.column_stack([orig["rot_0"], orig["rot_1"],
                                          orig["rot_2"], orig["rot_3"]])
            assert not np.allclose(quats, orig_quats, atol=1e-4)
        finally:
            os.unlink(out)

    def test_opacity_and_scales_unchanged(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, read_splat
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            T = _rotation_z_4x4(37.0)
            apply_transform_to_splat(src, T, out)
            orig = read_splat(src)
            result = read_splat(out)
            np.testing.assert_array_equal(result["opacity"], orig["opacity"])
            np.testing.assert_array_equal(result["scale_0"], orig["scale_0"])
            np.testing.assert_array_equal(result["scale_1"], orig["scale_1"])
            np.testing.assert_array_equal(result["scale_2"], orig["scale_2"])
        finally:
            os.unlink(out)

    def test_sh_dc_unchanged(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, read_splat
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            apply_transform_to_splat(src, _rotation_z_4x4(60.0), out)
            orig = read_splat(src)
            result = read_splat(out)
            np.testing.assert_array_equal(result["f_dc_0"], orig["f_dc_0"])
            np.testing.assert_array_equal(result["f_dc_1"], orig["f_dc_1"])
            np.testing.assert_array_equal(result["f_dc_2"], orig["f_dc_2"])
        finally:
            os.unlink(out)

    def test_double_transform_is_additive(self):
        """Applying T1 then T2 should equal applying T1@T2 directly."""
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, splat_positions
        src = str(_ASSETS / "scene_a.ply")
        with (tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f1,
              tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f2,
              tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f3):
            p1, p2, p3 = f1.name, f2.name, f3.name
        try:
            T1 = _translation_4x4(5, 0, 0)
            T2 = _rotation_z_4x4(90)
            apply_transform_to_splat(src, T1, p1)
            apply_transform_to_splat(p1, T2, p2)

            T_combined = (np.array(T2) @ np.array(T1)).tolist()
            apply_transform_to_splat(src, T_combined, p3)

            result_two_step = splat_positions(p2)
            result_combined = splat_positions(p3)
            np.testing.assert_allclose(result_two_step, result_combined, atol=1e-4)
        finally:
            for p in [p1, p2, p3]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def test_missing_source_raises(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat
        with pytest.raises(FileNotFoundError):
            apply_transform_to_splat("/nonexistent/source.ply", _identity_4x4(), "/tmp/out.ply")

    def test_output_is_valid_splat(self):
        from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, is_splat_ply
        src = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            apply_transform_to_splat(src, _translation_4x4(1, 2, 3), out)
            assert is_splat_ply(out)
        finally:
            os.unlink(out)


# ── merge_splats ──────────────────────────────────────────────────────────────

class TestMergeSplats:
    def test_merged_count_is_sum(self):
        from selfsuvis.pipeline.mapping.splat_io import merge_splats, splat_count
        a = str(_ASSETS / "scene_a.ply")
        b = str(_ASSETS / "scene_b.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            n = merge_splats([a, b], out)
            assert n == splat_count(a) + splat_count(b)
        finally:
            os.unlink(out)

    def test_single_path_roundtrip(self):
        from selfsuvis.pipeline.mapping.splat_io import merge_splats, splat_count
        a = str(_ASSETS / "scene_a.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            n = merge_splats([a], out)
            assert n == splat_count(a)
        finally:
            os.unlink(out)

    def test_three_way_merge(self):
        from selfsuvis.pipeline.mapping.splat_io import merge_splats, splat_count
        paths = [str(_ASSETS / f"scene_{s}.ply") for s in ["a", "b", "c"]]
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            n = merge_splats(paths, out)
            assert n == sum(splat_count(p) for p in paths)
        finally:
            os.unlink(out)

    def test_positions_are_concatenated(self):
        from selfsuvis.pipeline.mapping.splat_io import merge_splats, splat_positions
        a = str(_ASSETS / "scene_a.ply")
        b = str(_ASSETS / "scene_b.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            merge_splats([a, b], out)
            merged_pos = splat_positions(out)
            a_pos = splat_positions(a)
            b_pos = splat_positions(b)
            expected = np.concatenate([a_pos, b_pos])
            np.testing.assert_array_equal(merged_pos, expected)
        finally:
            os.unlink(out)

    def test_merged_is_valid_splat(self):
        from selfsuvis.pipeline.mapping.splat_io import is_splat_ply, merge_splats
        a = str(_ASSETS / "scene_a.ply")
        b = str(_ASSETS / "scene_b.ply")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            out = f.name
        try:
            merge_splats([a, b], out)
            assert is_splat_ply(out)
        finally:
            os.unlink(out)

    def test_empty_paths_raises(self):
        from selfsuvis.pipeline.mapping.splat_io import merge_splats
        with pytest.raises(ValueError, match="empty"):
            merge_splats([], "/tmp/out.ply")

    def test_missing_path_raises(self):
        from selfsuvis.pipeline.mapping.splat_io import merge_splats
        with pytest.raises(FileNotFoundError):
            merge_splats(["/nonexistent.ply"], "/tmp/out.ply")


# ── _fuse_splat_files (pipeline/mapper.py) ────────────────────────────────────

class TestFuseSplatFiles:
    def test_produces_fused_ply_on_success(self):
        from selfsuvis.pipeline.mapping.mapper import _fuse_splat_files
        from selfsuvis.pipeline.mapping.splat_io import is_splat_ply, splat_count
        a = str(_ASSETS / "scene_a.ply")
        b = str(_ASSETS / "scene_b.ply")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy scene_b into tmpdir so fused.ply lands there
            import shutil
            src = shutil.copy(b, os.path.join(tmpdir, "splat.ply"))
            fused_path = _fuse_splat_files(src, a, _identity_4x4(), "m1", "main")
            assert fused_path is not None
            assert os.path.isfile(fused_path)
            assert is_splat_ply(fused_path)
            # Combined count: a + b
            assert splat_count(fused_path) == splat_count(a) + splat_count(b)

    def test_returns_none_on_bad_source(self):
        from selfsuvis.pipeline.mapping.mapper import _fuse_splat_files
        a = str(_ASSETS / "scene_a.ply")
        result = _fuse_splat_files("/nonexistent.ply", a, _identity_4x4(), "m1", "main")
        assert result is None

    def test_temp_file_cleaned_up(self):
        import shutil

        from selfsuvis.pipeline.mapping.mapper import _fuse_splat_files
        a = str(_ASSETS / "scene_a.ply")
        b = str(_ASSETS / "scene_b.ply")
        with tempfile.TemporaryDirectory() as tmpdir:
            src = shutil.copy(b, os.path.join(tmpdir, "splat.ply"))
            _fuse_splat_files(src, a, _identity_4x4(), "m1", "main")
            # Temp aligned file should be gone
            aligned_tmp = os.path.join(tmpdir, "_aligned_tmp.ply")
            assert not os.path.exists(aligned_tmp)
