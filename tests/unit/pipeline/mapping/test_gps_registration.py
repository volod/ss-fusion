"""Unit tests for pipeline.gps_registration."""
import json
import math

import numpy as np
import pytest

from selfsuvis.pipeline.mapping.gps_registration import (
    _geodetic_to_ecef,
    _ecef_to_enu,
    build_registration_transform,
    gps_to_enu,
    register_mission_gps,
)


# ── gps_to_enu ────────────────────────────────────────────────────────────────

def test_gps_to_enu_same_point_is_origin():
    """A point equal to the origin maps to (0, 0, 0)."""
    e, n, u = gps_to_enu(48.0, 11.0, 500.0, 48.0, 11.0, 500.0)
    assert abs(e) < 1e-3
    assert abs(n) < 1e-3
    assert abs(u) < 1e-3


def test_gps_to_enu_east_displacement():
    """Moving east increases East component. Up may be non-zero for large offsets
    (Earth curvature: ~1km Up for 1° longitude displacement at equator)."""
    e, n, u = gps_to_enu(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert e > 100_000        # >100 km east
    assert abs(n) < 100       # negligible north
    # Up can be ~971m for 1° lon at equator due to Earth curvature — not a small-angle test


def test_gps_to_enu_north_displacement():
    """Moving north increases North component."""
    e, n, u = gps_to_enu(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert n > 100_000        # >100 km north
    assert abs(e) < 100


def test_gps_to_enu_altitude_displacement():
    """Moving up increases Up component."""
    e, n, u = gps_to_enu(48.0, 11.0, 100.0, 48.0, 11.0, 0.0)
    assert abs(u - 100.0) < 0.1   # ~100 m up
    assert abs(e) < 0.1
    assert abs(n) < 0.1


def test_gps_to_enu_small_offset():
    """50m north of origin should give ~50m North."""
    dlat = 50.0 / 111_320.0   # 50 m in degrees
    e, n, u = gps_to_enu(48.0 + dlat, 11.0, 0.0, 48.0, 11.0, 0.0)
    assert abs(n - 50.0) < 0.5   # within 0.5 m


def test_gps_to_enu_returns_floats():
    e, n, u = gps_to_enu(48.0, 11.0, 100.0, 48.0, 11.0, 0.0)
    assert isinstance(e, float)
    assert isinstance(n, float)
    assert isinstance(u, float)


# ── register_mission_gps ─────────────────────────────────────────────────────

def _make_frame(fid, lat, lon, alt, pose_status="success", R=None, t=None):
    gps = json.dumps({"lat": lat, "lon": lon, "alt": alt})
    if R is None:
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if t is None:
        t = [0.0, 0.0, 0.0]
    pose = json.dumps({"R": R, "t": t, "camera_id": 1, "image_name": f"frame_{fid}.jpg"})
    return {
        "id": str(fid),
        "gps_json": gps,
        "pose_json": pose if pose_status == "success" else None,
        "pose_status": pose_status,
    }


def test_register_no_gps_frames():
    """No GPS → origin is None, global_poses empty."""
    frames = [{"id": "1", "gps_json": None, "pose_json": None, "pose_status": "failed"}]
    origin, poses = register_mission_gps(frames)
    assert origin is None
    assert poses == {}


def test_register_empty_frames():
    origin, poses = register_mission_gps([])
    assert origin is None
    assert poses == {}


def test_register_origin_is_first_gps_frame():
    """ENU origin = first GPS-valid frame."""
    frames = [
        _make_frame(1, 48.0, 11.0, 100.0),
        _make_frame(2, 48.001, 11.001, 100.0),
    ]
    origin, poses = register_mission_gps(frames)
    assert origin["lat"] == pytest.approx(48.0)
    assert origin["lon"] == pytest.approx(11.0)
    assert origin["alt"] == pytest.approx(100.0)


def test_register_origin_frame_has_zero_enu():
    """The frame at the ENU origin has position_enu ≈ [0, 0, 0]."""
    frames = [_make_frame(1, 48.0, 11.0, 100.0)]
    origin, poses = register_mission_gps(frames)
    pos = poses["1"]["position_enu"]
    assert abs(pos[0]) < 1e-2
    assert abs(pos[1]) < 1e-2
    assert abs(pos[2]) < 1e-2


def test_register_second_frame_has_nonzero_offset():
    """Second frame at different GPS should have nonzero ENU offset."""
    frames = [
        _make_frame(1, 48.0, 11.0, 100.0),
        _make_frame(2, 48.001, 11.001, 100.0),
    ]
    origin, poses = register_mission_gps(frames)
    pos2 = poses["2"]["position_enu"]
    # Should be several tens of metres north and east
    assert pos2[0] > 10   # east
    assert pos2[1] > 50   # north


def test_register_gps_only_frame_has_no_rotation():
    """Frame with pose_status != success gets gps_only=True and R_enu=None."""
    frames = [_make_frame(1, 48.0, 11.0, 100.0, pose_status="failed")]
    origin, poses = register_mission_gps(frames)
    assert poses["1"]["gps_only"] is True
    assert poses["1"]["R_enu"] is None


def test_register_pose_success_has_rotation():
    """Registered frame with pose has R_enu populated and no gps_only key."""
    frames = [_make_frame(1, 48.0, 11.0, 100.0, pose_status="success")]
    origin, poses = register_mission_gps(frames)
    assert poses["1"].get("gps_only") is not True
    assert poses["1"]["R_enu"] is not None


def test_register_phase_is_1():
    """All Phase 1 poses have phase=1."""
    frames = [_make_frame(1, 48.0, 11.0, 100.0)]
    origin, poses = register_mission_gps(frames)
    assert poses["1"]["phase"] == 1


def test_register_json_string_gps():
    """gps_json as JSON string is parsed correctly."""
    frame = {
        "id": "x",
        "gps_json": '{"lat": 52.0, "lon": 13.0, "alt": 50.0}',
        "pose_json": None,
        "pose_status": "failed",
    }
    origin, poses = register_mission_gps([frame])
    assert origin["lat"] == pytest.approx(52.0)


def test_register_dict_gps():
    """gps_json as dict is accepted directly."""
    frame = {
        "id": "y",
        "gps_json": {"lat": 52.0, "lon": 13.0, "alt": 50.0},
        "pose_json": None,
        "pose_status": "failed",
    }
    origin, poses = register_mission_gps([frame])
    assert origin["lat"] == pytest.approx(52.0)


# ── build_registration_transform ────────────────────────────────────────────

def test_registration_transform_identity_same_origin():
    """Same origin → identity translation (zeros in last column)."""
    orig = {"lat": 48.0, "lon": 11.0, "alt": 100.0}
    T = build_registration_transform(orig, orig)
    T_arr = np.array(T)
    # Rotation block should be identity
    np.testing.assert_allclose(T_arr[:3, :3], np.eye(3), atol=1e-6)
    # Translation should be ~zero
    np.testing.assert_allclose(T_arr[:3, 3], [0, 0, 0], atol=1e-3)
    assert T_arr[3, 3] == pytest.approx(1.0)


def test_registration_transform_shape():
    orig = {"lat": 48.0, "lon": 11.0, "alt": 100.0}
    ref = {"lat": 48.001, "lon": 11.001, "alt": 100.0}
    T = build_registration_transform(orig, ref)
    assert len(T) == 4
    for row in T:
        assert len(row) == 4


def test_registration_transform_nonzero_offset():
    orig = {"lat": 48.0, "lon": 11.0, "alt": 100.0}
    ref = {"lat": 47.999, "lon": 10.999, "alt": 100.0}
    T = build_registration_transform(orig, ref)
    east, north = T[0][3], T[1][3]
    # Should be non-zero (different GPS positions)
    assert abs(east) > 10
    assert abs(north) > 50
