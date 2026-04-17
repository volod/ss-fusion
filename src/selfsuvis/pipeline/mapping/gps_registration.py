"""GPS-to-ENU coordinate registration (Phase 1).

Converts per-frame GPS coordinates (lat, lon, alt) and pycolmap camera poses
to a unified ENU (East-North-Up) coordinate frame, enabling:
  - Cross-mission spatial comparison
  - Robot advisory API queries by GPS position
  - Phase 2 ICP fusion (as metric pose input)

Phase 1 (this module): GPS-based ENU registration.
  - ENU origin: first GPS-valid frame of the mission.
  - Camera pose in ENU: pycolmap pose composed with GPS-derived position.
  - Registration error: None (GPS accuracy is the only error bound).

Phase 2 (future): ICP-based refinement.
  - Uses Phase 1 global_pose_json as initial alignment for Open3D ICP.
  - Populates registration_error from ICP residual.

ENU convention (ROS REP-103):
  - X: East
  - Y: North
  - Z: Up

Usage:
    from selfsuvis.pipeline.mapping.gps_registration import register_mission_gps, gps_to_enu

    enu_origin, global_poses = register_mission_gps(frames)
    # frames: list of {gps_json: {lat, lon, alt}, pose_json: {R, t, ...}, ...}
    # enu_origin: {lat, lon, alt}  — the ENU reference point
    # global_poses: {frame_id: {position_enu: [x,y,z], R_enu: [[...]], ...}}
"""
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

# WGS-84 semi-major axis (metres)
_WGS84_A = 6_378_137.0
# WGS-84 first eccentricity squared
_WGS84_E2 = 6.69437999014e-3


# ── WGS-84 ↔ ECEF ──────────────────────────────────────────────────────────

def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates to ECEF (Earth-Centred Earth-Fixed)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
    x = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - _WGS84_E2) + alt_m) * math.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def _ecef_to_enu(
    ecef: np.ndarray,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_ecef: np.ndarray,
) -> np.ndarray:
    """Transform an ECEF point to ENU relative to the given origin."""
    lat = math.radians(origin_lat_deg)
    lon = math.radians(origin_lon_deg)
    # Rotation matrix from ECEF to ENU
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    R = np.array([
        [-sin_lon,          cos_lon,         0       ],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat],
    ], dtype=np.float64)
    delta = ecef - origin_ecef
    return R @ delta


def gps_to_enu(
    lat: float,
    lon: float,
    alt: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> Tuple[float, float, float]:
    """Convert a GPS coordinate to ENU relative to the given origin.

    Returns:
        (east_m, north_m, up_m) in metres.
    """
    origin_ecef = _geodetic_to_ecef(origin_lat, origin_lon, origin_alt)
    point_ecef = _geodetic_to_ecef(lat, lon, alt)
    enu = _ecef_to_enu(point_ecef, origin_lat, origin_lon, origin_ecef)
    return float(enu[0]), float(enu[1]), float(enu[2])


# ── Pose composition ─────────────────────────────────────────────────────────

def _compose_global_pose(
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    gps_enu: np.ndarray,
) -> Dict[str, Any]:
    """Compose pycolmap camera pose with GPS-derived ENU position.

    pycolmap convention: X_world = R @ X_cam + t  (world=colmap local frame)
    We anchor the colmap origin at gps_enu by adding the GPS offset to t.

    Returns a global_pose_json dict with:
        position_enu: [east, north, up]   — camera centre in ENU metres
        R_enu:        3×3 rotation matrix — camera rotation in ENU frame
        t_enu:        [tx, ty, tz]        — translation in ENU frame
    """
    # Camera centre in colmap world frame: C = -R^T @ t
    C_colmap = -R_cam.T @ t_cam
    # Anchor to ENU: camera centre in global ENU = GPS position + colmap offset
    # For Phase 1 we use the GPS position directly as the camera's ENU position.
    # (pycolmap's translation is in the local SfM frame; Phase 1 uses GPS only.)
    position_enu = gps_enu
    return {
        "position_enu": position_enu.tolist(),
        "R_enu": R_cam.tolist(),
        "t_enu": t_cam.tolist(),
        "phase": 1,
    }


# ── Main registration function ────────────────────────────────────────────────

def register_mission_gps(
    frames: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Phase 1 GPS-to-ENU registration for a mission.

    For each frame that has both a valid GPS fix (gps_json) and a pycolmap pose
    (pose_json with pose_status=success), computes its ENU position relative to
    the mission's GPS origin (first GPS-valid frame).

    Args:
        frames: List of frame dicts. Each dict should have:
                  id, gps_json (str|dict|None), pose_json (str|dict|None),
                  pose_status.

    Returns:
        (enu_origin, global_poses)
          enu_origin:   {lat, lon, alt} of the ENU reference point,
                        or None if no GPS-valid frames found.
          global_poses: {frame_id: global_pose_json dict} for all registered frames.
                        Empty dict if no frames can be registered.
    """
    # Find first GPS-valid frame to use as ENU origin
    enu_origin: Optional[Dict[str, Any]] = None
    for frame in frames:
        gps = _parse_json_field(frame.get("gps_json"))
        if gps and gps.get("lat") is not None:
            enu_origin = {
                "lat": float(gps["lat"]),
                "lon": float(gps["lon"]),
                "alt": float(gps.get("alt", 0.0)),
            }
            break

    if enu_origin is None:
        logger.warning("GPS registration: no GPS-valid frames found in mission")
        return None, {}

    origin_lat = enu_origin["lat"]
    origin_lon = enu_origin["lon"]
    origin_alt = enu_origin["alt"]
    origin_ecef = _geodetic_to_ecef(origin_lat, origin_lon, origin_alt)

    global_poses: Dict[str, Any] = {}
    n_registered = 0
    n_gps_only = 0

    for frame in frames:
        frame_id = frame.get("id") or frame.get("frame_id")
        gps = _parse_json_field(frame.get("gps_json"))
        pose = _parse_json_field(frame.get("pose_json"))

        if not gps or gps.get("lat") is None:
            continue

        lat = float(gps["lat"])
        lon = float(gps["lon"])
        alt = float(gps.get("alt", 0.0))
        point_ecef = _geodetic_to_ecef(lat, lon, alt)
        enu_vec = _ecef_to_enu(point_ecef, origin_lat, origin_lon, origin_ecef)

        if pose and frame.get("pose_status") == "success":
            # Full registration: GPS position + pycolmap rotation
            R_cam = np.array(pose["R"], dtype=np.float64)
            t_cam = np.array(pose["t"], dtype=np.float64)
            global_pose = _compose_global_pose(R_cam, t_cam, enu_vec)
            n_registered += 1
        else:
            # GPS-only: position known, orientation unknown
            global_pose = {
                "position_enu": enu_vec.tolist(),
                "R_enu": None,
                "t_enu": None,
                "phase": 1,
                "gps_only": True,
            }
            n_gps_only += 1

        if frame_id is not None:
            global_poses[frame_id] = global_pose

    logger.info(
        "GPS registration: origin=(%.6f, %.6f, %.1fm) registered=%d gps_only=%d",
        origin_lat, origin_lon, origin_alt, n_registered, n_gps_only,
    )
    return enu_origin, global_poses


def _parse_json_field(value: Any) -> Optional[Dict]:
    """Parse a field that may be a JSON string, dict, or None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# ── SE(3) registration transform ─────────────────────────────────────────────

def build_registration_transform(
    enu_origin: Dict[str, Any],
    reference_origin: Dict[str, Any],
) -> List[List[float]]:
    """Build a 4×4 SE(3) registration transform between two ENU origins.

    Used for global_map_missions.registration_transform_json.
    The transform maps from mission ENU to global-map ENU (Phase 1: pure translation).

    Args:
        enu_origin:       {lat, lon, alt} of the mission's ENU origin.
        reference_origin: {lat, lon, alt} of the global map's ENU origin.

    Returns:
        4×4 SE(3) matrix as nested float list (row-major).
    """
    east, north, up = gps_to_enu(
        enu_origin["lat"], enu_origin["lon"], enu_origin["alt"],
        reference_origin["lat"], reference_origin["lon"], reference_origin["alt"],
    )
    # Phase 1: rotation = identity (GPS doesn't give us orientation offset between origins)
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = east
    T[1, 3] = north
    T[2, 3] = up
    return T.tolist()
