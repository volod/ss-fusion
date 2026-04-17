"""3D Gaussian Splatting PLY I/O utilities.

Handles read/write of the standard 3DGS binary PLY format produced by
nerfstudio splatfacto and compatible with the original graphdeco-inria
gaussian-splatting codebase.

PLY format — 59 float32 properties per Gaussian:
  x, y, z             — position
  nx, ny, nz          — normals (always 0)
  f_dc_0..2           — SH degree-0 (DC) coefficients (base colour)
  f_rest_0..44        — SH degree-1..3 coefficients (view-dependent)
  opacity             — logit-encoded opacity
  scale_0..2          — log-encoded scale along each Gaussian axis
  rot_0..3            — rotation quaternion (WXYZ order)

Companion metadata (splat_meta.json next to the PLY):
  {origin_lat, origin_lon, origin_alt}  — GPS ENU origin used during Phase 1
                                           registration; required for ICP fusion.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── property names ────────────────────────────────────────────────────────────

_PROPS_XYZ    = ["x", "y", "z"]
_PROPS_NORMALS = ["nx", "ny", "nz"]
_PROPS_DC     = ["f_dc_0", "f_dc_1", "f_dc_2"]
_PROPS_REST   = [f"f_rest_{i}" for i in range(45)]
_PROPS_OPAC   = ["opacity"]
_PROPS_SCALE  = ["scale_0", "scale_1", "scale_2"]
_PROPS_ROT    = ["rot_0", "rot_1", "rot_2", "rot_3"]

ALL_PROPERTIES: list = (
    _PROPS_XYZ + _PROPS_NORMALS + _PROPS_DC + _PROPS_REST
    + _PROPS_OPAC + _PROPS_SCALE + _PROPS_ROT
)  # 59 total


# ── dtype ─────────────────────────────────────────────────────────────────────

_SPLAT_DTYPE = np.dtype([(p, "f4") for p in ALL_PROPERTIES])


# ── write ─────────────────────────────────────────────────────────────────────

def write_splat(path: str, data: np.ndarray) -> None:
    """Write a 3DGS PLY file from a structured numpy array.

    Args:
        path: output file path (e.g. maps/mission_id/splat.ply).
        data: structured array with dtype=_SPLAT_DTYPE (or compatible superset).
              Must contain at minimum the 59 standard 3DGS properties.
    """
    from plyfile import PlyData, PlyElement  # type: ignore

    # Ensure all required properties are present
    missing = [p for p in ALL_PROPERTIES if p not in data.dtype.names]
    if missing:
        raise ValueError(f"write_splat: missing properties: {missing}")

    # Cast to canonical dtype (picks only the 59 standard properties in order)
    canonical = np.empty(len(data), dtype=_SPLAT_DTYPE)
    for p in ALL_PROPERTIES:
        canonical[p] = data[p]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    el = PlyElement.describe(canonical, "vertex")
    PlyData([el], text=False).write(path)


def write_splat_from_arrays(
    path: str,
    positions: np.ndarray,               # (N, 3) float32
    opacities: np.ndarray,               # (N,)   float32  logit-encoded
    scales: np.ndarray,                  # (N, 3) float32  log-encoded
    rotations: np.ndarray,               # (N, 4) float32  WXYZ quaternion
    sh_dc: Optional[np.ndarray] = None,  # (N, 3) float32; zeros if None
    sh_rest: Optional[np.ndarray] = None,# (N,45) float32; zeros if None
) -> None:
    """Convenience wrapper: build structured array from component arrays then write."""
    n = len(positions)
    data = np.zeros(n, dtype=_SPLAT_DTYPE)
    data["x"], data["y"], data["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    # normals stay 0
    if sh_dc is not None:
        data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
    if sh_rest is not None:
        for i in range(45):
            data[f"f_rest_{i}"] = sh_rest[:, i]
    data["opacity"] = opacities
    data["scale_0"], data["scale_1"], data["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = (
        rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    )
    write_splat(path, data)


# ── read ──────────────────────────────────────────────────────────────────────

def read_splat(path: str) -> np.ndarray:
    """Read a 3DGS PLY file and return a structured numpy array.

    Returns array with dtype containing all properties present in the file
    (may be a superset of the standard 59 for extended formats).
    Raises FileNotFoundError if path does not exist.
    """
    from plyfile import PlyData  # type: ignore

    if not os.path.isfile(path):
        raise FileNotFoundError(f"splat.ply not found: {path}")
    ply = PlyData.read(path)
    return np.array(ply["vertex"].data)


def splat_positions(path: str) -> np.ndarray:
    """Return (N, 3) float32 array of Gaussian positions from a PLY file.

    Minimal read — only extracts x/y/z for use as ICP point cloud input.
    """
    data = read_splat(path)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def splat_count(path: str) -> int:
    """Return the number of Gaussians in a PLY file without loading all data."""
    from plyfile import PlyData  # type: ignore
    ply = PlyData.read(path)
    return len(ply["vertex"].data)


def is_splat_ply(path: str) -> bool:
    """Return True if the PLY file contains 3DGS Gaussian splat data.

    Checks for the presence of f_dc_0 or opacity in the vertex properties.
    """
    try:
        from plyfile import PlyData  # type: ignore
        ply = PlyData.read(path)
        names = {p.name for p in ply["vertex"].properties}
        return bool(names & {"f_dc_0", "opacity", "scale_0"})
    except Exception:
        return False


# ── companion metadata ────────────────────────────────────────────────────────

def _meta_path(splat_path: str) -> str:
    base, _ = os.path.splitext(splat_path)
    return base + "_meta.json"


def write_splat_metadata(
    splat_path: str,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write GPS ENU origin metadata alongside a splat.ply.

    Creates <splat_base>_meta.json with {origin_lat, origin_lon, origin_alt}.
    Required for Phase 2 ICP fusion to align splats across missions.
    """
    meta: Dict[str, Any] = {
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "origin_alt": origin_alt,
    }
    if extra:
        meta.update(extra)
    with open(_meta_path(splat_path), "w") as f:
        json.dump(meta, f, indent=2)


def read_splat_metadata(splat_path: str) -> Optional[Dict[str, Any]]:
    """Read companion GPS metadata for a splat.ply. Returns None if missing."""
    mp = _meta_path(splat_path)
    if not os.path.isfile(mp):
        return None
    with open(mp) as f:
        return json.load(f)


# ── SE(3) transform helpers ───────────────────────────────────────────────────

def _rot_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a WXYZ unit quaternion.

    Uses the numerically stable Shepperd method.  R must be a proper
    rotation (det ≈ 1).  Returns float32 array [w, x, y, z].
    """
    R = R.astype(np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Vectorised Hamilton product q1 ⊗ q2, both WXYZ convention.

    q1 may be shape (4,) — broadcast applied to each row of q2 (N, 4).
    Returns same shape as q2.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2 = q2[:, 0]; x2 = q2[:, 1]; y2 = q2[:, 2]; z2 = q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1).astype(np.float32)


# ── transform / merge ─────────────────────────────────────────────────────────

def apply_transform_to_splat(
    path_in: str,
    transform_4x4: List[List[float]],
    path_out: str,
) -> int:
    """Apply a SE(3) rigid-body transform to every Gaussian in a splat.ply.

    The transform is the 4×4 matrix returned by ICP (source → target frame).

    What changes:
      positions (x/y/z)     — p' = R·p + t
      rotation quaternions  — q' = q_align ⊗ q_gs   (covariance frame rotation)

    What is copied unchanged:
      scales (log-encoded, intrinsic to Gaussian shape)
      opacity
      SH DC term (degree-0 is rotationally invariant)
      SH rest (degrees 1–3; Wigner-D rotation is deferred — acceptable for
               spatial-memory / advisory use; see docs/architecture.md)

    Args:
        path_in:        Source splat.ply path.
        transform_4x4:  4×4 SE(3) list-of-lists from IcpResult.transform_4x4.
        path_out:       Output path (may equal path_in for in-place; written
                        atomically via a temp file).

    Returns:
        Number of Gaussians written.

    Raises:
        FileNotFoundError: if path_in does not exist.
    """
    T = np.array(transform_4x4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    data = read_splat(path_in)

    # ── 1. rotate + translate positions ──────────────────────────────────────
    pos = np.column_stack([
        data["x"].astype(np.float64),
        data["y"].astype(np.float64),
        data["z"].astype(np.float64),
    ])                                       # (N, 3)
    pos_t = (R @ pos.T).T + t               # (N, 3)
    data["x"] = pos_t[:, 0].astype(np.float32)
    data["y"] = pos_t[:, 1].astype(np.float32)
    data["z"] = pos_t[:, 2].astype(np.float32)

    # ── 2. compose alignment rotation into stored Gaussian quaternions ────────
    q_align = _rot_matrix_to_quat_wxyz(R)   # (4,) WXYZ
    quats = np.column_stack([
        data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"],
    ]).astype(np.float32)                   # (N, 4) WXYZ
    quats_new = _quat_multiply_wxyz(q_align, quats)
    # Normalise (float32 accumulation can drift slightly)
    norms = np.linalg.norm(quats_new, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    quats_new = (quats_new / norms).astype(np.float32)
    data["rot_0"] = quats_new[:, 0]
    data["rot_1"] = quats_new[:, 1]
    data["rot_2"] = quats_new[:, 2]
    data["rot_3"] = quats_new[:, 3]

    # All other properties (scales, opacity, SH) are unchanged.
    write_splat(path_out, data)
    return len(data)


def merge_splats(paths: List[str], path_out: str) -> int:
    """Concatenate multiple splat.ply files into a single merged PLY.

    All inputs must be valid 3DGS PLY files (same 59-property schema).
    Gaussians are simply concatenated — no deduplication or blending.

    Args:
        paths:    List of input splat.ply paths (at least one).
        path_out: Output path for the merged splat.

    Returns:
        Total number of Gaussians in the merged splat.

    Raises:
        ValueError:       if paths is empty.
        FileNotFoundError: if any path does not exist.
    """
    if not paths:
        raise ValueError("merge_splats: paths must not be empty")
    arrays = [read_splat(p) for p in paths]
    merged = np.concatenate(arrays)
    write_splat(path_out, merged)
    return len(merged)
