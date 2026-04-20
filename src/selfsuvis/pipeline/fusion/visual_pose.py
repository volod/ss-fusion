"""Visual-pose constraint fusion via Sim(3) alignment (Umeyama algorithm).

Converts SfM camera centres (arbitrary local frame, unknown scale) to ENU
measurements by fitting the optimal rotation, translation, and scale that
maps the SfM positions to the corresponding GPS-ENU positions.

The resulting per-frame aligned positions are returned as PlatformMeasurement
instances with covariance derived from the alignment residual, ready to be
fed into the platform Kalman filter alongside GPS.

Reference:
    Umeyama, S. (1991). "Least-squares estimation of transformation parameters
    between two point patterns." IEEE T-PAMI, 13(4), 376-380.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from selfsuvis.pipeline.fusion.measurements import PlatformMeasurement

logger = logging.getLogger(__name__)

# Minimum aligned frames needed for a reliable Sim(3) estimate
_MIN_ALIGNED_FRAMES = 6
# Fallback measurement std when alignment residual is zero (perfect fit)
_MIN_STD_M = 0.5


def _umeyama_sim3(
    src: np.ndarray,  # (N, 3) source points (SfM)
    dst: np.ndarray,  # (N, 3) destination points (ENU)
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Fit the optimal Sim(3): dst ≈ scale * R @ src + t.

    Returns:
        scale, R (3×3), t (3,), rmse (m)
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma2_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
    if sigma2_src < 1e-12:
        # Degenerate: all SfM points are the same
        return 1.0, np.eye(3), mu_dst - mu_src, float(np.linalg.norm(dst_c))

    Sigma = (dst_c.T @ src_c) / n  # (3, 3)
    U, D, Vt = np.linalg.svd(Sigma)

    # Handle reflection
    det_UVt = float(np.linalg.det(U @ Vt))
    S_diag = np.ones(3)
    if det_UVt < 0:
        S_diag[-1] = -1.0

    R = U @ np.diag(S_diag) @ Vt
    scale = float((D * S_diag).sum() / sigma2_src)
    t = mu_dst - scale * R @ mu_src

    # RMSE
    residuals = dst - (scale * (src @ R.T) + t)
    rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    return scale, R, t, rmse


def align_sfm_to_enu(
    sfm_positions: Sequence[Dict[str, Any]],
    gps_enu_by_t: Dict[float, Tuple[float, float, float]],
    gps_std_m: float = 5.0,
    sfm_base_std_m: float = 2.0,
    min_frames: int = _MIN_ALIGNED_FRAMES,
) -> Tuple[Optional[Dict[str, Any]], List[PlatformMeasurement]]:
    """Align SfM camera centres to GPS-ENU and produce visual-pose measurements.

    Args:
        sfm_positions: List of dicts with keys "t_sec" and "position" {"x","y","z"}
                       (camera centres in the SfM local frame, arbitrary scale).
        gps_enu_by_t:  Mapping from frame timestamp to (x, y, z) GPS-ENU position.
        gps_std_m:     GPS measurement noise (metres), used for co-located pair check.
        sfm_base_std_m: Baseline SfM positional uncertainty after alignment (metres).
        min_frames:    Minimum number of co-located SfM+GPS frames for Umeyama.

    Returns:
        (alignment_info_dict, list of PlatformMeasurement with kind="sfm_position")
        alignment_info_dict is None if insufficient data.
    """
    # Build co-located pairs (SfM + GPS at the same timestamp, ±0.1 s)
    src_pts, dst_pts, t_secs = [], [], []
    for sfm_f in sfm_positions:
        t = float(sfm_f.get("t_sec", 0.0))
        pos = sfm_f.get("position", {})
        # Find closest GPS sample
        best_t = min(gps_enu_by_t.keys(), key=lambda gt: abs(gt - t), default=None)
        if best_t is None or abs(best_t - t) > 0.5:
            continue
        gps_xyz = gps_enu_by_t[best_t]
        src_pts.append([pos["x"], pos["y"], pos["z"]])
        dst_pts.append(list(gps_xyz))
        t_secs.append(t)

    if len(src_pts) < min_frames:
        logger.info(
            "Visual-pose alignment skipped: only %d co-located SfM+GPS frames "
            "(need ≥ %d)", len(src_pts), min_frames,
        )
        return None, []

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)

    scale, R, t_vec, rmse = _umeyama_sim3(src, dst)

    # Measurement std: base + residual (RMSE of alignment)
    meas_std = max(_MIN_STD_M, sfm_base_std_m + rmse)
    meas_var = meas_std ** 2
    cov = tuple(
        tuple(float(v) for v in row)
        for row in np.diag([meas_var, meas_var, meas_var])
    )

    logger.info(
        "Visual-pose Sim(3) alignment: %d frames | scale=%.3f | RMSE=%.2f m | "
        "meas_std=%.2f m",
        len(src_pts), scale, rmse, meas_std,
    )

    # Build measurements for ALL SfM frames (not only the ones used for fitting)
    measurements: List[PlatformMeasurement] = []
    for sfm_f in sfm_positions:
        t = float(sfm_f.get("t_sec", 0.0))
        pos = sfm_f.get("position", {})
        p_sfm = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
        p_enu = scale * R @ p_sfm + t_vec
        measurements.append(
            PlatformMeasurement(
                kind="sfm_position",
                t_sec=t,
                values=(float(p_enu[0]), float(p_enu[1]), float(p_enu[2])),
                covariance=cov,
                source="sfm_umeyama",
                frame="enu",
                quality="nominal" if rmse < gps_std_m else "approx_world_frame",
            )
        )

    alignment_info = {
        "n_aligned_frames": len(src_pts),
        "scale": round(scale, 6),
        "rmse_m": round(rmse, 4),
        "meas_std_m": round(meas_std, 4),
    }
    return alignment_info, measurements
