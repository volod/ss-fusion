"""Sparse 3D map builder for the local full-analysis pipeline.

Builds a point cloud from mission frames either via pycolmap SfM
(camera centres) or a PCA projection of frame embeddings as fallback.
Optionally trains a 3D Gaussian Splat using gsplat (CUDA required).
Saves ``sparse_map.npz`` and ``map_stats.json`` into *map_dir*.

Returns
-------
dict with keys:
    npz_path    : str  — absolute path to saved .npz
    points      : np.ndarray shape (N, 3)
    colours     : np.ndarray shape (N, 3)  — RGB in [0, 1]
    sfm_poses   : int  — number of frames with recovered SfM pose
    scene_count : int  — number of SfM connected components
    method      : str  — "sfm" | "pca_dino" | "pca_clip" | "failed"
    splat_ply   : str | None  — path to 3DGS PLY (when gsplat ran)
    viewer_html : str | None  — path to standalone HTML viewer
    gsplat_method : str       — gsplat method or "skipped"
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.mapping import run_sfm

logger = get_logger(__name__)


def _sfm_point_cloud(
    video_path: str,
    video_id: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int, Optional[List], List[Dict[str, Any]]]:
    """Run pycolmap SfM and return point cloud plus per-frame anchor positions.

    sfm_frames is the raw frame list from run_sfm (with pose_json per frame) — used
    by the gsplat builder.  Returns (None, None, 0, 0, None, []) if SfM fails.
    """
    sfm_result = run_sfm(video_path, video_id, video_id)
    success_frames = [f for f in sfm_result["frames"] if f["pose_status"] == "success"]
    sfm_poses = len(success_frames)
    scene_count = sfm_result.get("scene_count", 0)
    logger.info("SfM: %d/%d poses recovered, %d scene(s)",
                sfm_poses, len(sfm_result["frames"]), scene_count)

    if sfm_poses == 0:
        return None, None, sfm_poses, scene_count, sfm_result["frames"], []

    pts, cols = [], []
    frame_positions: List[Dict[str, Any]] = []
    max_t = max(sfm_result["frames"][-1]["t_sec"], 1.0)
    for f in success_frames:
        pose = f["pose_json"]
        R = np.array(pose["R"])   # 3×3
        t = np.array(pose["t"])   # (3,)
        centre = (-R.T @ t).astype(np.float32)      # camera centre in world coords
        pts.append(centre)
        ratio = f["t_sec"] / max_t
        cols.append([ratio, 0.3, 1.0 - ratio])
        frame_positions.append(
            {
                "frame_path": f.get("frame_path"),
                "frame_id": f.get("id"),
                "t_sec": float(f["t_sec"]),
                "position": {
                    "x": float(centre[0]),
                    "y": float(centre[1]),
                    "z": float(centre[2]),
                },
            }
        )

    return (
        np.array(pts, dtype=np.float32),
        np.array(cols, dtype=np.float32),
        sfm_poses,
        scene_count,
        sfm_result["frames"],
        frame_positions,
    )


def _pca_point_cloud(
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str, List[Dict[str, Any]]]:
    """Project frame embeddings to 3D via PCA and return per-frame anchor positions."""
    dino_model = models.get("dino")
    model_for_pca = dino_model or models["clip"]
    step = max(1, len(frame_list) // 200)
    sampled_frames = frame_list[::step]
    imgs = [Image.open(fp).convert("RGB") for fp, _ in sampled_frames]
    embeds = model_for_pca.encode_images(imgs)           # (N, D)

    emb_centered = embeds - embeds.mean(axis=0)
    _, _, Vt = np.linalg.svd(emb_centered, full_matrices=False)
    points = (emb_centered @ Vt[:3].T).astype(np.float32)   # (N, 3)

    n = len(points)
    ratios = np.linspace(0, 1, n)
    colours = np.stack([ratios, np.full(n, 0.4), 1.0 - ratios], axis=1).astype(np.float32)

    method = "pca_dino" if dino_model else "pca_clip"
    logger.info("PCA point cloud: %d points (method=%s)", n, method)
    frame_positions = []
    for (frame_path, t_sec), point in zip(sampled_frames, points):
        frame_positions.append(
            {
                "frame_path": frame_path,
                "t_sec": float(t_sec),
                "position": {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                },
            }
        )
    return points, colours, method, frame_positions


def export_ply(
    points: np.ndarray,
    colours: np.ndarray,
    ply_path: Path,
) -> Path:
    """Write a coloured point cloud to a PLY file.

    Parameters
    ----------
    points   : (N, 3) float32 XYZ coordinates
    colours  : (N, 3) float32 RGB values in [0, 1]
    ply_path : destination path (parent dir must exist)

    Returns the path written.  Viewable in MeshLab, CloudCompare, Blender, etc.
    """
    from plyfile import PlyData, PlyElement

    rgb = (colours.clip(0, 1) * 255).astype(np.uint8)
    vertex = np.empty(len(points), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"]   = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"]  = rgb[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(ply_path))
    logger.info("PLY exported: %s (%d points)", ply_path, len(points))
    return Path(ply_path)


def build_sparse_map(
    video_path: str,
    video_id: str,
    map_dir: Path,
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
    run_sfm_flag: bool = True,
    run_gsplat_flag: bool = True,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Build sparse 3D map and optionally a 3D Gaussian Splat, saving to *map_dir*.

    Parameters
    ----------
    video_path      : absolute path to source video
    video_id        : unique video identifier
    map_dir         : output directory (created if absent)
    frame_list      : list of (frame_path, t_sec) tuples from frame extraction
    models          : dict with "clip" and optionally "dino" embedder instances
    run_sfm_flag    : attempt pycolmap SfM when True; go straight to PCA when False
    run_gsplat_flag : train 3D Gaussians with gsplat when True (CUDA required)
    device          : torch device string passed to gsplat
    """
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    npz_path   = map_dir / "sparse_map.npz"
    stats_path = map_dir / "map_stats.json"

    points3d: Optional[np.ndarray] = None
    colours:  Optional[np.ndarray] = None
    sfm_poses   = 0
    scene_count = 0
    method      = "none"
    sfm_frames: Optional[List] = None
    frame_positions: List[Dict[str, Any]] = []

    if run_sfm_flag:
        logger.info("Running Structure-from-Motion (pycolmap) …")
        try:
            points3d, colours, sfm_poses, scene_count, sfm_frames, frame_positions = _sfm_point_cloud(
                video_path, video_id
            )
            if points3d is not None:
                method = "sfm"
                logger.info("SfM point cloud: %d camera centres", len(points3d))
        except Exception as exc:
            logger.warning("SfM failed (%s) — using PCA fallback", exc)
    else:
        logger.info("SfM skipped — using PCA point-cloud fallback")

    if points3d is None:
        logger.info("Building PCA 3D point cloud from frame embeddings …")
        try:
            points3d, colours, method, frame_positions = _pca_point_cloud(frame_list, models)
        except Exception as exc:
            logger.warning("PCA fallback failed (%s)", exc)
            points3d = np.zeros((1, 3), dtype=np.float32)
            colours  = np.ones((1, 3),  dtype=np.float32)
            method   = "failed"
            frame_positions = []

    np.savez(str(npz_path), points=points3d, colours=colours)
    ply_path = export_ply(points3d, colours, map_dir / "sparse_map.ply")

    # ── 3D Gaussian Splatting ─────────────────────────────────────────────────
    splat_ply    = None
    viewer_html  = None
    gsplat_method = "skipped"
    if run_gsplat_flag:
        logger.info("Building 3D Gaussian Splat (gsplat) …")
        try:
            from selfsuvis.pipeline.mapping.gsplat import build_gaussian_splat
            gs = build_gaussian_splat(
                frame_list=frame_list,
                map_dir=map_dir,
                sfm_frames=sfm_frames,
                device=device,
            )
            gsplat_method = gs["method"]
            if not gs["skipped"]:
                splat_ply   = gs["splat_ply"]
                viewer_html = gs["viewer_html"]
                logger.info("  Gaussian Splat: %d Gaussians → %s  (%.1fs)",
                            gs["point_count"], Path(splat_ply).name, gs["train_sec"])
                logger.info("  Viewer HTML: %s", viewer_html)
            else:
                logger.info("  Gaussian Splat skipped: %s", gs["reason"])
        except Exception as exc:
            logger.warning("gsplat step failed: %s", exc, exc_info=True)
            gsplat_method = "error"

    map_stats = {
        "method":        method,
        "point_count":   int(len(points3d)),
        "sfm_poses":     sfm_poses,
        "scene_count":   scene_count,
        "frame_anchor_count": len(frame_positions),
        "npz":           str(npz_path),
        "ply":           str(ply_path),
        "gsplat_method": gsplat_method,
        "splat_ply":     splat_ply,
        "viewer_html":   viewer_html,
    }
    stats_path.write_text(json.dumps(map_stats, indent=2), encoding="utf-8")

    logger.info("3D map saved: %d points  method=%s", len(points3d), method)
    logger.info("  NPZ: %s", npz_path)
    logger.info("  PLY: %s  (open in MeshLab / CloudCompare / Blender)", ply_path)
    if splat_ply:
        logger.info("  3DGS: %s  (open view_splat.html in browser for interactive view)", splat_ply)
    return {
        "npz_path":      str(npz_path),
        "ply_path":      str(ply_path),
        "points":        points3d,
        "colours":       colours,
        "sfm_poses":     sfm_poses,
        "scene_count":   scene_count,
        "method":        method,
        "frame_positions": frame_positions,
        "splat_ply":     splat_ply,
        "viewer_html":   viewer_html,
        "gsplat_method": gsplat_method,
    }
