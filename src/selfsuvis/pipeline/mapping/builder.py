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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.mapping import run_sfm

logger = get_logger(__name__)

_MIN_USEFUL_MAP_POINTS = 50
_MIN_USEFUL_SFM_POSES = 20


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


def _visual_pca_point_cloud(
    frame_list: List[Tuple[str, float]],
) -> Tuple[np.ndarray, np.ndarray, str, List[Dict[str, Any]]]:
    """CPU-only PCA fallback from low-resolution frame appearance.

    The local pipeline can build SfM in a background thread while foreground
    steps offload/restore CLIP and DINO on GPU. Reusing those shared embedders
    for fallback geometry is therefore brittle. This fallback intentionally uses
    simple image features so sparse-map repair never depends on model dtype,
    device, or thread state.
    """
    if not frame_list:
        return (
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            "pca_pixels",
            [],
        )

    step = max(1, len(frame_list) // 200)
    sampled_frames = frame_list[::step]
    features: List[np.ndarray] = []
    colours_src: List[np.ndarray] = []
    max_t = max((float(t_sec) for _, t_sec in sampled_frames), default=1.0) or 1.0

    for fp, t_sec in sampled_frames:
        with Image.open(fp) as img:
            rgb = img.convert("RGB")
            thumb = rgb.resize((32, 32), Image.Resampling.BILINEAR)
            arr = np.asarray(thumb, dtype=np.float32) / 255.0
        flat = arr.reshape(-1)
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        time_feature = np.array([float(t_sec) / max_t], dtype=np.float32)
        features.append(np.concatenate([flat, mean, std, time_feature]).astype(np.float32))
        colours_src.append(mean.astype(np.float32))

    matrix = np.stack(features, axis=0)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if len(sampled_frames) == 1 or float(np.abs(centered).sum()) == 0.0:
        points = np.zeros((len(sampled_frames), 3), dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[: min(3, len(vt))]
        projected = centered @ components.T
        if projected.shape[1] < 3:
            projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))
        points = projected[:, :3].astype(np.float32)
        scale = np.percentile(np.abs(points), 95)
        if scale > 0:
            points = points / float(scale)

    colours = np.stack(colours_src, axis=0).astype(np.float32)
    frame_positions: List[Dict[str, Any]] = []
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
    logger.info("Visual PCA point cloud: %d points (method=pca_pixels)", len(points))
    return points, colours, "pca_pixels", frame_positions


def _interpolate_frame_positions(
    frame_list: List[Tuple[str, float]],
    sparse_positions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Interpolate sparse SfM camera centres across all frame timestamps."""
    if not frame_list or not sparse_positions:
        return []
    ordered = sorted(
        (
            float(item.get("t_sec", 0.0)),
            float(item.get("position", {}).get("x", 0.0)),
            float(item.get("position", {}).get("y", 0.0)),
            float(item.get("position", {}).get("z", 0.0)),
        )
        for item in sparse_positions
        if isinstance(item, dict) and isinstance(item.get("position"), dict)
    )
    if not ordered:
        return []
    src_t = np.array([row[0] for row in ordered], dtype=np.float32)
    src_x = np.array([row[1] for row in ordered], dtype=np.float32)
    src_y = np.array([row[2] for row in ordered], dtype=np.float32)
    src_z = np.array([row[3] for row in ordered], dtype=np.float32)
    if len(src_t) == 1:
        src_t = np.array([src_t[0], src_t[0] + 1e-3], dtype=np.float32)
        src_x = np.array([src_x[0], src_x[0]], dtype=np.float32)
        src_y = np.array([src_y[0], src_y[0]], dtype=np.float32)
        src_z = np.array([src_z[0], src_z[0]], dtype=np.float32)

    interpolated: List[Dict[str, Any]] = []
    for frame_path, t_sec in frame_list:
        t = float(t_sec)
        interpolated.append(
            {
                "frame_path": frame_path,
                "t_sec": t,
                "position": {
                    "x": float(np.interp(t, src_t, src_x)),
                    "y": float(np.interp(t, src_t, src_y)),
                    "z": float(np.interp(t, src_t, src_z)),
                },
            }
        )
    return interpolated


def _position_lookup(frame_positions: List[Dict[str, Any]]) -> tuple[Dict[str, Dict[str, float]], List[Tuple[float, Dict[str, float]]]]:
    by_path: Dict[str, Dict[str, float]] = {}
    by_time: List[Tuple[float, Dict[str, float]]] = []
    for item in frame_positions:
        pos = item.get("position")
        if not isinstance(pos, dict):
            continue
        entry = {
            "x": float(pos.get("x", 0.0)),
            "y": float(pos.get("y", 0.0)),
            "z": float(pos.get("z", 0.0)),
        }
        if item.get("frame_path"):
            by_path[str(item["frame_path"])] = entry
        if item.get("t_sec") is not None:
            by_time.append((float(item["t_sec"]), entry))
    by_time.sort(key=lambda pair: pair[0])
    return by_path, by_time


def _nearest_position(
    frame_path: Optional[str],
    t_sec: float,
    *,
    by_path: Dict[str, Dict[str, float]],
    by_time: List[Tuple[float, Dict[str, float]]],
) -> Optional[Dict[str, float]]:
    if frame_path and frame_path in by_path:
        return by_path[frame_path]
    if not by_time:
        return None
    nearest_t, nearest_pos = min(by_time, key=lambda pair: abs(pair[0] - t_sec))
    if abs(nearest_t - t_sec) <= 1.5:
        return nearest_pos
    return None


def _bbox_area(bbox: Optional[Iterable[float]]) -> float:
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _priority_color(priority_label: str) -> np.ndarray:
    label = str(priority_label or "other").lower()
    if label == "human":
        return np.array([0.95, 0.25, 0.25], dtype=np.float32)
    if label == "vehicle":
        return np.array([0.20, 0.45, 0.95], dtype=np.float32)
    if label == "artificial":
        return np.array([0.25, 0.80, 0.35], dtype=np.float32)
    return np.array([0.65, 0.65, 0.65], dtype=np.float32)


def _build_semantic_enriched_point_cloud(
    *,
    frame_positions: List[Dict[str, Any]],
    yolo_detection_results: Optional[List[Dict[str, Any]]] = None,
    tracking_results: Optional[Dict[str, Any]] = None,
    depth_results: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a denser pseudo-3D cloud from trajectory anchors and detections.

    This is not metrically exact geometry. It produces a richer structural map
    when SfM is too sparse by anchoring object observations around the recovered
    or interpolated camera trajectory using bbox layout and coarse depth cues.
    """
    if not frame_positions:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    by_path, by_time = _position_lookup(frame_positions)
    depth_by_t: Dict[float, Dict[str, Any]] = {
        float(r.get("t_sec")): r.get("depth", {})
        for r in (depth_results or [])
        if r.get("t_sec") is not None and isinstance(r.get("depth"), dict)
    }

    points: List[List[float]] = []
    colours: List[np.ndarray] = []

    ordered_positions = sorted(frame_positions, key=lambda item: float(item.get("t_sec", 0.0)))
    for idx, item in enumerate(ordered_positions):
        pos = item.get("position", {})
        center = np.array(
            [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))],
            dtype=np.float32,
        )
        points.append(center.tolist())
        colours.append(np.array([0.95, 0.75, 0.15], dtype=np.float32))

        prev_pos = ordered_positions[max(0, idx - 1)].get("position", pos)
        next_pos = ordered_positions[min(len(ordered_positions) - 1, idx + 1)].get("position", pos)
        tangent = np.array(
            [
                float(next_pos.get("x", 0.0)) - float(prev_pos.get("x", 0.0)),
                float(next_pos.get("y", 0.0)) - float(prev_pos.get("y", 0.0)),
                0.0,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(tangent))
        if norm > 1e-6:
            tangent = tangent / norm
            lateral = np.array([-tangent[1], tangent[0], 0.0], dtype=np.float32)
            width = 0.08
            points.append((center + lateral * width).tolist())
            points.append((center - lateral * width).tolist())
            colours.append(np.array([0.85, 0.55, 0.10], dtype=np.float32))
            colours.append(np.array([0.85, 0.55, 0.10], dtype=np.float32))

    obs_frames: List[Dict[str, Any]] = []
    obs_frames.extend(yolo_detection_results or [])
    if tracking_results:
        obs_frames.extend(tracking_results.get("frames", []) or [])

    seen_keys: set[Tuple[str, float, str, str]] = set()
    track_points: Dict[str, List[np.ndarray]] = {}
    track_colors: Dict[str, np.ndarray] = {}
    for frame in obs_frames:
        detections = frame.get("detections") or []
        if not detections:
            continue
        t_sec = float(frame.get("t_sec", 0.0))
        anchor = _nearest_position(
            frame.get("frame_path"),
            t_sec,
            by_path=by_path,
            by_time=by_time,
        )
        if anchor is None:
            continue
        anchor_vec = np.array([anchor["x"], anchor["y"], anchor["z"]], dtype=np.float32)
        depth_info = depth_by_t.get(t_sec, {})
        depth_percentiles = depth_info.get("percentiles", [0.5, 0.5, 0.5, 0.5, 0.5])
        depth_mid = float(depth_percentiles[2]) if len(depth_percentiles) >= 3 else 0.5

        for det in detections:
            label = str(det.get("label", "") or "")
            bbox = det.get("bbox_norm")
            if not bbox or len(bbox) != 4:
                continue
            track_id = det.get("track_id")
            dedupe_key = (
                str(frame.get("frame_path", "")),
                round(t_sec, 3),
                label,
                str(track_id if track_id is not None else tuple(round(float(v), 4) for v in bbox)),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area = max(_bbox_area(bbox), float(det.get("mask_area_norm", 0.0) or 0.0))
            spread = 0.10 + 1.10 * float(np.sqrt(max(area, 1e-5)))
            lateral = (cx - 0.5) * (0.9 + 2.0 * spread)
            forward = (0.75 - cy) * (0.45 + 0.8 * (1.0 - depth_mid))
            vertical = (0.5 - depth_mid) * 0.35 + (0.5 - cy) * 0.12
            base = anchor_vec + np.array([lateral, forward, vertical], dtype=np.float32)
            color = _priority_color(str(det.get("priority_label", "other")))

            points.append(base.tolist())
            colours.append(color)
            points.append((base + np.array([spread * 0.25, 0.0, 0.0], dtype=np.float32)).tolist())
            points.append((base + np.array([-spread * 0.25, 0.0, 0.0], dtype=np.float32)).tolist())
            colours.append(color * 0.9)
            colours.append(color * 0.9)

            if track_id is not None:
                track_key = f"{label}:{track_id}"
                track_points.setdefault(track_key, []).append(base)
                track_colors[track_key] = color

    for track_key, samples in track_points.items():
        if len(samples) < 2:
            continue
        centroid = np.mean(np.stack(samples, axis=0), axis=0)
        points.append(centroid.astype(np.float32).tolist())
        colours.append(track_colors.get(track_key, np.array([0.3, 0.9, 0.9], dtype=np.float32)))

    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32), np.asarray(colours, dtype=np.float32)


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
    depth_results: Optional[List[Dict[str, Any]]] = None,
    yolo_detection_results: Optional[List[Dict[str, Any]]] = None,
    tracking_results: Optional[Dict[str, Any]] = None,
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
    t0 = time.time()
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
    quality_note = ""

    if run_sfm_flag:
        logger.info("Running Structure-from-Motion (pycolmap) …")
        try:
            points3d, colours, sfm_poses, scene_count, sfm_frames, frame_positions = _sfm_point_cloud(
                video_path, video_id
            )
            if points3d is not None:
                method = "sfm"
                logger.info("SfM point cloud: %d camera centres", len(points3d))
                if len(points3d) < _MIN_USEFUL_MAP_POINTS or sfm_poses < _MIN_USEFUL_SFM_POSES:
                    logger.info(
                        "SfM map is sparse (%d points, %d poses); enriching sparse geometry for downstream artifacts",
                        len(points3d),
                        sfm_poses,
                    )
                    try:
                        interpolated_positions = _interpolate_frame_positions(frame_list, frame_positions)
                        enriched_points, enriched_colours = _build_semantic_enriched_point_cloud(
                            frame_positions=interpolated_positions or frame_positions,
                            yolo_detection_results=yolo_detection_results,
                            tracking_results=tracking_results,
                            depth_results=depth_results,
                        )
                        if len(enriched_points) >= max(len(points3d), len(frame_list)):
                            points3d, colours = enriched_points, enriched_colours
                            frame_positions = interpolated_positions or frame_positions
                            method = "sfm_sparse+semantic_pseudo3d"
                            quality_note = (
                                "SfM was too sparse; interpolated trajectory and semantic pseudo-3D anchors were used"
                            )
                        else:
                            raise RuntimeError("semantic enrichment did not add enough structure")
                    except Exception as enrich_exc:
                        logger.warning("Semantic enrichment after sparse SfM failed (%s); falling back to visual PCA", enrich_exc)
                        try:
                            pca_points, pca_colours, pca_method, pca_positions = _visual_pca_point_cloud(frame_list)
                            points3d, colours = pca_points, pca_colours
                            frame_positions = pca_positions
                            method = f"sfm_sparse+{pca_method}"
                            quality_note = "SfM was too sparse; PCA fallback geometry used"
                        except Exception as pca_exc:
                            logger.warning("PCA fallback after sparse SfM failed (%s); keeping sparse SfM map", pca_exc)
        except Exception as exc:
            logger.warning("SfM failed (%s) — using PCA fallback", exc)
    else:
        logger.info("SfM skipped — using PCA point-cloud fallback")

    if points3d is None:
        logger.info("Building fallback 3D point cloud …")
        try:
            pca_points, pca_colours, pca_method, pca_positions = _pca_point_cloud(frame_list, models)
            enriched_points, enriched_colours = _build_semantic_enriched_point_cloud(
                frame_positions=pca_positions,
                yolo_detection_results=yolo_detection_results,
                tracking_results=tracking_results,
                depth_results=depth_results,
            )
            if len(enriched_points) >= max(len(pca_points), len(frame_list)):
                points3d, colours, method, frame_positions = (
                    enriched_points,
                    enriched_colours,
                    "semantic_pseudo3d",
                    pca_positions,
                )
                quality_note = "No SfM poses were recovered; semantic pseudo-3D anchors were built on PCA trajectory"
            else:
                points3d, colours, method, frame_positions = pca_points, pca_colours, pca_method, pca_positions
        except Exception as exc:
            logger.warning("Embedding PCA fallback failed (%s); using CPU visual PCA fallback", exc)
            try:
                points3d, colours, method, frame_positions = _visual_pca_point_cloud(frame_list)
                quality_note = "No SfM poses were recovered; visual PCA fallback geometry used"
            except Exception as visual_exc:
                logger.warning("Visual PCA fallback failed (%s)", visual_exc)
                points3d = np.zeros((1, 3), dtype=np.float32)
                colours  = np.ones((1, 3),  dtype=np.float32)
                method   = "failed"
                frame_positions = []
                quality_note = "3D map construction failed; generated placeholder geometry"

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
        "quality_degraded": bool(
            len(points3d) < _MIN_USEFUL_MAP_POINTS
            or sfm_poses < _MIN_USEFUL_SFM_POSES
            or str(method).startswith("sfm_sparse+")
        ),
        "quality_note": (
            quality_note
            or (
                "SfM was too sparse; PCA fallback geometry used"
                if str(method).startswith("sfm_sparse+")
                else ""
            )
        ),
        "elapsed_sec":   round(time.time() - t0, 3),
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
        "quality_note": (
            quality_note
            or (
                "SfM was too sparse; PCA fallback geometry used"
                if str(method).startswith("sfm_sparse+")
                else ""
            )
        ),
        "quality_degraded": bool(
            len(points3d) < _MIN_USEFUL_MAP_POINTS
            or sfm_poses < _MIN_USEFUL_SFM_POSES
            or str(method).startswith("sfm_sparse+")
        ),
        "elapsed_sec":   time.time() - t0,
        "splat_ply":     splat_ply,
        "viewer_html":   viewer_html,
        "gsplat_method": gsplat_method,
    }
