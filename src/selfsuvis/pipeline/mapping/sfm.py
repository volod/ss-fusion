"""Structure-from-Motion via pycolmap.

Runs pycolmap incremental mapping on a dense set of frames extracted at SFM_FPS
(default 2 fps — separate from the sparse search keyframes).  Writes a
pose_json dict per frame into the returned results list.

Camera model is controlled by PYCOLMAP_CAMERA_MODEL (default: SIMPLE_RADIAL).

The dense frame extraction uses a dedicated subdirectory
  frames/{video_id}_sfm/
so it does not pollute the sparse search keyframes in frames/{video_id}/.

Usage (called by pipeline/indexer.py after Pass A extraction):

    from selfsuvis.pipeline.mapping.sfm import run_sfm
    results = run_sfm(video_path, video_id, mission_id)
    # results: List[{"frame_path": str, "t_sec": float,
    #                "pose_json": dict | None, "pose_status": str}]

pose_status values:
    "success"  — pycolmap produced a pose for this frame
    "failed"   — registered frame but no valid pose
    "skipped"  — frame was not registered by pycolmap
"""
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from selfsuvis.pipeline.core import ensure_dir, get_logger, settings

logger = get_logger(__name__)
_MIN_SFM_FRAMES = 3

_COLMAP_CAMERA_MODELS = frozenset(
    {"SIMPLE_RADIAL", "RADIAL", "PINHOLE", "OPENCV", "FULL_OPENCV", "SIMPLE_PINHOLE"}
)


def _validate_camera_model(model: str) -> str:
    """Return a validated COLMAP camera model string, falling back to SIMPLE_RADIAL."""
    if model in _COLMAP_CAMERA_MODELS:
        return model
    logger.warning(
        "Unknown PYCOLMAP_CAMERA_MODEL %r; falling back to SIMPLE_RADIAL", model
    )
    return "SIMPLE_RADIAL"


def _extract_dense_frames(
    video_path: str, video_id: str, fps: float
) -> List[str]:
    """Extract dense frames for SfM into frames/{video_id}_sfm/.

    Returns sorted list of absolute frame paths.
    """
    out_dir = os.path.join(settings.FRAMES_DIR, f"{video_id}_sfm")
    ensure_dir(out_dir)
    pattern = os.path.join(out_dir, "frame_%010d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps},format=yuv420p",
        "-color_range", "2",
        "-q:v", "2",
        pattern,
    ]
    logger.info("SfM dense extraction: video_id=%s fps=%.1f", video_id, fps)
    subprocess.run(cmd, check=True, timeout=settings.FFMPEG_TIMEOUT_SEC)
    frames = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("frame_")
    )
    logger.info("SfM dense extraction complete: %d frames", len(frames))
    return frames


def _run_pycolmap(
    image_dir: str,
    output_dir: str,
    camera_model: str,
) -> List[Any]:
    """Run pycolmap incremental mapping.

    Returns a list of pycolmap Reconstruction objects sorted by size (largest first),
    one per connected component. Returns an empty list on failure or if pycolmap is
    not installed.

    pycolmap is an optional dependency; if not installed, logs a warning and
    returns [] (all frames get pose_status='failed').
    """
    try:
        import pycolmap  # type: ignore
    except ImportError:
        logger.warning(
            "pycolmap is not installed; SfM will be skipped. "
            "Install with: pip install pycolmap"
        )
        return []

    ensure_dir(output_dir)
    database_path = os.path.join(output_dir, "database.db")

    try:
        _configure_pycolmap_logging(pycolmap)

        # Feature extraction
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = camera_model
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.max_image_size = settings.PYCOLMAP_MAX_IMAGE_SIZE
        extraction_options.num_threads = settings.PYCOLMAP_NUM_THREADS

        camera_mode = (
            pycolmap.CameraMode.SINGLE
            if settings.PYCOLMAP_SINGLE_CAMERA
            else pycolmap.CameraMode.AUTO
        )
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=camera_mode,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )

        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.num_threads = settings.PYCOLMAP_NUM_THREADS
        verification_options = pycolmap.TwoViewGeometryOptions()
        verification_options.min_num_inliers = max(15, settings.PYCOLMAP_INIT_MIN_NUM_INLIERS // 2)

        matching_mode = settings.PYCOLMAP_MATCHING.strip().lower()
        if matching_mode == "exhaustive":
            pycolmap.match_exhaustive(
                database_path,
                matching_options=matching_options,
                verification_options=verification_options,
            )
        else:
            pairing_options = pycolmap.SequentialPairingOptions()
            pairing_options.overlap = settings.PYCOLMAP_SEQUENTIAL_OVERLAP
            pairing_options.quadratic_overlap = True
            pairing_options.loop_detection = False
            pairing_options.num_threads = settings.PYCOLMAP_NUM_THREADS
            pycolmap.match_sequential(
                database_path,
                matching_options=matching_options,
                pairing_options=pairing_options,
                verification_options=verification_options,
            )

        # Incremental reconstruction — may produce multiple disconnected components
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.num_threads = settings.PYCOLMAP_NUM_THREADS
        pipeline_options.multiple_models = True
        pipeline_options.min_model_size = 5
        pipeline_options.min_num_matches = 15
        pipeline_options.mapper.num_threads = settings.PYCOLMAP_NUM_THREADS
        pipeline_options.mapper.init_min_num_inliers = settings.PYCOLMAP_INIT_MIN_NUM_INLIERS
        pipeline_options.mapper.init_min_tri_angle = settings.PYCOLMAP_INIT_MIN_TRI_ANGLE
        pipeline_options.mapper.init_max_forward_motion = settings.PYCOLMAP_INIT_MAX_FORWARD_MOTION
        pipeline_options.mapper.abs_pose_min_inlier_ratio = settings.PYCOLMAP_ABS_POSE_MIN_INLIER_RATIO
        pipeline_options.mapper.abs_pose_min_num_inliers = max(
            20, settings.PYCOLMAP_INIT_MIN_NUM_INLIERS // 2
        )
        pipeline_options.mapper.filter_min_tri_angle = min(
            1.5, settings.PYCOLMAP_INIT_MIN_TRI_ANGLE
        )
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=output_dir,
            options=pipeline_options,
        )
        if not maps:
            logger.warning("pycolmap returned no reconstructions for %s", image_dir)
            return []

        # Sort by number of registered images (largest component first)
        reconstructions = sorted(
            maps.values(), key=lambda r: r.num_reg_images(), reverse=True
        )
        logger.info(
            "pycolmap: %d connected component(s), sizes=%s",
            len(reconstructions),
            [r.num_reg_images() for r in reconstructions],
        )
        return reconstructions
    except Exception as exc:
        logger.error("pycolmap failed: %s", exc)
        return []


def _configure_pycolmap_logging(pycolmap) -> None:
    try:
        pycolmap.logging.minloglevel = settings.PYCOLMAP_MIN_LOG_LEVEL
        pycolmap.logging.logtostderr = False
        pycolmap.logging.logtostdout = False
        pycolmap.logging.alsologtostderr = False
    except Exception:
        pass


def _pose_to_dict(image) -> Dict[str, Any]:
    """Convert a pycolmap Image object to a serialisable dict."""
    cam_from_world = image.cam_from_world()
    R = cam_from_world.rotation.matrix().tolist()
    t = cam_from_world.translation.tolist()
    return {
        "R": R,
        "t": t,
        "camera_id": image.camera_id,
        "image_name": image.name,
    }


def run_sfm(
    video_path: str,
    video_id: str,
    mission_id: str,
) -> Dict[str, Any]:
    """Run the full SfM pipeline for a mission.

    1. Extract dense frames at SFM_FPS.
    2. Run pycolmap incremental mapping (may produce multiple connected components).
    3. Map poses back to frame paths by image name.
    4. Assign scene_index to each frame based on which component it belongs to.

    Args:
        video_path: Absolute path to the source video.
        video_id:   Unique video identifier (used for frame output dir).
        mission_id: Mission identifier (used for colmap output dir).

    Returns:
        {
            "frames": List of dicts per dense frame:
                        frame_path, t_sec, pose_json, pose_status, scene_index
            "scene_count": int  — number of disconnected SfM components (≥0)
        }
        scene_index is 0-based; None when pose_status != "success".
    """
    fps = settings.SFM_FPS
    camera_model = _validate_camera_model(settings.PYCOLMAP_CAMERA_MODEL)

    frame_paths = _extract_dense_frames(video_path, video_id, fps)
    if not frame_paths:
        logger.warning("SfM: no frames extracted for video_id=%s", video_id)
        return {"frames": [], "scene_count": 0}
    if len(frame_paths) < _MIN_SFM_FRAMES:
        logger.info(
            "SfM skipped for %s: need at least %d dense frames, got %d. Using PCA fallback.",
            mission_id,
            _MIN_SFM_FRAMES,
            len(frame_paths),
        )
        results = [
            {
                "frame_path": frame_path,
                "t_sec": idx / fps,
                "pose_json": None,
                "pose_status": "failed",
                "scene_index": None,
            }
            for idx, frame_path in enumerate(frame_paths)
        ]
        logger.info(
            "SfM complete: mission=%s frames=%d poses=%d/%d scenes=%d",
            mission_id,
            len(results),
            0,
            len(results),
            0,
        )
        return {"frames": results, "scene_count": 0}

    image_dir = os.path.dirname(frame_paths[0])
    output_dir = os.path.join(settings.MAPS_DIR, mission_id, "colmap")

    reconstructions = _run_pycolmap(image_dir, output_dir, camera_model)

    # Build name → (scene_index, Image) lookup across all components
    name_to_scene: Dict[str, Any] = {}
    for scene_idx, recon in enumerate(reconstructions):
        for img in recon.images.values():
            name_to_scene[img.name] = (scene_idx, img)

    sfm_failed = len(reconstructions) == 0

    results: List[Dict[str, Any]] = []
    for idx, frame_path in enumerate(frame_paths):
        t_sec = idx / fps
        fname = os.path.basename(frame_path)

        entry = name_to_scene.get(fname)
        if entry is not None:
            scene_idx, image = entry
            pose_json = _pose_to_dict(image)
            pose_status = "success"
        elif sfm_failed:
            scene_idx = None
            pose_json = None
            pose_status = "failed"
        else:
            scene_idx = None
            pose_json = None
            pose_status = "skipped"

        results.append(
            {
                "frame_path": frame_path,
                "t_sec": t_sec,
                "pose_json": pose_json,
                "pose_status": pose_status,
                "scene_index": scene_idx,
            }
        )

    success_count = sum(1 for r in results if r["pose_status"] == "success")
    scene_count = len(reconstructions)
    logger.info(
        "SfM complete: mission=%s frames=%d poses=%d/%d scenes=%d",
        mission_id, len(results), success_count, len(results), scene_count,
    )
    return {"frames": results, "scene_count": scene_count}
