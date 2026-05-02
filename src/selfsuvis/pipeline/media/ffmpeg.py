import os
from typing import List, Tuple

from selfsuvis.pipeline.core import ensure_dir, get_logger, settings
from selfsuvis.pipeline.media.subprocess_common import run_checked


def _frame_dir(video_id: str) -> str:
    return os.path.join(settings.FRAMES_DIR, video_id)


def _frame_pattern(out_dir: str) -> str:
    return os.path.join(out_dir, "frame_%010d.jpg")


def _frame_paths(out_dir: str, fps: float) -> List[Tuple[str, float]]:
    frames = sorted(fname for fname in os.listdir(out_dir) if fname.startswith("frame_"))
    return [(os.path.join(out_dir, fname), idx / fps) for idx, fname in enumerate(frames)]


def extract_frames(video_path: str, video_id: str) -> List[Tuple[str, float]]:
    logger = get_logger(__name__)
    out_dir = _frame_dir(video_id)
    ensure_dir(out_dir)
    pattern = _frame_pattern(out_dir)
    fps = settings.SAMPLE_FPS_MAX
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",       # suppress info/warnings; keep real errors
        "-i", video_path,
        "-vf", f"fps={fps},format=yuv420p",  # explicit format avoids deprecated yuvj420p auto-select
        "-color_range", "2",        # full (pc) range — required for MJPEG/JPEG output
        "-q:v", "2",
        pattern,
    ]
    logger.info("Running ffmpeg for video_id=%s", video_id)
    run_checked(cmd, timeout=settings.FFMPEG_TIMEOUT_SEC)
    return _frame_paths(out_dir, fps)
