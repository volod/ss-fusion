import os
import subprocess
from typing import List, Tuple

from selfsuvis.pipeline.core import ensure_dir, get_logger, settings


def extract_frames(video_path: str, video_id: str) -> List[Tuple[str, float]]:
    logger = get_logger(__name__)
    out_dir = os.path.join(settings.FRAMES_DIR, video_id)
    ensure_dir(out_dir)
    pattern = os.path.join(out_dir, "frame_%010d.jpg")
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
    subprocess.run(cmd, check=True, timeout=settings.FFMPEG_TIMEOUT_SEC)

    frames = sorted([f for f in os.listdir(out_dir) if f.startswith("frame_")])
    frame_paths = []
    for idx, fname in enumerate(frames):
        t_sec = idx / fps
        frame_paths.append((os.path.join(out_dir, fname), t_sec))
    return frame_paths
