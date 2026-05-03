"""Audio extraction and subtitle-to-frame mapping utilities.

Extracts the audio track from a video file using ffmpeg (16 kHz mono WAV —
the format Whisper expects), then provides helpers to map time-stamped
transcript segments to frame timestamps.

Usage::

    from selfsuvis.pipeline.media.audio import extract_audio, map_subtitles_to_frames

    wav_path = extract_audio(video_path, output_dir)       # or None on failure
    subtitle_map = map_subtitles_to_frames(segments, frame_timestamps)
    # subtitle_map: {t_sec: "text spoken near this frame"}
"""

import os
import subprocess

from selfsuvis.pipeline.core import ensure_dir, get_logger, settings
from selfsuvis.pipeline.media.fs_common import output_path_with_suffix
from selfsuvis.pipeline.media.subprocess_common import run_captured

logger = get_logger(__name__)

# Window (±) around a frame timestamp used when matching subtitle segments.
# A value of 3.0 means we look 3 s before and 3 s after the frame time.
_DEFAULT_WINDOW_SEC = 3.0


# ── Audio extraction ──────────────────────────────────────────────────────────


def extract_audio(video_path: str, output_dir: str) -> str | None:
    """Extract the audio track from *video_path* into *output_dir* as a 16 kHz mono WAV.

    Returns the path to the WAV file on success, or ``None`` if the video has
    no audio track or ffmpeg fails.  The output filename mirrors the video
    basename with a ``.wav`` suffix.

    The 16 kHz / mono / PCM-s16le format matches Whisper's native input
    format and avoids any additional resampling by the ASR pipeline.
    """
    ensure_dir(output_dir)
    wav_path = output_path_with_suffix(video_path, output_dir, ".wav")

    # First, probe whether an audio stream is present.
    if not _has_audio_stream(video_path):
        logger.info("audio_extractor: no audio stream in %s — ASR skipped", video_path)
        return None

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",                          # drop video
        "-acodec", "pcm_s16le",         # uncompressed 16-bit PCM
        "-ar", "16000",                 # 16 kHz
        "-ac", "1",                     # mono
        wav_path,
    ]
    try:
        result = run_captured(cmd, timeout=settings.FFMPEG_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        logger.warning("audio_extractor: ffmpeg timed out extracting %s", video_path)
        return None
    except FileNotFoundError:
        logger.warning("audio_extractor: ffmpeg not found; ASR unavailable")
        return None

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:300]
        logger.warning("audio_extractor: ffmpeg failed for %s: %s", video_path, stderr)
        return None

    size = os.path.getsize(wav_path)
    if size < 1024:
        # Suspiciously small — probably an empty audio track
        logger.debug("audio_extractor: extracted WAV is nearly empty (%d bytes)", size)
        return None

    logger.info("audio_extractor: extracted %s → %s (%.1f MB)", video_path, wav_path, size / 1e6)
    return wav_path


def _has_audio_stream(video_path: str) -> bool:
    """Return True if ffprobe reports at least one audio stream."""
    try:
        result = run_captured(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=nw=1",
                video_path,
            ],
            timeout=30,
            text=True,
        )
        return "audio" in result.stdout.lower()
    except Exception:
        return True  # assume audio is present if probe fails


# ── Subtitle ↔ frame mapping ──────────────────────────────────────────────────


def map_subtitles_to_frames(
    segments: list[dict],
    frame_timestamps: list[float],
    window_sec: float = _DEFAULT_WINDOW_SEC,
) -> dict[float, str]:
    """Map Whisper transcript segments to video frame timestamps.

    Parameters
    ----------
    segments:
        List of dicts with at minimum ``{"text": str, "timestamp": (start, end)}``,
        which is the format returned by HuggingFace's ASR pipeline with
        ``return_timestamps=True``.  Also accepts ``{"text": str, "start": float,
        "end": float}`` (e.g. from faster-whisper).
    frame_timestamps:
        Sorted list of frame t_sec values from the indexing pass.
    window_sec:
        Half-width of the time window around each frame.  A segment is
        included if its interval overlaps [t_sec - window_sec, t_sec + window_sec].

    Returns
    -------
    Dict[float, str]
        Mapping ``t_sec → subtitle_text``.  Frames with no nearby transcript
        text are absent from the dict (callers should use ``.get()``).
    """
    # Normalise segment dicts to {text, start, end}
    normalised = _normalise_segments(segments)
    if not normalised:
        return {}

    result: dict[float, str] = {}
    for t_sec in frame_timestamps:
        lo = t_sec - window_sec
        hi = t_sec + window_sec
        texts = []
        for seg in normalised:
            seg_start = seg["start"]
            seg_end = seg["end"]
            if seg_end >= lo and seg_start <= hi:
                text = seg["text"].strip()
                if text:
                    texts.append(text)
        if texts:
            result[t_sec] = " ".join(texts)

    return result


def _normalise_segments(raw: list[dict]) -> list[dict]:
    """Normalise various ASR segment formats to ``{text, start, end}``."""
    out = []
    for seg in raw:
        text = seg.get("text", "").strip()
        if not text:
            continue
        # HuggingFace pipeline format: {"timestamp": (start, end), "text": ...}
        ts = seg.get("timestamp")
        if ts is not None and isinstance(ts, (tuple, list)) and len(ts) == 2:
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start + 3.0
            out.append({"text": text, "start": start, "end": end})
            continue
        # faster-whisper / direct format: {"start": float, "end": float, "text": ...}
        if "start" in seg and "end" in seg:
            out.append({
                "text": text,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            })
            continue
        # Fallback: no timing info — skip segment
        logger.debug("audio_extractor: unrecognised segment format; skipping: %r", seg)
    return out
