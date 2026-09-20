"""GPS metadata extraction from drone video files.

Priority order:
1. SRT sidecar file   — DJI .srt (same basename, .srt extension); GPS_SIDECAR_PATH overrides path.
2. ffprobe atoms      — MP4 ISO 6709 location atom (single fix, expanded to all frames).
3. GPMF binary stream — GoPro Max/Hero telemetry (detection only; full parse deferred to v2).
4. Null fallback      — returns None for all frames with a warning.

Output: List[Optional[dict]] matching len(frame_timestamps_ms).
Each item is {lat, lon, alt, timestamp_ms} or None.
"""

import json
import os
import re
import subprocess
from typing import Any

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.media.subprocess_common import run_captured

logger = get_logger(__name__)

# Matches "lat : 47.5577" / "latitude: 47.5577" / "Lattitude(N): 47.5577"
_LAT_RE = re.compile(r"latt?itude\s*[:(N]\s*([+-]?\d+\.\d+)", re.IGNORECASE)
_LON_RE = re.compile(r"longit?ude\s*[:(EW]\s*([+-]?\d+\.\d+)", re.IGNORECASE)
_ALT_RE = re.compile(r"alt(?:itude)?\s*[:(m]\s*([+-]?\d+\.?\d*)", re.IGNORECASE)


def _run_ffprobe(video_path: str, args: list[str], timeout: int = 30) -> str | None:
    """Run ffprobe and return stdout, or None on error/timeout."""
    cmd = ["ffprobe", "-v", "quiet"] + args + [video_path]
    try:
        result = run_captured(cmd, timeout=timeout, text=True)
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _extract_from_ffprobe_atoms(video_path: str) -> dict[str, float] | None:
    """Extract a single GPS fix from the MP4 location atom.

    Returns {lat, lon, alt} or None.
    """
    out = _run_ffprobe(video_path, ["-print_format", "json", "-show_format"])
    if not out:
        return None
    try:
        tags = json.loads(out).get("format", {}).get("tags", {})
        location = tags.get("com.apple.quicktime.location.ISO6709", "") or tags.get("location", "")
        if not location:
            return None
        # ISO 6709 annex H: +47.5577+008.4697+482.123/ or ±lat±lon±alt/
        m = re.match(r"([+-]\d+\.\d+)([+-]\d+\.\d+)([+-]\d+\.?\d*)?", location)
        if not m:
            return None
        return {
            "lat": float(m.group(1)),
            "lon": float(m.group(2)),
            "alt": float(m.group(3)) if m.group(3) else 0.0,
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _parse_srt_file(srt_path: str) -> list[dict[str, Any]]:
    """Parse a DJI SRT sidecar file.

    Returns list of {lat, lon, alt, timestamp_ms} sorted by timestamp_ms.
    Returns [] if the file is missing or contains no GPS data.
    """
    try:
        with open(srt_path, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError:
        return []

    records: list[dict[str, Any]] = []
    # SRT blocks separated by blank lines
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # lines[0] = sequence number, lines[1] = timecode, lines[2..] = text
        text = "\n".join(lines[2:])

        lat_m = _LAT_RE.search(text)
        lon_m = _LON_RE.search(text)
        alt_m = _ALT_RE.search(text)
        if not (lat_m and lon_m):
            continue

        # Parse start timecode "HH:MM:SS,mmm --> ..."
        tc_m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not tc_m:
            continue
        h, mn, s, ms_str = tc_m.groups()
        # Normalise sub-second part to milliseconds (handle 1–3 digit fractional)
        ms_part = int(ms_str.ljust(3, "0")[:3])
        timestamp_ms = (int(h) * 3600 + int(mn) * 60 + int(s)) * 1000 + ms_part

        records.append(
            {
                "lat": float(lat_m.group(1)),
                "lon": float(lon_m.group(1)),
                "alt": float(alt_m.group(1)) if alt_m else 0.0,
                "timestamp_ms": float(timestamp_ms),
            }
        )

    return sorted(records, key=lambda r: r["timestamp_ms"])


def _extract_from_srt(video_path: str) -> list[dict[str, Any]]:
    """Return GPS records from an SRT sidecar, or [] if none found."""
    srt_path = settings.GPS_SIDECAR_PATH or (os.path.splitext(video_path)[0] + ".srt")
    if not os.path.isfile(srt_path):
        return []
    records = _parse_srt_file(srt_path)
    if records:
        logger.debug("GPS: %d fixes from SRT sidecar %s", len(records), srt_path)
    return records


def _extract_from_gpmf(video_path: str) -> list[dict[str, Any]]:
    """Detect GPMF telemetry stream (GoPro). Full parse deferred to v2.

    Returns [] always (detection only); logs a debug message if stream found.
    """
    out = _run_ffprobe(
        video_path,
        ["-print_format", "json", "-show_streams", "-select_streams", "data"],
    )
    if not out:
        return []
    try:
        streams = json.loads(out).get("streams", [])
    except json.JSONDecodeError:
        return []
    if any("GoPro" in str(s) or "gpmd" in str(s).lower() for s in streams):
        logger.debug(
            "GPS: GPMF stream detected in %s but full parsing not yet implemented; "
            "falling back to null.",
            video_path,
        )
    return []


def _interpolate_gps(
    records: list[dict[str, Any]],
    frame_timestamps_ms: list[float],
) -> list[dict[str, Any] | None]:
    """Linearly interpolate GPS records to match frame timestamps.

    Clamps to the first/last record for timestamps outside the recorded range.

    Args:
        records: GPS records sorted by timestamp_ms.
        frame_timestamps_ms: Target frame timestamps in milliseconds.

    Returns:
        List of {lat, lon, alt, timestamp_ms} (one per frame) or None per frame
        if records is empty.
    """
    if not records:
        return [None] * len(frame_timestamps_ms)

    rec_ts = [float(r["timestamp_ms"]) for r in records]
    result: list[dict[str, Any] | None] = []

    for ts in frame_timestamps_ms:
        ts = float(ts)
        if ts <= rec_ts[0]:
            result.append({**records[0], "timestamp_ms": ts})
            continue
        if ts >= rec_ts[-1]:
            result.append({**records[-1], "timestamp_ms": ts})
            continue

        # Binary search for surrounding records
        lo, hi = 0, len(rec_ts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if rec_ts[mid] <= ts:
                lo = mid
            else:
                hi = mid

        t0, t1 = rec_ts[lo], rec_ts[hi]
        r0, r1 = records[lo], records[hi]
        alpha = (ts - t0) / (t1 - t0) if t1 != t0 else 0.0

        result.append(
            {
                "lat": r0["lat"] + alpha * (r1["lat"] - r0["lat"]),
                "lon": r0["lon"] + alpha * (r1["lon"] - r0["lon"]),
                "alt": r0["alt"] + alpha * (r1["alt"] - r0["alt"]),
                "timestamp_ms": ts,
            }
        )

    return result


def extract_gps(
    video_path: str,
    frame_timestamps_ms: list[float],
) -> list[dict[str, Any] | None]:
    """Extract GPS and interpolate to frame timestamps.

    Priority: SRT sidecar → ffprobe atoms → GPMF → null fallback.

    Args:
        video_path: Path to the video file.
        frame_timestamps_ms: Frame timestamps in milliseconds (one per keyframe).

    Returns:
        List of {lat, lon, alt, timestamp_ms} or None, same length as frame_timestamps_ms.
    """
    # Priority 1: SRT sidecar (most common for DJI drones)
    records = _extract_from_srt(video_path)
    if records:
        return _interpolate_gps(records, frame_timestamps_ms)

    # Priority 2: ffprobe ISO 6709 atom (single fix → broadcast to all frames)
    single_fix = _extract_from_ffprobe_atoms(video_path)
    if single_fix:
        logger.debug(
            "GPS: single fix from ffprobe atom lat=%.5f lon=%.5f",
            single_fix["lat"],
            single_fix["lon"],
        )
        return [{**single_fix, "timestamp_ms": float(ts)} for ts in frame_timestamps_ms]

    # Priority 3: GPMF (GoPro) — detection only, falls through to null
    _extract_from_gpmf(video_path)

    # Priority 4: Null fallback
    logger.info(
        "GPS: no GPS data found for %s; all %d frames will have null GPS",
        video_path,
        len(frame_timestamps_ms),
    )
    return [None] * len(frame_timestamps_ms)
