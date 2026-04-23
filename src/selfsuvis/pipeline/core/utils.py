import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from selfsuvis.pipeline.core.config import settings


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Number of hex digits taken from the SHA-256 digest to form a 64-bit Qdrant point ID.
# Changing this value invalidates all existing Qdrant data — wipe and re-index.
_POINT_ID_HEX_DIGITS = 16


def stable_point_id(*parts: Any) -> int:
    # Uses SHA-256. Changing this function changes all Qdrant point IDs;
    # existing indexed data must be wiped and re-indexed after an upgrade.
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int(h.hexdigest()[:_POINT_ID_HEX_DIGITS], 16)


def now_ts() -> float:
    return time.time()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_datetime(value: Any) -> Optional[datetime]:
    """Normalise epoch/datetime values to timezone-aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    raise TypeError(f"Unsupported datetime value: {type(value)!r}")


def datetime_to_ts(value: Any) -> Optional[float]:
    """Return Unix timestamp seconds for datetime-like values."""
    dt = to_utc_datetime(value)
    return dt.timestamp() if dt is not None else None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_allowed_path(user_path: str, must_be_file: bool = False, must_be_dir: bool = False) -> Optional[str]:
    """
    Resolve user-supplied path against allowed base directories.
    Returns the resolved absolute path if allowed, else None.

    Fail-closed: returns None when ALLOWED_INDEX_PATHS is empty so that
    path-based endpoints are disabled rather than open to the whole filesystem.
    """
    allowed = settings.ALLOWED_INDEX_PATHS
    if not allowed:
        return None
    if isinstance(allowed, str):
        allowed = [p for p in allowed.split(",") if p.strip()]

    resolved = os.path.abspath(os.path.realpath(user_path))
    for base in allowed:
        base_abs = os.path.abspath(os.path.realpath(base))
        try:
            common = os.path.commonpath([base_abs, resolved])
            if common == base_abs:
                if must_be_file and not os.path.isfile(resolved):
                    return None
                if must_be_dir and not os.path.isdir(resolved):
                    return None
                return resolved
        except ValueError:
            continue
    return None


def resolve_allowed_paths_for_walk(user_dir: str) -> Optional[str]:
    """Resolve directory for os.walk. Returns None if not allowed."""
    return resolve_allowed_path(user_dir, must_be_dir=True)


def file_sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class RateTimer:
    def __init__(self):
        self.start = time.time()
        self.count = 0

    def tick(self, n: int = 1) -> None:
        self.count += n

    def rate(self) -> float:
        elapsed = time.time() - self.start
        if elapsed <= 0:
            return 0.0
        return self.count / elapsed
