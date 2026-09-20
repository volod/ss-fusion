import hashlib
import time
from datetime import UTC, datetime
from typing import Any

from selfsuvis.pipeline.core.config import settings
from ss_kit.paths import ensure_dir as ensure_kit_dir
from ss_kit.security import resolve_allowed_path as resolve_kit_path
from ss_kit.security import stable_point_id as stable_point_id


def ensure_dir(path: str) -> None:
    ensure_kit_dir(path)


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_utc_datetime(value: Any) -> datetime | None:
    """Normalise epoch/datetime values to timezone-aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    raise TypeError(f"Unsupported datetime value: {type(value)!r}")


def datetime_to_ts(value: Any) -> float | None:
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


def resolve_allowed_path(
    user_path: str, must_be_file: bool = False, must_be_dir: bool = False
) -> str | None:
    """
    Resolve user-supplied path against allowed base directories.
    Returns the resolved absolute path if allowed, else None.

    Fail-closed: returns None when ALLOWED_INDEX_PATHS is empty so that
    path-based endpoints are disabled rather than open to the whole filesystem.
    """
    return resolve_kit_path(
        user_path,
        settings.ALLOWED_INDEX_PATHS,
        must_be_file=must_be_file,
        must_be_dir=must_be_dir,
    )


def resolve_allowed_paths_for_walk(user_dir: str) -> str | None:
    """Resolve directory for os.walk. Returns None if not allowed."""
    return resolve_allowed_path(user_dir, must_be_dir=True)


class RateTimer:
    def __init__(self) -> None:
        self.start = time.time()
        self.count = 0

    def tick(self, n: int = 1) -> None:
        self.count += n

    def rate(self) -> float:
        elapsed = time.time() - self.start
        if elapsed <= 0:
            return 0.0
        return self.count / elapsed
