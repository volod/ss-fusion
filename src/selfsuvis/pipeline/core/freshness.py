"""Freshness scoring and staleness decay helpers for realtime events."""

from datetime import datetime, timezone
from typing import Any


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def freshness_seconds(event_time: Any, ingest_time: Any) -> float:
    delta = (_parse_time(ingest_time) - _parse_time(event_time)).total_seconds()
    return max(0.0, float(delta))


def staleness_weight(
    freshness_sec: float,
    *,
    soft_limit_sec: float = 10.0,
    hard_expiry_sec: float = 60.0,
) -> float:
    freshness = max(0.0, float(freshness_sec or 0.0))
    if freshness >= hard_expiry_sec:
        return 0.0
    if freshness <= soft_limit_sec:
        return 1.0
    span = max(1e-6, hard_expiry_sec - soft_limit_sec)
    return max(0.0, 1.0 - ((freshness - soft_limit_sec) / span))


def expire_event(
    event: dict[str, Any],
    *,
    hard_expiry_sec: float = 60.0,
) -> bool:
    return float(event.get("freshness_sec", 0.0) or 0.0) >= hard_expiry_sec


def apply_freshness(event: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(event)
    enriched["freshness_sec"] = round(
        freshness_seconds(enriched.get("event_time"), enriched.get("ingest_time")),
        4,
    )
    return enriched


def downweight_score(
    score: float,
    freshness_sec: float,
    *,
    soft_limit_sec: float = 10.0,
    hard_expiry_sec: float = 60.0,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(score or 0.0)
            * staleness_weight(
                freshness_sec,
                soft_limit_sec=soft_limit_sec,
                hard_expiry_sec=hard_expiry_sec,
            ),
        ),
    )
