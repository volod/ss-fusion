"""Typed realtime event envelopes for streaming threat/runtime fusion."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar


def _to_iso8601(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp value is required")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    return dict(payload)


def _normalize_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


@dataclass
class _BaseEvent:
    event_time: str
    ingest_time: str
    node_id: str
    sensor_type: str
    sector_id: str
    payload: dict[str, Any]
    freshness_sec: float = 0.0

    event_kind: ClassVar[str] = "base"

    def __post_init__(self) -> None:
        self.event_time = _to_iso8601(self.event_time)
        self.ingest_time = _to_iso8601(self.ingest_time)
        self.node_id = _normalize_text(self.node_id, field_name="node_id")
        self.sensor_type = _normalize_text(self.sensor_type, field_name="sensor_type")
        self.sector_id = str(self.sector_id or "").strip() or "unknown"
        self.payload = _normalize_payload(self.payload)
        self.freshness_sec = max(0.0, float(self.freshness_sec or 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_kind": self.event_kind,
            "event_time": self.event_time,
            "ingest_time": self.ingest_time,
            "node_id": self.node_id,
            "sensor_type": self.sensor_type,
            "sector_id": self.sector_id,
            "payload": dict(self.payload),
            "freshness_sec": float(self.freshness_sec),
        }


@dataclass
class SensorEvent(_BaseEvent):
    event_kind: ClassVar[str] = "sensor"


@dataclass
class ThreatEvent(_BaseEvent):
    event_kind: ClassVar[str] = "threat"


@dataclass
class NodeHealthEvent(_BaseEvent):
    event_kind: ClassVar[str] = "node_health"
