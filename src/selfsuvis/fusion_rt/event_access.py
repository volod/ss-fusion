"""Shared access helpers for normalized realtime event envelopes."""

from typing import Any


def event_kind(event: dict[str, Any]) -> str:
    return str(event.get("event_kind", "")).strip().lower()


def event_sensor_type(event: dict[str, Any]) -> str:
    return str(event.get("sensor_type") or "").strip().lower()


def event_node_id(event: dict[str, Any], *, default: str = "unknown") -> str:
    text = str(event.get("node_id", default) or "").strip()
    return text or default


def event_sector_id(event: dict[str, Any], *, default: str = "unknown") -> str:
    text = str(event.get("sector_id", default) or "").strip()
    return text or default


def event_payload(event: dict[str, Any]) -> dict[str, Any]:
    return dict(event.get("payload") or {})


def event_freshness_sec(event: dict[str, Any]) -> float:
    return float(event.get("freshness_sec", 0.0) or 0.0)


def payload_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(payload.get(key, default) or default)


def payload_text(payload: dict[str, Any], key: str, default: str = "") -> str:
    return str(payload.get(key, default) or default)
