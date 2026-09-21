"""Persist contract camera and sensor events into fusion site_events."""

from datetime import UTC, datetime
from typing import Any

from selfsuvis.fusion_rt.config import fusion_settings
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.storage.common import jsonb

logger = get_logger(__name__)

VALID_MODALITIES = frozenset({"camera", "audio", "rf", "thermal", "vibration", "custom"})

_SENSOR_TYPE_MODALITY = {
    "camera": "camera",
    "audio": "audio",
    "acoustic": "audio",
    "rf": "rf",
    "thermal": "thermal",
    "vibration": "vibration",
    "lorawan": "custom",
}


def _as_mapping(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _clamp_confidence(value: Any, default: float = 0.7) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def sensor_event_to_site_row(event: Any, *, zone_id: str) -> dict[str, Any]:
    data = _as_mapping(event)
    sensor_type = str(data.get("sensor_type") or "").strip().lower()
    modality = _SENSOR_TYPE_MODALITY.get(sensor_type, "custom")
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    confidence = _clamp_confidence(
        payload.get("confidence"), 0.85 if payload.get("motion") else 0.7
    )
    return {
        "ts": parse_ts(data.get("event_time")),
        "zone_id": zone_id,
        "sensor_id": str(data.get("node_id") or "unknown"),
        "modality": modality,
        "confidence": confidence,
        "payload": data,
        "artifact_uri": None,
    }


def camera_event_to_site_row(event: Any, *, zone_id: str) -> dict[str, Any]:
    data = _as_mapping(event)
    return {
        "ts": parse_ts(data.get("started_at")),
        "zone_id": zone_id,
        "sensor_id": str(data.get("camera") or "unknown"),
        "modality": "camera",
        "confidence": _clamp_confidence(data.get("score"), 0.5),
        "payload": data,
        "artifact_uri": None,
    }


async def persist_site_event(pool: Any, row: dict[str, Any]) -> None:
    """Insert one site_events row and ensure the zone exists."""
    if pool is None:
        return
    modality = row["modality"]
    if modality not in VALID_MODALITIES:
        logger.warning("skip persist: unknown modality %s", modality)
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO zones (zone_id, label)
            VALUES ($1, $2)
            ON CONFLICT (zone_id) DO NOTHING
            """,
            row["zone_id"],
            row["zone_id"],
        )
        await conn.execute(
            """
            INSERT INTO site_events
                (ts, zone_id, sensor_id, modality, confidence, payload, artifact_uri)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            """,
            row["ts"],
            row["zone_id"],
            row["sensor_id"],
            row["modality"],
            row["confidence"],
            jsonb(row["payload"], default={}),
            row["artifact_uri"],
        )


class SiteEventPersister:
    """Write MQTT camera and sensor events into site_events for the correlator."""

    def __init__(self, pool: Any, zone_id: str | None = None) -> None:
        self._pool = pool
        self._zone_id = zone_id or fusion_settings.SITE_ID

    async def on_sensor_event(self, event: Any) -> None:
        try:
            await persist_site_event(
                self._pool, sensor_event_to_site_row(event, zone_id=self._zone_id)
            )
        except Exception:
            logger.exception("persist sensor-event failed")

    async def on_camera_event(self, event: Any) -> None:
        try:
            await persist_site_event(
                self._pool, camera_event_to_site_row(event, zone_id=self._zone_id)
            )
        except Exception:
            logger.exception("persist camera-event failed")
