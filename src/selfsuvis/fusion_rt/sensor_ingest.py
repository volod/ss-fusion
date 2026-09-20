"""Bridge contract sensor events and camera detections into the realtime threat pipeline.

``SensorEvent`` messages arrive over MQTT from ss_sens. ``CameraEvent`` objects are
produced locally from Frigate. Sector IDs for LoRaWAN already sit on the contract
event; camera threats default to sector ``unknown`` unless a camera-sector map is
supplied.
"""

from datetime import datetime, timezone
from typing import Any

from .events import ThreatEvent

_SENSOR_KIND = "sensor"


def _payload_from_attrs(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields:
        value = getattr(obj, field, None)
        if value is not None:
            payload[field] = value
    return payload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_event_dict(event: Any) -> dict[str, Any]:
    if hasattr(event, "to_dict"):
        data = event.to_dict()
    elif hasattr(event, "model_dump"):
        data = event.model_dump(mode="json")
    else:
        data = dict(event)
    if "event_kind" not in data:
        data["event_kind"] = _SENSOR_KIND
    return data


def camera_event_to_threat(event: Any) -> ThreatEvent | None:
    """Convert a ``CameraEvent`` to a ``ThreatEvent`` if it carries detections.

    Returns ``None`` for low-confidence events so the caller can skip them.
    """
    label: str = getattr(event, "label", "") or ""
    score: float = float(getattr(event, "score", 0.0) or 0.0)

    if not label and score < 0.05:
        return None

    threat_score = min(1.0, score)

    payload = {
        **_payload_from_attrs(event, ("region",)),
        "threat_type": "camera_detection",
        "score": threat_score,
        "label": label,
        "camera": getattr(event, "camera", "unknown"),
    }

    return ThreatEvent(
        event_time=getattr(event, "started_at", None) or _now_iso(),
        ingest_time=_now_iso(),
        node_id=f"frigate:{getattr(event, 'camera', 'unknown')}",
        sensor_type="camera",
        sector_id="unknown",
        payload=payload,
    )


class SensorEventIngestor:
    """Feed contract sensor events and camera detections into a threat aggregator.

    Args:
        threat_aggregator: ``RealtimeThreatAggregator`` instance to receive events.
        camera_sector_map: Optional mapping from camera name to sector_id so
                           Frigate cameras can be placed on the threat grid.
    """

    def __init__(
        self,
        threat_aggregator: Any,
        camera_sector_map: dict[str, str] | None = None,
    ) -> None:
        self._agg = threat_aggregator
        self._cam_sectors = camera_sector_map or {}

    async def on_sensor_event(self, event: Any) -> None:
        try:
            self._agg.consume(_as_event_dict(event))
        except Exception:
            pass

    async def on_camera_event(self, event: Any) -> None:
        try:
            threat = camera_event_to_threat(event)
            if threat is None:
                return
            cam = getattr(event, "camera", "")
            if cam in self._cam_sectors:
                threat.sector_id = self._cam_sectors[cam]
            self._agg.consume(threat.to_dict())
        except Exception:
            pass
