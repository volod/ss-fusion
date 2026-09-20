"""Combined camera + sensor site snapshot for fusion-rt.

Sensor rows are rebuilt from contract ``sensor-state`` (and ``sensor-event``)
messages published by ss-sens. Camera rows are a rolling window of
``camera-event`` messages the video side publishes on the topic map.
"""

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ss_contracts.models import SensorState


class CameraEventSummary(BaseModel):
    """Recent detections from one Frigate camera."""

    camera: str
    last_seen: datetime
    recent_detections: list[dict[str, Any]] = Field(default_factory=list)
    active_labels: list[str] = Field(default_factory=list)
    total_events: int = 0
    session_id: str | None = None
    rtsp_url: str | None = None


class CombinedSiteState(BaseModel):
    """Snapshot of sensors (from the mesh) plus local camera detections."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sensors: list[SensorState] = Field(default_factory=list)
    cameras: list[CameraEventSummary] = Field(default_factory=list)
    sensor_count: int = 0
    camera_count: int = 0
    active_motion: bool = False


def _as_mapping(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="python")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)


def _parse_time(value: Any) -> datetime:
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


class CombinedSiteSnapshot:
    """Merge contract sensor state with a rolling camera-event window."""

    def __init__(self, camera_window_sec: int | None = None) -> None:
        self._lock = asyncio.Lock()
        self._camera_window_sec = int(camera_window_sec if camera_window_sec is not None else 120)
        self._sensors: dict[str, SensorState] = {}
        self._cameras: dict[str, deque[Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}

    def set_camera_sessions(self, sessions: list[dict[str, Any]]) -> None:
        """Merge RTSP-bridge session metadata into /site/cameras."""
        self._sessions = {
            str(item.get("camera", "")): item for item in sessions if item.get("camera")
        }

    async def ingest_sensor_state(self, state: Any) -> None:
        data = _as_mapping(state)
        message = SensorState.model_validate(data)
        async with self._lock:
            self._sensors[message.dev_eui] = message

    async def ingest_sensor_event(self, event: Any) -> None:
        """Upsert a sensor row from a ``sensor-event`` when retained state has not arrived yet."""
        data = _as_mapping(event)
        node_id = str(data.get("node_id") or "").strip()
        if not node_id:
            return
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        event_time = _parse_time(data.get("event_time"))
        fields = {
            "dev_eui": node_id,
            "last_seen": event_time,
            "reading_count": 1,
            "temperature_c": payload.get("temperature_c"),
            "humidity_pct": payload.get("humidity_pct"),
            "co2_ppm": payload.get("co2_ppm"),
            "pressure_hpa": payload.get("pressure_hpa"),
            "battery_v": payload.get("battery_v"),
            "motion": payload.get("motion"),
            "rssi": payload.get("rssi"),
            "snr": payload.get("snr"),
        }
        message = SensorState.model_validate({k: v for k, v in fields.items() if v is not None})
        async with self._lock:
            existing = self._sensors.get(node_id)
            if existing is not None:
                merged = existing.model_dump()
                merged.update({k: v for k, v in message.model_dump(exclude_none=True).items()})
                merged["reading_count"] = max(int(existing.reading_count or 1), 1)
                self._sensors[node_id] = SensorState.model_validate(merged)
            else:
                self._sensors[node_id] = message

    async def ingest_camera_event(self, event: Any) -> None:
        camera = str(getattr(event, "camera", "") or "unknown")
        started_at = _parse_time(getattr(event, "started_at", None))

        class _Row:
            pass

        row = event
        if not hasattr(event, "started_at") or not isinstance(
            getattr(event, "started_at"), datetime
        ):
            row = _Row()
            row.event_id = getattr(event, "event_id", "")
            row.camera = camera
            row.label = getattr(event, "label", "")
            row.score = float(getattr(event, "score", 0.0) or 0.0)
            row.started_at = started_at
            row.has_snapshot = bool(getattr(event, "has_snapshot", False))
        async with self._lock:
            if camera not in self._cameras:
                self._cameras[camera] = deque()
            self._cameras[camera].append(row)
            cutoff = datetime.now(UTC) - timedelta(seconds=self._camera_window_sec)
            queue = self._cameras[camera]
            while queue and _parse_time(queue[0].started_at) < cutoff:
                queue.popleft()

    async def get_state(self) -> CombinedSiteState:
        async with self._lock:
            sensors = list(self._sensors.values())
            cameras = [
                self._summarize_camera(camera, queue)
                for camera, queue in self._cameras.items()
                if queue
            ]
        for row in cameras:
            session = self._sessions.get(row.camera) or {}
            if session.get("session_id"):
                row.session_id = session.get("session_id")
            if session.get("rtsp_url"):
                row.rtsp_url = session.get("rtsp_url")
        active_motion = any(s.motion for s in sensors if s.motion)
        return CombinedSiteState(
            sensors=sensors,
            cameras=cameras,
            sensor_count=len(sensors),
            camera_count=len(cameras),
            active_motion=active_motion,
        )

    @staticmethod
    def _summarize_camera(camera: str, queue: "deque[Any]") -> CameraEventSummary:
        events = list(queue)
        labels = list({e.label for e in events})
        recent = [
            {
                "event_id": e.event_id,
                "label": e.label,
                "score": e.score,
                "started_at": _parse_time(e.started_at).isoformat(),
                "has_snapshot": e.has_snapshot,
            }
            for e in sorted(events, key=lambda item: _parse_time(item.started_at), reverse=True)[
                :10
            ]
        ]
        return CameraEventSummary(
            camera=camera,
            last_seen=max(_parse_time(e.started_at) for e in events),
            recent_detections=recent,
            active_labels=labels,
            total_events=len(events),
        )
