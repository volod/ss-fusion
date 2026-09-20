"""FusionContractConsumer dispatches camera-event and sensor-event payloads."""

import json
from datetime import UTC, datetime

from selfsuvis.fusion_rt.mqtt_consumer import FusionContractConsumer
from ss_kit.mqtt import TopicBuilder


async def test_handle_message_dispatches_sensor_and_camera() -> None:
    events: list[object] = []
    cameras: list[object] = []

    async def on_event(model: object) -> None:
        events.append(model)

    async def on_camera(model: object) -> None:
        cameras.append(model)

    consumer = FusionContractConsumer(on_sensor_event=on_event, on_camera_event=on_camera)
    topics = TopicBuilder()
    event_topic = topics.build("sensor-event", site_id="local", node_id="70b3d57ed0060001")
    camera_topic = topics.build("camera-event", site_id="local", camera="entrance")
    event_body = {
        "event_kind": "sensor",
        "event_time": "2026-09-19T07:59:58.412345Z",
        "ingest_time": "2026-09-19T08:00:01.250000Z",
        "node_id": "70b3d57ed0060001",
        "sensor_type": "lorawan",
        "sector_id": "grid:50450:30523",
        "payload": {"temperature_c": 21.4},
        "freshness_sec": 0.0,
    }
    camera_body = {
        "event_id": "e1",
        "camera": "entrance",
        "label": "person",
        "score": 0.8,
        "top_score": 0.8,
        "event_type": "new",
        "started_at": datetime(2026, 9, 19, 8, 0, tzinfo=UTC).isoformat(),
        "has_snapshot": True,
        "has_clip": False,
        "region": {},
        "raw": {},
    }
    await consumer.handle_message(event_topic, json.dumps(event_body).encode("utf-8"))
    await consumer.handle_message(camera_topic, json.dumps(camera_body).encode("utf-8"))
    assert len(events) == 1
    assert getattr(events[0], "node_id") == "70b3d57ed0060001"
    assert len(cameras) == 1
    assert getattr(cameras[0], "camera") == "entrance"


async def test_handle_message_ignores_invalid_payload() -> None:
    seen: list[object] = []

    async def on_event(model: object) -> None:
        seen.append(model)

    consumer = FusionContractConsumer(on_sensor_event=on_event)
    topic = TopicBuilder().build("sensor-event", site_id="local", node_id="x")
    await consumer.handle_message(topic, b"not-json")
    assert seen == []
