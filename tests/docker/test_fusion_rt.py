"""Fusion-rt Docker suite: health, HTTP ingest, MQTT uplink to incident.

Requires the slim Docker test stack (fusion-rt, postgres, redis, mosquitto).
Set RUN_API_TESTS=1 (the tests image does this).
"""

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import requests

FUSION_RT_URL = os.getenv("FUSION_RT_URL", "http://localhost:8001")
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SITE_ID = os.getenv("COOP_SITE_ID", "local")
ASSETS = Path(os.getenv("ASSETS_DIR", Path(__file__).resolve().parents[1] / "assets"))
RUN_API_TESTS = os.getenv("RUN_API_TESTS", "").lower() in {"1", "true", "yes"}

if not RUN_API_TESTS:
    pytest.skip(
        "API integration tests disabled; set RUN_API_TESTS=1 to run them",
        allow_module_level=True,
    )


def _headers() -> dict[str, str]:
    api_key = os.getenv("API_KEY", "")
    return {"X-Api-Key": api_key} if api_key else {}


def _wait_fusion(timeout_sec: int = 120) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(f"{FUSION_RT_URL}/health", timeout=2)
            body = resp.json() if resp.content else {}
            if resp.status_code == 200 and body.get("postgres") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope="session", autouse=True)
def wait_for_fusion_rt():
    if not _wait_fusion():
        pytest.skip("fusion-rt not reachable")


def _publish(topic: str, payload: dict) -> None:
    import aiomqtt

    async def _run() -> None:
        async with aiomqtt.Client(hostname=MQTT_HOST, port=MQTT_PORT) as client:
            await client.publish(topic, json.dumps(payload), qos=1)

    asyncio.run(_run())


def test_health_reports_postgres_and_redis() -> None:
    resp = requests.get(f"{FUSION_RT_URL}/health", timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert body["postgres"] == "ok"
    assert body["redis"] == "ok"
    assert "dlq_depth" in body


def test_http_camera_event_ingest() -> None:
    ts = datetime.now(UTC).isoformat()
    resp = requests.post(
        f"{FUSION_RT_URL}/api/v1/events/camera",
        json={
            "ts": ts,
            "zone_id": "http-zone",
            "sensor_id": "test-cam",
            "confidence": 0.8,
        },
        headers=_headers(),
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "event_id" in data
    assert data["zone_id"] == "http-zone"


def test_fixture_uplink_and_camera_event_create_incident() -> None:
    from ss_kit.mqtt import TopicBuilder

    now = datetime.now(UTC)
    now_iso = now.isoformat().replace("+00:00", "Z")
    zone_id = SITE_ID
    headers = _headers()

    zone_resp = requests.post(
        f"{FUSION_RT_URL}/api/v1/zones",
        json={"zone_id": zone_id, "label": "Local site"},
        headers=headers,
        timeout=10,
    )
    assert zone_resp.status_code in {201, 409}

    rule_resp = requests.post(
        f"{FUSION_RT_URL}/api/v1/rules",
        json={
            "rule_id": "fusion-rt-mqtt-gate",
            "label": "MQTT uplink plus camera",
            "modalities": ["camera", "custom"],
            "zone_id": zone_id,
            "window_s": 60,
            "min_confidence": 0.5,
            "enabled": True,
        },
        headers=headers,
        timeout=10,
    )
    assert rule_resp.status_code in {201, 409}

    sensor = json.loads((ASSETS / "contracts/golden/sensor-event/lorawan.json").read_text())
    sensor["event_time"] = now_iso
    sensor["ingest_time"] = now_iso
    camera = json.loads((ASSETS / "contracts/golden/camera-event/new.json").read_text())
    camera["started_at"] = now_iso

    topics = TopicBuilder()
    _publish(
        topics.build("sensor-event", site_id=SITE_ID, node_id=sensor["node_id"]),
        sensor,
    )
    _publish(
        topics.build("camera-event", site_id=SITE_ID, camera=camera["camera"]),
        camera,
    )

    deadline = time.time() + 20
    incidents: list[dict] = []
    while time.time() < deadline:
        resp = requests.get(
            f"{FUSION_RT_URL}/api/v1/incidents",
            params={"status": "active", "zone": zone_id},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        incidents = resp.json().get("incidents", [])
        if any(inc.get("rule_id") == "fusion-rt-mqtt-gate" for inc in incidents):
            break
        time.sleep(0.5)
    else:
        pytest.fail(f"no incident after MQTT fixtures; last={incidents}")

    match = next(inc for inc in incidents if inc.get("rule_id") == "fusion-rt-mqtt-gate")
    assert match["zone_id"] == zone_id
    assert set(match["modalities"]) == {"camera", "custom"}
