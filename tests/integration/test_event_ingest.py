"""Integration tests: event ingest E2E flow.

Requires running Docker stack (postgres, redis, api).
Run with: pytest tests/integration/ -m integration
"""

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = "http://localhost:8000"


@pytest.fixture
def headers(api_key):
    return {"X-Api-Key": api_key} if api_key else {}


@pytest.fixture
def api_key():
    import os

    return os.getenv("API_KEY", "")


async def test_post_event_appears_in_db(headers):
    ts = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{API_URL}/api/v1/events/camera",
            json={
                "ts": ts,
                "zone_id": "test-zone",
                "sensor_id": "test-cam",
                "confidence": 0.8,
            },
            headers=headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "event_id" in data
    assert data["zone_id"] == "test-zone"


async def test_correlator_creates_incident_visible_in_site_state(headers):
    """Seed events, wait for correlator, check site/state."""
    ts = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient(timeout=10.0) as client:
        # First create a zone
        await client.post(
            f"{API_URL}/api/v1/zones",
            json={"zone_id": "e2e-zone", "label": "E2E Zone"},
            headers=headers,
        )

        # Create a rule
        await client.post(
            f"{API_URL}/api/v1/rules",
            json={
                "rule_id": "e2e-test-rule",
                "label": "E2E Test",
                "modalities": ["camera"],
                "zone_id": "e2e-zone",
                "window_s": 30,
                "min_confidence": 0.5,
                "enabled": True,
            },
            headers=headers,
        )

        # Post a matching event
        await client.post(
            f"{API_URL}/api/v1/events/camera",
            json={"ts": ts, "zone_id": "e2e-zone", "sensor_id": "cam", "confidence": 0.9},
            headers=headers,
        )

        # Wait for correlator tick (5s + buffer)
        await asyncio.sleep(7)

        resp = await client.get(f"{API_URL}/api/v1/site/state", headers=headers)
        assert resp.status_code == 200
        zones = {z["zone_id"]: z for z in resp.json()["zones"]}
        assert "e2e-zone" in zones
        assert zones["e2e-zone"]["risk_level"] is not None


async def test_acknowledge_removes_from_site_state(headers):
    """Acknowledged incident should not appear in site/state."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{API_URL}/api/v1/incidents?status=active", headers=headers)
        incidents = resp.json().get("incidents", [])
        if not incidents:
            pytest.skip("No active incidents to test with")

        inc_id = incidents[0]["incident_id"]
        await client.post(f"{API_URL}/api/v1/incidents/{inc_id}/acknowledge", headers=headers)

        resp = await client.get(f"{API_URL}/api/v1/site/state", headers=headers)
        zones = resp.json()["zones"]
        all_active_ids = [i["incident_id"] for z in zones for i in z["active_incidents"]]
        assert inc_id not in all_active_ids
