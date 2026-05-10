"""Integration tests: correlator E2E.

Requires running Docker stack.
Run with: pytest tests/integration/ -m integration
"""

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = "http://localhost:8000"


@pytest.fixture
def headers():
    import os

    key = os.getenv("API_KEY", "")
    return {"X-Api-Key": key} if key else {}


async def test_seed_rule_and_events_trigger_incident(headers):
    ts = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"{API_URL}/api/v1/zones",
            json={"zone_id": "corr-zone", "label": "Correlator Zone"},
            headers=headers,
        )
        await client.post(
            f"{API_URL}/api/v1/rules",
            json={
                "rule_id": "corr-e2e",
                "label": "Correlator E2E",
                "modalities": ["camera", "audio"],
                "zone_id": "corr-zone",
                "window_s": 60,
                "min_confidence": 0.5,
                "enabled": True,
            },
            headers=headers,
        )
        for modality in ("camera", "audio"):
            await client.post(
                f"{API_URL}/api/v1/events/{modality}",
                json={
                    "ts": ts,
                    "zone_id": "corr-zone",
                    "sensor_id": f"{modality}-1",
                    "confidence": 0.8,
                },
                headers=headers,
            )

        await asyncio.sleep(7)  # Wait for correlator tick

        resp = await client.get(
            f"{API_URL}/api/v1/incidents?zone=corr-zone&status=active",
            headers=headers,
        )
        incidents = resp.json()["incidents"]
        assert len(incidents) >= 1
        assert incidents[0]["zone_id"] == "corr-zone"
