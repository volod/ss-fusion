"""GET /site/state and /site/threat on fusion-rt."""

from datetime import UTC, datetime
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from selfsuvis.fusion_rt.aggregator import RealtimeThreatAggregator
from selfsuvis.fusion_rt.routers.site import router
from selfsuvis.fusion_rt.site_snapshot import CombinedSiteSnapshot
from ss_contracts.models import SensorState


def _app():
    app = FastAPI()
    app.include_router(router)
    snap = CombinedSiteSnapshot()
    threat = RealtimeThreatAggregator()
    app.state.site_snapshot = snap
    app.state.site_threat_aggregator = threat
    return app, snap, threat


def test_site_state_and_threat() -> None:
    app, snap, threat = _app()
    import asyncio

    asyncio.run(
        snap.ingest_sensor_state(
            SensorState(
                dev_eui="aabb",
                last_seen=datetime(2026, 9, 19, 8, 0, tzinfo=UTC),
                reading_count=1,
                temperature_c=20.0,
            )
        )
    )
    threat.consume(
        {
            "event_kind": "sensor",
            "event_time": "2026-09-19T08:00:00+00:00",
            "ingest_time": "2026-09-19T08:00:01+00:00",
            "node_id": "aabb",
            "sensor_type": "lorawan",
            "sector_id": "grid:1:2",
            "payload": {"temperature_c": 20.0},
            "freshness_sec": 0.0,
        }
    )
    with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        client = TestClient(app)
        state = client.get("/site/state")
        assert state.status_code == 200
        body = state.json()
        assert body["sensor_count"] == 1
        threat_body = client.get("/site/threat")
        assert threat_body.status_code == 200
        dumped = threat_body.json()
        assert "last_update" in dumped
        assert "global_threat_map" in dumped


def test_site_state_503_without_snapshot() -> None:
    app = FastAPI()
    app.include_router(router)
    with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        client = TestClient(app)
        assert client.get("/site/state").status_code == 503
        assert client.get("/site/threat").status_code == 503
