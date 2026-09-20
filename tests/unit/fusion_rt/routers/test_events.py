"""Unit tests for POST /api/v1/events/{modality}."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

NOW = datetime.now(UTC)
_VALID_BODY = {
    "ts": NOW.isoformat(),
    "zone_id": "north-gate",
    "sensor_id": "cam-01",
    "confidence": 0.85,
}


class _Row(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_app(conn):
    from fastapi import FastAPI

    from selfsuvis.fusion_rt.routers.v1.events import router

    app = FastAPI()
    app.include_router(router)

    pool = AsyncMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    app.state.db_pool = pool
    return app


@pytest.fixture
def conn():
    c = AsyncMock()
    row = _Row(
        event_id=uuid4(),
        ts=NOW,
        zone_id="north-gate",
        sensor_id="cam-01",
        modality="camera",
        confidence=0.85,
        payload={},
        artifact_uri=None,
        created_at=NOW,
    )
    c.fetchrow = AsyncMock(return_value=row)
    c.fetchval = AsyncMock(return_value=0)
    return c


def test_post_valid_event(conn):
    with (
        patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms,
        patch("selfsuvis.fusion_rt.deps.sensor_rate_limit"),
    ):
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        app = _make_app(conn)
        client = TestClient(app)
        resp = client.post("/events/camera", json=_VALID_BODY)
        assert resp.status_code == 200
        assert "event_id" in resp.json()


def test_post_unknown_modality(conn):
    with (
        patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms,
        patch("selfsuvis.fusion_rt.deps.sensor_rate_limit"),
    ):
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        app = _make_app(conn)
        client = TestClient(app)
        resp = client.post("/events/unknown_xyz", json=_VALID_BODY)
        assert resp.status_code == 422


def test_post_invalid_artifact_uri(conn):
    with (
        patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms,
        patch("selfsuvis.fusion_rt.deps.sensor_rate_limit"),
        patch(
            "selfsuvis.fusion_rt.routers.v1.events.resolve_allowed_path",
            side_effect=PermissionError,
        ),
    ):
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        app = _make_app(conn)
        client = TestClient(app)
        body = {**_VALID_BODY, "artifact_uri": "/etc/passwd"}
        resp = client.post("/events/camera", json=body)
        assert resp.status_code == 422


def test_post_empty_payload_defaults(conn):
    """payload field defaults to {} when omitted."""
    with (
        patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms,
        patch("selfsuvis.fusion_rt.deps.sensor_rate_limit"),
    ):
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        app = _make_app(conn)
        client = TestClient(app)
        resp = client.post("/events/audio", json=_VALID_BODY)
        assert resp.status_code == 200


def test_post_sensor_key_wrong_scope(conn):
    """Valid key but missing ingest scope → 403."""
    conn.fetchval = AsyncMock(side_effect=[1, None])
    conn.fetchrow = AsyncMock(return_value=MagicMock(scopes=["read"]))

    with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
        ms.API_KEY = "site-key"
        ms.API_AUTH_REQUIRED = True
        app = _make_app(conn)
        client = TestClient(app)
        resp = client.post("/events/camera", json=_VALID_BODY, headers={"X-Sensor-Key": "somekey"})
        assert resp.status_code in (401, 403)
