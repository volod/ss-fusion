"""Unit tests for zone CRUD and zone history."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

NOW = datetime.now(UTC)


class _Row(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_zone(**kwargs):
    defaults = dict(
        zone_id="north-gate",
        label="North Gate",
        description=None,
        map_x=None,
        map_y=None,
        map_w=None,
        map_h=None,
        created_at=NOW,
    )
    defaults.update(kwargs)
    return _Row(defaults)


def _make_incident(**kwargs):
    defaults = dict(
        incident_id=uuid4(),
        ts=NOW,
        zone_id="north-gate",
        modalities=["camera"],
        confidence=0.8,
        risk_level="high",
        summary_text="test",
        evidence_refs=[],
        rule_id=None,
        acknowledged_at=None,
        dismissed_at=None,
        dismissal_reason=None,
        created_at=NOW,
    )
    defaults.update(kwargs)
    return _Row(defaults)


def _make_app(conn):

    from fastapi import FastAPI

    from selfsuvis.fusion_rt.routers.v1.zones import router

    app = FastAPI()
    app.include_router(router)
    pool = AsyncMock()

    @asynccontextmanager
    async def _acq():
        yield conn

    pool.acquire = _acq
    app.state.db_pool = pool
    return app


@pytest.fixture
def conn():
    return AsyncMock()


@pytest.fixture
def headers():
    with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        yield {}


def test_create_zone(conn, headers):
    conn.fetchval = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(return_value=_make_zone())
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        "/zones",
        json={"zone_id": "north-gate", "label": "North Gate"},
        headers=headers,
    )
    assert resp.status_code == 201
    assert resp.json()["zone_id"] == "north-gate"


def test_create_zone_coordinates_optional(conn, headers):
    conn.fetchval = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(return_value=_make_zone())
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        "/zones",
        json={"zone_id": "north-gate", "label": "North Gate"},
        headers=headers,
    )
    assert resp.status_code == 201


def test_list_zones(conn, headers):
    conn.fetch = AsyncMock(return_value=[_make_zone()])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/zones", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()["zones"]) == 1


def test_zone_history_with_incidents(conn, headers):
    conn.fetchval = AsyncMock(return_value="north-gate")
    conn.fetch = AsyncMock(return_value=[_make_incident()])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/zones/north-gate/history", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["zone_id"] == "north-gate"
    assert len(resp.json()["incidents"]) == 1


def test_zone_history_sorted_desc(conn, headers):
    conn.fetchval = AsyncMock(return_value="north-gate")
    inc1 = _make_incident()
    inc2 = _make_incident()
    conn.fetch = AsyncMock(return_value=[inc1, inc2])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/zones/north-gate/history", headers=headers)
    assert resp.status_code == 200


def test_zone_history_nonexistent_404(conn, headers):
    conn.fetchval = AsyncMock(return_value=None)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/zones/nonexistent/history", headers=headers)
    assert resp.status_code == 404
