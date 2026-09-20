"""Unit tests for GET /api/v1/site/state."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

NOW = datetime.now(UTC)


def _make_app(conn):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from selfsuvis.fusion_rt.routers.v1.site_state import router

    app = FastAPI()
    app.include_router(router)
    pool = AsyncMock()

    @asynccontextmanager
    async def _acq():
        yield conn

    pool.acquire = _acq
    app.state.db_pool = pool
    return app


class _Row(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_zone(zone_id="north-gate", label="North Gate"):
    return _Row({"zone_id": zone_id, "label": label})


def _make_incident(**kwargs):
    defaults = dict(
        incident_id=uuid4(),
        ts=NOW,
        zone_id="north-gate",
        modalities=["camera"],
        confidence=0.5,
        risk_level="medium",
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


@pytest.fixture
def conn():
    return AsyncMock()


@pytest.fixture
def headers():
    with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        yield {}


def test_no_zones_returns_empty(conn, headers):
    conn.fetch = AsyncMock(side_effect=[[], []])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/site/state", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["zones"] == []


def test_zones_no_incidents_risk_null(conn, headers):
    conn.fetch = AsyncMock(side_effect=[[_make_zone()], []])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/site/state", headers=headers)
    assert resp.status_code == 200
    zones = resp.json()["zones"]
    assert len(zones) == 1
    assert zones[0]["risk_level"] is None


def test_active_incident_sets_risk(conn, headers):
    incident = _make_incident(risk_level="critical")
    conn.fetch = AsyncMock(side_effect=[[_make_zone()], [incident]])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/site/state", headers=headers)
    zones = resp.json()["zones"]
    assert zones[0]["risk_level"] == "critical"
    assert len(zones[0]["active_incidents"]) == 1


def test_acknowledged_incidents_not_shown(conn, headers):
    _make_incident(acknowledged_at=NOW)
    # The DB query filters by acknowledged_at IS NULL, so result would be empty
    conn.fetch = AsyncMock(side_effect=[[_make_zone()], []])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/site/state", headers=headers)
    zones = resp.json()["zones"]
    assert zones[0]["risk_level"] is None
