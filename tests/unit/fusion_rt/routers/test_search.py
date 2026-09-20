"""Unit tests for incident search."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

NOW = datetime.now(timezone.utc)


class _Row(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_row(**kwargs):
    defaults = dict(
        incident_id=uuid4(),
        ts=NOW,
        zone_id="north-gate",
        modalities=["camera"],
        confidence=0.8,
        risk_level="high",
        summary_text="Drone convergence",
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

    from selfsuvis.fusion_rt.routers.v1.incidents import router

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
def headers():
    with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        yield {}


def test_search_returns_matching(headers):
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[_make_row()])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents/search?q=drone", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()["incidents"]) == 1


def test_search_no_results(headers):
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents/search?q=nonexistent_xyz", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["incidents"] == []


def test_search_with_zone_filter(headers):
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[_make_row()])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents/search?q=drone&zone=north-gate", headers=headers)
    assert resp.status_code == 200
