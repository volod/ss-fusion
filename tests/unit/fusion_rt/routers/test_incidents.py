"""Unit tests for incident endpoints."""

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
        modalities=["camera", "audio"],
        confidence=0.92,
        risk_level="critical",
        summary_text="test",
        evidence_refs=[],
        rule_id="r1",
        acknowledged_at=None,
        dismissed_at=None,
        dismissal_reason=None,
        created_at=NOW,
    )
    defaults.update(kwargs)
    return _Row(defaults)


def _make_app(conn):
    from contextlib import asynccontextmanager

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
def conn():
    return AsyncMock()


@pytest.fixture
def headers():
    with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        yield {}


def test_list_incidents_active_default(conn, headers):
    conn.fetch = AsyncMock(return_value=[_make_row()])
    conn.fetchval = AsyncMock(return_value="north-gate")
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()["incidents"]) == 1


def test_list_incidents_unknown_zone_404(conn, headers):
    conn.fetchval = AsyncMock(return_value=None)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents?zone=nonexistent", headers=headers)
    assert resp.status_code == 404


def test_get_incident_not_found(conn, headers):
    conn.fetchrow = AsyncMock(return_value=None)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get(f"/incidents/{uuid4()}", headers=headers)
    assert resp.status_code == 404


def test_get_incident_found(conn, headers):
    row = _make_row()
    conn.fetchrow = AsyncMock(return_value=row)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get(f"/incidents/{row.incident_id}", headers=headers)
    assert resp.status_code == 200
    assert "incident_id" in resp.json()


def test_acknowledge_sets_acknowledged_at(conn, headers):
    row = _make_row(acknowledged_at=NOW)
    conn.fetchrow = AsyncMock(return_value=row)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(f"/incidents/{row.incident_id}/acknowledge", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["acknowledged_at"] is not None


def test_dismiss_with_reason(conn, headers):
    row = _make_row(dismissed_at=NOW, dismissal_reason="false positive")
    conn.fetchrow = AsyncMock(return_value=row)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        f"/incidents/{row.incident_id}/dismiss",
        json={"reason": "false positive"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["dismissed_at"] is not None


def test_dismiss_without_reason(conn, headers):
    row = _make_row(dismissed_at=NOW, dismissal_reason=None)
    conn.fetchrow = AsyncMock(return_value=row)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        f"/incidents/{row.incident_id}/dismiss",
        json={},
        headers=headers,
    )
    assert resp.status_code == 200


def test_add_note(conn, headers):
    inc_id = uuid4()
    note_row = _Row(
        note_id=uuid4(), incident_id=inc_id, body="test note", operator_id="op1", created_at=NOW
    )
    conn.fetchval = AsyncMock(return_value=str(inc_id))
    conn.fetchrow = AsyncMock(return_value=note_row)
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        f"/incidents/{inc_id}/notes",
        json={"body": "test note", "operator_id": "op1"},
        headers=headers,
    )
    assert resp.status_code == 201


def test_add_note_body_too_long(conn, headers):
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        f"/incidents/{uuid4()}/notes",
        json={"body": "x" * 4001},
        headers=headers,
    )
    assert resp.status_code == 422


def test_export_json(conn, headers):
    conn.fetch = AsyncMock(return_value=[_make_row()])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents/export?format=json", headers=headers)
    assert resp.status_code == 200
    assert "incidents" in resp.json()


def test_export_csv(conn, headers):
    conn.fetch = AsyncMock(return_value=[_make_row()])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents/export?format=csv", headers=headers)
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_export_limit_too_large(conn, headers):
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/incidents/export?limit=10001", headers=headers)
    assert resp.status_code == 422
