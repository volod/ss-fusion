"""Unit tests for rules CRUD."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

NOW = datetime.now(UTC)


class _Row(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_row(**kwargs):
    defaults = dict(
        rule_id="drone-test",
        label="Drone test",
        modalities=["camera", "audio"],
        zone_id=None,
        window_s=60,
        min_confidence=0.7,
        enabled=True,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(kwargs)
    return _Row(defaults)


def _make_app(conn):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from selfsuvis.fusion_rt.routers.v1.rules import router

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


_VALID_BODY = {
    "rule_id": "drone-test",
    "label": "Drone test",
    "modalities": ["camera", "audio"],
    "window_s": 60,
    "min_confidence": 0.7,
    "enabled": True,
}


def test_create_rule(conn, headers):
    conn.fetchval = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(return_value=_make_row())
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/rules", json=_VALID_BODY, headers=headers)
    assert resp.status_code == 201


def test_create_rule_window_zero(conn, headers):
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/rules", json={**_VALID_BODY, "window_s": 0}, headers=headers)
    assert resp.status_code == 422


def test_create_rule_window_too_large(conn, headers):
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/rules", json={**_VALID_BODY, "window_s": 3601}, headers=headers)
    assert resp.status_code == 422


def test_create_rule_confidence_too_high(conn, headers):
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/rules", json={**_VALID_BODY, "min_confidence": 1.1}, headers=headers)
    assert resp.status_code == 422


def test_create_rule_empty_modalities(conn, headers):
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/rules", json={**_VALID_BODY, "modalities": []}, headers=headers)
    assert resp.status_code == 422


def test_create_rule_invalid_modality(conn, headers):
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        "/rules", json={**_VALID_BODY, "modalities": ["invalid_mod"]}, headers=headers
    )
    assert resp.status_code == 422


def test_list_rules(conn, headers):
    conn.fetch = AsyncMock(return_value=[_make_row()])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/rules", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()["rules"]) == 1


def test_put_rule(conn, headers):
    conn.fetchrow = AsyncMock(side_effect=[_make_row(), _make_row(label="Updated")])
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.put("/rules/drone-test", json={"label": "Updated"}, headers=headers)
    assert resp.status_code == 200


def test_delete_rule(conn, headers):
    conn.execute = AsyncMock(return_value="DELETE 1")
    app = _make_app(conn)
    client = TestClient(app)
    resp = client.delete("/rules/drone-test", headers=headers)
    assert resp.status_code == 204
