"""Shared fixtures for v1 router unit tests."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

NOW = datetime.now(UTC)
NOW_ISO = NOW.isoformat()


class _Row(dict):
    """Minimal asyncpg Record stand-in: supports row["key"] and row.key."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_event_row(**kwargs):
    defaults = {
        "event_id": uuid4(),
        "ts": NOW,
        "zone_id": "north-gate",
        "sensor_id": "cam-01",
        "modality": "camera",
        "confidence": 0.85,
        "payload": {},
        "artifact_uri": None,
        "created_at": NOW,
    }
    defaults.update(kwargs)
    return _Row(defaults)


def _make_incident_row(**kwargs):
    defaults = {
        "incident_id": uuid4(),
        "ts": NOW,
        "zone_id": "north-gate",
        "modalities": ["camera", "audio"],
        "confidence": 0.92,
        "risk_level": "critical",
        "summary_text": "Drone convergence — audio, camera in north-gate (critical)",
        "evidence_refs": [],
        "rule_id": "drone-test",
        "acknowledged_at": None,
        "dismissed_at": None,
        "dismissal_reason": None,
        "created_at": NOW,
    }
    defaults.update(kwargs)
    return _Row(defaults)


def _make_zone_row(**kwargs):
    defaults = {
        "zone_id": "north-gate",
        "label": "North Gate",
        "description": None,
        "map_x": None,
        "map_y": None,
        "map_w": None,
        "map_h": None,
        "created_at": NOW,
    }
    defaults.update(kwargs)
    return _Row(defaults)


def _make_note_row(**kwargs):
    defaults = {
        "note_id": uuid4(),
        "incident_id": uuid4(),
        "body": "Investigated and confirmed",
        "operator_id": "op-1",
        "created_at": NOW,
    }
    defaults.update(kwargs)
    return _Row(defaults)


def _make_rule_row(**kwargs):
    defaults = {
        "rule_id": "drone-test",
        "label": "Drone test",
        "modalities": ["camera", "audio"],
        "zone_id": None,
        "window_s": 60,
        "min_confidence": 0.7,
        "enabled": True,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(kwargs)
    return _Row(defaults)


@pytest.fixture
def mock_pool():
    conn = AsyncMock()
    pool = AsyncMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool, conn


@pytest.fixture
def app_with_pool(mock_pool):
    pool, conn = mock_pool
    from fastapi import FastAPI

    from selfsuvis.fusion_rt.routers.v1 import router

    app = FastAPI()
    app.include_router(router)
    app.state.db_pool = pool
    app.state.sse_subscribers = {}
    return app, pool, conn
