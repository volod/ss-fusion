"""Unit tests for fusion-rt /health."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(pool=None):
    from selfsuvis.fusion_rt.health import router

    app = FastAPI()
    app.include_router(router)
    app.state.db_pool = pool
    app.state.sse_subscribers = {}
    return app


def test_health_postgres_ok():
    pool = AsyncMock()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)

    @asynccontextmanager
    async def _acq():
        yield conn

    pool.acquire = _acq
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.llen = AsyncMock(return_value=0)
    mock_redis.aclose = AsyncMock()

    with (
        patch("selfsuvis.fusion_rt.health.fusion_settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
    ):
        ms.HEALTH_REDIS_URL = "redis://localhost"
        ms.CORRELATOR_ENABLED = False
        ms.DRONE_AUDIO_MODEL_PATH = ""
        ms.DRONE_AUDIO_WATCH_DIR = ""
        client = TestClient(_make_app(pool))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["postgres"] == "ok"
        assert resp.json()["redis"] == "ok"


def test_health_redis_unreachable():
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=ConnectionError("refused"))

    with (
        patch("selfsuvis.fusion_rt.health.fusion_settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
    ):
        ms.HEALTH_REDIS_URL = "redis://localhost"
        ms.CORRELATOR_ENABLED = False
        client = TestClient(_make_app())
        resp = client.get("/health")
        assert resp.json()["redis"] == "error"
        assert resp.json()["status"] == "down"
        assert resp.status_code == 503
