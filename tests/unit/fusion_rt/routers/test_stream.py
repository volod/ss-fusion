"""Unit tests for SSE stream endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def _make_app():
    from fastapi import FastAPI

    from selfsuvis.fusion_rt.routers.v1.stream import router

    app = FastAPI()
    app.include_router(router)
    app.state.sse_subscribers = {}
    return app


def test_stream_missing_token_returns_401():
    with patch("selfsuvis.fusion_rt.routers.v1.stream.fusion_settings") as ms:
        ms.API_KEY = "secret"
        ms.API_AUTH_REQUIRED = True
        app = _make_app()
        client = TestClient(app)
        # Use stream=True to avoid hanging on SSE connection
        with client.stream("GET", "/events/stream") as resp:
            assert resp.status_code == 401


def test_stream_invalid_token_returns_401():
    with patch("selfsuvis.fusion_rt.routers.v1.stream.fusion_settings") as ms:
        ms.API_KEY = "secret"
        ms.API_AUTH_REQUIRED = True
        app = _make_app()
        client = TestClient(app)
        with client.stream("GET", "/events/stream?token=wrong") as resp:
            assert resp.status_code == 401


def test_stream_valid_token_does_not_raise():
    """_validate_token with correct token should not raise."""
    from fastapi import HTTPException

    from selfsuvis.fusion_rt.routers.v1.stream import _validate_token

    with patch("selfsuvis.fusion_rt.routers.v1.stream.fusion_settings") as ms:
        ms.API_KEY = "secret"
        ms.API_AUTH_REQUIRED = True
        try:
            _validate_token("secret")
        except HTTPException:
            raise AssertionError("valid token should not raise")
