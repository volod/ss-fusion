"""Integration tests: webhook retry queue.

Requires running Docker stack with Redis.
Run with: pytest tests/integration/ -m integration
"""

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = "http://localhost:8000"


@pytest.fixture
def headers():
    import os

    key = os.getenv("API_KEY", "")
    return {"X-Api-Key": key} if key else {}


async def test_dlq_depth_visible_in_health(headers):
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{API_URL}/health")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "dlq_depth" in data
        assert isinstance(data["dlq_depth"], int)
