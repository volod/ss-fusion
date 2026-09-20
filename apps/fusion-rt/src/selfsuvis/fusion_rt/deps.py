"""Auth and rate-limit helpers for the fusion-rt FastAPI app."""

import hashlib

from fastapi import Header, HTTPException, Request

from selfsuvis.fusion_rt.config import fusion_settings
from ss_kit.security import MAX_RATE_LIMIT_CLIENTS, RateLimiter
from ss_kit.web import check_api_key, client_key

_rate_limiter = RateLimiter(
    lambda: (
        float(fusion_settings.RATE_LIMIT_PER_MIN) / 60.0,
        float(fusion_settings.RATE_LIMIT_BURST),
    )
)
_sensor_rate_limiter = RateLimiter((10.0, 10.0))

# Backward-compatible exports used by tests.
_MAX_LIMITERS = MAX_RATE_LIMIT_CLIENTS
_limiters = _rate_limiter.buckets
_RateLimiter = RateLimiter


def _get_client_key(request: Request) -> str:
    """Derive client key for rate limiting.

    When TRUST_PROXY_HEADERS is True, uses X-Forwarded-For. Only enable
    TRUST_PROXY_HEADERS behind a trusted reverse proxy that strips this header.
    """
    return client_key(request, trust_proxy_headers=fusion_settings.TRUST_PROXY_HEADERS)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    check_api_key(
        x_api_key,
        fusion_settings.API_KEY,
        required=fusion_settings.API_AUTH_REQUIRED,
    )


def rate_limit(request: Request) -> None:
    if fusion_settings.RATE_LIMIT_PER_MIN <= 0:
        return
    if not _rate_limiter.check(_get_client_key(request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def sensor_rate_limit(sensor_id: str) -> None:
    """10 req/s per sensor_id. Called after EventEnvelope is parsed."""
    if not _sensor_rate_limiter.check(sensor_id):
        raise HTTPException(status_code=429, detail="Sensor rate limit exceeded")


async def require_sensor_key(
    request: Request,
    x_sensor_key: str = Header(default=""),
) -> None:
    """Authenticate ingest requests.

    If no sensor_keys rows exist: fall back to site-level require_api_key.
    If rows exist and key missing/invalid: 401.
    If key found but scope 'ingest' absent: 403.
    """
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        require_api_key(x_sensor_key)
        return

    async with pool.acquire() as conn:
        row_count = await conn.fetchval("SELECT COUNT(*) FROM sensor_keys")

    if row_count == 0:
        require_api_key(x_sensor_key)
        return

    if not x_sensor_key:
        raise HTTPException(status_code=401, detail="X-Sensor-Key header required")

    key_hash = hashlib.sha256(x_sensor_key.encode()).hexdigest()

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT scopes FROM sensor_keys WHERE key_hash = $1", key_hash)

    if row is None:
        raise HTTPException(status_code=401, detail="Invalid sensor key")

    if "ingest" not in row["scopes"]:
        raise HTTPException(status_code=403, detail="Sensor key lacks ingest scope")
