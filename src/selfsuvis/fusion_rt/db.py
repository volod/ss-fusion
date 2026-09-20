"""Database pool utilities for the fusion-rt FastAPI app."""

import asyncpg
from fastapi import FastAPI, HTTPException, Request

from selfsuvis.fusion_rt.config import fusion_settings
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)


async def init_db_pool(app: FastAPI) -> None:
    """Initialize the fusion asyncpg pool and attach it to app state."""
    db_url = fusion_settings.FUSION_DATABASE_URL
    if not db_url:
        app.state.db_pool = None
        logger.warning("FUSION_DATABASE_URL not configured; fusion DB operations unavailable")
        return
    app.state.db_pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=1,
        max_size=10,
        timeout=10,
    )


async def close_db_pool(app: FastAPI) -> None:
    """Close the asyncpg pool if it exists."""
    pool: asyncpg.Pool | None = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()


def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return the fusion DB pool from request app state or raise 503."""
    pool: asyncpg.Pool | None = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="FUSION_DATABASE_URL not configured")
    return pool
