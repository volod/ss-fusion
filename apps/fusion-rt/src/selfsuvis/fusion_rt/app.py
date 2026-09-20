"""Fusion-rt FastAPI app: incidents, rules, zones, threat, and scene synthesis."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from selfsuvis.fusion_rt.config import fusion_settings, validate_fusion_settings
from selfsuvis.fusion_rt.db import close_db_pool, init_db_pool
from selfsuvis.fusion_rt.health import router as health_router
from selfsuvis.fusion_rt.migrate import migrate as migrate_fusion
from selfsuvis.fusion_rt.routers.site import router as site_router
from selfsuvis.fusion_rt.routers.v1 import router as v1_router
from selfsuvis.fusion_rt.runtime import cancel_tasks, start_correlator_tasks, start_site_runtime
from selfsuvis.pipeline.core import get_logger
from ss_kit.web import SecurityHeadersMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down fusion-rt resources."""
    try:
        validate_fusion_settings()
        if fusion_settings.FUSION_DATABASE_URL:
            await migrate_fusion(fusion_settings.FUSION_DATABASE_URL, verbose=False)
    except Exception as exc:
        logger.warning("fusion schema bootstrap failed: %s", exc)
    await init_db_pool(app)
    app.state.sse_subscribers: dict[str, asyncio.Queue] = {}

    mqtt_task = start_site_runtime(app)
    correlator_task, webhook_task = start_correlator_tasks(app)

    try:
        yield
    finally:
        await cancel_tasks([correlator_task, webhook_task, mqtt_task])
        await close_db_pool(app)


app = FastAPI(title="fusion-rt", lifespan=lifespan)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(v1_router)
app.include_router(site_router)
app.include_router(health_router)
