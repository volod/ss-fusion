"""v1 API router — mounts all fusion-rt v1 sub-routers."""

from fastapi import APIRouter

from selfsuvis.fusion_rt.routers.v1.events import router as events_router
from selfsuvis.fusion_rt.routers.v1.incidents import router as incidents_router
from selfsuvis.fusion_rt.routers.v1.rules import router as rules_router
from selfsuvis.fusion_rt.routers.v1.site_state import router as site_state_router
from selfsuvis.fusion_rt.routers.v1.stream import router as stream_router
from selfsuvis.fusion_rt.routers.v1.zones import router as zones_router

router = APIRouter(prefix="/api/v1", tags=["v1"])

router.include_router(events_router)
router.include_router(site_state_router)
router.include_router(incidents_router)
router.include_router(zones_router)
router.include_router(rules_router)
router.include_router(stream_router)
