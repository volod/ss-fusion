"""fusion-rt OpenAPI serves incidents and live site routes, not cameras."""

from fastapi import FastAPI

from selfsuvis.fusion_rt.routers.site import router as site_router
from selfsuvis.fusion_rt.routers.v1 import router as v1_router


def test_fusion_rt_openapi_includes_moved_routes() -> None:
    app = FastAPI()
    app.include_router(v1_router)
    app.include_router(site_router)
    paths = set(app.openapi()["paths"])
    assert "/api/v1/incidents" in paths
    assert "/api/v1/rules" in paths
    assert "/api/v1/zones" in paths
    assert "/site/state" in paths
    assert "/site/threat" in paths
    assert "/site/synthesis" in paths
    assert "/site/cameras" not in paths
    assert any(getattr(route, "path", None) == "/site/stream" for route in app.routes)
