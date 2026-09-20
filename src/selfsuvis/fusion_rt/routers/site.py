"""Live site snapshot from contract sensor events plus published camera events.

GET /site/state, /site/threat, /site/synthesis, WS /site/stream.
"""

import asyncio
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from selfsuvis.fusion_rt.config import fusion_settings
from selfsuvis.fusion_rt.deps import require_api_key
from selfsuvis.fusion_rt.site_snapshot import CombinedSiteState
from selfsuvis.pipeline.core import get_logger
from ss_kit.security import keys_match

logger = get_logger(__name__)

router = APIRouter(tags=["site"])


def _snapshot(request: Request):
    snap = getattr(request.app.state, "site_snapshot", None)
    if snap is None:
        raise HTTPException(status_code=503, detail="site snapshot is not running")
    return snap


def _ws_auth(token: str) -> None:
    if not fusion_settings.API_KEY:
        if fusion_settings.API_AUTH_REQUIRED:
            raise HTTPException(status_code=401, detail="Server authentication not configured")
        return
    if not token or not keys_match(token, fusion_settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/site/state", response_model=CombinedSiteState)
async def site_state(request: Request, _: None = Depends(require_api_key)) -> CombinedSiteState:
    snapshot = _snapshot(request)
    return await snapshot.get_state()


@router.get("/site/threat")
async def site_threat(request: Request, _: None = Depends(require_api_key)) -> dict[str, Any]:
    agg = getattr(request.app.state, "site_threat_aggregator", None)
    if agg is None:
        raise HTTPException(status_code=503, detail="threat aggregator is not running")
    return agg.snapshot()


@router.get("/site/synthesis")
async def site_synthesis(
    request: Request,
    force: bool = False,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    synthesizer = getattr(request.app.state, "scene_synthesizer", None)
    if synthesizer is None:
        raise HTTPException(status_code=503, detail="scene synthesizer is not running")
    result = await synthesizer.synthesize(force=force)
    return result.model_dump(mode="json")


@router.websocket("/site/stream")
async def site_stream(websocket: WebSocket, token: str = Query(default="")) -> None:
    try:
        _ws_auth(token)
    except HTTPException:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    snapshot = getattr(websocket.app.state, "site_snapshot", None)
    try:
        while True:
            if snapshot is not None:
                state = await snapshot.get_state()
                await websocket.send_json(state.model_dump(mode="json"))
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.debug("site stream closed", exc_info=True)
