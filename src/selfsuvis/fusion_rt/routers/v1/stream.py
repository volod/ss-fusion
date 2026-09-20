"""GET /api/v1/events/stream — SSE endpoint for incident notifications."""

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from selfsuvis.fusion_rt.config import fusion_settings
from selfsuvis.pipeline.core import get_logger
from ss_kit.security import keys_match

logger = get_logger(__name__)

router = APIRouter()


def _validate_token(token: str) -> None:
    if not fusion_settings.API_KEY:
        if fusion_settings.API_AUTH_REQUIRED:
            raise HTTPException(status_code=401, detail="Server authentication not configured")
        return
    if not token or not keys_match(token, fusion_settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/events/stream", summary="SSE event stream")
async def events_stream(
    request: Request,
    token: str = Query(default=""),
) -> EventSourceResponse:
    _validate_token(token)

    subscriber_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    subscribers: dict[str, asyncio.Queue] = getattr(request.app.state, "sse_subscribers", {})
    subscribers[subscriber_id] = queue
    if not hasattr(request.app.state, "sse_subscribers"):
        request.app.state.sse_subscribers = subscribers

    async def _generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield event
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            subscribers.pop(subscriber_id, None)

    return EventSourceResponse(_generator())
